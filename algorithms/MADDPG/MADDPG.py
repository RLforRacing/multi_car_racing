import numpy as np
import matplotlib.pyplot as plt
import os, sys, random 
import tensorflow as tf
from tensorflow.python.keras.backend import update
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import MultivariateNormalTriL
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'benchmarking'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tools'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'gym_multi_car_racing'))
sys.path.append(os.path.join(os.path.dirname(__file__)))
from model_tester_maddpg import TesterAgent
from gym import spaces
from aux_functions import *
from multi_car_racing import *
from memory_noise_models import *
from config_default_MADDPG import *


class MADDPGTesterAgent(TesterAgent):
    def __init__(self,
                 save_path='../../.',
                 **kwargs
                 ):

        super().__init__(**kwargs)
        self.agent = self._load_model(save_path)
        

    def _load_model(self, save_path):
        noise_type = 'ou'
        noise_std = np.array([0.0,0.2], dtype=np.float32)
        noise_mean = np.array([0.0, -0.2], dtype=np.float32)
        action_dim = 2
        state_shape = (84,96,3)

        train_or_test = 'test'
        ckpt_load = 100

        self.num_cars = 2
        self.action_divide_factor_steer = 4
        self.action_divide_factor_gb = 6

        config_write = dict(num_cars=self.num_cars, noise_type=noise_type, noise_std=noise_std, noise_mean=noise_mean, 
                            train_or_test=train_or_test, ckpt_load=ckpt_load, action_dim=action_dim)
        config = set_config_default_MADDPG()
        config = add_settings_to_config(config, config_write)

        agent = MADDPG(config, state_shape)
        return agent


    def state_to_action(self, s):
        """
        This function should take the most recent state and return the
        action vector in the exact same form it is needed for the environment.
        If you are using frame buffer see example in _update_frame_buffer
        how to take care of that.
        """
        action, _ = self.agent.get_action(s)
        for i in range(self.num_cars):
            action[i][0] /= self.action_divide_factor_steer
            action[i][1:] /= self.action_divide_factor_gb
        return np.array(action)

    
    @staticmethod
    def setup_action_space(env):
        """
        This should be the same action space setup function that you used for training.
        Make sure that the actions set here are the same as the ones used to train the model.
        """
        env.action_space = spaces.Box( env.action_lb, env.action_ub, dtype=np.float32)  # (steer, gas, brake) x N

    @staticmethod
    def get_observation_type():
        """
        Simply return 'frames' or 'features'
        """
        return 'frames'


class MADDPG(object):

    def __init__(self, config, state_shape):

        self.config = config
        self.state_shape = state_shape
        if len(self.state_shape) != 3 or not isinstance(self.state_shape, tuple):
            assert 1==0, 'State shape must be tuple and of length 3, check if both correct!'
       
        self.num_cars = self.config['num_cars']

        self.noise_type = self.config['noise_type'].lower()
        self.noise_std = np.array(self.config['noise_std'], dtype=np.float32).reshape(-1, )
        self.noise_mean = np.array(self.config['noise_mean'], dtype=np.float32).reshape(-1, )
        self.noise_scale = self.config['noise_scale']

        self.action_dim = self.config['action_dim']
        self.tau = self.config['tau']
        self.gamma = self.config['gamma']
        self.max_memory_size = self.config['max_memory_size']
        self.batch_size = self.config['batch_size']

        self.max_to_keep = self.config['max_to_keep']

        model_save_suffices_single = ['saved_models/MADDPG/actor', 'saved_models/MADDPG/critic', 'saved_models/MADDPG/target_actor', 'saved_models/MADDPG/target_critic']
        self.model_save_suffices = []
        self.model_save_paths = dict()
        for i in range(self.num_cars):
            model_save_suffices_i = ['{}_{}'.format(j, (i+1)) for j in model_save_suffices_single]
            self.model_save_suffices.extend(model_save_suffices_i)

            self.model_save_paths['actor_{}_path'.format(i+1)] = '{}/{}'.format(self.config['directory_to_save'], self.model_save_suffices[4*i])
            self.model_save_paths['critic_{}_path'.format(i+1)] = '{}/{}'.format(self.config['directory_to_save'], self.model_save_suffices[4*i + 1])
            self.model_save_paths['target_actor_{}_path'.format(i+1)] = '{}/{}'.format(self.config['directory_to_save'], self.model_save_suffices[4*i + 2])
            self.model_save_paths['target_critic_{}_path'.format(i+1)] = '{}/{}'.format(self.config['directory_to_save'], self.model_save_suffices[4*i + 3])

        for _, model_save_path in self.model_save_paths.items():
            if not os.path.isdir(model_save_path):
                os.makedirs(model_save_path)

        # setup the model parameters
        self._set_network_params()
        # set action noise 
        self._set_noise_dist()
        # create/load main and target models
        self._set_or_init_networks()
        # create checkpoints
        self._set_checkpoints()

        # experience replay buffer 
        self.memory  = ReplayBuffer(self.max_memory_size, self.state_shape, self.action_dim, self.num_cars)
    

    def _set_or_init_networks(self):

        # create main and target models
        self.actor_models = []
        self.target_actor_models = []

        self.critic_models = []
        self.target_critic_models = []

        for i in range(self.num_cars):
            self.actor_models.append(self._build_actor_network(name='actor_{}'.format(i+1)))
            self.target_actor_models.append(self._build_actor_network(name='target_actor_{}'.format(i+1)))

            self.critic_models.append(self._build_critic_network(name='critic_{}'.format(i+1)))
            self.target_critic_models.append(self._build_critic_network(name='target_critic_{}'.format(i+1)))

            if self.config['train_or_test'] == 'train' and self.config['ckpt_load'] <= 0:
                # initialize the target and actual models in the same way
                self.target_actor_models[i].set_weights(self.actor_models[i].get_weights())
                self.target_critic_models[i].set_weights(self.critic_models[i].get_weights())

            else:
                # create the temp checkpoints to load the weights
                checkpoint_actor = tf.train.Checkpoint(model=self.actor_models[i])
                checkpoint_target_actor = tf.train.Checkpoint(model=self.target_actor_models[i])

                checkpoint_critic = tf.train.Checkpoint(model=self.critic_models[i])
                checkpoint_target_critic = tf.train.Checkpoint(model=self.target_critic_models[i])

                checkpoint_actor.restore(self.model_save_paths['actor_{}_path'.format(i+1)] + '/ckpt-{}'.format(self.config['ckpt_load']))
                checkpoint_target_actor.restore(self.model_save_paths['target_actor_{}_path'.format(i+1)] + '/ckpt-{}'.format(self.config['ckpt_load']))
                checkpoint_critic.restore(self.model_save_paths['critic_{}_path'.format(i+1)] + '/ckpt-{}'.format(self.config['ckpt_load']))
                checkpoint_target_critic.restore(self.model_save_paths['target_critic_{}_path'.format(i+1)] + '/ckpt-{}'.format(self.config['ckpt_load']))


    def _set_network_params(self):
        # set learning rates 
        self.init_lr_actor = self.config['init_lr_actor']
        self.init_lr_critic = self.config['init_lr_critic']
        
        # set the optimizers
        self.optimizers_actor = []
        self.optimizers_critic = []
        for i in range(self.num_cars):
            self.optimizers_actor.append(tf.keras.optimizers.Adam(learning_rate = self.init_lr_actor))
            self.optimizers_critic.append(tf.keras.optimizers.Adam(learning_rate = self.init_lr_critic))
        self.initializer_layer = tf.keras.initializers.GlorotNormal()

        # set exponential decay learning rate params
        self.decay_lr_steps = self.config['decay_lr_steps']
        self.decay_lr_rate = self.config['decay_lr_rate']


    def _set_noise_dist(self):
        if self.noise_type == 'ou':
            if self.noise_std.size != self.action_dim:
                assert 1==0, 'OU Noise std shape is not compatible with provided action dimension!'
            else:
                self.noise = NoiseGenerator(self.noise_mean, self.noise_std)
        else:
            if self.noise_std.size != self.action_dim:
                assert 1==0, 'Normal noise scale shape is not compatible with provided action dimension!'
            else:
                self.noise_cov = np.diag(self.noise_scale)
            self.noise_std = compute_cholesky_if_possible(self.cov_noise, type_jitter='add')
            self.noise = MultivariateNormalTriL(self.noise_mean, self.noise_std)
        
        # set exponential decay scaling params
        self.decay_noise_steps = self.config['decay_noise_steps']
        self.decay_noise_rate = self.config['decay_noise_rate']


    def _set_checkpoints(self):
        
        self.checkpoint_actors = []
        self.manager_actors = []

        self.checkpoint_critics = []
        self.manager_critics = []

        self.checkpoint_target_actors = []
        self.manager_target_actors = []

        self.checkpoint_target_critics = []
        self.manager_target_critics = []

        for i in range(self.num_cars):
            # set checkpoints
            self.checkpoint_actors.append(tf.train.Checkpoint(model=self.actor_models[i]))
            self.manager_actors.append(tf.train.CheckpointManager(self.checkpoint_actors[i], self.model_save_paths['actor_{}_path'.format(i+1)], max_to_keep = self.max_to_keep))

            self.checkpoint_critics.append(tf.train.Checkpoint(model=self.critic_models[i]))
            self.manager_critics.append(tf.train.CheckpointManager(self.checkpoint_critics[i], self.model_save_paths['critic_{}_path'.format(i+1)], max_to_keep = self.max_to_keep))

            self.checkpoint_target_actors.append(tf.train.Checkpoint(model=self.target_actor_models[i]))
            self.manager_target_actors.append(tf.train.CheckpointManager(self.checkpoint_target_actors[i], self.model_save_paths['target_actor_{}_path'.format(i+1)], max_to_keep = self.max_to_keep))

            self.checkpoint_target_critics.append(tf.train.Checkpoint(model=self.target_critic_models[i]))
            self.manager_target_critics.append(tf.train.CheckpointManager(self.checkpoint_target_critics[i], self.model_save_paths['target_critic_{}_path'.format(i+1)], max_to_keep = self.max_to_keep))


    def _build_actor_network(self, name='actor'):
        # state is the input to the network
        input_layer = tf.keras.Input(shape=self.state_shape)
        net = tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(4,4), padding='valid', use_bias=False, activation='relu', kernel_initializer=self.initializer_layer)(input_layer)
        net = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(3,3), padding='valid', use_bias=False, activation='relu', kernel_initializer=self.initializer_layer)(net)
        net = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(3,3), padding='valid', use_bias=False, activation='relu', kernel_initializer=self.initializer_layer)(net)
        net = tf.keras.layers.Flatten()(net)

        net = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=self.initializer_layer)(net)
        
        # get the actions
        if self.action_dim == 2:
            act = tf.keras.layers.Dense(self.action_dim, activation='tanh', kernel_initializer=self.initializer_layer)(net)
        elif self.action_dim in [3,4]:
            act = tf.keras.layers.Dense(self.action_dim, activation='sigmoid', kernel_initializer=self.initializer_layer)(net)

        # build actor model
        actor_model = tf.keras.Model(inputs=input_layer, outputs=act, name=name)
        return actor_model


    def _build_critic_network(self, name='critic'):

        # state is the input to the network
        multistate_shape = (self.state_shape[0], self.state_shape[1], self.state_shape[2]*self.num_cars)

        input_layer_state = tf.keras.Input(shape=multistate_shape)
        net_state = tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(4,4), padding='valid', use_bias=False, activation='relu', kernel_initializer=self.initializer_layer)(input_layer_state)
        net_state = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(3,3), padding='valid', use_bias=False, activation='relu', kernel_initializer=self.initializer_layer)(net_state)
        net_state = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(3,3), padding='valid', use_bias=False, activation='relu', kernel_initializer=self.initializer_layer)(net_state)
        net_state = tf.keras.layers.Flatten()(net_state)

        # pass action thru network
        input_layer_action = tf.keras.Input(shape=(self.action_dim*self.num_cars, ))
        net_action = input_layer_action

        # concatenate action and state outs, pass thru the network
        net_together = tf.keras.layers.Concatenate()([net_state, net_action])
        net_together = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=self.initializer_layer)(net_together)
        net_together = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=self.initializer_layer)(net_together)
        net_together = tf.keras.layers.Dense(32, activation='relu', kernel_initializer=self.initializer_layer)(net_together)
        
        # get the output
        out = tf.keras.layers.Dense(1, kernel_initializer=self.initializer_layer)(net_together)

        # build critic model
        critic_model = tf.keras.Model(inputs=[input_layer_state, input_layer_action], outputs=out, name=name)
        return critic_model
                

    def generate_noise(self): 
        if self.noise_type == 'ou':
            return self.noise.generate()
        else:
            return self.noise.sample(sample_shape=1).numpy().reshape(-1, )


    def get_action(self, state):
        state = process_image(state)

        action_before = []
        action = []
        # get the actions from actor networks
        for i in range(self.num_cars):
            action_before_i = self.actor_models[i](np.expand_dims(state[i, :, :, :], axis=0), training=False).numpy()
            action_before_i = action_before_i[0] + self.generate_noise()
            
            if self.action_dim == 2:
                action_i = [action_before_i[0], action_before_i[1].clip(0, 1),  -action_before_i[1].clip(-1, 0)]
            elif self.action_dim == 3:
                action_i = [2*action_before_i[0] - 1, action_before_i[1], action_before_i[2]]
            elif self.action_dim == 4:
                right_left = 1 if action_before_i[-1] > 0.5 else -1
                action_i = [action_before_i[0], action_before_i[1], action_before_i[2]]
                action_i[0] *= right_left

            action_i = np.clip(np.array(action_i), [-1,0,0], a_max=[1,1,1])
            action.append(action_i)
            action_before.append(action_before_i)
        return action, action_before


    # @tf.function
    def update_target_network_params(self):
        for i in range(self.num_cars):
            for v, t in zip(self.critic_models[i].trainable_variables, self.target_critic_models[i].trainable_variables):
                t.assign(t * (1-self.tau) + v * self.tau)

            for v, t in zip(self.actor_models[i].trainable_variables, self.target_actor_models[i].trainable_variables):
                t.assign(t * (1-self.tau) + v * self.tau)
    

    # @tf.function
    def update_agent_critic(self, states, actions, rewards, next_states, agent_id):
        batch_size, num_agents, height, width, ch = states.shape
        with tf.GradientTape(persistent=True) as tape:
            for i in range(self.num_cars):
                next_actions_i = self.target_actor_models[i](next_states[:, i, :, :, :], training=True)
                if i == 0:
                    full_next_actions = next_actions_i
                else:
                    full_next_actions = tf.concat([full_next_actions, next_actions_i], axis=1)

            full_next_states = tf.reshape(next_states, (batch_size, height, width, -1))
            next_q_vals = self.target_critic_models[agent_id]([full_next_states, full_next_actions], training=True)
            y = tf.expand_dims(rewards[:, agent_id], axis=1) + self.gamma * next_q_vals

            full_states = tf.reshape(states, (batch_size, height, width, -1))
            full_actions = tf.reshape(actions, (batch_size, -1))
            loss_critic = tf.math.reduce_mean( tf.square( self.critic_models[agent_id]([full_states, full_actions], training=True) - y ) ) 

        critic_gradients = tape.gradient(loss_critic, self.critic_models[agent_id].trainable_variables)
        self.optimizers_critic[agent_id].apply_gradients(zip(critic_gradients, self.critic_models[agent_id].trainable_variables))
        
        
    # @tf.function
    def update_agent_actor(self, states, actions, rewards, next_states, agent_id):
        batch_size, num_agents, height, width, ch = states.shape
        with tf.GradientTape(persistent=True) as tape:
            for i in range(self.num_cars):
                actor_actions_i = self.actor_models[i](states[:, i, :, :, :], training=True)
                if i == 0:
                    full_actor_actions = actor_actions_i
                else:
                    full_actor_actions = tf.concat([full_actor_actions, actor_actions_i], axis=1)
            
            full_states = tf.reshape(states, (batch_size, height, width, -1))
            q_vals = self.critic_models[agent_id]([full_states, full_actor_actions], training=True)
            loss_actor = -tf.math.reduce_mean(q_vals)
        
        actor_gradients = tape.gradient(loss_actor, self.actor_models[agent_id].trainable_variables)
        self.optimizers_actor[agent_id].apply_gradients(zip(actor_gradients, self.actor_models[agent_id].trainable_variables))


    def train_step(self):

        for agent_id in range(self.num_cars):
            # sample the batch
            states, actions, rewards, next_states, _ = self.memory.sample(self.batch_size)
            
            states = tf.constant(states, dtype=tf.float32)
            next_states = tf.constant(next_states, dtype=tf.float32)
            actions = tf.constant(actions, dtype=tf.float32)
            rewards = tf.constant(rewards, dtype=tf.float32)

            # update networks
            self.update_agent_critic(states, actions, rewards, next_states, agent_id)
            self.update_agent_actor(states, actions, rewards, next_states, agent_id)
        self.update_target_network_params()
     

    def save_models(self):
        for i in range(self.num_cars):
            self.manager_actors[i].save()
            self.manager_critics[i].save()
            self.manager_target_actors[i].save()
            self.manager_target_critics[i].save()


    def reset_update_noise(self, ep_num):
        if self.noise_type == 'ou':
            self.noise.reset()
            self.noise_std *= decayed_learning_rate(1, ep_num, self.decay_noise_steps, self.decay_noise_rate, staircase=True)
        else:
            updated_noise_scale = decayed_learning_rate(self.noise_scale, ep_num, self.decay_noise_steps, self.decay_noise_rate, staircase=True)
            self.noise_cov = updated_noise_scale * np.eye(self.action_dim, dtype=tf.float32)
            self.noise_std = compute_cholesky_if_possible(self.cov_noise, type_jitter='add')
            self.noise = MultivariateNormalTriL(self.noise_mean, self.noise_std)


    def update_network_lrs(self, ep_num):
        for i in range(self.num_cars):
            self.optimizers_actor[i].learning_rate.assign(decayed_learning_rate(self.init_lr_actor, ep_num, self.decay_lr_steps, self.decay_lr_rate, staircase=True))
            self.optimizers_critic[i].learning_rate.assign(decayed_learning_rate(self.init_lr_critic, ep_num, self.decay_lr_steps, self.decay_lr_rate, staircase=True))


    def create_writer(self):
        summary_path_main = '{}/graphs'.format(self.config['directory_to_save'])
        if not os.path.isdir(summary_path_main):
            os.mkdir(summary_path_main)
        summary_path = '{}/{}'.format(summary_path_main, 'MADDPG')
        if not os.path.isdir(summary_path):
            os.mkdir(summary_path)
        
        self.writer = tf.summary.create_file_writer(summary_path)


    def def_summary(self, avg_reward, step_reward, step, prefix):
        """ 
        Add training values to a TF Summary object for Tensorboard
        """
        if not hasattr(self, 'writer'):
            return 

        with self.writer.as_default(): 
            for i in range(self.num_cars):
                tf.summary.scalar(name=prefix + 'avg_reward_{}'.format(i+1), data=tf.constant(avg_reward[i], dtype=tf.float32), step=step)
                tf.summary.scalar(name=prefix + 'step_reward_{}'.format(i+1), data=tf.constant(step_reward[i], dtype=tf.float32), step=step)

    







