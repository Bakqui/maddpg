import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv():
    def __init__(self):

        # self.agents = self.world.policy_agents
        # set required vectorized gym env property
        ## Blotto
        self.n = 2
        self.K = 5
        self.W = [1.,1.,1.,1.,1.]
        self.N_list = [50, 10] #[Na, Nb]


        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in range(self.n):
            total_action_space = []
            # physical action space

            u_action_space = spaces.Discrete(self.K)

            total_action_space.append(u_action_space)

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.K,), dtype=np.float32))


        # rendering
        # self.shared_viewer = shared_viewer
        # if self.shared_viewer:
        #     self.viewers = [None]
        # else:
        #     self.viewers = [None] * self.n
        # self._reset_render()

    def step(self, action_n_sm):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        # reward = np.sum(np.sign(np.around(action_n[0]-action_n[1], decimals=0)) * self.W)

        # reward_a = np.sum((np.around(action_n[0]-action_n[1], decimals=0)>0) * self.W)
        # reward_b = np.sum((np.around(action_n[0]-action_n[1], decimals=0)<0) * self.W)

        # reward_a = np.sum((np.around(action_n[0]-action_n[1], decimals=0)>0) * self.W)
        # reward_b = np.sum((np.around(action_n[0]-action_n[1], decimals=0)<0) * self.W)

        action_n = [sm*Nn for sm, Nn in zip(action_n_sm, self.N_list)]


        reward_a = np.sum(((np.around(action_n[0], decimals=0)-np.around(action_n[1], decimals=0))>0) * self.W)
        reward_b = np.sum(((np.around(action_n[0], decimals=0)-np.around(action_n[1], decimals=0))<0) * self.W)

        # print(np.around(action_n[0]-action_n[1], decimals=1))
        # print('A party:', np.around(action_n[0]))
        # print('B party:', np.around(action_n[1]))
        # print('Occ',np.sign(action_n[0]-action_n[1]))
        # print('A Rew:',reward)
        # print('B Rew:',-reward)
        for agent in range(self.n):
            obs_n.append(self.W)
            # reward_n.append(reward)
            done_n.append(True)
            info_n['n'].append(True)
        # reward_n = [reward,-reward]
        reward_n = [reward_a, reward_b]

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for agent in range(self.n):
            obs_n.append(self.W)
        # for agent in self.agents:
        #     obs_n.append(self._get_obs(agent))
        return obs_n
