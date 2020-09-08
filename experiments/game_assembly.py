import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete


# round - minimizing round-off error
def rnd_keep_sum(Nn, sm):
    if Nn < 1:
        return np.zeros(len(sm), dtype=np.int32)
    else:
        fn = Nn*sm
    lbound = np.floor(fn)
    lbound = np.asarray(lbound, dtype=np.int32)
    r_error = fn - lbound
    sort_idx = np.argsort(r_error)
    origin_idx = np.argsort(sort_idx)
    diff = int(np.round(fn.sum(), decimals=0) - lbound.sum())
    rst = lbound[sort_idx]
    rst[-diff:] += 1
    rst = rst[origin_idx]

    return rst


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv():
    def __init__(self):

        # self.agents = self.world.policy_agents
        # set required vectorized gym env property
        ## Blotto
        self.n = 2
        self.K = 6
        self.Wa = [6., 4., 2., 1., 1., 1.]#[5.,4.,3.,2.,1.]
        self.Wb = [6., 2., 3., 1., 2., 1.]#[1.,2.,3.,4.,5.]
        self.N_list = [60, 30] #[Na, Nb]


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


        reward_a = np.sum(((np.around(action_n[0], decimals=0)-np.around(action_n[1], decimals=0))>0) * self.Wa)
        reward_b = np.sum(((np.around(action_n[0], decimals=0)-np.around(action_n[1], decimals=0))<0) * self.Wb)

        # print(np.around(action_n[0]-action_n[1], decimals=1))
        # print('A party:', np.around(action_n[0]))
        # print('B party:', np.around(action_n[1]))
        # print('Occ',np.sign(action_n[0]-action_n[1]))
        # print('A Rew:',reward)
        # print('B Rew:',-reward)
        obs_n.append(np.array(self.Wa)/np.max(self.Wa))
        obs_n.append(np.array(self.Wb)/np.max(self.Wb))
        for agent in range(self.n):

            # reward_n.append(reward)
            done_n.append(True)
            info_n['n'].append(True)
        # reward_n = [reward,-reward]
        reward_n = [reward_a, reward_b]

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        obs_n.append(np.array(self.Wa)/np.max(self.Wa))
        obs_n.append(np.array(self.Wb)/np.max(self.Wb))
        # for agent in range(self.n):
        #     obs_n.append(self.W)
        # for agent in self.agents:
        #     obs_n.append(self._get_obs(agent))
        return obs_n


class MultistepBlottoEnv():
    def __init__(self):

        # self.agents = self.world.policy_agents
        # set required vectorized gym env property

        # Blotto
        self.n = 2
        self.K = 4  # 18
        self.Wa = np.array([100., 100., 100., 100.], dtype=np.float32)#[6.,3.,6.,2.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]#[5.,4.,3.,2.,1.]
        self.Wb = np.array([100., 100., 100., 100.], dtype=np.float32)#[6.,6.,3.,1.,2.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]#[1.,2.,3.,4.,5.]
        self.capacity = np.array([13, 13, 13, 13], dtype=np.int32)
        self.member = np.zeros((2, 4), dtype=np.int32)
        self.N_list = np.array([43, 9], dtype=np.int32)#[197, 103] #[Na, Nb]

        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

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
                    act_space = spaces.MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(2*self.K + 1,), dtype=np.float32))

        # rendering
        # self.shared_viewer = shared_viewer
        # if self.shared_viewer:
        #     self.viewers = [None]
        # else:
        #     self.viewers = [None] * self.n
        # self._reset_render()

    def step(self, action_n_sm):
        obs_n = []
        info_n = {'n': []}

        
        action_n = np.array([rnd_keep_sum(Nn, sm)
                             for sm, Nn
                             in zip(action_n_sm, self.N_list)],
                            dtype=np.int32)
        total_a = np.add(action_n[0], action_n[1])
        is_excess = total_a > self.capacity
        reward_a, reward_b = 0, 0
        for comm in range(self.K):
            if self.capacity[comm] == 0:
                reward_a -= action_n[0, comm]
                reward_b -= action_n[1, comm]
                continue
            if is_excess[comm]:
                selected = self.lottery(action_n, total_a, comm)
            else:
                selected = action_n[:, comm]

            self.N_list -= selected
            self.member[:, comm] += selected
            # sparse reward? or not?
            reward_a += selected[0]
            reward_b += selected[1]

            self.capacity[comm] -= selected.sum()

        done_n = [self.N_list[i] == 0 for i in range(self.n)]

        if all(done_n):
            tie = (self.member[0, :] == self.member[1, :])
            if np.any(tie):
                self.tiebreaking()
            reward_a += np.dot(self.Wa, self.member[0, :] > self.member[1, :])
            reward_b += np.dot(self.Wb, self.member[0, :] < self.member[1, :])

        for agent in range(self.n):
            obs = np.hstack((self.capacity,
                             self.member[agent, :],
                             self.N_list[agent]))
            obs_n.append(obs)
            info_n['n'].append(True)
        reward_n = [reward_a, reward_b]

        return np.array(obs_n), np.array(reward_n), np.array(done_n), info_n

    def reset(self):
        self.capacity = np.array([13, 13, 13, 13], dtype=np.int32)
        self.member = np.zeros((2, 4), dtype=np.int32)
        self.N_list = np.array([43, 9], dtype=np.int32)
        obs_n = []
        for agent in range(self.n):
            obs = np.hstack((self.capacity,
                             self.member[agent, :],
                             self.N_list[agent]))
            obs_n.append(obs)

        return np.array(obs_n)

    def tiebreaking(self):
        for comm in range(self.K):
            if self.member[0, comm] == self.member[1, comm]:
                coin = np.random.choice(2)
                self.member[coin, comm] += 1

    def lottery(self, action_n, total_a, comm):
        selected = np.zeros(2, dtype=np.int32)
        for lot in range(self.capacity[comm]):
            prob_a = action_n[0, comm] / total_a[comm]
            if np.random.random_sample() > prob_a:
                if action_n[0, comm] >= 1:
                    selected[0] += 1
                    action_n[0, comm] -= 1
                elif action_n[1, comm] >= 1:
                    selected[1] += 1
                    action_n[1, comm] -= 1
            else:
                if action_n[1, comm] >= 1:
                    selected[1] += 1
                    action_n[1, comm] -= 1
                elif action_n[0, comm] >= 1:
                    selected[0] += 1
                    action_n[0, comm] -= 1

        return selected

