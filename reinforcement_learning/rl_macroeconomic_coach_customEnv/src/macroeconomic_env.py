import gym
import gym.spaces
import random
import math
import csv
import numpy as np
from scipy.stats import norm


class MacroeconomicEnv(gym.Env):
    """
    An environment for optimal consumption/savings policy in a macroeconomic life-cycle model.
    Based on [Bluhm and Cutura 2020](...)
    """

    def __init__(self, **config):
        config_defaults = {
            # Parameter choice based on https://www.sas.upenn.edu/~jesusfv/Guide_Parallel.pdf
            # https://github.com/davidzarruk/Parallel_Computing
            "nx": 350,    # Grid length for savings
            "xmin": 0.1,    # Lower bound for savings
            "xmax": 4.0,           # Upper bound for savings
            "ne": 9,      # Number of states (labor income shocks)
            "ssigma_eps": 0.02058,   # Variance of discrete markov chain process
            "llambda_eps": 0.99,      # Autoregressive parameter of markov chain process
            "m": 1.5,  # Grid width of markov chain process
            "ssigma": 2,    # Risk aversion parameter in utility function
            "bbeta": 0.97,  # Discount factor
            "steps": 10.0,   # Number of timesteps per episode (years in life-cycle model)
            "r": 0.07,  # One-period interest rate on savings
            "w": 5,  # Labour income
        }

        for key, val in config_defaults.items():
            val = config.get(key, val)  # Override defaults with constructor parameters
            self.__dict__[key] = val  # Creates variables

        self.csv_file = '/opt/ml/output/data/macroeconomic.csv'

        # Make instance of class to generate grids and transition probabilities based on
        # https://github.com/davidzarruk/Parallel_Computing/blob/master/Python_main.py
        data_generator = DataGenerator(self.nx, self.xmax, self.xmin, self.ne, self.ssigma_eps, self.llambda_eps,
                                       self.m)
        self.xgrid = data_generator.make_xgrid()
        self.egrid = data_generator.make_egrid()
        self.p = data_generator.make_transition_matrix()

        # Take exponential of the grid e
        for i in range(0, self.ne):
            self.egrid[i] = math.exp(self.egrid[i])

        # Construct observation space
        space_low = [0.0, self.xmin, min(self.egrid)]
        space_high = [self.steps-1, self.xmax, max(self.egrid)]
        self.observation_space = gym.spaces.Box(low=np.array(space_low), high=np.array(space_high), dtype=np.float32)

        # Construct action space (amount of savings over the interval [xmin, xmax])
        self.action_space = gym.spaces.Box(low=np.array([self.xmin]), high=np.array([self.xmax]), dtype=np.float32)
        self.episode = 0
        self.reset()

    def reset(self):
        """

        """
        self.infos = []
        self.episode += 1
        self.t = 0.0
        self.x = np.random.choice(self.xgrid)
        self.e = np.random.choice(self.egrid)
        return np.array([self.t, self.x, self.e])

    def step(self, action):
        """

        """
        cons = (1 + self.r) * self.x + self.e * self.w - action[0]

        if cons > 0:
            reward = math.pow(cons, (1 - self.ssigma)) / (1 - self.ssigma)
        else:
            reward = math.pow(-10.0, 5)

        # create additional info
        info = {}
        info['episode'] = self.episode
        info['t'] = self.t
        info['x'] = self.x
        info['e'] = self.e

        info['cons'] = cons
        info['reward'] = reward

        transition_probas = self.p[np.where(self.egrid == self.e)[0][0], :]
        self.t += 1
        self.x = action[0]
        self.e = np.random.choice(self.egrid, p=transition_probas)
        observation = np.array([self.t, self.x, self.e])

        done = self.t > self.steps - 1

        info['t_next'] = self.t
        info['x_next'] = action[0]
        info['e_next'] = self.e
        info['done'] = done
        self.infos.append(info)

        if done:
            # Save it to file
            keys = self.infos[0].keys()
            if self.episode == 3:
                with open(self.csv_file, 'w', newline='') as f:
                    dict_writer = csv.DictWriter(f, keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(self.infos)
            else:
                with open(self.csv_file, 'a', newline='') as f:
                    dict_writer = csv.DictWriter(f, keys)
                    dict_writer.writerows(self.infos)

        return observation, reward, done, info


class DataGenerator:
    """
    Data Generating Process based on https://www.sas.upenn.edu/~jesusfv/Guide_Parallel.pdf
    """
    def __init__(self, nx, xmax, xmin, ne, ssigma_eps, llambda_eps, m):
        self.nx = nx
        self.xmax = xmax
        self.xmin = xmin
        self.ne = ne
        self.ssigma_eps = ssigma_eps
        self.llambda_eps = llambda_eps
        self.m = m

    def make_xgrid(self):
        """

        :return:
        """
        xgrid = np.zeros(self.nx)
        size = self.nx
        xstep = (self.xmax - self.xmin) / (size - 1)
        it = 0
        for i in range(0, self.nx):
            xgrid[i] = self.xmin + it * xstep
            it += 1
        return xgrid

    def make_egrid(self):
        """

        :return:
        """
        self.egrid = np.zeros(self.ne)
        size = self.ne
        ssigma_y = math.sqrt(math.pow(self.ssigma_eps, 2) / (1 - math.pow(self.llambda_eps, 2)))
        estep = 2 * ssigma_y * self.m / (size - 1)
        it = 0
        for i in range(0, self.ne):
            self.egrid[i] = (-self.m * math.sqrt(math.pow(self.ssigma_eps, 2) / (1 - math.pow(self.llambda_eps, 2))) +
                        it * estep)
            it += 1
        return self.egrid

    def make_transition_matrix(self):
        """

        :return:
        """
        p = np.zeros((self.ne, self.ne))
        mm = self.egrid[1] - self.egrid[0]
        for j in range(0, self.ne):
            for k in range(0, self.ne):
                if k == 0:
                    p[j, k] = norm.cdf((self.egrid[k] - self.llambda_eps * self.egrid[j] + (mm / 2)) / self.ssigma_eps)
                elif k == self.ne - 1:
                    p[j, k] = 1 - norm.cdf((self.egrid[k] - self.llambda_eps * self.egrid[j] - (mm / 2)) /
                                           self.ssigma_eps)
                else:
                    p[j, k] = norm.cdf((self.egrid[k] - self.llambda_eps * self.egrid[j] + (mm / 2)) /
                                       self.ssigma_eps) - norm.cdf((self.egrid[k] - self.llambda_eps * self.egrid[j] -
                                                                    (mm / 2)) / self.ssigma_eps)
        return p
