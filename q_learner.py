import numpy as np

EPSILON_MIN = 0.005  # Aprendemos mientras el incremento de aprendizaje sea superior a dicho valor
MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 200
MAX_NUM_STEPS = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / MAX_NUM_STEPS  # Caida de epsilon en cada paso
ALPHA = 0.05  # Ratio de aprendizaje del agente
GAMMA = 0.98  # Factor de descuento
NUM_DISCRETE_BINS = 30
EPSILON = 1.00


class QLearner(object):

    def __init__(self, environment):
        self.obs_shape = environment.observation_space.shape
        self.obs_high = environment.observation_space.high
        self.obs_low = environment.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins

        self.action_shape = environment.action_space.n
        self.q = np.zeros((self.obs_bins, self.obs_bins, self.action_shape))

        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON

    def discretize(self, obs):
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, obs):
        discrete_obs = self.discretize(obs)

        # Epsilon-Greedy
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY

        if np.random.random() > self.epsilon:
            return np.argmax(self.q[discrete_obs])
        else:
            return np.random.choice([action for action in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        discrete_obs = self.discretize(obs)
        discrete_next_obs = self.discretize(next_obs)
        td_target = reward + self.gamma * np.max(self.q[discrete_next_obs])
        td_error = td_target - self.q[discrete_obs][action]
        self.q[discrete_obs][action] += self.alpha * td_error
