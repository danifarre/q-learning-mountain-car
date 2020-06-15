import numpy as np
import gym
from ql_mountain_car import q_learner as ql

MAX_NUM_EPISODES = 50000


def train(agent, environment):
    best_reward = -float('inf')

    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = environment.reset()
        total_reward = 0.0

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, info = environment.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward

        if total_reward > best_reward:
            best_reward = total_reward

        print("Episodio número {} con recompensa: {}, mejor recompensa: {}, epsilon: {}".format(episode, total_reward,
                                                                                                best_reward,
                                                                                                agent.epsilon))

    return np.argmax(agent.q, axis=2)


def test(agent, environment, policy):
    done = False
    obs = environment.reset()
    total_reward = 0.0  # Recompensa total de cada episodio

    while not done:
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, info = environment.step(action)
        obs = next_obs
        total_reward += reward

    return total_reward


if __name__ == "__main__":
    environment = gym.make("MountainCar-v0")
    agent = ql.QLearner(environment)

    learned_policy = train(agent, environment)

    monitor_path = "./monitor_output"
    environment = gym.wrappers.Monitor(environment, monitor_path, force=True)

    for _ in range(1000):
        test(agent, environment, learned_policy)

    environment.close()
