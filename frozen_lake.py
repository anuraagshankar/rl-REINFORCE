import gym
import numpy as np
from scipy.special import softmax
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize the environment

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)


"""
Assume - state represented in one-hot -> 16 states -> each state represented using a feature vector of size 16
s -> shape = (n_states, n_states)
represent state-value function as a linear function of the state
w -> shape = (n_states, )
v(s, w) = np.sum(w * s[i])

Feature vector x(s, a) can be represented as vector of size 6 (n_states, n_actions, n_states * n_actions)
represent policy parameter as a linear function of the feature vector
theta -> shape = (n_states * n_actions, )
h(s, a, theta) = np.sum(theta * x[s, a])
pi[s] = softmax on h along s
"""

n_states = env.observation_space.n
n_actions = env.action_space.n
n_features = n_states * n_actions

states = np.zeros((n_states, n_states))
# Converting to one-hot
states[np.arange(n_states), np.arange(n_states)] = 1

feature_vector = np.zeros((n_states, n_actions, n_states * n_actions))
for state in range(n_states):
    for action in range(n_actions):
        # Converting to one-hot
        feature_vector[state, action, state * n_actions + action] = 1

# For visualizing the runs


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_preferences(state, feature_vector, theta):
    action_preferences = np.sum(
        feature_vector[state] * theta, axis=1)  # shape: (n_actions)
    return softmax(action_preferences)  # shape: (n_actions)


def choose_action(state, feature_vector, theta):
    pi = get_preferences(state, feature_vector, theta)
    # Sample from probability distribution
    return np.random.choice(np.arange(n_actions), p=pi)


def get_state_value(states, state, w):
    return np.sum(states[state] * w)


def reinforce(num_eps, alpha_w, alpha_theta):
    w = np.zeros(n_states)
    theta = np.zeros(n_states * n_actions)

    total_returns = []
    for ep in tqdm(range(num_eps)):
        # Generate state, action, reward tuples following the given policy
        state = env.reset()
        done = False
        state_action_reward = []
        while not done:
            action = choose_action(state, feature_vector, theta)
            next_state, reward, done, _ = env.step(action)
            state_action_reward.append((state, action, reward))
            state = next_state

        # Improve the policy and value function
        returns = 0
        for state, action, reward in reversed(state_action_reward):
            returns += reward
            delta = returns - get_state_value(states, state, w)

            grad_w = states[state]
            w = w + alpha_w * delta * grad_w

            pi = get_preferences(state, feature_vector, theta)
            grad_theta = feature_vector[state, action] - np.sum(
                [pi[a] * feature_vector[state, a] for a in range(n_actions) if a != action], axis=0)
            theta = theta + alpha_theta * delta * grad_theta

        total_returns.append(returns)
    return total_returns, w, theta


"""
Working parameters:
is_slippery = False:
	num_eps = 1000
	alpha_w = 0.005
	alpha_theta = 0.1

is_slippery = True:
	num_eps = 10000
	alpha_w = 0.1
	alpha_theta = 0.05
"""

alpha_w = 0.1
alpha_theta = 0.05

num_eps = 10000

total_returns, w, theta = reinforce(num_eps, alpha_w, alpha_theta)
print(f'Successful Runs: {np.sum(total_returns)}')

rolling_returns = moving_average(total_returns, n=num_eps // 40)
if len(rolling_returns) > 10000:
    rolling_returns = [rr for i, rr in enumerate(
        rolling_returns) if i % (num_eps//10000) == 0]

sns.set(rc={'figure.figsize': (8, 8)})
ax = sns.lineplot(x=np.arange(len(rolling_returns)) + 1, y=rolling_returns)
ax.set(xlabel='Episode', ylabel='Rolling Return')
plt.show()
