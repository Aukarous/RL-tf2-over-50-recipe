"""
The SARSA algorithm can be applied to model-free control problems and allows us to
optimize the value function of an unknown MDP.
"""

import numpy as np
import random

from value_function_utils import visualize_grid_action_values
from temporal_diff import GridworldV2Env


def sarsa(env: GridworldV2Env, max_episodes):
	grid_action_values = np.zeros((len(env.distinct_states), env.action_space.n))
	grid_action_values[env.goal_state] = 1
	grid_action_values[env.bomb_state] = -1
	gamma, alpha = 0.99, 0.01  # discount factor, learning rate
	q = grid_action_values  # q: state-action-value function
	# outer loop
	for episode in range(max_episodes):
		step_num = 1
		done = False
		state = env.reset()
		action = greedy_policy(q[state], 1)
		# inner loop with the SARSA learning update step
		while not done:
			next_state, reward, done = env.step(action)
			step_num += 1
			decayed_epsilon = gamma ** step_num
			next_action = greedy_policy(q_values=q[next_state], epsilon=decayed_epsilon)
			q[state][action] += \
				alpha * (reward + gamma * q[next_state][next_action] - q[state][action])
			state = next_state
			action = next_action
	
	visualize_grid_action_values(grid_action_values)


def greedy_policy(q_values, epsilon):
	"""epsilon-greedy policy that the agent will use"""
	if random.random() >= epsilon:
		return np.argmax(q_values)
	else:
		return random.randint(0, 3)


if __name__ == "__main__":
	max_episodes_ = 4000
	env_ = GridworldV2Env(step_cost=0.1, max_ep_length=30)
	sarsa(env_, max_episodes=max_episodes_)
