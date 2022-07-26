"""
Value-based reinforcement learning works by learning the state-value function or the
action-value function in a given environment. This recipe will show you how to create
and update the value function for the Maze environment to obtain an optimal policy.
Learning value functions, especially in model-free RL problems where a model of the
environment is not available, can prove to be quite effective, especially for RL problems
with low-dimensional state space.
基于价值的强化学习通过学习特定环境中的状态-价值函数或行动-价值函数来实现。此菜谱将向您展示如何创建和更新
迷宫环境的值函数，以获得最佳策略。学习值函数，特别是在无模型的 RL 问题中，在没有环境模型的情况下，
可以证明是非常有效的，特别是对于低维状态空间的 RL 问题。
"""
from stochastic_env import MazeEnv
import numpy as np
from typing import List
from value_function_utils import visualize_maze_values


env = MazeEnv()
print(f'Observation space: {env.observation_space}')
print(f'Action space: {env.action_space}')

state_dim = env.distinct_states
state_values = np.zeros(state_dim)
q_values = np.zeros((state_dim, env.action_space.n))
policy = np.zeros(state_dim)

discount = 0.9


def calculate_values(state, action):
	"""
	Evaluate value function for given state and action
	Args:
		state: int, Valid (discrete) state in discrete `env.observation_space`
		action: int, Valid (discrete) action in `env.action_space`
	Returns:
		v_sum: value for given state, action
	"""
	v_sum = 0
	slip_action = env.slip_action_map[action]
	env.set_state(state)
	slip_next_state, slip_reward, _ = env.step(slip_action, slip=False)
	
	# We need a list of transitions in the environment to be able to calculate the rewards,
	# as per the Bellman equations. Let's create a transitions list and append the
	# newly obtained environment transition information:
	transitions = [(slip_reward, slip_next_state, env.slip)]
	
	# Let's obtain another transition using the state and the action, this time without
	# stochasticity. We can do this by not using slip_action and setting slip=False
	# while stepping through the Maze environment:
	env.set_state(state)
	next_state, reward, _ = env.step(action, slip=False)
	transitions.append((reward, next_state, 1 - env.slip))
	
	# There is only one more step needed to complete the calculate_values
	# function, which is to calculate the values:
	for reward, next_state, pi in transitions:
		v_sum += pi * (reward + discount * state_values[next_state])
	
	return v_sum


# Now, we can start implementing the state/action value learning. We will begin by
# defining the max_iteration hyper_parameters:
max_iteration = 1000

# Let's implement the state-value function learning loop using value iteration:
for i in range(max_iteration):
	v_s = np.zeros(state_dim)
	for state in range(state_dim):
		if env.index_to_coordinate_map[int(state / 8)] == env.goal_pos:
			continue
		v_max = float('-inf')
		for action in range(env.action_space.n):
			v_sum = calculate_values(state, action)
			v_max = max(v_max, v_sum)
		
		v_s[state] = v_max
	
	state_values = np.copy(v_s)

# Now that we have the state-value function learning loop implemented, let's
# move on and implement the action-value function:
for state in range(state_dim):
	for action in range(env.action_space.n):
		q_values[state, action] = calculate_values(state, action)

# With the action-value function computed, we are only one step away from
# obtaining the optimal policy. Let's go get it!
for state in range(state_dim):
	policy[state] = np.argmax(q_values[state, :])

# We can print the Q values (the state-action values) and the policy using the
# following lines of code:

print(f'Q-values: {q_values}')
print('Action mapping: [0--UP; 1--DOWN; 2--LEFT; 3--RIGHT]')
print(f'optimal_policy: {policy}')

visualize_maze_values(q_values, env)


"""
how it works

Value iteration-based value function learning follows Bellman equations, and the optimal
policy is obtained from the Q-value function by simply choosing the action with the
highest Q/action-value.

"""