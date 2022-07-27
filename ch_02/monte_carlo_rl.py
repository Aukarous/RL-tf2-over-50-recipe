"""
This recipe provides the ingredients for building a Monte Carlo prediction and control
algorithm so that you can build your RL agents. Similar to the temporal difference
learning algorithm, Monte Carlo learning methods can be used to learn both the state
and the action value functions. Monte Carlo methods have zero bias since they learn
from complete episodes with real experience, without approximate predictions. These
methods are suitable for applications that require good convergence properties. The
following diagram illustrates the value that's learned by the Monte Carlo method for the
GridworldV2 environment:
Monte-Carlo 算法用于预测和控制；
与时序差分算法类似，MC算法即可以学习状态函数，也可以学习动作价值函数。MC算法没有bias，因为MC从完整的
episode中学习真实经验，没有近似预测。
MC算法适用于需要较好收敛性的应用
"""

# We will start by implementing the monte_carlo_prediction algorithm
# and visualizing the learned value function for each state in the GridworldV2
# environment. After that, we will implement an epsilon-greedy policy and the
# monte_carlo_control algorithm to construct an agent that will act in an RL
# environment.


import numpy as np
from temporal_diff import GridworldV2Env
from value_function_utils import visualize_grid_state_values, visualize_grid_action_values


def monte_carlo_prediction(env: GridworldV2Env, max_episodes: int):
	returns = {state: [] for state in env.distinct_states}
	grid_state_values = np.zeros(len(env.distinct_states))
	grid_state_values[env.goal_state] = 1
	grid_state_values[env.bomb_state] = -1
	gamma = 0.99  # discount factor
	# outer loop, Outer loops are commonplace in all RL agent training code:
	for episode in range(max_episodes):
		g_t = 0
		state = env.reset()
		done = False
		trajectory = []
		while not done:
			action = env.action_space.sample()
			# random policy
			next_state, reward, done = env.step(action)
			trajectory.append((state, reward))
			state = next_state
		
		for idx, (state, reward) in enumerate(trajectory[::-1]):
			g_t = gamma * g_t + reward
			# first visit MC prediction
			if state not in np.array(trajectory[::-1])[:, 0][idx + 1:]:
				returns[str(state)].append(g_t)
				grid_state_values[state] = np.mean(returns[str(state)])
	visualize_grid_state_values(grid_state_values.reshape((3, 4)))


def epsilon_greedy_policy(action_logits, epsilon=0.2):
	idx = np.argmax(action_logits)
	probs = []
	epsilon_decay_factor = np.sqrt(sum([a ** 2 for a in action_logits]))
	if epsilon_decay_factor == 0:
		epsilon_decay_factor = 1.0
	for i, a in enumerate(action_logits):
		if i == idx:
			probs.append(round(1 - epsilon + (epsilon / epsilon_decay_factor), 3))
		else:
			probs.append(round(epsilon / epsilon_decay_factor, 3))
	residual_err = sum(probs) - 1
	residual = residual_err / len(action_logits)
	res = np.array(probs) - residual
	return res


def monte_carlo_control(env: GridworldV2Env, max_episodes):
	# initial values for the state-action values:
	grid_state_action_values = np.zeros((12, 4))
	grid_state_action_values[3] = 1
	grid_state_action_values[7] = -1
	# initializing the returns for all the possible state and action pairs:
	returns = {}
	possible_states = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
	possible_actions = ["0", "1", "2", "3"]
	for state in possible_states:
		for action in possible_actions:
			returns[state + ',' + action] = []
	
	# define the outer loop for each episode and then the inner loop for each step in
	# an episode. By doing this, we can collect trajectories of experience until the
	# end of an episode:
	gamma = 0.99
	for episode in range(max_episodes):
		g_t = 0
		state = env.reset()
		trajectory = []
		while True:
			action_values = grid_state_action_values[state]
			probs = epsilon_greedy_policy(action_logits=action_values)
			action = np.random.choice(np.arange(4), p=probs)  # random policy
			next_state, reward, done = env.step(action)
			trajectory.append((state, action, reward))
			state = next_state
			if done: break
		# Now that we have a full trajectory for an episode in the inner loop, we can
		# implement our Monte Carlo Control update to update the state-action values:
		for step in reversed(trajectory):
			g_t = gamma * g_t + step[2]
			returns[str(step[0]) + ',' + str(step[1])].append(g_t)
			t_ = returns[str(step[0]) + ',' + str(step[1])]
			grid_state_action_values[step[0]][step[1]] = np.mean(t_)
	
	# Once the outer loop completes, we can visualize the state-action values
	visualize_grid_action_values(grid_state_action_values)


if __name__ == "__main__":
	max_episodes_ = 4000
	env_ = GridworldV2Env(step_cost=-0.1, max_ep_length=30)
	print(f"==============Monte Carlo Prediction=================")
	monte_carlo_prediction(env_, max_episodes=max_episodes_)
	print(f"++++++++++++++Monte Carlo Control++++++++++++++++++++")
	monte_carlo_control(env_, max_episodes=max_episodes_)
