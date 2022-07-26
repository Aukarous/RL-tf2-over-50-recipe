"""
The learning environment is a simulator that provides observations for the RL agent,
supports a set of actions that the RL agent can perform by executing the actions, and
returns the resultant/new observation as a result of the agent taking the action.
"""

import gym
import numpy as np
from typing import List


class MazeEnv(gym.Env):
	"""Maze learning environment that represents a simple 2D map with cells representing
	the location of the agent, their goal, walls, coins, and empty space:"""
	
	def __init__(self, stochastic=True):
		self.map = np.asarray(['SWFWG', 'OOOOO', 'WOOOW', 'FOWFW'])
		self.observation_space = gym.spaces.Discrete(1)
		self.dim = (4, 5)
		self.img_map = np.ones(self.dim)
		if stochastic:
			self.slip = True
		
		self.distinct_states = 112
		self.action_space = gym.spaces.Discrete(4)
		self.obstacles = [(0, 1), (0, 3), (2, 0), (2, 4), (3, 2), (3, 4)]
		for x in self.obstacles:
			self.img_map[x[0]][x[1]] = 0
		
		# Clock-wise action slip for stochasticity
		self.slip_action_map = {0: 3, 1: 2, 2: 0, 3: 1, }
		self.slip_probability = 0.1
		self.start_pos = (0, 0)
		self.goal_pos = (0, 4)
		# a lookup table in the form of a dictionary to map indices to cells
		# in the Maze environment:
		self.index_to_coordinate_map = {
			0: (0, 0),
			1: (1, 0),
			2: (3, 0),
			3: (1, 1),
			4: (2, 1),
			5: (3, 1),
			6: (0, 2),
			7: (1, 2),
			8: (2, 2),
			9: (1, 3),
			10: (2, 3),
			11: (3, 3),
			12: (0, 4),
			13: (1, 4), }
		self.coordinate_to_index_map = dict((v, k)
											for k, v in self.index_to_coordinate_map.items())
		self.state = self.coordinate_to_index_map[self.start_pos]
	
	def num2coin(self, n: int):
		"""a method that will handle the coins and their statuses in the Maze,
		where 0 means that the coin wasn't collected by the agent and 1 means that the coin
		was collected by the agent"""
		coinlist = [
			(0, 0, 0),
			(1, 0, 0),
			(0, 1, 0),
			(0, 0, 1),
			(1, 1, 0),
			(1, 0, 1),
			(0, 1, 1),
			(1, 1, 1),
			]
		return list(coinlist[n])
	
	def coin2num(self, v: List):
		if sum(v) < 2:
			return np.inner(v, [1, 2, 3])
		else:
			return np.inner(v, [1, 2, 3]) + 1
	
	def set_state(self, state: int) -> None:
		"""a setter function to set the state of the environment. This is useful for
		algorithms such as value iteration, where each and every state needs to be
		visited in the environment for it to calculate values:
		Args:
            state (int): A valid state in the Maze env int: [0, 112]
		"""
		self.state = state
	
	def reset(self):
		"""return the initial state"""
		self.state = self.coordinate_to_index_map[self.start_pos]
		return self.state
	
	def step(self, action, slip=True):
		"""
		Run one step into the Maze env
        Args:
            state (Any): Current index state of the maze
            action (int): Discrete action for up, down, left, right
            slip (bool, optional): Stochasticity in the env. Defaults to True.

        Raises:
            ValueError: If invalid action is provided as input

        Returns:
            Tuple : Next state, reward, done, _
		"""
		self.slip = slip
		if self.slip:
			if np.random.rand() < self.slip_probability:
				action = self.slip_action_map[action]
		# update the state of the maze based on the action that's taken:
		cell = self.index_to_coordinate_map[int(self.state / 8)]
		if action == 0:
			c_next = cell[1]
			r_next = max(0, cell[0] - 1)
		elif action == 1:
			c_next = cell[1]
			r_next = min(self.dim[0] - 1, cell[0] + 1)
		elif action == 2:
			c_next = max(0, cell[1] - 1)
			r_next = cell[0]
		elif action == 3:
			c_next = min(self.dim[1] - 1, cell[1] + 1)
			r_next = cell[0]
		else:
			raise ValueError(f"Invalid action:L{action}")
		
		# determine whether the agent has reached the goal:
		if (r_next == self.goal_pos[0]) and (c_next == self.goal_pos[1]):
			v_coin = self.num2coin(self.state % 8)
			self.state = (8 * self.coordinate_to_index_map[(r_next, c_next)] + self.state % 8)
			return self.state, float(sum(v_coin)), True
		
		else:
			# handle cases when the action results in hitting an obstacle/wall:
			if (r_next, c_next) in self.obstacles:
				return self.state, 0.0, False
			else:  # e action leads to collecting a coin:
				v_coin = self.num2coin(self.state % 8)
				if (r_next, c_next) == (0, 2):
					v_coin[0] = 1
				elif (r_next, c_next) == (3, 0):
					v_coin[1] = 1
				elif (r_next, c_next) == (3, 3):
					v_coin[2] = 1
				self.state = 8 * self.coordinate_to_index_map[(r_next, c_next)] + self.coin2num(
						v_coin)
				return self.state, 0.0, False
	
	def render(self):
		"""implement a render function that will print out a text version of the current
		state of the Maze environment:"""
		cell = self.index_to_coordinate_map[int(self.state / 8)]
		desc = self.map.tolist()
		desc[cell[0]] = (
				desc[cell[0]][:cell[1]]
				+ "\x1b[1;34m"  # Blue font
				+ "\x1b[4m"  # Underline
				+ "\x1b[1m"  # Bold
				+ "\x1b[7m"  # Reversed
				+ desc[cell[0]][cell[1]]
				+ "\x1b[0m"
				+ desc[cell[0]][cell[1] + 1:]
		)
		print("\n".join("".join(row) for row in desc))


if __name__ == "__main__":
	env = MazeEnv()
	obs = env.reset()
	env.render()
	done = False
	step_num = 1
	action_list = ['UP', 'DOWN', 'RIGHT', 'LEFT']
	while not done:
		# sample a random action from the action space
		action0 = env.action_space.sample()
		next_obs, reward, done = env.step(action0)
		print(f"step {step_num} action:L {action_list[action0]} reward:{reward} done: {done}")
		step_num += 1
		env.render()
	
	env.close()

"""
Our map, as defined in step 1 in the How to do it... section, represents the state of the
learning environment. The Maze environment defines the observation space, the action
space, and the rewarding mechanism for implementing a Markov decision process (MDP).
We sampled a valid action from the action space of the environment and stepped the
environment with the chosen action, which resulted in us getting the new observation,
reward, and a done status Boolean (representing whether the episode has finished) as
the response from the Maze environment. The env.render() method converts the environment's
internal grid representation into a simple text/string grid and prints it for
easy visual understanding.
我们的地图，如“如何做到”部分中的步骤1所定义的，代表了学习环境的状态。迷宫环境定义了观察空间、行动空间和
实施马可夫决策过程(mDP)的奖励机制。我们从环境的行为空间中采样了一个有效的行为，并用所选择的行为步进了
环境，这导致我们得到了新的观察、奖励和一个已完成的状态布尔值(表示该事件是否已经完成)作为迷宫环境的响应。
Render ()方法将环境的内部网格表示转换为一个简单的文本/字符串网格，并将其打印出来，以便于视觉理解。
"""
