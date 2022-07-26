"""
The agent's location is represented by the blue cell in the grid, while the goal and a
mine/bomb/obstacle's location is represented in the grid using green and red cells,
respectively. The agent (blue cell) needs to find its way through the grid to reach the goal
(green cell) without running over the mine/bomb (red cell).
The learning environment is a simulator that provides the
observation for the RL agent, supports a set of actions that the RL agent can perform by
executing the actions, and returns the resultant/new observation as a result of the agent
taking the action.
"""

import copy
import sys
import gym
import numpy as np
from collections import defaultdict
from SimpleImageViewer import SimpleImageViewer

EMPTY = BLACK = 0
WALL = GRAY = 1
AGENT = BLUE = 2
BOMB = RED = 3
GOAL = GREEN = 4
SUCCESS = PINK = 5

COLOR_MAP = {
	BLACK: [0.0, 0.0, 0.0],
	GRAY: [0.5, 0.5, 0.5],
	BLUE: [0.0, 0.0, 1.0],
	RED: [1.0, 0.0, 0.0],
	GREEN: [0.0, 1.0, 0.0],
	}
# action mapping
NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4


class GridworldEnv(gym.Env):
	def __init__(self, max_steps=100):
		self.grid_layout = """
		1 1 1 1 1 1 1 1
        1 2 0 0 0 0 0 1
        1 0 1 1 1 0 0 1
        1 0 1 0 1 0 0 1
        1 0 1 4 1 0 0 1
        1 0 3 0 0 0 0 1
        1 0 0 0 0 0 0 1
        1 1 1 1 1 1 1 1
		"""
		
		self.initial_grid_state = np.fromstring(self.grid_layout, dtype=int, sep=" ")
		self.initial_grid_state = self.initial_grid_state.reshape(8, 8)
		self.grid_state = copy.deepcopy(self.initial_grid_state)
		self.observation_space = gym.spaces.Box(low=0, high=6, shape=self.grid_state.shape)
		self.img_shape = [256, 256, 3]
		self.metadata = {"render.modes": ['human']}
		
		# define the action space and the mapping between the actions and the movement of the
		# agent in the grid:
		self.action_space = gym.spaces.Discrete(5)
		self.actions = [NOOP, UP, DOWN, LEFT, RIGHT]
		self.action_pos_dict = defaultdict(
				lambda: [0, 0],
				{
					NOOP: [0, 0],
					UP: [-1, 0],
					DOWN: [1, 0],
					LEFT: [0, -1],
					RIGHT: [0, 1]
					}, )
		# wrap up the __init__ function by initializing the agent's start and goal
		# states using the get_state() method (which we will implement in the next step):
		self.agent_state, self.goal_state = self.get_state()
		self.step_num = 0  # to keep track of number of steps
		self.max_steps = max_steps
		self.done = False
		self.info = {"status": "Live"}
		self.viewer = None
	
	def get_state(self):
		"""returns the start and goal state for the Gridworld environment:"""
		start_state = np.where(self.grid_state == AGENT)
		goal_state = np.where(self.grid_state == GOAL)
		start_or_goal_not_found = not (start_state[0] and goal_state[0])
		if start_or_goal_not_found:
			sys.exit(
					"Start and/or goal state not present in the gridworld"
					" check the grid layout"
					)
		
		start_state = (start_state[0][0], start_state[1][0])
		goal_state = (goal_state[0][0], goal_state[1][0])
		return start_state, goal_state
	
	def step(self, action):
		"""execute the action and retrieve the next state/observation, the associated reward,
		and whether the episode ended:"""
		action = int(action)
		reward = 0.0
		next_state = (
			self.agent_state[0] + self.action_pos_dict[action][0],
			self.agent_state[1] + self.action_pos_dict[action][1])
		
		next_state_invalid = (next_state[0] < 0 or next_state[0] >= self.grid_state.shape[0]) or (
				next_state[1] < 0 or next_state[1] >= self.grid_state.shape[1])
		
		if next_state_invalid:
			# leave the agent state unchanged
			next_state = self.agent_state
			self.info['status'] = 'Next state is invalid'
		
		next_agent_state = self.grid_state[next_state[0], next_state[1]]
		
		# calculate reward
		if next_agent_state == EMPTY:
			# Move agent from previous state to the next state on the grid
			self.info['status'] = 'Agent moved to a new cell'
			self.grid_state[next_state[0], next_state[1]] = AGENT
			self.grid_state[self.agent_state[0], self.agent_state[1]] = EMPTY
			self.agent_state = copy.deepcopy(next_state)
		elif next_agent_state == WALL:
			self.info['status'] = 'Agent bumped into a wall'
			reward -= 0.1
		# Terminal states
		elif next_agent_state == GOAL:
			self.info['status'] = 'Agent reached the GOAL'
			self.done = True
			reward = 1
		elif next_agent_state == BOMB:
			self.info['status'] = 'Agent stepped on a BOMB'
			self.done = True
			reward = -1
		else:
			self.done = False
		
		self.step_num += 1
		
		# check if max steps per episode has been reached
		if self.step_num >= self.max_steps:
			self.done = True
			self.info['status'] = 'Max steps reached'
		
		if self.done:
			done = True
			terminal_state = copy.deepcopy(self.grid_state)
			terminal_info = copy.deepcopy(self.info)
			
			_ = self.reset()
			return terminal_state, reward, done, terminal_info
		
		return self.grid_state, reward, self.done, self.info
	
	def reset(self) -> np.array:
		"""reset() method, which resets the Gridworld environment when an
		episode completes (or if a request to reset the environment is made):"""
		self.grid_state = copy.deepcopy(self.initial_grid_state)
		self.agent_state, self.goal_state = self.get_state()
		self.step_num = 0
		self.done = False
		self.info['status'] = 'Live'
		return self.grid_state
	
	def gridarray_to_image(self, img_shape=None):
		if img_shape is None:
			img_shape = self.img_shape
		
		observation = np.random.randn(*img_shape) * 0.0
		scale_x = int(observation.shape[0] / self.grid_state.shape[0])
		scale_y = int(observation.shape[1] / self.grid_state.shape[1])
		
		for i in range(self.grid_state.shape[0]):
			for j in range(self.grid_state.shape[1]):
				for k in range(3):
					pixel_value = COLOR_MAP[self.grid_state[i, j]][k]
					observation[
					i * scale_x: (i + 1) * scale_x,
					j * scale_y:(j + 1) * scale_y,
					k
					] = pixel_value
		return (255 * observation).astype(np.uint8)
	
	def render(self, mode='human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return
		
		img = self.gridarray_to_image()
		if mode == 'rgb_array':
			return img
		elif mode == 'human':
			if self.viewer is None:
				self.viewer = SimpleImageViewer()
			self.viewer.imshow(img)
	
	def close(self):
		self.render(close=True)
	
	@staticmethod
	def get_action_meanings():
		return ["NOOP", "DOWN", "UP", "LEFT", "RIGHT"]


if __name__ == "__main__":
	env = GridworldEnv()
	obs = env.reset()
	done = False
	step_num = 1  # run one episode
	while not done:
		# sample a random action from the action space
		action = env.action_space.sample()
		next_obs, reward, done, info = env.step(action)
		print(f'Step: {step_num} reward: {reward} done: {done} info: {info}')
		step_num += 1
		env.render()
	
	env.close()
