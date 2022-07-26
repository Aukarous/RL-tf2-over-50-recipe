from collections import namedtuple
import gym
import gym_gridworlds
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from build_env_mechanism import GridworldEnv


class Brain(keras.Model):
	def get_config(self):
		pass
	
	def __init__(self, action_dim=5, input_shape=(1, 8 * 8)):
		super(Brain, self).__init__()
		self.dense1 = layers.Dense(32, input_shape=input_shape, activation='relu')
		self.logits = layers.Dense(action_dim)
	
	def call(self, inputs):
		x = tf.convert_to_tensor(inputs)
		# if len(x.shape) >= 2 and x.shape[0] != 1:
		# 	x = tf.reshape(x, (1, -1))
		
		return self.logits(self.dense1(x))
	
	def process(self, observations):
		action_logits = self.predict_on_batch(observations)
		return action_logits


class Agent(object):
	def __init__(self, action_dim=5, input_shape=(1, 8 * 8)):
		self.brain = Brain(action_dim, input_shape)
		self.brain.compile(loss='categorical_crossentropy',
						   optimizer='adam',
						   metrics=['accuracy'])
		self.policy = self.policy_mlp
	
	def policy_mlp(self, observations):
		# observations = observations.reshape(-1, 1)	# ????????
		action_logits = self.brain.process(observations)
		action = tf.random.categorical(tf.math.log(action_logits), num_samples=1)
		return tf.squeeze(action, axis=1)
	
	def get_action(self, observations):
		return self.policy(observations)
	
	def learn(self, samples):
		raise NotImplementedError


def evaluate(agent_, env: gym.Env, render=True):
	obs, episode_reward, done, step_num = env.reset(), 0.0, False, 0	# obs is None ！！
	while not done:
		action = agent_.get_action(obs)
		obs, reward, done, info = env.step(action)
		episode_reward += reward
		step_num += 1
		if render:
			env.render()
	
	return step_num, episode_reward, done, info


def rollout(agent, env: gym.Env, render=False):
	obs, episode_reward, done, step_num = env.reset(), 0.0, False, 0
	observations, actions = [], []
	episode_reward = 0.0
	while not done:
		action = agent.get_action(obs)
		next_obs, reward, done, info = env.step(action)
		# save experience
		observations.append(np.array(obs).reshape(1, -1))
		# convert to numpy & reshape (8,8) to (1, 64)
		actions.append(action)
		episode_reward += reward
		obs = next_obs
		step_num += 1
		if render:
			env.render()
	env.close()
	return observations, actions, episode_reward


if __name__ == "__main__":
	env = gym.make('Gridworld-v0')
	agent = Agent(env.action_space.n, env.observation_space.shape)
	for episode in tqdm(range(10)):
		steps, episode_reward, done, info = evaluate(agent, env)
		print(f'Ep_Reward: {episode_reward: .2f} steps: {steps}, done: {done} info: {info}')
		
	env.close()

# env_ = GridworldEnv()
# brain = Brain(env_.action_space.n)
# agent = Agent(brain)
# obs_batch, actions_batch, episode_reward = rollout(agent, env=env_)
#
# assert len(obs_batch) == len(actions_batch)
#
# # Let's now roll out multiple complete trajectories to collect experience data:
# # Trajectory: (obs_batch, actions_batch, episode_reward)
# # Rollout 100 episodes; Maximum possible steps = 100 * 100 = 10e4
# trajectories = [rollout(agent, env_, render=True) for _ in tqdm(range(100))]
# sample_ep_rewards = [rollout(agent, env_)[-1] for _ in tqdm(range(1000))]
# plt.hist(sample_ep_rewards, bins=10, histtype='bar')
