"""
In environments where the action space is continuous, meaning that the actions are
real-valued, a real-valued, continuous policy distribution is necessary. A continuous
probability distribution can be used to represent an RL agent's policy when the action
space of the environment contains real numbers. In a general sense, such distributions can
be used to describe the possible results of a random variable when the random variable
can take any (real) value.
在行动空间是连续的环境中，意味着行动是实值的，一个实值的，连续的策略分布是必要的。
当环境的操作空间包含实数时，连续概率分布可以用来表示 RL 代理的策略。
在一般意义上，这样的分布可以用来描述当随机变量可以取任何(实)值时随机变量的可能结果。


"""
from typing import List

import tensorflow_probability as tfp
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gym
import gym_gridworlds
from tqdm import tqdm
from build_env_mechanism import GridworldEnv

mu, sigma = 0.0, 1.0
continuous_policy = tfp.distributions.Normal(loc=mu, scale=sigma)

# sample_actions = continuous_policy.sample(10)

for i in range(10):
	action = continuous_policy.sample(10)
	print(action)

mu_1, covariance_diag = [0.0, 0.0], [3.0, 3.0]
continuous_multidim_policy = tfp.distributions.MultivariateNormalDiag(loc=mu_1,
																	  scale_diag=covariance_diag)
for i in range(10):
	action = continuous_multidim_policy.sample(1)
	print(action)

sample_actions = continuous_multidim_policy.sample(500)
sns.jointplot(sample_actions[:, 0], sample_actions[:, 1], kind='scatter')
plt.show()


class ContinuousPolicy(object):
	def __init__(self, action_dim):
		self.distribution = None
		self.action_dim = action_dim
	
	def sample(self, mu, var):
		self.distribution = tfp.distributions.Normal(loc=mu, scale=sigma)
		return self.distribution.sample(1)
	
	def get_action(self, mu, var):
		action_ = self.sample(mu, var)
		return action_


class Brain(keras.Model):
	def get_config(self):
		pass
	
	def __init__(self, action_dim=5, input_shape=(1, 8 * 8)):
		"""Initialize the Agent's Brain model
		 Args:
		     action_dim (int): Number of actions
		 """
		super(Brain, self).__init__()
		self.dense1 = layers.Dense(32, input_shape=input_shape, activation='relu')
		self.logits = layers.Dense(action_dim)
	
	def call(self, inputs, training=None, mask=None):
		x = tf.convert_to_tensor(inputs)
		# if len(x.shape) >= 2 and x.shape[0] != 1:
		# 	x = tf.reshape(x, (1, -1))
		logits = self.logits(self.dense1(x))
		return logits
	
	def process(self, observations):
		"""Process batch observations using `call(inputs)` behind-the-scenes"""
		action_logits = self.predict_on_batch(observations)
		return action_logits


class Agent(object):
	"""
	simple agent class that utilizes the ContinuousPolicy object
	to act in continuous action space environments:
	"""
	
	def __init__(self, action_dim=5, input_dim=(1, 8 * 8)):
		self.brain = Brain(action_dim, input_dim)
		self.brain.compile(loss='categorical_crossentropy',
						   optimizer='adam',
						   metrics=['accuracy'])
		self.policy = self.policy_mlp
	
	def policy_mlp(self, observations: np.array):
		observations = observations.reshape(1, -1)
		action_logits = self.brain.process(observations)
		action_ = tf.random.categorical(tf.math.log(action_logits), num_samples=1)
		return action_  # tf.squeeze(action, axis=1)
	
	def get_action(self, observations: np.array):
		return self.policy(observations)
	
	def learn(self, samples):
		raise NotImplementedError


class ContinuousMultiDimensionalPolicy(object):
	def __init__(self, num_actions):
		self.action_dim = num_actions
		self.distribution = None
	
	def sample(self, mu, convariance_diag):
		self.distribution = tfp.distributions.MultivariateNormalDiag(loc=mu,
																	 scale_diag=covariance_diag)
		return self.distribution.sample(1)
	
	def get_action(self, mu, covariance_diag_):
		action_ = self.sample(mu, covariance_diag_)
		return action_


def evaluate(agent: gym, env: gym.Env, render=True):
	"""
	evaluate an agent in an environment with a continuous action space to
	assess episodic performance
	"""
	obs, episode_reward, done, step_num = env.reset(), 0.0, False, 0
	while not done:
		action_ = agent.get_action(obs)
		obs, reward, done, info_ = env.step(action_)
		episode_reward += reward
		step_num += 1
		if render:
			env.render()
	return step_num, episode_reward, done, info_


if __name__ == "__main__":
	env = gym.make('Gridworld-v0')
	agent = Agent(env.action_space.n, env.observation_space.shape)
	for episode in tqdm(range(10)):
		steps, episode_reward, done, info = evaluate(agent, env)
		print(f"EpReward: {episode_reward: .2f} steps: {steps} done:{done} info:{info}")
	
	env.close()
