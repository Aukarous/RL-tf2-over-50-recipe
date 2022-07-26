import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from typing import List
import numpy as np

binary_policy = tfp.distributions.Bernoulli(probs=0.5)
for i in range(5):
	action = binary_policy.sample(1)
	print('action:', action)

sample_actions = binary_policy.sample(500)
sns.distplot(sample_actions)
plt.show()

action_dim = 4  # dimension of the discrete action space
action_probabilities = [0.25, 0.25, 0.25, 0.25]
discrete_policy = tfp.distributions.Multinomial(probs=action_probabilities, total_count=1)
for i in range(5):
	action = discrete_policy.sample(1)
	print(action)
sns.distplot(discrete_policy.sample(1))
plt.show()


def entropy(action_probs: List):
	# calculate the entropy of a discrete policy
	return -tf.reduce_sum(action_probs * tf.math.log(action_probs), axis=-1)


print(entropy(action_probabilities))


class DiscretePolicy(object):
	# discrete policy class
	def __int__(self, num_actions):
		self.action_dim = num_actions
	
	def sample(self, action_logits):
		self.distribution = tfp.distributions.Multinomial(logits=action_logits, total_count=1)
		return self.distribution.sample(1)
	
	def get_action(self, action_logits):
		action = self.sample(action_logits)
		return np.where(action)[-1]  # return the action index
	
	def entropy(self, action_probabilities):
		return -tf.reduce_sum(action_probabilities * tf.math.log(action_probabilities), axis=-1)
	
	def evaluate(self, agent, env, render=True):
		"""ealuate the agent in a given environment:"""
		obs, episode_reward, done, step_num = env.reset(), 0.0, False, 0
		while not done:
			action = agent.get_action(obs)
			obs, reward, done, info = env.step(action)
			episode_reward += reward
			step_num += 1
			if render:
				env.render()
		return step_num, episode_reward, done, info
