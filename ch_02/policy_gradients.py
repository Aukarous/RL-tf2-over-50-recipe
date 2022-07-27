"""
The first one is applying the policy function, which is going to be represented using
a neural network implemented in TensorFlow 2.x.
The second part is applying the Agent class' implementation, while the final part will
be to apply a trainer function, which is used to train the policy gradient-based agent
in a given RL environment.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gym


class PolicyNet(keras.Model):
	def get_config(self):
		pass
	
	def __init__(self, action_dim=1):
		super(PolicyNet, self).__init__()
		self.fc1 = layers.Dense(24, activation='relu')
		self.fc2 = layers.Dense(36, activation='relu')
		self.fc3 = layers.Dense(action_dim, activation='softmax')
	
	def call(self, x):
		"""be called to process inputs to the model"""
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		return x
	
	def process(self, observations):
		"""Process batch observations using call(x) behind-the-scenes"""
		action_probabilities = self.predict_on_batch(observations)
		return action_probabilities


class Agent(object):
	def __init__(self, action_dim=1):
		"""Agent with a neural-network brain powered policy"""
		self.policy_net = PolicyNet(action_dim=action_dim)
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
		self.gamma = 0.99
	
	def policy(self, observation):
		"""a policy helper function that takes an observation as input, has it
		processed by the policy network, and returns the action as the output"""
		observation = observation.reshape(1, -1)
		observation = tf.convert_to_tensor(observation, dtype=tf.float32)
		action_logits = self.policy_net(observation)
		action = tf.random.categorical(tf.math.log(action_logits), num_samples=1)
		return action
	
	def get_action(self, observation):
		"""Another helper function to get the action from the agent"""
		action = self.policy(observation).numpy()
		res = action.squeeze()
		return res
	
	def learn(self, states, rewards, actions):
		""" learning updates for the policy gradient algorithm. Let's
		initialize the learn function with an empty list for discounted rewards"""
		discounted_reward = 0
		discounted_rewards = []
		rewards.reverse()
		# calculate the discounted rewards while using the episodic rewards as input
		for r in rewards:
			discounted_reward = r + self.gamma * discounted_reward
			discounted_rewards.append(discounted_reward)
		discounted_rewards.reverse()
		# crucial step of calculating the policy gradient and update
		# the parameters of the neural network policy using an optimizer:
		for state, reward, action in zip(states, discounted_rewards, actions):
			with tf.GradientTape() as tape:
				action_probabilities = self.policy_net(np.array([state]), training=True)
				loss = self.loss(action_probabilities, action, reward)
			grads = tape.gradient(loss, self.policy_net.trainable_variables)
			self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
	
	def loss(self, action_probabilities, action, reward):
		"""loss function that we referred to in the previous step to
		calculate the policy parameter updates:"""
		dist = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
		log_prob = dist.log_prob(action)
		loss = -log_prob * reward
		return loss


def train(agent: Agent, env: gym.Env, episodes: int, render=True):
	"""Train `agent` in `env` for `episodes`
	
	Args:
		agent: agent to train
		env: environment to train the agent
		episodes: number of episodes to train
		render: True=Enable/False=Disable rendering
	"""
	# begin with the outer loop implementation of the agent training function
	for episode in range(episodes):
		done = False
		state = env.reset()
		total_reward = 0
		rewards, states, actions = [], [], []
		# inner loop to finalize the train function
		while not done:
			action = agent.get_action(state)
			next_state, reward, done, _ = env.step(action)
			rewards.append(reward)
			states.append(state)
			actions.append(action)
			state = next_state
			total_reward += reward
			if render:
				env.render()
			if done:
				agent.learn(states, rewards, actions)
				print('\n')
			print(f"Episode: {episode} ep_reward: {total_reward}", end="\r")


if __name__ == "__main__":
	agent_ = Agent()
	episodes_ = 2  # increase number of episodes to train
	env_ = gym.make("MountainCar-v0")
	# set render=True to visualize Agent's actions in the env
	train(agent_, env_, episodes_, render=True)
	env_.close()
