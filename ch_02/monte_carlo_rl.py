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


def monte_carlo_prediction():



def outer_loop():
	for episode in range(max_episodes):
	
	
	







































