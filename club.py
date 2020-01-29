import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph, csr_matrix
import scipy
import os 
from utils import *
from scipy.sparse.csgraph import connected_components

class CLUB():
	def __init__(self, dimension, iteration, user_num, item_num, pool_size, item_feature_matrix, true_user_feature_matrix, true_payoffs, alpha, alpha_2, delta, sigma, beta, state):
		self.state=state
		self.dimension=dimension
		self.iteration=iteration
		self.user_num=user_num
		self.item_num=item_num
		self.pool_size=pool_size
		self.item_feature_matrix=item_feature_matrix
		self.true_user_feature_matrix=true_user_feature_matrix
		self.true_payoffs=true_payoffs
		self.user_feature=np.zeros((self.user_num, self.dimension))
		self.I=np.identity(self.user_num)
		self.adj=np.ones((self.user_num, self.user_num))
		self.L=np.zeros((self.user_num, self.user_num))
		self.cluster_list=np.array(list(range(self.user_num)))
		self.cluster_num=0
		self.alpha=alpha
		self.alpha_2=alpha_2
		self.delta=delta
		self.sigma=sigma
		self.beta=beta
		self.beta_list=[]
		self.covariance={i: self.alpha*np.identity(self.dimension) for i in range(self.user_num)}
		self.bias=np.zeros((self.user_num, self.dimension))
		self.served_user_list=[]
		self.user_cluster_cov={i: np.identity(self.dimension) for i in range(self.user_num)}
		self.user_cluster_bias=np.zeros((self.user_num, self.dimension))
		self.user_cluster_feature=np.zeros((self.user_num, self.dimension))
		self.CBPrime = np.zeros(self.user_num)
		self.user_counters = np.zeros(self.user_num)

	def update_cluster_by_connected_components(self, user_index):
		self.cluster_num, self.cluster_list=connected_components(csr_matrix(self.adj))

	def update_graph(self, user_index):
		user_f_diff=np.linalg.norm(self.user_feature[user_index]-self.user_feature, axis=1)
		cb_prime_sum=self.CBPrime[user_index]+self.CBPrime
		ratio=user_f_diff/cb_prime_sum
		big_index=ratio>1.0
		self.adj[big_index, user_index]=0.0
		self.adj[user_index][big_index]=0.0

	def update_cluster_feature(self, user_index):
		user_cluster_index=self.cluster_list[user_index]
		all_users_in_the_cluster=list(np.where(self.cluster_list==user_cluster_index)[0])
		self.user_cluster_cov[user_index]=np.identity(self.dimension)
		self.user_cluster_bias[user_index]=np.zeros(self.dimension)
		for u in all_users_in_the_cluster:
			self.user_cluster_cov[user_index]+=self.covariance[u]-np.identity(self.dimension)
			self.user_cluster_bias[user_index]+=self.bias[u]

		new_cluster_feature=np.dot(np.linalg.pinv(self.user_cluster_cov[user_index]), self.user_cluster_bias[user_index])
		for u in all_users_in_the_cluster:
			self.user_cluster_feature[u]=new_cluster_feature

	def update_user_feature(self, true_payoff, selected_item_feature, user_index):
		self.covariance[user_index]+=np.outer(selected_item_feature, selected_item_feature)
		self.bias[user_index]+=true_payoff*selected_item_feature
		self.user_feature[user_index]=np.dot(np.linalg.pinv(self.covariance[user_index]), self.bias[user_index])
		self.CBPrime[user_index]=self.alpha_2*np.sqrt(float(1+np.log(1+self.user_counters[user_index])/float(1+self.user_counters[user_index])))
		self.user_counters[user_index]+=1

	def update_beta(self, user_index, time):
		cluster_cov=self.user_cluster_cov[user_index]
		cluster_cov_inv=np.linalg.pinv(cluster_cov)
		a=np.linalg.det(cluster_cov)**(1/2)
		b=np.linalg.det(self.alpha*np.identity(self.dimension))**(-1/2)
		self.beta=self.sigma*np.sqrt(2*np.log(a*b/self.delta))+np.sqrt(self.alpha)*np.linalg.norm(self.user_feature[user_index])
		self.beta=np.sqrt(self.alpha)+np.sqrt(2*np.log(1/self.delta)+self.dimension*np.log(1+self.iteration/(self.dimension*self.alpha)))

		self.beta_list.extend([self.beta])

	def select_item(self, user_index, item_pool, time):
		cluster_cov=self.user_cluster_cov[user_index]
		cluster_cov_inv=np.linalg.pinv(cluster_cov)
		est_payoffs=[]
		self.update_beta(user_index, time)
		for it in item_pool:
			x=self.item_feature_matrix[it]
			x_norm=np.sqrt(np.dot(np.dot(x, cluster_cov_inv), x))
			self.beta=0.1*np.sqrt(np.log(time+1))
			est_payoff=np.dot(self.user_cluster_feature[user_index], x)+self.beta*x_norm
			est_payoffs.extend([est_payoff])

		itt=np.argmax(est_payoffs)
		id_=item_pool[itt]
		selected_item_feature=self.item_feature_matrix[id_]
		true_payoff=self.true_payoffs[user_index, id_]+np.random.normal(scale=self.sigma)
		true_max_payoff=np.max(self.true_payoffs[user_index][item_pool])
		regret=true_max_payoff-true_payoff
		return true_payoff, regret, selected_item_feature, x_norm

	def run(self, user_array, item_pool_array, iteration):
		regret_error=[0]
		learning_error=[]
		cluster_num=[]
		x_norm_list=[]
		for time in range(iteration):
			print('time/iteration', time, iteration, '~~~ CLUB')
			item_pool=item_pool_array[time]
			user_index=user_array[time]
			self.served_user_list.extend([user_index])
			self.served_user_list=list(np.unique(self.served_user_list))
			true_payoff, regret, selected_item_feature, x_norm=self.select_item(user_index, item_pool, time)
			x_norm_list.extend([x_norm])
			self.update_user_feature(true_payoff, selected_item_feature, user_index)
			self.update_graph(user_index)
			self.update_cluster_by_connected_components(user_index)
			self.update_cluster_feature(user_index)
			regret_error.extend([regret_error[-1]+regret])
			learning_error.extend([np.linalg.norm(self.true_user_feature_matrix-self.user_feature)])
			cluster_num.extend([self.cluster_num])
		return regret_error[1:], learning_error, cluster_num, self.beta_list, x_norm_list






