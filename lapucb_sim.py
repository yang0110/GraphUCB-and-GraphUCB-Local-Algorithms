import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 

class LAPUCB_SIM(): 
	def __init__(self, dimension,iteration, user_num, item_num, pool_size, item_feature_matrix, true_user_feature_matrix, true_payoffs, true_adj, true_lap, alpha, delta, sigma, beta, thres, state):
		self.true_adj=true_adj
		self.state=state
		self.dimension=dimension
		self.iteration=iteration
		self.user_num=user_num
		self.item_num=item_num
		self.pool_size=pool_size
		self.item_feature_matrix=item_feature_matrix
		self.true_user_feature_matrix=true_user_feature_matrix
		self.true_payoffs=true_payoffs
		self.user_feature_matrix=np.zeros((self.user_num, self.dimension))
		self.thres=thres
		self.adj=true_adj
		self.lap=true_lap
		self.L=self.lap.copy()+0.01*np.identity(self.user_num)
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.beta=beta
		self.user_bias={}
		self.user_v={}
		self.user_xx={}
		self.user_avg={}
		self.user_ridge=np.zeros((self.user_num, self.dimension))
		self.user_ls=np.zeros((self.user_num, self.dimension))
		self.beta_list=[]
		self.user_counter={}
		self.graph_error=[]
		self.user_h={}

	def initialized_parameter(self):
		for u in range(self.user_num):
			self.user_v[u]=self.alpha*np.identity(self.dimension)
			self.user_avg[u]=np.zeros(self.dimension)
			self.user_xx[u]=0.1*np.identity(self.dimension)
			self.user_bias[u]=np.zeros(self.dimension)
			self.user_counter[u]=0
			self.user_h[u]=np.zeros((self.dimension, self.dimension))

	def update_beta(self, user_index):
		sum_A=np.zeros((self.dimension, self.dimension))
		for uu in range(self.user_num):
			if uu==user_index:
				pass
			else:
				sum_A+=((self.L[user_index, uu])**2)*np.linalg.inv(self.user_xx[uu])
		self.user_h[user_index]=self.user_xx[user_index]+self.alpha**2*sum_A+2*self.alpha*self.L[user_index, user_index]*np.identity(self.dimension)
		a=np.linalg.det(self.user_v[user_index])**(1/2)
		b=np.linalg.det(self.alpha*np.identity(self.dimension))**(-1/2)
		d=self.sigma*np.sqrt(2*np.log(a*b/self.delta))
		if self.user_counter[user_index]==0:
			if self.state==1:
				self.user_avg[user_index]=np.dot(self.user_ls.T, self.L[user_index])
			else:
				self.user_avg[user_index]=np.dot(self.true_user_feature_matrix.T, self.L[user_index])
		else:
			if self.state==1:
				self.user_avg[user_index]=np.dot(self.user_ls.T, self.L[user_index])
			else:
				self.user_avg[user_index]=np.dot(self.true_user_feature_matrix.T, self.L[user_index])
		c=np.sqrt(self.alpha)*np.linalg.norm(self.user_avg[user_index])
		self.beta=d+c
		self.beta_list.extend([self.beta])

	def select_item(self, item_pool, user_index, time):
		item_fs=self.item_feature_matrix[item_pool]
		estimated_payoffs=np.zeros(self.pool_size)
		self.update_beta(user_index)
		h_inv=np.linalg.pinv(self.user_h[user_index])
		for j in range(self.pool_size):
			x=item_fs[j]
			x_norm=np.sqrt(np.dot(np.dot(x, h_inv),x))
			mean=np.dot(x, self.user_feature_matrix[user_index])
			est_y=mean+self.beta*x_norm
			estimated_payoffs[j]=est_y
			ucb=self.beta*x_norm

		max_index=np.argmax(estimated_payoffs)
		selected_item_index=item_pool[max_index]
		selected_item_feature=item_fs[max_index]
		true_payoff=self.true_payoffs[user_index, selected_item_index]+np.random.normal(scale=self.sigma)
		max_ideal_payoff=np.max(self.true_payoffs[user_index][item_pool])
		regret=max_ideal_payoff-true_payoff
		return true_payoff, selected_item_feature, regret, x_norm, ucb

	def update_user_feature_upon_ridge(self, true_payoff, selected_item_feature, user_index):
		x=selected_item_feature
		self.user_xx[user_index]+=np.outer(x, x)
		self.user_v[user_index]+=np.outer(x, x)
		self.user_bias[user_index]+=true_payoff*x
		xx_inv=np.linalg.pinv(self.user_xx[user_index])
		v_inv=np.linalg.pinv(self.user_v[user_index])
		self.user_ls[user_index]=np.dot(xx_inv, self.user_bias[user_index])
		self.user_ridge[user_index]=np.dot(v_inv, self.user_bias[user_index])
		for u in range(self.user_num):
			v_inv=np.linalg.pinv(self.user_v[u])
			xx_inv=np.linalg.pinv(self.user_xx[u])
			self.user_avg[u]=np.dot(self.user_ls.T, self.L[u])
			self.user_feature_matrix[u]=self.user_ls[u]-self.alpha*np.dot(v_inv, self.user_avg[u])

	def run(self, user_array, item_pool_array, iteration):
		self.initialized_parameter()
		cumulative_regret=[0]
		learning_error_list=np.zeros(iteration)
		x_norm_list=[]
		sum_x_norm=[0]
		ucb_list=[]
		avg_norm_list=[]
		inst_regret=[]
		for time in range(iteration):
			print('time/iteration', time, iteration,'~~~GraphUCB-Local')
			user_index=user_array[time]
			item_pool=item_pool_array[time]
			true_payoff, selected_item_feature, regret, x_norm, ucb=self.select_item(item_pool,user_index, time)
			x_norm_list.extend([x_norm])
			self.user_counter[user_index]+=1
			self.update_user_feature_upon_ridge(true_payoff, selected_item_feature, user_index)
			error=np.linalg.norm(self.user_feature_matrix-self.true_user_feature_matrix)
			cumulative_regret.extend([cumulative_regret[-1]+regret])
			learning_error_list[time]=error 
			inst_regret.extend([regret])
			ucb_list.extend([ucb])
			sum_x_norm.extend([sum_x_norm[-1]+x_norm])

		return np.array(cumulative_regret[1:]), learning_error_list, self.beta_list, x_norm_list, avg_norm_list,inst_regret, ucb_list, sum_x_norm[1:]
