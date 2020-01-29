import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy

class GOB():
	def __init__(self, dimension,iteration, user_num, item_num, pool_size, item_feature_matrix, true_user_feature_matrix, true_payoffs, true_adj, true_lap, alpha, delta, sigma, b, state):
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
		self.true_user_feature_vector=true_user_feature_matrix.flatten()
		self.user_feature_vector=np.zeros(self.user_num*self.dimension)
		self.user_feature_matrix=np.zeros((self.user_num, self.dimension))
		self.user_feature_matrix_converted=np.zeros((self.user_num, self.dimension))
		self.user_ridge=np.zeros((self.user_num, self.dimension))
		self.user_ls=np.zeros((self.user_num, self.dimension))
		self.I=np.identity(self.user_num*self.dimension)
		self.adj=true_adj
		self.lap=true_lap
		self.L=self.lap.copy()+np.identity(self.user_num)
		self.A=np.kron(self.L, np.identity(self.dimension))
		self.A_inv=np.linalg.pinv(self.A)
		self.A_inv_sqrt=scipy.linalg.sqrtm(self.A_inv)
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.beta=0
		self.b=b
		self.covariance=np.identity(self.user_num*self.dimension)
		self.bias=np.zeros(self.user_num*self.dimension)
		self.beta_list=[]
		self.graph_error=[]
		self.user_xx={}
		self.user_v={}
		self.user_bias={}
		self.user_counter={}
		self.real_beta_list=[]

	def initial(self):
		for u in range(self.user_num):
			self.user_xx[u]=np.zeros((self.dimension, self.dimension))
			self.user_v[u]=self.alpha*np.identity(self.dimension)
			self.user_bias[u]=np.zeros(self.dimension)
			self.user_counter[u]=0

	def update_beta(self):
		a=np.linalg.det(self.covariance)
		#b=np.linalg.det(self.alpha*self.A)**(-1/2)
		self.beta=self.sigma*np.sqrt(np.log(a/self.delta))+np.sqrt(self.alpha)*np.linalg.norm(np.dot(np.real(self.A_inv_sqrt), self.user_feature_vector))
		self.beta_list.extend([self.beta])
		diff=self.user_feature_vector-self.true_user_feature_vector
		real_beta=np.sqrt(np.dot(np.dot(diff, self.covariance), diff))
		self.real_beta_list.extend([real_beta])
		
	def select_item(self, item_pool, user_index, time):
		item_fs=self.item_feature_matrix[item_pool]
		item_feature_array=np.zeros((self.pool_size, self.user_num*self.dimension))
		item_feature_array[:,user_index*self.dimension:(user_index+1)*self.dimension]=item_fs
		estimated_payoffs=np.zeros(self.pool_size)
		cov_inv=np.linalg.pinv(self.covariance)
		if self.state==False:
			self.update_beta()
			for j in range(self.pool_size):
				item_index=item_pool[j]
				x=self.item_feature_matrix[item_index]
				x_long=np.zeros((self.dimension*self.user_num))
				x_long[user_index*self.dimension:(user_index+1)*self.dimension]=x
				co_x=np.dot(self.A_inv_sqrt, x_long)
				x_norm=np.sqrt(np.dot(np.dot(co_x, cov_inv), co_x))
				self.beta=0.1*np.sqrt(np.log(time+1))
				est_y=np.dot(self.user_feature_vector, co_x)+self.beta*x_norm
				estimated_payoffs[j]=est_y
				ucb=self.beta_list[time]*x_norm
		else: 
			for j in range(self.pool_size):
				item_index=item_pool[j]
				x=self.item_feature_matrix[item_index]
				x_long=np.zeros((self.dimension*self.user_num))
				x_long[user_index*self.dimension:(user_index+1)*self.dimension]=x
				co_x=np.dot(np.real(self.A_inv_sqrt), x_long)
				x_norm=np.sqrt(np.dot(np.dot(co_x, cov_inv),co_x))
				est_y=np.dot(self.user_feature_vector, co_x)+self.beta*x_norm*np.sqrt(np.log(time+1))
				estimated_payoffs[j]=est_y

		max_index=np.argmax(estimated_payoffs)
		selected_item_index=item_pool[max_index]
		selected_item_feature=item_fs[max_index]
		true_payoff=self.true_payoffs[user_index, selected_item_index]+np.random.normal(scale=self.sigma)
		max_ideal_payoff=np.max(self.true_payoffs[user_index][item_pool])
		regret=max_ideal_payoff-true_payoff
		return true_payoff, selected_item_feature, regret, x_norm, ucb

	def update_user_feature(self, true_payoff, selected_item_feature, user_index):
		x_long=np.zeros(self.dimension*self.user_num)
		x_long[user_index*self.dimension:(user_index+1)*self.dimension]=selected_item_feature
		co_x=np.dot(np.real(self.A_inv_sqrt), x_long)
		self.user_xx[user_index]+=np.outer(selected_item_feature, selected_item_feature)
		self.user_v[user_index]+=np.outer(selected_item_feature, selected_item_feature)
		self.user_bias[user_index]+=true_payoff*selected_item_feature
		self.covariance+=np.outer(co_x, co_x)
		self.bias+=true_payoff*co_x
		cov_inv=np.linalg.pinv(self.covariance)
		self.user_feature_vector=np.dot(cov_inv, self.bias)
		self.user_feature_matrix=self.user_feature_vector.reshape((self.user_num, self.dimension))
		self.user_feature_matrix_converted=np.dot(np.real(self.A_inv_sqrt),self.user_feature_vector).reshape((self.user_num, self.dimension))


	def run(self, user_array, item_pool_array, iteration):
		cumulative_regret=[0]
		learning_error_list=np.zeros(iteration)
		x_norm_list=[]
		sum_x_norm=[0]
		ucb_list=[]
		self.initial()		
		for time in range(iteration):	
			print('time/iteration', time, iteration, '~~~GOB')
			user_index=user_array[time]
			item_pool=item_pool_array[time]
			self.user_counter[user_index]+=1
			true_payoff, selected_item_feature, regret, x_norm, ucb=self.select_item(item_pool, user_index, time)
			x_norm_list.extend([x_norm])
			ucb_list.extend([ucb])
			self.update_user_feature(true_payoff, selected_item_feature, user_index)
			cumulative_regret.extend([cumulative_regret[-1]+regret])
			error=np.linalg.norm(self.user_feature_matrix_converted.flatten()-self.true_user_feature_vector)
			learning_error_list[time]=error
			sum_x_norm.extend([sum_x_norm[-1]+x_norm])


		return np.array(cumulative_regret[1:]), learning_error_list, self.beta_list, x_norm_list, ucb_list, sum_x_norm[1:], self.real_beta_list
