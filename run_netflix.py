import numpy as np 
import networkx as nx
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph  
import scipy
import os 
from sklearn import datasets
os.chdir('/Kaige_Research/Code/graph_bandit/code/')
from linucb import LINUCB
from t_sampling import TS
from gob import GOB 
from lapucb import LAPUCB
from lapucb_sim import LAPUCB_SIM
from club import CLUB
from utils import *
path='../bandit_results/netflix/'
#np.random.seed(2018)

user_num=20
item_num=300
dimension=10
pool_size=20
iteration=1000
loop=5
sigma=0.01# noise
delta=0.1# high probability
alpha=0.25# regularizer
alpha_2=0.125# edge delete CLUB
beta=0.125 # exploration for CLUB, SCLUB and GOB
state=False # False for artificial dataset, True for real dataset
thres=0

input_path='../processed_data/netflix/'
user_feature_matrix_ori=np.load(input_path+'user_feature_matrix_100.npy')
item_feature_matrix_ori=np.load(input_path+'item_feature_matrix_500.npy')

user_length=user_feature_matrix_ori.shape[0]
item_length=item_feature_matrix_ori.shape[0]


linucb_regret_matrix=np.zeros((loop, iteration))
linucb_error_matrix=np.zeros((loop, iteration))
gob_regret_matrix=np.zeros((loop, iteration))
gob_error_matrix=np.zeros((loop, iteration))
lapucb_regret_matrix=np.zeros((loop, iteration))
lapucb_error_matrix=np.zeros((loop, iteration))
lapucb_sim_regret_matrix=np.zeros((loop, iteration))
lapucb_sim_error_matrix=np.zeros((loop, iteration))
club_regret_matrix=np.zeros((loop, iteration))
club_error_matrix=np.zeros((loop, iteration))
gob_graph_matrix=np.zeros((loop, iteration))
lapucb_graph_matrix=np.zeros((loop, iteration))
lapucb_sim_graph_matrix=np.zeros((loop, iteration))

for l in range(loop):
	print('loop/total_loop', l, loop)
	user_list=np.random.choice(range(user_length), size=user_num)
	item_list=np.random.choice(range(item_length), size=300)
	user_feature_matrix=user_feature_matrix_ori[user_list]
	item_feature_matrix=item_feature_matrix_ori[item_list]
	noise_matrix=np.random.normal(scale=sigma, size=(user_num, item_num))
	true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)+noise_matrix
	user_seq=np.random.choice(range(user_num), size=iteration)
	item_pool_seq=np.random.choice(range(item_num), size=(iteration, pool_size))

	true_adj=rbf_kernel(user_feature_matrix, gamma=0.75)
	np.fill_diagonal(true_adj,0)

	# payoffs=true_payoffs.ravel()
	# plt.figure(figsize=(5,5))
	# plt.hist(payoffs)
	# plt.ylabel('Counts', fontsize=12)
	# plt.xlabel('Payoffs', fontsize=12)
	# plt.tight_layout()
	# plt.savefig(path+'hist_payoffs_netflix'+'.png', dpi=100)
	# plt.show()


	# edges=true_adj.ravel()
	# plt.figure(figsize=(5,5))
	# plt.hist(edges)
	# plt.ylabel('Counts', fontsize=12)
	# plt.xlabel('Edge weights', fontsize=12)
	# plt.tight_layout()
	# plt.savefig(path+'hist_edge_weights_netflix'+'.png', dpi=100)
	# plt.show()

	D=np.diag(np.sum(true_adj, axis=1))
	lap=D-true_adj
	true_lap=np.zeros((user_num, user_num))
	for i in range(user_num):
		for j in range(user_num):
			if D[i,i]==0:
				true_lap[i,j]=0
			else:
				true_lap[i,j]=-true_adj[i,j]/D[i,i]
	np.fill_diagonal(true_lap, 1)


	linucb_model=LINUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, 0.25, delta, sigma, state)
	gob_model=GOB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, true_lap, 0.5, delta, sigma, 0.15, state)
	lapucb_model=LAPUCB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, true_lap, noise_matrix, 0.25, delta, sigma, beta, thres, state)
	lapucb_sim_model=LAPUCB_SIM(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, true_lap, noise_matrix, 0.25, delta, sigma, beta, thres, state)
	club_model = CLUB(dimension, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, 0.25, 0.2, delta, sigma, beta, state)

	linucb_regret, linucb_error, linucb_beta, linucb_x_norm, linucb_inst_regret, linucb_ucb, linucb_sum_x_norm, linucb_real_beta=linucb_model.run(user_seq, item_pool_seq, iteration)
	gob_regret, gob_error, gob_beta,  gob_x_norm, gob_ucb, gob_sum_x_norm, gob_real_beta=gob_model.run(user_seq, item_pool_seq, iteration)
	lapucb_regret, lapucb_error, lapucb_beta, lapucb_x_norm, lapucb_inst_regret, lapucb_ucb, lapucb_sum_x_norm, lapucb_real_beta=lapucb_model.run(user_seq, item_pool_seq, iteration)
	lapucb_sim_regret, lapucb_sim_error, lapucb_sim_beta, lapucb_sim_x_norm, lapucb_sim_avg_norm, lapucb_sim_inst_regret, lapucb_sim_ucb, lapucb_sim_sum_x_norm=lapucb_sim_model.run( user_seq, item_pool_seq, iteration)
	club_regret, club_error,club_cluster_num, club_beta, club_x_norm=club_model.run(user_seq, item_pool_seq, iteration)

	linucb_regret_matrix[l], linucb_error_matrix[l]=linucb_regret, linucb_error
	gob_regret_matrix[l], gob_error_matrix[l]=gob_regret, gob_error
	lapucb_regret_matrix[l], lapucb_error_matrix[l]=lapucb_regret, lapucb_error
	lapucb_sim_regret_matrix[l], lapucb_sim_error_matrix[l]=lapucb_sim_regret, lapucb_sim_error
	club_regret_matrix[l], club_error_matrix[l]=club_regret, club_error


linucb_regret=np.mean(linucb_regret_matrix, axis=0)
linucb_error=np.mean(linucb_error_matrix, axis=0)
gob_regret=np.mean(gob_regret_matrix, axis=0)
gob_error=np.mean(gob_error_matrix, axis=0)

lapucb_regret=np.mean(lapucb_regret_matrix, axis=0)
lapucb_error=np.mean(lapucb_error_matrix, axis=0)

lapucb_sim_regret=np.mean(lapucb_sim_regret_matrix, axis=0)
lapucb_sim_error=np.mean(lapucb_sim_error_matrix, axis=0)

club_regret=np.mean(club_regret_matrix, axis=0)
club_error=np.mean(club_error_matrix, axis=0)



plt.figure(figsize=(5,5))
plt.plot(linucb_regret,'-.', markevery=0.1,linewidth=2, markersize=8, label='LinUCB')
plt.plot(gob_regret, '-p', color='orange', markevery=0.1,linewidth=2, markersize=8, label='GOB.Lin')
plt.plot(lapucb_sim_regret, '-s', color='g', markevery=0.1,linewidth=2, markersize=8, label='GraphUCB-Local')
plt.plot(lapucb_regret, '-o', color='r', markevery=0.1,linewidth=2, markersize=8, label='GraphUCB')
plt.plot(club_regret,'-*',color='k',  markevery=0.1,linewidth=2, markersize=8, label='CLUB')
plt.ylabel('Cumulative Regret', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.legend(loc=0, fontsize=16)
plt.tight_layout()
plt.savefig(path+'netflix_backup'+'.png', dpi=100)
plt.show()



plt.figure(figsize=(5,5))
plt.plot(linucb_error,'-.', markevery=0.1, label='LinUCB')
plt.plot(gob_error, '-p', color='orange', markevery=0.1, label='GOB.Lin')
plt.plot(lapucb_sim_error, '-s',color='g',  markevery=0.1, label='GraphUCB-Local')
plt.plot(lapucb_error, '-o',color='r',  markevery=0.1, label='GraphUCB')
plt.plot(club_error, '-*', color='k', markevery=0.1,  label='CLUB')
plt.ylabel('Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.tight_layout()
#plt.savefig(path+'error'+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(club_cluster_num, label='Club')
plt.ylabel('cluster num')
plt.xlabel('Time')
plt.show()



