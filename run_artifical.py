import numpy as np 
import pandas as pd
import networkx as nx
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
sns.set_style("white")
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph  
import scipy
import os 
from sklearn import datasets
os.chdir('C:/DATA/Kaige_Research/Code/graph_bandit/graphucb_code/')
from linucb import LINUCB
from gob import GOB 
from lapucb import LAPUCB
from lapucb_sim import LAPUCB_SIM
from club import CLUB
from utils import *
path='../bandit_results/simulated/'
save_path='../bandit_results/camera_ready_results/'
#np.random.seed(2018)


user_num=10
item_num=500
dimension=5
pool_size=50
iteration=1000
loop=10
sigma=0.01# noise
delta=0.01# high probability
alpha=1 # regularizer
alpha_2=0.15# edge delete CLUB
epsilon=8 # Ts
beta=0.125 # exploration for CLUB, SCLUB and GOB
thres=0.0
state=False # False for artificial dataset, True for real dataset

user_seq=np.random.choice(range(user_num), size=iteration)
item_pool_seq=np.random.choice(range(item_num), size=(iteration, pool_size))
item_feature_matrix=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))

#er_adj=ER_graph(user_num, 0.4)
#np.save(path+'er_binary_graph.npy', er_adj)
er_adj=np.load(path+'er_binary_graph.npy')


#ba_adj=BA_graph(user_num, 5)
#np.save(path+'ba_binary_graph.npy', ba_adj)
ba_adj=np.load(path+'ba_binary_graph.npy')


#ws_adj=WS_graph(user_num, 8, 0.2)
#np.save(path+'ws_binary_graph.npy', ws_adj)
ws_adj=np.load(path+'ws_binary_graph.npy')

random_matrix=np.random.normal(size=(user_num, dimension))
rbf_adj=rbf_kernel(random_matrix, gamma=0.05)
rbf_adj[rbf_adj<0.5]=0
np.fill_diagonal(rbf_adj, 0)
#rbf_adj=np.load(path+'rbf_sparse_graph.npy')
true_adj=rbf_adj.copy()
true_adj=er_adj.copy()
true_adj=ba_adj.copy()
true_adj=ws_adj.copy()


D=np.diag(np.sum(true_adj, axis=1))
true_lap=np.zeros((user_num, user_num))
for i in range(user_num):
	for j in range(user_num):
		if D[i,i]==0:
			true_lap[i,j]=0
		else:
			true_lap[i,j]=-true_adj[i,j]/D[i,i]

np.fill_diagonal(true_lap, 1)

#true_com_lap=csgraph.laplacian(true_adj)
#true_norm_lap=csgraph.laplacian(true_adj, normed=True)


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

for l in range(loop):
	print('loop/total_loop', l, loop)
	user_feature_matrix=dictionary_matrix_generator(user_num, dimension, true_lap, 7)
	true_payoffs=np.dot(user_feature_matrix, item_feature_matrix.T)

	linucb_model=LINUCB(dimension, iteration, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, delta, sigma, state)

	gob_model=GOB(dimension, iteration, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, true_lap, alpha, delta, sigma, beta, state)

	lapucb_model=LAPUCB(dimension, iteration, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, true_lap, alpha, delta, sigma, beta, thres, 1)

	lapucb_sim_model=LAPUCB_SIM(dimension, iteration, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, true_adj, true_lap, alpha, delta, sigma, beta, thres, 1)

	club_model = CLUB(dimension, iteration, user_num, item_num, pool_size, item_feature_matrix, user_feature_matrix, true_payoffs, alpha, alpha_2, delta, sigma, beta, state)

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


linucb_mean=np.mean(linucb_regret_matrix, axis=0)
linucb_sd=linucb_regret_matrix.std(0)

gob_mean=np.mean(gob_regret_matrix, axis=0)
gob_sd=gob_regret_matrix.std(0)

lapucb_mean=np.mean(lapucb_regret_matrix, axis=0)
lapucb_sd=lapucb_regret_matrix.std(0)

lapucb_sim_mean=np.mean(lapucb_sim_regret_matrix, axis=0)
lapucb_sim_sd=lapucb_sim_regret_matrix.std(0)

club_mean=np.mean(club_regret_matrix, axis=0)
club_sd=club_regret_matrix.std(0)

# plt.figure(figsize=(5,5))
# plt.errorbar(range(iteration),linucb_mean, linucb_sd*0.95, markevery=0.2, marker='.')
# plt.show()
x=range(iteration)
plt.figure(figsize=(5,5))
plt.plot(x, linucb_mean, '-.', markevery=0.1, linewidth=2, markersize=8, label='LinUCB')
plt.fill_between(x, linucb_mean-linucb_sd, linucb_mean+linucb_sd, color='b', alpha=0.2)
plt.plot(x, gob_mean, '-p', color='orange', markevery=0.1, linewidth=2, markersize=8, label='Gob.Lin')
plt.fill_between(x, gob_mean-gob_sd, gob_mean+gob_sd, color='orange', alpha=0.2)
plt.plot(x, lapucb_sim_mean, '-s', color='g', markevery=0.1, linewidth=2, markersize=8, label='GraphUCB-Local')
plt.fill_between(x, lapucb_sim_mean-lapucb_sim_sd, lapucb_sim_mean+lapucb_sim_sd, color='g', alpha=0.2)
plt.plot(x, lapucb_mean, '-o', color='r', markevery=0.1, linewidth=2, markersize=8, label='GraphUCB')
plt.fill_between(x, lapucb_mean-lapucb_sd, lapucb_mean+lapucb_sd, color='r', alpha=0.2)
plt.plot(x, club_mean, '-*', color='k', markevery=0.1, linewidth=2, markersize=8, label='CLUB')
plt.fill_between(x, club_mean-club_sd, club_mean+club_sd, color='k', alpha=0.2)
plt.ylabel('Cumulative Regret', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.legend(loc=2, fontsize=14)
plt.tight_layout()
#plt.savefig(save_path+'ws_cum_regret'+'.png', dpi=100)
plt.show()


# df=pd.DataFrame(columns=['Time', 'LinUCB', 'Gob.Lin', 'GraphUCB', 'GraphUCB-Local', 'CLUB'])
# time_points=list(range(iteration))*loop
# df['Time']=time_points
# df['LinUCB']=linucb_regret_matrix.ravel()
# df['Gob.Lin']=gob_regret_matrix.ravel()
# df['GraphUCB']=lapucb_regret_matrix.ravel()
# df['GraphUCB-Local']=lapucb_sim_regret_matrix.ravel()
# df['CLUB']=club_regret_matrix.ravel()

# fig, ax=plt.subplots(1,1, figsize=(5,5))
# sns.lineplot(x='Time', y='LinUCB', err_style='band', markers='o', data=df, color='blue', label='LinUCB')
# sns.lineplot(x='Time', y='Gob.Lin', err_style='band', data=df, color='orange', label='Gob.Lin', markers=True)
# sns.lineplot(x='Time', y='GraphUCB', err_style='band', data=df, color='red', label='GraphUCB', markers=True)
# sns.lineplot(x='Time', y='GraphUCB-Local', err_style='band', data=df, color='green', label='GraphUCB-Local', markers=True)
# sns.lineplot(x='Time', y='CLUB', err_style='band', data=df, color='black', label='CLUB', markers=True)
# ax.legend(loc=2, fontsize=14)
# ax.set_ylabel('Cumulative Regret', fontsize=16)
# ax.set_xlabel('Time', fontsize=16)
# #plt.savefig(save_path+'rbf_cum_regret'+'.png', dpi=100)
# plt.show()



# ax.set_ylabel('Cumulative Regret')
# ax.xlabel('Time')
# ax.legend(loc=0)
# linucb_regret=np.mean(linucb_regret_matrix, axis=0)
# linucb_error=np.mean(linucb_error_matrix, axis=0)
# gob_regret=np.mean(gob_regret_matrix, axis=0)
# gob_error=np.mean(gob_error_matrix, axis=0)

# lapucb_regret=np.mean(lapucb_regret_matrix, axis=0)
# lapucb_error=np.mean(lapucb_error_matrix, axis=0)

# lapucb_sim_regret=np.mean(lapucb_sim_regret_matrix, axis=0)
# lapucb_sim_error=np.mean(lapucb_sim_error_matrix, axis=0)

# club_regret=np.mean(club_regret_matrix, axis=0)
# club_error=np.mean(club_error_matrix, axis=0)

# plt.figure(figsize=(5,5))
# plt.plot(linucb_regret,'-.', markevery=0.1, linewidth=2, markersize=8, label='LinUCB')
# plt.plot(gob_regret, '-p', color='orange', markevery=0.1,linewidth=2, markersize=8,  label='GOB.Lin')
# plt.plot(lapucb_sim_regret, '-s', color='g', markevery=0.1, linewidth=2, markersize=8, label='GraphUCB-Local')
# plt.plot(lapucb_regret, '-o', color='r', markevery=0.1,linewidth=2, markersize=8,  label='GraphUCB')
# plt.plot(club_regret,'-*', color='k', markevery=0.1,linewidth=2, markersize=8,  label='CLUB')
# plt.ylabel('Cumulative Regret', fontsize=16)
# plt.xlabel('Time', fontsize=16)
# plt.ylim([0,80])
# # plt.title('sp=%s, sm=%s'%(np.round(sparsity, decimals=1), np.round(smoothness, decimals=1)), fontsize=16)
# plt.legend(loc=1, fontsize=14)
# plt.tight_layout()
# plt.savefig(path+'ws'+'.png', dpi=100)
# plt.show()



# plt.figure(figsize=(5,5))
# plt.plot(linucb_error,'-.', markevery=0.1, label='LinUCB')
# plt.plot(gob_error, '-p', color='orange', markevery=0.1, label='GOB.Lin')
# plt.plot(lapucb_sim_error, '-s',color='g', markevery=0.1, label='GraphUCB-Local')
# plt.plot(lapucb_error, '-o', color='r', markevery=0.1, label='GraphUCB')
# plt.plot(club_error, '-*', color='k', markevery=0.1, label='CLUB')
# plt.ylabel('Error', fontsize=14)
# plt.xlabel('Time', fontsize=14)
# plt.legend(loc=1, fontsize=14)
# plt.tight_layout()
# #plt.savefig(path+'error'+'.png', dpi=100)
# plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(linucb_beta,'-.', markevery=0.05, label='LinUCB')
# plt.plot(gob_beta, '-p', color='orange', markevery=0.05, label='GOB.Lin')
# plt.plot(lapucb_sim_beta, '-s', markevery=0.05, label='GraphUCB-Local')
# plt.plot(lapucb_beta, '-o', color='red', markevery=0.05, label='GraphUCB')
# #plt.plot(club_beta,'-p', markevery=0.1, label='CLUB')
# plt.ylabel('Beta', fontsize=16)
# plt.xlabel('Time', fontsize=16)
# plt.legend(loc=0, fontsize=16)
# plt.tight_layout()
# #plt.savefig(path+'beta'+'.png', dpi=100)
# plt.show()



# plt.figure(figsize=(5,5))
# plt.plot(linucb_x_norm,'-.', markevery=0.05, label='LinUCB')
# plt.plot(gob_x_norm, '-p', color='orange', markevery=0.05, label='GOB.Lin')
# plt.plot(lapucb_sim_x_norm, '-s', markevery=0.05, label='GraphUCB-Local')
# plt.plot(lapucb_x_norm, '-o',color='red', markevery=0.05, label='GraphUCB')
# #plt.plot(club_x_norm, label='CLUB')
# plt.ylabel('x norm', fontsize=16)
# plt.xlabel('Time', fontsize=16)
# plt.legend(loc=1, fontsize=16)
# plt.tight_layout()
# #plt.savefig(path+'x_norm'+'.png', dpi=100)
# plt.show()



# plt.figure(figsize=(5,5))
# plt.plot(linucb_ucb,'-.', markevery=0.1, label='LinUCB')
# plt.plot(gob_ucb, '-p', color='orange', markevery=0.1, label='GOB.Lin')
# plt.plot(lapucb_sim_ucb, '-s', markevery=0.1, label='GraphUCB-Local')
# plt.plot(lapucb_ucb, '-o',color='red', markevery=0.1, label='GraphUCB')
# #plt.plot(club_x_norm, label='CLUB')
# plt.ylabel('UCB', fontsize=16)
# plt.xlabel('Time', fontsize=16)
# plt.legend(loc=1, fontsize=16)
# plt.tight_layout()
# #plt.savefig(path+'UCB'+'.png', dpi=100)
# plt.show()

# payoffs=true_payoffs.ravel()
# plt.figure(figsize=(5,5))
# plt.hist(payoffs)
# plt.ylabel('Counts', fontsize=12)
# plt.xlabel('Payoffs', fontsize=12)
# plt.tight_layout()
# plt.savefig(path+'hist_payoffs'+'.png', dpi=100)
# plt.clf()