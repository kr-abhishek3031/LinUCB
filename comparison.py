import numpy as np
import random
import math 
import matplotlib.pyplot as plt
best_re = 0
#######initialization
T = 20000 # number of iterations
regret_ind = [1]
i=2 
while i<= T:
	regret_ind.append(i)
	i+=2
#print(regret_ind)
#######
#Ucb
ucb_reg = []
ucb_count = np.zeros(10)
ucb_est = np.zeros(10)
ucb_re = 0
#######
#LinUcb
linucb_reg =[]
linucb_re = 0
linucb_count = np.zeros(10)
best_count = np.zeros(10)
theta_star = [] #mu_star is theta_star
#explo_round = input("Fraction of Exploration Rounds:")
#explo_round = float()
ucb_para = input("UCB Parameter:")
ucb_para = float(ucb_para)
alpha = input("Alpha: ")
alpha = float(alpha)
ad_feat = []
for i in range(0,10):
	temp = random.sample(range(1, 100),10)
	temp[i] = temp[i] * 10
	s =sum(temp)
	temp = [x/s for x in temp]
	ad_feat.append(temp)
	t = random.sample(range(1,100),10)
	t[i] = t[i] *10
	s1 =sum(t)
	t = [x/s1 for x in t]
	theta_star.append(t)
theta_star =np.array(theta_star)
ad_feat = np.array(ad_feat)
A = [np.identity(10)]
b = [np.zeros(10)]
est_theta = [np.zeros(10)]
p = np.zeros(10)
for i in range(9):
	A = np.append([np.identity(10)], A, axis=0)
	b = np.append([np.zeros(10)], b, axis=0)
	est_theta = np.append([np.zeros(10)], est_theta, axis = 0)

#######
#TS
ts_reg = []
ts_rew = 0
d = 10 #dimension
delta = input("Delta : ") #preferably 0.05 
delta = float(delta)
B = [np.identity(d)] #B deals with contexts, 10 is dimension of context
mu_hat = [np.zeros(d)] #parameter which we want to estimate
f = [np.zeros(d)] 
for i in range(d-1):
	B = np.append([np.identity(10)], B, axis=0)
	mu_hat = np.append([np.zeros(10)], mu_hat, axis=0)
	f = np.append([np.zeros(10)], f, axis = 0)
mu_star = theta_star
eps = 1/(math.log(int(T))) #from remark 2 of the paper TS for CMAB
R = 0.01
v = R * math.sqrt((24/eps)*d*(math.log(1/delta)))
m = 0
#######
best_reg = []

#################
#Creating 100 features of 100 users 
temp1 = random.sample(range(1, 100),10)

ind = 0 #to generate context randomly from 0 to 9
temp1[ind] = temp1[ind] * 10 #here context is more similar to the first
s =sum(temp1)
temp1 = [x/s for x in temp1]
temp1 = np.array(temp1)
feature = [temp1]
for i in range(99):
	temp1 = random.sample(range(1, 100),10)

	ind = int(i/10)#to generate context randomly from 0 to 9
	temp1[ind] = temp1[ind] *10 #here context is more similar to the first
	s =sum(temp1)
	temp1 = [x/s for x in temp1]
	temp1 = np.array(temp1)
	feature = np.append(feature, [temp1], axis=0)
#print(feature)
#print(len(feature))
for i in range(T): # 2 is number iterations, time 
	#observe features
	'''
	temp1 = random.sample(range(1, 100),10)

	ind = random.randint(0,9)#to generate context randomly from 0 to 9
	temp1[ind] = temp1[ind] * 10 #here context is more similar to the first
	s =sum(temp1)
	temp1 = [x/s for x in temp1]
	temp1 = np.array(temp1)
	#temp1 is a random user
	'''
	random_no = np.random.rand()
	ind = random.randint(0,99)
	temp1= feature[ind]
	for j in range(len(ad_feat)):
		if j ==0 :
			features = [ad_feat[j] * temp1]
		else:
			features = np.append(features, [ad_feat[j]*temp1], axis=0)


	#########################
	#Thompson sampling 
	ran_sample = np.array([]) 
	for j in range(d):
		#print(mu_hat[j])
		#print(v*v*(np.linalg.inv(B[j])))
		sam = np.random.multivariate_normal(mu_hat[j], v*v*(np.linalg.inv(B[j])))#generating sample
		if j ==0:
			ran_sample = [sam]
		else:
			ran_sample = np.append(ran_sample, [sam], axis =0 )
	
	#mu_hat = ran_sample

	temp_val = np.array([])
	for k in range(d):
		t10 = np.matmul(np.transpose(features[k]), ran_sample[k])
		if k ==0:
			temp_val = [t10]
		else:
			temp_val = np.append(temp_val, [t10] , axis =0)
	opt_arm_ind = np.argmax(temp_val)

	rew = np.matmul(np.transpose(features[opt_arm_ind]), mu_star[opt_arm_ind])
	ts_rew += rew
	if random_no < rew :
		reward2 = 1
	else :
		reward2 =0
	#####
	#update step
	resh = np.reshape(features[opt_arm_ind], (1, 10))
	resh1 = np.reshape(features[opt_arm_ind], (10,1))
	B[opt_arm_ind] = B[opt_arm_ind] + np.matmul(resh1, resh)
	f[opt_arm_ind]= f[opt_arm_ind] + features[opt_arm_ind]*reward2
	mu_hat[opt_arm_ind] = np.matmul(np.linalg.inv(B[opt_arm_ind]), f[opt_arm_ind])
	
	#end TS
	######

	####################################
	#ucb
	
	if i < 10:
		temp_re = np.matmul(np.transpose(features[i]), theta_star[i])
		ucb_count[i] +=1
		if random_no < temp_re :
			reward1 = 1
		else :
			reward1 =0
		ucb_est[i] = reward1
		ucb_re += temp_re
	else:
		t1 = np.zeros(10)
		for k in range(10):
			estimate = math.sqrt((ucb_para * math.log(i))/(2*ucb_count[k]))
			t1[k] = ucb_est[k] + estimate
		best_ind = np.argmax(t1)
		ucb_count[best_ind] += 1 
		temp_re = np.matmul(np.transpose(features[best_ind]), theta_star[best_ind])
		if random_no < temp_re :
			reward1 = 1
		else :
			reward1 =0
		ucb_est[best_ind] = ((ucb_count[best_ind]-1)*ucb_est[best_ind] + reward1)/ ucb_count[best_ind]
		ucb_re += temp_re

	#end UCB

	####################################
	#Best 
	t2 = np.zeros(10)
	for j in range(10):
		t2[j] = np.matmul(np.transpose(features[j]), theta_star[j])

	b_ind = np.argmax(t2)
	best_re += t2[b_ind] 
	best_count[b_ind] +=1
	#end
	####################################

	####################################
	#LinUCB
	for k in range(len(theta_star)):
		temp2 = np.linalg.inv(A[k])
		est_theta[k] = np.matmul( temp2, b[k])
		
		s2 = sum(est_theta[k])
		if s2 > 0:
			s3 = [x/s2 for x in est_theta[k]]
			est_theta[k]= s3
	
		temp3 = np.transpose(features[k])
		p[k] = alpha * (math.sqrt(np.matmul(np.matmul(temp3, temp2), features[k]))) + np.matmul(np.transpose(est_theta[k]), features[k]) 
	opt_ind = np.argmax(p) 
	linucb_count[opt_ind] += 1

	 
	temp_rew = np.matmul(np.transpose(features[opt_ind]), theta_star[opt_ind])
	if random_no < temp_rew:
		reward = 1
	else:
		reward =0
	linucb_re += temp_rew
	res = np.reshape(features[opt_ind], (1, 10))
	res1 = np.reshape(features[opt_ind], (10,1))
	A[opt_ind] = A[opt_ind] + np.matmul(res1, res)
	b[opt_ind] = b[opt_ind] + reward * features[opt_ind]

	#end LinUCB
	##########

	if i == regret_ind[m]:
		linucb_reg.append(linucb_re)
		ucb_reg.append(ucb_re)
		ts_reg.append(ts_rew)
		best_reg.append(best_re)
		m += 1
linucb_reg = np.array(linucb_reg)
print(linucb_reg)
ucb_reg = np.array(ucb_reg)
ts_reg = np.array(ts_reg)
best_reg = np.array(best_reg)
diff_linucb_reg = best_reg - linucb_reg
diff_ucb_reg = best_reg - ucb_reg
diff_ts_reg = best_reg - ts_reg
regret_ind= regret_ind[:-1]
plt.plot(regret_ind, diff_linucb_reg, 'g', regret_ind, diff_ucb_reg, 'b', regret_ind, diff_ts_reg, 'k')
print(ucb_count)
print(best_count)
print(linucb_count)
print (ucb_re)
print(linucb_re)
print(ts_rew)
print(best_re)
plt.show()