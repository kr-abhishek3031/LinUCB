import numpy as np
import math
import random
##initialization
ts_rew = 0
ad_feat = []
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
mu_star = [] #it is the mean parameter from which 
#print(mu_hat)
for i in range(d):
	temp = random.sample(range(1, 100),10)
	temp[i] = temp[i] * 10
	s =sum(temp)
	temp = [x/s for x in temp]
	ad_feat.append(temp)
	t = random.sample(range(1,100),10)
	s1 =sum(t)
	t = [x/s1 for x in t]
	mu_star.append(t)
mu_star = np.array(mu_star)
t = 10000 # number of iterations
eps = 1/(math.log(t)) #from remark 2 of the paper TS for CMAB
R = 0.1
v = R * math.sqrt((24/eps)*d*(math.log(1/delta)))
##############
#generate features feature[i] is context for arm i
for i in range(t):
	temp1 = random.sample(range(1, 100),10)
	
	ind = random.randint(0,9)#to generate context randomly from 0 to 9
	temp1[ind] = temp1[ind] * 10 #here context is more similar to the first
	s =sum(temp1)
	temp1 = [x/s for x in temp1]
	temp1 = np.array(temp1)
	#temp1 is a random user

	for j in range(10):
		if j ==0 :
			feature = [ ad_feat[j] * temp1]
			#print (type(features))

		else:
			feature = np.append(feature, [ad_feat[j]*temp1], axis=0)
	feature = np.array(feature)
	#print("feature",feature)
	ran_sample = np.array([]) 
	count = 0
	for j in range(d):
		#print(mu_hat[j])
		#print(v*v*(np.linalg.inv(B[j])))
		sam = np.random.multivariate_normal(mu_hat[j], v*v*(np.linalg.inv(B[j])))#generating sample
		if j ==0:
			ran_sample = [sam]
		else:
			ran_sample = np.append(ran_sample, [sam], axis =0 )
		count += 1

		#print (ran_sample)
	#print("hello")
	#mu_hat = ran_sample
	temp_val = np.array([])
	for k in range(d):
		t10 = np.matmul(np.transpose(feature[k]), ran_sample[k])
		if k ==0:
			temp_val = [t10]
		else:
			temp_val = np.append(temp_val, [t10] , axis =0)
	opt_arm_ind = np.argmax(temp_val)

	rew = np.matmul(np.transpose(feature[opt_arm_ind]), mu_star[opt_arm_ind])
	ts_rew += rew
	#####
	#update step
	resh = np.reshape(feature[opt_arm_ind], (1, 10))
	resh1 = np.reshape(feature[opt_arm_ind], (10,1))
	B[opt_arm_ind] = B[opt_arm_ind] + np.matmul(resh1, resh)
	f[opt_arm_ind]= f[opt_arm_ind] = feature[opt_arm_ind]*rew
	mu_hat[opt_arm_ind] = np.matmul(np.linalg.inv(B[opt_arm_ind]), f[opt_arm_ind])

print(ts_rew)