import numpy as np
import sys

def thompsonWithHint(means,hz,sortedMeans):
	arms = len(sortedMeans)
	hint = sortedMeans[arms-2]
	#print(sortedMeans,hint)
	count = np.zeros((arms,1))
	history = np.zeros((hz,1))
	pulls = np.zeros((arms,hz))
	reward_perarm = np.zeros((arms,1))
	ep = 0.07
	for i in range(hz):
		thoms = np.zeros((arms,1))
		diff = np.zeros((arms,1))
		std = np.zeros((arms,1))
		safe = {}
		max_value = 0
		best_arm = 0
		for j in range(arms):
			alpha = reward_perarm[j,0]+1
			beta = count[j,0]-reward_perarm[j,0]+1
			thoms[j] = np.random.beta(alpha,beta)
			betaMeans = (alpha)/(alpha+beta)
			if betaMeans > hint:
				safe[j] = betaMeans 
				if safe[j] > max_value:
					max_value = safe[j]
					best_arm = j
			var = (alpha*beta)/(np.power((alpha+beta),2)*(alpha+beta+1))
			diff[j] = np.abs(betaMeans-hint)
			std[j] = np.power(var,0.5)
		# arm = np.argmin(diff)
		# if pullArm(ep):
		# 	arm = np.argmin(diff)
		# else:
		# 	arm = np.argmax(thoms)
		if len(safe) != 0:
			# arm = max(safe, key=safe.get)
			arm = best_arm
			#print(arm)
		else:
			arm = np.argmax(thoms)
		pulls[arm,i] = pullArm(means[arm])
		history[i,0] = pulls[arm,i]
		count[arm,0] = count[arm,0]+1
		reward_perarm[arm] = reward_perarm[arm] + pulls[arm,i]
	return np.max(means)*hz-np.sum(history)

def thompson(means,hz):
	arms = len(means)
	count = np.zeros((arms,1))
	history = np.zeros((hz,1))
	pulls = np.zeros((arms,hz))
	reward_perarm = np.zeros((arms,1))
	for i in range(hz):
		thoms = np.zeros((arms,1))
		max_value = 0
		best_arm = 0
		for j in range(arms):
			thoms[j] = np.random.beta(reward_perarm[j]+1,count[j]-reward_perarm[j]+1)
			if thoms[j]>max_value:
				max_value = thoms[j]
				best_arm = j
		arm = best_arm
		pulls[arm,i] = pullArm(means[arm])
		history[i,0] = pulls[arm,i]
		count[arm,0] = count[arm,0]+1
		reward_perarm[arm] = reward_perarm[arm] + pulls[arm,i]
	return np.max(means)*hz-np.sum(history)

def KL(p,q):
    res = 0
    if p > 0: 
        res = res + p*(np.log(p) - np.log(q))
    if p < 1: 
        res = res + (1-p)*(np.log(1-p) - np.log(1-q))
    return res

def pullArm(k):
    r = np.random.uniform(0,1)
    if r < k: 
        return 1
    else: 
        return 0

def pickArm(reward_perarm,t,c,count):
	ucbkl = []
	#ucbkl = np.zeros((len(reward_perarm))).astype(float)
	for i in range(len(reward_perarm)):
		rhs = np.log(t)+c*np.log(np.log(t))
		rhs = rhs/count[i]
		# if i == 0:
		# 	print(count[i])
		emp = float(reward_perarm[i])/float(count[i])
		#print(emp)
		low = emp
		high = 1
		maxIter = 25
		precision = 1e-6
		for j in range(maxIter):
			m = (low + high)/2
			kl = KL(emp,m)
			if (abs(kl - rhs) < precision): break
			if (kl <= rhs): low = m
			else: high = m
		ucbkl.append(m)
	#print(ucbkl[:2])
	return np.argmax(ucbkl)

def klucb(means,hz):
	arms = len(means)
	count = np.zeros((arms))
	history = np.zeros((hz))
	pulls = np.zeros((arms,hz))
	reward_perarm = np.zeros((arms))
	c = 3
	for i in range(min(arms,hz)):
		pulls[i,i] = pullArm(means[i])
		history[i] = pulls[i,i]
		count[i] = count[i]+1
		reward_perarm[i] = reward_perarm[i] + pulls[i,i]
	#print(reward_perarm)
	for i in range(min(arms,hz),hz):
		arm = pickArm(reward_perarm,i,c,count)
		#print(arm)
		pulls[arm,i] = pullArm(means[arm])
		history[i] = pulls[arm,i]
		count[arm] = count[arm]+1
		reward_perarm[arm] = reward_perarm[arm] + pulls[arm,i]
	return np.max(means)*hz - np.sum(history)

def ucb(means, hz):
	arms = len(means)
	count = np.zeros((arms,1))
	history = np.zeros((hz,1))
	pulls = np.zeros((arms,hz))
	reward_perarm = np.zeros((arms,1))
	ucbt = np.zeros((arms,1))
	for i in range(min(arms,hz)):
		pulls[i,i] = pullArm(means[i])
		history[i,0] = pulls[i,i]
		count[i,0] = count[i,0]+1
		reward_perarm[i] = reward_perarm[i] + pulls[i,i]
	for i in range(min(arms,hz),hz):
		for j in range(arms):
			if count[j,0]>0:
				ucbt[j,0] = reward_perarm[j,0]/count[j,0]+np.power((2*np.log(i))/count[j,0],0.5)
				#print(j, reward_perarm[j,0]/count[j,0])
		#print(ucbt.T)
		arm = np.argmax(ucbt)
		pulls[arm,i] = pullArm(means[arm])
		history[i,0] = pulls[arm,i]
		count[arm,0] = count[arm,0]+1
		reward_perarm[arm] = reward_perarm[arm] + pulls[arm,i]
	actual_max = np.max(means)
	#print(history)
	return actual_max*hz-np.sum(history)

def epsilon(means, ep, hz):
	arms = len(means)
	count = np.zeros((arms,1))
	pulls = np.zeros((arms,hz))
	history = np.zeros((hz,1))
	reward_perarm = np.zeros((arms,1))
	empirical = np.zeros((arms,1))
	for i in range(min(arms,hz)):
		pulls[i,i] = pullArm(means[i])
		history[i,0] = pulls[i,i]
		count[i,0] = count[i,0]+1
		reward_perarm[i] = reward_perarm[i] + pulls[i,i]
		empirical[i] = float(reward_perarm[i])/float(count[i,0])
	#print(empirical)
	for i in range(min(arms,hz),hz):
		k = pullArm(ep)
		if k:
			arm = np.random.randint(arms)
		else:
			arm = np.argmax(empirical)
		pulls[arm,i] = pullArm(means[arm])
		history[i,0] = pulls[arm,i]
		count[arm,0] = count[arm,0]+1
		reward_perarm[arm] = reward_perarm[arm] + pulls[arm,i]
		empirical[arm,0] = float(reward_perarm[arm])/float(count[arm,0])
	#print(empirical)
	actual_max = np.max(means)
	#print(history)
	return actual_max*hz-np.sum(history)

def get_arg(args):
    instance = str(args[2])
    algorithm = str(args[4])
    seed = int(args[6])
    epsilon = float(args[8])
    horizon = int(args[10])
    return instance, algorithm, seed, epsilon, horizon

if __name__ == '__main__':
	ins,al,rs,ep,hz = get_arg(sys.argv)
	f = open(ins)
	means = []
	np.random.seed(rs)
	for arm in f:
		means.append(float(arm.replace("\n","")))
	#print(al)
	if al == "epsilon-greedy":
		reg = epsilon(means,ep,hz)
	if al == "ucb":
		reg = ucb(means,hz)
	if al == "kl-ucb":
		reg = klucb(means,hz)
	if al == "thompson-sampling":
		reg = thompson(means,hz)
	if al == "thompson-sampling-with-hint":
		reg = thompsonWithHint(means,hz,np.sort(means))
	print(ins + "," + al + "," + str(rs) + "," + str(ep) + "," + str(hz) + "," + str(reg))