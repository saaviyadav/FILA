import numpy as np
import matplotlib.pyplot as plt
import sys

class Grid:
	def __init__(self, kingMove, stochastic):
		self.rows = 7
		self.cols = 10
		self.wind = [0,0,0,1,1,1,2,2,1,0]
		self.start = [3,0]
		self.end = [3,7]
		self.disc = 1
		self.kingMove = kingMove
		if kingMove == 0: self.actions = 4
		else: self.actions = 8
		self.stochastic = stochastic

	def getAction(self,s,QValue,epsilon):
		if self.kingMove == 0: validActions = [0,1,2,3]
		else: validActions = [0,1,2,3,4,5,6,7]
		action = 0
		if (np.random.uniform() < epsilon): action = np.random.choice(validActions)
		else:
			maxQValue = - float("inf")
			for a in validActions:
				if QValue[s[0]][s[1]][a] > maxQValue:
					maxQValue = QValue[s[0]][s[1]][a]
					action = a
		return action

	def getTarget(self,QValue,method,epsilon,state,action):
		reward = -1
		if method == 0:
			target = reward + self.disc*QValue[state[0]][state[1]][action]
		elif method == 1:
			target = reward + self.disc*max(QValue[state[0]][state[1]])
		else:
			target = reward + (epsilon)*self.disc*np.average(QValue[state[0]][state[1]]) + (1-epsilon)*self.disc*max(QValue[state[0]][state[1]])
		return target

	def getIterations(self,QValue,epsilon,alpha,method):
		iterations = 0
		state = self.start
		action = self.getAction(state,QValue,epsilon)
		while(state != self.end):
			x = state[0]
			y = state[1]
			x += self.wind[y]
			if (action == 0 or action == 4 or action == 7): x += 1
			elif (action == 2 or action == 5 or action == 6): x -= 1
			if (action == 1 or action == 4 or action == 5): y += 1
			elif (action == 3 or action == 6 or action == 7): y -= 1
			if self.stochastic == 1: x += np.random.choice([-1,0,1])
			nextState = state[:]
			nextState[0] = max(min(self.rows-1,x),0)
			nextState[1] = max(min(self.cols-1,y),0)
			nextAction = self.getAction(nextState,QValue,epsilon)
			target = self.getTarget(QValue,method,epsilon,nextState,nextAction)
			QValue[state[0]][state[1]][action] += alpha*(target - QValue[state[0]][state[1]][action])
			iterations += 1
			state = nextState
			action = nextAction
		return iterations

	def Sarsa(self,episodes,alpha):
		runs = []
		steps = np.zeros(episodes)
		seeds = range(10)
		for seed in seeds:
			np.random.seed(seed)
			QValue = np.zeros((self.rows,self.cols,self.actions))
			totalTime = np.zeros(episodes)
			for episode in range(episodes):
				epsilon = 0.1
				it = self.getIterations(QValue,epsilon,alpha,0)
				if episode>0: totalTime[episode] += it+totalTime[episode-1]
				else: totalTime[episode] += it
				steps[episode] += it
			runs.append(totalTime)
		avg = np.mean(runs,axis=0)
		steps = [i/10 for i in steps]
		plt.plot(avg,range(avg.shape[0]))
		plt.xlabel('Time steps')
		plt.ylabel('Episodes')
		plt.title('Episodes vs Time Steps for Sarsa')
		plt.savefig('timestep'+str(self.kingMove)+str(self.stochastic)+'.png')

		plt.clf()
		plt.plot(range(len(steps)),steps)
		plt.ylabel('Average Steps')
		plt.xlabel('Episodes')
		plt.title('Average Steps vs Episodes for Sarsa')
		plt.savefig('avgsteps'+str(self.kingMove)+str(self.stochastic)+'.png')

if __name__ == '__main__':
	kingMove = int(sys.argv[1])
	stochastic = int(sys.argv[2])
	episodes = int(sys.argv[3])
	alpha = 0.5
	grid = Grid(kingMove, stochastic)
	grid.Sarsa(episodes, alpha)
