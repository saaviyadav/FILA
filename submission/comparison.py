import numpy as np
import matplotlib.pyplot as plt
import gridworld
import sys

class Compare(gridworld.Grid):
    def Comparison(self,numEpisodes,alpha):
        runs1 = []
        runs2 = []
        runs3 = []
        steps1 = np.zeros(numEpisodes)
        steps2 = np.zeros(numEpisodes)
        steps3 = np.zeros(numEpisodes)
        seeds = range(10)
        for seed in seeds:
            np.random.seed(seed)
            QValue1 = np.zeros((self.rows,self.cols,self.actions))
            totalTime1 = np.zeros(numEpisodes)
            QValue2 = np.zeros((self.rows,self.cols,self.actions))
            totalTime2 = np.zeros(numEpisodes)
            QValue3 = np.zeros((self.rows,self.cols,self.actions))
            totalTime3 = np.zeros(numEpisodes)

            for episode in range(numEpisodes):
                epsilon = 0.1
                t1 = self.getIterations(QValue1,epsilon,alpha,0)
                t2 = self.getIterations(QValue2,epsilon,alpha,1)
                t3 = self.getIterations(QValue3,epsilon,alpha,2)
                if episode>0: 
                    totalTime1[episode] += t1+totalTime1[episode-1]
                    totalTime2[episode] += t2+totalTime2[episode-1]
                    totalTime3[episode] += t3+totalTime3[episode-1]
                else: 
                    totalTime1[episode] += t1
                    totalTime2[episode] += t2
                    totalTime3[episode] += t3
                steps1[episode] += t1
                steps2[episode] += t2
                steps3[episode] += t3
            runs1.append(totalTime1)
            runs2.append(totalTime2)
            runs3.append(totalTime3)
        avg1 = np.mean(runs1,axis=0)
        steps1 = [i/10 for i in steps1]
        avg2 = np.mean(runs2,axis=0)
        steps2 = [i/10 for i in steps2]
        avg3 = np.mean(runs3,axis=0)
        steps3 = [i/10 for i in steps3]
        plt.plot(avg1,range(avg1.shape[0]))
        plt.plot(avg2,range(avg2.shape[0]))
        plt.plot(avg3,range(avg3.shape[0]))
        plt.xlabel('Time steps')
        plt.ylabel('Episodes')
        plt.legend(["Sarsa", "Qlearn", "Expected Sarsa"])
        plt.title('Sarsa, Expected Sarsa, and Q-learning')
        plt.savefig('Timestep_comp.png')

        # plt.clf()
        # plt.plot(range(len(steps1)),steps1)
        # plt.plot(range(len(steps2)),steps2)
        # plt.plot(range(len(steps3)),steps3)
        # plt.ylabel('Average Steps')
        # plt.xlabel('Episodes')
        # plt.legend(["Sarsa", "Qlearn", "Expected Sarsa"])
        # plt.savefig('Comp_steps.png')

if __name__ == '__main__':
	kingMove = int(sys.argv[1])
	stochastic = int(sys.argv[2])
	episodes = int(sys.argv[3])
	alpha = 0.5
	grid = Compare(kingMove, stochastic)
	grid.Comparison(170,0.5)