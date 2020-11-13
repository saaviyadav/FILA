import argparse
import numpy as np
import pulp
from pulp import LpMinimize, LpProblem, lpSum, LpVariable,LpSolverDefault

def lp(numStates,numActions,end,rewards,probability,transitions,discount):
    problem = LpProblem("MDP", LpMinimize)
    values = LpVariable.dict("valueFunction", range(numStates))
    LpSolverDefault.msg = 0
    problem += lpSum([values[i] for i in range(numStates)])
    V = np.zeros((numStates, 1), dtype = LpVariable)
    for i in range(numStates):
        V[i][0] = values[i]
    for i in transitions:
        if i not in end:
            for j in transitions[i]:
                value = 0
                for trans in transitions[i][j]:
                    value += trans[2]*(trans[1]+discount*values[trans[0]])
                problem += (values[i] >= value)
    # additional constraint for episodic MDPs
    for i in end :
    	problem += (values[i] == 0)
    # solve the linear programming problem
    problem.solve()
    newValues = np.zeros(numStates)
    for i in range(numStates):
        newValues[i] = values[i].varValue
    # getting the optimum policy
    
    actions = np.zeros(numStates)
    for i in transitions:
        if i not in end:
            maxValue = -float('inf')
            bestAction = 0
            for j in transitions[i]:
                value = 0
                for trans in transitions[i][j]:
                    value += trans[2]*(trans[1]+discount*newValues[trans[0]])
                if value > maxValue:
                    maxValue = value
                    bestAction = j
            actions[i] = bestAction
    return newValues, actions

def evaluate(oldValues,numStates,numActions,start,end,transitions,discount,policy):
    newValues = np.copy(oldValues)
    tol = 1e-15
    while True:
        oldValues = np.copy(newValues)
        for i in transitions:
            if i not in end:
                score = 0
                for trans in transitions[i][policy[i]]:
                    score += trans[2]*(trans[1]+discount*newValues[trans[0]])
                newValues[i] = score
                    #print(newValues[i])
        if np.sum(abs(oldValues-newValues))<tol:
            break
    return newValues

def howardIteration(numStates,numActions,start,end,rewards,probability,transitions,mdtype,discount): 
    policy = np.ones(numStates)
    tol = 1e-15
    oldPolicy = np.zeros(numStates)
    t = 0
    np.random.seed(0)
    policy = np.zeros(numStates)
    for i in transitions:
        if i not in end:
            for j in transitions[i]:
                policy[i] = j
                break
    #print(policy)
    oldValues = np.zeros(numStates)
    newValues = np.zeros(numStates)
    k = 0

    while(True):
        k =k+1
        oldValues = np.copy(newValues)
        oldPolicy = np.copy(policy)
        newValues = evaluate(oldValues,numStates,numActions,start,end,transitions,discount,policy)
        #print(newValues)
        oldValues = np.copy(newValues)
        breakLoop = True
        for i in transitions:
            if i not in end:
                maxScore = -float('inf')
                bestAction = -1
                for j in transitions[i]:
                    transValue = 0
                    for trans in transitions[i][j]:
                        transValue += trans[2]*(trans[1]+discount*newValues[trans[0]])
                    if transValue > maxScore:
                        maxScore = transValue
                        bestAction = j
                policy[i] = bestAction
                #newValues[i] = maxScore
                if (oldPolicy[i] != policy[i]):
                    breakLoop = False
                    break
        # for i in range(numStates):
        #     policy[i] = actions[i]
        #print(newValues,policy)
        if breakLoop:
            break
    return oldValues, policy

def valueIteration(numStates,numActions,start,end,rewards,probability,transitions,mdtype,discount,policy):
    oldValue = np.zeros(numStates)
    actions = np.zeros(numStates)
    tol = 1e-10
    maxIter = 100000
    t = 0
    newValue = np.zeros(numStates)
    while(t<maxIter):
        oldValue = np.copy(newValue)
        for i in transitions:
            if i not in end:
                maxScore = -float('inf')
                bestAction = -1
                for j in transitions[i]:
                    transValue = 0
                    for trans in transitions[i][j]:
                        transValue += trans[2]*(trans[1]+discount*oldValue[trans[0]])
                    if transValue > maxScore:
                        maxScore = transValue
                        bestAction = j
                actions[i] = bestAction
                newValue[i] = maxScore
        if np.sum(abs(oldValue-newValue))<tol: break
        oldValue = np.copy(newValue)
        t = t+1
    return oldValue, actions
    

def parseInput(mdp):
    numStates = 0
    numActions = 0
    start = 0
    end = []
    rewards = []
    probability = []
    transitions = {}
    mdtype = ""
    discount = 0
    f = open(mdp,"r")
    for line in f:
        lines = line.split(" ")
        n = len(lines)
        if lines[0] == "numStates":
            numStates = int(lines[1])
        if lines[0] == "numActions":
            numActions = int(lines[1])
            rewards = np.resize(rewards,(numStates,numStates,numActions))
            probability = np.resize(probability,(numStates,numStates,numActions))
        if lines[0] == "start":
            start = int(lines[1])
        if lines[0] == "end":
            if int(lines[1]) != -1:
                for j in range(n-1):
                    end.append(int(lines[j+1]))
            #print(end)
        if lines[0] == "transition":
            #print(float(lines[4]),lines[3], lines[4], lines[5])
            s = int(lines[1])
            a = int(lines[2])
            sNew = int(lines[3])
            rew = float(lines[4])
            prob = float(lines[5])
            try: transitions[s][a].append((sNew, rew, prob))
            except:
                try: transitions[s][a] = [(sNew, rew, prob)]
                except: transitions[s] = {a: [(sNew, rew, prob)]}
            rewards[int(lines[1]),int(lines[3]),int(lines[2])] = float(lines[4])
            probability[int(lines[1]),int(lines[3]),int(lines[2])] = float(lines[5])
        if lines[0] == "mdtype":
            mdtype = lines[1]
        if lines[0] == "discount":
            try:
                discount = float(lines[1])
            except:
                discount = float(lines[2])
    return numStates,numActions,start,end,rewards,probability,transitions,mdtype,discount

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp",type=str,default = "../")
    parser.add_argument("--algorithm",type=str,default="vi")
    args = parser.parse_args()
    [numStates,numActions,start,end,rewards,probability,transitions,mdtype,discount] = parseInput(args.mdp)
    policy = np.ones([numStates, numActions])

    if args.algorithm == "vi":
        [values,actions] = valueIteration(numStates,numActions,start,end,rewards,probability,transitions,mdtype,discount,policy)

    if args.algorithm == "hpi":
        [values,actions] = howardIteration(numStates,numActions,start,end,rewards,probability,transitions,mdtype,discount)

    if args.algorithm == "lp":
        [values,actions] = lp(numStates,numActions,end,rewards,probability,transitions,discount)

    for i in range(numStates):
        print("{:.6f}".format(float(values[i]))+" "+str(int(actions[i])))
        #print(str(float(values[i]))+" "+str(int(actions[i])))
        #float(int(values[i]*1e10))/1e10

    #mdpi = MDP(args.mdp)
    #mdpi.lp()
