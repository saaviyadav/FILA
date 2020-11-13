import numpy as np
import sys

def decode(gridfile, policyfile):
    f = open(gridfile)
    maze = f.read().strip().split("\n")
    w = len(maze[0].split())
    h = len(maze)
    states = []
    for i in maze:
        for j in i.split():
            states.append(int(j))
    start = states.index(2)
    end = states.index(3)
    disc = 1
    f.close()
    p = open(policyfile)
    data = p.readlines()
    value = []
    policy = []
    for row in data:
        vals = row.split()
        value.append(float(vals[0]))
        policy.append(int(vals[1]))
    st = start
    directions = {0: 'N',1: 'S',2: 'E',3: 'W'}
    while (st != end):
        print(directions[policy[st]], end=' ')
        north = st - w
        south = st + w
        east = st + 1
        west = st - 1
        if policy[st] == 0:
            st = north
        elif policy[st] == 1:
            st = south
        elif policy[st] == 2:
            st = east
        elif policy[st] == 3:
            st = west

decode(sys.argv[2], sys.argv[4])
