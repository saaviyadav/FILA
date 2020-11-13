import numpy as np
import sys

class Grid:
    def __init__(self,file):
        f = open(file)
        maze = f.read().strip().split("\n")
        self.w = len(maze[0].split())
        self.h = len(maze)
        self.states = []
        for i in maze:
            for j in i.split():
                self.states.append(int(j))
        self.start = self.states.index(2)
        self.end = self.states.index(3)
        self.disc = 1

    def print_trans(self):
        print("numStates", len(self.states))
        print("numActions", 4)
        print("start", self.start)
        print("end", self.end) 
        for i in range(len(self.states)):
            if self.states[i] == 1 or self.states[i] == 3:
                continue
            else:
                north = i - self.w
                south = i + self.w
                east = i + 1
                west = i - 1
                if self.states[north] != 1 and north >= 0 and north < len(self.states):
                    print("transition", i, 0, north, -1, 1)
                if self.states[south] != 1 and south >= 0 and south < len(self.states):
                    print("transition", i, 1, south, -1, 1)
                if self.states[east] != 1 and east >= 0 and east < len(self.states):
                    print("transition", i, 2, east, -1, 1)
                if self.states[west] != 1 and west >= 0 and west < len(self.states):
                    print("transition", i, 3, west, -1, 1)
        print("mdptype episodic")
        print("discount", self.disc)

if __name__ == "__main__":
    #print(sys.argv[2])
    grid = Grid(sys.argv[2])
    grid.print_trans()
