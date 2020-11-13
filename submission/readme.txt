Type ./script.sh to run all the tasks.

gridworld.py has the code to form the mdp and for sarsa and all other variants of agents.
To run the above file type:   python3 gridworld.py p1 p2 episodes
where p1 = 1(if king's move has to be selected otherwise 0),
      p2 = 1(if stochasticity has to be added otherwise 0),
      episodes = number of episodes.

Plots convention is as follows:-
1) avgsteps00.png for avg steps vs episodes where first zero indicates if king's move is selected or not and the other zero indicates if stochasticity is added.
2) timestep00.png for episodes vs time steps where first zero indicates if king's move is selected or not and the other zero indicates if stochasticity is added.

comparison.py has the code for comparison between the different agents and saves the png file.
To run the above file type:   python3 comparison.py p1 p2 episodes
where p1 = 1(if king's move has to be selected otherwise 0),
      p2 = 1(if stochasticity has to be added otherwise 0),
      episodes = number of episodes.

One plot for episodes vs timestep assuming no king's move and stochasticity is generated and saved.