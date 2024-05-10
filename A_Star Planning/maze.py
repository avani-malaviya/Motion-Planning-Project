# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:45:50 2023

@author: Bijo Sebastian
"""

import search
import copy
import maze_maps
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
import numpy as np

#Flag to enable plots
#Disbaled in VPL
enable_plots  = False
class Maze:
  """
  This class outlines the structure of the maze problem
  """
  
  maze_map = []# To store map data, start and goal points
  
  # Legal moves
  # [delta_x, delta_y, description]
  eight_neighbor_actions = {'up':[-1, 0], 'down':[1, 0], 'left': [0, -1], 'right': [0, 1], 'ul': [-1, -1], 'ur': [-1, 1], 'dl': [1, -1], 'dr': [1, 1], 's': [0,0]}

  if enable_plots:
      #Setup plot
      plt.close('all')
      map_plot_copy = []
      plot_colormap_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=19.0)
      fig,ax = plt.subplots(1)
      plt.axis('equal')
  
  def plot_map(self):
      """
      Plot
      """
      if enable_plots:
          start = self.getStartState()
          goal = self.getGoalState()
          self.map_plot_copy[start[0]][start[1]] = maze_maps.start_id
          self.map_plot_copy[goal[0]][goal[1]] = maze_maps.goal_id
          plt.imshow(self.map_plot_copy, cmap=plt.cm.tab20c, norm=self.plot_colormap_norm)
          plt.show()
      
  # default constructor
  def __init__(self, id):
      """
      Sets the map as defined in file maze_maps
      """
      #Set up the map to be used
      self.maze_map = maze_maps.maps_dictionary[id]
      self.state_expansion_counter = 0
      if enable_plots:
          self.map_plot_copy = copy.deepcopy(self.maze_map.map_data)
          self.plot_map()
      return
     
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     start_state = self.maze_map.start
     return start_state
 
  def getGoalState(self):
     """
     Returns the start state for the search problem 
     """
     goal_state =  self.maze_map.goal
     return goal_state
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     if state == self.getGoalState():
         return True
     else:
         return False

  def isObstacle(self, state):
      """
        state: Search state
     
      Returns True if and only if the state is an obstacle
      """
      if self.maze_map.map_data[state[0]][state[1]] == maze_maps.obstacle_id:
          return True
      else:
          return False
  
  def getStateExpansionCount(self):
      """
      Returns number of state expansions
      """
      return self.state_expansion_counter

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     if enable_plots:
         #Update changes on the plot copy
         if self.map_plot_copy[state[0]][state[1]] == maze_maps.fringe_id:
             self.map_plot_copy[state[0]][state[1]] = maze_maps.expanded_id
     
     successors = []
     self.state_expansion_counter = self.state_expansion_counter + 1
     for action in self.eight_neighbor_actions:
         
         #Get individual action
         del_x, del_y = self.eight_neighbor_actions.get(action)
         
         #Get successor
         new_successor = [state[0] + del_x , state[1] + del_y]
         new_action = action
         
         # Check for obstacle 
         if self.isObstacle(new_successor):
             continue
          
         
         if enable_plots:
             #Update changes on the plot copy
             if self.map_plot_copy[new_successor[0]][new_successor[1]] == maze_maps.free_space_id1 or self.map_plot_copy[new_successor[0]][new_successor[1]] == maze_maps.free_space_id2:
                 self.map_plot_copy[new_successor[0]][new_successor[1]] = maze_maps.fringe_id
         
         #Check cost
         if self.maze_map.map_data[new_successor[0]][new_successor[1]] == maze_maps.free_space_id2:
             new_cost = maze_maps.free_space_id2_cost
         else:
             new_cost = maze_maps.free_space_id1_cost 
             
         successors.append([new_successor, new_action, new_cost])
         
     return successors

if __name__ == '__main__':
    
    current_maze = Maze(1)
    path, path_nodes = search.weighted_AStarSearch(current_maze, 'E', [])
    print(path_nodes)
    result = pd.DataFrame(path_nodes)
    result.to_csv("Path1.csv", header=False, index=False)
    if path:
        row,col = current_maze.getStartState() 
        for action in path:
            del_x, del_y = current_maze.eight_neighbor_actions.get(action)
            row = row + del_x
            col = col + del_y
            if enable_plots:
                current_maze.map_plot_copy[row][col] = 10
        if current_maze.isGoalState([row, col]):
            print("Found a path of ", len(path)," moves by expanding ",current_maze.getStateExpansionCount()," nodes")
            if enable_plots:
                current_maze.plot_map()
        else:
            print('Not a valid path')
        
    else:        
        print("Could not find a path")  



    current_maze = Maze(2)

    path, path_nodes2 = search.weighted_AStarSearch(current_maze, 'E', [path_nodes])
    print(path_nodes2)
    result = pd.DataFrame(path_nodes)
    result.to_csv("Path2.csv", header=False, index=False)
    if path:
        row,col = current_maze.getStartState() 
        for action in path:
            del_x, del_y = current_maze.eight_neighbor_actions.get(action)
            row = row + del_x
            col = col + del_y
            if enable_plots:
                current_maze.map_plot_copy[row][col] = 10
        if current_maze.isGoalState([row, col]):
            print("Found a path of ", len(path)," moves by expanding ",current_maze.getStateExpansionCount()," nodes")
            if enable_plots:
                current_maze.plot_map()
        else:
            print('Not a valid path')
        
    else:        
        print("Could not find a path")  


        
    current_maze = Maze(3)

    path, path_nodes3 = search.weighted_AStarSearch(current_maze, 'E', [path_nodes,path_nodes2])
    print(path_nodes3)
    result = pd.DataFrame(path_nodes)
    result.to_csv("Path3.csv", header=False, index=False)
    if path:
        #Check path validity
        row,col = current_maze.getStartState() 
        for action in path:
            del_x, del_y = current_maze.eight_neighbor_actions.get(action)
            row = row + del_x
            col = col + del_y
            if enable_plots:
                current_maze.map_plot_copy[row][col] = 10
        if current_maze.isGoalState([row, col]):
            print("Found a path of ", len(path)," moves by expanding ",current_maze.getStateExpansionCount()," nodes")
            if enable_plots:
                current_maze.plot_map()
        else:
            print('Not a valid path')
        
    else:        
        print("Could not find a path")  


import csv

with open("Map_Extraction/refinedmap.csv", newline='') as f:
    reader = csv.reader(f)
    map = list(reader)

max_row = int(map[-1][1])
max_col = int(map[-1][0])

map1 = [ [0]*(int(map[-1][0])+1) for i in range(int(map[-1][1])+1)]

for i in range(len(map)):
    if (float(map[i][2]) >100) : map1[int(map[i][1])][int(map[i][0])] = 3
    else: map1[int(map[i][1])][int(map[i][0])] = 16

for i in range(len(map1)):
    map1[i][-1] = 16
    map1[i][0] = 16    

for i in range(len(map1[1][:])):
    map1[-1][i] = 16

# Find the longest path
max_length = max(len(path_nodes), len(path_nodes2), len(path_nodes3))

# Extend the shorter paths with the last element
path1 = path_nodes + [path_nodes[-1]] * (max_length - len(path_nodes))
path2 = path_nodes2 + [path_nodes2[-1]] * (max_length - len(path_nodes2))
path3 = path_nodes3 + [path_nodes3[-1]] * (max_length - len(path_nodes3))

# Extract the x and y coordinates for each path
x1, y1 = zip(*path1)
x2, y2 = zip(*path2)
x3, y3 = zip(*path3)

# Create a figure and axis
fig, ax = plt.subplots()

map = np.transpose(map1)

# Plot the obstacles
for i in range(len(map)):
    for j in range(len(map[0])):
        if map[i][j] == 16:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='gray')
            ax.add_patch(rect)

# Set axis limits and labels
# ax.set_xlim(min(min(x1), min(x2), min(x3)) - 5, max(max(x1), max(x2), max(x3)) + 5)
# ax.set_ylim(min(min(y1), min(y2), min(y3)) - 5, max(max(y1), max(y2), max(y3)) + 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Robot Paths with Bounding Boxes')

# Initialize the plot lines and bounding boxes
line1, = ax.plot([], [], 'ro', lw=2)
line2, = ax.plot([], [], 'go', lw=2)
line3, = ax.plot([], [], 'bo', lw=2)
bbox1 = ax.add_patch(plt.Circle((0, 0), radius=3, fill=False, edgecolor='r'))
bbox2 = ax.add_patch(plt.Circle((0, 0), radius=3, fill=False, edgecolor='g'))
bbox3 = ax.add_patch(plt.Circle((0, 0), radius=3, fill=False, edgecolor='b'))

# Animation function
def animate(i):
    line1.set_data(x1[:i+1], y1[:i+1])
    line2.set_data(x2[:i+1], y2[:i+1])
    line3.set_data(x3[:i+1], y3[:i+1])
    bbox1.center = (x1[i], y1[i])
    bbox2.center = (x2[i], y2[i])
    bbox3.center = (x3[i], y3[i])
    return line1, line2, line3, bbox1, bbox2, bbox3


# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=max(len(path_nodes), len(path_nodes2), len(path_nodes3)), interval=200, blit=True)

# Save the animation as a GIF
ani.save('robot_paths.gif', writer='pillow', fps=5)

# Show the animation
plt.show()


    




        
    
 