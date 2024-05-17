# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:45:50 2023

@author: Bijo Sebastian
"""

import math
#import search
import copy
import maze_maps
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import operator
import functools
import numpy

class Node:
    def __init__(self, state, g, rhs, successors, predecessors, index):
        self.state = state
        self.g = g
        self.rhs = rhs
        self.successors = successors
        self.predecessors = predecessors
        self.index = index
        
#Flag to enable plots
#Disbaled in VPL
enable_plots  = False
class Maze:
  """
  This class outlines the structure of the maze problem
  """
  
  # Legal moves
  # [delta_x, delta_y, description]
  eight_neighbor_actions = {'up':[-1, 0], 'down':[1, 0], 'left': [0, -1], 'right': [0, 1], 'ul': [-1, -1], 'ur': [-1, 1], 'dl': [1, -1]}

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

  def getAllStates(self):
      map_lim_0 = len(self.maze_map.map_data)
      map_lim_1 = len(self.maze_map.map_data[0])
      states = []
      for i in range(map_lim_0):
          for j in range(map_lim_1):
              if not self.isObstacle([i, j]):
                  states.append([i, j])
      return states                
  
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

def heuristic(s1, s2):
    return int(math.dist(s1, s2)/math.sqrt(2))
    #return 0

def pred(u, problem, states):
    predecessors = []
    for state in states:
        successors = problem.getSuccessors(state)
        for next in successors:
            if next[0] == u:
                predecessors.append([state,next[1],next[2]])
                break
    return predecessors

def CalculateKey(s, s_start, km):
    return [min(s.g, s.rhs) + heuristic(s.state, s_start.state) + km, min(s.g, s.rhs)] 

def Initialize(problem, s_start):
    U = []
    states = problem.getAllStates()
    states_with_cost = []
    goal_state = None
    index = 0
    s_index = 0
    for state in states:
        if problem.isGoalState(state):
            predecessors = pred(state, problem, states)
            successors = problem.getSuccessors(state)
            goal_state = Node(state, math.inf, 0, successors, predecessors, index)
            states_with_cost.append(goal_state)
        else:
            predecessors = pred(state, problem, states)
            successors = problem.getSuccessors(state)
            node = Node(state, math.inf, math.inf, successors, predecessors, index)
            states_with_cost.append(node)
        if state == s_start:
            s_index = index    
        index = index + 1
    index = 0
    for state in states_with_cost:
        predecessors = state.predecessors
        successors = state.successors
        for i in range(len(successors)):
            node = successors[i][0]
            for s in states_with_cost:
                if node == s.state:
                    action = state.successors[i][1]
                    state.successors[i].append(action)
                    state.successors[i][1] = s.index
                    break
        for i in range(len(predecessors)):
            node = predecessors[i][0]
            for s in states_with_cost:
                if node == s.state:
                    action = state.predecessors[i][1]
                    state.predecessors[i].append(action)
                    state.predecessors[i][1] = s.index
                    break
        states_with_cost[state.index] = state                                    
    km = 0
    s_start = states_with_cost[s_index]

    U.append([goal_state, CalculateKey(goal_state, s_start, km)])  
    return U, states_with_cost, km, s_start


def UpdateVertex(u, problem, U, states_with_cost, s_start, km):
    if not problem.isGoalState(u.state):
        successors = u.successors
        cost = []
        for successor in successors:
            successor_with_cost = states_with_cost[successor[1]]
            cost.append(successor[2] + successor_with_cost.g)
        u.rhs = min(cost)
        states_with_cost[u.index] = u

    u_in_U = False
    for node in U:
        if node[0].state == u.state:
            u_in_U = True
            index = U.index(node)
            break
    if u_in_U:
        U.pop(index)

    if u.g != u.rhs:
        U.append([u, CalculateKey(u, s_start, km)])

    return U, states_with_cost               

def keyComp(key1, key2):
    ans = False
    if key1[0] < key2[0]:
        ans = True
        return ans
    elif key1[0] == key2[0] and key1[1] < key2[1]:
        ans = True
        return ans
    return ans

def keyComp1(u1, u2):
    if keyComp(u1[1],u2[1]):
        return -1
    elif u1[1] == u2[1]:
        return 0
    else:
        return 1

def ComputeShortestPath(problem, U, states_with_cost, s_start, km):
    U.sort(key= functools.cmp_to_key(keyComp1))   
    while s_start.rhs != s_start.g or keyComp(U[0][1], CalculateKey(s_start, s_start, km)):

        k_old = U[0][1]
        u = U.pop(0)
        u = u[0]
        k_new = CalculateKey(u, s_start, km)
        if keyComp(k_old, k_new):
            U.append([u, k_new])
        elif u.g > u.rhs:
            u.g = u.rhs
            states_with_cost[u.index] = u
            predecessors = u.predecessors
            for node in predecessors:
                predecessor_with_cost = states_with_cost[node[1]]
                U, states_with_cost = UpdateVertex(predecessor_with_cost, problem, U, states_with_cost, s_start, km)
        else:
            u.g = math.inf
            states_with_cost[u.index] = u
            predecessors = u.predecessors
            for node in predecessors:
                predecessor_with_cost = states_with_cost[node[1]]
                U, states_with_cost = UpdateVertex(predecessor_with_cost, problem, U, states_with_cost, s_start, km)
            U, states_with_cost = UpdateVertex(u, problem, U, states_with_cost, s_start, km)    

        U.sort(key= functools.cmp_to_key(keyComp1))
        s_start = states_with_cost[s_start.index]
    return U, states_with_cost                                

    
if __name__ == '__main__':

    sensing = 50
    maze1 = Maze(1)
    maze2 = Maze(2)
    maze3 = Maze(3)
    s_start_1 = maze1.getStartState()
    s_start_2 = maze2.getStartState()
    s_start_3 = maze3.getStartState()
    maze1.plot_map()
    maze2.plot_map()
    maze3.plot_map()                        

    U1, states_with_cost_1, km1, s_start_1 = Initialize(maze1, s_start_1)
    U2, states_with_cost_2, km2, s_start_2 = Initialize(maze2, s_start_2)
    U3, states_with_cost_3, km3, s_start_3 = Initialize(maze3, s_start_3)

    states_with_cost_1_original = copy.deepcopy(states_with_cost_1)
    states_with_cost_2_original = copy.deepcopy(states_with_cost_2)
    states_with_cost_3_original = copy.deepcopy(states_with_cost_3)

    for state in states_with_cost_1:
        if math.dist(state.state, s_start_2.state) <= 5 or math.dist(state.state, s_start_3.state) <= 5:
            for i in range(len(state.predecessors)):
                state.predecessors[i][2] = math.inf
                predecessor = states_with_cost_1[state.predecessors[i][1]]
                for j in range(len(predecessor.successors)):
                    if predecessor.successors[j][0] == state.state:
                        predecessor.successors[j][2] = math.inf
                        break
                states_with_cost_1[predecessor.index] = predecessor
            states_with_cost_1[state.index] = state

    for state in states_with_cost_2:
        if math.dist(state.state, s_start_3.state) <= 5 or math.dist(state.state, s_start_1.state) <= 5:
            for i in range(len(state.predecessors)):
                state.predecessors[i][2] = math.inf
                predecessor = states_with_cost_2[state.predecessors[i][1]]
                for j in range(len(predecessor.successors)):
                    if predecessor.successors[j][0] == state.state:
                        predecessor.successors[j][2] = math.inf
                        break
                states_with_cost_2[predecessor.index] = predecessor
            states_with_cost_2[state.index] = state

    for state in states_with_cost_3:
        if math.dist(state.state, s_start_1.state) <= 5 or math.dist(state.state, s_start_2.state) <= 5:
            for i in range(len(state.predecessors)):
                state.predecessors[i][2] = math.inf
                predecessor = states_with_cost_3[state.predecessors[i][1]]
                for j in range(len(predecessor.successors)):
                    if predecessor.successors[j][0] == state.state:
                        predecessor.successors[j][2] = math.inf
                        break
                states_with_cost_3[predecessor.index] = predecessor
            states_with_cost_3[state.index] = state                               

    s_start_1 = states_with_cost_1[s_start_1.index]
    s_start_2 = states_with_cost_1[s_start_2.index]
    s_start_3 = states_with_cost_1[s_start_3.index]
    U1, states_with_cost_1 = ComputeShortestPath(maze1, U1, states_with_cost_1, s_start_1, km1)
    U2, states_with_cost_2 = ComputeShortestPath(maze2, U2, states_with_cost_2, s_start_2, km2)
    U3, states_with_cost_3 = ComputeShortestPath(maze3, U3, states_with_cost_3, s_start_3, km3)
    s_start_1 = states_with_cost_1[s_start_1.index]
    s_start_2 = states_with_cost_1[s_start_2.index]
    s_start_3 = states_with_cost_1[s_start_3.index]
    s_last_1 = s_start_1
    s_last_2 = s_start_2
    s_last_3 = s_start_3

    path1 = [s_start_1.state]
    path2 = [s_start_2.state]
    path3 = [s_start_3.state]
    actions1 = []
    actions2 = []
    actions3 = []
    iter = 0
    while not maze1.isGoalState(s_start_1.state) or not maze2.isGoalState(s_start_2.state) or not maze3.isGoalState(s_start_3.state):
        iter += 1
        print(iter)
        #if s_start_1.g == math.inf or s_start_2.g == math.inf or s_start_3.g == math.inf:
        #    print('No path')
        #    break
        if not maze1.isGoalState(s_start_1.state):
            successors = s_start_1.successors
            cost = []
            for successor in successors:
                successor_with_cost = states_with_cost_1[successor[1]]
                cost.append(successor[2] + successor_with_cost.g)
            index = cost.index(min(cost))
            s_start_1 = states_with_cost_1[successors[index][1]]
            path1.append(s_start_1.state)
            actions1.append(successors[index][3])

        if not maze2.isGoalState(s_start_2.state):
            successors = s_start_2.successors
            cost = []
            for successor in successors:
                successor_with_cost = states_with_cost_2[successor[1]]
                cost.append(successor[2] + successor_with_cost.g)
            index = cost.index(min(cost))
            s_start_2 = states_with_cost_2[successors[index][1]]
            path2.append(s_start_2.state)
            actions2.append(successors[index][3])

        if not maze3.isGoalState(s_start_3.state):
            successors = s_start_3.successors
            cost = []
            for successor in successors:
                successor_with_cost = states_with_cost_3[successor[1]]
                cost.append(successor[2] + successor_with_cost.g)
            index = cost.index(min(cost))
            s_start_3 = states_with_cost_3[successors[index][1]]
            path3.append(s_start_3.state)
            actions3.append(successors[index][3])

        if not maze1.isGoalState(s_start_1.state):
            states_with_cost_1_old = copy.deepcopy(states_with_cost_1)
            for state in states_with_cost_1:
                index = state.index
                state.predecessors = states_with_cost_1_original[index].predecessors
                state.successors = states_with_cost_1_original[index].successors
                states_with_cost_1[index] = state
            changed_states_1 = []
            for state in states_with_cost_1:
                if math.dist(state.state, s_start_2.state) <= 5 or math.dist(state.state, s_start_3.state) <= 5:
                    for i in range(len(state.predecessors)):
                        state.predecessors[i][2] = math.inf
                        predecessor = states_with_cost_1[state.predecessors[i][1]]
                        for j in range(len(predecessor.successors)):
                            if predecessor.successors[j][0] == state.state:
                                predecessor.successors[j][2] = math.inf
                            break
                        states_with_cost_1[predecessor.index] = predecessor
                    states_with_cost_1[state.index] = state
            for state in states_with_cost_1:        
                index = state.index
                succ = state.successors
                succ_old = states_with_cost_1_old[index].successors
                for i in range(len(succ)):
                    if succ[i][2] != succ_old[i][2]:
                        changed_states_1.append(state)
                        break
            if len(changed_states_1) > 0:
                km1 = km1 + heuristic(s_last_1.state,s_start_1.state)
                s_last_1 = s_start_1
                for state in changed_states_1:
                    U1, states_with_cost_1 = UpdateVertex(state, maze1, U1, states_with_cost_1, s_start_1, km1)
                    s_last_1 = states_with_cost_1[s_last_1.index]
                    s_start_1 = states_with_cost_1[s_start_1.index]
                U1, states_with_cost_1 = ComputeShortestPath(maze1, U1, states_with_cost_1, s_start_1, km1)
                s_last_1 = states_with_cost_1[s_last_1.index]
                s_start_1 = states_with_cost_1[s_start_1.index]

        if not maze2.isGoalState(s_start_2.state):
            states_with_cost_2_old = copy.deepcopy(states_with_cost_2)
            for state in states_with_cost_2:
                index = state.index
                state.predecessors = states_with_cost_2_original[index].predecessors
                state.successors = states_with_cost_2_original[index].successors
                states_with_cost_2[index] = state
            changed_states_2 = []
            for state in states_with_cost_2:
                if math.dist(state.state, s_start_3.state) <= 5 or math.dist(state.state, s_start_1.state) <= 5:
                    for i in range(len(state.predecessors)):
                        state.predecessors[i][2] = math.inf
                        predecessor = states_with_cost_2[state.predecessors[i][1]]
                        for j in range(len(predecessor.successors)):
                            if predecessor.successors[j][0] == state.state:
                                predecessor.successors[j][2] = math.inf
                            break
                        states_with_cost_2[predecessor.index] = predecessor
                    states_with_cost_2[state.index] = state
            for state in states_with_cost_2:        
                index = state.index
                succ = state.successors
                succ_old = states_with_cost_2_old[index].successors
                for i in range(len(succ)):
                    if succ[i][2] != succ_old[i][2]:
                        changed_states_2.append(state)
                        break
            if len(changed_states_2) > 0:
                km2 = km2 + heuristic(s_last_2.state,s_start_2.state)
                s_last_2 = s_start_2
                for state in changed_states_2:
                    U2, states_with_cost_2 = UpdateVertex(state, maze2, U2, states_with_cost_2, s_start_2, km2)
                    s_last_2 = states_with_cost_2[s_last_2.index]
                    s_start_2 = states_with_cost_2[s_start_2.index]
                U2, states_with_cost_2 = ComputeShortestPath(maze2, U2, states_with_cost_2, s_start_2, km2)
                s_last_2 = states_with_cost_2[s_last_2.index]
                s_start_2 = states_with_cost_2[s_start_2.index]

        if not maze3.isGoalState(s_start_3.state):
            states_with_cost_3_old = copy.deepcopy(states_with_cost_3)
            for state in states_with_cost_3:
                index = state.index
                state.predecessors = states_with_cost_3_original[index].predecessors
                state.successors = states_with_cost_3_original[index].successors
                states_with_cost_3[index] = state
            changed_states_3 = []
            for state in states_with_cost_3:
                if math.dist(state.state, s_start_1.state) <= 5 or math.dist(state.state, s_start_2.state) <= 5:
                    for i in range(len(state.predecessors)):
                        state.predecessors[i][2] = math.inf
                        predecessor = states_with_cost_3[state.predecessors[i][1]]
                        for j in range(len(predecessor.successors)):
                            if predecessor.successors[j][0] == state.state:
                                predecessor.successors[j][2] = math.inf
                            break
                        states_with_cost_3[predecessor.index] = predecessor
                    states_with_cost_3[state.index] = state
            for state in states_with_cost_3:        
                index = state.index
                succ = state.successors
                succ_old = states_with_cost_3_old[index].successors
                for i in range(len(succ)):
                    if succ[i][2] != succ_old[i][2]:
                        changed_states_3.append(state)
                        break
            if len(changed_states_3) > 0:
                km3 = km3 + heuristic(s_last_3.state,s_start_3.state)
                s_last_3 = s_start_3
                for state in changed_states_3:
                    U3, states_with_cost_3 = UpdateVertex(state, maze3, U3, states_with_cost_3, s_start_3, km3)
                    s_last_3 = states_with_cost_3[s_last_3.index]
                    s_start_3 = states_with_cost_3[s_start_3.index]
                U3, states_with_cost_3 = ComputeShortestPath(maze3, U3, states_with_cost_3, s_start_3, km3)
                s_last_3 = states_with_cost_3[s_last_3.index]
                s_start_3 = states_with_cost_3[s_start_3.index]                                  
                             


    
    current_maze = Maze(1)
    print(path1)
    result = pd.DataFrame(path1)
    result.to_csv("D_Star/Path1.csv", header=False, index=False)
    path = actions1
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

    print(path2)
    result = pd.DataFrame(path2)
    result.to_csv("D_Star/Path2.csv", header=False, index=False)
    path = actions2
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
    print(path3)
    result = pd.DataFrame(path3)
    result.to_csv("D_Star/Path3.csv", header=False, index=False)
    path = actions3
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



def within_bounding_box(node1, node2):
    return abs(node1[0] - node2[0]) <= 2 and abs(node1[1] - node2[1]) <= 2

for i in range(min(len(path3), len(path2), len(path1))):
    node1 = path3[i]
    node2 = path2[i]
    node3 = path1[i]

    # Check if any two nodes are within each other's bounding boxes
    if (node1 == node2 or node1 == node3 or node2 == node3 or
        within_bounding_box(node1, node2) or
        within_bounding_box(node1, node3) or
        within_bounding_box(node2, node3)):
        print('bad!!')
import csv
import numpy as np
import matplotlib.animation as animation

with open("D_star/refinedmap.csv", newline='') as f:
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
max_length = max(len(path1), len(path2), len(path3))

# Extend the shorter paths with the last element
path1 = path1 + [path1[-1]] * (max_length - len(path1))
path2 = path2 + [path2[-1]] * (max_length - len(path2))
path3 = path3 + [path3[-1]] * (max_length - len(path3))

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
#ax.axis('equal')

# Initialize the plot lines and bounding boxes
line1, = ax.plot([], [], 'ro', lw=2)
line2, = ax.plot([], [], 'go', lw=2)
line3, = ax.plot([], [], 'bo', lw=2)
bbox1 = ax.add_patch(plt.Circle((0, 0), radius=2, fill=False, edgecolor='r'))
bbox2 = ax.add_patch(plt.Circle((0, 0), radius=2, fill=False, edgecolor='g'))
bbox3 = ax.add_patch(plt.Circle((0, 0), radius=2, fill=False, edgecolor='b'))

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
ani = animation.FuncAnimation(fig, animate, frames=max(len(path1), len(path2), len(path3)), interval=200, blit=True)

# Save the animation as a GIF
ani.save('D_star/robot_paths.gif', writer='pillow', fps=5)

# Show the animation
plt.show()        


    




        
    
 