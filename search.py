# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:45:50 2023

@author: Bijo Sebastian
"""

"""
Implement your search algorithms here
"""

import operator
import math

def heuristic_1(problem, state):
    """
    Euclidean distance
    """
    "*** YOUR CODE HERE ***"
    goal = problem.getGoalState()
    return int(math.dist(state, goal))

def heuristic_2(problem, state):
    """
    Manhattan distance
    """
    "*** YOUR CODE HERE ***"
    goal = problem.getGoalState()
    man_dist = 0
    for i in range(len(goal)):
        man_dist += abs(goal[i] - state[i])
    return man_dist    

def weighted_AStarSearch(problem, heuristic_ip):
    """
    Pop the node that having the lowest combined cost plus heuristic
    heuristic_ip can be M, E, or a number 
    if heuristic_ip is M use Manhattan distance as the heuristic function
    if heuristic_ip is E use Euclidean distance as the heuristic function
    if heuristic_ip is a number, use weighted A* Search with Euclidean distance as the heuristic function and the integer being the weight
    """
    "*** YOUR CODE HERE ***"
    path = []
    fringe = []
    closed = []
    start_state = problem.getStartState()
    h = 0
    try:
        weight = int(heuristic_ip)
        h = weight * heuristic_1(problem, start_state)
    except ValueError:
        if heuristic_ip == 'M':
            h = heuristic_2(problem, start_state)
        elif heuristic_ip == 'E':
            h = heuristic_1(problem, start_state)
        else:
            print('Wrong heuristic argument')
            return path
    start_node = [start_state, [], 0, h]
    fringe.append(start_node)
    while len(fringe) > 0:
        fringe.sort(key= operator.itemgetter(3))
        curr_node = fringe.pop(0)
        closed.append(curr_node)
        curr_state = curr_node[0]
        curr_path = curr_node[1]
        curr_cost = curr_node[2]
        if problem.isGoalState(curr_state):
            path = curr_path
            return path
        successors = problem.getSuccessors(curr_state)
        for next_node in successors:
            next_state = next_node[0]
            next_path = curr_path.copy()
            next_path.append(next_node[1])
            next_cost = curr_cost + next_node[2]
            h = 0
            try:
                weight = int(heuristic_ip)
                h = weight * heuristic_1(problem, next_state)
            except ValueError:
                if heuristic_ip == 'M':
                    h = heuristic_2(problem, next_state)
                elif heuristic_ip == 'E':
                    h = heuristic_1(problem, next_state)
                else:
                    print('Wrong heuristic argument')
                    return path
            next_tot_cost = next_cost + h
            next_node = [next_state, next_path, next_cost, next_tot_cost]
            if any([next_state == node[0] for node in closed]):
                continue
            in_fringe = False
            for node in fringe:
                if next_state == node[0]:
                    in_fringe = True
                    if next_cost < node[2]:
                        fringe.remove(node)
                        fringe.append(next_node)
                    break
            if not in_fringe:
                fringe.append(next_node)
    return path

def breadthFirstSearch(problem):
    return weighted_AStarSearch(problem, '0')

def uniformCostSearch(problem):
    return breadthFirstSearch(problem)                    