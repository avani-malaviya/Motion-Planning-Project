
import operator
import math
import pandas as pd
import queue


def heuristic_1(problem, state):
  """
  Euclidian Distance
  """
  goal = problem.getGoalState() 
  return (math.sqrt((state[0]-goal[0])**2 + (state[1]-goal[1])**2))
  

def heuristic_2(problem, state):
  """
  Manhattan distance
  """
  goal = problem.getGoalState() 
  return(abs(state[0]-goal[0]) + abs(state[1]-goal[1]))


def weighted_AStarSearch(problem, heuristic_ip, otherPaths):
    """
    Pop the node that having the lowest combined cost plus heuristic
    heuristic_ip can be M, E, or a number 
    if heuristic_ip is M use Manhattan distance as the heuristic function
    if heuristic_ip is E use Euclidean distance as the heuristic function
    if heuristic_ip is a number, use weighted A* Search with Euclidean distance as the heuristic function and the integer being the weight
    """
    "*** YOUR CODE HERE ***"
    fringe = queue.PriorityQueue()
    closed_set = []
    fringe.put((0, ((problem.getStartState(),0), [], 0)))

    while not fringe.empty():
        p, node = fringe.get()
        state_t, path, cost = node
        state = state_t[0]

        if problem.isGoalState(state):
            path_nodes = [problem.getStartState()]
            i = 0
            for action in path:
                state = path_nodes[i]
                del_x, del_y = problem.eight_neighbor_actions.get(action)
                new_successor = [state[0] + del_x , state[1] + del_y]
                path_nodes.append(new_successor)
                i += 1
            
            return path, path_nodes

        if state_t in closed_set:
            continue

        closed_set.append(state_t)

        successors = problem.getSuccessors(state)
        for successor_state, action, step_cost in successors:
            if not otherPaths: 
                if (successor_state, len(path)) not in closed_set:    
                    new_path = path + [action]
                    new_cost = cost + step_cost

                    if heuristic_ip == 'E':
                        heuristic = heuristic_1(problem, successor_state)
                    if heuristic_ip == 'M':
                        heuristic = heuristic_2(problem, successor_state)
                    if heuristic_ip.isdigit():
                        heuristic = int(heuristic_ip)*heuristic_1(problem, successor_state)

                    fringe.put((new_cost + heuristic, [(successor_state, len(new_path)), new_path , new_cost]))

            else:
                check_successor = True

                for otherPath in otherPaths:                    
                    if math.sqrt((successor_state[0] - otherPath[min(len(path)+1, len(otherPath)-1)][0])**2 + (successor_state[1] - otherPath[min(len(path)+1, len(otherPath)-1)][1])**2) <= 10:
                        check_successor = False 
                        break

                if check_successor and (successor_state, len(path)) not in closed_set:
                    new_path = path + [action]
                    new_cost = cost + step_cost
                    if heuristic_ip == 'E':
                        heuristic = heuristic_1(problem, successor_state)
                    elif heuristic_ip == 'M':
                        heuristic = heuristic_2(problem, successor_state)
                    elif heuristic_ip.isdigit():
                        heuristic = int(heuristic_ip) * heuristic_1(problem, successor_state)
                    fringe.put((new_cost + heuristic, [(successor_state, len(new_path)), new_path, new_cost]))

                elif not check_successor:
                    fringe.put((cost + 1, [(state, len(path)+1), path + ['s'], cost + 1]))


    return [], []
