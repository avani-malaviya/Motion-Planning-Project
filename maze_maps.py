import csv
import numpy as np
import pandas as pd

#Definitions based on color map
start_id = 1
goal_id = 8
obstacle_id = 16
beacon_id = 12
free_space_id1 = 3
free_space_id2 = 18
free_space_id1_cost = 1
free_space_id2_cost = 3
fringe_id = 4
expanded_id = 6

import csv

with open('refinedmap.csv', newline='') as f:
    reader = csv.reader(f)
    map = list(reader)

map1 = [ [0]*(int(map[-1][0])+1) for i in range(int(map[-1][1])+1)]

for i in range(len(map)):
    if (float(map[i][2]) >100) : map1[int(map[i][1])][int(map[i][0])] = 3
    else: map1[int(map[i][1])][int(map[i][0])] = 16

for i in range(len(map1)):
    map1[i][-1] = 16
    map1[i][0] = 16    
    

class Maps:
    """
    This class outlines the structure of the maps
    """    
    map_data = []
    start = []
    goal = []
    
#Maze maps
map_1 = Maps()
map_1.map_data = map1
# map_1.map_data = [
#      [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
#      [16,  3,  8,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
#      [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
#      [16,  3,  3,  3,  3,  3,  3,  3,  3, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],         
#      [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
#      [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
#      [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
#      [16,  3,  3, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,  3,  3, 16],
#      [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16,  3,  3,  3,  3,  3,  3, 16],
#      [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16,  3,  3,  3,  3,  3,  3, 16],
#      [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16,  3,  3,  3,  3,  3,  3, 16],
#      [16, 16, 16, 16, 16, 16, 16, 16, 16,  3,  3,  3, 16,  3,  3,  3,  3,  3,  3, 16],
#      [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
#      [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
#      [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
#      [16,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
#      [16,  1,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 16],
#      [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
#      ]

map_1.goal = [2, 40]
map_1.start = [26,2]
print(map_1.map_data)
print(f"{len(map_1.map_data)}, {len(map_1.map_data[0])}")

map_2 = Maps()
map_3 = Maps()

maps_dictionary = {1:map_1, 2:map_2, 3:map_3}


 
