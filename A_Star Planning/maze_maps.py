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
    
# Add 5x5 padding of 16 around all existing 16s
for row in range(1, max_row):
    for col in range(1, max_col):
        if map1[row][col] == 16:
            for i in range(max(0, row - 2), min(max_row + 1, row + 1)):
                for j in range(max(0, col - 2), min(max_col + 1, col + 1)):
                    if i != row or j != col:
                        map1[i][j] = 16

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

map_1.start = [40,9]
map_1.goal = [20, 120]

# print(map_1.map_data)
# print(f"{len(map_1.map_data)}, {len(map_1.map_data[0])}")

map_2 = Maps()
map_3 = Maps()

map_2.map_data = map1
map_2.start = [30, 20]
map_2.goal = [40, 100]

map_3.map_data = map1
map_3.start = [30, 12]
map_3.goal = [20, 100]

maps_dictionary = {1:map_1, 2:map_2, 3:map_3}


 
