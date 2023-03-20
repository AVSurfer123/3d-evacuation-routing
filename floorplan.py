import numpy as np
import os
import sys
import math

from scipy import ndimage

import pathfinder

# import the code from plane_segmentation
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import plane_segmentation.test_open3d as points

# FUNCTIONS TO USE LATER


# distance between two tuples
def distance(start, end):
    return math.sqrt((start[1] - end[1])**2+(start[0] - end[0])**2)

# place classifier on plan
def place_classifier(plan, cell_list, id):
    for cell in cell_list:
        plan[cell[0],cell[1]] = id

# place egress on plan
def place_shortest_egress(input_plan, output_plan, start, exits):
    # find evacuation path in the form of a list of tuples
    # for i, exit in enumerate(exits):
    #     if i == 0:
    #         shortest_d = math.sqrt((start[1] - exit[1])**2+(start[0] - exit[0])**2)
    #         closest_exit = exit
    #     else:
    #         d =  math.sqrt((start[1] - exit[1])**2+(start[0] - exit[0])**2)
    #         if d < shortest_d:
    #             closest_exit = exit
    
    # make sure that there is at least one exit
    for i, exit in enumerate(exits):               
        # for first loop (assign everything as if the shortest)
        if i == 0:
            # find egress path
            shortest_egress = pathfinder.main(input_plan, start, exit)
            # find length of egress path
            shortest_egress_length = 0
            for j in range(len(shortest_egress)-1):
                shortest_egress_length += distance(shortest_egress[j], shortest_egress[j+1])
            
            
        # for all the other loops (if shorter is found, replace original)
        else:
            # find egress path
            egress = pathfinder.main(input_plan, start, exit)
            
            # find length of egress path
            if len(egress) != 0:
                egress_length = 0
                for j in range(len(egress)-1):
                    egress_length += distance(egress[j], egress[j+1])
                if egress_length < shortest_egress_length:
                    shortest_egress = egress
                    shortest_egress_length = egress_length
                          
    # place start (2)
    place_classifier(output_plan, [start], 2)
    
    
    
    # place egress (4)
    # make an exception case if the above function doesn't find a path
    if shortest_egress_length > 0:
        # pop the first and last tuples because they're the start and exit
        shortest_egress = shortest_egress[1:-1]
        for cell in shortest_egress:
            output_plan[cell[0],cell[1]] = 4
        return True
    else:
        return False

#############################

# LOAD NP.ARRAY TO PLAN
raw_plan = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                    [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

# LOADING THE EXIT ARRAY (3)
exits = [(0,11),(0,12),(0,13),(1,0),(1,22),(11,0),(12,11),(12,12),(12,13),(11,22)]
place_classifier(raw_plan, exits, 0)

plan = np.copy(raw_plan)
place_classifier(plan, exits, 3)

# plan = np.load('numpy_grid.npy')

# GENERATE LENGTH AND WIDTH OF PLAN
LENGTH = points.RESOLUTION * plan.shape[0]
WIDTH = points.RESOLUTION * plan.shape[1]


# CREATING THE WALKABLE ARRAY (0)
start_id = np.where(plan == 0)
# print(walk_id)
starts = np.c_[start_id[0], start_id[1]]
starts = list(map(tuple, starts))

# CREATING THE WALL ARRAY (1)
walls_id = np.where(plan == 1)
# print(walls_id)
walls = np.c_[walls_id[0], walls_id[1]]
walls = list(map(tuple, walls))


# CREATING THE WALL OFFSETS (5)

# perform binary diltation
struct1 = ndimage.generate_binary_structure(2, 1)
struct2 = ndimage.generate_binary_structure(2, 2)

MIN_WIDTH = 0.05
iterations = 0.5*MIN_WIDTH/points.RESOLUTION

dilated_plan = ndimage.binary_dilation(raw_plan, structure=struct1, iterations=math.floor(1)).astype(raw_plan.dtype)
offset_plan = np.copy(dilated_plan)

offsets_plan = raw_plan != dilated_plan
offsets_id = np.where(offsets_plan == 1)
offsets = np.c_[offsets_id[0], offsets_id[1]]
offsets = list(map(tuple, offsets))


place_classifier(dilated_plan, exits, 3)
place_classifier(dilated_plan, offsets, 5)

exits_id = np.where(dilated_plan == 3)
# print(walls_id)
dilated_exits = np.c_[exits_id[0], exits_id[1]]
dilated_exits = list(map(tuple, dilated_exits))

starts_id = np.where(dilated_plan == 0)
# print(walls_id)
dilated_starts = np.c_[starts_id[0], starts_id[1]]
dilated_starts = list(map(tuple, dilated_starts))






