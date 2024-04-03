#!/usr/bin/env python

"""
Mobile robot simulation setup
@author: Bijo Sebastian 
"""

#Import libraries
import time

#Import files
import sim_interface
import csv
import numpy as np

def main():

    with open('Paths.csv', newline='') as f:
        reader = csv.reader(f)
        Paths = list(reader)


    # Swapping x and y (cause matrix and coppeliasim have mismatch) and also converting to floats
    for i in range(len(Paths)):
        temp = float(Paths[i][0])
        Paths[i][0] = float(Paths[i][1])
        Paths[i][1] = temp

    # print(Paths)

    # Mapping CV pixels to Coppelisim 
    x_cv = [0,127]
    y_cv = [0,86]
    x_Csim = [-4.25, 4.25]
    y_Csim = [2.8, -2,8]

    for i in range(len(Paths)):
        Paths[i][0] = x_Csim[0] + Paths[i][0]* (x_Csim[1]-x_Csim[0])/(x_cv[1]-x_cv[0])
        Paths[i][1] = y_Csim[0] + Paths[i][1]* (y_Csim[1]-y_Csim[0])/(y_cv[1]-y_cv[0])

    print(Paths[0][:])

    if (sim_interface.sim_init()):

        #Create three robot and setup interface for all three 
        robot1 = sim_interface.youBot(1)
        robot2 = sim_interface.youBot(2)
        robot3 = sim_interface.youBot(3)

        #Start simulation
        if (sim_interface.start_simulation()):
            
            #Set goal state
            robot1.localize_robot()
            robot2.localize_robot()
            robot3.localize_robot()


            print(robot1.current_state)
            
            for i in range(len(Paths)):
                robot1.goal_state = [Paths[i][0], Paths[i][1], robot1.current_state[2]]
                robot2.goal_state = robot2.current_state
                robot3.goal_state = robot3.current_state
                
                
                while not robot1.robot_at_goal() or not robot2.robot_at_goal() or not robot3.robot_at_goal():
                    #Run the control loops for three robots
                    robot1.run_controller()
                    robot2.run_controller()
                    robot3.run_controller()
            
            
            #Set arm position
            robot1.set_arm_position_youbot([0.0, 0.0, 52.0, 72.0, 0.0])
            robot2.set_arm_position_youbot([0.0, 0.0, 52.0, 72.0, 0.0])
            robot3.set_arm_position_youbot([0.0, 0.0, 52.0, 72.0, 0.0])
            time.sleep(0)

        else:
            print ('Failed to start simulation')
    else:
        print ('Failed connecting to remote API server')
    
    #shutdown
    sim_interface.sim_shutdown()
    time.sleep(2.0)
    return

#run
if __name__ == '__main__':

    main()        
    print ('Program ended')
            

 