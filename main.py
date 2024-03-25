#!/usr/bin/env python

"""
Mobile robot simulation setup
@author: Bijo Sebastian 
"""

#Import libraries
import time

#Import files
import sim_interface
import numpy as np

def main():
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
            robot1.goal_state = robot1.current_state #[0.0, 1.0, -np.pi/4.0]
            robot2.goal_state = robot2.current_state #[1.0, 0.0, -np.pi/4.0]
            robot3.goal_state = [0.0, -1.0, -np.pi/4.0]
            
            while not robot1.robot_at_goal() or not robot2.robot_at_goal() or not robot3.robot_at_goal():
                #Run the control loops for three robots
                robot1.run_controller()
                robot2.run_controller()
                robot3.run_controller()
            
            
            #Set arm position
            robot1.set_arm_position_youbot([0.0, 0.0, 0.0, 72.0, 0.0])
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
            

 