import numpy as np
import robot_params
import time
import control

try:
  import sim
except:
  print ('--------------------------------------------------------------')
  print ('"sim.py" could not be imported. This means very probably that')
  print ('either "sim.py" or the remoteApi library could not be found.')
  print ('Make sure both are in the same folder as this file,')
  print ('or appropriately adjust the file "sim.py"')
  print ('--------------------------------------------------------------')
  print ('')

client_ID = []


def sim_init():
  global sim
  global client_ID
  
  #Initialize sim interface
  sim.simxFinish(-1) # just in case, close all opened connections
  client_ID=sim.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to CoppeliaSim    
  if client_ID!=-1:
    print ('Connected to remote API server')
    return True
  else:
    return False


def start_simulation():
  global sim
  global client_ID

  ###Start the Simulation: Keep printing out status messages!!!
  res = sim.simxStartSimulation(client_ID, sim.simx_opmode_oneshot_wait)

  if res == sim.simx_return_ok:
    print ("---!!! Started Simulation !!! ---")
    return True
  else:
    return False


def sim_shutdown():
  #Gracefully shutdown simulation

  global sim
  global client_ID

  #Stop simulation
  res = sim.simxStopSimulation(client_ID, sim.simx_opmode_oneshot_wait)
  if res == sim.simx_return_ok:
    print ("---!!! Stopped Simulation !!! ---")

  # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
  sim.simxGetPingTime(client_ID)

  # Now close the connection to CoppeliaSim:
  sim.simxFinish(client_ID)      

  return


class youBot:    

    # Handles
    wheel_handles = None 
    arm_handles = None
    youbot_handle = None
    
    # state variables
    goal_state = [0.0, 1.0, -np.pi/4.0] #x, y, theta
    current_state = [0.0, 0.0, 0.0] #x, y, theta
    
    robot_control = control.Go_to_goal_controller()

    def __init__(self, id):
        """
        Sets up the youBot
        """
        #Set up the name for the robot, must match coppeliasim 
        self.name = "/youBot"+str(id)
        self.get_handles() 
        self.set_vel_youbot(0.0, 0.0, 0.0)
        return      

    def get_handles(self):
      #Get the handles to the sim items
    
      # Get handles
      res , self.youbot_handle = sim.simxGetObjectHandle(client_ID, self.name, sim.simx_opmode_blocking)
      self.wheel_handles = [-1, -1, -1, -1]
      res,  self.wheel_handles[0] = sim.simxGetObjectHandle(client_ID, self.name + "/rollingJoint_fl", sim.simx_opmode_blocking)
      res,  self.wheel_handles[1] = sim.simxGetObjectHandle(client_ID, self.name + "/rollingJoint_rl", sim.simx_opmode_blocking)
      res,  self.wheel_handles[2] = sim.simxGetObjectHandle(client_ID, self.name + "/rollingJoint_rr", sim.simx_opmode_blocking)
      res,  self.wheel_handles[3] = sim.simxGetObjectHandle(client_ID, self.name + "/rollingJoint_fr", sim.simx_opmode_blocking)
      self.arm_handles = [-1, -1, -1, -1, -1]
      res,  self.arm_handles[0] = sim.simxGetObjectHandle(client_ID, self.name + "/youBotArmJoint0", sim.simx_opmode_blocking)
      res,  self.arm_handles[1] = sim.simxGetObjectHandle(client_ID, self.name + "/youBotArmJoint1", sim.simx_opmode_blocking)
      res,  self.arm_handles[2] = sim.simxGetObjectHandle(client_ID, self.name + "/youBotArmJoint2", sim.simx_opmode_blocking)
      res,  self.arm_handles[3] = sim.simxGetObjectHandle(client_ID, self.name + "/youBotArmJoint3", sim.simx_opmode_blocking)
      res,  self.arm_handles[4] = sim.simxGetObjectHandle(client_ID, self.name + "/youBotArmJoint4", sim.simx_opmode_blocking)
      
      # Get the position of the YouBot for the first time in streaming mode
      res , youbot_1_Position = sim.simxGetObjectPosition(client_ID, self.youbot_handle, -1 , sim.simx_opmode_streaming)
      res , youbot_1_Orientation = sim.simxGetObjectOrientation(client_ID, self.youbot_handle, -1 , sim.simx_opmode_streaming)
      
      # Stop all joint actuations:Make sure Youbot is stationary
      for i in range(4):
          res = sim.simxSetJointTargetVelocity(client_ID, self.wheel_handles[i], 0.0, sim.simx_opmode_streaming)
      
      #Set arm to staright up
      for i in range(5):
          res = sim.simxSetJointTargetPosition(client_ID, self.arm_handles[i], 0.0, sim.simx_opmode_streaming)
      
      print ("Succesfully obtained handles")
    
      return
    
    
    
    def localize_robot(self):
      #Function that will return the current location of youbot
      #PS. THE ORIENTATION WILL BE RETURNED IN RADIANS        
      
      res , youbot_Position = sim.simxGetObjectPosition(client_ID, self.youbot_handle, -1 , sim.simx_opmode_buffer)
      res , youbot_Orientation = sim.simxGetObjectOrientation(client_ID, self.youbot_handle, -1 , sim.simx_opmode_buffer)
      
      x = youbot_Position[0]
      y = youbot_Position[1]
      theta  = youbot_Orientation[1]
      
      self.current_state = [x,y,theta]
      return            
    
    def set_vel_youbot(self, Vx, Vy, W):
      #Function to set the linear and rotational velocity of youbot

      #print("Velocities", Vx, Vy, W)
              
      # Set velocity
      sim.simxSetJointTargetVelocity(client_ID, self.wheel_handles[0], -Vx -Vy -W, sim.simx_opmode_oneshot_wait)
      sim.simxSetJointTargetVelocity(client_ID, self.wheel_handles[1], -Vx +Vy -W, sim.simx_opmode_oneshot_wait)
      sim.simxSetJointTargetVelocity(client_ID, self.wheel_handles[2], -Vx -Vy +W, sim.simx_opmode_oneshot_wait)
      sim.simxSetJointTargetVelocity(client_ID, self.wheel_handles[3], -Vx +Vy +W, sim.simx_opmode_oneshot_wait)
      
      return  
    
    def set_arm_position_youbot(self, theta_desired):
      
      #Set arm to desired configuration
      for i in range(5):
          sim.simxSetJointTargetPosition(client_ID, self.arm_handles[i], np.deg2rad(theta_desired[i]), sim.simx_opmode_oneshot_wait)
          
      return
  
    def robot_at_goal(self):
        #Check if robot at goal
        #print (self.current_state)
        #print (self.goal_state)
        self.localize_robot()
        return self.robot_control.at_goal(self.current_state, self.goal_state)
    
  
    def run_controller(self):
        #Run the gtg_control lop for the robot

        #Localise the robot 
        self.localize_robot()
        print(self.current_state)
        #Check if robot is at goal
        if not self.robot_at_goal():
            #Run control loop 
            Vx, Vy, W = self.robot_control.gtg(self.current_state, self.goal_state)
            self.set_vel_youbot(Vx, Vy, W)
            time.sleep(0.5)
        else:
            #Stop robot
            print("Reached local goal")
            self.set_vel_youbot(0.0, 0.0, 0.0)
      
  

