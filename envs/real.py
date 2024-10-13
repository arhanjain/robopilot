from droid.droid.robot_env import RobotEnv
import numpy as np
class RealWorld:
    def __init__(self):
        self.env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
        print("-------- Server Live! --------")

    def reset(self):
        return self.env.reset() 

    def step(self, action, ee_state):
        if action is None or (action[0] == 0 and action[1] == 1 and action[2] == 2):
            return 
        action = action + [np.pi,0,0, 0]
        action = np.array(action)
        print(action)
        return self.env.step(action)
