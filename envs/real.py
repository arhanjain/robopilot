from droid.droid.robot_env import RobotEnv
import numpy as np
class RealWorld:
    def __init__(self):
        self.env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
        print("-------- Server Live! --------")
        self.mini = np.array([np.inf, np.inf, np.inf])
        self.maxi = np.array([-np.inf, -np.inf, -np.inf])

    def reset(self):
        return self.env.reset() 

    def step(self, action, ee_state):
        if action is None or (action[0] == 0 and action[1] == 0 and action[2] == 0):
            return 

        self.mini = np.minimum(self.mini, action)
        self.maxi = np.maximum(self.maxi, action)

        print(self.mini,self.maxi)
        action = action + [np.pi,0,0, 0]
        action = np.array(action)
        print(action)
        return self.env.step(action)
