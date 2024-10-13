from droid.droid.robot_env import RobotEnv



class RealWorld:
    def __init__(self):
        self.env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")

    def reset(self):
        return self.env.reset() 

    def step(self, action):
        return self.env.step(action)
