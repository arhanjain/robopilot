import numpy as np
import pickle
import json

from envs.mujoco import MujocoSim

class WrappedEnv:
    def __init__(self, env):
        self.env_class = env
        self.env = self.env_class()
        self.trajectory = None
        self.ee_states = []
        self.traj_num = 0
    
    def step(self, action, ee_state):
        """
        action: [x, y, z]
        ee_state: 1 for open and 0 for close
        """
        # Store open or close gripper state
        self.ee_states.append(ee_state)
        if isinstance(self.env, MujocoSim):
            self.env.set_ee_pos(action)
            self.env.step()
        else:
            obs = self.env.step(action, ee_state)["image"]["844212071484"]
        # Update trajectory
        if self.trajectory is None:
            self.trajectory = [[None, action]]
        else:
            self.trajectory.append([None, action])
        
    def reset(self):
        # Save tuples of trajectory and end effector state
        res = {"traj": self.trajectory, "ee_states": self.ee_states}
        with open(f"trajectories/traj_{self.traj_num}.pkl", "wb") as f:
            pickle.dump(res, f)
        # with open(f"trajectories/traj_{self.traj_num}.pkl", "w") as f:
        #     json.dump(res, f)
        # Reset class variables
        self.env = self.env_class()
        self.trajectory = None
        self.ee_states = []
        # Increment trajectory number
        self.traj_num += 1
