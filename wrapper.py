import numpy as np

from server import FrankaSim

class WrappedFrankaSim:
    def __init__(self):
        self.franka = FrankaSim()
        self.trajectory = None
    
    def step(self, action):
        self.franka.set_ee_position(action)
        self.franka.step()
        if self.trajectory is None:
            self.trajectory = np.array([[None, action]])
        else:
            self.trajectory = np.concatenate((self.trajectory, np.array([[self.trajectory[-1, 1], action]])), axis=0)
        
    def reset(self):
        self.franka = FrankaSim()
        self.trajectory = None