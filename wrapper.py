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
            self.trajectory = [[None, action]]
        else:
            self.trajectory.append([None, action])
        
    def reset(self):
        self.franka = FrankaSim()
        self.trajectory = None