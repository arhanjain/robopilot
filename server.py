# import zerorpc
#
# class RobotServer:
#     def initiate(self)

import mujoco
import mujoco.viewer
import numpy as np
import time
from threading import Thread
import zerorpc

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 1.0

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Maximum allowable joint velocity in rad/s. Set to 0 to disable.
max_angvel = 0.0

class FrankaSim:
    def __init__(self):
        assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

        # Load the model and data.
        model = mujoco.MjModel.from_xml_path("./mjctrl/franka_emika_panda/scene.xml")
        data = mujoco.MjData(model)
        self.data = data
        self.model = model

        # Override the simulation timestep.
        model.opt.timestep = dt

        # End-effector site we wish to control, in this case a site attached to the last
        # link (wrist_3_link) of the robot.
        self.site_id = model.site("attachment_site").id

        # Name of bodies we wish to apply gravity compensation to.
        body_names = [
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        ]
        body_ids = [model.body(name).id for name in body_names]
        if gravity_compensation:
            model.body_gravcomp[body_ids] = 1.0

        # Get the dof and actuator ids for the joints we wish to control.
        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        ]
        self.dof_ids = np.array([model.joint(name).id for name in joint_names])
        # Note that actuator names are the same as joint names in this case.
        self.actuator_ids = np.array([model.actuator(name).id for name in joint_names])

        # Initial joint configuration saved as a keyframe in the XML file.
        self.key_id = model.key("home").id

        # Mocap body we will control with our mouse.
        self.mocap_id = model.body("target").mocapid[0]

        # Pre-allocate numpy arrays.
        self.jac = np.zeros((6, model.nv))
        self.diag = damping * np.eye(6)
        self.error = np.zeros(6)
        self.error_pos = self.error[:3]
        self.error_ori = self.error[3:]
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        
        self.viewer = mujoco.viewer.launch_passive(
            model=model, 
            data=data, 
            show_left_ui=False, 
            show_right_ui=False
            )

        # Reset the simulation to the initial keyframe.
        mujoco.mj_resetDataKeyframe(model, data, self.key_id)

        # Initialize the camera view to that of the free camera.
        mujoco.mjv_defaultFreeCamera(model, self.viewer.cam)

        # Toggle site frame visualization.
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE


    # Define a trajectory for the end-effector site to follow.
    def circle(self, t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
        """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
        as a function of time t and frequency f."""
        x = r * np.cos(2 * np.pi * f * t) + h
        y = r * np.sin(2 * np.pi * f * t) + k
        return np.array([x, y])

        
    def step(self):
        data = self.data
        model = self.model
        dof_ids = self.dof_ids
        actuator_ids = self.actuator_ids

            # while self.viewer.is_running():
        step_start = time.time()

        # Set the target position of the end-effector site.
        # data.mocap_pos[self.mocap_id, 0:2] = self.circle(data.time, 0.1, 0.5, 0.0, 0.5)

        # Position error.
        self.error_pos[:] = data.mocap_pos[self.mocap_id] - data.site(self.site_id).xpos

        # Orientation error.
        mujoco.mju_mat2Quat(self.site_quat, data.site(self.site_id).xmat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, data.mocap_quat[self.mocap_id], self.site_quat_conj)
        mujoco.mju_quat2Vel(self.error_ori, self.error_quat, 1.0)

        # Get the Jacobian with respect to the end-effector site.
        mujoco.mj_jacSite(model, data, self.jac[:3], self.jac[3:], self.site_id)

        # Solve system of equations: J @ dq = error.
        dq = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.diag, self.error)

        # Scale down joint velocities if they exceed maximum.
        if max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > max_angvel:
                dq *= max_angvel / dq_abs_max

        # Integrate joint velocities to obtain joint positions.
        q = data.qpos.copy()
        mujoco.mj_integratePos(model, q, dq, integration_dt)

        # Set the control signal.
        np.clip(q, *model.jnt_range.T, out=q)
        data.ctrl[actuator_ids] = q[dof_ids]

        # Step the simulation.
        mujoco.mj_step(model, data)

        self.viewer.sync()
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    def set_ee_pos(self, xyz):
        self.data.mocap_pos[self.mocap_id] = xyz
        


if __name__ == "__main__":
    
    server = zerorpc.Server(FrankaSim())
    server.bind("tcp://0.0.0.0:4242")
    server.run()


