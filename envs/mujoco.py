import mujoco
import mujoco.viewer
import time
import numpy as np

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.005

# Nullspace P gain.
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

# Maximum allowable joint velocity in rad/s.
# max_angvel = 0.785
max_angvel = 1.5

class MujocoSim:
    def __init__(self):
        assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

        # Load the model and data.
        self.model = mujoco.MjModel.from_xml_path("./mjctrl/franka_emika_panda/scene.xml")
        self.data = mujoco.MjData(self.model)

        # Enable gravity compensation. Set to 0.0 to disable.
        self.model.body_gravcomp[:] = float(gravity_compensation)
        self.model.opt.timestep = dt

        # End-effector site we wish to control.
        site_name = "attachment_site"
        self.site_id = self.model.site(site_name).id

        # Get the dof and actuator ids for the joints we wish to control. These are copied
        # from the XML file. Feel free to comment out some joints to see the effect on
        # the controller.
        joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in joint_names])

        # Initial joint configuration saved as a keyframe in the XML file.
        key_name = "home"
        self.key_id = self.model.key(key_name).id
        self.q0 = self.model.key(key_name).qpos

        # Mocap body we will control with our mouse.
        mocap_name = "target"
        self.mocap_id = self.model.body(mocap_name).mocapid[0]

        # Pre-allocate numpy arrays.
        self.jac = np.zeros((6, self.model.nv))
        self.diag = damping * np.eye(6)
        self.eye = np.eye(self.model.nv)
        self.twist = np.zeros(6)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)


        self.viewer  = mujoco.viewer.launch_passive(
                model=self.model,
                data=self.data,
                show_left_ui=False,
                show_right_ui=False,
                )

        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)

        # Enable site frame visualization.
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        
    def step(self):
        data = self.data
        twist = self.twist
        site_quat = self.site_quat
        site_quat_conj = self.site_quat_conj
        error_quat = self.error_quat
        mocap_id = self.mocap_id
        jac = self.jac
        model = self.model
        diag = self.diag
        eye = self.eye
        q0 = self.q0
        viewer = self.viewer
        actuator_ids = self.actuator_ids
        dof_ids = self.dof_ids
        site_id = self.site_id
        step_start = time.time()

        # Spatial velocity (aka twist).
        dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
        twist[:3] = Kpos * dx / integration_dt
        mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
        mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
        twist[3:] *= Kori / integration_dt

        # Jacobian.
        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

        # Damped least squares.
        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)

        # Nullspace control biasing joint velocities towards the home configuration.
        dq += (eye - np.linalg.pinv(jac) @ jac) @ (Kn * (q0 - data.qpos[dof_ids]))

        # Clamp maximum joint velocity.
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > max_angvel:
            dq *= max_angvel / dq_abs_max

        # Integrate joint velocities to obtain joint positions.
        q = data.qpos.copy()  # Note the copy here is important.
        mujoco.mj_integratePos(model, q, dq, integration_dt)
        np.clip(q, *model.jnt_range.T, out=q)

        # Set the control signal and step the simulation.
        data.ctrl[actuator_ids] = q[dof_ids]
        mujoco.mj_step(model, data)

        viewer.sync()
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


    def set_ee_pos(self, xyz):
        if xyz is None or (xyz[0] == 0 and xyz[1] == 1 and xyz[2] == 2):
            return
        self.data.mocap_pos[self.mocap_id] = xyz
        
