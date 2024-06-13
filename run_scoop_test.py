import os
import time
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
import random

# Initialize gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments()

# Configure simulation
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 5000.0
sim_params.substeps = 1
sim_params.flex.shape_collision_margin = 1e-4  # Default is 0.001
sim_params.flex.shape_collision_distance = 1e-4  # Default is 0.001

# Create simulation
sim = gym.create_sim(0, 0, gymapi.SIM_FLEX, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# Add ground plane with custom orientation
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)  # Change the normal vector to tilt the plane
plane_params.distance = 0  # Adjust the distance of the plane from the origin if necessary
gym.add_ground(sim, plane_params)

# Define asset root and asset files
gripper_asset_root = "./lc_soft_enable_wide_grip/"
gripper_asset_file = "lc_soft_enable_wide_grip.urdf"
# gripper_asset_root = "./"
# gripper_asset_file = 'franka_description/robots/franka_panda_fem_simple_v4_with_arm.urdf'

object_asset_root = "./examples/rectangle/"
object_asset_file = "soft_body.urdf"
# platform_asset_root = "./examples/"
# platform_asset_file = "platform.urdf"

# Load gripper, object, and platform assets
asset_options = gymapi.AssetOptions()
asset_options.flip_visual_attachments = False
asset_options.armature = 0.0  # Additional moment of inertia due to motors
# 1e-4  # Collision distance for rigid bodies. Minkowski sum of collision
# mesh and sphere. Default value is large, so set explicitly
asset_options.thickness = 0.0
asset_options.linear_damping = 1.0  # Linear damping for rigid bodies
asset_options.angular_damping = 0.0  # Angular damping for rigid bodies
# asset_options.disable_gravity = True
# Activates PD position, velocity, or torque controller, instead of doing
# DOF control in post-processing
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

asset_options.fix_base_link = True
gripper_asset = gym.load_asset(sim, gripper_asset_root, gripper_asset_file, asset_options)
asset_options.fix_base_link = False
asset_options.disable_gravity = False
object_asset = gym.load_asset(sim, object_asset_root, object_asset_file, asset_options)

# Create environment
num_envs = 1
spacing = 2.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
envs = []
gripper_handles = []
object_handles = []

for i in range(num_envs):
    # Create environment
    env = gym.create_env(sim, env_lower, env_upper, num_envs)
    envs.append(env)

    # Set gripper pose
    gripper_pose = gymapi.Transform()
    gripper_pose.p = gymapi.Vec3(0, 0, 0.3)
    gripper_pose.r = gymapi.Quat.from_euler_zyx(0, np.pi, 0)  # Rotate 180 degrees around x-axis

    # Create gripper actor
    gripper_handle = gym.create_actor(env, gripper_asset, gripper_pose, "gripper", i, 1)
    gripper_handles.append(gripper_handle)

    # Set object pose
    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(0.0, 0, 0.015)

    # Create object actor
    object_handle = gym.create_actor(env, object_asset, object_pose, "object", i, 2)
    object_handles.append(object_handle)

# Set viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# Set camera
cam_pos = gymapi.Vec3(.5, .5, .5)
cam_target = gymapi.Vec3(0, 0, 0.2)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Simulation loop
i = 0
while not gym.query_viewer_has_closed(viewer):
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Get and update state
    for env, (gripper_handle, object_handle) in zip(envs, zip(gripper_handles, object_handles)):
        # Close fingers to scoop
        dof_states = gym.get_actor_dof_states(env, gripper_handle, gymapi.STATE_ALL)
        dof_positions = dof_states['pos']

        # Example of closing gripper and lifting
        dof_positions[1] = np.min((i/2000, 1))  # Close left tip
        dof_positions[3] = np.min((i/2000, 1))   # Close right tip
        gym.set_actor_dof_position_targets(env, gripper_handle, dof_positions)

        # Example of closing gripper with torque noise
        dof_torques = np.zeros(dof_states['effort'].shape)

        # Apply random torque noise to joint 1 and joint 3
        torque_noise_1 = random.uniform(-0.01, 0.01)  # Random torque noise for joint 1
        torque_noise_3 = random.uniform(-0.01, 0.01)  # Random torque noise for joint 3

        dof_torques[1] = torque_noise_1  # Add torque noise to joint 1
        dof_torques[3] = torque_noise_3  # Add torque noise to joint 3

        gym.apply_actor_dof_efforts(env, gripper_handle, dof_torques)

    # Update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    i += 1

# Clean up
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)