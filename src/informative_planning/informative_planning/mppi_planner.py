#!/usr/bin/env python3

import torch
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Point, Pose, Twist
from ros_gz_interfaces.msg import ParamVec
from nav_msgs.msg import OccupancyGrid, GridCells, Odometry
import numpy as np
import tf_transformations
from std_msgs.msg import Header, Float32, Float64, Int32, Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from usv_navigation.utils import pose_to_numpy, SyncSubscription
from vrx_custom_msgs.srv import RemoveObject
from ros_gz_interfaces.msg import ParamVec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation
from scipy.ndimage import shift
from informative_planning.utils import *
from informative_planning.params import *
from functools import partial
from informative_planning.mppi_torch.mppi import MPPIPlanner
import onnxruntime as ort
import os
import yaml

# set device for torch operations:
if torch.cuda.is_available():
    def_device = 'cuda'
else:
    def_device = 'cpu'

path = '/vrx_ws/src/informative_planning/informative_planning/mppi_config.yaml'
config = yaml.safe_load(open(path))

class MPPIInfPlannerNode(Node): 
    """
    Hybrid planner
    MPPI for global/exploration planning
    pure pursuit for tracking
    Publish velocity command in speed/yaw
    
    later todo - make it modular to change planners at each level
    """
    
    def __init__(self):
        
        super().__init__("mppi_planner")
        
        self.mode_publisher = self.create_publisher(Int32, 'wamv/guidance_mode', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist,'wamv/cmd_vel',10)
        self.ce_publisher = self.create_publisher(Float64, 'wamv/control_effort', 10)
        self.traj_publisher = self.create_publisher(Float64MultiArray, 'mppi/trajs', 10)
        self.traj_costs_publisher = self.create_publisher(Float64MultiArray, 'mppi/costs', 10) 
        self.int_pos_publisher = self.create_publisher(Float64MultiArray, 'wamv/intercepted_positions', 10)
        self.target_pos_publisher = self.create_publisher(Pose, 'wamv/nearest_target', 10)

        self.odom_subscriber = self.create_subscription(Odometry, '/wamv/sensors/position/ground_truth_odometry', self.odom_callback, 10)
        self.og_subscriber = self.create_subscription(OccupancyGrid, '/gridmap', self.og_callback,10)
        self.visit_og_subscriber = self.create_subscription(OccupancyGrid, '/visit_score_map', self.visit_og_callback,10)
        self.object_det_subscriber = self.create_subscription(ParamVec, '/wamv/yolo/results', self.object_detect_callback, 10)
        self.object_pos_subscriber = self.create_subscription(PoseArray, '/buoy_positions', self.object_position_callback, 10)
        self.num_intercept_subscriber = self.create_subscription(Int32, 'wamv/num_interceptions', self.num_intercepted_callback, 10)

        wind_df = {'/vrx/debug/wind/direction': Float32, '/vrx/debug/wind/speed': Float32}
        self.wind_sub = SyncSubscription(self, wind_df, self.wind_callback,10,1)
        self.occupancy_grid = None

        self.origin = origin
        self.r_tol = r_tol
        self.pos = None
        self.trigger = False       
        self.wind_trigger = False
        self.p_low = p_low
        self.p_high = p_high
        self.alpha, self.beta = alpha, beta # dynamic occupancy grid update factor
        self.d, self.theta = d_max, theta  # field of view distance and angle
        self.wind_factor = 0.03
        self.desired_speed = u
        self.cmd_vel = [u, 0.0]  # initial command velocity
        self.cmd_vel_history = np.empty((0, 2))
        self.l_low, self.l_high = to_logodds(p_low), to_logodds(p_high)
        self.decay_rate = decay_rate

        self.guidance_mode = 0  # default to exploration mode
        self.intercepted = 0
        self.wait_until_explore = 0
        self.wait_until_intercept = 0
        self.t_step = time_step #time_step for planner
        self.horizon = config['mppi']['horizon']
        self.T = config['mppi']['horizon'] * self.t_step  # total prediction horizon
        self.r_min = config['mppi']['u_min'][1]
        self.r_max = config['mppi']['u_max'][1]
        self.u_min = config['mppi']['u_min'][0]
        self.u_max = config['mppi']['u_max'][0]
        self.gamma = config['mppi']['rollout_var_discount']
        self.gamma_seq = torch.tensor([self.gamma ** (t) for t in range(self.horizon)], device=torch.device(def_device))
        self.tracking_timer = self.create_timer(0.1, self.guidance)
        self.ipp_timer = self.create_timer(config['mppi']['control_step'], self.informative_global_planner)
        self.timer = self.create_timer(0.1, self.cmd_vel_callback)                       
        self.path_followed_0 = []
        self.path_followed_1 = []
        self.intercept_positions = []
        
        # MPPI init
        self.mppi_planner = MPPIPlanner(cfg=config['mppi'], nx=3, 
                                        dynamics=self.state_transition,running_cost=self.compute_cost,
                                        )
        self.prev_action = self.cmd_vel.copy()

        self.tracking_mode = 'pure_pursuit'
        self.track_time_step = track_time_step

        self.counter = 1 # for plotting

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = [
            ('CUDAExecutionProvider', {"device_id": 0}),
            'CPUExecutionProvider'
        ]
        self.sess = ort.InferenceSession(
            '/vrx_ws/src/informative_planning/models/model_jetson_sim.onnx',
            sess_options=so,
            providers=providers
        )
        self.get_logger().info(f'Loaded ONNX model with providers: {self.sess.get_providers()}')
        self.input_names = [inp.name for inp in self.sess.get_inputs()]
        
    def odom_callback(self, msg):
            
        self.pos = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        q=[orientation_q.x,orientation_q.y,orientation_q.z,orientation_q.w]
        self.angular_pos = tf_transformations.euler_from_quaternion(q)
        self.psi= self.angular_pos[2]
        self.linear_vel = msg.twist.twist.linear
        self.angular_vel = msg.twist.twist.angular        
        self.r = self.angular_vel.z
        self.trans, self.rot = pose_to_numpy(msg.pose.pose)
        
        if self.guidance_mode == 0:
            self.path_followed_0.append([self.pos.x, self.pos.y])
        else:
            self.path_followed_1.append([self.pos.x, self.pos.y])
        
        if self.trigger == False:
            self.trigger = True
            self.start_time = self.get_time()

        if self.get_time() - self.start_time > mission_time:
            self.get_logger().info("Mission time over. Stopping the planner.")
            rclpy.shutdown()
                   
        elif self.intercepted >= max_targets:
            self.get_logger().info(f"Successfully intercepted {max_targets} targets. Stopping the planner.")
            rclpy.shutdown()

    def num_intercepted_callback(self, msg):
        self.intercepted = msg.data

    def wind_callback(self,dir:Float32,speed:Float32):
        self.wind_speed = speed.data
        self.wind_dir = dir.data * np.pi / 180.0  # Convert degrees to radians
        self.wind_factor = 0.02 + 0.0015 * self.wind_speed
        if not self.wind_trigger:
            self.wind_trigger = True
    
    def og_callback(self, msg:OccupancyGrid):
        
        if self.trigger == False or self.wind_trigger == False:
            return
        
        w, h = msg.info.width, msg.info.height
        self.grid_size = [w, h]
        self.resolution = msg.info.resolution
        og = np.array(msg.data, dtype=np.int8).reshape(h,w)
        og = og/100.0
        og[og == -1/100] = 0.5
        self.occupancy_grid = og.copy()
        
    def visit_og_callback(self, msg:OccupancyGrid):

        if self.occupancy_grid is None:
            return
        
        visit_og = np.array(msg.data, dtype=np.int8).reshape(self.grid_size[1],self.grid_size[0])
        visit_og = visit_og/100.0
        self.visit_og = visit_og.copy()

    def guidance(self):

        if self.guidance_mode == 0:
            return        

        self.get_logger().info("Intercepting")
        if self.tracking_mode == 'mppi':
            self.tracking_controller()
        elif self.tracking_mode == 'pure_pursuit':
            self.pure_pursuit()

        return

    def informative_global_planner(self):

        if self.trigger == False or self.occupancy_grid is None:
            return

        t1 = self.get_time()
        
        # spatiotemporal target predictions
        binary_grid = self.occupancy_grid.copy()
        ids = binary_grid > 0.5
        binary_grid[ids] = 1
        binary_grid[~ids] = 0

        self.pred_unc_grids = self.predictions(binary_grid)

        action = self.mppi_planner.command(np.array([self.pos.x, self.pos.y, self.psi]))
        self.prev_action = action.cpu().numpy().tolist()
        u, r = self.prev_action
        psi_d = ssa(self.psi + r * self.t_step)

        t2 = self.get_time()
        self.get_logger().info(f"Time taken for informative global planning step: {t2 - t1} s")

        if self.guidance_mode == 0:
            self.cmd_vel = [u, r]

        return None

    def cmd_vel_callback(self):

        twist_msg = Twist()
        twist_msg.linear.x = self.cmd_vel[0]
        twist_msg.angular.z = self.cmd_vel[1]

        self.cmd_vel_history = np.append(self.cmd_vel_history, [self.cmd_vel], axis=0)
        # calculate control effort
        if len(self.cmd_vel_history) > 1:
            delta_u = self.cmd_vel_history[1:, 0] - self.cmd_vel_history[:-1, 0]
            delta_r = self.cmd_vel_history[1:, 1] - self.cmd_vel_history[:-1, 1]

            # normalise
            delta_u /= (self.u_max - self.u_min)
            delta_r /= (self.r_max - self.r_min)

            ce = np.mean(np.abs(delta_u) + np.abs(delta_r))
            self.ce_publisher.publish(Float64(data=ce))

        self.cmd_vel_publisher.publish(twist_msg)
        self.mode_publisher.publish(Int32(data=self.guidance_mode))
        return

    def predictions(self, grid):

        vx = self.wind_speed * np.cos(self.wind_dir)
        vy = self.wind_speed * np.sin(self.wind_dir)

        # Pad grid to 100x100 to match neural network input size
        extended_grid = np.zeros((100, 100), dtype=np.float32)
        extended_grid[0:self.grid_size[0], 0:self.grid_size[1]] = grid

        input_grids = np.array([extended_grid.copy() for _ in range(int(self.T // self.t_step))])
        input_grids = np.expand_dims(input_grids, axis=-1)  # Add channel dimension
        input_params = np.array([[vx, vy, t] for t in np.arange(self.t_step, self.T + self.t_step, self.t_step)])

        inputs = {self.input_names[0]: input_grids.astype(np.float32),
                  self.input_names[1]: input_params.astype(np.float32)}
        predictions = self.sess.run(None, inputs)[0]

        predictions[predictions < 0.01] = 0
        predictions = torch.tensor(predictions.squeeze(-1), dtype=torch.float32, device=torch.device(def_device))

        # Crop back to actual grid size (model input was padded to 100x100)
        predictions = predictions[:, :self.grid_size[1], :self.grid_size[0]]

        return predictions

    def predictions_without_nn(self, grid):
        
        predictions = []
        
        for i in range(self.horizon):
            pred_grid = self.generate_pred_grid(grid, (i+1)*self.t_step)
            predictions.append(pred_grid)

        return torch.tensor(np.array(predictions), dtype=torch.float32, device=torch.device(def_device))

    def generate_pred_grid(self, grid, t):

        num_ob = np.sum(grid > 0.5)
        
        ob_indices = np.argwhere(grid > 0.5)
        ob_indices = [(ind[1], ind[0]) for ind in ob_indices]

        vx = self.wind_speed * np.cos(self.wind_dir)
        vy = self.wind_speed * np.sin(self.wind_dir)
        
        if num_ob == 0:
            return grid.copy()

        dx = self.wind_factor * vx * t
        dy = self.wind_factor * vy * t

        sig_x = 0.5 * np.linalg.norm([dx, dy])
        sig_y = 0.2 * np.linalg.norm([dx, dy])

        if np.linalg.norm([dx, dy]) == 0:
            sig_x = 0.1*t
            sig_y = 0.1*t
            
        theta = np.arctan2(vy, vx)
        D = np.diag([sig_x**2, sig_y**2])
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        Sig = R @ D @ R.T
        invS = np.linalg.inv(Sig)
        norm = 1/ (2 * np.pi * sig_x * sig_y)

        centers = np.array(ob_indices) + np.array([int(dx), int(dy)])

        result = np.zeros((100, 100))
        x, y = np.linspace(0, 99, 100), np.linspace(0, 99, 100)
        X, Y = np.meshgrid(x, y)

        min_thresh = 0.01
        local_grid_size = 10

        for ind in centers:
            x0, y0 = ind
            
            # Calculate grid bounds with bounds checking
            i_min = max(0, int(x0 - local_grid_size//2))
            i_max = min(100, int(x0 + local_grid_size//2 + (local_grid_size % 2)))
            j_min = max(0, int(y0 - local_grid_size))
            j_max = min(100, int(y0 + local_grid_size + (local_grid_size % 2)))
            
            # Skip if window is completely outside the grid
            if i_max <= i_min or j_max <= j_min:
                continue
            
            # Calculate centered coordinates for local grid
            x_centered = X[j_min:j_max, i_min:i_max] - x0
            y_centered = Y[j_min:j_max, i_min:i_max] - y0

            # to make the local gaussian with finite support
            local_gaussian = norm * np.exp(-0.5 * (invS[0, 0] * x_centered**2 + invS[1, 1] * y_centered**2 + 
                                             2 * invS[0, 1] * x_centered * y_centered))

            local_gaussian[local_gaussian < min_thresh] = 0

            clipped = (x0-2*sig_x <= 0 or x0-2*sig_x >= 100 or y0-2*sig_y <= 0 or y0-2*sig_y >= 100)
            if np.sum(local_gaussian)> 0 and not clipped:
                local_gaussian = local_gaussian/np.sum(local_gaussian)

            result[j_min:j_max, i_min:i_max] += local_gaussian

        return result

    def state_transition(self, state, action, t=None):
        
        u = action[:, 0]
        r = action[:, 1]

        # limit the change in heading
        psi = state[:, 2] + r * self.t_step
        # update state
        psi = (psi + torch.pi) % (2 * torch.pi) - torch.pi
        x_new = state[:, 0] + u * torch.cos(psi) * self.t_step
        y_new = state[:, 1] + u * torch.sin(psi) * self.t_step

        ## if out of bounds, clamp
        x_new = torch.clamp(x_new, self.origin[0], self.origin[0] + self.grid_size[0]*self.resolution)
        y_new = torch.clamp(y_new, self.origin[1], self.origin[1] + self.grid_size[1]*self.resolution)

        new_state = torch.stack([x_new, y_new, psi], axis=-1)

        return new_state, action

    def nominal_trajectories(self):
        # nominal trajectories based on constant speed and yaw rate

        u_vals = torch.linspace(self.u_min, self.u_max, steps=3, device=def_device)
        r_vals = torch.linspace(-0.15, 0.15, steps=7, device=def_device)
        nominal_traj = torch.cartesian_prod(u_vals, r_vals)  # shape (num_traj, 2)

        return nominal_traj

    def explore_costs(self, states, actions, fov_all=None):

        K = states.shape[0]
        T = states.shape[1]
        device = states.device

        ## expand visit_og to batch_size
        visit_og = torch.tensor(self.visit_og, device=device).float()
        visit_og_batch = visit_og.unsqueeze(0).expand(K, -1, -1).clone()

        explore_cost = torch.zeros((K, T), device=device)

        for t in range(T):
            if fov_all is not None:
                fov_batch = fov_all[:, t]                                          # Jetson: from pre-computed (K,T,H,W)
            else:
                fov_batch = batch_get_fov(                                         # dev machine: one (K,H,W) at a time
                    states[:, t, 0], states[:, t, 1], states[:, t, 2], device=device
                )

            ## compute information gain - sum of visit grid values in fov
            explore_cost[:, t] = torch.sum(visit_og_batch * fov_batch, dim=(-1, -2))

            ## advance visit grid: decay unvisited, reset visited cells to 0
            visit_og_batch = visit_og_batch + self.decay_rate * self.t_step
            visit_og_batch = torch.where(fov_batch, torch.zeros_like(visit_og_batch), visit_og_batch)
            visit_og_batch = torch.clamp(visit_og_batch, 0, 1)

        explore_cost /= (self.grid_size[0] * self.grid_size[1])  # normalize by grid size
        explore_cost_sum = torch.sum(self.gamma_seq[:T] * explore_cost, dim=(-1))  # discounted sum over time

        return -explore_cost_sum

    def target_costs(self, states, actions, fov_all=None):

        K = states.shape[0]
        T = states.shape[1]
        device = states.device

        # World coords (cached module-level constants) — move to device once
        x_world = x_world_grid.to(device)  # (H, W)
        y_world = y_world_grid.to(device)

        pred_unc = self.pred_unc_grids.to(device)  # (T, H, W)

        # ── Loop version (dev machine / Gazebo): peak ~6×(K,H,W) = 6×5.7 MB ──
        # Swap with the vectorised block below for Jetson (no Gazebo, more VRAM headroom).
        target_cost_sum = torch.zeros(K, device=device)
        for t in range(T):
            if fov_all is not None:
                fov_t = fov_all[:, t].float()                                      # Jetson: from pre-computed (K,T,H,W)
            else:
                fov_t = batch_get_fov(                                             # dev machine: one (K,H,W) at a time
                    states[:, t, 0], states[:, t, 1], states[:, t, 2], device=device
                ).float()

            visible_cells_t = pred_unc[t].unsqueeze(0) * fov_t  # (K, H, W)

            center_x = states[:, t, 0].view(K, 1, 1)
            center_y = states[:, t, 1].view(K, 1, 1)
            psi      = states[:, t, 2].view(K, 1, 1)

            dx = x_world - center_x  # (K, H, W)
            dy = y_world - center_y
            dist = torch.sqrt(dx**2 + dy**2)
            del dx, dy

            angles = torch.atan2(y_world - center_y, x_world - center_x)
            rel_bearings = ssa(angles - psi)  # (K, H, W)
            del angles

            fov_distances = dist * fov_t
            del dist

            track_ig = -visible_cells_t * torch.exp(-0.04 * fov_distances**2 - 4 * rel_bearings**2)
            del fov_distances, rel_bearings, visible_cells_t

            target_cost_sum += self.gamma_seq[t] * torch.sum(track_ig * fov_t, dim=(-1, -2))

        # ── Vectorised version (Jetson / no Gazebo): peak ~6×(K,T,H,W) = 6×86 MB ──
        # fov_float = fov_all.float()  # (K, T, H, W)
        # pred_unc_b = pred_unc.unsqueeze(0)  # (1, T, H, W)
        # visible_cells = pred_unc_b * fov_float  # (K, T, H, W)
        # center_x = states[:, :, 0].view(K, T, 1, 1)
        # center_y = states[:, :, 1].view(K, T, 1, 1)
        # psi      = states[:, :, 2].view(K, T, 1, 1)
        # dx = x_world - center_x
        # dy = y_world - center_y
        # dist = torch.sqrt(dx**2 + dy**2)
        # del dx, dy
        # angles = torch.atan2(y_world - center_y, x_world - center_x)
        # rel_bearings = ssa(angles - psi)
        # del angles
        # fov_distances = dist * fov_float
        # track_ig = -visible_cells * torch.exp(-0.04 * fov_distances**2 - 4 * rel_bearings**2)
        # target_costs = torch.sum(track_ig * fov_float, dim=(-1, -2))  # (K, T)
        # target_cost_sum = torch.sum(self.gamma_seq[:T] * target_costs, dim=-1)  # (K,)

        return target_cost_sum

    def path_costs(self, states, actions):

        u = actions[:, :, 0]
        r = actions[:, :, 1]

        # promote higher speed and limit turns
        u_norm = (u - self.u_min) / (self.u_max - self.u_min)
        r_norm = torch.abs(r) / (self.r_max)
        
        path_cost = (1 - u_norm)**2 + r_norm**2
        path_cost = torch.sum(self.gamma_seq * path_cost, dim=-1)  # discounted sum over time

        return path_cost
    
    def acceleration_costs(self, states, actions):

        # states are K x T x nx
        # actions are K x T x nu
        
        u = actions[:, :, 0]
        r = actions[:, :, 1]

        u_ = torch.roll(u, shifts=1, dims=1)
        u_[:,0] = self.prev_action[0]
        r_ = torch.roll(r, shifts=1, dims=1)
        r_[:,0] = self.prev_action[1]
        
        delta_u = (u[:, :] - u_[:, :]) / (self.u_max - self.u_min)
        delta_r = (r[:, :] - r_[:, :])/(self.r_max - self.r_min) 
        # limit accelerations        

        action_cost = delta_u**2 + delta_r**2
        action_cost = torch.sum(self.gamma_seq * action_cost, dim=-1)  # discounted sum over time

        return action_cost
        
    def collision_costs(self, states):
        
        # states are K x T x nx
        K = states.shape[0]
        n_points = states.shape[1]
        device = states.device

        xt = states[:, :, 0]
        yt = states[:, :, 1]

        x_min, y_min = self.origin[0], self.origin[1]
        x_max = self.origin[0] + self.grid_size[0]*self.resolution
        y_max = self.origin[1] + self.grid_size[1]*self.resolution

        d_left, d_right = xt - x_min, x_max - xt
        d_top, d_bottom = yt - y_min, y_max - yt

        d_min = torch.minimum(torch.minimum(d_left, d_right), torch.minimum(d_top, d_bottom))
        # barrier function for collision
        d_safe = 1.0
        coll_cost = torch.nn.functional.softplus(d_safe - d_min)
        
        # discounted sum over time
        coll_cost = torch.sum(self.gamma_seq * coll_cost, dim=-1)

        return coll_cost

    @torch.no_grad()
    def compute_cost(self, states, actions):

        # Dev machine: FoV computed per-timestep inside each cost fn
        # uncomment below and pass fov_all to both cost fns for faster vectorised path.
        # fov_all = batch_get_fov_kt(
        #     states[:, :, 0], states[:, :, 1], states[:, :, 2],
        #     device=states.device
        # )  # (K, T, H, W)
        # explore_cost = self.explore_costs(states, actions, fov_all=fov_all)
        # track_cost = self.target_costs(states, actions, fov_all=fov_all)

        explore_cost = self.explore_costs(states, actions)
        track_cost = self.target_costs(states, actions)
        path_cost = self.path_costs(states, actions)
        coll_cost = self.collision_costs(states)

        cost = a1 * explore_cost + a2 * track_cost + a3 * coll_cost + a4 * path_cost
        print('mean, min, max explore cost:', torch.mean(explore_cost).item(), torch.min(explore_cost).item(), torch.max(explore_cost).item())
        print('mean, min, max track cost:', torch.mean(track_cost).item(), torch.min(track_cost).item(), torch.max(track_cost).item())
        print('mean, min, max path cost:', torch.mean(path_cost).item(), torch.min(path_cost).item(), torch.max(path_cost).item())
        print('mean, min, max collision cost:', torch.mean(coll_cost).item(), torch.min(coll_cost).item(), torch.max(coll_cost).item())

        self.counter +=1
        self.publish_traj(states[:,:,:2], cost)

        return cost

    def publish_traj(self, trajs, costs):

        # trajs: K x T x 2 tensor (x, y only)
        # costs: K tensor
        K, T, _ = trajs.shape

        trajs_msg = Float64MultiArray()
        trajs_msg.layout.dim = [
            MultiArrayDimension(label='K', size=K, stride=K*T*2),
            MultiArrayDimension(label='T', size=T, stride=T*2),
            MultiArrayDimension(label='xy', size=2, stride=2),
        ]
        trajs_msg.layout.data_offset = 0
        trajs_msg.data = trajs.cpu().numpy().flatten().tolist()
        self.traj_publisher.publish(trajs_msg)

        costs_msg = Float64MultiArray()
        costs_msg.layout.dim = [
            MultiArrayDimension(label='K', size=K, stride=1),
        ]
        costs_msg.layout.data_offset = 0
        costs_msg.data = costs.cpu().numpy().flatten().tolist()
        self.traj_costs_publisher.publish(costs_msg)

        int_pos_msg = Float64MultiArray()
        int_pos_msg.layout.dim = [
            MultiArrayDimension(label='N', size=len(self.intercept_positions), stride=2),
            MultiArrayDimension(label='xy', size=2, stride=2),
        ]

        int_pos_msg.layout.data_offset = 0
        int_pos_msg.data = []
        if len(self.intercept_positions) > 0:
            int_pos_msg.data = np.array(self.intercept_positions).flatten().tolist()
        self.int_pos_publisher.publish(int_pos_msg)

        # publish nearest target position
        if self.nearest_target_pos is not None:
            target_pos_msg = Pose()
            target_pos_msg.position.x = self.nearest_target_pos[0]
            target_pos_msg.position.y = self.nearest_target_pos[1]
            self.target_pos_publisher.publish(target_pos_msg)

    ### Pure pursuit guidance for tracking
    def pure_pursuit(self):
        
        xt, yt = self.nearest_target_pos[0], self.nearest_target_pos[1]
        x, y = self.pos.x, self.pos.y
        psi_des = np.arctan2(yt - y, xt - x)

        k_r = 1/self.track_time_step
        r = ssa(psi_des - self.psi)* k_r
        self.cmd_vel = [self.desired_speed, r]

    ### MPPI tracking controller

    def compute_cost_track(self, states, actions):
        
        action_cost = self.path_costs_track(states, actions)
        tracking_cost = self.track_cost(states, actions)
        
        print('mean, min, max action cost:', torch.mean(action_cost).item(), torch.min(action_cost).item(), torch.max(action_cost).item())
        print('mean, min, max tracking cost:', torch.mean(tracking_cost).item(), torch.min(tracking_cost).item(), torch.max(tracking_cost).item())
        cost = 0.1 * action_cost + tracking_cost
        
        return cost

    def path_costs_track(self, states, actions):
        
        # states are K x T x nx
        # actions are K x T x nu
        
        u = actions[:, :, 0]
        r = actions[:, :, 1]

        u_norm = (u - self.u_min) / (self.u_max - self.u_min)
        r_norm = torch.abs(r) / (self.r_max)

        action_cost = torch.sum((1 - u_norm)**2, dim=(-1)) + torch.sum(r_norm**2, dim=(-1))

        return action_cost
    
    def track_cost(self, states, actions):

        # states are K x T x nx
        K = states.shape[0]
        T = states.shape[1]
        device = states.device

        x_v = states[:, :, 0]
        y_v = states[:, :, 1]

        weighted_cost = torch.zeros((K,), device=device)

        # forward simulate target dynamics
        init_pos = self.nearest_target_pos
        target_positions = []
        
        for i in range(1,T+1):
            x = init_pos[0] + self.wind_factor * self.wind_speed * np.cos(self.wind_dir) * i * self.t_step
            y = init_pos[1] + self.wind_factor * self.wind_speed * np.sin(self.wind_dir) * i * self.t_step
            target_positions.append([x,y])
            
        target_positions = torch.tensor(target_positions, device=device).unsqueeze(0)
        
        # compute cost based on dist to target and relative bearing to target
        psi_v = states[:, :, 2]
        x_t, y_t = target_positions[:, :, 0], target_positions[:, :, 1] # position of target
        bearing = torch.atan2(y_t - y_v, x_t - x_v)
        del_bearing = bearing - psi_v
        del_bearing = (del_bearing + torch.pi) % (2 * torch.pi) - torch.pi
        dist = torch.sqrt((x_t - x_v)**2 + (y_t - y_v)**2)

        # reward to approach target with low bearing
        costs = 2 * torch.exp(-0.02 * dist**2 - 2 * del_bearing**2 )
        
        ## cost for interception
        r_int = 1.0
        theta_int = torch.pi/6
        
        int_mask = (dist < r_int) & (torch.abs(del_bearing) < theta_int)

        t_idxs = torch.arange(T, device=device).float()
        int_reward = 10 * (1- (t_idxs / T))
        int_reward = int_reward.unsqueeze(0).expand(K, -1)
    
        intercept_times = torch.where(int_mask, 
                                    t_idxs.unsqueeze(0).expand(K, -1), 
                                    torch.full((K, T), T, device=device))  # (K, T)
        first_intercept_time, _ = torch.min(intercept_times, dim=1)  # (K,)

        # Mask for trajectories that actually intercept (first_intercept_time < T)
        has_intercept = first_intercept_time < T  # (K,)

        # Create mask for first interception timestep only
        first_intercept_mask = (t_idxs.unsqueeze(0) == first_intercept_time.unsqueeze(1)) & has_intercept.unsqueeze(1)  # (K, T)

        # Apply interception reward only at first interception timestep
        interception_rewards = torch.where(first_intercept_mask, -int_reward, torch.zeros_like(int_reward))
        
        disc_cost = - costs - interception_rewards  # shape (K, T)
        weighted_cost += torch.sum(disc_cost, dim=1)  # shape (K,)       
        
        return weighted_cost

    def state_transition_track(self, state, action, t=None):
        
        u = action[:, 0]
        r = action[:, 1]

        # limit the change in heading
        psi = state[:, 2] + r * self.track_time_step
        psi = (psi + torch.pi) % (2  * torch.pi) - torch.pi
        x_new = state[:, 0] + u * torch.cos(psi) * self.track_time_step
        y_new = state[:, 1] + u * torch.sin(psi) * self.track_time_step

        # if out of bounds, stay in same position
        out_bounds = (x_new < self.origin[0]) | (x_new > self.origin[0] + self.grid_size[0]*self.resolution) | \
                        (y_new < self.origin[1]) | (y_new > self.origin[1] + self.grid_size[1]*self.resolution)
        x_new = torch.where(out_bounds, state[:, 0], x_new)
        y_new = torch.where(out_bounds, state[:, 1], y_new)
        
        new_state = torch.stack([x_new, y_new, psi], axis=-1)

        return new_state, action

    ### For mode switch and gazebo service to delet object
    def call_delete_object_service(self, pos):
        
        client = self.create_client(RemoveObject, 'delete_object')
        while not client.wait_for_service(0.1):
            self.get_logger().warn('waiting for delete object service')
        request = RemoveObject.Request()
        request.position.x = pos[0]
        request.position.y = pos[1]
        
        future = client.call_async(request)
        future.add_done_callback(partial(self.callback_delete_object, pos=pos))
        
    def callback_delete_object(self, future, pos):
        try:
            response = future.result()
            if response.success:
                self.intercepted += 1
                self.guidance_mode = 0  # switch back to exploration
                self.get_logger().info(f"Successfully deleted object at position: {pos}")
            else:
                self.get_logger().info(f"Failed to delete object at position: {pos}")
                self.guidance_mode = 0  # switch back to exploration
        
        except Exception as e:
            self.get_logger().info(f"Service call failed: {e}")
        
    def object_detect_callback(self, msg: ParamVec):
        
        if len(msg.params) == 0 or self.pos is None:
            self.guidance_mode = 0
            self.nearest_target_pos = None
            return

        self.nearest_target_pos = None
        near_dist = 100
        # register closest target to track
        for ob in msg.params:
            ob_pos = ob.value.double_array_value
            if ob.value.integer_value == 7 or ob.value.integer_value == 8:
                dist = np.sqrt((self.pos.x - ob_pos[0])**2 + (self.pos.y - ob_pos[1])**2)
                if dist < near_dist:
                    near_dist = dist
                    self.nearest_target_pos = ob_pos
                

        ## switching condition new - wait until switching
        if near_dist < switching_radius:
            if self.guidance_mode == 0 and self.wait_until_intercept < 4:
                self.wait_until_intercept += 1
            elif self.wait_until_intercept >=4:
                self.wait_until_intercept = 0
                self.guidance_mode = 1
        elif near_dist < switching_radius + 2.0 and self.guidance_mode == 1:
            pass
        else:
            self.guidance_mode = 0
            self.wait_until_intercept = 0
            
    def object_position_callback(self, msg):
        
        if len(msg.poses) == 0 or self.pos is None:
            return
        
        # delete object if within 2 meters
        for ob in msg.poses:
            pos = ob.position
            dist = np.sqrt((self.pos.x - pos.x)**2 + (self.pos.y - pos.y)**2)
            # relative bearing from usv bl to object pos            
            bearing = np.arctan2(pos.y - self.pos.y, pos.x - self.pos.x) - self.psi
            if dist < 2.0 and np.abs(bearing) < np.pi/6:
                self.call_delete_object_service([pos.x, pos.y])
                self.intercept_positions.append([pos.x, pos.y])
                return

    def plot_samples(self, trajs, costs=None):
        
        # given K trajs plot them
        # also plot occupancy grid
        K = trajs.shape[0]

        # append current position to the trajs at the start
        current_pos = torch.tensor([[self.pos.x, self.pos.y]], device=trajs.device)
        trajs = torch.cat((current_pos.unsqueeze(0).repeat(K,1,1), trajs), dim=1)

        fig, ax = plt.subplots(figsize=(11, 8))
        norm = mcolors.Normalize(vmin=0, vmax=1)
        im = ax.imshow(self.occupancy_grid, cmap='viridis', origin='lower', norm=norm)
        
        # If costs are provided, use them to color trajectories
        if costs is not None:
            # Convert costs to numpy if they're tensors
            costs_np = costs.cpu().numpy()
            costs_np = costs_np.flatten()[:K]
            # Normalize costs for colormap (lower cost = better = warmer colors)
            # Invert costs so that better trajectories (lower cost) get warmer colors
            normalized_costs = (costs_np - costs_np.min()) / (costs_np.max() - costs_np.min() + 1e-8)
            inverted_costs = 1.0 - normalized_costs  # Invert so lower cost = higher value
        
            # Create colormap for trajectories (plasma gives good contrast)
            traj_colormap = plt.cm.plasma
        
            for i in range(K):
                color = traj_colormap(-costs_np[i])
                plt.plot(trajs[i,:,0].cpu().numpy()-self.origin[0], 
                        trajs[i,:,1].cpu().numpy()-self.origin[1], 
                        color=color, linewidth=1.0, alpha=0.7)
        
            # Add colorbar for trajectory costs
            sm = plt.cm.ScalarMappable(cmap=traj_colormap, 
                                    norm=plt.Normalize(vmin=-costs_np.max(), vmax=-costs_np.min()))
            sm.set_array([])
            cbar_traj = plt.colorbar(sm, ax=ax, label='-Trajectory Cost',)
            cbar_traj.ax.tick_params(labelsize=12)
        
        else:
            # Default plotting with black lines
            for i in range(K):
                plt.plot(trajs[i,:,0].cpu().numpy()-self.origin[0], 
                        trajs[i,:,1].cpu().numpy()-self.origin[1], 
                        'black', linewidth=0.5, alpha=0.3)

        # colorbar for occupancy grid
        plt.colorbar(im, ax=ax, label='Occupancy Probability')
        save_dir = '/vrx_ws/src/informative_planning/images/mppi_test_feb25'
        os.makedirs(save_dir, exist_ok=True)

        if len(self.path_followed_0) > 0: 
            path_0 = np.array(self.path_followed_0) - np.array(self.origin)
            plt.scatter(path_0[:, 0], path_0[:, 1], color='black', s=10)

        if len(self.path_followed_1) > 0:
            path_1 = np.array(self.path_followed_1) - np.array(self.origin)
            plt.scatter(path_1[:, 0], path_1[:, 1], color='gray', s=10)

        if len(self.intercept_positions) > 0:
            intercept_pos = np.array(self.intercept_positions) - np.array(self.origin)
            plt.scatter(intercept_pos[:,0], intercept_pos[:,1], marker='x', color='red', s=100)

        # Add grid lines
        ax.grid(True, color='gray', linestyle='--', alpha=0.7)
        
        # Add title and labels
        ax.set_xlabel('X (m)', fontsize=16)
        ax.set_ylabel('Y (m)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(loc='upper right')
        
        # # Save the figure
        filename = os.path.join(save_dir, f'map_{self.counter:04d}.png')
        plt.savefig(filename, dpi=250, bbox_inches='tight')
        plt.tight_layout()
        # Set axis limits to show only 100x100 grid
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        plt.close()
        
    def get_time(self):
        return self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec*1e-9

def main(args=None):
    rclpy.init()
    node = MPPIInfPlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
