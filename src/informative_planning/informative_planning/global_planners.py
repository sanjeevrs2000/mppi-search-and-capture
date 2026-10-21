#!usr/bin/env python3

import torch
import rclpy 
from rclpy.node import Node 
from geometry_msgs.msg import PoseArray, Point, Pose, Twist
from ros_gz_interfaces.msg import ParamVec
from nav_msgs.msg import OccupancyGrid, GridCells, Odometry
import numpy as np
import tf_transformations
from std_msgs.msg import Header, Float32, Float64, Int32, Float64MultiArray, MultiArrayDimension
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

config = yaml.safe_load(open('/vrx_ws/src/informative_planning/informative_planning/mppi_config.yaml'))

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
        
        self.cmd_vel_publisher = self.create_publisher(Twist,'wamv/cmd_vel',10)
        self.ce_publisher = self.create_publisher(Float64, 'wamv/control_effort', 10)
        self.traj_publisher = self.create_publisher(Float64MultiArray, 'mppi/trajs', 10)
        self.traj_costs_publisher = self.create_publisher(Float64MultiArray, 'mppi/costs', 10) 
        
        self.odom_subscriber = self.create_subscription(Odometry, '/wamv/sensors/position/ground_truth_odometry', self.odom_callback, 10)
        self.og_subscriber = self.create_subscription(OccupancyGrid, '/gridmap', self.og_callback,10)
        self.visit_og_subscriber = self.create_subscription(OccupancyGrid, '/visit_score_map', self.visit_og_callback,10)

        wind_df = {'/vrx/debug/wind/direction': Float32, '/vrx/debug/wind/speed': Float32}
        self.wind_sub = SyncSubscription(self, wind_df, self.wind_callback,10,1)

        self.origin = origin
        self.pos = None
        self.occupancy_grid = None
        self.visit_og = None
        self.trigger = False       
        self.wind_trigger = False
        self.p_low = p_low
        self.p_high = p_high
        self.alpha, self.beta = alpha, beta # dynamic occupancy grid update factor
        self.d, self.theta = d_max, theta  # field of view distance and angle
        self.wind_factor = 0.03
        self.desired_speed = u
        self.cmd_vel = [u, 0.0]  # initial command velocity
        self.l_low, self.l_high = to_logodds(p_low), to_logodds(p_high)
        self.decay_rate = decay_rate
        self.t_step = time_step #time_step for planner
        self.horizon = config['mppi']['horizon']
        self.T = self.horizon * self.t_step  # total prediction horizon
        self.r_min = config['mppi']['u_min'][1]
        self.r_max = config['mppi']['u_max'][1]
        self.u_min = config['mppi']['u_min'][0]
        self.u_max = config['mppi']['u_max'][0]
        self.gamma = config['mppi']['rollout_var_discount']
        self.gamma_seq = torch.tensor([self.gamma ** (t) for t in range(self.horizon)], device=torch.device(def_device))
        self.ipp_timer = self.create_timer(config['mppi']['control_step'], self.informative_global_planner)
        
        # MPPI init
        self.mppi_planner = MPPIPlanner(cfg=config['mppi'], nx=3, 
                                        dynamics=self.state_transition,running_cost=self.compute_cost,
                                        )
        self.prev_action = self.cmd_vel.copy()

        # ONNX Runtime session init
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

        self.guidance_mode = 0  # default to exploration mode
        self.intercepted = 0
        self.counter = 0
        self.path_followed_0 = []
        self.path_followed_1 = []
        self.intercept_positions = []
        self.cmd_vel_history = np.empty((0,2))

    def velocity_command(self):
        
        return self.cmd_vel

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

    def informative_global_planner(self):

        if self.trigger == False or self.occupancy_grid is None or self.visit_og is None:
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

        t2 = self.get_time()
        self.get_logger().info(f"Time taken for informative global planning step: {t2 - t1} s")

        if self.guidance_mode == 0:
            self.cmd_vel = [u, r]
        
        return None

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
                fov_batch = fov_all[:, t]                                          
            else:
                fov_batch = batch_get_fov(                                         
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

        x_world = x_world_grid.to(device)  # (H, W)
        y_world = y_world_grid.to(device)

        pred_unc = self.pred_unc_grids.to(device)  # (T, H, W)

        target_cost_sum = torch.zeros(K, device=device)
        for t in range(T):
            if fov_all is not None:
                fov_t = fov_all[:, t].float()
            else:
                fov_t = batch_get_fov(                                     
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
            if cost_ablation == 'tracking_coefficients':
                track_ig = -visible_cells_t
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

        # Dev machine: FoV computed per-timestep inside each cost fn (no 86 MB alloc).
        # Jetson: uncomment below and pass fov_all to both cost fns for faster vectorised path.
        # fov_all = batch_get_fov_kt(
        #     states[:, :, 0], states[:, :, 1], states[:, :, 2],
        #     device=states.device
        # )  # (K, T, H, W)

        # information gain - entropy + tracking ig
        explore_cost = self.explore_costs(states, actions)           # pass fov_all=fov_all for Jetson
        track_cost = self.target_costs(states, actions)              # pass fov_all=fov_all for Jetson
        path_cost = self.path_costs(states, actions)
        coll_cost = self.collision_costs(states)

        c_e, c_t, c_c, c_p = a1, a2, a3, a4

        if cost_ablation == 'exploration':
            c_e = 0.0
        elif cost_ablation == 'tracking':
            c_t = 0.0
        elif cost_ablation == 'path':
            c_p = 0.0
        elif cost_ablation == 'exploration+path' or cost_ablation == 'path+exploration':
            c_e = 0.0
            c_p = 0.0

        c_e *= scale_a1
        c_t *= scale_a2
        c_c *= scale_a3
        c_p *= scale_a4

        # cost = a1 * explore_cost + a2 * track_cost + a3 * coll_cost + a4 * path_cost
        cost = c_e * explore_cost + c_t * track_cost + c_c * coll_cost + c_p * path_cost
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

    def get_time(self):
        return self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec*1e-9

class RHPlannerNode(Node):

    def __init__(self):
        
        super().__init__("rh_planner")
        
        self.origin = origin
        self.r_tol = r_tol
        self.x_tol = x_tol
        self.pos = None
        self.trigger = False        
        self.wind_trigger = False
        self.occupancy_grid = None
        self.p_low = p_low
        self.p_high = p_high
        self.alpha, self.beta = alpha, beta # dynamic occupancy grid update factor
        self.d, self.theta = d_max, theta  # field of view distance and angle
        self.wind_factor = 0.03
        self.desired_speed = u
        self.cmd_vel = [u, 0.0]  # initial command velocity
        self.l_low, self.l_high = to_logodds(p_low), to_logodds(p_high)
        self.t_step = time_step
        self.decay_rate = decay_rate
        self.horizon = config['mppi']['horizon']
        self.gamma = config['mppi']['rollout_var_discount']
        self.T = self.t_step * self.horizon
        self.guidance_mode = 0 # 0 - global planning, 1 - tracking
        self.wps = PoseArray()
        self.wp_idx = 0
        self.t_delay = 0
        self.counter = 0
        self.path_followed = []

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
        # publishers and subscribers
        self.odom_subscriber = self.create_subscription(Odometry, '/wamv/sensors/position/ground_truth_odometry', self.odom_callback, 10)
        self.og_subscriber = self.create_subscription(OccupancyGrid, '/gridmap', self.og_callback, 10)
        self.mode_subscriber = self.create_subscription(Int32, '/wamv/guidance_mode', self.mode_callback, 10)
        wind_df = {'/vrx/debug/wind/direction': Float32, '/vrx/debug/wind/speed': Float32}
        self.wind_sub = SyncSubscription(self, wind_df, self.wind_callback,10,1)

        self.traj_publisher = self.create_publisher(Float64MultiArray, 'mppi/trajs', 10)
        self.traj_costs_publisher = self.create_publisher(Float64MultiArray, 'mppi/costs', 10)
        self.timer = self.create_timer(0.1, self.replan)

        self.yaw_rates = torch.linspace(-0.26, 0.26, 11)
        self.speed_list = torch.linspace(1.0, 2.0, 3)
        self.action_set = torch.cartesian_prod(self.speed_list, self.yaw_rates).to(def_device)
        self.visit_og = None
        self.visit_og_subscriber = self.create_subscription(OccupancyGrid, '/visit_score_map', self.visit_og_callback, 10)
        self.gamma_seq = torch.tensor([self.gamma ** t for t in range(self.horizon)], device=torch.device(def_device))
        self.u_min = float(self.speed_list.min())
        self.u_max = float(self.speed_list.max())
        self.r_min = float(self.yaw_rates.min())
        self.r_max = float(self.yaw_rates.max())
        self.prev_action = self.cmd_vel.copy()

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
        
        self.path_followed.append([self.pos.x, self.pos.y])
        
        if self.trigger == False:
            self.trigger = True
            self.start_time = self.get_time()

        if self.get_time() - self.start_time > mission_time:
            self.get_logger().info("Mission time over. Stopping the planner.")
            rclpy.shutdown()
                 
    def wind_callback(self,dir:Float32,speed:Float32):
        self.wind_speed = speed.data
        self.wind_dir = dir.data * np.pi / 180.0  # Convert degrees to radians
        self.wind_factor = 0.02 + 0.0015 * self.wind_speed
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

    def visit_og_callback(self, msg: OccupancyGrid):

        if self.occupancy_grid is None:
            return

        visit_og = np.array(msg.data, dtype=np.int8).reshape(self.grid_size[1], self.grid_size[0])
        visit_og = visit_og / 100.0
        self.visit_og = visit_og.copy()

    def mode_callback(self, msg:Int32):
        
        if self.guidance_mode == 0 and msg.data == 1:
            self.wps = PoseArray()
            self.wp_idx = 0

        self.guidance_mode = msg.data

    def velocity_command(self):

        return self.cmd_vel
    
    def replan(self):

        if self.guidance_mode != 0 or self.occupancy_grid is None or self.visit_og is None:
            return

        t1 = self.get_time()

        # spatiotemporal target predictions
        binary_grid = self.occupancy_grid.copy()
        ids = (binary_grid > 0.5)
        binary_grid[~ids] = 0
        binary_grid[ids] = 1
        self.pred_unc_grids = self.predictions(binary_grid)

        states = self.get_trajectories(self.action_set, torch.device(def_device))

        # expand constant (u, r) per-trajectory to K x T x 2 for cost functions
        K = self.action_set.shape[0]
        n_points = states.shape[1]
        actions_expanded = self.action_set.unsqueeze(1).expand(K, n_points, -1)

        cost = self.compute_cost(states, actions_expanded)

        best_idx = torch.argmin(cost).item()
        best_action = self.action_set[best_idx]

        u, r = best_action.cpu().numpy()
        self.cmd_vel = [float(u), float(r)]
        self.prev_action = [float(u), float(r)]

        self.publish_traj(states[:, :, :2], cost)

        t2 = self.get_time()
        self.get_logger().info(f"Time taken for planning: {t2-t1} seconds")

        return None

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

    def get_trajectories(self, action, device):

        # get trajectories for receding horizon planner
        # action set: K x 2 (speed, yaw_rate) — same format as nominal_trajectories in MPPIInfPlannerNode
        K = action.shape[0]
        n_points = int(self.T / self.t_step)

        u = action[:, 0].to(device)  # (K,)
        r = action[:, 1].to(device)  # (K,)

        states = torch.zeros((K, n_points, 3), device=device)

        x = torch.full((K,), self.pos.x, device=device)
        y = torch.full((K,), self.pos.y, device=device)
        psi = torch.full((K,), self.psi, device=device)

        for t in range(n_points):
            psi = psi + r * self.t_step
            psi = (psi + torch.pi) % (2 * torch.pi) - torch.pi
            x = x + u * torch.cos(psi) * self.t_step
            y = y + u * torch.sin(psi) * self.t_step
            states[:, t, 0] = x
            states[:, t, 1] = y
            states[:, t, 2] = psi

        return states

    def explore_costs(self, states, actions, fov_all=None):

        K = states.shape[0]
        T = states.shape[1]
        device = states.device

        visit_og = torch.tensor(self.visit_og, device=device).float()
        visit_og_batch = visit_og.unsqueeze(0).expand(K, -1, -1).clone()

        explore_cost = torch.zeros((K, T), device=device)

        for t in range(T):
            if fov_all is not None:
                fov_batch = fov_all[:, t]
            else:
                fov_batch = batch_get_fov(
                    states[:, t, 0], states[:, t, 1], states[:, t, 2], device=device
                )

            explore_cost[:, t] = torch.sum(visit_og_batch * fov_batch, dim=(-1, -2))

            visit_og_batch = visit_og_batch + self.decay_rate * self.t_step
            visit_og_batch = torch.where(fov_batch, torch.zeros_like(visit_og_batch), visit_og_batch)
            visit_og_batch = torch.clamp(visit_og_batch, 0, 1)

        explore_cost /= (self.grid_size[0] * self.grid_size[1])
        explore_cost_sum = torch.sum(self.gamma_seq[:T] * explore_cost, dim=(-1))

        return -explore_cost_sum

    def target_costs(self, states, actions, fov_all=None):

        K = states.shape[0]
        T = states.shape[1]
        device = states.device

        x_world = x_world_grid.to(device)  # (H, W)
        y_world = y_world_grid.to(device)

        pred_unc = self.pred_unc_grids.to(device)  # (T, H, W)

        target_cost_sum = torch.zeros(K, device=device)
        for t in range(T):
            if fov_all is not None:
                fov_t = fov_all[:, t].float()
            else:
                fov_t = batch_get_fov(
                    states[:, t, 0], states[:, t, 1], states[:, t, 2], device=device
                ).float()

            visible_cells_t = pred_unc[t].unsqueeze(0) * fov_t  # (K, H, W)

            center_x = states[:, t, 0].view(K, 1, 1)
            center_y = states[:, t, 1].view(K, 1, 1)
            psi      = states[:, t, 2].view(K, 1, 1)

            dx = x_world - center_x
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

        return target_cost_sum

    def path_costs(self, states, actions):

        u = actions[:, :, 0]
        r = actions[:, :, 1]

        # promote higher speed and limit turns
        u_range = self.u_max - self.u_min
        u_norm = (u - self.u_min) / u_range if u_range > 0 else torch.ones_like(u)
        r_norm = torch.abs(r) / self.r_max

        path_cost = (1 - u_norm)**2 + r_norm**2
        path_cost = torch.sum(self.gamma_seq[:path_cost.shape[1]] * path_cost, dim=-1)

        return path_cost

    def acceleration_costs(self, states, actions):

        u = actions[:, :, 0]
        r = actions[:, :, 1]

        u_ = torch.roll(u, shifts=1, dims=1)
        u_[:, 0] = self.prev_action[0]
        r_ = torch.roll(r, shifts=1, dims=1)
        r_[:, 0] = self.prev_action[1]

        u_range = self.u_max - self.u_min
        delta_u = (u - u_) / u_range if u_range > 0 else torch.zeros_like(u)
        delta_r = (r - r_) / (self.r_max - self.r_min)

        action_cost = delta_u**2 + delta_r**2
        action_cost = torch.sum(self.gamma_seq[:action_cost.shape[1]] * action_cost, dim=-1)

        return action_cost

    def collision_costs(self, states):

        # states are K x T x nx — softplus boundary barrier (same as MPPIInfPlannerNode)
        K = states.shape[0]
        device = states.device

        xt = states[:, :, 0]
        yt = states[:, :, 1]

        x_min, y_min = self.origin[0], self.origin[1]
        x_max = self.origin[0] + self.grid_size[0] * self.resolution
        y_max = self.origin[1] + self.grid_size[1] * self.resolution

        d_left, d_right = xt - x_min, x_max - xt
        d_top, d_bottom = yt - y_min, y_max - yt

        d_min = torch.minimum(torch.minimum(d_left, d_right), torch.minimum(d_top, d_bottom))
        d_safe = 1.0
        coll_cost = torch.nn.functional.softplus(d_safe - d_min)
        coll_cost = torch.sum(self.gamma_seq[:coll_cost.shape[1]] * coll_cost, dim=-1)

        return coll_cost

    @torch.no_grad()
    def compute_cost(self, states, actions):

        explore_cost = self.explore_costs(states, actions)
        track_cost = self.target_costs(states, actions)
        path_cost = self.path_costs(states, actions)
        coll_cost = self.collision_costs(states)

        # cost = a1 * explore_cost + a2 * track_cost + a3 * coll_cost + a4 * path_cost
        cost = a1 * explore_cost + a2 * track_cost + a3 * coll_cost + a4 * path_cost
        self.get_logger().info(
            f'explore: [{torch.mean(explore_cost).item():.3f}, {torch.min(explore_cost).item():.3f}, {torch.max(explore_cost).item():.3f}]  '
            f'track: [{torch.mean(track_cost).item():.3f}, {torch.min(track_cost).item():.3f}, {torch.max(track_cost).item():.3f}]  '
            f'coll: [{torch.mean(coll_cost).item():.3f}, {torch.min(coll_cost).item():.3f}, {torch.max(coll_cost).item():.3f}]'
        )

        return cost
        
    ## get step costs and save the map prediction
    def simulate_og_map(self, states, visualize_=False):
            
        # states: K x T x nx
        K = states.shape[0]
        n_points = states.shape[1]
        device = states.device

        batch_trajs = states[:,:,:2]

        self.counter += 1

        og = torch.tensor(self.occupancy_grid, device=device, dtype=torch.float32)
        og_batch = og.unsqueeze(0).repeat(K, 1, 1)
        ig = torch.zeros((K,), device=device)
        
        for i in range(1, n_points-1):

            x1 = batch_trajs[:, i-1, 0]
            y1 = batch_trajs[:, i-1, 1]
            x2 = batch_trajs[:, i, 0]
            y2 = batch_trajs[:, i, 1]

            psi_d = torch.atan2(y2 - y1, x2 - x1)

            fov_batch = batch_get_fov(x2, y2, psi_d, device)
            
            # simulate og update in batch
            og_batch = self.simulate_map_update(x2, y2, psi_d, fov_batch, og_batch, i, device)

            # compute tracking information gain for each trajectory
            ig_step = self.compute_track_ig(x2, y2, psi_d, fov_batch, i)
            ig += ig_step

        # compute mutual information from occupancy grid
        entropy_maps = - og_batch * torch.log(og_batch) - (1 - og_batch) * torch.log(1 - og_batch)
        pred_mean_entropy = torch.sum(entropy_maps, dim=(-2, -1))/(self.grid_size[0] * self.grid_size[1])
        
        og_ = torch.tensor(self.occupancy_grid, device=device, dtype=torch.float32)
        entropy_map = -og_ * torch.log(og_) - (1 - og_) * torch.log(1 - og_)
        mean_entropy_now = torch.sum(entropy_map)/(self.grid_size[0] * self.grid_size[1])

        diff_entr = mean_entropy_now - pred_mean_entropy

        ig /= (n_points)

        # plot one of the traj
        if visualize_:
            for i in range(K): 
                path = batch_trajs[i].cpu().numpy()
                og = og_batch[i].cpu().numpy()
                # Visualize original and simulated occupancy grids
                plt.figure(figsize=(12, 6))
                norm = mcolors.Normalize(vmin=0, vmax=1)
                # Plot original occupancy grid
                plt.subplot(1, 2, 1)
                plt.imshow(self.occupancy_grid, origin='lower', norm=norm)
                path_array = np.array(self.path_followed) - np.array(self.origin)
                plt.colorbar(label='Probability')
                plt.title('Occupancy Grid')
                plt.axis('off')

                # Plot simulated occupancy grid
                plt.subplot(1, 2, 2)
                plt.imshow(og, origin='lower', norm=norm)
                plt.colorbar(label='Probability')
                plt.title('Simulated Occupancy Grid')
                cand_path = np.array(path) - np.array(self.origin)
                plt.plot(cand_path[:, 0], cand_path[:, 1], 'red', linewidth=2, label='Path')
                # Convert self.path_followed to numpy array
                plt.plot(path_array[:, 0], path_array[:, 1], 'black', linewidth=2, label='Path Followed')
                plt.axis('off')
                plt.show()
                plt.close()                
        
        return diff_entr, ig

    def simulate_map_update(self, x, y, psi, fov_mask, og_maps, k, device):
        
        K = x.shape[0]
        
        # Apply prediction updates if available
        if k > 0:
            grid_prev = torch.tensor(self.pred_grids[k-1], device=device, dtype=torch.float32)
            grid_curr = torch.tensor(self.pred_grids[k], device=device, dtype=torch.float32)
            diff_mask = (grid_curr - grid_prev) != 0

            grid_curr_expanded = grid_curr.unsqueeze(0).expand(K, -1, -1)
            update_vals = self.alpha * grid_curr_expanded + self.beta * og_maps
            # Expand diff_mask to match og_maps dimensions
            diff_mask_expanded = diff_mask.unsqueeze(0).expand(K, -1, -1)
            og_maps = torch.where(diff_mask_expanded, update_vals, og_maps)

        # # simulate occupancy grid map step

        og_maps = torch.clamp(og_maps, self.p_low, self.p_high)
        log_odds_grid = torch.log(og_maps / (1 - og_maps))

        i_coords = torch.arange(self.grid_size[0], device=device).float()
        j_coords = torch.arange(self.grid_size[1], device=device).float()

        i_grid, j_grid = torch.meshgrid(i_coords, j_coords)
        
        x_world = (j_grid + 0.5) * self.resolution + self.origin[0]
        y_world = (i_grid + 0.5) * self.resolution + self.origin[1]

        x_world = x_world.unsqueeze(0).expand(K, -1, -1)
        y_world = y_world.unsqueeze(0).expand(K, -1, -1)
        
        dx = x_world - x.view(K, 1, 1)
        dy = y_world - y.view(K, 1, 1)
        distances = torch.sqrt(dx**2 + dy**2)
        
        # apply fov mask to update only cells within fov
        occ_mask = fov_mask & (og_maps > 0.5)
        free_mask = fov_mask & (og_maps <= 0.5)
        
        # apply updates with predictions from inverse sensor model
        if occ_mask.any():
            occ_distances = torch.where(occ_mask, distances, torch.zeros_like(distances))
            p = batch_sensor_model(occ_distances)
            log_odds_grid += torch.where(occ_mask, log_odds_grid + torch.log(p/(1-p)), torch.zeros_like(log_odds_grid))

        if free_mask.any():
            free_distances = torch.where(free_mask, distances, torch.zeros_like(distances))
            p0 = batch_negative_sensor_model(free_distances)
            log_odds_grid += torch.where(free_mask, log_odds_grid + torch.log(p0/(1-p0)), torch.zeros_like(log_odds_grid))

        # convert log odds to probability
        log_odds_grid = torch.clamp(log_odds_grid, self.l_low, self.l_high)
        og_maps = torch.sigmoid(log_odds_grid)

        return og_maps

    def pred_map(self, grid, t):
       
        grid_snapshots = []

        Rx, Ry = 0,0
        for i in np.arange(self.t_step,t+self.t_delay,self.t_step):

            dx = self.wind_factor * self.wind_speed  * np.cos(self.wind_dir) * (self.t_step) / self.resolution
            dy = self.wind_factor * self.wind_speed * np.sin(self.wind_dir) * (self.t_step) / self.resolution
            Rx += dx
            Ry += dy
            shift_x = np.round(Rx)
            shift_y = np.round(Ry)
            Rx -= shift_x
            Ry -= shift_y

            grid = shift(grid, shift=(shift_y, shift_x), order=1, mode='constant', cval=self.p_low)
            grid_snapshots.append(grid.copy())
                
        return grid_snapshots[self.t_delay//self.t_step:]

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

        # keep on CPU; moved to GPU in compute_track_ig
        predictions = torch.tensor(predictions.squeeze(-1), dtype=torch.float32)

        # Crop back to actual grid size (model input was padded to 100x100)
        predictions = predictions[:, :self.grid_size[1], :self.grid_size[0]]

        return predictions

    def compute_track_ig(self, x, y, psi, fov_mask, i):
        
        K = fov_mask.shape[0]
        device = fov_mask.device
        
        # Get prediction uncertainty grid for time step i
        pred_unc_batch = self.pred_unc_grids[i].to(device).unsqueeze(0).expand(K, -1, -1)  # (K, H, W)
        visible_cells = pred_unc_batch * fov_mask.float()  # (K, H, W)
        
        # Create coordinate grids for distance calculation
        i_coords = torch.arange(self.grid_size[0], device=device).float()
        j_coords = torch.arange(self.grid_size[1], device=device).float()
        j_grid, i_grid = torch.meshgrid(j_coords, i_coords, indexing='ij')  # (H, W)
        
        # Convert to world coordinates
        x_world = (i_grid + 0.5) * self.resolution + self.origin[0]  # (H, W)
        y_world = (j_grid + 0.5) * self.resolution + self.origin[1]  # (H, W)
        
        # Expand for all trajectories
        x_world_batch = x_world.unsqueeze(0).expand(K, -1, -1)  # (K, H, W)
        y_world_batch = y_world.unsqueeze(0).expand(K, -1, -1)  # (K, H, W)
        
        center_x = x.view(K, 1, 1)
        center_y = y.view(K, 1, 1)
        dx = x_world_batch - center_x
        dy = y_world_batch - center_y
        distances = torch.sqrt(dx**2 + dy**2)  # (K, H, W)
        
        angles = torch.atan2(dy, dx)  # (K, H, W)
        rel_bearings = torch.abs(ssa(angles - psi.view(K, 1, 1)))  # (K, H, W)

        # Apply FOV mask to distances
        fov_distances = distances * fov_mask.float()  # (K, H, W)
        
        # Valid mask: probability between 0.01 and 1, and within FOV
        valid_mask = (visible_cells > 0.01) & fov_mask
        
        ig = visible_cells * torch.exp(-0.01 * fov_distances**2)
        ig_values = torch.where(valid_mask, ig, torch.zeros_like(visible_cells))
        
        total_ig = torch.sum(ig_values, dim=(-2, -1))  # (K,)

        return total_ig

    ### LOS guidance
    def los_guidance(self):
        
        if not self.trigger:
            self.cmd_vel = [0.0, 0.0] # u, r
            return
                
        if self.wp_idx >= len(self.wps.poses)-1:
            # publish 0
            self.cmd_vel = [0.0, 0.0] # u, r
            return

        x, y = self.pos.x, self.pos.y                                
        x1, y1 = self.wps.poses[self.wp_idx].position.x, self.wps.poses[self.wp_idx].position.y
        x2, y2 = self.wps.poses[self.wp_idx+1].position.x, self.wps.poses[self.wp_idx+1].position.y
        path_vec = np.array([x2 - x1, y2 - y1])/np.linalg.norm(np.array([x2 - x1, y2 - y1]))
        
        # computing the cross track error
        proj = np.array([x1, y1]) + np.dot(np.array([x, y])-np.array([x1, y1]),path_vec) * path_vec
        cross_track = np.array([x, y]) - proj
        along_track = np.array([x2, y2]) - proj
        
        if np.linalg.norm(np.array([x, y]) - np.array([x2, y2])) < self.r_tol or np.linalg.norm(along_track) < self.x_tol:
                
            if self.wp_idx < len(self.wps.poses)-2:
                self.wp_idx += 1
            else:
                self.wp_idx = len(self.wps.poses)-1
                # publish 0
                self.cmd_vel = [0.0, 0.0] # u, r
                return
                   
        los_point = proj + (l_d * path_vec)

        psi_d = np.arctan2(los_point[1]-y, los_point[0]-x)
        psi_d = ssa(psi_d)
        r = psi_d * self.desired_speed / l_d  # fix this - make it consistent with velocity controller
        self.cmd_vel = [self.desired_speed, r]

    def get_time(self):
        return self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec*1e-9
