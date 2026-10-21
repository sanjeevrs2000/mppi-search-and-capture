#!usr/bin/env python3

import rclpy 
from rclpy.node import Node 
from geometry_msgs.msg import PoseArray, Point, Pose, Twist
from ros_gz_interfaces.msg import ParamVec
from nav_msgs.msg import OccupancyGrid, GridCells, Odometry
import numpy as np
import tf_transformations
from std_msgs.msg import Header, Float32, Float64, Int32
from usv_navigation.utils import pose_to_numpy, SyncSubscription
import cv2
from scipy.ndimage import shift as ndimage_shift
from informative_planning.utils import *
from informative_planning.params import *
import os
import onnxruntime as ort

"""
TODO - implement global planners for baselines
1) Frontier based exploration
2)  Coverage planner

TODO - switching logic for baseline
When to switch? - greedy or based on distance threshold? 
Also to consider - switch back to global planner if the target is too far away?
"""

class CoveragePlan(Node):

    def __init__(self):

        super().__init__("coverage_planner")

        self.origin = origin
        self.grid_size = grid_size
        self.resolution = resolution
        self.r_tol = r_tol
        self.x_tol = x_tol        
        self.wps = PoseArray()
        self.wp_idx = 0
        self.guidance_mode = 0 # 0 for exploration; 1 for tracking
        self.pos = None
        self.trigger = False
        self.desired_speed = u
        self.cmd_vel = [u, 0.0]

        # publishers and subscribers
        self.odom_subscriber = self.create_subscription(Odometry, '/wamv/sensors/position/ground_truth_odometry', self.odom_callback, 10)
        self.mode_subscriber = self.create_subscription(Int32, '/wamv/guidance_mode', self.mode_callback, 10)
        self.timer = self.create_timer(0.1, self.los_guidance)
        # init waypoints
        self.init_waypoints()

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
        
        if self.trigger == False:
            self.trigger = True
            self.start_time = self.get_time()

    def mode_callback(self, msg):
        self.guidance_mode = msg.data

    def velocity_command(self):
        return self.cmd_vel

    def init_waypoints(self):

        self.wps.poses = []
        width_between_sweeps = 25
        grid_width, grid_height = self.grid_size
        origin_x, origin_y = self.origin
        wp_dist = 10

        # sweep right then left: i iterates over columns (y), j over rows (x)
        for i in range(0, grid_width, int(width_between_sweeps / self.resolution)):
            y = origin_y + i * self.resolution
            if (i // int(width_between_sweeps / self.resolution)) % 2 == 0:
                # sweep bottom to top
                for j in range(0, grid_height, wp_dist):
                    x = origin_x + j * self.resolution
                    waypoint = Pose()
                    waypoint.position.x = x
                    waypoint.position.y = y
                    self.wps.poses.append(waypoint)
            else:
                # sweep top to bottom
                for j in range(grid_height, 0, -wp_dist):
                    x = origin_x + j * self.resolution
                    waypoint = Pose()
                    waypoint.position.x = x
                    waypoint.position.y = y
                    self.wps.poses.append(waypoint)

    ### LOS guidance
    def los_guidance(self):
        
        if not self.trigger:
            self.cmd_vel = [0.0, 0.0] # u, r
            return
                
        elif self.wp_idx >= len(self.wps.poses)-1:
            # publish 0
            self.cmd_vel = [0.0, 0.0] # u, r
            return

        if self.guidance_mode == 1: # tracking mode
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
        r = (psi_d - self.psi)/time_step
        self.cmd_vel = [self.desired_speed, r] ## change from psi to r

    def get_time(self):
        return self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec*1e-9

class FrontierPlan(Node):

    def __init__(self):

        super().__init__("frontier_planner")

        self.origin = origin
        self.grid_size = grid_size
        self.resolution = resolution
        self.r_tol = r_tol
        self.x_tol = x_tol     
        self.guidance_mode = 0 # 0 for exploration; 1 for tracking
        self.occupancy_grid = None
        self.visit_og = None
        self.pos = None
        self.trigger = False
        self.cmd_vel = [u, 0.0]
        self.desired_speed = u
        self.wp_set = False

        # publishers and subscribers
        self.odom_subscriber = self.create_subscription(Odometry, '/wamv/sensors/position/ground_truth_odometry', self.odom_callback, 10)
        self.og_subscriber = self.create_subscription(OccupancyGrid, '/gridmap', self.og_callback,10)
        self.visit_og_subscriber = self.create_subscription(OccupancyGrid, '/visit_score_map', self.visit_og_callback,10)
        self.mode_subscriber = self.create_subscription(Int32, '/wamv/guidance_mode', self.mode_callback, 10)
        self.timer = self.create_timer(0.1, self.replan)

        self.target_wp_publisher = self.create_publisher(Pose, '/planner/target_wp', 10)
        self.viz_counter = 0

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

        if self.trigger == False:
            self.trigger = True
            self.start_time = self.get_time()

    def mode_callback(self, msg):

        self.guidance_mode = msg.data

        # reset frontier waypoint if in tracking mode
        if self.guidance_mode == 1 and self.wp_set:
            self.wp_set = False

    def velocity_command(self):
        return self.cmd_vel

    def og_callback(self, msg:OccupancyGrid):
        
        if self.trigger == False:
            return
        
        w, h = msg.info.width, msg.info.height
        self.grid_size = [w, h]
        self.resolution = msg.info.resolution
        og = np.array(msg.data, dtype=np.int8).reshape(h,w)
        og = og/100.0
        og[og == -1/100] = 0.5
        self.occupancy_grid = og.copy()

    def visit_og_callback(self, msg:OccupancyGrid):

        if not self.trigger or self.occupancy_grid is None:
            return

        w, h = msg.info.width, msg.info.height
        visit_og = np.array(msg.data, dtype=np.float32).reshape(h,w)        
        visit_og = visit_og/100.0        
        self.visit_og = visit_og.copy()

    def replan(self):
        if self.guidance_mode == 0 and not self.wp_set:
            frontiers = self.get_frontiers()
            if not frontiers:
                return

            best = min(frontiers, key=self.compute_cost)
            x, y, psi, _ = best

            self.wp = Pose()
            self.wp.position.x = x
            self.wp.position.y = y
            self.wp_set = True

            # publish target waypoint for plotter
            self.target_wp_publisher.publish(self.wp)

        if self.wp_set:
            self.los_guidance()

    def compute_cost(self, frontier):

        x, y, psi, size = frontier
        dist_ = np.linalg.norm(np.array([x, y]) - np.array([self.pos.x, self.pos.y]))
        size_w = 0.5
        return dist_ - size_w * size

    ### LOS guidance
    def los_guidance(self):

        if self.guidance_mode == 1: # tracking mode
            return

        if not self.trigger or not self.wp_set:
            self.cmd_vel = [0.0, 0.0] # u, r
            return

        ## set waypoint based on selected frontier (x, y, psi)

        x, y = self.pos.x, self.pos.y
        x2, y2 = self.wp.position.x, self.wp.position.y
        path_vec = np.array([np.cos(self.wp.orientation.z), np.sin(self.wp.orientation.z)])
        
        # computing the cross track error
        proj = np.array([x2, y2]) - np.dot(np.array([x, y])-np.array([x2, y2]),path_vec) * path_vec
        cross_track = np.array([x, y]) - proj
        along_track = np.array([x2, y2]) - proj
        
        if np.linalg.norm(np.array([x, y]) - np.array([x2, y2])) < self.r_tol or np.linalg.norm(along_track) < self.x_tol:
            self.wp_set = False
            self.cmd_vel = [0.0, 0.0] # u, r
            return
                   
        los_point = proj + (l_d * path_vec)

        psi_d = np.arctan2(los_point[1]-y, los_point[0]-x)
        psi_d = ssa(psi_d)
        r = (psi_d - self.psi)/time_step
        self.cmd_vel = [self.desired_speed, r] ## change from psi to r

    def get_frontiers(self):

        if self.occupancy_grid is None or not self.trigger:
            return []

        og = self.occupancy_grid
        H, W = og.shape
        xo, yo = self.origin
        res = self.resolution

        # --- masks ---
        unknown_mask = (og == 0.5).astype(np.uint8)
        free_mask = (og != 0.5).astype(np.uint8)

        contours, _ = cv2.findContours(free_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return []

        kernel = np.ones((3, 3), dtype=np.uint8)
        unknown_dilated = cv2.dilate(unknown_mask, kernel, iterations=1)

        # filter contour points to keep only those adjacent to unknown
        frontier_pts = []
        for cnt in contours:
            for pt in cnt:
                c, r = pt[0]   # cv2: (col, row)
                if unknown_dilated[r, c]:
                    frontier_pts.append([c, r])

        if not frontier_pts:
            return []

        frontier_pts = np.array(frontier_pts)   # (N, 2) col, row

        # split into fixed-size segments along the contour order
        # contour points are already spatially ordered - just chunk them
        seg_size = 35   # pixels per frontier segment - only tunable
        segments = [frontier_pts[i:i+seg_size] for i in range(0, len(frontier_pts), seg_size)]

        min_size = 10
        frontiers = []
        usv_col = (self.pos.x - xo) / res
        usv_row = (self.pos.y - yo) / res

        for seg in segments:

            if len(seg) < min_size:
                continue

            # centroid in pixel space (col, row)
            c_col = np.mean(seg[:, 0])
            c_row = np.mean(seg[:, 1])

            # snap centroid to nearest unknown pixel
            unknown_px = np.argwhere(unknown_mask == 1)   # (N, 2) row, col
            if len(unknown_px) > 0:
                dists = np.linalg.norm(unknown_px - np.array([c_row, c_col]), axis=1)
                nearest = unknown_px[np.argmin(dists)]
                c_row, c_col = float(nearest[0]), float(nearest[1])

            x_goal = xo + c_col * res
            y_goal = yo + c_row * res

            # PCA: seg is (col, row), swap to (row, col) for consistency
            seg_rc = seg[:, ::-1].astype(np.float32)   # (N, 2) row, col
            pts_c = seg_rc - seg_rc.mean(axis=0)

            if len(seg) >= 2:
                _, _, Vt  = np.linalg.svd(pts_c, full_matrices=False)
                principal = Vt[0]                                    # (row_dir, col_dir)
                normal    = np.array([-principal[1], principal[0]])  # perpendicular (col, row)
                normal   /= np.linalg.norm(normal)
            else:
                normal = np.array([1.0, 0.0])

            # vote which side is unknown
            def in_bounds(r, c): return 0 <= int(r) < H and 0 <= int(c) < W

            pos_votes, neg_votes = 0, 0
            for step in [3, 5, 8, 12]:
                r_p = int(c_row + step*normal[1]);  c_p = int(c_col + step*normal[0])
                r_n = int(c_row - step*normal[1]);  c_n = int(c_col - step*normal[0])
                if in_bounds(r_p, c_p): pos_votes += int(unknown_mask[r_p, c_p])
                if in_bounds(r_n, c_n): neg_votes += int(unknown_mask[r_n, c_n])

            if neg_votes > pos_votes:
                normal = -normal

            if pos_votes == 0 and neg_votes == 0:
                fallback = np.array([c_col - usv_col, c_row - usv_row], dtype=float)
                if np.linalg.norm(fallback) > 0:
                    normal = fallback / np.linalg.norm(fallback)

            yaw = float(np.arctan2(normal[1], normal[0]))
            frontiers.append((x_goal, y_goal, yaw, len(seg)))

        return frontiers

    def get_best_frontier(self, frontiers):

        min_cost, frontier_ = np.inf, None

        for f in frontiers:
            cost = self.compute_cost(f)
            if cost < min_cost:
                min_cost = cost
                frontier_ = f

        return frontier_

    def get_time(self):
        return self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec*1e-9


class MyopicGreedy(Node):

    def __init__(self):
        
        super().__init__("myopic_greedy")
        ## myopic planner that selects next best waypoint
        ## utility - same as receding horizon/mppi?

        self.origin = origin
        self.grid_size = grid_size
        self.resolution = resolution
        self.r_tol = r_tol
        self.x_tol = x_tol     
        self.guidance_mode = 0 # 0 for exploration; 1 for tracking
        self.occupancy_grid = None
        self.visit_og = None
        self.pos = None
        self.trigger = False
        self.cmd_vel = [u, 0.0]
        self.desired_speed = u
        self.wp_set = False

        # single-step planning horizon: waypoint is T_wp seconds ahead
        self.T_wp = 5.0

        # wind state (updated by subscriber)
        self.wind_speed = 0.0
        self.wind_dir = 0.0
        self.wind_factor = 0.03
        self.wind_trigger = False

        # occupancy grid update parameters
        self.p_low = p_low
        self.p_high = p_high

        # ONNX Runtime session for target predictions
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

        # action set: K candidates = n_cand headings × 2 speeds
        heading_list = np.linspace(-0.75 * np.pi, 0.75 * np.pi, n_cand)
        speed_list = np.array([1.0, 1.5, 2.0])
        self.action_set = np.array([[s, h] for s in speed_list for h in heading_list])

        # publishers and subscribers
        self.odom_subscriber = self.create_subscription(Odometry, '/wamv/sensors/position/ground_truth_odometry', self.odom_callback, 10)
        self.og_subscriber = self.create_subscription(OccupancyGrid, '/gridmap', self.og_callback,10)
        self.visit_og_subscriber = self.create_subscription(OccupancyGrid, '/visit_score_map', self.visit_og_callback,10)
        self.mode_subscriber = self.create_subscription(Int32, '/wamv/guidance_mode', self.mode_callback, 10)
        wind_df = {'/vrx/debug/wind/direction': Float32, '/vrx/debug/wind/speed': Float32}
        self.wind_sub = SyncSubscription(self, wind_df, self.wind_callback, 10, 1)
        self.timer = self.create_timer(0.1, self.replan)

        self.target_wp_publisher = self.create_publisher(Pose, '/planner/target_wp', 10)

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

        if self.trigger == False:
            self.trigger = True
            self.start_time = self.get_time()

    def mode_callback(self, msg):

        self.guidance_mode = msg.data

        # reset frontier waypoint if in tracking mode
        if self.guidance_mode == 1 and self.wp_set:
            self.wp_set = False

    def velocity_command(self):
        return self.cmd_vel

    def og_callback(self, msg:OccupancyGrid):
        
        if self.trigger == False:
            return
        
        w, h = msg.info.width, msg.info.height
        self.grid_size = [w, h]
        self.resolution = msg.info.resolution
        og = np.array(msg.data, dtype=np.int8).reshape(h,w)
        og = og/100.0
        og[og == -1/100] = 0.5
        self.occupancy_grid = og.copy()

    def visit_og_callback(self, msg:OccupancyGrid):

        if not self.trigger or self.occupancy_grid is None:
            return

        w, h = msg.info.width, msg.info.height
        visit_og = np.array(msg.data, dtype=np.float32).reshape(h,w)
        visit_og = visit_og/100.0
        self.visit_og = visit_og.copy()

    def wind_callback(self, dir: Float32, speed: Float32):
        self.wind_speed = speed.data
        self.wind_dir = dir.data * np.pi / 180.0
        self.wind_factor = 0.02 + 0.0015 * self.wind_speed
        if not self.wind_trigger:
            self.wind_trigger = True

    def replan(self):
        if self.guidance_mode == 0 and not self.wp_set:
            result = self.next_best_waypoint()
            if result[0] is None:
                return
            x, y, psi = result

            self.wp = Pose()
            self.wp.position.x = x
            self.wp.position.y = y
            self.wp_set = True

            self.target_wp_publisher.publish(self.wp)

        if self.wp_set:
            self.los_guidance()

    def predict_grid(self, grid, t):
        """Run ONNX model to predict target grid at t seconds ahead (same as MPPIInfPlannerNode.predictions)."""
        vx = self.wind_speed * np.cos(self.wind_dir) if self.wind_trigger else 0.0
        vy = self.wind_speed * np.sin(self.wind_dir) if self.wind_trigger else 0.0

        # Pad to 100x100 to match model input size
        extended_grid = np.zeros((100, 100), dtype=np.float32)
        extended_grid[0:self.grid_size[1], 0:self.grid_size[0]] = grid

        input_grids = extended_grid[np.newaxis, :, :, np.newaxis]  # (1, 100, 100, 1)
        input_params = np.array([[vx, vy, t]], dtype=np.float32)    # (1, 3)

        inputs = {self.input_names[0]: input_grids, self.input_names[1]: input_params}
        pred = self.sess.run(None, inputs)[0]  # (1, 100, 100, 1)

        pred = pred.squeeze()  # (100, 100)
        pred[pred < 0.01] = 0.0

        # Crop back to actual grid size
        return pred[:self.grid_size[1], :self.grid_size[0]]

    def compute_waypoint_cost(self, x, y, psi, pred_grid, spd, dpsi):
        """Single-waypoint cost using the same coefficients/structure as MPPI compute_cost.
        a1=1.0 (explore), a2=0.45 (target), a3=0.05 (collision), a4=0.1 (path).
        """
        xo, yo = self.origin
        x_max = xo + self.grid_size[0] * self.resolution
        y_max = yo + self.grid_size[1] * self.resolution

        # collision cost: softplus barrier on boundary proximity (same as MPPI)
        d_min = min(x - xo, x_max - x, y - yo, y_max - y)
        d_safe = 1.0
        coll_cost = float(np.logaddexp(0.0, d_safe - d_min))  # softplus(d_safe - d_min)

        fov_ids = get_fov(x, y, psi)  # (N, 2) as [row, col]

        if len(fov_ids) == 0 or self.visit_og is None:
            explore_cost = 0.0
            target_cost = 0.0
        else:
            rows = np.clip(fov_ids[:, 0], 0, self.occupancy_grid.shape[0] - 1)
            cols = np.clip(fov_ids[:, 1], 0, self.occupancy_grid.shape[1] - 1)

            # explore cost: sum of visit_og in FoV, normalised by grid size (mirrors MPPI)
            explore_cost = -np.sum(self.visit_og[rows, cols]) / (self.grid_size[0] * self.grid_size[1])

            # target cost: pred_grid weighted by distance & bearing, same kernel as MPPI
            cx = xo + (cols + 0.5) * self.resolution
            cy = yo + (rows + 0.5) * self.resolution
            dx = cx - x
            dy = cy - y
            dist = np.sqrt(dx**2 + dy**2)
            rel_bearings = ssa(np.arctan2(dy, dx) - psi)
            pred_vals = pred_grid[rows, cols]
            track_ig = -pred_vals * np.exp(-0.04 * dist**2 - 4.0 * rel_bearings**2)
            target_cost = float(np.sum(track_ig))  # no grid normalisation, same as MPPI

        # path cost: penalise slow speed and high yaw rate (mirrors MPPI path_costs)
        u_min, u_max, r_max = 1.0, 2.0, 0.53
        u_norm = np.clip((spd - u_min) / (u_max - u_min), 0.0, 1.0)
        r_norm = abs(dpsi / self.T_wp) / r_max  # equivalent yaw rate normalised
        path_cost = float((1.0 - u_norm)**2 + r_norm**2)

        a1, a2, a3, a4 = 1.0, 0.45, 0.05, 0.1
        return a1 * explore_cost + a2 * target_cost + a3 * coll_cost + a4 * path_cost

    def next_best_waypoint(self):
        """Select the single best waypoint T_wp seconds ahead from all candidate actions."""
        if self.occupancy_grid is None or not self.trigger:
            return None, None, None

        x0, y0, psi0 = self.pos.x, self.pos.y, self.psi

        # predict binary occupancy at T_wp seconds for tracking cost
        binary_grid = (self.occupancy_grid > 0.5).astype(np.float32)
        pred_grid = self.predict_grid(binary_grid, self.T_wp)

        best_cost = np.inf
        best_x, best_y, best_psi = x0, y0, psi0

        for spd, dpsi in self.action_set:
            psi_wp = ssa(psi0 + dpsi)
            x_wp = x0 + spd * np.cos(psi_wp) * self.T_wp
            y_wp = y0 + spd * np.sin(psi_wp) * self.T_wp

            cost = self.compute_waypoint_cost(x_wp, y_wp, psi_wp, pred_grid, spd, dpsi)
            if cost < best_cost:
                best_cost = cost
                best_x, best_y, best_psi = x_wp, y_wp, psi_wp

        return best_x, best_y, best_psi

    ### LOS guidance
    def los_guidance(self):

        if self.guidance_mode == 1: # tracking mode
            return

        if not self.trigger or not self.wp_set:
            self.cmd_vel = [0.0, 0.0] # u, r
            return

        ## set waypoint based on selected frontier (x, y, psi)

        x, y = self.pos.x, self.pos.y
        x2, y2 = self.wp.position.x, self.wp.position.y
        path_vec = np.array([np.cos(self.wp.orientation.z), np.sin(self.wp.orientation.z)])
        
        # computing the cross track error
        proj = np.array([x2, y2]) - np.dot(np.array([x, y])-np.array([x2, y2]),path_vec) * path_vec
        cross_track = np.array([x, y]) - proj
        along_track = np.array([x2, y2]) - proj
        
        if np.linalg.norm(np.array([x, y]) - np.array([x2, y2])) < self.r_tol or np.linalg.norm(along_track) < self.x_tol:
            self.wp_set = False
            self.cmd_vel = [0.0, 0.0] # u, r
            return
                   
        los_point = proj + (l_d * path_vec)

        psi_d = np.arctan2(los_point[1]-y, los_point[0]-x)
        psi_d = ssa(psi_d)
        r = (psi_d - self.psi)/time_step
        self.cmd_vel = [self.desired_speed, r] ## change from psi to r

    def get_time(self):
        return self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec*1e-9