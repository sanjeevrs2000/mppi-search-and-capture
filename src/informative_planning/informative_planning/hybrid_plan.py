#!usr/bin/env python3

import torch
import rclpy 
from rclpy.node import Node 
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseArray, Point, Pose, Twist
from ros_gz_interfaces.msg import ParamVec
from nav_msgs.msg import OccupancyGrid, GridCells, Odometry
import numpy as np
import tf_transformations
from std_msgs.msg import Header, Float32, Float64, Int32, Float64MultiArray, MultiArrayDimension
from usv_navigation.utils import pose_to_numpy, SyncSubscription
from vrx_custom_msgs.srv import RemoveObject
from ros_gz_interfaces.msg import ParamVec
from informative_planning.utils import *
from informative_planning.params import *
from functools import partial
from informative_planning.global_planners import MPPIInfPlannerNode, RHPlannerNode
from informative_planning.baseline_planners import FrontierPlan, CoveragePlan, MyopicGreedy
from informative_planning.tracking_controllers import PurePursuit
import yaml

path = '/vrx_ws/src/informative_planning/informative_planning/mppi_config.yaml'
config = yaml.safe_load(open(path))

class PlannerNode(Node): 
 
    def __init__(self):
        
        super().__init__("hybrid_planner")
        
        self.mode_publisher = self.create_publisher(Int32, '/wamv/guidance_mode', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/wamv/cmd_vel', 10)
        self.ce_publisher = self.create_publisher(Float64, '/wamv/control_effort', 10)
        self.int_pos_publisher = self.create_publisher(Float64MultiArray, '/wamv/intercepted_positions', 10)
        self.target_pos_publisher = self.create_publisher(Pose, '/wamv/nearest_target', 10)

        self.odom_subscriber = self.create_subscription(Odometry, '/wamv/sensors/position/ground_truth_odometry', self.odom_callback, 10)
        self.og_subscriber = self.create_subscription(OccupancyGrid, 'gridmap', self.og_callback,10)
        self.object_det_subscriber = self.create_subscription(ParamVec, '/wamv/yolo/results', self.object_detect_callback, 10)
        self.object_pos_subscriber = self.create_subscription(PoseArray, '/buoy_positions', self.object_position_callback, 10)
        self.num_intercept_subscriber = self.create_subscription(Int32, 'wamv/num_interceptions', self.num_intercepted_callback, 10)

        wind_df = {'/vrx/debug/wind/direction': Float32, '/vrx/debug/wind/speed': Float32}
        self.wind_sub = SyncSubscription(self, wind_df, self.wind_callback,10,1)

        self.cmd_vel_timer = self.create_timer(0.1, self.cmd_vel_callback)
        self.planner_timer = self.create_timer(0.1, self.planner)

        self.occupancy_grid = None
        self.origin = origin
        self.pos = None
        self.trigger = False        
        self.wind_trigger = False
        self.nearest_target_pos = None
        self.p_low = p_low
        self.p_high = p_high
        self.alpha, self.beta = alpha, beta # dynamic occupancy grid update factor
        self.d, self.theta = d_max, theta  # field of view distance and angle
        self.desired_speed = u
        self.cmd_vel = [u, 0.0]
        self.l_low, self.l_high = to_logodds(p_low), to_logodds(p_high)
        
        self.guidance_mode = 0  # 0 - global planner , 1 - tracking

        self.u_min = config['mppi']['u_min'][0]
        self.u_max = config['mppi']['u_max'][0]
        self.r_min = config['mppi']['u_min'][1]
        self.r_max = config['mppi']['u_max'][1]

        self.entropy_list = []
        self.avg_det_list = []
        self.cmd_vel_history = np.empty((0, 2))
        self.intercepted_positions = []
        self.intercepted = 0
        self.counter = 1 # for plotting
        self.wait_until_intercept = 0 #
        self.init_planners()

    def init_planners(self):

        if planner == 'receding_horizon':
            self.global_planner = RHPlannerNode()

        elif planner == 'mppi':
            self.global_planner = MPPIInfPlannerNode()
        
        elif planner == 'coverage':
            self.global_planner = CoveragePlan()
        
        elif planner == 'frontier':
            self.global_planner = FrontierPlan()

        elif planner == 'myopic':
            self.global_planner = MyopicGreedy()

        self.tracking_planner = PurePursuit()

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

        if self.get_time() - self.start_time > mission_time:
            self.get_logger().info("Mission time over. Stopping the planner.")
            rclpy.shutdown()

        elif self.intercepted >= max_targets:
            self.get_logger().info(f"Successfully intercepted {max_targets} targets. Stopping the planner.")
            rclpy.shutdown()

    def num_intercepted_callback(self, msg):
        self.intercepted = msg.data

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

    def wind_callback(self,dir:Float32,speed:Float32):
        self.wind_speed = speed.data
        self.wind_dir = dir.data * np.pi / 180.0  # Convert degrees to radians
        self.wind_trigger = True

    def planner(self):
        if self.guidance_mode == 1:
            self.cmd_vel = \
                self.tracking_planner.velocity_command([self.pos.x, self.pos.y, self.psi],
                                                       self.nearest_target_pos)

        elif self.guidance_mode == 0:
            self.cmd_vel = self.global_planner.velocity_command()

    def cmd_vel_callback(self):
        twist_msg = Twist()
        twist_msg.linear.x = self.cmd_vel[0]
        twist_msg.angular.z = self.cmd_vel[1]

        if self.trigger:
            self.cmd_vel_history = np.append(self.cmd_vel_history, [self.cmd_vel], axis=0)

        # calculate control effort over the whole mission
        if len(self.cmd_vel_history) > 1:
            delta_u = self.cmd_vel_history[1:, 0] - self.cmd_vel_history[:-1, 0]
            delta_r = self.cmd_vel_history[1:, 1] - self.cmd_vel_history[:-1, 1]
            delta_u /= (self.u_max - self.u_min)
            delta_r /= (self.r_max - self.r_min)
            ce = np.mean(np.abs(delta_u) + np.abs(delta_r))
            self.ce_publisher.publish(Float64(data=ce))

        self.cmd_vel_publisher.publish(twist_msg)
        self.mode_publisher.publish(Int32(data=self.guidance_mode))

        # publish list of intercpted positions
        int_pos_msg = Float64MultiArray()
        int_pos_msg.layout.dim = [
            MultiArrayDimension(label='N', size=len(self.intercepted_positions), stride=2),
            MultiArrayDimension(label='xy', size=2, stride=2),
        ]
        int_pos_msg.layout.data_offset = 0
        int_pos_msg.data = []
        if len(self.intercepted_positions) > 0:
            int_pos_msg.data = np.array(self.intercepted_positions).flatten().tolist()
        self.int_pos_publisher.publish(int_pos_msg)

        # publish nearest target position
        if self.nearest_target_pos is not None:
            target_msg = Pose()
            target_msg.position.x = self.nearest_target_pos[0]
            target_msg.position.y = self.nearest_target_pos[1]
            self.target_pos_publisher.publish(target_msg)

        return

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
                self.get_logger().info(f"Successfully deleted object at position: {pos}")
            else:
                self.get_logger().info(f"Failed to delete object at position: {pos}")
        
        except Exception as e:
            self.get_logger().info(f"Service call failed: {e}")
        
    def object_detect_callback(self, msg: ParamVec):
        
        if len(msg.params) == 0 or self.pos is None:
            self.guidance_mode = 0
            return

        self.nearest_target_pos = None
        near_dist = d_max
        # register closest target to track
        for ob in msg.params:
            ob_pos = ob.value.double_array_value
            if ob.value.integer_value == 7 or ob.value.integer_value == 8:
                dist = np.sqrt((self.pos.x - ob_pos[0])**2 + (self.pos.y - ob_pos[1])**2)
                if dist < near_dist:
                    near_dist = dist
                    self.nearest_target_pos = ob_pos
 
        if near_dist < switching_radius:
            if self.guidance_mode == 0 and self.wait_until_intercept < 3:
                self.wait_until_intercept += 1
            elif self.wait_until_intercept >=3:
                self.wait_until_intercept = 0
                self.guidance_mode = 1  # switch to tracking
        elif near_dist < switching_radius + 2.0 and self.guidance_mode == 1:
            # to avoid frequent switching?
            pass
        else:
            self.guidance_mode = 0  # switch to exploration
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
                self.intercepted_positions.append([pos.x, pos.y])
                return

    def get_time(self):
        return self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec*1e-9

def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()  
    global_plan_node = node.global_planner
    tracking_plan_node = node.tracking_planner

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(global_plan_node)
    executor.add_node(tracking_plan_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info(" Keyboard interrupt, shutting down")
    finally:
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()
        global_plan_node.destroy_node()
        tracking_plan_node.destroy_node()

if __name__ == "__main__": 
    main()