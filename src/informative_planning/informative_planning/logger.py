import rclpy 
from rclpy.node import Node 
from geometry_msgs.msg import PoseArray, Point, Pose
from ros_gz_interfaces.msg import ParamVec
from nav_msgs.msg import OccupancyGrid, GridCells, Odometry
import numpy as np
import tf_transformations
from std_msgs.msg import Header, Float32, Float64, Int32    
from geometry_msgs.msg import Twist
from usv_navigation.utils import pose_to_numpy, SyncSubscription
from informative_planning.utils import to_logodds, to_prob, sensor_model, negative_sensor_model, calc_4points_bezier_path, world_to_grid, BoatVisualization, compute_entropy
from informative_planning.params import save_name, mission_time, origin, grid_size, theta, d_max
import os
import csv

run_name = save_name  # Set your run name here
 
class LoggerNode(Node):
    def __init__(self):
        super().__init__('logger_node')
        self.get_logger().info('Logger node started')

        self.create_subscription(Odometry, '/wamv/sensors/position/ground_truth_odometry', self.odom_callback, 10)
        self.create_subscription(Float32, "/avg_detections", self.avg_detections_callback, 10)
        self.create_subscription(Int32, 'wamv/guidance_mode', self.mode_callback, 10)
        self.create_subscription(Float64, 'wamv/control_effort', self.ce_callback, 10)
        self.create_subscription(Int32, 'wamv/num_interceptions', self.num_intercepted_callback, 10)
        self.create_subscription(OccupancyGrid, 'gridmap', self.og_callback, 10)
        self.create_subscription(OccupancyGrid, '/visit_score_map', self.visit_og_callback, 10)

        self.log_timer = self.create_timer(10.0, self.log_callback)
        self.get_logger().info('Logger node initialized')
    
        self.trigger = False
        self.start_time = None
        self.guidance_mode = None
        self.avg_det_list = []
        self.path_followed = []
        self.ce_list = []
        self.num_int_list = []
        self.entropy_list = []
        self.visit_score_list = []

        self.occupancy_grid = None
        self.visit_og = None
        self.control_effort = -1

    def mode_callback(self, msg:Int32):
        self.guidance_mode = msg.data

    def ce_callback(self, msg:Float64):
        self.control_effort = msg.data
    
    def num_intercepted_callback(self, msg:Int32):
        self.num_int = msg.data

    def log_callback(self):
            
        if self.guidance_mode is None or self.trigger == False:
            return

        if self.get_time() - self.start_time < 10:
            return

        # calculate control effort, number of objects collected
        self.ce_list.append(self.control_effort)
        self.avg_det_list.append(self.avg_det)
        self.num_int_list.append(self.num_int)

        # log entropy (1 - mean Bernoulli entropy) of occupancy grid
        if self.occupancy_grid is not None:
            compute_entropy_vec = np.vectorize(compute_entropy)
            h = compute_entropy_vec(self.occupancy_grid)
            mean_entropy = np.mean(h)
            self.entropy_list.append(mean_entropy)

        # log mean visit score
        if self.visit_og is not None:
            self.visit_score_list.append(float(np.mean(self.visit_og)))

        # todo - save values at frequent intervals

        if self.get_time() - self.start_time > mission_time:
            
            self.save_data()
            self.get_logger().info('Data saved successfully')
            self.destroy_node()

        return

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
            init_pos = Pose()
            init_pos.position.x = self.pos.x
            init_pos.position.y = self.pos.y
            init_pos.orientation = orientation_q
            self.trigger = True
            self.start_time = self.get_time()
            self.avg_det_list.append(0)
            
    def og_callback(self, msg: OccupancyGrid):
        
        if self.trigger == False:
            return
        
        w, h = msg.info.width, msg.info.height
        self.grid_size = [w, h]
        self.resolution = msg.info.resolution
        og = np.array(msg.data, dtype=np.int8).reshape(h, w)
        og = og / 100.0
        og[og == -1/100] = 0.5
        self.occupancy_grid = og.copy()

    def visit_og_callback(self, msg: OccupancyGrid):

        if self.occupancy_grid is None:
            return

        visit_og = np.array(msg.data, dtype=np.int8).reshape(self.grid_size[1], self.grid_size[0])
        visit_og = visit_og / 100.0
        self.visit_og = visit_og.copy()
        
    def get_time(self):
        return self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec*1e-9

    def avg_detections_callback(self, msg:Float32):
        self.avg_det = msg.data

    def save_data(self):
        
        # Create directory if it doesn't exist
        data_dir = os.path.join('/vrx_ws/src/informative_planning/images', run_name)
        os.makedirs(data_dir, exist_ok=True)

        # Set file names based on run_name
        self.num_int_file = os.path.join(data_dir, f"numint_data.csv")
        self.ce_file = os.path.join(data_dir, f"ce_data.csv")
        self.entropy_file = os.path.join(data_dir, f"entropy_data.csv")
        self.visit_score_file = os.path.join(data_dir, f"visitscore_data.csv")

        # Create files with headers if they don't exist
        if not os.path.exists(self.num_int_file):
            with open(self.num_int_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['num_interceptions'])
        
        if not os.path.exists(self.ce_file):
            with open(self.ce_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['control_effort'])

        if not os.path.exists(self.entropy_file):
            with open(self.entropy_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['mean_entropy'])

        if not os.path.exists(self.visit_score_file):
            with open(self.visit_score_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['mean_visit_score'])

        # Write time-series data
        with open(self.num_int_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.num_int_list)

        with open(self.ce_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.ce_list)

        with open(self.entropy_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.entropy_list)

        with open(self.visit_score_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.visit_score_list)

        # --- Summary file alongside the time-series files ---
        summary_file = os.path.join(data_dir, f"{run_name}_summary.csv")
        summary = {
            'num_interceptions': self.num_int_list[-1] if self.num_int_list else 0,
            'final_ce':          self.ce_list[-1]      if self.ce_list      else -1,
            'final_entropy':     self.entropy_list[-1] if self.entropy_list else -1,
            'mean_visit_score':  float(np.mean(self.visit_score_list)) if self.visit_score_list else -1,
        }
        write_header = not os.path.exists(summary_file)
        with open(summary_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(summary)
    
def main():
    rclpy.init()
    logger_node = LoggerNode()
    rclpy.spin(logger_node)
    logger_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()