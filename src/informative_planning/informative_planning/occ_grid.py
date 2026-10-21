#!usr/bin/env python3

import rclpy 
from rclpy.node import Node 
from geometry_msgs.msg import PoseArray, Point, Pose
import rclpy.time
from ros_gz_interfaces.msg import ParamVec
from nav_msgs.msg import OccupancyGrid, GridCells, Odometry
import numpy as np
import tf_transformations
from std_msgs.msg import Header, Float32
from usv_navigation.utils import pose_to_numpy, SyncSubscription
from informative_planning.params import origin, grid_size, resolution, p_low, p_high, alpha, beta, d_max, theta, decay_rate
from informative_planning.utils import world_to_grid, grid_to_world, to_logodds, to_prob, sensor_model, negative_sensor_model, get_fov
from scipy.ndimage import shift

class OccupancyGridNode(Node): 

    def __init__(self): 
        
        super().__init__("occupancy_grid")

        self.create_subscription(ParamVec, '/wamv/yolo/results', self.per_callback, 10)
        self.create_subscription(Odometry, '/wamv/sensors/position/ground_truth_odometry', self.odom_callback, 10)
        self.map_publisher = self.create_publisher(OccupancyGrid, '/gridmap',10)
        self.visit_score_publisher =  self.create_publisher(OccupancyGrid, '/visit_score_map',10)
        
        wind_df = {'/vrx/debug/wind/direction': Float32, '/vrx/debug/wind/speed': Float32}
        self.wind_sub = SyncSubscription(self, wind_df, self.wind_callback,10,1)
        self.wind_trigger = False
        self.resolution = resolution
        self.grid_size = grid_size
        self.origin = origin
        self.wind_factor = 0.03

        self.occupancy_grid = np.full((self.grid_size[1], self.grid_size[0]), 0.5, dtype=float)
        self.visit_score = np.full((self.grid_size[1], self.grid_size[0]), 1.0, dtype=float)
        self.log_odds = np.full((self.grid_size[1], self.grid_size[0]),0, dtype=float)

        self.d, self.theta = d_max, theta  # field of view distance and angle
        self.p_low, self.p_high = p_low, p_high  # low and high occupancy probabilities to clip
        self.alpha, self.beta = alpha, beta # dynamic occupancy grid update factor
        self.decay_rate = decay_rate # decay rate per second
        self.time_step = 0.2 # time step of mapping/perception updates

        self.l_low = to_logodds(self.p_low)
        self.l_high = to_logodds(self.p_high)
              
        self.trigger = False

        ## initialise occupancy grid and visit score grid
        self.grid = OccupancyGrid()
        self.grid.info.height = self.grid_size[1]
        self.grid.info.width = self.grid_size[0]
        self.grid.info.resolution = self.resolution
        self.grid.info.origin.position.x = 0.0
        self.grid.info.origin.position.y = 0.0

        self.visit_og = OccupancyGrid()
        self.visit_og.info.height = self.grid_size[1]
        self.visit_og.info.width = self.grid_size[0]
        self.visit_og.info.resolution = self.resolution
        self.visit_og.info.origin.position.x = 0.0
        self.visit_og.info.origin.position.y = 0.0

        self.counter = 0
        self.prev_time = 0
        self.time_ = 0
        self.prev_time = 0

        self.Rx = 0
        self.Ry = 0
        # publish average number of detections
        self.n = 0 # count frames
        self.no_detections = 0 # count number of detections
        self.det_pub = self.create_publisher(Float32, '/avg_detections', 10)
        
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
            init_pos = Pose()
            init_pos.position.x = self.pos.x
            init_pos.position.y = self.pos.y
            init_pos.orientation = orientation_q
            self.trigger = True
        
    def per_callback(self, msg):
        
        if self.trigger == False:
            return
        
        self.grid.header = Header()
        self.grid.header.frame_id = 'map'
        self.grid.header.stamp = self.get_clock().now().to_msg()

        self.visit_og.header = Header()
        self.visit_og.header.frame_id = 'map'
        self.visit_og.header.stamp = self.get_clock().now().to_msg()

        self.update_map() ## prediction step
        self.log_odds = np.vectorize(to_logodds)(self.occupancy_grid)
        self.log_odds = np.clip(self.log_odds, self.l_low, self.l_high)

        oc_ids = []
        self.n += 1
        
        for ob in msg.params:
            pos = ob.value.double_array_value
            conf = ob.value.double_value
            # class ids 7, 8: mb_round_buoy_black, mb_round_buoy_orange
            if ob.value.integer_value == 7 or ob.value.integer_value == 8:
                self.no_detections += 1 
                id = world_to_grid(pos[0],pos[1])
                dist = np.sqrt( (self.pos.x - pos[0])**2 + (self.pos.y - pos[1])**2 )
                if id and dist>2 and dist<35:
                    idx,idy = id
                    p0 = sensor_model(dist)
                    self.log_odds[idy, idx] += np.log(p0/(1-p0))
                    oc_ids.append((idy,idx))
        fov = get_fov(self.pos.x, self.pos.y, self.psi)
                                   
        for id in fov:
            if  ((id[0],id[1]) not in set(oc_ids)):
                j,i = id
                xc, yc = grid_to_world(i, j)
                if xc is None or yc is None:
                    continue
                dist = np.sqrt((xc-self.pos.x)**2 + (yc-self.pos.y)**2)              
                p = negative_sensor_model(dist)
                self.log_odds[j,i] += np.log(p/(1-p))

        # Update visit score
        self.visit_score += self.decay_rate * self.time_step

        for id in fov:
            j,i = id
            self.visit_score[j,i] = 0.0

        self.visit_score = np.clip(self.visit_score, 0, 1)

        self.log_odds = np.clip(self.log_odds, self.l_low, self.l_high)
        self.occupancy_grid = np.vectorize(to_prob)(self.log_odds)
        
        og = self.occupancy_grid.copy()*100
        og[og == 50.0] = -1
        self.grid.data = og.astype(np.int8).flatten().tolist()

        visit_score_og = self.visit_score.copy()*100
        visit_score_og = visit_score_og.astype(np.int8).flatten().tolist()
        self.visit_og.data = visit_score_og

        self.map_publisher.publish(self.grid)
        self.visit_score_publisher.publish(self.visit_og)
        self.det_pub.publish(Float32(data=self.no_detections/self.n))        
        
        self.counter +=1
               
    def wind_callback(self,dir:Float32,speed:Float32):
        self.wind_speed = speed.data
        self.wind_dir = dir.data * np.pi / 180.0  # Convert degrees to radians
        self.wind_factor = 0.02 + 0.0015 * self.wind_speed
        if not self.wind_trigger:
            self.wind_trigger = True

    def update_map(self):
        
        grid = self.occupancy_grid.copy()
    
        grid[grid<=0.5] = self.p_low
        
        self.prev_time = self.time_
        self.time_ = self.get_time()
        if self.prev_time == 0:
            self.prev_time = self.time_
            
        dt = self.time_ - self.prev_time
        dx = self.wind_factor * self.wind_speed * np.cos(self.wind_dir) * dt / self.resolution
        dy = self.wind_factor * self.wind_speed * np.sin(self.wind_dir) * dt / self.resolution
        
        self.Rx += dx
        self.Ry += dy
        shift_x = np.round(self.Rx)
        shift_y = np.round(self.Ry)
        self.Rx -= shift_x
        self.Ry -= shift_y
        
        out = shift(grid, shift=(shift_y, shift_x), order=1, mode='constant', cval=self.p_low)
                
        self.occupancy_grid = np.where((out-grid)!=0, out*self.alpha + self.occupancy_grid*self.beta, self.occupancy_grid)
        

    def get_time(self):
        return self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec*1e-9


def main(args=None):
    rclpy.init(args=args) 
    node = OccupancyGridNode()  
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__": 
    main()
    