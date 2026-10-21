#!/usr/bin/env python3 
import rclpy 
from rclpy.node import Node 
from nav_msgs.msg import Odometry
import tf_transformations
from std_msgs.msg import Float64, Int32
from geometry_msgs.msg import Twist
from usv_navigation.utils import speed_thrust_wamv
from informative_planning.params import track_time_step, time_step
import numpy as np

class PD_Controller(Node): 
    def __init__(self): 

        super().__init__("pd_speed_controller")
        
        self.create_subscription(Odometry, '/wamv/sensors/position/ground_truth_odometry', self.odom_callback, 10)
        self.create_subscription(Twist, 'wamv/cmd_vel', self.command_callback, 10)
        self.create_subscription(Int32, 'wamv/guidance_mode', self.mode_callback, 10)

        self.left_thrust_publisher = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.right_thrust_publisher = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        
        self.trigger = False
        self.cmd_trigger = False
        self.timer = self.create_timer(0.02, self.controller)

        self.guidance_mode = 0
        self.time_step = time_step
        self.X = 275

    def command_callback(self, msg):
        
        if self.trigger == False:
            return

        self.u = msg.linear.x
        self.X = speed_thrust_wamv(self.u)

        r = msg.angular.z
        self.psi_d = self.ssa(self.psi + r * self.time_step)

        if self.cmd_trigger == False:
            self.cmd_trigger = True

    def mode_callback(self, msg):

        self.guidance_mode = msg.data

        if self.guidance_mode == 1:
            self.time_step = track_time_step
        else:
            self.time_step = time_step

    def odom_callback(self, msg):
        
        self.pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        orientation_q = msg.pose.pose.orientation
        q = [orientation_q.x,orientation_q.y,orientation_q.z,orientation_q.w]
        self.angular_pos = tf_transformations.euler_from_quaternion(q)
        self.psi = self.angular_pos[2]
        
        self.linear_vel = msg.twist.twist.linear
        self.angular_vel = msg.twist.twist.angular        
        self.r = self.angular_vel.z

        if self.trigger == False:
            self.trigger = True

    def controller(self):
        
        if self.trigger == False or self.cmd_trigger == False:
            return

        self.psi_error = self.ssa(self.psi_d - self.psi)

        m=333
        T=1
        k=T/m
        d=1/k
        wn=2
        zeta=0.8
        
        Kp = m*wn**2-k
        Kd = 2*m*zeta*wn-d
        
        N = -Kp*self.psi_error - Kd*self.r
        X = self.X

        #thrust allocation:
        lp_y = -1.027
        ls_y = 1.027

        b = ls_y - lp_y

        t_p = X*0.5 + N/b
        t_s = X*0.5 - N/b

        self.t_p = t_p
        self.t_s = t_s

        self.left_thrust_publisher.publish(Float64(data=self.t_p))
        self.right_thrust_publisher.publish(Float64(data=self.t_s))

    def ssa(self, angle):
        """ 
        Smallest Signed Angle (SSA) between -pi and pi
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

def main(args=None):    

    rclpy.init(args=args)
    node = PD_Controller()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__": 
    main()