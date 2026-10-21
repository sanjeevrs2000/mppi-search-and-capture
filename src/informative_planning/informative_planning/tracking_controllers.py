#!usr/bin/env python3

import numpy as np
from rclpy.node import Node
from informative_planning.params import *

class PurePursuit(Node):

    def __init__(self):
        
        super().__init__("pure_pursuit")

        self.U = u
        self.prev_action = [u, 0.0]
        self.t_step = track_time_step

    def velocity_command(self, pos, target_pos):

        xt, yt = target_pos[0], target_pos[1]
        x, y, psi = pos
        psi_des = np.arctan2(yt - y, xt - x)
        r = (psi_des - psi) / self.t_step
        vel_cmd = [self.U, r]
        self.get_logger().info(f"Intecepting")   

        return vel_cmd
