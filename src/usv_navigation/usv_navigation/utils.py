#!/usr/bin/env python3
import numpy as np
from rclpy.node import Node
import message_filters

FRONT_LEFT_K = np.array(
    [762.7223205566406, 0.0, 640.0, 0.0, 762.7223110198975, 360.0, 0.0, 0.0, 1.0]).reshape((3, 3))

BASE_LINK_TO_FRONT_LEFT_CAMERA_LINK_TF = (np.array([0.75, 0.1, 1.5]), np.array(
    [[0.9659258262890682, 0.0, 0.2588190451025209], [0.0, 1.0, 0.0], [-0.2588190451025209, 0.0, 0.9659258262890682]]))

SPEED_THRUST_TABLE = [
    [0.000,   0],
    [0.636,   125],
    [0.990,   250],
    [1.280,   375],
    [1.520,   500],
    [1.733,   625],
    [1.924,   750],
    [2.260,  1000],
    [2.568,  1250],
    [2.840,  1500],
    [3.093,  1750],
    [3.320,  2000],
    [4.830,  4000],
    [5.370,  5000],
]

thrust_table = np.asarray(SPEED_THRUST_TABLE)

speeds = np.array(thrust_table[:,0])
thrusts = np.array(thrust_table[:,1])


class SyncSubscription:
    def __init__(self, node: Node, df, callback, queue_size=10, approx_time=0) -> None:
        if approx_time == 0:
            node.get_logger().debug(
                f'Subs to topics in sync : {",".join(df.keys())}')
            self.msgs_sub = message_filters.TimeSynchronizer(
                [message_filters.Subscriber(node, df[topic], topic) for topic in df], queue_size=queue_size)
        else:
            self.msgs_sub = message_filters.ApproximateTimeSynchronizer(
                [message_filters.Subscriber(node, df[topic], topic) for topic in df],
                slop=approx_time, queue_size=queue_size, allow_headerless=True)
        self.msgs_sub.registerCallback(callback)


def pose_to_numpy(pose):
    rot_mat = quaternion_to_rotation_matrix(pose.orientation)
    trans = np.array([pose.position.x,
                      pose.position.y, pose.position.z])
    return trans, rot_mat


def quaternion_to_rotation_matrix(quaternion):
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w

    rotation_matrix = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y -
                2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x -
                2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 *
                x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )
    return rotation_matrix


def speed_thrust_wamv(speed):

    return np.interp(speed, speeds, thrusts)
