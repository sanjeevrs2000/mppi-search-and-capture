#include <rclcpp/rclcpp.hpp>
#include <gz/transport/Node.hh>
#include <gz/msgs/pose_v.pb.h>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/int32.hpp>

class PoseVListener : public rclcpp::Node
{
public:
  PoseVListener(const std::string &world_name) : Node("pose_v_listener"), initial_count_ (-1)
  {
    gz_node_.Subscribe("/world/" + world_name + "/dynamic_pose/info",
                       &PoseVListener::PoseCallback, this);

    pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/buoy_positions", 10);
    num_deleted_pub_ = this->create_publisher<std_msgs::msg::Int32>("/wamv/num_interceptions", 10);
  }

private:
  void PoseCallback(const gz::msgs::Pose_V &msg)
  {
    geometry_msgs::msg::PoseArray pose_array_msg;
    pose_array_msg.header.stamp = this->now();
    pose_array_msg.header.frame_id = "map";

    int current_count = 0;
    
    for (int i = 0; i < msg.pose_size(); ++i)
    {
      const auto &pose = msg.pose(i);
      const std::string &name = pose.name();

    if (name.find("round_buoy") != std::string::npos || name.find("buoy") != std::string::npos)
      {
        auto pos = pose.position();

        geometry_msgs::msg::Pose pose_msg;
        pose_msg.position.x = pos.x();
        pose_msg.position.y = pos.y();
        pose_msg.position.z = pos.z();

        pose_array_msg.poses.push_back(pose_msg);
        
        current_count++;
      }
    }
    pub_->publish(pose_array_msg);

    // set initial count if uninitialized
    if (initial_count_ == -1)
    {
      initial_count_ = current_count;
    }

    // publish number of deleted objects
    std_msgs::msg::Int32 num_deleted_msg;
    num_deleted_msg.data = initial_count_ - current_count;
    num_deleted_pub_->publish(num_deleted_msg);
  }

  gz::transport::Node gz_node_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr num_deleted_pub_;
  int initial_count_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  std::string world_name = "default";

  if (argc > 1)
  {
    world_name = argv[1];
  }

  auto node = std::make_shared<PoseVListener>(world_name);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
