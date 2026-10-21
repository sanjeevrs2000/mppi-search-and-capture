#include <rclcpp/rclcpp.hpp>
#include <gz/transport/Node.hh>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/entity.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/boolean.pb.h>
#include <vrx_custom_msgs/srv/remove_object.hpp>

#include <mutex>
#include <unordered_map>
#include <array>
#include <string>
#include <cmath>
#include <limits>

using namespace std::placeholders;

class GzObjectInteractServerNode : public rclcpp::Node
{
public:
    GzObjectInteractServerNode() : Node("object_delete_server")
    {
        
        this->declare_parameter<std::string>("world_name", "sydney_regatta");
        this->declare_parameter<double>("max_radius", 1.5);

        max_radius = this->get_parameter("max_radius").as_double();
        world_name = this->get_parameter("world_name").as_string();

        gz_node_.Subscribe("/world/" + world_name + "/dynamic_pose/info",
                           &GzObjectInteractServerNode::PoseCallback, this);

        server_ = this->create_service<vrx_custom_msgs::srv::RemoveObject>(
            "delete_object", std::bind(&GzObjectInteractServerNode::handle_request, this, _1, _2));
            
        pose_topic_ = "/world/" + world_name + "/dynamic_pose/info";
        remove_srv_ = "/world/" + world_name + "/remove";
    }

private:
    void PoseCallback(const gz::msgs::Pose_V &msg)
    {

        std::lock_guard<std::mutex> lk(map_mtx_);
        objects_.clear();

        for  (int i = 0; i< msg.pose_size(); ++i)
        {
            const auto &pose = msg.pose(i);
            const std::string &name = pose.name();

            if (name.find("round_buoy") != std::string::npos)
            {
                //save in dictionary with name as key and position as value                
                const auto &pose = msg.pose(i);
                const std::string &name = pose.name();

                if (name.find("round_buoy") != std::string::npos)
                {
                    objects_[name] = {
                    pose.position().x(),
                    pose.position().y(),
                    pose.position().z()
                };
                }
            }

        }
    }

    void handle_request(
        const std::shared_ptr<vrx_custom_msgs::srv::RemoveObject::Request> request,
        std::shared_ptr<vrx_custom_msgs::srv::RemoveObject::Response> response)
    {
        // Handle the delete object request
        const double tx = request->position.x;
        const double ty = request->position.y;
        
        std::string match_name;
        double min_dist = std::numeric_limits<double>::infinity();

        // check if an object in dictionary is within max_rad
        // and delete it from the world
        {
            std::lock_guard<std::mutex> lk(map_mtx_);
            for (const auto &kv : objects_)
            {
                const auto &name = kv.first;
                const auto &pos = kv.second;
                const double dx = pos[0] - tx;
                const double dy = pos[1] - ty;
                const double dist = std::sqrt(dx*dx + dy*dy);

                if (dist < min_dist && dist <=max_radius)
                {
                    min_dist = dist;
                    match_name = name;
                }
            }
        }

        if (match_name.empty())
        {
            response->success = false;
            response->object_name = "";
            response->message = "No object in range";

            RCLCPP_WARN(this->get_logger(), "%s (r=%.3f)", response->message.c_str(), max_radius);
            return;
        }

        //call remove service
        gz::msgs::Entity req;
        req.set_name(match_name);
        req.set_type(gz::msgs::Entity::MODEL);
        gz::msgs::Boolean rep;
        bool result = false;
        bool executed = gz_node_.Request(remove_srv_, req, 1000 /* ms */, rep, result);

        if (!executed)
        {
            response->success = false;
            response->object_name = match_name;
            response->message = "Remove service not executed";
            RCLCPP_ERROR(this->get_logger(), "[remove] Not executed for '%s'", match_name.c_str());
            return;
        }

        if (!result)
        {
            response->success = false;
            response->object_name = match_name;
            response->message = "Remove service timeout/no response";
            RCLCPP_ERROR(this->get_logger(), "[remove] No result for '%s'", match_name.c_str());
            return;
        }

        response->success = true;
        response->object_name = match_name;
        response->message = "Object deleted from world successfully";
        RCLCPP_INFO(this->get_logger(), "Deleted object '%s'", match_name.c_str());

    }

    std::string world_name;
    double max_radius;

    std::string pose_topic_;
    std::string remove_srv_;
    gz::transport::Node gz_node_;

    std::mutex map_mtx_;
    std::unordered_map<std::string, std::array<double, 3>> objects_;

    rclcpp::Service<vrx_custom_msgs::srv::RemoveObject>::SharedPtr server_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GzObjectInteractServerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}