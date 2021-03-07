#include <fstream>

#include <ros/ros.h>
#include <rosbag/bag.h>

#include "imu_integration/estimator/activity.hpp"

int main(int argc, char** argv) {
    std::string node_name{"imu_integration_estimator_node"};
    ros::init(argc, argv, node_name);
    
    imu_integration::estimator::Activity activity;

    activity.Init();
    
    // 100 Hz:
    int loop = 5000;
    ros::Rate loop_rate(100);
    while (ros::ok() && (loop--)>0)
    {
        ros::spinOnce();

        activity.Run();

        loop_rate.sleep();
    } 
    activity.ComputeError();

    return EXIT_SUCCESS;
}