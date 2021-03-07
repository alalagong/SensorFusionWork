/*
 * @Description: IMU integration activity
 * @Author: Ge Yao
 * @Date: 2020-11-10 14:25:03
 */
#include <cmath>

#include "imu_integration/estimator/activity.hpp"
#include "glog/logging.h"
namespace imu_integration {

namespace estimator {

Activity::Activity(void) 
    : private_nh_("~"), 
    initialized_(false),
    // gravity acceleration:
    G_(0, 0, -9.81),
    // angular velocity bias:
    angular_vel_bias_(0.0, 0.0, 0.0),
    // linear acceleration bias:
    linear_acc_bias_(0.0, 0.0, 0.0)
{}

void Activity::Init(void) {
    // parse IMU config:
    private_nh_.param("imu/topic_name", imu_config_.topic_name, std::string("/sim/sensor/imu"));
    private_nh_.param("imu/integ_method", imu_config_.integ_method, std::string("mid"));
    imu_sub_ptr_ = std::make_shared<IMUSubscriber>(private_nh_, imu_config_.topic_name, 1000000);

    // a. gravity constant:
    private_nh_.param("imu/gravity/x", imu_config_.gravity.x,  0.0);
    private_nh_.param("imu/gravity/y", imu_config_.gravity.y,  0.0);
    private_nh_.param("imu/gravity/z", imu_config_.gravity.z, -9.81);
    G_.x() = imu_config_.gravity.x;
    G_.y() = imu_config_.gravity.y;
    G_.z() = imu_config_.gravity.z;

    // b. angular velocity bias:
    private_nh_.param("imu/bias/angular_velocity/x", imu_config_.bias.angular_velocity.x,  0.0);
    private_nh_.param("imu/bias/angular_velocity/y", imu_config_.bias.angular_velocity.y,  0.0);
    private_nh_.param("imu/bias/angular_velocity/z", imu_config_.bias.angular_velocity.z,  0.0);
    angular_vel_bias_.x() = imu_config_.bias.angular_velocity.x;
    angular_vel_bias_.y() = imu_config_.bias.angular_velocity.y;
    angular_vel_bias_.z() = imu_config_.bias.angular_velocity.z;

    // c. linear acceleration bias:
    private_nh_.param("imu/bias/linear_acceleration/x", imu_config_.bias.linear_acceleration.x,  0.0);
    private_nh_.param("imu/bias/linear_acceleration/y", imu_config_.bias.linear_acceleration.y,  0.0);
    private_nh_.param("imu/bias/linear_acceleration/z", imu_config_.bias.linear_acceleration.z,  0.0);
    linear_acc_bias_.x() = imu_config_.bias.linear_acceleration.x;
    linear_acc_bias_.y() = imu_config_.bias.linear_acceleration.y;
    linear_acc_bias_.z() = imu_config_.bias.linear_acceleration.z;

    // parse odom config:
    private_nh_.param("pose/frame_id", odom_config_.frame_id, std::string("inertial"));
    private_nh_.param("pose/topic_name/ground_truth", odom_config_.topic_name.ground_truth, std::string("/pose/ground_truth"));
    private_nh_.param("pose/topic_name/estimation", odom_config_.topic_name.estimation, std::string("/pose/estimation"));

    odom_ground_truth_sub_ptr = std::make_shared<OdomSubscriber>(private_nh_, odom_config_.topic_name.ground_truth, 1000000);
    odom_estimation_pub_ = private_nh_.advertise<nav_msgs::Odometry>(odom_config_.topic_name.estimation, 500);
}

bool Activity::Run(void) {
    if (!ReadData())
        return false;

    while(HasData()) {
        if (UpdatePose()) {
            PublishPose();
        }
        ReadData();
    }
    return true;
}

bool Activity::ReadData(void) {
    // fetch IMU measurements into buffer:
    // size_t before_imu = imu_data_buff_.size();
    // LOG(INFO) << "before has imu " << before_imu ;
    imu_sub_ptr_->ParseData(imu_data_buff_);
    // size_t after_imu = imu_data_buff_.size();
    // LOG(INFO) << "after has imu " << after_imu ;
    // size_t add_imu = after_imu - before_imu;

    if (static_cast<size_t>(0) == imu_data_buff_.size())
        return false;

    // if (!initialized_) {
    // size_t before_odom = odom_data_buff_.size();
    // LOG(INFO) << "before has odom " << before_odom ;
    odom_ground_truth_sub_ptr->ParseData(odom_data_buff_);
    // size_t after_odom = odom_data_buff_.size();
    // LOG(INFO) << "after has odom " << after_odom ;
    // size_t add_odom = after_odom - before_odom;

    if (static_cast<size_t>(0) == odom_data_buff_.size())
        return false;
    // }

    return true;
}

bool Activity::HasData(void) {
    if (imu_data_buff_.size() < static_cast<size_t>(3))
        return false;

    if (
        !initialized_ && 
        static_cast<size_t>(0) == odom_data_buff_.size()
    ) {
        return false;
    }

    return true;
}

bool Activity::UpdatePose(void) {
    if (!initialized_) {
        // use the latest measurement for initialization:
        OdomData odom_data = odom_data_buff_.back();
        IMUData imu_data = imu_data_buff_.back();

        pose_ = odom_data.pose;
        vel_ = odom_data.vel;

        initialized_ = true;

        odom_data_buff_.clear();
        imu_data_buff_.clear();

        // keep the latest IMU measurement for mid-value integration:
        imu_data_buff_.push_back(imu_data);
        odom_data_buff_.push_back(odom_data);

        // size_curr = 1;
        // index_imu_curr = 0;
        index_odom_curr_ = 0;
    } else {
        //
        // TODO: implement your estimation here
        //
        if(imu_data_buff_.size() < static_cast<size_t>(2) )
            return false;
        
        Eigen::Vector3d velocity_delta, angular_delta;

        if(imu_config_.integ_method == std::string("euler"))
        {
            const IMUData &imu_data_prev = imu_data_buff_.at(0);
            const IMUData &imu_data_curr = imu_data_buff_.at(1);

            double delta_t = imu_data_curr.time - imu_data_prev.time;
            
            Eigen::Matrix3d R_prev = pose_.block<3,3>(0,0);
            Eigen::Matrix3d R_curr;
            // get deltas:
            GetAngularDeltaEuler(0, delta_t, angular_delta);          
            // update orientation:
            UpdateOrientation(angular_delta, R_curr, R_prev);
            // get velocity delta:
            GetVelocityDeltaEuler(0, R_prev, delta_t, velocity_delta);
            // update position:
            UpdatePosition(delta_t, velocity_delta);
        } else if (imu_config_.integ_method == std::string("mid"))
        {
            double delta_t = 0;
            Eigen::Matrix3d R_prev = pose_.block<3,3>(0,0);
            Eigen::Matrix3d R_curr;
            // get deltas:
            GetAngularDeltaMid(0, 1, angular_delta);          
            // update orientation:
            UpdateOrientation(angular_delta, R_curr, R_prev);
            // get velocity delta:
            GetVelocityDeltaMid_RK4(0, 1, R_curr, R_prev, delta_t, velocity_delta);
            // update position:
            UpdatePosition(delta_t, velocity_delta);
        } else if (imu_config_.integ_method == std::string("rk4"))
        {
            double delta_t = 0;
            Eigen::Matrix3d R_prev = pose_.block<3,3>(0,0);
            Eigen::Matrix3d R_curr;
            // get deltas:
            GetAngularDeltaRK4(0, 1, angular_delta);          
            // update orientation:
            UpdateOrientation(angular_delta, R_curr, R_prev);
            // get velocity delta:
            GetVelocityDeltaMid_RK4(0, 1, R_curr, R_prev, delta_t, velocity_delta);
            // update position:
            UpdatePosition(delta_t, velocity_delta);
        }

        // save
        // OdomData odom_data;
        // odom_data.time = imu_data_buff_.at(1).time;
        // odom_data.pose = pose_;
        // result_buff_.push_back(odom_data);

        // move forward -- 
        // NOTE: this is NOT fixed. you should update your buffer according to the method of your choice:
        imu_data_buff_.pop_front();
        // index_odom_curr_++;
    }
    
    return true;
}

bool Activity::PublishPose() {
    // a. set header:
    message_odom_.header.stamp = ros::Time::now();
    message_odom_.header.frame_id = odom_config_.frame_id;
    
    // b. set child frame id:
    message_odom_.child_frame_id = odom_config_.frame_id;

    // b. set orientation:
    Eigen::Quaterniond q(pose_.block<3, 3>(0, 0));
    message_odom_.pose.pose.orientation.x = q.x();
    message_odom_.pose.pose.orientation.y = q.y();
    message_odom_.pose.pose.orientation.z = q.z();
    message_odom_.pose.pose.orientation.w = q.w();

    // c. set position:
    Eigen::Vector3d t = pose_.block<3, 1>(0, 3);
    message_odom_.pose.pose.position.x = t.x();
    message_odom_.pose.pose.position.y = t.y();
    message_odom_.pose.pose.position.z = t.z();  

    // d. set velocity:
    message_odom_.twist.twist.linear.x = vel_.x();
    message_odom_.twist.twist.linear.y = vel_.y();
    message_odom_.twist.twist.linear.z = vel_.z(); 

    odom_estimation_pub_.publish(message_odom_);

    return true;
}

/**
 * @brief  get unbiased angular velocity in body frame
 * @param  angular_vel, angular velocity measurement
 * @return unbiased angular velocity in body frame
 */
inline Eigen::Vector3d Activity::GetUnbiasedAngularVel(const Eigen::Vector3d &angular_vel) {
    return angular_vel - angular_vel_bias_;
}

/**
 * @brief  get unbiased linear acceleration in navigation frame
 * @param  linear_acc, linear acceleration measurement
 * @param  R, corresponding orientation of measurement
 * @return unbiased linear acceleration in navigation frame
 */
inline Eigen::Vector3d Activity::GetUnbiasedLinearAcc(
    const Eigen::Vector3d &linear_acc,
    const Eigen::Matrix3d &R
) {
    return R*(linear_acc - linear_acc_bias_) - G_;
}

/**
 * @brief  get angular delta by euler
 * @param  index_prev, previous imu measurement buffer index
 * @param  angular_delta, angular delta output
 * @return true if success false otherwise
 */
bool Activity::GetAngularDeltaEuler(
    const size_t index_prev, double &delta_t,
    Eigen::Vector3d &angular_delta
) {
    if(imu_data_buff_.size() <= index_prev)
        return false;
    
    const IMUData &imu_data_prev = imu_data_buff_.at(index_prev);
    
    Eigen::Vector3d angular_vel_prev = GetUnbiasedAngularVel(imu_data_prev.angular_velocity);
    
    angular_delta = delta_t*angular_vel_prev;
    
    return true;
}

/**
 * @brief  get angular delta by mid-value
 * @param  index_curr, current imu measurement buffer index
 * @param  index_prev, previous imu measurement buffer index
 * @param  angular_delta, angular delta output
 * @return true if success false otherwise
 */
bool Activity::GetAngularDeltaMid(
    const size_t index_curr, const size_t index_prev,
    Eigen::Vector3d &angular_delta
) {
    //
    // TODO: this could be a helper routine for your own implementation
    //
    if (
        index_curr <= index_prev ||
        imu_data_buff_.size() <= index_curr
    ) {
        return false;
    }

    const IMUData &imu_data_curr = imu_data_buff_.at(index_curr);
    const IMUData &imu_data_prev = imu_data_buff_.at(index_prev);

    double delta_t = imu_data_curr.time - imu_data_prev.time;

    Eigen::Vector3d angular_vel_curr = GetUnbiasedAngularVel(imu_data_curr.angular_velocity);
    Eigen::Vector3d angular_vel_prev = GetUnbiasedAngularVel(imu_data_prev.angular_velocity);

    angular_delta = 0.5*delta_t*(angular_vel_curr + angular_vel_prev);

    return true;
}

/**
 * @brief  get angular delta by RK4
 * @param  index_prev, previous imu measurement buffer index
 * @param  index_prev, previous imu measurement buffer index
 * @param  q_last, corresponding orientation of previous imu measurement
 * @param  angular_delta, angular delta output
 * @return true if success false otherwise
 */
bool Activity::GetAngularDeltaRK4(
    const size_t index_curr, const size_t index_prev, 
    Eigen::Vector3d &angular_delta
) {
    if (
        index_curr <= index_prev ||
        imu_data_buff_.size() <= index_curr
    ) {
        return false;
    }

    const IMUData &imu_data_curr = imu_data_buff_.at(index_curr);
    const IMUData &imu_data_prev = imu_data_buff_.at(index_prev);

    double delta_t = imu_data_curr.time - imu_data_prev.time;
    
    Eigen::Vector3d angular_vel_prev = GetUnbiasedAngularVel(imu_data_prev.angular_velocity);
    Eigen::Vector3d angular_vel_curr = GetUnbiasedAngularVel(imu_data_curr.angular_velocity);
    Eigen::Vector3d angular_vel_mid = 0.5f * (angular_vel_curr + angular_vel_prev);

    //*********** 对四元数微分方程进行RK4 ************
    // Eigen::Quaterniond k1, k2, k3, k4;
    // Eigen::Quaterniond q1_linear, q2_linear, q3_linear, q4_linear;

    // q1_linear = q_last;
    // k1 = q1_linear * Eigen::Quaterniond(0, 0.5*angular_vel_prev);

    // q2_linear = q_last + 0.5*delta_t*k1;
    // k2 = q2_linear * Eigen::Quaterniond(0, 0.5*angular_vel_mid);

    // q3_linear = q_last + 0.5*delta_t*k2;
    // k3 = q3_linear * Eigen::Quaterniond(0, 0.5*angular_vel_mid);

    // q4_linear = q_last * delta_t*k3;
    // k4 = q4_linear * Eigen::Quaterniond(0, 0.5*angular_vel_curr);

    // q_curr = q_last + 1.f/6.f*delta_t*(k1 + 2.f*k2 + 2.f*k3 + k4);
    // q_curr.normalize();

    //********** 对旋转矢量微分方程进行RK4 ***********
    Eigen::Vector3d k1, k2, k3, k4;
    
    k1 = angular_vel_prev;
    k2 = angular_vel_mid + 1.f/4.f*delta_t*k1.cross(angular_vel_mid);
    k3 = angular_vel_mid + 1.f/4.f*delta_t*k2.cross(angular_vel_mid);
    k4 = angular_vel_curr + 1.f/2.f*delta_t*k3.cross(angular_vel_curr);
    
    angular_delta = delta_t/6.f*(k1 + 2*k2 + 2*k3 + k4);

    return true;
}

/**
 * @brief  get velocity delta
 * @param  index_curr, current imu measurement buffer index
 * @param  index_prev, previous imu measurement buffer index
 * @param  R_curr, corresponding orientation of current imu measurement
 * @param  R_prev, corresponding orientation of previous imu measurement
 * @param  velocity_delta, velocity delta output
 * @return true if success false otherwise
 */
bool Activity::GetVelocityDeltaMid_RK4(
    const size_t index_curr, const size_t index_prev,
    const Eigen::Matrix3d &R_curr, const Eigen::Matrix3d &R_prev, 
    double &delta_t, Eigen::Vector3d &velocity_delta
) {
    //
    // TODO: this could be a helper routine for your own implementation
    //
    if (
        index_curr <= index_prev ||
        imu_data_buff_.size() <= index_curr
    ) {
        return false;
    }

    const IMUData &imu_data_curr = imu_data_buff_.at(index_curr);
    const IMUData &imu_data_prev = imu_data_buff_.at(index_prev);

    delta_t = imu_data_curr.time - imu_data_prev.time;

    Eigen::Vector3d linear_acc_curr = GetUnbiasedLinearAcc(imu_data_curr.linear_acceleration, R_curr);
    Eigen::Vector3d linear_acc_prev = GetUnbiasedLinearAcc(imu_data_prev.linear_acceleration, R_prev);
    
    velocity_delta = 0.5*delta_t*(linear_acc_curr + linear_acc_prev);

    return true;
}

/**
 * @brief  get velocity delta by Euler
 * @param  index_prev, previous imu measurement buffer index
 * @param  R_prev, corresponding orientation of previous imu measurement
 * @param  velocity_delta, velocity delta output
 * @return true if success false otherwise
 */
bool Activity::GetVelocityDeltaEuler(
    const size_t index_prev, const Eigen::Matrix3d &R_prev, 
    double &delta_t, Eigen::Vector3d &velocity_delta
) {
    if(imu_data_buff_.size() <= index_prev)
        return false;

    const IMUData &imu_data_prev = imu_data_buff_.at(index_prev);
    
    Eigen::Vector3d linear_acc_prev = GetUnbiasedLinearAcc(imu_data_prev.linear_acceleration, R_prev);

    velocity_delta = delta_t*linear_acc_prev;

    return true;
}

/**
 * @brief  update orientation with effective rotation angular_delta
 * @param  angular_delta, effective rotation
 * @param  R_curr, current orientation
 * @param  R_prev, previous orientation
 * @return void
 */
void Activity::UpdateOrientation(
    const Eigen::Vector3d &angular_delta,
    Eigen::Matrix3d &R_curr, Eigen::Matrix3d &R_prev
) {
    //
    // TODO: this could be a helper routine for your own implementation
    //
    // magnitude:
    double angular_delta_mag = angular_delta.norm();
    // direction:
    Eigen::Vector3d angular_delta_dir = angular_delta.normalized();

    // build delta q:
    double angular_delta_cos = cos(angular_delta_mag/2.0);
    double angular_delta_sin = sin(angular_delta_mag/2.0);
    Eigen::Quaterniond dq(
        angular_delta_cos, 
        angular_delta_sin*angular_delta_dir.x(), 
        angular_delta_sin*angular_delta_dir.y(), 
        angular_delta_sin*angular_delta_dir.z()
    );
    Eigen::Quaterniond q(pose_.block<3, 3>(0, 0));
    
    // update:
    q = q*dq;
    
    // write back:
    R_prev = pose_.block<3, 3>(0, 0);
    pose_.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    R_curr = pose_.block<3, 3>(0, 0);
}

/**
 * @brief  update orientation with effective velocity change velocity_delta
 * @param  delta_t, timestamp delta 
 * @param  velocity_delta, effective velocity change
 * @return void
 */
void Activity::UpdatePosition(const double &delta_t, const Eigen::Vector3d &velocity_delta) {
    //
    // TODO: this could be a helper routine for your own implementation
    //
    pose_.block<3, 1>(0, 3) += delta_t*vel_ + 0.5*delta_t*velocity_delta;
    vel_ += velocity_delta;
}

void Activity::ComputeError(void) {
    
    error_ = 0;
    // if(result_buff_.size() != odom_data_buff_.size())
    //      LOG(WARNING)<<"error ! " << odom_data_buff_.size() << " and " << result_buff_.size();

    if(odom_data_buff_.size() == 0 || result_buff_.size() == 0)
        return;
            
    // aligned
    const OdomData &odom_data_curr = odom_data_buff_.at(0);
    const OdomData &imu_data_curr = result_buff_.at(0);
    
    double delta_t = odom_data_curr.time - imu_data_curr.time;

    while(abs(delta_t) > 0.009)
    {
        if(delta_t > 0.009)
            result_buff_.pop_front();
        if(delta_t < -0.009)
            odom_data_buff_.pop_front();

        if(odom_data_buff_.size() == 0 || result_buff_.size() == 0)
            return; 
    }
    size_t num = odom_data_buff_.size() < result_buff_.size() ? odom_data_buff_.size() : result_buff_.size();

    for(size_t i = 0; i < num; ++i)
    {
        const OdomData &odom_data_curr = odom_data_buff_.at(i);
        const OdomData &imu_data_curr = result_buff_.at(i);

        double delta_t = odom_data_curr.time - imu_data_curr.time;
    
        // odom too late
        // if(delta_t > 0.009)
        // {
        //     LOG(INFO) << "Time different, Odom time is " << odom_data_curr.time << 
        //             " Imu time is " << imu_data_curr.time << std::endl;
            // double time_odom = odom_data_curr.time;
            // double time_imu = imu_data_curr.time;

            // while((time_odom - time_imu) > 0.009)
            // {
            //     if(index_odom_curr_ == 0)
            //         break;
            //     index_odom_curr_--;
            //     time_odom = odom_data_buff_.at(index_odom_curr_).time;
            // }
        // }
        // odom too early
        // else if(delta_t < -0.009)
        // {
        //     LOG(INFO) << "Time different, Odom time is " << odom_data_curr.time << 
        //             " Imu time is " << imu_data_curr.time << std::endl;
        //     double time_odom = odom_data_curr.time;
        //     double time_imu = imu_data_curr.time;

        //     while((time_odom - time_imu) < -0.009)
        //     {
        //         if(index_odom_curr_ >= odom_data_buff_.size()-1)
        //             break;
        //         index_odom_curr_++;
        //         time_odom = odom_data_buff_.at(index_odom_curr_).time;
        //     }
        // }else
            error_ += (odom_data_curr.pose.block<3,1>(0,3) - pose_.block<3,1>(0,3)).norm();
    }
    LOG(INFO) << "this traj mean error is " << error_/num;
}

} // namespace estimator

} // namespace imu_integration