/*
 * @Description: ICP 匹配模块
 * @Author: Ge Yao
 * @Date: 2020-10-24 21:46:45
 */
#include "lidar_localization/models/registration/icp_registration.hpp"
#include "glog/logging.h"

namespace lidar_localization {

ICPRegistration::ICPRegistration(
    const YAML::Node& node
) : icp_ptr_(new pcl::IterativeClosestPoint<CloudData::POINT, CloudData::POINT>()) {
    
    float max_corr_dist = node["max_corr_dist"].as<float>();
    float trans_eps = node["trans_eps"].as<float>();
    float euc_fitness_eps = node["euc_fitness_eps"].as<float>();
    int max_iter = node["max_iter"].as<int>();

    SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
}

ICPRegistration::ICPRegistration(
    float max_corr_dist, 
    float trans_eps, 
    float euc_fitness_eps, 
    int max_iter
) : icp_ptr_(new pcl::IterativeClosestPoint<CloudData::POINT, CloudData::POINT>()) {

    SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
}

bool ICPRegistration::SetRegistrationParam(
    float max_corr_dist, 
    float trans_eps, 
    float euc_fitness_eps, 
    int max_iter
) {
    icp_ptr_->setMaxCorrespondenceDistance(max_corr_dist);
    icp_ptr_->setTransformationEpsilon(trans_eps);
    icp_ptr_->setEuclideanFitnessEpsilon(euc_fitness_eps);
    icp_ptr_->setMaximumIterations(max_iter);

    LOG(INFO) << "ICP params:" << std::endl
              << "max_corr_dist: " << max_corr_dist << ", "
              << "trans_eps: " << trans_eps << ", "
              << "euc_fitness_eps: " << euc_fitness_eps << ", "
              << "max_iter: " << max_iter 
              << std::endl << std::endl;

    return true;
}

bool ICPRegistration::SetInputTarget(const CloudData::CLOUD_PTR& input_target) {
    icp_ptr_->setInputTarget(input_target);

    return true;
}

bool ICPRegistration::ScanMatch(const CloudData::CLOUD_PTR& input_source, 
                                const Eigen::Matrix4f& predict_pose, 
                                CloudData::CLOUD_PTR& result_cloud_ptr,
                                Eigen::Matrix4f& result_pose) {
    icp_ptr_->setInputSource(input_source);
    icp_ptr_->align(*result_cloud_ptr, predict_pose);
    result_pose = icp_ptr_->getFinalTransformation();

    return true;
}

//********** my custom ICP ***********

ClosedICPRegistration::ClosedICPRegistration(const YAML::Node& node)
{
    float max_corr_dist = node["max_corr_dist"].as<float>();
    float trans_eps = node["trans_eps"].as<float>();
    float euc_fitness_eps = node["euc_fitness_eps"].as<float>();
    int max_iter = node["max_iter"].as<int>();

    SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
}

ClosedICPRegistration::ClosedICPRegistration(float max_corr_dist, float trans_eps, float euc_fitness_eps, int max_iter)
{
    SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
}

bool ClosedICPRegistration::SetRegistrationParam(float max_corr_dist, float trans_eps, float euc_fitness_eps, int max_iter)
{
    max_corr_dist_ = max_corr_dist;
    trans_eps_ = trans_eps;
    max_iter_ = max_iter;
    euc_fitness_eps_ = euc_fitness_eps;

    LOG(INFO) << "ICP params:" << std::endl
          << "max_corr_dist: " << max_corr_dist << ", "
          << "trans_eps: " << trans_eps << ", "
          << "max_iter: " << max_iter 
          << std::endl << std::endl;
    return true;
}

bool ClosedICPRegistration::SetInputTarget(const CloudData::CLOUD_PTR& input_target)
{
    mat_target_.resize(3, input_target->size());
    for(int i = 0; i < input_target->size(); ++i)
    {
        mat_target_(0, i) = input_target->points[i].x;
        mat_target_(1, i) = input_target->points[i].y;
        mat_target_(2, i) = input_target->points[i].z;
    }

    return true;
}

bool ClosedICPRegistration::ScanMatch(const CloudData::CLOUD_PTR& input_source, const Eigen::Matrix4f& predict_pose, 
                                        CloudData::CLOUD_PTR& result_cloud_ptr, Eigen::Matrix4f& result_pose)
{
    // CloudData::CLOUD_PTR predict_cloud(new CloudData::CLOUD());
    // pcl::transformPointCloud(*input_source, *predict_cloud, predict_pose);

    // source pointcloud mat
    mat_source_.resize(3, input_source->size());
    for(int i = 0; i < input_source->size(); ++i)
    {
        mat_source_(0, i) = input_source->points[i].x;
        mat_source_(1, i) = input_source->points[i].y;
        mat_source_(2, i) = input_source->points[i].z;
    }

    result_pose = ICP::point2point(mat_source_, mat_target_, predict_pose, max_iter_, trans_eps_, euc_fitness_eps_, max_corr_dist_);
    
    // transformed point cloud
    // result_cloud_ptr->resize(mat_source_.cols());
    // for(int i = 0; i < result_cloud_ptr->size(); ++i)
    // {
    //     result_cloud_ptr->points[i].x = mat_source_(0, i);
    //     result_cloud_ptr->points[i].y = mat_source_(1, i);
    //     result_cloud_ptr->points[i].z = mat_source_(2, i);
    // }
    pcl::transformPointCloud(*input_source, *result_cloud_ptr, result_pose);

    return true;
}                                        

}