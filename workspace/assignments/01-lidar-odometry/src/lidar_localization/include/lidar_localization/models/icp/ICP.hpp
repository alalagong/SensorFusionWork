/*
 * @Description: ICP SVD实现
 * @Author: Gong Yi Qun
 * @Date: 2021-01-31 21:46:57
 */

#ifndef LIDAR_LOCALIZATION_MODELS_ICP_ICP_HPP_
#define LIDAR_LOCALIZATION_MODELS_ICP_ICP_HPP_

#include "lidar_localization/tools/nanoflann.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>
#include "glog/logging.h"

namespace lidar_localization {
namespace ICP
{
    struct KDTree
    {
        typedef nanoflann::L2_Simple_Adaptor<float,KDTree> Metric;
        typedef nanoflann::KDTreeSingleIndexAdaptor<Metric,KDTree,3,int> KDTreeIndex;

        KDTree(const Eigen::Matrix3Xf& point_mat) : data_matrix_(point_mat)
        {
            index_ = new KDTreeIndex(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10));
            index_->buildIndex();
        }
        ~KDTree() { delete index_; }

        //@ 返回最近的一个点
        inline int closest(const float *query_point) const {
            int out_indices;
            float out_distances_sq;
            nanoflann::KNNResultSet<float,int> result(1);
            result.init(&out_indices, &out_distances_sq);
            index_->findNeighbors(result, query_point, nanoflann::SearchParams());
            return out_indices;
        }

        //@ 返回点的数目（列数）
        inline size_t kdtree_get_point_count() const {return data_matrix_.cols();}

        //@ 返回p1点和第idx_p2个点
        inline float kdtree_distance(const float *p1,const size_t idx_p2,size_t size) const {
            float s=0;
            for (size_t i=0; i<size; i++) {
                const float d= p1[i]-data_matrix_.coeff(i,idx_p2);
                s+=d*d;
            }
            return s;
        }
        
        //@ 返回[dim][idx]
        inline float kdtree_get_pt(const size_t idx, int dim) const {
            return data_matrix_.coeff(dim,idx);
        }
        template <class BBOX> bool kdtree_get_bbox(BBOX&) const {return false;}

        //@ data
        const Eigen::Matrix3Xf& data_matrix_;
        KDTreeIndex* index_;
    };

    // Eigen::Matrix4f point2point(Eigen::Matrix3Xf& X, Eigen::Matrix3Xf& Y, int max_iter, float trans_eps, float max_corr_dist = 0.1);
    // Eigen::Matrix4f point2plane(Eigen::Matrix3Xf& X, Eigen::Matrix3Xf& Y, Eigen::Matrix3Xf& N, int max_iter, float trans_eps, float max_corr_dist = 0.1);

//? why  multiple definition of ????
static Eigen::Matrix4f point2point(Eigen::Matrix3Xf& X, Eigen::Matrix3Xf& Y, Eigen::Matrix4f predict_transfom, 
                                int max_iter, float trans_eps, float euc_fitness_eps, float max_corr_dist = 0.1) 
{
    // build kdtree to find closest point
    KDTree kdtree(Y);

    Eigen::Matrix3Xf Y_permutated = Eigen::Matrix3Xf::Zero(3, X.cols());
    Eigen::VectorXf W = Eigen::VectorXf::Zero(X.cols());
    Eigen::Matrix3Xf Xo1 = X;
    Eigen::Matrix3Xf Xo2 = X;
    Eigen::Matrix3Xf X_trans = X;   
    Eigen::Affine3f transformation;
    static int flag = 0;      
    transformation.matrix() = predict_transfom;  // init value
    X_trans = transformation*X;
    double match_error = 1e6;

    for(int icp=0; icp<max_iter; ++icp) {
        //[1] Find closest point
        #pragma omp parallel for
        for(int i=0; i<X.cols(); ++i) {
            Y_permutated.col(i) = Y.col(kdtree.closest(X_trans.col(i).data()));
        }
        // LOG(INFO) << "before align two pooint is " <<  X.col(5).transpose() << " and " << Y_permutated.col(5).transpose() << std::endl;
        // Eigen::VectorXf point_error = (X_trans-Y_permutated).colwise().norm();        
        // match_error = point_error.norm();
        // Eigen::Affine3f transformation_start = transformation;
        //[2] Computer rotation and translation
        // for(int outer=0; outer<30; ++outer) {         
                
            //[2.1] Compute weights
            W = (X_trans-Y_permutated).colwise().norm();
            
            // std::vector<float> W_v;
            // for(int i=0; i<W.rows(); ++i)
            // {
            //     W_v.push_back(W(i));
            // }

            // int median = floor(W.rows()*0.7-1);
            // std::nth_element(W_v.begin(),W_v.begin()+median, W_v.end());
            // double p = std::max(max_corr_dist, W_v[median]);
            double p = max_corr_dist*max_corr_dist;
            // tukey weight
            for(int i=0; i<W.rows(); ++i) {
                if(W(i) > p) 
                    W(i) = 0.0;
                else 
                    W(i) = 1.0; //std::pow((1.0 - std::pow(W(i)/p,2.0)), 2.0);;
            }
            // W = Eigen::VectorXf::Ones(X_trans.cols());
            // Normalize weight vector
            Eigen::VectorXf w_normalized = W/W.sum();
            
            //[2.2] Rotation and translation update
            // De-mean
            Eigen::Vector3f X_mean, Y_mean;
            for(int i=0; i<3; ++i) {
                X_mean(i) = (X_trans.row(i).array()*w_normalized.transpose().array()).sum();
                Y_mean(i) = (Y_permutated.row(i).array()*w_normalized.transpose().array()).sum();
            }
            X_trans.colwise() -= X_mean;
            Y_permutated.colwise() -= Y_mean;
            
            //[2.3] Compute transformation
            Eigen::Affine3f transformation_temp;
            Eigen::Matrix3f sigma = X_trans * w_normalized.asDiagonal() * Y_permutated.transpose();
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(sigma  , Eigen::ComputeFullU | Eigen::ComputeFullV);
            // if(svd.matrixU().determinant()*svd.matrixV().determinant() < 0.0) {
            //     // Eigen::Vector3f S = Eigen::Vector3f::Ones(); S(0) = -1.0;S(1) = -1.0;S(2) = -1.0;
            //     transformation_temp.linear().noalias() = -svd.matrixV()*svd.matrixU().transpose();
            // } else {
            //     transformation_temp.linear().noalias() = svd.matrixV()*svd.matrixU().transpose();
            // }
            double det = (svd.matrixV()*svd.matrixU().transpose()).determinant();
            Eigen::Vector3f S = Eigen::Vector3f::Ones(); 
            // S(0) = 1.f/det;
            // S(1) = 1.f/det;
            S(2) = det;
            transformation_temp.linear().noalias() = svd.matrixV()*S.asDiagonal()*svd.matrixU().transpose();
            transformation_temp.translation().noalias() = Y_mean - transformation_temp.linear()*X_mean;
            
            //[2.4] Re-apply mean
            X_trans.colwise() += X_mean;
            Y_permutated.colwise() += Y_mean;
            
            //[2.5] Apply transformation
            Eigen::Matrix3Xf X_trans_new = transformation_temp*X_trans;

            // [2.6] is use transfom
            // Eigen::VectorXf point_error = (X_trans_new-Y_permutated).colwise().norm();
            // double err = point_error.transpose()* w_normalized.asDiagonal() * point_error;
            // if(err > match_error+euc_fitness_eps)
            // {               
            //     LOG(INFO) << "error " << match_error << "increase to " << err << " so break ";
            //     break;
            // }   
            // else
            // {
            if(transformation_temp.translation().norm() < trans_eps) break;
            if(fabs((acos(transformation_temp.linear().trace()-1.f) / 2.f)) < trans_eps) break;
                
            X_trans = X_trans_new;
            transformation = transformation_temp*transformation;
                // match_error = err;
            // }    

            //[3] Stopping criteria
            // double stop1 = (X_trans-Xo1).colwise().norm().maxCoeff();
            // Xo1 = X_trans;
            // if(stop1 < trans_eps) break;
        // }
        //[4] Stopping criteria
        // float translation_magnitude = transformation.translation().norm() - transformation_start.translation().norm();
        // if(translation_magnitude < trans_eps) break;
        // float rotation_magnitude = fabs(acos(transformation.linear().trace()-1.f / 2.f)) - 
        //                             fabs(acos(transformation_start.linear().trace()-1.f / 2.f));
        // if(rotation_magnitude < trans_eps) break;
        
        // double stop2 = (X_trans-Xo2).colwise().norm().maxCoeff();
        // Xo2 = X_trans;
        // if(stop2 < trans_eps) break;
    }

    X = transformation*X;
    return transformation.matrix();
}

static Eigen::Matrix4f point2plane(Eigen::Matrix3Xf& X, Eigen::Matrix3Xf& Y, Eigen::Matrix3Xf& N, int max_iter, float trans_eps, float max_corr_dist = 0.1)
{
    typedef Eigen::Matrix<float, 6, 6> Matrix66;
    typedef Eigen::Matrix<float, 6, 1> Vector6;
    typedef Eigen::Block<Matrix66, 3, 3> Block33;
        
    KDTree kdtree(Y);

    // Buffers
    Eigen::Matrix3Xf Y_permutated = Eigen::Matrix3Xf::Zero(3, X.cols());
    Eigen::Matrix3Xf Y_normals = Eigen::Matrix3Xf::Zero(3, X.cols());
    Eigen::VectorXf W = Eigen::VectorXf::Zero(X.cols());
    Eigen::Matrix3Xf Xo1 = X;
    Eigen::Matrix3Xf Xo2 = X;
    Eigen::Affine3f transformation;

    // ICP
    for(int icp=0; icp<max_iter; ++icp) {
        // Find closest point
        #pragma omp parallel for
        for(int i=0; i<X.cols(); ++i) {
            int id = kdtree.closest(X.col(i).data());
            Y_permutated.col(i) = Y.col(id);
            Y_normals.col(i) = Y_normals.col(id);
        }
        
        // Computer rotation and translation
        for(int outer=0; outer<100; ++outer) {
            // Compute weights
            W = (Y_normals.array()*(X-Y_permutated).array()).colwise().sum().abs().transpose();
            for(int i=0; i<W.rows(); ++i) {
                if(W(i) > max_corr_dist) 
                    W(i) = 0.0;
                else 
                    W(i) = std::pow((1.0 - std::pow(W(i)/max_corr_dist,2.0)), 2.0);
            }
            // Normalize weight vector
            Eigen::VectorXf w_normalized = W/W.sum();
            // Rotation and translation update

            // De-mean
            Eigen::Vector3f X_mean;
            for(int i=0; i<3; ++i)
                X_mean(i) = (X.row(i).array()*w_normalized.transpose().array()).sum();
            X.colwise() -= X_mean;
            Y_permutated.colwise() -= X_mean;
            // Prepare LHS and RHS
            Matrix66 LHS = Matrix66::Zero();
            Vector6 RHS = Vector6::Zero();
            Block33 TL = LHS.topLeftCorner<3,3>();
            Block33 TR = LHS.topRightCorner<3,3>();
            Block33 BR = LHS.bottomRightCorner<3,3>();
            Eigen::MatrixXf C = Eigen::MatrixXf::Zero(3,X.cols());
            #pragma omp parallel
            {
                #pragma omp for
                for(int i=0; i<X.cols(); i++) {
                    C.col(i) = X.col(i).cross(Y_normals.col(i));
                }
                #pragma omp sections nowait
                {
                    #pragma omp section
                    for(int i=0; i<X.cols(); i++) TL.selfadjointView<Eigen::Upper>().rankUpdate(C.col(i), W(i));
                    #pragma omp section
                    for(int i=0; i<X.cols(); i++) TR += (C.col(i)*Y_normals.col(i).transpose())*W(i);
                    #pragma omp section
                    for(int i=0; i<X.cols(); i++) BR.selfadjointView<Eigen::Upper>().rankUpdate(Y_normals.col(i), W(i));
                    #pragma omp section
                    for(int i=0; i<C.cols(); i++) {
                        double dist_to_plane = -((X.col(i) - Y_permutated.col(i)).dot(Y_normals.col(i)))*W(i);
                        RHS.head<3>() += C.col(i)*dist_to_plane;
                        RHS.tail<3>() += Y_normals.col(i)*dist_to_plane;
                    }
                }
            }
            LHS = LHS.selfadjointView<Eigen::Upper>();
            // Compute transformation
            Eigen::Affine3f transformation;
            Eigen::LDLT<Matrix66> ldlt(LHS);
            RHS = ldlt.solve(RHS);
            transformation  = Eigen::AngleAxisf(RHS(0), Eigen::Vector3f::UnitX()) *
                              Eigen::AngleAxisf(RHS(1), Eigen::Vector3f::UnitY()) *
                              Eigen::AngleAxisf(RHS(2), Eigen::Vector3f::UnitZ());
            transformation.translation() = RHS.tail<3>();
            // Apply transformation
            X = transformation*X;
            // Re-apply mean
            X.colwise() += X_mean;
            Y_permutated.colwise() += X_mean;
            // Stopping criteria
            double stop1 = (X-Xo1).colwise().norm().maxCoeff();
            Xo1 = X;
            if(stop1 < trans_eps) break;
        }
        // Stopping criteria
        double stop2 = (X-Xo2).colwise().norm().maxCoeff() ;
        Xo2 = X;
        if(stop2 < trans_eps) break;
    }
    return transformation.matrix();
}



} // namespace icp
}

#endif // !LIDAR_LOCALIZATION_MODELS_ICP_ICP_HPP_



