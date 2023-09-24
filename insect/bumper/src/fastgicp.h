//
// Created by pbustos on 23/09/23.
// Lidar3D odometry using Fast_GICP    https://github.com/SMRT-AIST/fast_gicp/tree/master
//
// To use instantiate the class as a regular variable and call the method assign with the LiDar data.
// A trajectory will start immediately. Call reset() to reinitiate it to (0,0,0)
//

#ifndef BUMPER_FASTGICP_H
#define BUMPER_FASTGICP_H

#include <chrono>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/memory.h>

#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#ifdef USE_VGICP_CUDA
    #include <fast_gicp/ndt/ndt_cuda.hpp>
    #include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

class FastGICP
{
    public:
        FastGICP();
        //pcl::PointCloud<pcl::PointXY>::Ptr  align(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud);
        Eigen::Isometry3d align(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud);

        void reset();

    private:
        std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;
        pcl::PointCloud<pcl::PointXY>::Ptr  trajectory;
        bool first_time = true;

        fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> fgicp;

        // other options
        // fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
        // fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ> gicp;
        // fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ> gicp;
        // gicp.setResolution(1.0);
        // gicp.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL);
};


#endif //BUMPER_FASTGICP_H
