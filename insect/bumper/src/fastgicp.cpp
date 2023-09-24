//
// Created by pbustos on 23/09/23.
//

#include "fastgicp.h"

FastGICP::FastGICP()
{
    fgicp.setMaxCorrespondenceDistance(1.0);
    fgicp.setNumThreads(8);
    trajectory = pcl::PointCloud<pcl::PointXY>::Ptr(new pcl::PointCloud<pcl::PointXY>());
}

//pcl::PointCloud<pcl::PointXY>::Ptr FastGICP::align(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud)
//{
//    if(source_cloud->empty())
//    { std::cout << "Empty cloud. Returning" << std::endl; return {};}
//
//    // remove invalid points around origin
//    source_cloud->erase(
//            std::remove_if(source_cloud->begin(), source_cloud->end(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
//            source_cloud->end());
////    target_cloud->erase(
////            std::remove_if(target_cloud->begin(), target_cloud->end(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
////            target_cloud->end());
//
//    // downsampling
//    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
//    voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);
//
//    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
//    voxelgrid.setInputCloud(source_cloud);
//    voxelgrid.filter(*filtered);
//    source_cloud = filtered;
//
//    // align
//    if(first_time)
//    {
//        fgicp.setInputTarget(source_cloud);
//        poses.resize(1);
//        trajectory->clear();
//        poses[0].setIdentity();
//        first_time = false;
//    }
//    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
//    //auto t1 = std::chrono::high_resolution_clock::now();
//    fgicp.setInputSource(source_cloud);
//    fgicp.align(*aligned);
//    fgicp.swapSourceAndTarget();
//
//    // accumulate pose
//    poses.emplace_back(poses.back() * fgicp.getFinalTransformation().cast<double>());
//    trajectory->emplace_back(pcl::PointXY(poses.back()(0, 3), poses.back()(1, 3)));
//    //auto t2 = std::chrono::high_resolution_clock::now();
//    //double fitness_score = fgicp.getFitnessScore();
//    //double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
//    //std::cout << "multi:" << single << "[msec] " << "source:" << source_cloud->size()
//    //    << "[pts] score:" << fitness_score << " length: " << trajectory->size() << std::endl;;
//
//    return trajectory;
//}

Eigen::Isometry3d FastGICP::align(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud)
{
    if(source_cloud->empty())
    { std::cout << "Empty cloud. Returning" << std::endl; return {};}

    // remove invalid points around origin
    source_cloud->erase(
            std::remove_if(source_cloud->begin(), source_cloud->end(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
            source_cloud->end());
//    target_cloud->erase(
//            std::remove_if(target_cloud->begin(), target_cloud->end(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
//            target_cloud->end());

    // downsampling
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
    voxelgrid.setInputCloud(source_cloud);
    voxelgrid.filter(*filtered);
    source_cloud = filtered;

    // align
    if(first_time)
    {
        fgicp.setInputTarget(source_cloud);
        poses.resize(1);
        trajectory->clear();
        poses[0].setIdentity();
        first_time = false;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
    //auto t1 = std::chrono::high_resolution_clock::now();
    fgicp.setInputSource(source_cloud);
    fgicp.align(*aligned);
    fgicp.swapSourceAndTarget();

    // accumulate pose
    poses.emplace_back(poses.back() * fgicp.getFinalTransformation().cast<double>());
    //auto t2 = std::chrono::high_resolution_clock::now();
    //double fitness_score = fgicp.getFitnessScore();
    //double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
    //std::cout << "multi:" << single << "[msec] " << "source:" << source_cloud->size()
    //    << "[pts] score:" << fitness_score << " length: " << trajectory->size() << std::endl;;

    return poses.back();
}

void FastGICP::reset()
{
    first_time = true;
}


//#ifdef USE_VGICP_CUDA
//  std::cout << "--- ndt_cuda (P2D) ---" << std::endl;
//  fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ> ndt_cuda;
//  ndt_cuda.setResolution(1.0);
//  ndt_cuda.setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
//
//  std::cout << "--- ndt_cuda (D2D) ---" << std::endl;
//  ndt_cuda.setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
//
//  std::cout << "--- vgicp_cuda (parallel_kdtree) ---" << std::endl;
//  fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ> vgicp_cuda;
//  vgicp_cuda.setResolution(1.0);
//  // vgicp_cuda uses CPU-based parallel KDTree in covariance estimation by default
//  // on a modern CPU, it is faster than GPU_BRUTEFORCE
//  // vgicp_cuda.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE);
//
//  std::cout << "--- vgicp_cuda (gpu_bruteforce) ---" << std::endl;
//  // use GPU-based bruteforce nearest neighbor search for covariance estimation
//  // this would be a good choice if your PC has a weak CPU and a strong GPU (e.g., NVIDIA Jetson)
//  vgicp_cuda.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
//
//  std::cout << "--- vgicp_cuda (gpu_rbf_kernel) ---" << std::endl;
//  // use RBF-kernel-based covariance estimation
//  // extremely fast but maybe a bit inaccurate
//  vgicp_cuda.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL);
//  // kernel width (and distance threshold) need to be tuned
//  vgicp_cuda.setKernelWidth(0.5);
//
//#endif