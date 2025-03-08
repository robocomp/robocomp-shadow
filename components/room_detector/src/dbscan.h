//
// Created by pbustos on 21/10/24.
//

#ifndef PERSON_TRACKER_DBSCAN_H
#define PERSON_TRACKER_DBSCAN_H

#include <vector>
#include <map>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <cppitertools/enumerate.hpp>
#include "common_types.h"
#include <opencv2/opencv.hpp>

namespace rc
{
        /**
        * @brief Perform DBSCAN clustering on a set of points and return the clusters as polygons.
        * @return A pair containing the polygons of the clusters and the number of points in each cluster.
       */
        std::tuple<std::vector<QPolygonF>, std::vector<cv::RotatedRect>, std::vector<size_t>>
            dbscan(const std::vector<Eigen::Vector2d> &points, float eps, int min_points);

        /**
        * @brief Overloading to take float 2D points and return the clusters as polygons.
        * @return A pair containing the polygons of the clusters and the number of points in each cluster.
       */
        std::tuple<std::vector<QPolygonF>, std::vector<cv::RotatedRect>, std::vector<size_t>>
            dbscan(const std::vector<Eigen::Vector2f> &points, float eps, int min_points);

        /**
        * @brief Overloading to take double 3D points and return the 2D projected clusters as polygons.
        * @return A pair containing the polygons of the clusters and the number of points in each cluster.
        */
        std::tuple<std::vector<QPolygonF>, std::vector<cv::RotatedRect>, std::vector<size_t>>
            dbscan(const std::vector<Eigen::Vector3d> &points, float eps, int min_points);

        /**
          * @brief Clusters the points
          * @return A map of cluster label to points
          */
        std::map<size_t, std::vector<Eigen::Vector3d>>
            dbscan_get_point_clusters(const std::vector<Eigen::Vector3d> &points, float eps, int min_points);

        /**
        * @brief Overloading of the dbscan function to work with corners.
        * @param corners
        * @param eps
        * @param min_points
        * @return
        */
        std::tuple<std::vector<QPolygonF>, std::vector<cv::RotatedRect>, std::vector<size_t>>
            dbscan(const Corners &corners, float eps, int min_points);

        /**
        * @brief Overloading of the dbscan function to work with circular buffer of corners
        * @param corners
        * @param eps
        * @param min_points
        * @return
        */
        std::tuple<std::vector<QPolygonF>, std::vector<cv::RotatedRect>, std::vector<size_t>>
            dbscan(const boost::circular_buffer<Corner> &corners, float eps, int min_points);


        /**
        * @brief Perform DBSCAN clustering on a set of scalar values
        * @points A vector of scalar types to be clustered.
        * @return A pair containing the cluster and the number of points in each cluster.
        */
       template <typename T>
        std::map<size_t, std::pair<std::vector<T>, std::vector<size_t>>>  // map of cluster label to points and indices
            dbscan1D(const std::vector<T> &points, float eps, int min_points)
                requires std::is_floating_point_v<T> || std::is_integral_v<T>;

        /**
        * @brief Perform K-means clustering on a set of points and return the centroids of the clusters.
        * @param points A vector of points to be clustered.
        * @param number_clusters The number of clusters to form.
        * @return A tuple containing the centroids of the clusters and the points assigned to each cluster.
        */
        std::tuple<std::vector<Eigen::VectorXd>, std::vector<std::vector<Eigen::VectorXd>>>
            kmeans(const std::vector<Eigen::VectorXd> &points, const int number_clusters);

}
#endif //PERSON_TRACKER_DBSCAN_H
