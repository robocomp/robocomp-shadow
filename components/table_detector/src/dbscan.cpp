//
// Created by pbustos on 21/10/24.
//

#include "dbscan.h"
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <cppitertools/enumerate.hpp>
#include <cppitertools/range.hpp>
#include <vector>
#include <map>
#include <QVector2D>

namespace rc
{
    std::tuple<std::vector<QPolygonF>, std::vector<cv::RotatedRect>, std::vector<size_t>>
    dbscan(const std::vector<Eigen::Vector2d> &points, float eps, int min_points)
    {
        if(points.empty()) {std::cout << __FUNCTION__ << " No points" << std::endl; return {};}
        arma::mat arma_data(2, points.size());
        for (const auto &[i, p]: points | iter::enumerate)
        {
            arma_data(0, i) = p.x();
            arma_data(1, i) = p.y();
        }
        arma::Row<size_t> assignments;
        mlpack::dbscan::DBSCAN<> dbscan(eps, min_points);
//        mlpack::DBSCAN<> dbscan(eps, min_points);
        dbscan.Cluster(arma_data, assignments);
        std::map<size_t, std::vector<cv::Point2f>> clustersMap;
        for (const auto i: iter::range(assignments.n_elem))
        {
            size_t label = assignments[i];
            if (label != (size_t) -1)
                clustersMap[label].emplace_back(points[i].x(), points[i].y());  // -1 indicates noise
        }

        // compute polygons
        std::vector<QPolygonF> list_poly;
        std::vector<size_t> votes;
        std::vector<cv::Point2f> hull;
        std::vector<cv::RotatedRect> rects;
        for (const auto &pair: clustersMap)
        {
            // Calculate the convex hull of the cluster
            std::vector<cv::Point2f> hull;
            cv::convexHull(pair.second, hull);
            rects.emplace_back(cv::minAreaRect(pair.second));

            // Convert the convex hull to a QPolygonF
            QPolygonF poly;
            for (const auto &p: hull)
                poly << QPointF(p.x, p.y);
            list_poly.emplace_back(poly);
            votes.emplace_back(pair.second.size());
        }
        return {list_poly, rects, votes};
    };

    std::tuple<std::vector<QPolygonF>, std::vector<cv::RotatedRect>, std::vector<size_t>>
    dbscan(const std::vector<Eigen::Vector2f> &points, float eps, int min_points)
    {
        std::vector<Eigen::Vector2d> dpoints;
        std::ranges::transform(points, std::back_inserter(dpoints), [](const auto &p)
                { return Eigen::Vector2d(p.x(), p.y());});
        return dbscan(dpoints, eps, min_points);
    }

    std::tuple<std::vector<QPolygonF>, std::vector<cv::RotatedRect>, std::vector<size_t>>
          dbscan(const std::vector<Eigen::Vector3d> &points, float eps, int min_points)
    {
        std::vector<Eigen::Vector2d> dpoints;
        std::ranges::transform(points, std::back_inserter(dpoints), [](const auto &p)
                { return Eigen::Vector2d(p.x(), p.y());});
        return dbscan(dpoints, eps, min_points);
    }

    std::map<size_t, std::vector<Eigen::Vector3d>>
           dbscan_get_point_clusters(const std::vector<Eigen::Vector3d> &points, float eps, int min_points)
    {
        if(points.empty()) {std::cout << __FUNCTION__ << " No points" << std::endl; return {};}
        arma::mat arma_data(3, points.size());
        for (const auto &[i, p]: points | iter::enumerate)
        {
            arma_data(0, i) = p.x();
            arma_data(1, i) = p.y();
            arma_data(2, i) = p.z();
        }
        arma::Row<size_t> assignments;
//        mlpack::DBSCAN<> dbscan(eps, min_points);
        mlpack::dbscan::DBSCAN<> dbscan(eps, min_points);
        dbscan.Cluster(arma_data, assignments);
        std::map<size_t, std::vector<Eigen::Vector3d>> clustersMap;
        //qInfo() << __FUNCTION__ << "Number of assignments: " << assignments.n_elem << " Number of points: " << points.size();
        for (const auto i: iter::range(assignments.n_elem))
            if (size_t label = assignments[i]; label != static_cast<size_t>(-1))
                clustersMap[label].emplace_back(points[i].x(), points[i].y(), points[i].z());  // -1 indicates noise

        // Sort the clusters by size
        std::vector<std::pair<size_t, std::vector<Eigen::Vector3d>>> sortedClustersVec(clustersMap.begin(), clustersMap.end());
        std::ranges::sort(sortedClustersVec, [](const auto &a, const auto &b) {return a.second.size() > b.second.size();});

        // Reconstruct the map from the sorted vector
        std::map<size_t, std::vector<Eigen::Vector3d>> sortedClustersMap;
        for (const auto &[label, cluster] : sortedClustersVec)
            sortedClustersMap[label] = cluster;

        return sortedClustersMap;
    }

    // 1D DBSCAN
    template <typename T>
       std::map<size_t, std::pair<std::vector<T>, std::vector<size_t>>>  // map
            dbscan1D(const std::vector<T> &points, const float eps, const int min_points)
                requires std::is_floating_point_v<T> || std::is_integral_v<T>
    {
        if(points.empty()) {std::cout << __FUNCTION__ << " No points" << std::endl; return {};}
        arma::mat arma_data(1, points.size());
        for (const auto &[i, p]: points | iter::enumerate)
            arma_data(0, i) = static_cast<float>(p);

        arma::Row<size_t> assignments;
//        mlpack::DBSCAN<> dbscan(eps, min_points);
        mlpack::dbscan::DBSCAN<> dbscan(eps, min_points);

        dbscan.Cluster(arma_data, assignments);
        std::map<size_t, std::pair<std::vector<T>, std::vector<size_t>>> clustersMap;
        for (const auto i: iter::range(assignments.n_elem))
        {
            if (size_t label = assignments[i]; label != static_cast<size_t>(-1)) // -1 indicates noise
            {
                clustersMap[label].first.emplace_back(points[i]);
                clustersMap[label].second.emplace_back(i);
            }
        }
        return clustersMap;
    };

    std::tuple<std::vector<QPolygonF>, std::vector<cv::RotatedRect>, std::vector<size_t>>
    dbscan(const Corners &corners, float eps, int min_points)
    {
        std::vector<Eigen::Vector3d> points;
        std::ranges::transform(corners, std::back_inserter(points), [](const auto &c)
                { return Eigen::Vector3d{std::get<1>(c).x(), std::get<1>(c).y(), 1.0 };});
        return dbscan(points, eps, min_points);
    }

    std::tuple<std::vector<QPolygonF>, std::vector<cv::RotatedRect>, std::vector<size_t>>
    dbscan(const boost::circular_buffer<Corner> &corners, float eps, int min_points)
    {
        std::vector<Eigen::Vector3d> points;
        for (const auto &[_, p, timestamp]: corners)
            points.emplace_back(p.x(), p.y(), 1.0);
        return dbscan(points, eps, min_points);
    }

    /**
     * @brief Perform K-means clustering on a set of points and return the centroids of the clusters.
     *
     * @param points A vector of points to be clustered.
     * @return A tuple with the centroids of the clusters and the points in each cluster.
     */
    std::tuple<std::vector<Eigen::VectorXd>, std::vector<std::vector<Eigen::VectorXd>>>
            kmeans(const std::vector<Eigen::VectorXd> &points, const int number_clusters)
    {
        constexpr  size_t maxIterations = 1000;
        const std::string initMethod = "kmeans++";

        // convert points to armadillo data
        arma::mat arma_data(points[0].size(), points.size());
        for (size_t i = 0; i < points.size(); ++i)
            arma_data.col(i) = arma::vec(reinterpret_cast<arma::uword>(points[i].data()), points[i].size());

        // 2. Perform K-means clustering
//        mlpack::KMeans<> kmeans(maxIterations);
        mlpack::kmeans::KMeans<> kmeans(maxIterations);
        arma::Row<size_t> assignments;
        arma::mat centroids;
        kmeans.Cluster(arma_data, number_clusters, assignments, centroids);

        // 3. Convert assignments and centroids back-to Eigen vectors
        std::vector<Eigen::VectorXd> eigen_centroids(number_clusters);
        std::vector<std::vector<Eigen::VectorXd>> clustered_points(number_clusters);
        for (const auto &i : iter::range(number_clusters))
        {
            eigen_centroids[i].resize(static_cast<long>(centroids.col(i).n_rows));
            for (const auto &j : iter::range(eigen_centroids[i].size()))
                eigen_centroids[i](j) = static_cast<float>(centroids.col(i)(j));
        }
        for (const auto &[i, p] : iter::enumerate(points))
            clustered_points[assignments(i)].push_back(points[i]);

        return std::make_tuple(eigen_centroids, clustered_points);
    };

    // Explicit instantiation for long (add more types as needed) is required to avoid linker errors and keep
    // the code in .cpp
    template std::map<size_t, std::pair<std::vector<unsigned long>, std::vector<size_t>>>
    dbscan1D<unsigned long>(const std::vector<unsigned long>& points, const float eps, const int min_points);

};


