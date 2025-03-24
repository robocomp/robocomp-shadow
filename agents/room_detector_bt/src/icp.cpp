//
// Created by robolab on 3/14/24.
//

#include "icp.h"

icp::icp(const std::vector<Eigen::Vector2d>& source_points, const std::vector<Eigen::Vector2d>& target_points)
        : source_points_(source_points), target_points_(target_points)
{
}

// set source points and target points
void icp::setPoints(const std::vector<Eigen::Vector2d>& source_points, const std::vector<Eigen::Vector2d>& target_points)
{
    source_points_ = source_points;
    target_points_ = target_points;
}

Eigen::Matrix2d icp::rotation() const {
    return R_;
}

Eigen::Vector2d icp::translation() const {
    return t_;
}

Eigen::Vector2d icp::transformPoint(const Eigen::Vector2d& point) const {
    return R_ * point + t_;
}

void icp::align()
{
//    //print source and target points
//    std::cout << "Source points: " << std::endl;
//    for (const auto& point : source_points_) {
//        std::cout << point << std::endl;
//    }
//
//    //print target points
//    std::cout << "Target points: " << std::endl;
//    for (const auto& point : target_points_) {
//        std::cout << point << std::endl;
//    }

    // Inicializar R y t como una matriz identidad y un vector cero respectivamente
    R_ = Eigen::Matrix2d::Identity();
    t_ = Eigen::Vector2d::Zero();

    // Iteraciones de ICP
    const int max_iterations = 10000;
    for (int iter = 0; iter < max_iterations; ++iter)
    {
        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> correspondences;
        // Asociar cada punto de origen con el punto más cercano en el conjunto de puntos objetivo
        std::vector<Eigen::Vector2d> source_points_copy = source_points_;
        std::vector<Eigen::Vector2d> target_points_copy = target_points_;

        for (auto source_iter = source_points_copy.begin(); source_iter != source_points_copy.end();)
        {
            double min_distance = std::numeric_limits<double>::max();
            Eigen::Vector2d closest_point = *source_iter;
            auto target_iter = target_points_copy.begin();
            auto closest_target_iter = target_iter;

            // Encontrar el punto más cercano en el conjunto de puntos objetivo
            while (target_iter != target_points_copy.end())
            {
                if (double d = (*source_iter - *target_iter).norm(); d < min_distance)
                {
                    min_distance = d;
                    closest_point = *target_iter;
                    closest_target_iter = target_iter;
                }
                ++target_iter;
            }

            // Almacenar la correspondencia encontrada
            correspondences.push_back({*source_iter, closest_point});  // TODO: Cambiar a emplace_back

            // Eliminar los puntos correspondientes de sus vectores originales
            source_iter = source_points_copy.erase(source_iter);
            target_points_copy.erase(closest_target_iter);
        }

        // Calcular el centroide de cada conjunto de puntos
        Eigen::Vector2d centroid_source = Eigen::Vector2d::Zero();
        Eigen::Vector2d centroid_target = Eigen::Vector2d::Zero();
        for (const auto& correspondence : correspondences)
        {
            centroid_source += correspondence.first;
            centroid_target += correspondence.second;
        }
        centroid_source /= source_points_.size();
        centroid_target /= target_points_.size();

        // Construir las matrices de covarianza
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        for (const auto& correspondence : correspondences)
        {
            Eigen::Vector2d dx = correspondence.first - centroid_source;
            Eigen::Vector2d cx = correspondence.second - centroid_target;
            H += dx * cx.transpose();
        }

        // Calcular la matriz de rotación usando SVD
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix2d V = svd.matrixV();
        Eigen::Matrix2d U = svd.matrixU();
        Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
        S(1, 1) = (V * U.transpose()).determinant();
        R_ = V * S * U.transpose();

        // Calcular el vector de traslación
        t_ = centroid_target - R_ * centroid_source;

        // Transformar los puntos de origen usando la transformación encontrada
        std::vector<Eigen::Vector2d> transformed_source_points;
        for (const auto& source_point : source_points_)
            transformed_source_points.push_back(transformPoint(source_point));

        // Calcular el error de alineación
        double error = 0;
        for (size_t i = 0; i < transformed_source_points.size(); ++i)
            error += (transformed_source_points[i] - correspondences[i].second).norm();

//        std::cout << "Current iter::" << iter << std::endl;

        // Salir si el error es suficientemente pequeño
        if (error < 0.001) {
            std::cout << "Converged in " << iter << " iterations" << std::endl;
            break;
        }
    }
}

void icp::drawPoints() const
{
    cv::Mat img(500, 500, CV_8UC3, cv::Scalar(255, 255, 255));

    // Dibujar puntos fuente
    for (const auto& point : source_points_) {
        cv::circle(img, cv::Point(point.x() * 100 + 250, point.y() * 100 + 250), 5, cv::Scalar(0, 0, 255), -1);
        cv::putText(img, std::to_string(point.x()) + " " + std::to_string(point.y()), cv::Point(point.x() * 100 + 260, point.y() * 100 + 250),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
    }

    // Dibujar puntos objetivo
    for (const auto& point : target_points_) {
        cv::circle(img, cv::Point(point.x() * 100 + 250, point.y() * 100 + 250), 5, cv::Scalar(0, 255, 0), 10);
        cv::putText(img, std::to_string(point.x()) + " " + std::to_string(point.y()), cv::Point(point.x() * 100 + 260, point.y() * 100 + 150),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
    }

    // Dibujar puntos transformados
    for (const auto& point : source_points_) {
        auto transformed_point = transformPoint(point);
        cv::circle(img, cv::Point(transformed_point.x() * 100 + 250, transformed_point.y() * 100 + 250), 3, cv::Scalar(255, 0, 0), -1);
        cv::putText(img, std::to_string(transformed_point.x()) + " " + std::to_string(transformed_point.y()), cv::Point(transformed_point.x() * 100 + 260, transformed_point.y() * 100 + 200),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
    }

    cv::imshow("Source, Target and Transformed Points", img);
    cv::waitKey(0);
}