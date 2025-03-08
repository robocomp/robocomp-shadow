//
// Created by pbustos on 9/12/24.
//

#include "actionable_origin.h"
#include <optimiser.h>
#include "pch.h"

namespace rc
{
    ActionableOrigin::ActionableOrigin()
    {
        corner_buffer.set_capacity(56);
        corner_robot_buffer.set_capacity(56);
        error_buffer.set_capacity(25);
    }

    void ActionableOrigin::initialize(const rc::ActionableOrigin &parent,
                                      const std::tuple<QGraphicsPolygonItem*,
                                      QGraphicsEllipseItem*> &robot_drawing)
    {
        const auto &[robot_body, white_circle] = robot_drawing;
        this->robot_draw = robot_body;
        this->white_circle = white_circle;
        initialized = true;
    }

    Match ActionableOrigin::project_corners(const Corners &corners, long lidar_timestamp)
    {
        // projects measured corners to origing odometry frame and stores in circular buffer
        for(const auto &[v, p, _]: corners)
        {
            // convert corner to homogeneous coordinates
            Eigen::Vector3d cc;
            cc << Eigen::Vector2d{p.x(), p.y()}, 1.0;
            // project the corner to room frame
            const auto &q = robot_pose.matrix() * cc;  // Rx + T  (from robot RS to global RS)
            // store corners in local storage to facilitate procreation of new offspring
            corner_buffer.push_back(std::make_tuple(v, QPointF(q.x(), q.y()), lidar_timestamp));
            corner_robot_buffer.push_back(std::make_tuple(v, QPointF(p.x(), p.y()), lidar_timestamp));
        }
        return Match();
    }

    Eigen::Affine2d ActionableOrigin::update_robot_pose(const Eigen::Affine2d &pose)
    {
        robot_pose = pose;
        robot_draw->setPos(robot_pose.translation().x(), robot_pose.translation().y());
        robot_draw->setRotation(qRadiansToDegrees(Eigen::Rotation2Df(robot_pose.rotation()).angle()));
        return robot_pose;
    }
    Eigen::Vector3d ActionableOrigin::update_target(const LidarPoints &points)
    {
        return std::accumulate(points.begin(), points.end(), Eigen::Vector3d{0.0, 0.0, 0.0}) / static_cast<double>(points.size());
    }
    bool ActionableOrigin::ready_to_procreate() const
    {
      // In the intializer_actionable, this method checks if there are enough stable corners to start sampling
      // In the other actionables, this method checks if there is a large error in the projection of the corners
        return reproduce;
    }
    bool ActionableOrigin::check_viability()
    {
      // if room_detector.sample_room_concept returns a room, then the robot is viable
      const auto &[polys, _, votes] = rc::dbscan(corner_buffer, 100, 3);
      const auto viable = rc::Room_Detector::sample_room_concept(polys, votes, this->robot_pose);
      if (viable.has_value())
      {
          robot_pose = std::get<2>(viable.value());
          room = std::get<0>(viable.value());
          corner_buffer.clear();   // since they were defined it the progenitor's frame
          corner_robot_buffer.clear();   // since they were defined it the progenitor's frame
          robot_draw->setPos(robot_pose.translation().x(), robot_pose.translation().y());
          robot_draw->setRotation(qRadiansToDegrees(Eigen::Rotation2Df(robot_pose.rotation()).angle()));
          return true;
      }
      return false;
    }

    void ActionableOrigin::remove_robot_draw(QGraphicsScene *scene)
    {
        if (robot_draw != nullptr)
        {scene->removeItem(robot_draw); delete robot_draw;}
        // if (white_circle != nullptr)
        // { scene->removeItem(white_circle); delete white_circle;}
    }

    void ActionableOrigin::set_robot_opacity(float opacity)
    {
        robot_draw->setOpacity(opacity);
    }

    void ActionableOrigin::define_actionable_color(const QColor &color)
    {
        this->color = color;
    }

    void ActionableOrigin::set_error(double error)
    {
        error_buffer.push_back(error);
        this->error = error;
    }

    std::vector<Eigen::Vector3d> ActionableOrigin::get_corners_3d() const
    {
        return room.get_corners_3d();
    }

    Eigen::Affine2d ActionableOrigin::get_robot_pose() const
    {
        return robot_pose;
    }

    QGraphicsItem * ActionableOrigin::get_robot_draw() const
    {
        return robot_draw;
    }

    QColor ActionableOrigin::get_actionable_color() const
    {
        return color;
    }

    double ActionableOrigin::get_error() const
    {
        return error;
    }

    double ActionableOrigin::get_buffer_error() const
    {
        if (error_buffer.empty()) return 0.0; // Si el buffer está vacío, devuelve 0
        return std::accumulate(error_buffer.begin(), error_buffer.end(), 0.0) / static_cast<float>(error_buffer.size());
    }

    double ActionableOrigin::get_normalized_error(double eps) const
    {
        return 1.0 / (1.0 + (eps * get_buffer_error()));
    }

    rc::Room ActionableOrigin::get_room() const
    {
        return room;
    }

    void ActionableOrigin::set_energy(double energy_)
    {
       energy = energy_;
    }

    double ActionableOrigin::get_energy() const
    {
        return energy;
    }

    boost::circular_buffer<Corner> ActionableOrigin::get_corner_buffer() const
    {
        return corner_buffer;
    }

    boost::circular_buffer<Corner> ActionableOrigin::get_corner_robot_buffer() const
    {
        return corner_robot_buffer;
    }

    bool ActionableOrigin::operator==(const ActionableOrigin &other) const
    {
        return false;
    }
} // rc

//Eigen::Matrix<double, 2, Eigen::Dynamic> mat(3, points.size());
//for(const auto &[i, p] : points | iter::enumerate)
//    mat.col(i) = points[i].head(2);
//// remove outliers and junk points
//mat = mat.array().max(-10000.0).min(10000.0);
//mat = mat.array().isNaN().select(0.0, mat);
//// Calculate the mean of each row, aka the center of the room
//const Eigen::Vector2d center = mat.rowwise().mean();
//qDebug() << __FUNCTION__ << "Center frame: " << center.x() << center.y();
//// substract the mean to all points
//mat.colwise() -= center;
//// compute the covariance matrix
//const Eigen::Matrix2d cov = (mat * mat.adjoint()) / static_cast<double>(mat.cols() - 1);
//// compute the eigenvalues and eigenvectors, aka the dimensions of the room
//Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen_solver;
//eigen_solver.compute(cov);
//Eigen::Vector2d values = eigen_solver.eigenvalues().real().cwiseSqrt()*2.f;
//Eigen::Index i;
//values.maxCoeff(&i);
//Eigen::Vector2d max_vector = eigen_solver.eigenvectors().real().col(i);
//qDebug() << __FUNCTION__ << " START_SEARCH:" << values.x() << values.y() << atan2(max_vector.x(), max_vector.y()) << i;
//// compute center_frame
//center_frame = Eigen::Translation2d(center) * Eigen::Rotation2Dd(atan2(max_vector.y(), max_vector.x()));