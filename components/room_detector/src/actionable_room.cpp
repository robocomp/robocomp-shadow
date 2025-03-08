//
// Created by pbustos on 9/12/24.
//

#include "actionable_room.h"

#include <iosfwd>
#include <optimiser.h>
#include <tuple>
#include <vector>
#include "pch.h"

namespace rc
{
    ActionableRoom::ActionableRoom()
    {
        corner_buffer.set_capacity(56);
        corner_robot_buffer.set_capacity(56);
        error_buffer.set_capacity(25);
    }

    bool ActionableRoom::initialize(const ActionableRoom &progenitor,
                                    const std::vector<ActionableRoom> &actionables,
                                    QGraphicsPolygonItem *robot_draw_)
    {
        // the parent is another room, so we get its corner_buffer and pose to propose a different room that captures better the reality
        const auto &[polys, _, votes] = rc::dbscan(progenitor.get_corner_buffer(), 100, 3);
        if (const auto viable = rc::Room_Detector::sample_room_concept(polys, votes, progenitor.get_robot_pose()); viable.has_value())
        {
            define_actionable_color(robot_draw_->brush().color());
            robot_draw = robot_draw_;
            robot_pose = std::get<2>(viable.value());
            room = std::get<0>(viable.value());
            room_draw = new QGraphicsPolygonItem(room.get_qt_polygon());
            corner_buffer.clear();   // since they were defined it the progenitor's frame
            corner_robot_buffer.clear();   // since they were defined it the progenitor's frame
            robot_draw->setPos(robot_pose.translation().x(), robot_pose.translation().y());
            robot_draw->setRotation(qRadiansToDegrees(Eigen::Rotation2Df(robot_pose.rotation()).angle()));
            if (std::ranges::find(actionables, *this) == actionables.end())
                return true;
        }
        return false;
    }

    bool ActionableRoom::initialize(const ActionableOrigin &progenitor,
                                    const std::vector<ActionableRoom> &actionables,
                                    QGraphicsPolygonItem *robot_draw_)
    {
        const auto &[polys, _, votes] = rc::dbscan(progenitor.get_corner_buffer(), 100, 3);
        if (const auto viable = rc::Room_Detector::sample_room_concept(polys, votes, progenitor.get_robot_pose()); viable.has_value())
        {
            define_actionable_color(robot_draw_->brush().color());
            robot_draw = robot_draw_;
            corner_buffer = progenitor.get_corner_buffer();
            corner_robot_buffer = progenitor.get_corner_robot_buffer();
            robot_pose = std::get<2>(viable.value());
            room = std::get<0>(viable.value());
            room_draw = new QGraphicsPolygonItem(room.get_qt_polygon());
            return true;
        }
        return false;
    }

    bool ActionableRoom::check_viability()
    {
        // if room_detector.sample_room_concept returns a room, then the robot is viable
        const auto &[polys, _, votes] = rc::dbscan(corner_buffer, 100, 3);
        if (const auto viable = rc::Room_Detector::sample_room_concept(polys, votes, this->robot_pose); viable.has_value())
        {
            robot_pose = std::get<2>(viable.value());
            room = std::get<0>(viable.value());
            corner_buffer.clear();   // since they were defined it the progenitor's frame
            corner_robot_buffer.clear();   // since they were defined it the progenitor's frame
            robot_draw->setPos(robot_pose.translation().x(), robot_pose.translation().y());
            robot_draw->setRotation(qRadiansToDegrees(Eigen::Rotation2Df(robot_pose.rotation()).angle()));
            return true;
        }
        qDebug() << __FUNCTION__ << "Room is not viable according to sample_room_concept";
        return false;
    }

    Match ActionableRoom::project_corners(const Corners &corners, long lidar_timestamp)
    {
        // get nominal corners and project them to the robot sensor space. Match them with the measurement corners
        std::vector<Eigen::Vector3d> corners_global, corners_global_past;    // for measured corners transformed to global frame

        const auto corners_by_timestamp = get_corners_by_timestamp(lidar_timestamp);
        for(const auto &[v, p, timestamp]: corners)
        {
            // convert corner to homogeneous coordinates
            Eigen::Vector3d cc;
            cc << Eigen::Vector2d{p.x(), p.y()}, 1.0;
            // project the corner to room frame
            const auto &q = robot_pose.matrix() * cc;  // Rx + T  (from robot RS to global RS)
//            qDebug() << "Act corner" << q.x() << q.y();
            // store corners in local storage to facilitate procreation of new offspring
            corner_buffer.push_back(std::make_tuple(v, QPointF(q.x(), q.y()), lidar_timestamp));
            corner_robot_buffer.push_back(std::make_tuple(v, QPointF(p.x(), p.y()), lidar_timestamp));
            corners_global.emplace_back(q);
        }

        for(const auto &p: corners_by_timestamp)
        {
            // convert corner to homogeneous coordinates
            // store corners in local storage to facilitate procreation of new offspring
            const auto &q_past = robot_pose.matrix() * p;  // Rx + T  (from robot RS to global RS)
            corners_global_past.emplace_back(q_past);
//            qDebug() << "Past corner" << q_past.x() << q_past.y();
        }
        // match measured in room and nominal
        const auto nominal_corners_3d = room.get_corners_3d();

        Match match = rc::hungarian(corners_global, nominal_corners_3d, robot_pose);
        Match match_past = rc::hungarian(corners_global_past, nominal_corners_3d, robot_pose);
        //qDebug() << __FUNCTION__ << "Match size act: " << match.size() << "Match size past: " << match_past.size();

        // Set the error based on the computed values
        double total_error_act = compute_total_error(match);
        double total_error = compute_total_error(match_past);
        this->set_error(std::min(total_error_act, total_error));

        const auto &[x, y, angle] = optimise(match, robot_pose);
        robot_pose = Eigen::Translation2d(x, y) * Eigen::Rotation2Dd(angle);
        return match_past;
    }

    double ActionableRoom::compute_total_error(const Match &matches)
    {
        if (matches.empty()) {
            return std::numeric_limits<double>::max();
        }
        return std::accumulate(matches.begin(), matches.end(), 0.0,
                               [](double acc, const auto& m) { return acc + std::get<4>(m); });
    }


    Eigen::Affine2d ActionableRoom::update_robot_pose(const Eigen::Affine2d &pose)
    {
        // does nothing
        return robot_pose;
    }
    Eigen::Vector3d ActionableRoom::update_target(const LidarPoints &points)
    {
        return robot_pose.inverse() * Eigen::Vector3d{0.0, 0.0, 1.0};  // center of the room in robot's reference frame
    }

    void ActionableRoom::remove_robot_draw(QGraphicsScene *scene)
    {
        if (robot_draw != nullptr)
         { scene->removeItem(robot_draw); delete robot_draw;}
    }

    void ActionableRoom::set_robot_opacity(float opacity)
    {
        robot_draw->setOpacity(opacity);
    }

    void ActionableRoom::define_actionable_color(const QColor &color)
    {
        this->color = color;
    }

    void ActionableRoom::set_error(double error)
    {
        error_buffer.push_back(error);
        this->error = error;
    }

    std::vector<Eigen::Vector3d> ActionableRoom::get_corners_3d() const
    {
        return room.get_corners_3d();
    }

    Eigen::Affine2d ActionableRoom::get_robot_pose() const
    {
        return robot_pose;
    }

    Eigen::Affine2d ActionableRoom::get_robot_pose_in_meters() const
    {
        // Convert the robot pose to meters
        return Eigen::Translation2d(robot_pose.translation() / 1000.0) * robot_pose.rotation();
    }

    QGraphicsItem * ActionableRoom::get_robot_draw() const
    {
        return robot_draw;
    }

    QGraphicsItem * ActionableRoom::get_room_draw() const
    {
        return room_draw;
    }

    QColor ActionableRoom::get_actionable_color() const
    {
        return color;
    }

    double ActionableRoom::get_error() const
    {
        return error;
    }

    double ActionableRoom::get_buffer_error() const
    {
        if (error_buffer.empty()) return 0.0; // Si el buffer está vacío, devuelve 0
        return std::accumulate(error_buffer.begin(), error_buffer.end(), 0.0) / static_cast<float>(error_buffer.size());
    }

    void ActionableRoom::compute_prediction_fitness(double eps)
    {
        this->value = 1.0 / (1.0 + (eps * get_buffer_error()));
        set_robot_opacity(value);
    }

    rc::Room ActionableRoom::get_room() const
    {
        return room;
    }

    void ActionableRoom::set_energy(double energy_)
    {
       energy = energy_;
    }

    double ActionableRoom::get_energy() const
    {
        return energy;
    }

    double ActionableRoom::get_value() const
    {
        return value;
    }
    boost::circular_buffer<Corner> ActionableRoom::get_corner_buffer() const
    {
        return corner_buffer;
    }

    std::vector<Eigen::Vector3d> ActionableRoom::get_corners_by_timestamp(long timestamp) const
    {
        if (corner_robot_buffer.empty())
        {
            qDebug() << "Empty buffer. Returning...";
            return {};
        }

        std::vector<Eigen::Vector3d> result;

        long closestTimestamp = std::get<2>(corner_robot_buffer.front()); // Inicializar con el primer timestamp
        long minDifference = std::abs(closestTimestamp - timestamp);

        for (const auto& [_, point, corner_timestamp] : corner_robot_buffer) {
            long diff = std::abs(corner_timestamp - timestamp);
            if (diff < minDifference) {
                // Nuevo timestamp más cercano encontrado, actualizar
                minDifference = diff;
                closestTimestamp = corner_timestamp;

                // Limpiar el resultado y agregar el nuevo elemento
                result.clear();
                result.push_back(Eigen::Vector3d{point.x(), point.y(), 1});
            } else if (diff == minDifference) {
                // Si el timestamp es igual de cercano, agregar al resultado
                result.push_back(Eigen::Vector3d{point.x(), point.y(), 1});
            }
        }
        return result;
    }

    bool ActionableRoom::operator==(const ActionableRoom &other) const
    {
         if(get_room().get_rotation() == other.get_room().get_rotation())
            return fabs(other.get_room().get_depth() - get_room().get_depth()) < 300 and
                   fabs(other.get_room().get_width() - get_room().get_width()) < 300;

        return fabs(other.get_room().get_depth() - get_room().get_width()) < 300 and
               fabs(other.get_room().get_width() - get_room().get_depth()) < 300;
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