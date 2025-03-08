//
// Created by pbustos on 9/12/24.
//

#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <vector>
#include <QPointF>
#include <QGraphicsPolygonItem>
#include <Eigen/Dense>
#include <boost/circular_buffer.hpp>


enum class STATE
{
    START_SEARCH, MOVE_TO_CENTER, AT_CENTER, MOVE_ODOM_FRAME_TO_TARGET, MOVE_PHANTOMS_TO_TARGET,
    SELECT_NEW_TARGET, INITIAL_ACCUMULATION, MOVE_AROUND_AFFORDANCE, IDLE, INITIALIZE, SAMPLE_ROOMS, PROCESS_ROOMS
};

using RetVal = std::tuple<STATE, double, double>;
using RobotSpeed = std::tuple<double, double>;
using Boost_Circular_Buffer = boost::circular_buffer<Eigen::Matrix<double, 3, Eigen::Dynamic>>;
using LidarPoints = std::vector<Eigen::Vector3d>;   // 3D points with projective coordinates
// types for the features
using Lines = std::vector<std::pair<int, QLineF>>;
using Par_lines = std::vector<std::pair<QLineF, QLineF>>;
using Corner = std::tuple<int, QPointF, long>;
using Corners =  std::vector<Corner>;
using All_Corners = std::vector<std::tuple<QPointF, QPointF, QPointF, QPointF>>;
using Features = std::tuple<Lines, Par_lines, Corners, All_Corners>;
using Center = std::pair<QPointF, int>;  // center of a polygon and number of votes

// nominal_corners, measurement_corners_in_room, norm of measurement_corner_in_robot, angle of measurement_corner_in_robot, error
using Match = std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, double, double, double>>;
using Target = Eigen::Vector3d;

// types for the features
using Lines = std::vector<std::pair<int, QLineF>>;
using Par_lines = std::vector<std::pair<QLineF, QLineF>>;
using All_Corners = std::vector<std::tuple<QPointF, QPointF, QPointF, QPointF>>;
using Features = std::tuple<Lines, Par_lines, Corners, All_Corners>;
using Center = std::pair<QPointF, int>;  // center of a polygon and number of votes



#endif //COMMON_TYPES_H
