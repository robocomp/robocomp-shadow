//
// Created by pbustos on 2/12/22.
//

#ifndef FORCEFIELD_ROOM_DETECTOR_H
#define FORCEFIELD_ROOM_DETECTOR_H

#include <vector>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <QtCore>
#include "room.h"
#include <ranges>
#include "common_types.h"

// Room_Detector class is a class that detects rooms in a 2D space. It uses a set of Hough lines to detect the rooms.
// The class has a method detect that receives a set of lines and returns a Room object.
// The class has a method compute_features that receives a set of lines and returns a tuple with the following features:
// 1. A set of lines
// 2. A set of parallel lines
// 3. A set of corners
// 4. A set of rooms

namespace rc   // aka RoboComp
{
    class Room_Detector
    {
        public:
            static Features compute_features(const std::vector<Eigen::Vector2d> &line, QGraphicsScene *scene= nullptr);
            static Features compute_features(const std::vector<Eigen::Vector3d> &line, QGraphicsScene *scene= nullptr);
            [[nodiscard]] static std::optional<std::tuple<Room, std::vector<QPointF>, Eigen::Affine2d>>
                sample_room_concept(const std::vector<QPolygonF> &polys, const std::vector<unsigned long> &votes, const Eigen::Affine2d &robot_pose);

            static Eigen::Vector3d estimate_room_sizes(const Eigen::Vector2d &room_center, std::vector<Eigen::Vector2d> &floor_line_cart);
            static Lines get_hough_lines(std::vector<Eigen::Vector2d> &floor_line_cart, const Eigen::Vector2d &estimated_size);
            static Par_lines get_parallel_lines(const  Lines &lines, const Eigen::Vector2d &estimated_size);
            static Corners get_corners(Lines &elines);
            static All_Corners get_rooms(const Eigen::Vector2d &estimated_size, const Corners &corners);
            static void filter_lines_by_length(const Lines &lines, std::vector <Eigen::Vector2d> &floor_line_cart);

            // aux
            static double euc_distance_between_points(const QPointF &p1, const QPointF &p2);
            static QPointF get_most_distant_point(const QPointF &p, const QPointF &p1, const QPointF &p2);
            [[nodiscard]] static std::vector<Center> reorder_points_CCW(const std::vector<Center> &points);

            // draw
            void draw_par_lines_on_2D_tab(const Par_lines &par_lines, QGraphicsScene *scene);
            static void draw_lines_on_2D_tab(const Lines &lines, QGraphicsScene *scene);
            static void draw_corners_on_2D_tab(const Corners &corners, const std::vector<Eigen::Vector2d> &model_corners, QGraphicsScene *scene);
            void draw_triple_corners_on_2D_tab(const All_Corners &double_corners, QString color, QGraphicsScene *scene);

            // local data
            static Eigen::Vector2d to_eigen(const QPointF &p);
            static Eigen::Vector2d to_eigen(const cv::Point2d &p);
            static QPointF to_qpointf(const cv::Point2d &p);
    };

} // rc

#endif //FORCEFIELD_ROOM_DETECTOR_H
