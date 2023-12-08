//
// Created by pbustos on 2/12/22.
//

#ifndef FORCEFIELD_ROOM_DETECTOR_H
#define FORCEFIELD_ROOM_DETECTOR_H

#include <vector>
#include <abstract_graphic_viewer/abstract_graphic_viewer.h>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <QtCore>
#include "room.h"
#include "door_detector.h"
#include <cmath>
#include <qcustomplot/qcustomplot.h>
#include <ranges>

namespace rc
{
    class Room_Detector
    {
        using Lines = std::vector<std::pair<int, QLineF>>;
        using Par_lines = std::vector<std::pair<QLineF, QLineF>>;
        using Corners =  std::vector<std::tuple<int, QPointF>>;
        using All_Corners = std::vector<std::tuple<QPointF, QPointF, QPointF, QPointF>>;
        using Features = std::tuple<Lines, Par_lines, Corners, All_Corners>;


        public:
            void init(QCustomPlot *customPlot_);
            Room detect(const std::vector<std::vector<Eigen::Vector2f>> &lines, AbstractGraphicViewer *viewer=nullptr, bool draw_lines=false);
            Features compute_features(const std::vector<std::vector<Eigen::Vector2f>> &lines, AbstractGraphicViewer *viewer);

        private:
            Eigen::Vector3f estimate_room_sizes(const Eigen::Vector2f &room_center, std::vector<Eigen::Vector2f> &floor_line_cart) const;
            Lines get_hough_lines(std::vector<Eigen::Vector2f> &floor_line_cart) const;
            Par_lines get_parallel_lines(const  Lines &lines, const Eigen::Vector2f &estimated_size);
            Corners get_corners(Lines &elines);
            All_Corners get_rooms(const Eigen::Vector2f &estimated_size, const Corners &corners);
            void filter_lines_by_length(const Lines &lines, std::vector <Eigen::Vector2f> &floor_line_cart);

            // aux
            float euc_distance_between_points(const QPointF &p1, const QPointF &p2) const;
            QPointF get_most_distant_point(const QPointF &p, const QPointF &p1, const QPointF &p2) const;

            // draw
            //void draw_on_2D_tab(const std::vector<Eigen::Vector2f> &lines, QString color, int size, bool clean, AbstractGraphicViewer *viewer);
            void draw_par_lines_on_2D_tab(const Par_lines &par_lines, AbstractGraphicViewer *viewer);
            void draw_lines_on_2D_tab(const Lines &lines, AbstractGraphicViewer *viewer);
            void draw_corners_on_2D_tab(const Corners &corners, const std::vector<Eigen::Vector2f> &model_corners, AbstractGraphicViewer *viewer);
            void draw_triple_corners_on_2D_tab(const All_Corners &double_corners, QString color, AbstractGraphicViewer *viewer);

            // local data
            Room current_room;

            Eigen::Vector2f to_eigen(const QPointF &p) const;
            Eigen::Vector2f to_eigen(const cv::Point2f &p) const;
            QPointF to_qpointf(const cv::Point2f &p) const;

            // QCustomPlot
            QCustomPlot *custom_plot;
            QCPGraph *triple_sig, *corners_sig, *par_lines_sig;
            void draw_timeseries(float triple_val, float corners_val, float par_lines_val);
    };

} // rc

#endif //FORCEFIELD_ROOM_DETECTOR_H
