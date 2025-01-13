/*
 *    Copyright (C) 2023 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
	\brief
	@author authorname
*/

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <genericworker.h>
#include "doublebuffer/DoubleBuffer.h"
#include <Eigen/Eigen>
#include "abstract_graphic_viewer/abstract_graphic_viewer.h"
#include <fps/fps.h>
#include <timer/timer.h>
#include <qcustomplot/qcustomplot.h>

class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
        SpecificWorker(TuplePrx tprx, bool startup_check);
        ~SpecificWorker() override;
        bool setParams(RoboCompCommonBehavior::ParameterList params) override;
        void SegmentatorTrackingPub_setTrack (RoboCompVisualElementsPub::TObject target);

    public slots:
        void compute() override;
        int startup_check();
        void initialize(int period) override;

    private:
        bool startup_check_flag;
        bool cancel_from_mouse = false;     // cancel current target from mouse right click

        //Graphics
        AbstractGraphicViewer *viewer;

        // Robot
        using RobotPose = std::pair<Eigen::Transform<double, 3, 1>, Eigen::Transform<double, 3, 1>>;
        // robot pose in a global reference frame computed with lidar odometry

        struct Params
        {
            float ROBOT_WIDTH = 460;  // mm
            float ROBOT_LENGTH = 480;  // mm
            float ROBOT_SEMI_WIDTH = ROBOT_WIDTH / 2.f;     // mm
            float ROBOT_SEMI_LENGTH = ROBOT_LENGTH / 2.f;    // mm
            float TILE_SIZE = 100;   // mm
            float MIN_DISTANCE_TO_TARGET = ROBOT_WIDTH / 2.f; // mm
            std::string LIDAR_NAME_LOW = "bpearl";
            std::string LIDAR_NAME_HIGH = "helios";
            float MAX_LIDAR_LOW_RANGE = 10000;  // mm
            float MAX_LIDAR_HIGH_RANGE = 10000;  // mm
            float MAX_LIDAR_RANGE = 10000;  // mm used in the grid
            int LIDAR_LOW_DECIMATION_FACTOR = 2;
            int LIDAR_HIGH_DECIMATION_FACTOR = 1;
            QRectF GRID_MAX_DIM{-6000, -6000, 12000, 12000};
            RoboCompGridder::TDimensions gdim;
            float CARROT_DISTANCE = 400;   // mm
            float CARROT_ANGLE = M_PI_4 / 6.f;   // rad
            long PERIOD_HYSTERESIS = 2; // to avoid oscillations in the adjustment of the lidar thread period
            int PERIOD = 100;    // ms (20 Hz) for compute timer
            float MIN_ANGLE_TO_TARGET = 0.1;   // rad
            int MPC_HORIZON = 8;
            bool USE_MPC = true;
            unsigned int ELAPSED_TIME_BETWEEN_PATH_UPDATES = 3000;
            int NUM_PATHS_TO_SEARCH = 3;
            float MIN_DISTANCE_BETWEEN_PATHS = 500; // mm
            bool DISPLAY = true;
            size_t MAX_PATH_STEPS = 1000;
            float TRACKING_DISTANCE_TO_TARGET = 1200; // mm

            // colors
            QColor TARGET_COLOR= {"orange"};
            QColor LIDAR_COLOR = {"green"};
            QColor PATH_COLOR = {"orange"};
            QColor SMOOTHED_PATH_COLOR = {"magenta"};
        };
        Params params;

        // FPS
        FPSCounter fps;
        int hz = 0;

        // Timer
        rc::Timer<> clock;

        // Target
        struct Target
        {
            bool active = false;
            bool global = false;  // defined in global coordinates
            bool completed = false;
            Eigen::Vector2f point = Eigen::Vector2f(0, 0);
            QPointF qpoint;
            Eigen::Vector2f original;    // original target in global coordinates
            bool new_target = false;   // true if target has been updated from outside
            bool is_being_tracked = false;   // true if target is being tracked
            void set(const QPointF &p, bool global_ = false)
            {
                point = Eigen::Vector2f(p.x(), p.y());
                qpoint = p;
                active = true;
            }
            void set(const Eigen::Vector2f &p, bool global_ = false)
            {
                point = p;
                qpoint = QPointF(p.x(), p.y());
                active = true;
            }
            void set_original(const Eigen::Vector2f &t) { original = t; }
            void set_new(bool v) { new_target = v; }
            bool is_new() const { return new_target;}
            void unset() { active = false; }
            void set_being_tracked(bool v) { is_being_tracked = v; }
            [[nodiscard]] bool is_tracked() const { return is_being_tracked; }
            [[nodiscard]] bool is_valid() const { return active; };
            [[nodiscard]] Eigen::Vector2f pos_eigen() const { return point; }
            [[nodiscard]] QPointF pos_qt() const { return qpoint; }
            [[nodiscard]] bool is_global() const { return global; }
            [[nodiscard]] bool is_completed() const { return completed; }
            [[nodiscard]] float angle_to_robot() const { return atan2(point.x(), point.y()); }
            [[nodiscard]] float distance_to_robot() const { return point.norm(); }
            [[nodiscard]] Eigen::Vector2f point_at_distance(float distance) const
            {
                Eigen::Vector2f result;
                return point.normalized() * distance;
            }
            void print() const
            {
                qInfo() << "Target: ";
                qInfo() << "    point: " << point.x() << " " << point.y();
                qInfo() << "    qpoint: " << qpoint.x() << " " << qpoint.y();
                qInfo() << "    global: " << global;
                qInfo() << "    active: " << active;
                qInfo() << "    dist to robot: " << point.norm();
                qInfo() << "    angle to robot: " << angle_to_robot();
                qInfo() << "    original: " << original.x() << " " << original.y();
                qInfo() << "    new: " << new_target;
                qInfo() << "    completed: " << completed;
            }
            static Target invalid() { Target t; t.active=false; return t; };
            Eigen::Vector2f get_original() const { return original; };

        };
        DoubleBuffer<Target, Target> target_buffer;
        std::vector<Eigen::Vector2f> current_path;

        // Frechet distance calculus
        float distance_between_paths(const std::vector<Eigen::Vector2f> &pathA, const std::vector<Eigen::Vector2f> &pathB); //approximats the frechet distance

        // Draw
        void draw_paths(const RoboCompGridder::TPaths &paths, QGraphicsScene *scene, bool erase_only=false);
        void draw_global_target(const Eigen::Vector2f &point, QGraphicsScene *scene, bool erase_only=false);
        void draw_path(const vector<Eigen::Vector2f> &path, QGraphicsScene *scene, bool erase_only=false);
        void draw_smoothed_path(const RoboCompGridPlanner::Points &path, QGraphicsScene *scene, const QColor &color, bool erase_only=false);
        void draw_lidar(const std::vector<Eigen::Vector3f> &points, int decimate=1);
        void draw_point_color(const Eigen::Vector2f &point, QGraphicsScene *scene, bool erase_only, QColor color);

    // Lidar Thread
        DoubleBuffer<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3f>> buffer_lidar_data;
        std::thread read_lidar_th;
        void read_lidar();

        // Do some work
        std::optional<pair<Eigen::Transform<double, 3, 1>, Eigen::Transform<double, 3, 1>>> get_robot_pose_and_change();    // robot pose from external component
        RoboCompGridPlanner::TPoint get_carrot_from_path(const std::vector<Eigen::Vector2f> &path,
                                                         float threshold_dist, float threshold_angle); // get close point in current path
        Eigen::Vector2f  compute_closest_target_to_grid_border(const Eigen::Vector2f &target);
        Target transform_target_to_global_frame(const Eigen::Transform<double, 3, 1> &robot_pose, const Target &target);
        RoboCompGridder::Result compute_line_of_sight_target(const Target &target);
        RoboCompGridder::Result compute_plan_from_grid(const Target &target, const RobotPose &robot_pose_and_change);
        RoboCompGridPlanner::TPlan
        convert_plan_to_control(const RoboCompGridder::Result &res, const Target &target);
        bool robot_is_at_target(const Target &target_);
        void inject_ending_plan();
        RoboCompGridder::Result compute_path(const Eigen::Vector2f &source, const Target &target, const RobotPose &robot_pose_and_change);
        double path_length(const std::vector<Eigen::Vector2f> &path) const;

        // publish
        void send_and_publish_plan(const RoboCompGridPlanner::TPlan &plan);

        // state-machine
        enum class State {IDLE, COMPUTE, WAIT, STOP, ERROR};
        State state = State::IDLE;

        // QCustomPlot
        QCustomPlot custom_plot;
        QCPGraph *side_vel, *adv_vel, *track_dist;
        void draw_timeseries(float side, float adv, float track);

};

#endif
