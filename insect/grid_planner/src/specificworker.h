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
#include "grid.h"
#include "doublebuffer/DoubleBuffer.h"
#include <Eigen/Eigen>
#include "abstract_graphic_viewer/abstract_graphic_viewer.h"
#include <fps/fps.h>
#include <timer/timer.h>

class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
        SpecificWorker(TuplePrx tprx, bool startup_check);
        ~SpecificWorker() override;
        bool setParams(RoboCompCommonBehavior::ParameterList params) override;
        void SegmentatorTrackingPub_setTrack (RoboCompVisualElements::TObject target) override;

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
        Eigen::Transform<double, 3, 1> robot_pose; // robot pose in a global reference frame computed with lidar odometry

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
            int LIDAR_LOW_DECIMATION_FACTOR = 1;
            int LIDAR_HIGH_DECIMATION_FACTOR = 1;
            QRectF GRID_MAX_DIM{-6000, -6000, 12000, 12000};
            float CARROT_DISTANCE = 400;   // mm
            float CARROT_ANGLE = M_PI_4 / 6.f;   // rad
            long PERIOD_HYSTERESIS = 2; // to avoid oscillations in the adjustment of the lidar thread period
            int PERIOD = 50;    // ms (20 Hz) for compute timer
            float MIN_ANGLE_TO_TARGET = 1.f;   // rad
            int MPC_HORIZON = 8;
            bool USE_MPC = true;
            unsigned int ELAPSED_TIME_BETWEEN_PATH_UPDATES = 3000;
            int NUM_PATHS_TO_SEARCH = 3;
            float MIN_DISTANCE_BETWEEN_PATHS = 500; // mm
        };
        Params params;

        // grid
        Grid grid;

        // FPS
        FPSCounter fps;

        // Timer
        rc::Timer<> clock;

        // Lidar Thread
        DoubleBuffer<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3f>> buffer_lidar_data;
        std::thread read_lidar_th;
        void read_lidar();

        // Target
        struct Target
        {
            bool active = false;
            bool global = false;  // defined in global coordinates
            bool completed = false;
            Eigen::Vector2f point = Eigen::Vector2f(0, 0);
            QPointF qpoint;
            void set(const QPointF &p, bool global_ = false)
            {
                point = Eigen::Vector2f(p.x(), p.y());
                qpoint = p;
                global = global_;
                active = true;
                completed = false;
            }
            void set(const Eigen::Vector2f &p, bool global_ = false)
            {
                point = p;
                global = global_;
                qpoint = QPointF(p.x(), p.y());
                active = true;
                completed = false;
            }
            void unset() { active = false; }

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
            }
            static Target invalid() { Target t; t.active=false; return t; };

        };
        DoubleBuffer<Target, Target> target_buffer;

        // Frechet distance calculus
        double frechet_distance(const std::vector<Eigen::Vector2f>& pathA, const std::vector<Eigen::Vector2f>& pathB);
        float max_distance(const std::vector<Eigen::Vector2f> &pathA, const std::vector<Eigen::Vector2f> &pathB); //approximats the frechet distance

        // Draw
        void draw_paths(const vector<std::vector<Eigen::Vector2f>> &paths, QGraphicsScene *scene, bool erase_only=false);
        void draw_lidar(const RoboCompLidar3D::TPoints &points, int decimate=1);
        void draw_subtarget(const Eigen::Vector2f &point, QGraphicsScene *scene, bool erase_only=false);
        void draw_global_target(const Eigen::Vector2f &point, QGraphicsScene *scene);
        void draw_path(const vector<Eigen::Vector2f> &path, QGraphicsScene *scene, bool erase_only=false);
        void draw_smoothed_path(const std::vector<Eigen::Vector2f> &path, QGraphicsScene *scene, bool erase_only=false);

        // Do some work
        Eigen::Transform<double, 3, 1> get_robot_pose();    // robot pose from external component
        RoboCompGridPlanner::TPoint get_carrot_from_path(const std::vector<Eigen::Vector2f> &path,
                                                         float threshold_dist, float threshold_angle); // get close point in current path
        Eigen::Vector2f  border_subtarget(const Eigen::Vector2f &target);
        Target transform_target_to_global_frame(const Eigen::Transform<double, 3, 1> &robot_pose, const Target &target);
        RoboCompGridPlanner::TPlan compute_line_of_sight_target(const Target &target);
        RoboCompGridPlanner::TPlan compute_plan_from_grid(const Target &target);
        void adapt_grid_size(const Target &target,  const RoboCompGridPlanner::Points &path);   // EXPERIMENTAL
        RoboCompGridPlanner::TPlan
        convert_plan_to_control(const RoboCompGridPlanner::TPlan &plan, const Target &target);

    // publish
        void send_and_publish_plan(RoboCompGridPlanner::TPlan plan);

    bool is_robot_at_target(const Target &target_, const Target &original_target_);
};

#endif
