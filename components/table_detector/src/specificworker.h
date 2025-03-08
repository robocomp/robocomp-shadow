/*
 *    Copyright (C) 2024 by YOUR NAME HERE
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
//#define HIBERNATION_ENABLED

#include <genericworker.h>
#include <protected_map.h>

#include "abstract_graphic_viewer/abstract_graphic_viewer.h"
#include <random>
#include "room.h"
#include <tuple>
#include <timer/timer.h>
#include <fps/fps.h>
#include <Eigen/Dense>
#include "qcustomplot/qcustomplot.h"
#include "actionable_room.h"
#include "actionable_thing.h"
#include "actionable_origin.h"
#include "common_types.h"
#include "protected_map.h"
#include <csignal>
#include <atomic>
#include <thread>
#include "viewer3D.h"

class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
        SpecificWorker(TuplePrx tprx, bool startup_check);
        ~SpecificWorker();
        bool setParams(RoboCompCommonBehavior::ParameterList params);

    public slots:
        void initialize(int period);
        void compute();
        void emergency();
        void restore();
        int startup_check();

        // Signal handler for SIGINT
        static void signal_handler(int signal)
        { if (signal == SIGINT)
                stop_flag.store(true);
        }

    signals:
        void updateRobotTransform(Eigen::Affine2d);
        void createRoomTransform();
        void updateRoomTransform();
        void createTableTransform();
        void updateTableTransform();

    private:
        bool startup_check_flag;
        static std::atomic<bool> stop_flag;
        struct Params
        {
            double ROBOT_WIDTH = 460;  // mm
            double ROBOT_LENGTH = 480;  // mm
            double MAX_ADV_SPEED = 1000; // mm/s
            double MAX_ROT_SPEED = 1.0; // rad/s
            double STOP_THRESHOLD = 700; // mm
            double ADVANCE_THRESHOLD = ROBOT_WIDTH * 3; // mm
            double LIDAR_FRONT_SECTION = 0.2; // rads, aprox 12 degrees
            double MIN_DIST_TO_CENTER = 200; // mm
            double MIN_DIST_TO_WALL = 100; // mm
            std::string LIDAR_NAME_HIGH = "helios";
            QRectF GRID_MAX_DIM{-5000, 2500, 10000, -5000};
            double MAX_CORNER_DIFF = 1000; // mm
            int MPC_NUM_STEPS = 10;
            int MAX_DIST_POINTS_TO_SHOW = 75; // points to show in plot
            const int CICULAR_BUFFER_SIZE = 2;
            float div_value_slider = 1000.f;
            float TRACE_SCALING_FACTOR = 50;
        };
        Params params;

        // UI
        bool reset_optimiser = false;
        bool clean_buffer = false;

        struct Phantom_Room
        {
            rc::Room room;
            Eigen::Affine2d robot_pose;
            QGraphicsPolygonItem *robot_draw;
        };

        // Actionables
        std::vector<rc::ActionableRoom> room_actionables;
        QVector<rc::ActionableThing*> thing_actionables;
        rc::ActionableOrigin initializer_actionable;
        rc::ActionableRoom *best_actionable = nullptr;

        std::shared_ptr<rc::ActionablesData> inner_model;

        /// lidar
        [[nodiscard]] std::pair<long, LidarPoints> read_lidar_helios() const;

        /// random number generator
        std::random_device rd;

        /// room
        //rc::Room_Detector room_detector;
        std::vector<Phantom_Room> active_rooms;
        rc::Room current_room;
        bool search_room(const std::vector<Eigen::Vector2d> &points);
        void update_room(const std::vector<Eigen::Vector2d> &points);

        std::tuple<Match, Target> evaluate(rc::ActionableRoom &act, const LidarPoints &points, const Eigen::Affine2d &robot_pose);
        void actionable_units(const LidarPoints &points, long lidar_timestamp);
        void competition_dynamics_lateral_inhibition(std::vector<rc::ActionableRoom> &actionables, double alpha,
                                                     double beta,
                                                     int iterations);
        std::optional<rc::ActionableRoom> check_new_candidate(const rc::ActionableOrigin &parent);
        std::optional<rc::ActionableRoom> check_new_candidate(const rc::ActionableRoom &parent);

        [[nodiscard]] std::vector<Eigen::Vector2d>
        get_and_accumulate_corners(const Eigen::Affine2d &pose,
                                       const std::vector<Eigen::Vector2d> &points,
                                       const std::vector<Eigen::Vector2d> &corners);
        [[nodiscard]] double keep_angle_between_minus_pi_and_pi(double angle);

        /// robot
        void move_robot(double adv, double rot) const;
        void stop_robot() const;
        void turn_robot(double angle);

        bool move_robot_to_target(const Eigen::Vector2d &target) const;
        Eigen::Affine2d robot_current_pose;

        /// timer
        rc::Timer<> clock;
        FPSCounter fps;

        /// search room
        // states of the  search room state machine
        RetVal move_to_center(const std::vector<Eigen::Vector2d> &points);
        RetVal move_to_random_point(const std::vector<Eigen::Vector2d> &points, const Eigen::Vector2d &target);
        Eigen::Vector2d select_new_target();

        /// auxiliarty functions
        void print_clusters(const std::vector<QPolygonF> &polys, const std::vector<unsigned long> &votes,
                        const std::vector<unsigned long> &centroids, const std::vector<unsigned long> &assignments) const;
        [[nodiscard]] Eigen::Affine2d affine3d_to_2d(const RoboCompFullPoseEstimation::FullPoseMatrix &pose);
        //[[nodiscard]] std::vector<QPointF> reorderRectanglePoints(const std::vector<QPointF> &points) const;
        [[nodiscard]] double gaussian(double x) const;

        /// draw
        AbstractGraphicViewer *viewer;
        QColor generate_random_color();
        void draw_lidar(auto &filtered_points, QGraphicsScene *scene);
        QGraphicsPolygonItem *robot_draw, *robot_draw_global;
        AbstractGraphicViewer *viewer_global;
        void draw_room_center(const Eigen::Vector2d &center, QGraphicsScene *scene, bool erase=false);
        void draw_corners(const auto &corner, QGraphicsScene *scene, QColor color, float opacity, int vector_position) const;
        void draw_corners_local(const std::vector<Eigen::Vector3d> &corner, QGraphicsScene *scene) const;
        void draw_polys(const std::vector<QPolygonF> &polys, QGraphicsScene *scene, const QColor &color);
        void draw_room(const auto &corners, const QColor &color, QGraphicsScene *scene, float opacity, int vector_position) const ;
        void draw_matching_corners_local(const std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, double, double, double>> &map, QGraphicsScene *scene);
        void draw_robot_global(const Eigen::Affine2d &pose);
        void draw_robots_global(QGraphicsItem *robot, const Eigen::Affine2d &pose);
        void draw_circular_queue(const Boost_Circular_Buffer &buffer, QGraphicsScene *scene, bool erase = false);
        void draw_residuals(const auto &points, QGraphicsScene *scene, bool erase = false);
        void draw_residuals_in_room_frame(std::vector<Eigen::Vector3d> points, QGraphicsScene *scene, bool erase = false);

        // 3D
        rc::Viewer3D *viewer3D;

        /// plotter
        QCustomPlot *plot;
        void plot_free_energy(double distance, double traza);
        void plot_multiple_free_energy(const std::vector<rc::ActionableRoom> &actionables);

        // affordances
        void move_around_affordance(const std::vector<Eigen::Vector2d> &points);


};
#endif
