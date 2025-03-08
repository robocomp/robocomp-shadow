//
// Created by pbustos on 9/12/24.
//

#ifndef ACTIONABLE_THING_H
#define ACTIONABLE_THING_H

#include <vector>
#include "room.h"
#include "common_types.h"
#include "dbscan.h"
#include "actionable_room.h"
#include <Eigen/Dense>
#include "custom_factor_expressions.h"
//#include <nanoflann.hpp>
#include <atomic>
#include "protected_map.h"

namespace rc
{
    class ActionableThing
    {
        public:
        struct Params
            {
                double POINTS_SIGMA = 0.01;
                double ADJ_SIGMA = 0.6;
                double ALIGNMENT_SIGMA = 0.6;
                double PRIOR_CX_SIGMA = 3;
                double PRIOR_CY_SIGMA = 3;
                double PRIOR_ALPHA_SIGMA = 3;
                double PRIOR_WIDTH_SIGMA = 0.01;
                double PRIOR_DEPTH_SIGMA = 0.01;
                double INIT_ANGLE_VALUE = 0.0;
                double INIT_WIDTH_VALUE = 0.7;
                double INIT_DEPTH_VALUE = 0.7;
            };
            //using ProtectedMap_shr = std::shared_ptr<rc::ProtectedMap<rc::ConceptsEnum, std::shared_ptr<Actionable>>>;
            explicit ActionableThing(const std::shared_ptr<rc::ActionablesData> &innermodel_);

            void run(std::atomic<bool> &stop_flag)
            {
                while (stop_flag.load() == false)
                {
                    // wait for a current room
                        // call project on existing models
                        // check for residuals
                            // call initialize
                                // wait for low trace while doing affordances
                            // add new model
                }
            }
            //////////////////////////////////////////////////////////
            bool initialize(const rc::ActionableRoom &best_actionable_room,
                            const std::vector<Eigen::Vector3d> residuals,
                            const Params &params_,
                            QGraphicsScene *scene);

            LidarPoints project(const LidarPoints &residuals, const rc::ActionableRoom &current_room, bool reset_optimiser, QGraphicsScene *scene);
            double get_error() const;
            double get_traza() const { return covariance.trace();; };
            gtsam::Vector5 get_fridge() const { return inner_model->fridge->means; };
            static Params params;

        private:
            std::shared_ptr<rc::ActionablesData> inner_model;
            //gtsam::Vector5 fridge; // params: cx, cy, alpha, w, d

            gtsam::Matrix covariance;
            Eigen::Affine2d robot_pose;
            QGraphicsPolygonItem *thing_draw = nullptr;
            Eigen::AlignedBox2d box;
            Eigen::Vector3d target;
            boost::circular_buffer<double> error_buffer;
            bool initialized = false;
            QColor color;
            double error = 0.0f;
            double energy = 0.0;
            boost::circular_buffer<Eigen::Vector3d> residuals_queue; // buffer to hold residuals for the last N poinsts

            gtsam::Vector5 factor_graph_expr_points(const std::vector<Eigen::Vector3d> &residual_points,
                                                    const gtsam::Vector5 &initial_fridge,
                                                    const rc::ActionableRoom &current_room,
                                                    bool reset_optimiser,
                                                    const Eigen::Vector3d &mass_center, QGraphicsScene *scene);

            //using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, NanoFlannPointCloud>, NanoFlannPointCloud, 3 /* dim */ >;
            void plotUncertaintyEllipses(const Eigen::MatrixXd& covariance, const Eigen::Matrix<double, 5, 1> &params, QGraphicsScene* scene);
            Match project_corners(const Corners &corners, long lidar_timestamp);
            double compute_total_error(const Match &matches);
            Eigen::Affine2d update_robot_pose(const Eigen::Affine2d &pose);
            bool check_viability();
            void remove_thing_draw(QGraphicsScene *scene);
            void set_thing_opacity(float opacity);
            void define_actionable_color(const QColor &color);
            void set_error(double error);
            std::vector<Eigen::Vector3d> get_corners_3d() const;
            Eigen::Affine2d get_robot_pose() const;
            QGraphicsItem * get_thing_draw() const;
            QColor get_actionable_color() const;
            double get_buffer_error() const;
            double get_prediction_fitness(double eps = 0.001) const;
            void set_energy(double energy_);
            double geAlignedBox2dt_energy() const;
            bool operator==(const ActionableThing &other) const;
            std::vector<Eigen::Vector3d> add_new_points(const std::vector<Eigen::Vector3d> &points);

            void draw_box(const std::vector<Eigen::AlignedBox2d> &boxes, QGraphicsScene *scene);
            void draw_fridge(const auto &corners, const QColor &color, QGraphicsScene *scene) const;
            void draw_clusters(const std::vector<cv::RotatedRect> &rects, QGraphicsScene *scene);
            void draw_qrects(const std::vector<QRectF> &rects, QGraphicsScene *scene);
            void draw_point(Eigen::Vector3d point, QGraphicsScene *scene, bool erase=false);
            void draw_residuals_in_room_frame(const std::vector<Eigen::Vector3d> &points, QGraphicsScene *scene, bool erase=false);
            std::vector<Eigen::Vector3d> add_new_points_2(const std::vector<Eigen::Vector3d> &points);
            void filter_with_dbscan(std::vector<Eigen::Vector3d> vector);
     };
} // rc

#endif //ACTIONABLE_THING_H
