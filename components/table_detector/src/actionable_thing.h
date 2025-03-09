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
//#include <nanoflann.hpp>
#include <atomic>
#include <gtsam/base/Matrix.h>
#include "protected_map.h"
#include "custom_factor_expressions.h"

namespace rc
{
    class ActionableThing: public  QObject
    {
        Q_OBJECT
        public:
        struct Params
            {
                double POINTS_SIGMA = 0.1;
                double HEIGHT_SIGMA = 0.1;
                double ADJ_SIGMA = 0.6;
                double ALIGNMENT_SIGMA = 0.6;
                double PRIOR_CX_SIGMA = 3;
                double PRIOR_CY_SIGMA = 3;
                double PRIOR_CZ_SIGMA = 3;
                double PRIOR_ALPHA_SIGMA = 0.1;
                double PRIOR_BETA_SIGMA = 0.1;
                double PRIOR_GAMMA_SIGMA = 0.1;
                double PRIOR_WIDTH_SIGMA = 1;
                double PRIOR_DEPTH_SIGMA = 1;
                double PRIOR_HEIGHT_SIGMA = 1;

                double INIT_ALPHA_VALUE = 0.0;
                double INIT_BETA_VALUE = 0.0;
                double INIT_GAMMA_VALUE = 0.0;
                double INIT_WIDTH_VALUE = 0.5;
                double INIT_DEPTH_VALUE = 0.5;
                double INIT_HEIGHT_VALUE = 0.7;
                double INIT_BETA_SOFTMAX_VALUE = 3.0;
            };
            explicit ActionableThing(const std::shared_ptr<rc::ActionablesData> &innermodel_, QObject* parent = nullptr);
            ~ActionableThing() = default;

            //////////////////////////////////////////////////////////
            bool initialize(const rc::ActionableRoom &best_actionable_room,
                            const std::vector<Eigen::Vector3d> residuals,
                            QGraphicsScene *scene);

            LidarPoints project(const LidarPoints &residuals, const rc::ActionableRoom &current_room, QGraphicsScene *scene);
            double get_error() const;
            double get_traza() const { return covariance.trace();; };
            gtsam::Vector9 get_table() const { return inner_model->table->means; };
            static Params params;

        public slots:
            void reset_optimiser_slot() { reset_optimiser = true;}
            void clean_buffer_slot() { clean_buffer = true; }
            void change_beta_slot(double beta_) { factors::beta = beta_; }
            void change_gamma_slot(double gamma_) { factors::gamma = gamma_; }
            void change_width_prior(double width_) { params.PRIOR_WIDTH_SIGMA = width_; }

        private:
            std::shared_ptr<rc::ActionablesData> inner_model;
            gtsam::Matrix99 covariance;
            Eigen::Affine2d robot_pose;
            QGraphicsPolygonItem *thing_draw = nullptr;
            Eigen::AlignedBox2d box;
            Eigen::Vector3d target;
            //boost::circular_buffer<double> error_buffer;
            std::vector<Eigen::Vector3d> points_memory_buffer;
            bool initialized = false;
            QColor color;
            double error = 0.0f;
            double energy = 0.0;
            boost::circular_buffer<Eigen::Vector3d> residuals_queue; // buffer to hold residuals for the last N poinsts
            bool reset_optimiser = false;
            bool clean_buffer = false;


            gtsam::Vector5 factor_graph_expr_points(const std::vector<Eigen::Vector3d> &residual_points,
                                                    const gtsam::Vector9 &initial_table,
                                                    const rc::ActionableRoom &current_room,
                                                    bool reset_optimiser,
                                                    const Eigen::Vector3d &mass_center, QGraphicsScene *scene);

            //using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, NanoFlannPointCloud>, NanoFlannPointCloud, 3 /* dim */ >;
            void plotUncertaintyEllipses(const Eigen::MatrixXd &covariance, const Eigen::Matrix<double, 9, 1> &params, QGraphicsScene *scene);
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
            void set_energy(double energy_);
            double geAlignedBox2dt_energy() const;
            bool operator==(const ActionableThing &other) const;
            std::vector<Eigen::Vector3d> add_new_points(const std::vector<Eigen::Vector3d> &points);

            void draw_box(const std::vector<Eigen::AlignedBox2d> &boxes, QGraphicsScene *scene);
            void draw_clusters(const std::vector<cv::RotatedRect> &rects, QGraphicsScene *scene);
            void draw_qrects(const std::vector<QRectF> &rects, QGraphicsScene *scene);
            void draw_point(Eigen::Vector3d point, QGraphicsScene *scene, bool erase=false);
            void draw_residuals_in_room_frame(const std::vector<Eigen::Vector3d> &points, QGraphicsScene *scene, bool erase=false);
            std::vector<Eigen::Vector3d> add_new_points_2(const std::vector<Eigen::Vector3d> &points);
            void filter_with_dbscan(std::vector<Eigen::Vector3d> vector);

            void draw_table(const auto &params, const QColor &color, QGraphicsScene *scene) const;
            gtsam::Vector9 factor_graph_expr_points_table_top(const std::vector<Eigen::Vector3d> &residual_points,
                                                              const gtsam::Vector9 &initial_table,
                                                              const ActionableRoom &current_room,
                                                              const Eigen::Vector3d &mass_center, QGraphicsScene *scene);

         std::tuple<Eigen::Matrix3d, Eigen::Vector3d> compute_covariance_matrix(const std::vector<Eigen::Vector3d> &points);
         std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d, Eigen::Matrix3d>
         compute_OBB(const std::vector<Eigen::Vector3d> &points);
         std::vector<Eigen::Vector3d> generatePerimeterPoints(double width, double depth, double height, double centerX, double centerY, double distance);
    };
} // rc

#endif //ACTIONABLE_THING_H
