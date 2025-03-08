//
// Created by pbustos on 24/02/25.
//

#ifndef FRIDGE_H
#define FRIDGE_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>

// Concept class Fridge. It is a placeholder for the fridge object. Similar to Room.
namespace rc
{
    class Rect final
    {
        public:
            Rect(double cx_, double cy_, double theta_, double width_, double depth_) :  cx(cx_), cy(cy_), theta(theta_), width(width_), depth(depth_)
            {};

            enum class Sides { TOP, RIGHT,BOTTOM, LEFT};
            double get_width() const { return width; }
            double get_depth() const { return depth; }
            double get_theta() const { return theta; }
            double get_x() const { return cx; }
            double get_y() const { return cy; }
            void set_width(double w) { width = w; }
            void set_depth(double d) { depth = d; }
            void set_theta(double t) { theta = t; }
            void set_x(double x_) { cx = x_; }
            void set_y(double y_) { cy = y_; }
            Eigen::Vector2d get_center() const { return Eigen::Vector2d(cx, cy); }
            std::vector<Sides> get_occluded_sides(const Eigen::Affine2d &robot_pose_in_room) const;
            Eigen::Vector2d get_closest_side(const Eigen::Affine2d &robot_pose_in_room) const;
            Eigen::Affine2d get_side_mid_point_in_robot(const Eigen::Affine2d &robot_pose_in_room, Sides side) const;
            Eigen::Vector2d get_side_mid_point_in_room(Sides side) const;
            Eigen::Vector2d get_side_mid_point(Sides side) const;

        private:
            double cx; //meters
            double cy;
            double theta;
            double width;
        double depth;
    };

    class Plane final
    {
    public:
        Plane(double cx_, double cy_, double theta_, double width_, double depth_, double height_) :  cx(cx_), cy(cy_), theta(theta_), width(width_), depth(depth_), height(height_)
        {};

        enum class Sides { TOP, RIGHT,BOTTOM, LEFT};
        double get_width() const { return width; }
        double get_depth() const { return depth; }
        double get_theta() const { return theta; }
        double get_height() const { return height; }
        double get_x() const { return cx; }
        double get_y() const { return cy; }
        void set_width(double w) { width = w; }
        void set_depth(double d) { depth = d; }
        void set_theta(double t) { theta = t; }
        void set_height(double h) { height = h; }
        void set_x(double x_) { cx = x_; }
        void set_y(double y_) { cy = y_; }
        Eigen::Vector2d get_center() const { return Eigen::Vector2d(cx, cy); }
        std::vector<Sides> get_occluded_sides(const Eigen::Affine2d &robot_pose_in_room) const;
        Eigen::Vector2d get_closest_side(const Eigen::Affine2d &robot_pose_in_room) const;
        Eigen::Affine2d get_side_mid_point_in_robot(const Eigen::Affine2d &robot_pose_in_room, Sides side) const;
        Eigen::Vector2d get_side_mid_point_in_room(Sides side) const;
        Eigen::Vector2d get_side_mid_point(Sides side) const;

    private:
        double cx; //meters
        double cy;
        double theta;
        double width;
        double depth;
        double height;

    };

    class Fridge
    {
        public:
            Fridge(): means{0.0, 0.0, par.INIT_ANGLE_VALUE, par.INIT_WIDTH_VALUE, par.INIT_DEPTH_VALUE},
                      base{means(0), means(1), means(2), means(3), means(4)}
            {};
            void print() // print the fridge parameters
            {
                std::cout << "Fridge: " << means.transpose() << std::endl;
            }
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
            gtsam::Vector5 means; // cx, cy, alpha, w, d
            gtsam::Vector5 sigmas;
            Rect base;
            Params par;
    };

    class Table
    {
        public:
            Table() : means{0.0, 0.0, 0.0, par.INIT_ALPHA_VALUE, par.INIT_BETA_VALUE, par.INIT_GAMMA_VALUE, par.INIT_WIDTH_VALUE, par.INIT_DEPTH_VALUE, par.INIT_HEIGHT_VALUE},
                      cx(means(0)), cy(means(1)), cz(means(2)), alpha(means(3)), beta(means(4)), gamma(means(5)), width(means(6)), depth(means(7)), height(means(8))
            {

            };
            void print() // print the fridge parameters
            {
                std::cout << "Table: " << means.transpose() << std::endl;
            }
            [[nodiscard]] double point_distance(const Eigen::Vector3d &p) const
            {
                // Transform the point to the orthotope's coordinate system
                const gtsam::Pose3 pose(gtsam::Rot3::Rz(gamma), gtsam::Point3(cx, cy, cz));
                gtsam::Point3 lp = pose.transformTo(gtsam::Point3(p.x(), p.y(), p.z()));

                // Compute the distance in the local coordinate system
                const double dx = std::max({-width / 2.0 - lp.x(), lp.x() - width / 2.0});
                const double dy = std::max({-height / 2.0 - lp.y(), lp.y() - height / 2.0});
                const double dz = std::max({-depth / 2.0- lp.z(), lp.z() - depth / 2.0});
                return std::sqrt(dx * dx + dy * dy + dz * dz);
            }
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
                double INIT_ALPHA_VALUE = 0.0;
                double INIT_BETA_VALUE = 0.0;
                double INIT_GAMMA_VALUE = 0.0;
                double INIT_ANGLE_VALUE = 0.0;
                double INIT_WIDTH_VALUE = 1;
                double INIT_DEPTH_VALUE = 0.7;
                double INIT_HEIGHT_VALUE = 0.7;
            };

                gtsam::Vector9 means;
                gtsam::Vector9 sigmas;
                Params par;
                double &cx, &cy, &cz, &alpha, &beta, &gamma, &width, &depth, &height;
    };
};// namespace rc


#endif //FRIDGE_H
