#ifndef FRIDGEPOINTSFACTOR_H
#define FRIDGEPOINTSFACTOR_H

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <Eigen/Core>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>


namespace factors
{
    inline double beta = 10;  // for the relu
    inline double gamma = 1; // for the softabs


    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// FRIDGE custom factor expressions
    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////// Auxiliary functions

     /**
     * \brief Normalizes a value by a given denominator.
     *
     * \param denom The denominator as a gtsam::Vector1.
     * \param val The value to be normalized.
     * \param H1 Optional Jacobian for the normalization with respect to the denominator.
     * \param H2 Optional Jacobian for the normalization with respect to the value.
     * \return The normalized value.
     */
    static double normalize (const gtsam::Vector1 &denom, const double val, gtsam::OptionalJacobian<1, 1> H1, gtsam::OptionalJacobian<1, 1> H2)
    {
        if (H1) *H1 = gtsam::Matrix11::Zero();
        if (H2) *H2 = gtsam::Matrix11(1.0 / denom(0));
        return val/denom(0);
    }

    /**
     * \brief Returns the squared y coordinate as the distance to the side/wall.
     *
     * \param p Point in fridge coordinates.
     * \param H Optional Jacobian for the distance with respect to the point.
     * \return Distance to the side of the fridge.
     */
    static double dist2side_sqr(const gtsam::Vector2 &p, gtsam::OptionalJacobian<1, 2> H)
    {
        if (H) *H = (gtsam::Matrix12() << 0, 2*p.y()).finished();
//        std::cout << "Dist to side sqr: " << p.y()*p.y() << std::endl;
        return p.y()* p.y();
    }

    /**
     * @brief Función sigmoide y su derivada used in dist2seg
     */
    inline double logistic(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline double dlogistic(double x) {
        double s = logistic(x);
        return s * (1.0 - s);
    }

    static double softmin_local(const std::vector<double>& dists, const double k=1.0)
    {
        double numerator = 0.0;
        double denominator = 0.0;
        for (const double dist : dists)
        {
            const double w = std::exp(-k * dist);
            numerator += dist * w;
            denominator += w;
        }
        return numerator / denominator;
    }

    static std::vector<double> dsoftmin_local_dd(const std::vector<double>& dists, const double k)
    {
        double S = 0.0, N = 0.0;
        for (const double dist : dists)
        {
            const double e = std::exp(-k * dist);
            S += e;
            N += dist * e;
        }
        std::vector<double> grad(dists.size(), 0.0);
        for (size_t j = 0; j < dists.size(); j++)
        {
            double e = std::exp(-k * dists[j]);
            grad[j] = (e * (1 - k * dists[j]) * S + k * e * N) / (S * S);
        }
        return grad;
    }

    /**
    * @brief Softplus function as a smooth approximation to ReLU.
    *
    * The softplus function is defined as:
    * softplus(x) = ln(1 + exp(x))
    *
    * @param x The input value.
    * @return The result of the softplus function.
    */
     static double soft_max(double x, double beta)
    {
        if (x > 20.0)
            return x; // For large x, softplus(x) ≈ x
        return std::log1p(std::exp(beta*x));  // Using log1p(exp(x)) is numerically more stable for small x.
    }

    static double soft_max_derivative(double x, double beta)
    {
        return beta / (1.0 + std::exp(-beta*x));
    }

    static double soft_abs(double x, double gamma)
    {
        constexpr double eps = 1e-6;
        return std::sqrt(gamma * gamma * x * x + eps);
    }

    static double soft_abs_derivative(double x, double gamma)
    {
        constexpr double eps = 1e-6;
        return (gamma*gamma*x) / std::sqrt(gamma * gamma * x * x + eps);
    }

    /**
     * \brief Computes the skew-symmetric matrix of a given 3D vector.
     *
     * The skew-symmetric matrix is used in various vector cross product operations.
     *
     * \param v The input 3D vector.
     * \return The skew-symmetric matrix of the input vector.
     */
    inline Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
        Eigen::Matrix3d S;
        S <<     0, -v(2),  v(1),
                v(2),     0, -v(0),
                -v(1),  v(0),     0;
        return S;
    }


    //////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    /**
     * \brief Softmin function: given 4 values compute the minimum value in a smooth way.
     *  Given values compute the minimum value in a smooth way
     *  softmin = (sum_i d_i * exp(-k*d_i)) / (sum_i exp(-k*d_i))
     * \params v0, v1, v2, v3 Values to be compared.
     * \param H1 Optional Jacobian for the first value.
     * \param H2 Optional Jacobian for the second value.
     * \param H3 Optional Jacobian for the third value.
     * \param H4 Optional Jacobian for the fourth value.
     * \return The smooth minimum of the given values.
     */
    static double softmin(const double v0, const double v1, const double v2, const double v3, gtsam::OptionalJacobian<1, 1> H1,
                                                                                              gtsam::OptionalJacobian<1, 1> H2,
                                                                                              gtsam::OptionalJacobian<1, 1> H3,
                                                                                              gtsam::OptionalJacobian<1, 1> H4)
    {
        //  A higher value of k makes the function more sensitive to smaller differences between the input values, resulting in a smoother approximation of the minimum value.
        //  Conversely, a lower value of k would make the function less sensitive, leading to a less smooth approximation
        constexpr double k = 15;
        double numer = 0.0;
        double denom = 0.0;

        const double exp_v0 = std::exp(-k * v0);
        const double exp_v1 = std::exp(-k * v1);
        const double exp_v2 = std::exp(-k * v2);
        const double exp_v3 = std::exp(-k * v3);

        denom = exp_v0 + exp_v1 + exp_v2 + exp_v3;
        numer = v0 * exp_v0 + v1 * exp_v1 + v2 * exp_v2 + v3 * exp_v3;

        // Jacobians computation
        if (H1) *H1 = gtsam::Matrix11((exp_v0 * (1 - k * v0) * denom + k * exp_v0 * numer) / (denom * denom));
        if (H2) *H2 = gtsam::Matrix11((exp_v1 * (1 - k * v1) * denom + k * exp_v1 * numer) / (denom * denom));
        if (H3) *H3 = gtsam::Matrix11((exp_v2 * (1 - k * v2) * denom + k * exp_v2 * numer) / (denom * denom));
        if (H4) *H4 = gtsam::Matrix11((exp_v3 * (1 - k * v3) * denom + k * exp_v3 * numer) / (denom * denom));

        //std::cout << "Softmin: " << numer/denom << std::endl;
        return numer / denom;
    }

    static double dist2table_top(const gtsam::Vector9 &v, const gtsam::Vector3 &p,
                                 gtsam::OptionalJacobian<1, 9> H1,
                                 gtsam::OptionalJacobian<1, 3> H2)
    {
        // Distance from the point to the table top
        const double distance = p.z()*p.z();

        if (H1)
        {
            H1->setZero();
            //H1->block<1, 3>(0, 2) << 0.0 , 0.0 , 2*v(2);
        }
        if (H2)
        {
            H2->setZero();
            *H2 << 0.0 , 0.0, 2*p.z();
        }

        return distance;
    };

    /**
     * \brief computes the min distance of the 4 fridge middle points to a room side given in params.
     *
     * \param v Vector containing fridge parameters.
     * \param p
     * \param params Vector containing wall parameters: wall number, semi_depth, semi_width.
     * \param H1 Optional Jacobian for the transformation with respect to the fridge parameters.
     * \param H2 Optional Jacobian for the transformation with respect to the wall parameters.
     * \return Transformed point in wall coordinates.
     */
    static gtsam::Vector3 room2table(const gtsam::Vector9 &v, const gtsam::Vector3 &p,
                                     gtsam::OptionalJacobian<3, 9> H1,
                                     gtsam::OptionalJacobian<3, 3> H2)
    {
        // Construct the pose from v:
        // - Translation: v(0), v(1), v(2)
        // - Rotation: only v(5) is used for a rotation in Z
        const auto r2t = gtsam::Pose3(gtsam::Rot3::Rz(v(5)), gtsam::Point3(v(0), v(1), v(2)));

        // Create local matrices for the Jacobians of the pose and the point.
        gtsam::Matrix36 h_pose;   // Jacobian 3x6 wrt the pose
        gtsam::Matrix33 h_point;  // Jacobian 3x3 wrt the point

        // Transform the point p to the pose r2t system.
        const auto res = r2t.transformTo(p, h_pose, h_point);

        // compute the Jacobian of the transformation wrt v: d(r2t(p))/dv
        if (H1)
        {
            *H1 = gtsam::Matrix39::Zero();

            // Assign the derivatives with respect to the pose parameters:
            // - The first 3 columns correspond to the translation:
            H1->block<3,3>(0,0) = h_pose.block<3,3>(0,3); // translation
            // - The next 3 columns correspond to the rotation:
            H1->block<3,3>(0,3) = h_pose.block<3,3>(0,0); // rotation
            // - The last 3 columns (v(6:8)) remain zero.
            //H1->setZero();
        }
        // If the Jacobian with respect to p is requested, it is assigned directly.
        if (H2)
            *H2 = h_point;
            //H2->setZero();

        return res;
    }

    /**
     * \brief Transforms a point from the table frame to the top frame.
     *
     * \param v Vector containing the table parameters.
     * \param p Point in the table frame.
     * \param H1 Optional Jacobian for the transformation with respect to the table parameters.
     * \param H2 Optional Jacobian for the transformation with respect to the point.
     * \return Transformed point in the top frame.
    */
    static gtsam::Vector3 table2top(const gtsam::Vector9 &v, const gtsam::Vector3 &p,
                                    gtsam::OptionalJacobian<3, 9> H1,
                                    gtsam::OptionalJacobian<3, 3> H2)
    {
        // Translation: (0, 0, v(8)/2) and identity rotation. [semi-height]
        const double semi_height = v(8) / 2.0;
        const auto r2t = gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0.0, 0.0, semi_height));

        // Create local matrices for the Jacobians
        gtsam::Matrix36 h_pose;   // Jacobian 3x6 with respect to the pose (rot the trans)
        gtsam::Matrix33 h_point;  // Jacobian 3x3 with respect to the point

        // Transform the point p using r2t.
        const auto res = r2t.transformTo(p, h_pose, h_point);

        // Jacobian with respect to v: r2t depends on v only in v(8), which affects the translation in z,
        // and not on the rotation or translation parameters of the center.
        // The derivative should be placed in the corresponding block.
        if (H1)
        {
            *H1 = gtsam::Matrix39::Zero();  // Initialize to zero (size 3x9)
            H1->block<3, 1>(0, 8) = h_pose.block<3, 1>(0, 5);  // translation
        }

        // Jacobian with respect to p is assigned directly.
        if (H2)
            *H2 = h_point;
        return res;
    }

    /**
    * \brief
    *
    * \param v Vector containing the table parameters.
    * \param p Point in the table frame.
    * \param H1 Optional Jacobian for the transformation with respect to the table parameters.
    * \param H2 Optional Jacobian for the transformation with respect to the point.
    * \return Transformed point in the top frame.
   */
    static double stick_to_floor(const gtsam::Vector9 &v, gtsam::OptionalJacobian<1, 9> H1)
    {
        const auto bottom = gtsam::Vector3(v(0), v(1), -v(8)/2.0);
        const double dist = bottom.z() * bottom.z();

        if (H1)
        {
            *H1 = gtsam::Matrix19::Zero();  // Initialize to zero (size 3x9)
            H1->block<1, 1>(0, 2) << 2.0*bottom.z();  // translation
        }
        return dist;
    }

    /**
     * \brief If point is outside tabletop, computes the segment distance to the topside of the tabletop, otherwise returns 0.
     *
     *  Check the jacobian.tex file to see how the Jacobian is computed.
     *
     * \param v Vector containing table parameters.
     * \param side select the side of the table.
     * \param p Point in the table frame.
     * \param H1 Optional Jacobian for the transformation with respect to the table parameters.
     * \param H2 const Jacobian is zero
     * \param H3 Optional Jacobian for the transformation with respect to the point.
     * \return The minimum distance to the sides of the table.
     */
    static double min_dist_to_side_x(const gtsam::Vector9 &v, const double side, const gtsam::Vector3 &p,
                                     gtsam::OptionalJacobian<1, 9> H1,
                                     gtsam::OptionalJacobian<1, 1> H2,
                                     gtsam::OptionalJacobian<1, 3> H3)
    {
//         if (side < 0 or side > 4)
//             throw std::invalid_argument("Invalid side value. It should be between 1 and 4.");
//         if (v(6)==0 or v(7)==0)
//             throw std::invalid_argument("Invalid dimensions values. It should be positive.");
//
//         double vx = side==1 or side==3 ? v(7) : v(6);   // depth for side 1 and 3, width for side 2 and 4
//
//         gtsam::Pose3 t2s;
//         gtsam::Point3 A{-vx / 2.0, 0.0, 0.0};
//         gtsam::Point3 B{vx / 2.0, 0.0, 0.0};
//         switch (static_cast<int>(side))
//         {
//             case 1:
//                 t2s = gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0.0, vx / 2.0, 0.0));
//                 break;
//             case 2:
//                 t2s = gtsam::Pose3(gtsam::Rot3::Rz(-M_PI/2), gtsam::Point3(vx/2, 0.0, 0.0));
//                 break;
//             case 3:
//                 t2s = gtsam::Pose3(gtsam::Rot3::Rz(M_PI), gtsam::Point3(0.0, -vx/2, 0.0));
//                 break;
//             case 4:
//                 t2s = gtsam::Pose3(gtsam::Rot3::Rz(M_PI/2), gtsam::Point3(-vx/2, 0.0, 0.0));
//                 break;
//             default: ;
//         }
//
//         // Calculate the transformed point ps to the side frame and its Jacobians
//         gtsam::Matrix36 H_pose;
//         gtsam::Matrix33 H_point;
//         const gtsam::Point3 ps = t2s.transformTo(p, H_pose, H_point);
//         gtsam::Vector3 h_pose_vx;
//         if (side == 1 or side == 3)
//             h_pose_vx = H_pose.block<3, 1>(0, 4); // Extract column for v7
//         else
//             h_pose_vx = H_pose.block<3, 1>(0, 3); // Extract column for v6
//
//         const double ps_x = ps.x();
//         const double ps_y = ps.y();
//         const double t = ps_x / (B.x() - A.x()) + 0.5;
//         const double s0 = logistic(beta * t);
//         const double s1 = logistic(beta * (t - 1.0));
//         const double win = s0 * (1.0 - s1);
//         const double wleft = 1.0 - s0;
//         const double wright = s1;
//         const double din = ps_y * ps_y;
//         const double dleft = (ps - A).squaredNorm();
//         const double dright = (ps - B).squaredNorm();
//
//         const double dist = win * din + wleft * dleft + wright * dright;
//
//         // Apply the softplus gate to the *signed* y-coordinate
//         double outlier_gate = softplus(ps_y*gamma);
//         double filtered_dist = dist * outlier_gate;
//
//         // if (ps_y > 0.0)
// //              qInfo() << "[" << p.x() << p.y() << p.z() << "--" << ps.x() << ps.y() << ps.z() << "] y_coor" << ps_y << "t " << t << " dist" << dist << " filtered_dist" << filtered_dist;
//
//         // Derivative of the outlier gate.
//         gtsam::Vector3 d_outlier_gate_dps =  (gtsam::Vector3() << 0.0, softplus_derivative(ps_y), 0.0).finished();
//
//         if (H1)   // H1 = outlier_gate * ddist_dv7 + dist * doutlier_gate_dps * h_pose_v7
//         {
//             *H1 = gtsam::Matrix19::Zero(); // Use fixed-size Matrix19
//
//             // --- Term K ---
//             const auto ddin_dps = gtsam::Vector3(0.0, 2.0 * ps_y, 0.0);
//             const double ddin_dvx = ddin_dps.transpose() * h_pose_vx;
//             const double dwin_ds0 = 1.0 - s1;
//             const double dwin_ds1 = -s0;
//             const double ds0_dt1 = beta * logistic(beta * t) * (1.0 - logistic(beta * t));
//             const double ds1_dt1 = beta * logistic(beta * (t - 1.0)) * (1.0 - logistic(beta * (t - 1.0)));
//             const double dt1_dvx = (1.0 / vx) * h_pose_vx[0] + ps_x / (vx * vx);
//             const double dwin_dvx = (dwin_ds0 * ds0_dt1 + dwin_ds1 * ds1_dt1) * dt1_dvx;
//             const double dK_dvx = win * ddin_dvx + din * dwin_dvx;
//
//             // --- Term L ---
//             const double dleft_dvx = 2.0 * (ps - A).transpose() * h_pose_vx + (ps_x - A.x());
//             constexpr double dwleft_ds0 = -1.0;
//             const double dwleft_dvx = dwleft_ds0 * ds0_dt1 * dt1_dvx;
//             const double dL_dvx = wleft * dleft_dvx + dleft * dwleft_dvx;
//
//             // --- Term M ---
//             const double dright_dvx = 2.0 * (ps - B).transpose() * h_pose_vx - (ps_x - B.x());
//             constexpr double dwright_ds1 = 1.0;
//             const double dwright_dvx = dwright_ds1 * ds1_dt1 * dt1_dvx;
//             const double dM_dvx = wright * dright_dvx + dright * dwright_dvx;
//
//             // --- Combine ---
//             const double ddist_dvx = dK_dvx + dL_dvx + dM_dvx;
//
//             // Derivative of loss wrt vx
//             // Chain rule:  outlier_gate * ddist_dvx + dist * doutlier_gate_dps * h_pose_vx
//             if (side == 1 or side == 3)
//                 (*H1)(0, 7) =  outlier_gate * ddist_dvx + dist * d_outlier_gate_dps.transpose() * h_pose_vx;
//             else
//                 (*H1)(0, 6) =  outlier_gate * ddist_dvx + dist * d_outlier_gate_dps.transpose() * h_pose_vx;
//
// //            std::cout << "H1: " << *H1 << std::endl;
//         }
//         if (H2)
//             *H2 = gtsam::Matrix11::Zero(); // constants
//         if (H3)
//         {
//             *H3 = gtsam::Matrix13::Zero(); // Use fixed-size Matrix13
//
//             const auto ddin_dps = gtsam::Vector3(0.0, 2.0 * ps_y, 0.0);
//             const gtsam::Vector3 dleft_dps = 2.0 * (ps - A);
//             const gtsam::Vector3 dright_dps = 2.0 * (ps - B);
//
//             const double dt1_dpsx = 1.0 / vx;
//             const gtsam::Vector3 dt1_dps(dt1_dpsx, 0.0, 0.0);
//
//             const double dwin_ds0 = 1.0 - s1;
//             const double dwin_ds1 = -s0;
//             const double ds0_dt1 = beta * logistic(beta * t) * (1.0 - logistic(beta * t));
//             const double ds1_dt1 = beta * logistic(beta * (t - 1.0)) * (1.0 - logistic(beta * (t - 1.0)));
//             const gtsam::Vector3 dwin_dps = (dwin_ds0 * ds0_dt1 + dwin_ds1 * ds1_dt1) * dt1_dps;
//
//             constexpr double dwleft_ds0 = -1.0;
//             const gtsam::Vector3 dwleft_dps = dwleft_ds0 * ds0_dt1 * dt1_dps;
//
//             constexpr double dwright_ds1 = 1.0;
//             const gtsam::Vector3 dwright_dps = dwright_ds1 * ds1_dt1 * dt1_dps;
//
//             const gtsam::Vector3 ddist_dps = win * ddin_dps + din * dwin_dps
//                 + wleft * dleft_dps + dleft * dwleft_dps
//                 + wright * dright_dps + dright * dwright_dps;
//
//             // Derivative of loss wrt p
//             (*H3) = (outlier_gate * ddist_dps.transpose() + dist * d_outlier_gate_dps.transpose()) * H_point;
//
// //            std::cout << "H3: " << *H3 << std::endl;
//         }
   //      return filtered_dist;
        return{};
    };

     /**
     * \brief
     *
     *  Check the jacobian.tex file to see how the Jacobian is computed.
     *
     * \param v Vector containing table parameters.
     * \param side select the side of the table.
     * \param p Point in the table frame.
     * \param H1 Optional Jacobian for the transformation with respect to the table parameters.
     * \param H2 const Jacobian is zero
     * \param H3 Optional Jacobian for the transformation with respect to the point.
     * \return The minimum distance to the sides of the table.
     */
    static double min_dist_to_side_top(const gtsam::Vector9 &v, const double side, const gtsam::Vector3 &p,
                                     gtsam::OptionalJacobian<1, 9> H1,
                                     gtsam::OptionalJacobian<1, 1> H2,
                                     gtsam::OptionalJacobian<1, 3> H3)
    {
        // top side
        const double width = v(6);
        const double depth = v(7);
        const double height = v(8);

        gtsam::Matrix36 H_pose;
        gtsam::Matrix33 H_point;
        const auto t_top = gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0.0, 0.0, height / 2));
        gtsam::Vector3 ps = t_top.transformTo(p, H_pose, H_point);

        const double Lx = width / 2.0;
        const double Ly = depth / 2.0;
        const double sx = soft_abs(ps.x(), gamma);
        const double sy = soft_abs(ps.y(), gamma);
        const double dx = soft_max(sx - Lx, beta);
        const double dy = soft_max(sy - Ly, beta);

        const double dist = ps.z() * ps.z() + dx * dx + dy * dy;

        if (H1)
        {
            const double d_dx2_dux = 2.0 * dx;
            const double d_dy2_duy = 2.0 * dy;
            const double d_ux_d_vx = soft_max_derivative(sx-Lx, beta);
            const double d_uy_d_vy = soft_max_derivative(sy-Ly, beta);
            const double d_sx_d_px = soft_abs_derivative(ps.x(), gamma);
            const double d_sy_d_py = soft_abs_derivative(ps.y(), gamma);
            // Derivatives
            //const double d_D_d_w = sx * d_dx2_dux * d_ux_d_vx * d_sx_d_px * H_pose(0,3);
            const double d_D_d_w = -2.0 * soft_max(std::abs(ps.x()) - Lx, beta) * soft_max_derivative(std::abs(ps.x()) - Lx, beta);
            //const double  d_D_d_d = sy * d_dy2_duy * d_uy_d_vy * d_sy_d_py * H_pose(1,4);
            const double d_D_d_d = -2.0 * soft_max(std::abs(ps.y()) - Ly, beta) * soft_max_derivative(std::abs(ps.y()) - Ly, beta);
            const double d_D_d_h = 2.0 * ps.z() * H_pose(2, 5);
            *H1 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, d_D_d_w, d_D_d_d, d_D_d_h;

            double h6 = H1->block<1, 1>(0, 6)(0,0);
            double h7 = H1->block<1, 1>(0, 7)(0,0);
            if (h6<0 and h7>0 or h7<0 and h6>0)
            {
                qInfo() << "Data: ";
                qInfo() << "    H1(6) " << h6 << " H1(7) " << h7;
                qInfo() << "    [" << p.x() << p.y() << p.z() << " : " << ps.x() << ps.y() << ps.z() << "]";
                qInfo() << "    dx " << dx << " dy " << dy << " dist" << dist << "inside " << (dx < 0 && dy < 0) <<
                                "sx-lx " << sx-Lx << "sy-ly " << sy-Ly << "sx " << sx << "sy " << sy << " Lx " << Lx << " Ly " << Ly;
                qInfo() << "    sx " << sx << " d_dx2_dux " << d_dx2_dux << " d_ux_d_vx " << d_ux_d_vx << " d_sx_d_px " << d_sx_d_px << " H_pose(0,3) " << H_pose(0,3);
                qInfo() << "    sy " << sy << " d_dy2_duy " << d_dy2_duy << " d_uy_d_vy " << d_uy_d_vy << " d_sy_d_py " << d_sy_d_py << " H_pose(1,4) " << H_pose(1,4);
            }
            // qInfo() << "    v(0) " << v(0) << " v(1) " << v(1) << " v(2) " << v(2) << " v(3) " << v(3) <<
            //              " v(4) " << v(4) << " v(5) " << v(5) << " v(6) " << v(6) << " v(7) " << v(7) <<
            //              " v(8) " << v(8) ;

            //std::cout << *H1 << std::endl;

        }
        if (H2) *H2 = gtsam::Matrix11::Zero();
        if (H3)
        {
            const auto d_D_d_px = 2.0 * soft_max(sx-Lx, beta) * soft_max_derivative(sx-Lx, beta) * soft_abs_derivative(ps.x(), gamma) * H_point(0, 0);
            const auto d_D_d_py = 2.0 * soft_max(sy-Ly, beta) * soft_max_derivative(sy-Ly, beta) * soft_abs_derivative(ps.y(), gamma) * H_point(1, 1);
            const auto d_D_d_pz = 2.0 * ps.z() * H_point(2, 2);
            H3->setZero();
            *H3 << d_D_d_px, d_D_d_py, d_D_d_pz;
            //*H3 << d_D_d_px, d_D_d_py, 0.0;

            //std::cout << *H3 << std::endl;
            //qInfo() << "---------------";
        }
        return dist;
    };
};

#endif // FRIDGEPOINTSFACTOR_H
