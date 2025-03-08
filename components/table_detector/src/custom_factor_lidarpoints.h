#ifndef FRIDGEPOINTSFECTOR_H
#define FRIDGEPOINTSFECTOR_H

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Point2.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <cmath>
#include <algorithm>
#include <boost/optional.hpp>

class FridgePointsFactor final : public gtsam::NoiseModelFactor1<gtsam::Vector5>
{
  public:
    // Fixed LiDAR points in the robot's frame.
    std::vector<Eigen::Vector2d> points_;
    // Transformation from the room frame to the robot frame.
    Eigen::Affine2d T_R_to_r_;
    // Softmin softness parameter.
    double k_softmin_;

    // Constructor.
    FridgePointsFactor(const gtsam::Key fridgeKey,
                       const std::vector<Eigen::Vector2d>& points,
                       const Eigen::Affine2d& T_R_to_r,
                       const std::shared_ptr<gtsam::noiseModel::Isotropic>& noiseModel,
                       const double k_softmin = 10.0)
      :
      gtsam::NoiseModelFactor1<gtsam::Vector5>(noiseModel, fridgeKey),
        points_(points),
        T_R_to_r_(T_R_to_r),
        k_softmin_(k_softmin)
    {}

    // Use the signature required by your version:
    gtsam::Vector evaluateError(const gtsam::Vector5 &b, gtsam::Matrix *H) const override
    {
      // Unpack fridge parameters: [width, depth, angle, center_x, center_y]
      const double width   = b(0);
      const double depth   = b(1);
      const double angle   = b(2);
      const double centerX = b(3);
      const double centerY = b(4);

      double totalError = 0.0;
      Eigen::Matrix<double, 1, 5> J;
      if (H) { J.setZero(); }

      // Loop over each LiDAR point.
      for (const auto &p : points_)
      {
        // --- Transform point from robot frame to room frame ---
        // Note: Our transform T_R_to_r_ maps room -> robot,
        // so its inverse maps robot -> room.
//        Eigen::Vector2d p_room = T_R_to_r_ * Eigen::Vector2d(p.x() * 0.001, p.y() * 0.001); // direct transformation going upwards in tree
          Eigen::Vector2d p_room = {p.x(), p.y()};
        // --- Build candidate fridge transform T_R_to_f from room to fridge ---
        // The candidate fridge transform is built from the fridge parameters.
        // Translation by (centerX, centerY) and rotation by angle.
        Eigen::Affine2d T_R_to_f = Eigen::Translation2d(centerX, centerY) * Eigen::Rotation2Dd(angle);
        // Transform p_room to the fridge frame.
        Eigen::Vector2d p_fridge = T_R_to_f.inverse() * p_room; // inverse transformation going downwards in tree
        // --- Compute distances in fridge frame ---
        // In the fridge's local frame, the fridge is represented as a rectangle centered at (0,0)
        // with half-extents: halfW = width/2 and halfD = depth/2.
        const double halfW = width / 2.0;
        const double halfD = depth / 2.0;

        // Instead of using std::fabs, we use a smooth absolute value function.
        // Here is one simple differentiable approximation:
        auto smoothAbs = [](const double x, const double eps = 1e-4) -> double
        { return std::sqrt(x * x + eps); };

        // Compute smooth absolute differences.
        const double d_front = smoothAbs(p_fridge(0) - halfW);
        const double d_back  = smoothAbs(p_fridge(0) + halfW);
        const double d_left  = smoothAbs(p_fridge(1) - halfD);
        const double d_right = smoothAbs(p_fridge(1) + halfD);

        // Compute the softmin of the four distances.
        std::vector<double> dists = { d_front, d_back, d_left, d_right };
//        std::cout << "Distances: " << d_front << ", " << d_back << ", " << d_left << ", " << d_right << std::endl;
        const double pointCost = softmin(dists, k_softmin_);
//        std::cout << "Point cost: " << pointCost << std::endl;
        totalError += pointCost;

        if (H)
        {
          // Compute derivatives for softmin.
          double sumExp = 0.0;
          double sumExpTimes = 0.0;
          for (size_t j = 0; j < dists.size(); j++)
          {
            const double expVal = std::exp(-k_softmin_ * dists[j]);
            sumExp += expVal;
            sumExpTimes += dists[j] * expVal;
          }
          std::vector<double> dSoftmin_dd(4, 0.0);
          for (size_t j = 0; j < dists.size(); j++)
          {
            const double expVal = std::exp(-k_softmin_ * dists[j]);
            // Derivative of softmin: d/d d_j (sumExpTimes/sumExp)
            dSoftmin_dd[j] = (expVal * (1 - k_softmin_ * dists[j]) * sumExp + k_softmin_ * expVal * sumExpTimes) / (sumExp * sumExp);
          }

          // Next, compute derivatives of each distance with respect to fridge parameters.
          // First, compute the transformation details.
          const double dx = p_room(0) - centerX;
          const double dy = p_room(1) - centerY;
          const double cosA = std::cos(angle);
          const double sinA = std::sin(angle);
          const double fx = cosA * dx + sinA * dy;
          const double fy = -sinA * dx + cosA * dy;

          // For the smooth absolute, we differentiate smoothAbs(x) = sqrt(x^2 + eps).
          auto dsmoothAbs = [=](double x, double eps = 1e-4) -> double {
            return x / std::sqrt(x * x + eps);
          };

          // For d_front: function is smoothAbs(fx - halfW)
          const double dfront_dfx = dsmoothAbs(fx - halfW);
          const double dfront_dwidth = -0.5 * dsmoothAbs(fx - halfW);
          // For d_back: function is smoothAbs(fx + halfW)
          const double dback_dfx = dsmoothAbs(fx + halfW);
          const double dback_dwidth = 0.5 * dsmoothAbs(fx + halfW);
          // For d_left: function is smoothAbs(fy - halfD)
          const double dleft_dfy = dsmoothAbs(fy - halfD);
          const double dleft_ddepth = -0.5 * dsmoothAbs(fy - halfD);
          // For d_right: function is smoothAbs(fy + halfD)
          const double dright_dfy = dsmoothAbs(fy + halfD);
          const double dright_ddepth = 0.5 * dsmoothAbs(fy + halfD);

          // Now compute derivatives of fx and fy with respect to angle, centerX, centerY.
          const  double dfx_dangle = -sinA * dx + cosA * dy;
          const double dfx_dcx = -cosA;
          const double dfx_dcy = -sinA;
          const double dfy_dangle = -cosA * dx - sinA * dy;
          const double dfy_dcx = sinA;
          const double dfy_dcy = -cosA;

          // Chain rule to compute derivative for each parameter.
          const double dcost_dwidth = dSoftmin_dd[0] * dfront_dwidth + dSoftmin_dd[1] * dback_dwidth;
          const double dcost_ddepth = dSoftmin_dd[2] * dleft_ddepth + dSoftmin_dd[3] * dright_ddepth;
          const double dcost_dangle = dSoftmin_dd[0] * (dfront_dfx * dfx_dangle)
                                      + dSoftmin_dd[1] * (dback_dfx * dfx_dangle)
                                      + dSoftmin_dd[2] * (dleft_dfy * dfy_dangle)
                                      + dSoftmin_dd[3] * (dright_dfy * dfy_dangle);
          const double dcost_dcx = dSoftmin_dd[0] * (dfront_dfx * dfx_dcx)
                                   + dSoftmin_dd[1] * (dback_dfx * dfx_dcx)
                                   + dSoftmin_dd[2] * (dleft_dfy * dfy_dcx)
                                   + dSoftmin_dd[3] * (dright_dfy * dfy_dcx);
          const double dcost_dcy = dSoftmin_dd[0] * (dfront_dfx * dfx_dcy)
                                   + dSoftmin_dd[1] * (dback_dfx * dfx_dcy)
                                   + dSoftmin_dd[2] * (dleft_dfy * dfy_dcy)
                                   + dSoftmin_dd[3] * (dright_dfy * dfy_dcy);

          // Accumulate into Jacobian.
          J(0, 0) += dcost_dwidth;
          J(0, 1) += dcost_ddepth;
          J(0, 2) += dcost_dangle;
          J(0, 3) += dcost_dcx;
          J(0, 4) += dcost_dcy;
        }
      } // end loop over points

      if (H) { *H = J; }
      gtsam::Vector err(1);
      err(0) = totalError;
      qDebug() << "Grad" << J(0, 0) << ", " << J(0, 1) << ", " << J(0, 2) << ", " << J(0, 3) << ", " << J(0, 4);
      return err;
    }

  private:
    // Softmin: a differentiable approximation to the minimum.
    // dists: a vector of distances, k: softness parameter.
    template <typename T>
    T softmin(const std::vector<T>& dists, const T& k) const
    {
      T numerator = T(0);
      T denominator = T(0);
      for (size_t i = 0; i < dists.size(); i++)
      {
        T w = std::exp(-k * dists[i]);
        numerator += dists[i] * w;
        denominator += w;
      }
//      std::cout << "Numerator: " << numerator << ", Denominator: " << denominator << std::endl;
      return numerator / denominator;
    }
};

#endif // FRIDGEPOINTSFECTOR_H



// #ifndef FRIDGEPOINTSFECTOR_H
// #define FRIDGEPOINTSFECTOR_H
//
// #include <gtsam/nonlinear/NonlinearFactor.h>
// #include <gtsam/base/Vector.h>
// #include <gtsam/geometry/Point2.h>
// #include <Eigen/Core>
// #include <vector>
// #include <cmath>
// #include <algorithm>
// #include <boost/optional.hpp>
// #include <gtsam/base/Matrix.h>
//
// class FridgePointsFactor : public gtsam::NoiseModelFactor1<gtsam::Vector5>
// {
//     public:
//         // A vector of LiDAR points (in the room's frame). These are fixed observations.
//         std::vector<Eigen::Vector3d> points_;
//
//         // Constructor: takes a noise model, a key for the fridge parameters, and the LiDAR points.
//         FridgePointsFactor(gtsam::Key fridgeKey,
//                            const std::vector<Eigen::Vector3d> &points,
//                            const std::shared_ptr<gtsam::noiseModel::Isotropic> &noiseModel)
//             : gtsam::NoiseModelFactor1<gtsam::Vector5>(noiseModel, fridgeKey), points_(points)
//         {
//         }
//
//         // Implements the error function and its Jacobian.
//         // The fridge parameters b are a 5D vector: [width, depth, angle, center_x, center_y].
//         virtual gtsam::Vector evaluateError(const gtsam::Vector5 &b,
//                                             gtsam::Matrix *H) const override
//         {
//             // Extract fridge parameters.
//             double width = b(0);
//             double depth = b(1);
//             double angle = b(2);
//             double centerX = b(3);
//             double centerY = b(4);
//
//             // Initialize total error.
//             double totalError = 0.0;
//             // If a Jacobian is requested, initialize a 1x5 row vector to zero.
//             Eigen::Matrix<double, 1, 5> J;
//             if (H) { J.setZero(); }
//
//             // Loop over each LiDAR point.
//             for (std::size_t i = 0; i < points_.size(); i++)
//             {
//                 // Get the LiDAR point (in room frame).
//                 double pt_x = points_[i].x();
//                 double pt_y = points_[i].y();
//
//                 // Transform the point to the fridge frame.
//                 // 1. Translate by the fridge center.
//                 double dx = pt_x - centerX;
//                 double dy = pt_y - centerY;
//                 // 2. Rotate by -angle.
//                 double cosA = std::cos(angle);
//                 double sinA = std::sin(angle);
//                 double fx = cosA * dx + sinA * dy;
//                 double fy = -sinA * dx + cosA * dy;
//
//                 // In the fridge frame, the fridge is represented as a rectangle centered at (0,0)
//                 // with half-extents: halfW = width/2 and halfD = depth/2.
//                 double halfW = width / 2.0;
//                 double halfD = depth / 2.0;
//
//                 // Define a simple error: for example, let the cost be the difference between the absolute
//                 // x-coordinate (in fridge frame) and half the width.
//                 // (This is a simplification. In practice you might compute a smooth minimal distance to all four edges.)
//                 double error_i = std::fabs(fx) - halfW;
//                 totalError += error_i;
//
//                 // Compute the Jacobian for this point if requested.
//                 if (H)
//                 {
//                     // Let u = fx = cosA*dx + sinA*dy.
//                     // Then error_i = |u| - halfW.
//                     // The derivative d|u|/du is sign(u) (undefined at u = 0; here we assume u != 0).
//                     double sign_u = (fx >= 0.0) ? 1.0 : -1.0;
//                     // Derivatives of u with respect to the fridge parameters:
//                     // u does not depend on width or depth.
//                     double du_dangle = -sinA * dx + cosA * dy;
//                     double du_dcx = -cosA;
//                     double du_dcy = -sinA;
//                     // Also, the derivative of halfW with respect to width is 0.5.
//                     double dhalfW_dwidth = 0.5;
//
//                     // The error_i derivative with respect to each parameter is:
//                     // d(error_i)/d(width) = 0 (from u) - dhalfW/d(width)
//                     double derror_dwidth = -dhalfW_dwidth;
//                     // d(error_i)/d(depth) = 0 (no dependency)
//                     double derror_ddepth = 0.0;
//                     // d(error_i)/d(angle) = sign(u) * du/d(angle)
//                     double derror_dangle = sign_u * du_dangle;
//                     // d(error_i)/d(center_x) = sign(u) * du/d(center_x)
//                     double derror_dcx = sign_u * du_dcx;
//                     // d(error_i)/d(center_y) = sign(u) * du/d(center_y)
//                     double derror_dcy = sign_u * du_dcy;
//
//                     // Accumulate into the Jacobian matrix.
//                     J(0, 0) += derror_dwidth;
//                     J(0, 1) += derror_ddepth;
//                     J(0, 2) += derror_dangle;
//                     J(0, 3) += derror_dcx;
//                     J(0, 4) += derror_dcy;
//                 }
//             } // end for
//
//             if (H)
//             {
//                 *H = J;
//             }
//             gtsam::Vector err(1);
//             err(0) = totalError;
//             return err;
//         }
// };
//
// #endif // FRIDGEPOINTSFECTOR_H

//class FridgePointsFactor final : public gtsam::NoiseModelFactorN<gtsam::Vector5>
//{
//public:
//    // Fixed LiDAR points in the robot's frame.
//    std::vector<Eigen::Vector3d> points_;
//    // Transformation from the room frame to the robot frame.
//    Eigen::Affine2d T_R_to_r_;
//    // Softmin softness parameter.
//    double k_softmin_;
//    // Constructor
//    FridgePointsFactor(const gtsam::Key fridgeKey,
//                       const std::vector<Eigen::Vector3d>& points,
//                       const Eigen::Affine2d& T_R_to_r,
////                       const std::shared_ptr<gtsam::noiseModel::Isotropic>& noiseModel,
//                       const gtsam::noiseModel::Diagonal::shared_ptr& noiseModel,
//                       const double k_softmin = 10.0)
//            : gtsam::NoiseModelFactorN<gtsam::Vector5>
//                      (noiseModel, fridgeKey),
//              points_(points),
//              T_R_to_r_(T_R_to_r),
//              k_softmin_(k_softmin)
//    {}
//
//    // Método para calcular el error
//    gtsam::Vector evaluateError(const gtsam::Vector5 &b, gtsam::Matrix *H) const override
//    {
//        // Unpack fridge parameters: [width, depth, angle, center_x, center_y]
//        const double width   = b(0);
//        const double depth   = b(1);
//        const double angle   = b(2);
//        const double centerX = b(3);
//        const double centerY = b(4);
//
//        double totalError = 0.0;
//        Eigen::Matrix<double, 1, 5> J;
//        if (H) { J.setZero(); }
//
//        // Loop over each LiDAR point.
//        for (const auto &p : points_)
//        {
//            // --- Transform point from robot frame to room frame ---
//            // Note: Our transform T_R_to_r_ maps room -> robot,
//            // so its inverse maps robot -> room.
//            Eigen::Vector2d p_room = T_R_to_r_ * Eigen::Vector2d(p.x(), p.y()); // direct transformation going upwards in tree
//
//            // --- Build candidate fridge transform T_R_to_f from room to fridge ---
//            // The candidate fridge transform is built from the fridge parameters.
//            // Translation by (centerX, centerY) and rotation by angle.
//            Eigen::Affine2d T_R_to_f = Eigen::Translation2d(centerX, centerY) * Eigen::Rotation2Dd(angle);
//            // Transform p_room to the fridge frame.
//            Eigen::Vector2d p_fridge = T_R_to_f.inverse() * p_room; // inverse transformation going downwards in tree
//
//            // --- Compute distances in fridge frame ---
//            // In the fridge's local frame, the fridge is represented as a rectangle centered at (0,0)
//            // with half-extents: halfW = width/2 and halfD = depth/2.
//            const double halfW = width / 2.0;
//            const double halfD = depth / 2.0;
//
//            // Instead of using std::fabs, we use a smooth absolute value function.
//            // Here is one simple differentiable approximation:
//            auto smoothAbs = [](const double x, const double eps = 1e-4) -> double
//            { return std::sqrt(x * x + eps); };
//
//            // Compute smooth absolute differences.
//            const double d_front = smoothAbs(p_fridge(0) - halfW);
//            const double d_back  = smoothAbs(p_fridge(0) + halfW);
//            const double d_left  = smoothAbs(p_fridge(1) - halfD);
//            const double d_right = smoothAbs(p_fridge(1) + halfD);
//
//            // Compute the softmin of the four distances.
//            std::vector<double> dists = { d_front, d_back, d_left, d_right };
//            const double pointCost = softmin(dists, k_softmin_);
//            totalError += pointCost;
//
//            if (H)
//            {
//                // Compute derivatives for softmin.
//                double sumExp = 0.0;
//                double sumExpTimes = 0.0;
//                for (size_t j = 0; j < dists.size(); j++) {
//                    const double expVal = std::exp(-k_softmin_ * dists[j]);
//                    sumExp += expVal;
//                    sumExpTimes += dists[j] * expVal;
//                }
//                std::vector<double> dSoftmin_dd(4, 0.0);
//                for (size_t j = 0; j < dists.size(); j++) {
//                    const double expVal = std::exp(-k_softmin_ * dists[j]);
//                    // Derivative of softmin: d/d d_j (sumExpTimes/sumExp)
//                    dSoftmin_dd[j] = (expVal * (1 - k_softmin_ * dists[j]) * sumExp + k_softmin_ * expVal * sumExpTimes) / (sumExp * sumExp);
//                }
//
//                // Next, compute derivatives of each distance with respect to fridge parameters.
//                // First, compute the transformation details.
//                const double dx = p_room(0) - centerX;
//                const double dy = p_room(1) - centerY;
//                const double cosA = std::cos(angle);
//                const double sinA = std::sin(angle);
//                const double fx = cosA * dx + sinA * dy;
//                const double fy = -sinA * dx + cosA * dy;
//
//                // For the smooth absolute, we differentiate smoothAbs(x) = sqrt(x^2 + eps).
//                auto dsmoothAbs = [=](double x, double eps = 1e-4) -> double {
//                    return x / std::sqrt(x * x + eps);
//                };
//
//                // For d_front: function is smoothAbs(fx - halfW)
//                const double dfront_dfx = dsmoothAbs(fx - halfW);
//                const double dfront_dwidth = -0.5 * dsmoothAbs(fx - halfW);
//                // For d_back: function is smoothAbs(fx + halfW)
//                const double dback_dfx = dsmoothAbs(fx + halfW);
//                const double dback_dwidth = 0.5 * dsmoothAbs(fx + halfW);
//                // For d_left: function is smoothAbs(fy - halfD)
//                const double dleft_dfy = dsmoothAbs(fy - halfD);
//                const double dleft_ddepth = -0.5 * dsmoothAbs(fy - halfD);
//                // For d_right: function is smoothAbs(fy + halfD)
//                const double dright_dfy = dsmoothAbs(fy + halfD);
//                const double dright_ddepth = 0.5 * dsmoothAbs(fy + halfD);
//
//                // Now compute derivatives of fx and fy with respect to angle, centerX, centerY.
//                const  double dfx_dangle = -sinA * dx + cosA * dy;
//                const double dfx_dcx = -cosA;
//                const double dfx_dcy = -sinA;
//                const double dfy_dangle = -cosA * dx - sinA * dy;
//                const double dfy_dcx = sinA;
//                const double dfy_dcy = -cosA;
//
//                // Chain rule to compute derivative for each parameter.
//                const double dcost_dwidth = dSoftmin_dd[0] * dfront_dwidth + dSoftmin_dd[1] * dback_dwidth;
//                const double dcost_ddepth = dSoftmin_dd[2] * dleft_ddepth + dSoftmin_dd[3] * dright_ddepth;
//                const double dcost_dangle = dSoftmin_dd[0] * (dfront_dfx * dfx_dangle)
//                                            + dSoftmin_dd[1] * (dback_dfx * dfx_dangle)
//                                            + dSoftmin_dd[2] * (dleft_dfy * dfy_dangle)
//                                            + dSoftmin_dd[3] * (dright_dfy * dfy_dangle);
//                const double dcost_dcx = dSoftmin_dd[0] * (dfront_dfx * dfx_dcx)
//                                         + dSoftmin_dd[1] * (dback_dfx * dfx_dcx)
//                                         + dSoftmin_dd[2] * (dleft_dfy * dfy_dcx)
//                                         + dSoftmin_dd[3] * (dright_dfy * dfy_dcx);
//                const double dcost_dcy = dSoftmin_dd[0] * (dfront_dfx * dfx_dcy)
//                                         + dSoftmin_dd[1] * (dback_dfx * dfx_dcy)
//                                         + dSoftmin_dd[2] * (dleft_dfy * dfy_dcy)
//                                         + dSoftmin_dd[3] * (dright_dfy * dfy_dcy);
//
//                // Accumulate into Jacobian.
//                J(0, 0) += dcost_dwidth;
//                J(0, 1) += dcost_ddepth;
//                J(0, 2) += dcost_dangle;
//                J(0, 3) += dcost_dcx;
//                J(0, 4) += dcost_dcy;
//            }
//        } // end loop over points
//
//        if (H) { *H = J; }
//        gtsam::Vector err(1);
//        err(0) = totalError;
//        return err;
//    }
//
//private:
//    // Softmin: a differentiable approximation to the minimum.
//    // dists: a vector of distances, k: softness parameter.
//    template <typename T>
//    T softmin(const std::vector<T>& dists, const T& k) const
//    {
//        T numerator = T(0);
//        T denominator = T(0);
//        for (size_t i = 0; i < dists.size(); i++)
//        {
//            T w = std::exp(-k * dists[i]);
//            numerator += dists[i] * w;
//            denominator += w;
//        }
//        return numerator / denominator;
//    }
//};
