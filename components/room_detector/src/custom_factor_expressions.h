#ifndef FRIDGEPOINTSFACTOR_H
#define FRIDGEPOINTSFACTOR_H

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <Eigen/Core>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>

namespace factors
{
    static gtsam::Vector2 project_pose(const gtsam::Vector2 &v, gtsam::OptionalJacobian<2, 2> H)
    {
        if (H)
            *H = gtsam::Matrix22::Identity();
//        std::cout << __FUNCTION__ << " Center: " << v.transpose() << std::endl;
        return gtsam::Vector2(v.x(), v.y());
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// FRIDGE custom factor expressions
    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////// Auxiliary functions


    /**
        * \brief Returns the squared y coordinate as the distance to the side/wall.
        *
        * \param p Point in fridge coordinates.
        * \param H Optional Jacobian for the distance with respect to the point.
        * \return Distance to the side of the fridge.
        */
    static double normalize (const gtsam::Vector1 denom, const double val, gtsam::OptionalJacobian<1, 1> H1, gtsam::OptionalJacobian<1, 1> H2)
    {
        if (H1) *H1 = gtsam::Matrix11::Zero();
        if (H2) *H2 = gtsam::Matrix11(1.0 / denom(0));
//        std::cout << "Normalized value: " << val/denom(0) << " " << denom(0) << std::endl;
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
        // Derivada de la sigmoide: sigma'(x) = sigma(x)*(1 - sigma(x))
        double s = logistic(x);
        return s * (1.0 - s);
    }

    // Softmin function
    static double softmin_local(const std::vector<double>& dists, const double k=1.0)
    {
        double numerator = 0.0;
        double denominator = 0.0;
        for (size_t i = 0; i < dists.size(); i++)
        {
            double w = std::exp(-k * dists[i]);
            numerator += dists[i] * w;
            denominator += w;
        }
        return numerator / denominator;
    }

    // Derivative of softmin_local
    static std::vector<double> dsoftmin_local_dd(const std::vector<double>& dists, const double k)
    {
        double S = 0.0, N = 0.0;
        for (size_t i = 0; i < dists.size(); i++)
        {
            double e = std::exp(-k * dists[i]);
            S += e;
            N += dists[i] * e;
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
     static double softplus(double x)
    {
        // Using log1p(exp(x)) is numerically more stable for small x.
        return std::log1p(std::exp(x));
    }

    // The derivative of softplus is the logistic function.
     static double softplusDerivative(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }
    //////////////////////////////////////////////////////////////////////////////////////

    /**
    * @brief Calculates the (smoothed) distance from a point p (in the local frame of the 'side')
    * to the corresponding segment of the rectangle defined by v = [cx, cy, alpha, width, depth].
    *
    * @param v    Vector5 with (cx, cy, alpha, width, depth)
    * @param side Vector1 with the side index (1..4), following the clockwise order:
    *             1: top side,    2: right side,
    *             3: bottom side, 4: left side.
    * @param p    Vector2 with the position of the point in the local frame of that side.
    * @param H1   (optional) 1x5 Jacobian w.r.t. v
    * @param H2   (optional) 1x1 Jacobian w.r.t. side
    * @param H3   (optional) 1x2 Jacobian w.r.t. p
    * @return     Smoothed scalar distance.
    *
    * It is assumed that in this local frame:
    *  side=1 -> segment from (-width/2, +depth/2) to (+width/2, +depth/2)
    *  side=2 -> segment from (+width/2, +depth/2) to (+width/2, -depth/2)
    *  side=3 -> segment from (+width/2, -depth/2) to (-width/2, -depth/2)
    *  side=4 -> segment from (-width/2, -depth/2) to (-width/2, +depth/2)
    *
    * A sigmoid is used to smooth the transition between "interior projection" and
    * "near the ends".
    */
    static double dist2seg(const gtsam::Vector5 &v, const double side, const gtsam::Point2 &p, gtsam::OptionalJacobian<1, 5> H1,  gtsam::OptionalJacobian<1, 1> H2, gtsam::OptionalJacobian<1, 2> H3)
    {
        // Rectangle parameters
        const double w     = v(3);  // width
        const double d     = v(4);  // depth

        constexpr double beta = 20.0;

        // Definir los extremos del segmento en este marco local, según la cara 'side'.
        Eigen::Vector2d A_local, B_local;
        int s = static_cast<int>(side);
        switch(s) {
            case 1: // lado superior
                A_local = Eigen::Vector2d(-w/2.0, +d/2.0);
                B_local = Eigen::Vector2d(+w/2.0, +d/2.0);
                break;
            case 2: // lado derecho
                A_local = Eigen::Vector2d(+w/2.0, +d/2.0);
                B_local = Eigen::Vector2d(+w/2.0, -d/2.0);
                break;
            case 3: // lado inferior
                A_local = Eigen::Vector2d(+w/2.0, -d/2.0);
                B_local = Eigen::Vector2d(-w/2.0, -d/2.0);
                break;
            case 4: // lado izquierdo
                A_local = Eigen::Vector2d(-w/2.0, -d/2.0);
                B_local = Eigen::Vector2d(-w/2.0, +d/2.0);
                break;
            default:
                // side inválido
                if (H1) H1->setZero();
                if (H2) H2->setZero();
                return 9999.9;
        }

        // Definir los vectores del segmento y del punto a uno de los extremos
        const Eigen::Vector2d AB = B_local - A_local;  // segmento
        const Eigen::Vector2d AP = p - A_local;        // vector A->P
        const double AB2 = AB.dot(AB);                // |AB|^2

        // Si el segmento es degenerado (w=0 y d=0?), distancia = p a A_local
        if (AB2 < 1e-15) {
            const double dist_degen = (p - A_local).norm();
            if (H1) H1->setZero();
            if (H2) H2->setZero();
            return dist_degen;
        }

        // Cálculo de la proyección "t" en la línea infinita
        //    t < 0 => proyección cae antes de A
        //    t > 1 => proyección cae después de B
        //    0 <= t <= 1 => proyección en [A,B]
        double t = AP.dot(AB) / AB2;

        // Definición de sigmoides para suavizar la "saturación" de t en [0,1]
        //    s0 ~ "estamos por encima de t=0",  s1 ~ "estamos por encima de t=1"
        double s0 = logistic(beta * t);         // pasa de ~0 a ~1 cerca de t=0
        double s1 = logistic(beta * (t - 1.0)); // pasa de ~0 a ~1 cerca de t=1

        // "pesos" suaves para cada región
        //  - w_in:  en el interior (0 < t < 1)
        //  - w_left: cerca de A
        //  - w_right: cerca de B
        double w_in   = s0 * (1.0 - s1);
        double w_left = (1.0 - s0);
        double w_right= s1;

        // Partial distances
        //    - dist_in: distance to the infinite line => projection
        //    - dist_left: distance to A_local
        //    - dist_right: distance to B_local
        // We mix with the weights to get a single continuous distance.

        Eigen::Vector2d closest_in = A_local + t * AB;  // proyección en la línea
        Eigen::Vector2d diff_in = p - closest_in;
        double dist_in = diff_in.norm();

        Eigen::Vector2d diff_left = p - A_local;
        double dist_left = diff_left.norm();

        Eigen::Vector2d diff_right = p - B_local;
        double dist_right = diff_right.norm();

        // Smoothed final distance
        double dist = w_in * dist_in + w_left * dist_left + w_right * dist_right;

        // qDebug() << "w_in: " << w_in << " w_left: " << w_left << " w_right: " << w_right;
        // qDebug() << "dist_in: " << dist_in << " dist_left: " << dist_left << " dist_right: " << dist_right;
        // qDebug() << "Dist2seg: " << dist << " side: " << side;

        // Cálculo de los jacobianos si se solicitan
        // ------------------------------------------------------------
        // Derivadas respecto a v = (cx, cy, alpha, w, d) (H1) ---
        // En este marco local NO estamos usando (cx, cy, alpha), así que sus derivadas son 0. Solo w y d cambian A_local, B_local.

        if (H1)
        {
            // Preparamos contenedores:
            Eigen::Vector2d dA_dw, dA_dd, dB_dw, dB_dd;
            dA_dw.setZero();
            dA_dd.setZero();
            dB_dw.setZero();
            dB_dd.setZero();

            // Rellenar según side:
            // side=1 => A=(-w/2, +d/2), B=(+w/2, +d/2)
            // => dA/dw= (-1/2, 0), dA/dd= (0, +1/2)
            // => dB/dw= (+1/2, 0), dB/dd= (0, +1/2)
            switch (s)
            {
                case 1:
                    dA_dw = Eigen::Vector2d(-0.5, 0.0);
                    dA_dd = Eigen::Vector2d(0.0, 0.5);
                    dB_dw = Eigen::Vector2d(0.5, 0.0);
                    dB_dd = Eigen::Vector2d(0.0, 0.5);
                    break;
                case 2:
                    // A=(+w/2,+d/2), B=(+w/2,-d/2)
                    // => A_w=(+0.5,0), A_d=(0,+0.5)
                    // => B_w=(+0.5,0), B_d=(0,-0.5)
                    dA_dw = Eigen::Vector2d(0.5, 0.0);
                    dA_dd = Eigen::Vector2d(0.0, 0.5);
                    dB_dw = Eigen::Vector2d(0.5, 0.0);
                    dB_dd = Eigen::Vector2d(0.0, -0.5);
                    break;
                case 3:
                    // A=(+w/2,-d/2), B=(-w/2,-d/2)
                    dA_dw = Eigen::Vector2d(0.5, 0.0);
                    dA_dd = Eigen::Vector2d(0.0, -0.5);
                    dB_dw = Eigen::Vector2d(-0.5, 0.0);
                    dB_dd = Eigen::Vector2d(0.0, -0.5);
                    break;
                case 4:
                    // A=(-w/2,-d/2), B=(-w/2,+d/2)
                    dA_dw = Eigen::Vector2d(-0.5, 0.0);
                    dA_dd = Eigen::Vector2d(0.0, -0.5);
                    dB_dw = Eigen::Vector2d(-0.5, 0.0);
                    dB_dd = Eigen::Vector2d(0.0, 0.5);
                    break;
            }

            // 2) AB= B_local - A_local => dAB/dw= dB_dw - dA_dw, etc.
            Eigen::Vector2d dAB_dw = dB_dw - dA_dw;
            Eigen::Vector2d dAB_dd = dB_dd - dA_dd;

            // 3) AP= p - A_local => dAP/dw= -dA_dw, dAP/dd= -dA_dd
            Eigen::Vector2d dAP_dw = -dA_dw;
            Eigen::Vector2d dAP_dd = -dA_dd;

            // 4) t= AP·AB / (AB·AB).  =>
            //    dt/dw= [ (dAP_dw·AB + AP·dAB_dw)*(AB·AB) - (AP·AB)*2(AB·dAB_dw) ] / (AB·AB)^2
            //    (análogo para dd).
            //    *Hay que hacerlo con cuidado. Aquí, por brevedad, mostramos la idea:
            double ABdotAB = AB2;  // denominador
            double APdotAB = AP.dot(AB);

            // Función auxiliar para derivar t wrt w/d
            auto dt = [&](const Eigen::Vector2d &dAP_dv, const Eigen::Vector2d &dAB_dv) {
                double num = (dAP_dv.dot(AB) + AP.dot(dAB_dv)) * ABdotAB
                             - APdotAB * 2.0 * AB.dot(dAB_dv);
                return num / (ABdotAB * ABdotAB);
            };
            double dt_dw = dt(dAP_dw, dAB_dw);
            double dt_dd = dt(dAP_dd, dAB_dd);

            double ds0_dt = beta * dlogistic(beta * t);
            double ds1_dt = beta * dlogistic(beta * (t - 1.0));

            // w_in= s0*(1-s1), etc.
            auto dw_in_dt = ds0_dt * (1.0 - s1) - s0 * ds1_dt;
            auto dw_left_dt = -ds0_dt; // (1 - s0)
            auto dw_right_dt = ds1_dt;  // s1

            // Derivadas w.r.t. w o d se obtienen por la cadena:
            // dw_in/dw = dw_in/dt * dt/dw
            double dw_in_dw = dw_in_dt * dt_dw;
            double dw_left_dw = dw_left_dt * dt_dw;
            double dw_right_dw = dw_right_dt * dt_dw;
            double dw_in_dd = dw_in_dt * dt_dd;
            double dw_left_dd = dw_left_dt * dt_dd;
            double dw_right_dd = dw_right_dt * dt_dd;

            // dist_in= norm(p - (A_local + t*AB)) => depende de w,d vía A_local, AB, t
            // Podemos derivar "dist_in" w.r.t. w =>
            // d(dist_in)/dw= (1/dist_in)* [ (p - (A + t*AB)) · ( - (dA_dw + t*dAB_dw) - AB * dt_dw ) ]
            // etc. Por brevedad, lo hacemos en pasos:

            // closest_in = A_local + t*AB
            // => d(closest_in)/dw = dA_dw + t*dAB_dw + AB*dt_dw
            // (¡ojo!: la parte AB*dt_dw va multiplicando dt_dw, no sumando si no se hace carefully).
            // Realmente: d(closest_in)/dw = dA_dw + [ d(t*AB)/dw ]
            // => d(t*AB)/dw= d(t)/dw * AB + t*d(AB)/dw

            auto dClosest_in = [&](double dt_dv, const Eigen::Vector2d &dA_dv, const Eigen::Vector2d &dAB_dv) {
                return dA_dv + dt_dv * AB + t * dAB_dv;
            };

            Eigen::Vector2d dClosest_in_dw = dClosest_in(dt_dw, dA_dw, dAB_dw);
            Eigen::Vector2d dClosest_in_dd = dClosest_in(dt_dd, dA_dd, dAB_dd);

            // dist_in= || p - closest_in ||
            // => d(dist_in)/dw= (1/dist_in)* (p - closest_in) · ( - dClosest_in/dw )
            auto dDist_in = [&](const Eigen::Vector2d &dClosest_dv) {
                // derivada escalar:
                double val = 0.0;
                if (dist_in > 1e-12)
                {
                    val = (diff_in.dot(-dClosest_dv)) / dist_in;
                }
                return val;
            };
            double ddist_in_dw = dDist_in(dClosest_in_dw);
            double ddist_in_dd = dDist_in(dClosest_in_dd);

            // dist_left= || p - A_local ||
            // => d(dist_left)/dw= (1/dist_left)* (p - A_local)·(-dA_dw)
            auto dDist_simple = [&](double dist_val, const Eigen::Vector2d &diff,
                                    const Eigen::Vector2d &dA_dv) {
                if (dist_val < 1e-12) return 0.0;
                return (diff.dot(-dA_dv)) / dist_val;
            };
            double ddist_left_dw = dDist_simple(dist_left, diff_left, dA_dw);
            double ddist_left_dd = dDist_simple(dist_left, diff_left, dA_dd);
            double ddist_right_dw = dDist_simple(dist_right, diff_right, dB_dw);
            double ddist_right_dd = dDist_simple(dist_right, diff_right, dB_dd);

            // Finalmente, dist= w_in*dist_in + w_left*dist_left + w_right*dist_right
            // => d(dist)/dw = ...
            auto dDist_dv = [&](double dw_in_dv, double dw_left_dv, double dw_right_dv,
                                double ddist_in_dv, double ddist_left_dv, double ddist_right_dv) {
                return dw_in_dv * dist_in + w_in * ddist_in_dv
                       + dw_left_dv * dist_left + w_left * ddist_left_dv
                       + dw_right_dv * dist_right + w_right * ddist_right_dv;
            };

            double dDist_dw = dDist_dv(dw_in_dw, dw_left_dw, dw_right_dw,
                                       ddist_in_dw, ddist_left_dw, ddist_right_dw);
            double dDist_dd = dDist_dv(dw_in_dd, dw_left_dd, dw_right_dd,
                                       ddist_in_dd, ddist_left_dd, ddist_right_dd);

            // Asignamos:
            // dDist/d(cx)=0, dDist/d(cy)=0, dDist/d(alpha)=0, dDist/d(width)= dDist_dw, dDist/d(depth)= dDist_dd
            //print dDist_dw, dDist_dd
            H1->setZero(); // (1x5)
            (*H1)(0, 3) = dDist_dw; // wrt width
            (*H1)(0, 4) = dDist_dd; // wrt depth
        }
        // Derivadas respecto a side (H2) es zero
        if (H2) H2->setZero();

        // Derivadas respecto a p (H3) ---
        if (H3)
        {
            Eigen::RowVector2d J_p = Eigen::RowVector2d::Zero();
            // d(dist_in)/dp = (p - closest_in)^T / ||p - closest_in||
            double norm_in = dist_in;
            Eigen::RowVector2d d_dist_in_dp = (norm_in > 1e-6) ? Eigen::RowVector2d(diff_in.transpose() / norm_in) : Eigen::RowVector2d::Zero();
            // d(dist_left)/dp = (p - A_local)^T / ||p - A_local||
            double norm_left = dist_left;
            Eigen::RowVector2d d_dist_left_dp = (norm_left > 1e-6) ? Eigen::RowVector2d(diff_left.transpose() / norm_left) : Eigen::RowVector2d::Zero();
            // d(dist_right)/dp = (p - B_local)^T / ||p - B_local||
            double norm_right = dist_right;
            Eigen::RowVector2d d_dist_right_dp = (norm_right > 1e-6) ? Eigen::RowVector2d(diff_right.transpose() / norm_right) : Eigen::RowVector2d::Zero();

            J_p = w_in * d_dist_in_dp + w_left * d_dist_left_dp + w_right * d_dist_right_dp;
            *H3 = J_p;
        }
        return dist;
    }

/**
    * @brief Calculates the (smoothed) squared distance from a point p (in the local frame of the 'side')
    * to the corresponding segment of the rectangle defined by v = [cx, cy, alpha, width, depth].
    *
    * @param v    Vector5 with (cx, cy, alpha, width, depth)
    * @param side Vector1 with the side index (1..4), following the clockwise order:
    *             1: top side,    2: right side,
    *             3: bottom side, 4: left side.
    * @param p    Vector2 with the position of the point in the local frame of that side.
    * @param H1   (optional) 1x5 Jacobian w.r.t. v
    * @param H2   (optional) 1x1 Jacobian w.r.t. side
    * @param H3   (optional) 1x2 Jacobian w.r.t. p
    * @return     Smoothed scalar distance.
    *
    * It is assumed that in this local frame:
    *  side=1 -> segment from (-width/2, +depth/2) to (+width/2, +depth/2)
    *  side=2 -> segment from (+width/2, +depth/2) to (+width/2, -depth/2)
    *  side=3 -> segment from (+width/2, -depth/2) to (-width/2, -depth/2)
    *  side=4 -> segment from (-width/2, -depth/2) to (-width/2, +depth/2)
    *
    * A sigmoid is used to smooth the transition between "interior projection" and
    * "near the ends".
    */

    static double dist2seg_sqr(const gtsam::Vector5 &v, const double side, const gtsam::Point2 &p, gtsam::OptionalJacobian<1, 5> H1,  gtsam::OptionalJacobian<1, 1> H2, gtsam::OptionalJacobian<1, 2> H3)
    {
        // Extract fridge parameters.
        const double width = v(3);
        const double depth = v(4);

        // Define endpoints A_local and B_local for each side in the sides' local frame.
        Eigen::Vector2d A_local, B_local;
        int s = static_cast<int>(side);
        switch (s)
        {
            case 1: // top side: from (-width/2, 0) to (width/2, 0)
                A_local = Eigen::Vector2d(-width/2.0, 0.0);
                B_local = Eigen::Vector2d( width/2.0, 0.0);
                break;
            case 2: // right side: from (-depth/2, 0) to (depth/2, 0)
                A_local = Eigen::Vector2d(-depth/2.0, 0.0);
                B_local = Eigen::Vector2d( depth/2.0, 0.0);
                break;
            case 3: // bottom side: from (width/2, 0) to (-width/2, 0)
                A_local = Eigen::Vector2d( -width/2.0, 0.0);
                B_local = Eigen::Vector2d(width/2.0, 0.0);
                break;
            case 4: // left side: from (depth/2, 0) to (-depth/2, 0)
                A_local = Eigen::Vector2d( -depth/2.0, 0.0);
                B_local = Eigen::Vector2d(  depth/2.0, 0.0);
                break;
            default:
                if(H1) H1->setZero();
                if(H2) H2->setZero();
                if(H3) H3->setZero();
                return 9999.9;
    	}

	    // Compute the projection of the point onto the segment.
	    const Eigen::Vector2d AB = B_local - A_local;
	    const double t = p.x() / (B_local.x() - A_local.x()) + 0.5; // shifted to match the normalized range [0, 1]

	    // // Definition of sigmoids to smooth the "saturation" of t in [0,1]:  s0 ~ above t=0",  s1 ~ above de t=1
	    constexpr double beta = 10.0; // Steepness of the sigmoid. Higher makes the transition more abrupt
	    const double s0 = logistic(beta * t);         // pasa de ~0 a ~1 cerca de t=0
	    const double s1 = logistic(beta * (t - 1.0)); // pasa de ~0 a ~1 cerca de t=1

	    //  soft weights for each region: w_in:  inside (0 < t < 1)  - w_left: close to A - w_right: close to B
	    const double w_in   = s0 * (1.0 - s1);
	    const double w_left = (1.0 - s0);
	    const double w_right= s1;

	    // Compute squared distances:
	    const double d_in = p.y() * p.y();
	    // d_left = || p - A_local ||^2.
	    const double d_left = (Eigen::Vector2d(p.x(), p.y()) - A_local).squaredNorm();
	    // d_right = || p - B_local ||^2.
	    const double d_right = (Eigen::Vector2d(p.x(), p.y()) - B_local).squaredNorm();

	    // Smoothed final distance
	    const double dist = w_in * d_in + w_left * d_left + w_right * d_right;

        /////////////////////////////////////
        // compute Jacobians if requested
        ////////////////////////////////////

        // current side length
        const double sd = AB.norm();  // it is selected in the switch statement
        if (H1) // wrt to the fridge params
        {
            // Compute derivatives of s0 and s1 with respect to t.
            double ds0_dt = s0 * (1.0 - s0) * beta;
            double ds1_dt = s1 * (1.0 - s1) * beta;

            // dt/dw = -p.x() / (side_width^2).
            double dt_dw = -p.x() / (sd * sd);  // esto está probablemente mal. Hay dos derivadas aquí, wrt to w y d.
            //double dt_dw = -p.x() / std::pow(B_local.x() - A_local.x(), 2) * (dB_dw.x() - dA_dw.x());
            //double dt_dd = -p.x() / std::pow(B_local.x() - A_local.x(), 2) * (dB_dd.x() - dA_dd.x());

            // Derivative of s0 with respect to w.
            double ds0_dw = ds0_dt * dt_dw; // = - (beta * p.x()/(w^2)) * s0*(1-s0)
            // Derivative of s1 with respect to w.
            double ds1_dw = ds1_dt * dt_dw; // = - (beta * p.x()/(w^2)) * s1*(1-s1)

            // Now, weights:
            // w_in = s0*(1-s1), w_left = 1-s0, w_right = s1.
            // Their derivatives with respect to w:
            double dw_in_dw = ds0_dw * (1 - s1) - s0 * ds1_dw;
            double dw_left_dw = - ds0_dw;
            double dw_right_dw = ds1_dw;

            // Squared distances:
            // d_in = p.y()^2, (dd/dw = 0)
            // d_left = (p.x() + w/2)^2 + p.y()^2, so derivative w.r.t. w is:
            double dd_left_dw = p.x() + sd / 2.0;
            // d_right = (p.x() - w/2)^2 + p.y()^2, so derivative w.r.t. w is:
            double dd_right_dw = -(p.x() - sd / 2.0);

            // Compute the derivative of f(v,p) = w_in*d_in + w_left*d_left + w_right*d_right
            // with respect to w. (d_in is independent of w.)
            double dfdw = dw_in_dw * d_in
                          + dw_left_dw * d_left + w_left * dd_left_dw
                          + dw_right_dw * d_right + w_right * dd_right_dw;

            // Since f does not depend on cx,cy,alpha in this local formulation,
            // we assign the width derivative to the fourth element and zeros elsewhere.
            gtsam::Vector5 Jv = gtsam::Vector5::Zero();
            if (sd == 1 or sd == 3) Jv(3) = dfdw;
            if (sd == 2 or sd == 4) Jv(4) = dfdw;
            *H1 = Jv;
        }
        if (H2) H2->setZero();  // derivative with respect to the side
        if (H3) // derivative with respect to the point
        {
            // derivative of the in-segment distance with respect to the point  y*y
            // Compute derivatives of logistic functions.
            const double ds0_dt = dlogistic(beta * t) * beta;         // d s0/dt.
            const double ds1_dt = dlogistic(beta * (t - 1.0)) * beta;   // d s1/dt.

            // t depends on p_x only: dt/dp_x = 1/width, dt/dp_y = 0.
            const double dt_dp0 = 1.0 / sd;

            // Thus, the derivative of w_in with respect to p_x is:
            const double dw_in_dp0 = (ds0_dt * (1.0 - s1) - s0 * ds1_dt) * dt_dp0;

            // The interior squared distance is: d_in = (p_y)^2.
            // Its derivative with respect to p is [0, 2*p_y].
            const double dfdp_x = dw_in_dp0 * (p.y() * p.y());          // derivative w.r.t. p_x.
            const double dfdp_y = w_in * (2.0 * p.y());                // derivative w.r.t. p_y.
            Eigen::RowVector2d dfdp_center;
            dfdp_center << dfdp_x, dfdp_y;

            // Derivatives of the left term f_left = w_left * d_left with respect to p are:

            // dt/dp_x = 1/w, dt/dp_y = 0.
            const double dt_dp_x = 1.0 / sd;

            // Therefore, derivative of s0 w.r.t p_x:
            const double ds0_dp_x = ds0_dt * dt_dp_x;

            // Derivative of w_left w.r.t p_x:
            const double dw_left_dp_x = - ds0_dp_x;
            const double dw_left_dp_y = 0.0;

            // Derivative of d_left with respect to p:
            // ∇_p d_left = 2*(p - A_local)^T.
            const Eigen::RowVector2d dd_left_dp = 2.0 * (Eigen::Vector2d(p.x(), p.y()) - A_local).transpose();

            // Now, the derivative of the left term f_left = w_left * d_left with respect to p is:
            // ∇_p f_left = (∇_p w_left)*d_left + w_left * (∇_p d_left).
            Eigen::RowVector2d dfdp_left;
            dfdp_left << (dw_left_dp_x * d_left), (dw_left_dp_y * d_left);
            dfdp_left += w_left * dd_left_dp;

            // the derivative of the right term f_right = w_right * d_right with respect to p is:

            // derivative of s1 with respect to p.x):
            const double ds1_dp_x = ds1_dt * dt_dp_x;
            const double ds1_dp_y = 0.0;

            // So the derivative of w_right = s1 is:
            const double dwright_dp_x = ds1_dp_x;
            const double dwright_dp_y = ds1_dp_y;

            // Compute d_right = || p - B_local ||^2.
            double d_right = (p - B_local).squaredNorm();

            // Derivative of d_right with respect to p is 2*(p - B_local)^T.
            Eigen::RowVector2d dd_right_dp = 2.0 * (p - B_local).transpose();

            // Now, by the product rule:
            // d(f_right)/dp = (d(w_right)/dp) * d_right + w_right * (d(d_right)/dp)
            Eigen::RowVector2d dfdp;
            dfdp << (dwright_dp_x * d_right) + s1 * dd_right_dp(0),
                    (dwright_dp_y * d_right) + s1 * dd_right_dp(1);

            Eigen::RowVector2d dfdp_right;
            dd_right_dp = 2.0 * (p - B_local).transpose();

            // addding the three terms
            Eigen::RowVector2d Jp = dfdp_center + dfdp_left + dfdp_right;
            *H3 = Jp;
        }
	    return dist;
    }

    /**
     * \brief Returns the squared distance (y-w/2) of the fridge edge to the side/wall.
     *
     * \param p Point in fridge coordinates.
     * \param H Optional Jacobian for the distance with respect to the point.
     * \return Distance to the side of the fridge.
     */
    static double dist_side2wall_sqr(const gtsam::Vector5 &v, const gtsam::Vector2 &p, gtsam::OptionalJacobian<1, 5> H1,
                                                                                       gtsam::OptionalJacobian<1, 2> H2)
    {
        const double semi_width = v(3)/2.0;
        if (H1) *H1 = gtsam::Matrix15::Zero();
        if (H2) *H2 = (gtsam::Matrix12() << 0, 2*(p.y()+semi_width)).finished();
        //std::cout << "Dist to side sqr: " << p.y()*p.y() << std::endl;
        return (p.y()+semi_width)* (p.y()+semi_width);
    }

    /**
   * \brief Returns the angle of the fridge to the side/wall.
   *
   * \param p Point in fridge coordinates.
   * \param H Optional Jacobian for the distance with respect to the point.
   * \return Distance to the side of the fridge.
   */
    static double angle2wall_sqr(const gtsam::Vector5 &v, const double wall, gtsam::OptionalJacobian<1, 5> H1,
                                                                             gtsam::OptionalJacobian<1, 1> H2)
    {
        double wall_angle = 0.0;
        // Ensure the wall_index is valid
        //if (wall < 0 || wall >= 4) {throw std::invalid_argument("Invalid wall index. Must be between 0 and 3."); }
        switch (static_cast<int>(wall))
        {
            case 1:
                wall_angle = 0.0;
            break;
            case 2:
                wall_angle = -M_PI/2.0;
            break;
            case 3:
                wall_angle = M_PI;
            break;
            case 4:
                wall_angle = M_PI/2.0;
            break;
        }
        // Calculate the difference between the fridge angle and the wall angle
        const double fridge_angle = v(2);
        const double angle_diff = fridge_angle - wall_angle;

        // assign 1 to the third element of H1
        // compute the Jacobian of angle_diff wrt v(2): d atan2(sin(x), cos(x)) / dx = 1/(1 + x^2)

        if (H1) *H1 = (gtsam::Matrix15() << 0, 0, 2.0*angle_diff, 0, 0).finished();
        if (H2) *H2 = gtsam::Matrix11::Zero();
//        std::cout << "Ang to side : " << angle_diff*angle_diff << std::endl;
        return angle_diff*angle_diff;
    }

    /**
     * \brief Transforms a point from room coordinates to fridge coordinates.
     *
     * \param v Vector containing fridge parameters.
     * \param p Point in room coordinates.
     * \param H1 Optional Jacobian for the transformation with respect to the fridge parameters.
     * \param H2 Optional Jacobian for the transformation with respect to the point.
     * \return Transformed point in fridge coordinates.
     */
    static gtsam::Point2 room2fridge(const gtsam::Vector5 &v, const gtsam::Vector2 &p, gtsam::OptionalJacobian<2, 5> H1, gtsam::OptionalJacobian<2, 2> H2 )
    {
        const gtsam::Pose2 pose(v(0), v(1), v(2));
        gtsam::Matrix23 H1_pose;
        gtsam::Matrix22 H2_point;
        auto q = pose.transformTo(p, H1_pose, H2_point);

        //qDebug() << "Point" << p.transpose().x() << p.transpose().y() << "from room transformed to fridge" << q.transpose().x() << q.transpose().y();

        if (H1) // Compute the Jacobian of the transformation wrt the x,y,theta params
        {
            *H1 = gtsam::Matrix25::Zero();
            H1->block<2, 3>(0, 0) = H1_pose;
        }
        // Compute the Jacobian of the transformation wrt the point
        if (H2) *H2 = H2_point;
        return q;
    }

    /**
     * \brief Transforms a point from fridge coordinates to a specific side of the fridge.
     *
     * \param v Vector containing fridge parameters.
     * \param side The side of the fridge to which the point is transformed.
     * \param p Point in fridge coordinates.
     * \param H1 Optional Jacobian for the transformation with respect to the fridge parameters.
     * \param H2 Optional Jacobian for the transformation with respect to the side.
     * \param H3 Optional Jacobian for the transformation with respect to the point.
     * \return Transformed point in the coordinates of the specified side of the fridge.
     */
    static gtsam::Point2 fridge2side(const gtsam::Vector5 &v, const double side, const gtsam::Point2 &p, gtsam::OptionalJacobian<2, 5> H1,
                                                                                                         gtsam::OptionalJacobian<2, 1> H2,
                                                                                                         gtsam::OptionalJacobian<2, 2> H3)
    {
        const double semi_depth = v(4)/2.0;
        const double semi_width = v(3)/2.0;
        gtsam::Pose2 wall_pose;
        switch (static_cast<int>(side))
        {
            case 1:
                wall_pose = gtsam::Pose2(0.0, semi_depth, 0);
                break;
            case 2:
                wall_pose = gtsam::Pose2(semi_width, 0.0, -M_PI/2.0);
                break;
            case 3:
                wall_pose = gtsam::Pose2(0.0, -semi_depth, M_PI);
                break;
            case 4:
                wall_pose = gtsam::Pose2(-semi_width, 0.0, M_PI/2.0);
                break;
        }

        gtsam::Matrix23 H1_pose;
        gtsam::Matrix22 H3_point;
        const auto pw = wall_pose.transformTo(p, H1_pose, H3_point);

        qDebug() << "Fridge point" << p.transpose().x() << p.transpose().y() << "transformed to side" << side << pw.transpose().x() << pw.transpose().y();

        if (H1)    // Compute the Jacobian of the transformation wrt the fridge params
        {
            *H1 = gtsam::Matrix25::Zero();
            // copy the 1,2,3 columns of H1_pose to the 3,4,2 columns of H1
            H1->block<2, 2>(0, 3) = H1_pose.block<2, 2>(0, 0);
            //H1->block<2, 1>(0, 2) = H1_pose.block<2, 1>(0, 2);
        }
        if (H2) *H2 = gtsam::Matrix12::Zero();  // Compute the Jacobian of the transformation wrt the const side
        if (H3) *H3 = H3_point;  // Compute the Jacobian of the transformation wrt the point
        return pw;
    };

    // Smooth absolute function: differentiable approximation of abs(x) = sqrt(x) + eps where eps is a small constant.
    // static double smooth_abs(const double x, gtsam::OptionalJacobian<1, 1> H )
    // {
    //     constexpr double eps = 1e-4;
    //     if (H) *H = (gtsam::Matrix11() << x / std::sqrt(x * x + eps)).finished(); // dsmoothAbs/dx = x / sqrt(x^2 + eps)
    //     return std::sqrt(x * x + eps);
    // }
    //
    // // Smooth absolute function for 2 vectors: differentiable approximation of abs(x) = sqrt(x) + eps where eps is a small constant.
    // static gtsam::Point2 smooth_abs_2(const gtsam::Point2 &p, gtsam::OptionalJacobian<2, 2> H )
    // {
    //     constexpr double eps = 1e-4;
    //     // compute smooth_abs for each component of a two-dimensional vector
    //     const double x = p.x();
    //     const double y = p.y();
    //     const double norm = std::sqrt(x * x + y * y + eps);
    //     if (H) *H = (gtsam::Matrix22() << x / norm, y / norm, x / norm, y / norm).finished();
    //     return gtsam::Point2(std::sqrt(x * x + eps), std::sqrt(y * y + eps));
    // }

    /**
     * \brief Softmin function: given 4 values compute the minimum value in a smooth way.
     *  Given values compute the minimum value in a smooth way
     *  softmin = (sum_i d_i * exp(-k*d_i)) / (sum_i exp(-k*d_i))
     * \param v0-v3 values.
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

    /**
     * \brief Transforms fridge coordinates to room's wall coordinates.
     *
     * \param v Vector containing fridge parameters.
     * \param params Vector containing wall parameters: wall number, semi_depth, semi_width.
     * \param H1 Optional Jacobian for the transformation with respect to the fridge parameters.
     * \param H2 Optional Jacobian for the transformation with respect to the wall parameters.
     * \return Transformed point in wall coordinates.
     */
    static gtsam::Point2 fridge2wall(const gtsam::Vector5 &v, const gtsam::Vector3 &params, gtsam::OptionalJacobian<2, 5> H1,
                                                                                            gtsam::OptionalJacobian<2, 3> H2)
    {
        const double semi_depth = params(1);
        const double semi_width = params(2);
        gtsam::Pose2 wall_pose;
        switch (static_cast<int>(params(0)))
        {
            case 1:
                wall_pose = gtsam::Pose2(0.0, semi_depth, 0);
            break;
            case 2:
                wall_pose = gtsam::Pose2(semi_width, 0.0, -M_PI/2.0);
            break;
            case 3:
                wall_pose = gtsam::Pose2(0.0, -semi_depth, M_PI);
            break;
            case 4:
                wall_pose = gtsam::Pose2(-semi_width, 0.0, M_PI/2.0);
            break;
        }

        gtsam::Matrix23 H1_pose;
        gtsam::Matrix22 H1_point;
        const auto pw = wall_pose.transformTo(gtsam::Point2(v(0), v(1)), H1_pose, H1_point);
        //std::cout << " Wall: " << params(0) << "pose: " << wall_pose << " center in room: (" << v(0) << " " << v(1) << ") " << "center in wall: " << pw.transpose() << std::endl;
        if (H1)    // Compute the Jacobian of the transformation wrt the fridge params
        {
            *H1 = gtsam::Matrix25::Zero();
            H1->block<2, 2>(0, 0) = H1_point;  // H1_point is the Jacobian of the transformation wrt the point (v(0), v(1)) which is the center of the fridge
        }
        if (H2) *H2 = gtsam::Matrix23::Zero();   // Compute the Jacobian of the transformation wrt the params (consts)

        return pw;
    };

    /**
     * \brief computes the min distance of the 4 fridge middle points to a room side given in params.
     *
     * \param v Vector containing fridge parameters.
     * \param params Vector containing wall parameters: wall number, semi_depth, semi_width.
     * \param H1 Optional Jacobian for the transformation with respect to the fridge parameters.
     * \param H2 Optional Jacobian for the transformation with respect to the wall parameters.
     * \return Transformed point in wall coordinates.
     */
    static double closest_fridge_side_to_wall(const gtsam::Vector5 &v,
                                      const gtsam::Vector3 &params, gtsam::OptionalJacobian<1, 5> H1,
                                      gtsam::OptionalJacobian<1, 3> H2)
    {
        const double semi_depth = params(1);    // semi depth of the room
        const double semi_width = params(2);    // semi width of the room
        gtsam::Pose2 wall_pose;
        // for each wall side compute the transformation from room frame to wall frame. Y+ points outwards and X+ points to the right
        switch (static_cast<int>(params(0)))    // wall number
        {
            case 1:
                wall_pose = gtsam::Pose2(0.0, semi_depth, 0);
            break;
            case 2:
                wall_pose = gtsam::Pose2(semi_width, 0.0, -M_PI/2.0);
            break;
            case 3:
                wall_pose = gtsam::Pose2(0.0, -semi_depth, M_PI);
            break;
            case 4:
                wall_pose = gtsam::Pose2(-semi_width, 0.0, M_PI/2.0);
            break;
        }

        // pose to transform a point from fridge to room
        gtsam::Matrix23 H1f_pose, H2f_pose, H3f_pose, H4f_pose;
        gtsam::Matrix22 H1f_point, H2f_point, H3f_point, H4f_point;
        const gtsam::Pose2 room_to_fridge(v(0), v(1), v(2));    // both J are significant
        const gtsam::Vector2 side_1 = room_to_fridge.transformFrom(gtsam::Point2(0.0, v(4)/2.0), H1f_pose, H1f_point); // top side
        const gtsam::Vector2 side_2 = room_to_fridge.transformFrom(gtsam::Point2(v(3)/2.0, 0.0), H2f_pose, H2f_point); // right side
        const gtsam::Vector2 side_3 = room_to_fridge.transformFrom(gtsam::Point2(0.0, -v(4)/2.0), H3f_pose, H3f_point); // bottom side
        const gtsam::Vector2 side_4 = room_to_fridge.transformFrom(gtsam::Point2(-v(3)/2.0, 0.0), H4f_pose, H4f_point); // left side

        // transform all middle points of the fridge sides to the wall frame. HX_pose is constant (room params)
        gtsam::Matrix23 H1w_pose, H2w_pose, H3w_pose, H4w_pose;
        gtsam::Matrix22 H1w_point, H2w_point, H3w_point, H4w_point;
        const gtsam::Vector2 p1 = wall_pose.transformTo(side_1, H1w_pose, H1w_point);
        const gtsam::Vector2 p2 = wall_pose.transformTo(side_2, H2w_pose, H2w_point);
        const gtsam::Vector2 p3 = wall_pose.transformTo(side_3, H3w_pose, H3w_point);
        const gtsam::Vector2 p4 = wall_pose.transformTo(side_4, H4w_pose, H4w_point);

        // compute the squared distance between the middle point of the fridge side and the wall.
        // It is the squared y coordinate of the point. We add a penalty term if the point is outside the wall.
        const double penalty_gain = 3.0;
        const double dist1 = p1.y() * p1.y() + penalty_gain*softplus(p1.y());
        const double dist2 = p2.y() * p2.y() + penalty_gain*softplus(p2.y());
        const double dist3 = p3.y() * p3.y() + penalty_gain*softplus(p3.y());
        const double dist4 = p4.y() * p4.y() + penalty_gain*softplus(p4.y());

        // compute the softmin of the 4 distances
        const std::vector<double> dists = { dist1, dist2, dist3, dist4 };
        const double final_dist = softmin_local(dists);

        if (H1)    // Compute the Jacobian of the transformation wrt the fridge params: d(f(v,p))/dv
        {
            *H1 = gtsam::Matrix15::Zero();

            // Chain rule: df/dv = dsoft/di * di/p * dp/dv

            const double k = 10.0; // softness parameter for softmin
            std::vector<double> dsoft = dsoftmin_local_dd(dists, k); // dsoft[i] = ∂f/∂d_i

            // For each side, the derivative of d_i with respect to p_i is:
            // dd_i/dp_i = [ 0, 2 * p_i.y  + penalty_gain * softplusDerivative(p_i.y) ]
            gtsam::Matrix12 grad1, grad2, grad3, grad4;
            grad1 << 0.0, 2.0 * p1.y() + penalty_gain * softplusDerivative(p1.y());
            grad2 << 0.0, 2.0 * p2.y() + penalty_gain * softplusDerivative(p1.y());   ;
            grad3 << 0.0, 2.0 * p3.y() + penalty_gain * softplusDerivative(p1.y());   ;
            grad4 << 0.0, 2.0 * p4.y() + penalty_gain * softplusDerivative(p1.y());   ;

            // contribution of point i to dt/dv
             gtsam::Matrix25 dp1dv;
             dp1dv << (H1w_point * H1f_pose).col(0), (H1w_point * H1f_pose).col(1), (H1w_point * H1f_pose).col(2), (H1w_point * H1f_point).col(0), (H1w_point * H1f_point).col(1);
            // contribution of point i to dt/dv
            gtsam::Matrix25 dp2dv;
            dp2dv << (H2w_point * H2f_pose).col(0), (H2w_point * H2f_pose).col(1), (H2w_point * H2f_pose).col(2), (H2w_point * H2f_point).col(0), (H2w_point * H2f_point).col(1);
            // contribution of point i to dt/dv
            gtsam::Matrix25 dp3dv;
            dp3dv << (H3w_point * H3f_pose).col(0), (H3w_point * H3f_pose).col(1), (H3w_point * H3f_pose).col(2), (H3w_point * H3f_point).col(0), (H3w_point * H3f_point).col(1);
            // contribution of point i to dt/dv
            gtsam::Matrix25 dp4dv;
            dp4dv << (H4w_point * H4f_pose).col(0), (H4w_point * H4f_pose).col(1), (H4w_point * H4f_pose).col(2), (H4w_point * H4f_point).col(0), (H4w_point * H4f_point).col(1);

            // multiply by the gradient of the distance to the wall 1x2 * 2x5 = 1x5
            gtsam::Matrix15 J1 = grad1 * dp1dv;
            gtsam::Matrix15 J2 = grad2 * dp2dv;
            gtsam::Matrix15 J3 = grad3 * dp3dv;
            gtsam::Matrix15 J4 = grad4 * dp4dv;

            // Combine the contributions weighted by the softmin derivatives.
            gtsam::Matrix15 Jv = dsoft[0] * J1 + dsoft[1] * J2 + dsoft[2] * J3 + dsoft[3] * J4;
            *H1 = Jv;
        }
        if (H2) *H2 = gtsam::Matrix13::Zero();   // Compute the Jacobian of the transformation wrt the params (consts)

        return final_dist;
    };
};

#endif // FRIDGEPOINTSFACTOR_H
