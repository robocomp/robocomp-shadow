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

#include <genericworker.h>
#include <fstream>
#include <string>
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/stuff/sampler.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_with_hessian.h"
#include "g2o/types/slam2d/types_slam2d.h"
#include "g2o/core/factory.h"
#include <g2o/core/marginal_covariance_cholesky.h>
#include <memory>
#include <fps/fps.h>

using namespace g2o;

  G2O_USE_TYPE_GROUP(slam2d);
  //G2O_REGISTER_TYPE(EDGE_SE2_POINT_XY, EdgeSE2PointXY );
  //G2O_REGISTER_TYPE(VERTEX_XY, VertexPointXY );

class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
        SpecificWorker(TuplePrx tprx, bool startup_check);
        ~SpecificWorker();
        bool setParams(RoboCompCommonBehavior::ParameterList params);

        std::string G2Ooptimizer_optimize(std::string trajectory);

    public slots:
        void compute();
        int startup_check();
        void initialize(int period);

    private:
        bool startup_check_flag;

        struct PARAMS
        {
            int iterations; // Number of iterations using by the optimizer
        };
        PARAMS parameters;

        // fps
        FPSCounter fps;

        // Create the optimizer
        g2o::SparseOptimizer optimizer;
        std::unique_ptr<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>> linear_solver;
        std::unique_ptr<g2o::BlockSolverX> block_solver;
        std::unique_ptr<g2o::OptimizationAlgorithmLevenberg> algorithm;
        // Create an instance of the robust kernel
        g2o::RobustKernelHuber* robustKernel = new g2o::RobustKernelHuber;
};

#endif
