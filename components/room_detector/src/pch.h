//
// Created by pbustos on 7/12/24.
//

// pch.h
#ifndef PCH_H
#define PCH_H

// Add headers that you want to precompile here
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <boost/circular_buffer.hpp>
#include <Eigen/Geometry>
#include <numeric>
#include <tuple>
#include <vector>
#include <set>
#include <boost/math/constants/constants.hpp>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/BearingRangeFactor.h>
//#include <gtsam/sam/ExpressionFactor.h>
#include <gtsam/3rdparty/Eigen/Eigen/src/Core/Diagonal.h>
#include <gtsam/linear/NoiseModel.h>

#include <cppitertools/range.hpp>
#include <cppitertools/sliding_window.hpp>
#include <cppitertools/enumerate.hpp>
#include <cppitertools/sliding_window.hpp>
#include <cppitertools/combinations.hpp>
#include <cppitertools/zip.hpp>
#include <cppitertools/filter.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include "room_detector.h"
#include "dbscan.h"

#endif // PCH_H
