#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/nonlinear/expressions.h> // For expressions
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>

#include <vector>
#include <iostream>

using namespace gtsam;
using namespace std;

gtsam::Vector5 project_pose(const gtsam::Vector5 &v, OptionalJacobian<5, 5> H)
{
  if(H) *H = (gtsam::Matrix55() << 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1).finished();
  std::cout << "Center: " << v.transpose() << std::endl;
  return gtsam::Vector5(v.x(), v.y(), 0.0, 0.0, 0.0);
}

gtsam::Vector5 project_scale(const gtsam::Vector5 &v, OptionalJacobian<5, 5> H)
{
    // Devuelve las dimensiones del frigorífico para que el factor las reste de las dimensiones estimadas de los puntos
    if(H) *H = (gtsam::Matrix55() << 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1).finished();
    std::cout << "Dims: " << v.transpose() << std::endl;
    return gtsam::Vector5(0.0, 0.0, 0.0, v(3), v(4));
}

gtsam::Vector5 project_orientation(const gtsam::Vector5 &v, OptionalJacobian<5, 5> H)
{
    // transformar la orientación del frigorífico al marco de la pared más cercana
    if(H) *H = (gtsam::Matrix55() << 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1).finished();
    // Maybe return the Frobenious norm of the fridge rotation matrix in the wall frame
    // Or just the fridge rotation angle in the wall frame

    return gtsam::Vector5(0.0, 0.0, v(2), 0.0, 0.0);
}

int main()
{
    // 1. Calculate the centroid of the residual points (example data)
    vector<Point2> residualPoints =
    {
        Point2(1.0, 0.2), Point2(1.2, 0.3), Point2(0.8, 0.1),
        Point2(1.1, 0.4), Point2(0.9, 0.2)
    };
    Point2 centroid(0, 0);
    for (const auto& point : residualPoints)
        centroid = centroid + point;
    centroid = centroid / residualPoints.size();
    cout << "Centroid of residual points: " << centroid.transpose() << endl;

    // 2. Compute the closest wall to the centroid


    // 2. Define keys for the fridge params (cx, cy, alpha)
    const gtsam::Symbol fridgeSym('f', 1);
    Key fridgeKey = fridgeSym.key();

    // 3. Define a noise model (e.g., isotropic noise)
    auto noiseModel = noiseModel::Isotropic::Sigma(5, 0.1);  // Adjust sigma as needed

    // 4. Create an expression for the fridge center
    Expression<Vector5> fridge_center_expr(fridgeKey);

    // Create a constant expression for the centroid
    Vector5 par(centroid.x(), centroid.y(), 0.0, 0.0, 0.0);
    Expression<Vector5> centroid_expr(par);

    // Create an expression for the error (difference)
    Expression<Vector5> project_pose_(&project_pose, fridge_center_expr);

    // Prior for the fridge
    auto priorModel = noiseModel::Isotropic::Sigma(2, 0.5);
    const auto priorFactor = make_shared<PriorFactor<Point2>>(fridgeKey, Point2(1.0, 0.0), priorModel); // Example

    // 5. Create a Factor Graph
    ExpressionFactorGraph graph;
    graph.addExpressionFactor(project_pose_, par, noiseModel);

    // 6. Provide initial estimates for the fridge center
    Values initialEstimate;
    initialEstimate.insert(fridgeKey, Vector5(-1000.0, 1400.0, 0, 0, 0));  // Initial guess (can be arbitrary)
    std::cout << "Initial guess " << initialEstimate.at<Vector5>(fridgeKey).transpose() << std::endl;

    // 7. Optimize the graph using Levenberg-Marquardt
    //LevenbergMarquardtParams params;
    //params.setAbsoluteErrorTol(1e-5); // Set the absolute error tolerance
    //params.setRelativeErrorTol(1e-5); // Set the relative error tolerance
    //LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
    GaussNewtonOptimizer optimizer(graph, initialEstimate);
    Values result = optimizer.optimize();

    // 8. Print the optimized fridge center
    cout << "Optimized fridge center: " << result.at<Vector5>(fridgeKey).transpose() << endl;

    // 9. Calculate and print marginal covariances for all variables
    cout.precision(3);
    Marginals marginals(graph, result);
    cout << "x1 covariance:\n" << marginals.marginalCovariance(fridgeKey) << endl;

    return 0;
}
