#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/AdaptAutoDiff.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>

#include <vector>
#include <iostream>

using namespace gtsam;
using namespace std;

// Custom factor for the squared distance between fridge center and centroid of residuals
class FridgeCentroidFactor final : public NoiseModelFactor1<Point2>
{
private:
    Point2 centroid_;  // Centroid of the residual points (measurement)

public:
    FridgeCentroidFactor(Key key, const Point2& centroid, const SharedNoiseModel& model)
        : NoiseModelFactor1<Point2>(model, key), centroid_(centroid) {}

    Vector evaluateError(const Point2& fridge_center, gtsam::Matrix *H) const override
    {
        // Calculate the error (difference between fridge center and centroid)
        Vector2 error = fridge_center - centroid_;

        // Calculate the squared distance (error magnitude squared)
        double squared_distance = error.squaredNorm();

        // Optional Jacobian calculation
        if (H)
        {
             // *H = numericalDerivative11<Vector, Point2>([&](const Point2& current_fridge_center)
             //     { return this->evaluateError(current_fridge_center, nullptr);}, fridge_center);
            
            //Jacobian of the error (difference) with respect to the fridge_center. d(error)/d(fridge_center) = error
            *H = Matrix::Identity(2, 2);  // Identity
            *H = 2 * error.transpose() * (*H);
            std::cout << "Center: " << fridge_center.transpose() << "   Jacobian: " << *H << "    Error: " << squared_distance << std::endl;
        }

        // Return the squared distance as a 1D vector
        return (Vector1() << squared_distance).finished();
    }
};


int main() {
    // 1. Define keys for the fridge center (cx, cy)
    const gtsam::Symbol fridgeSym('f', 1);
    Key fridgeKey = fridgeSym.key();

    // 2. Calculate the centroid of the residual points (example data)
    vector<Point2> residualPoints = {
        Point2(1.0, 0.2), Point2(1.2, 0.3), Point2(0.8, 0.1),
        Point2(1.1, 0.4), Point2(0.9, 0.2)
    };

    Point2 centroid(0, 0);
    for (const auto& point : residualPoints) {
        centroid = centroid + point;
    }
    centroid = centroid / residualPoints.size();
    cout << "Centroid of residual points: " << centroid.transpose() << endl;


    // 3. Define a noise model (e.g., isotropic noise)
    auto noiseModel = noiseModel::Isotropic::Sigma(1, 0.1);  // Adjust sigma as needed

    // 4. Create the custom factor
    const auto centroidFactor = make_shared<FridgeCentroidFactor>(fridgeKey, centroid, noiseModel);

      // Prior for the fridge
    auto priorModel = noiseModel::Isotropic::Sigma(2, 0.5);
    const auto priorFactor = make_shared<PriorFactor<Point2>>(fridgeKey, Point2(1.0, 0.0), priorModel); // Example

    // 5. Create a Factor Graph
    NonlinearFactorGraph graph;
    graph.add(centroidFactor);
    graph.add(priorFactor);

    // 6. Provide initial estimates for the fridge center
    Values initialEstimate;
    initialEstimate.insert(fridgeKey, Point2(-100.0, 100.0));  // Initial guess (can be arbitrary)
    std::cout << "Initial guess" << initialEstimate.at<Point2>(fridgeKey).transpose() << std::endl;

    // 7. Optimize the graph using Levenberg-Marquardt
    // Create Levenberg-Marquardt parameters
    LevenbergMarquardtParams params;
    params.setAbsoluteErrorTol(1e-5); // Set the absolute error tolerance
    params.setRelativeErrorTol(1e-5); // Set the relative error tolerance
    LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
    Values result = optimizer.optimize();

    // 8. Print the optimized fridge center
    cout << "Optimized fridge center: " << result.at<Point2>(fridgeKey).transpose() << endl;

    return 0;
}
