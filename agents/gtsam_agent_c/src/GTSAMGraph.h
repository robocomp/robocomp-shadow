//
// Created by robolab on 11/11/24.
//

#ifndef GTSAM_AGENT_C_GTSAM_GRAPH_H
#define GTSAM_AGENT_C_GTSAM_GRAPH_H

#define RESET   "\033[0m"
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Key.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BearingRangeFactor.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/nonlinear/Marginals.h>
#include <Eigen/Dense>
#include <QGraphicsItem>
#include <QGraphicsScene>

using namespace std;
using namespace gtsam;
using namespace std::chrono;

using symbol_shorthand::X; // Pose
using symbol_shorthand::L; // Landmark

class GTSAMGraph
{
private:
    std::map<int, std::pair<gtsam::Key, gtsam::Point3>> landmark_keys; // Maps landmark IDs to GTSAM keys

    noiseModel::Diagonal::shared_ptr odometry_noise = noiseModel::Diagonal::Sigmas(Vector6(0.01, 0.01, 0.01, 0.005, 0.005, 0.005)); // x, y, z, roll, pitch, yaw noise
    noiseModel::Diagonal::shared_ptr landmark_noise = noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.5)); // x, y, z, roll, pitch, yaw noise
    noiseModel::Diagonal::shared_ptr prior_noise = noiseModel::Diagonal::Sigmas(Vector6(0.01, 0.01, 0.01, 0.005, 0.005, 0.005));
    noiseModel::Diagonal::shared_ptr prior_landmark_noise = noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.01));

    ISAM2Params parameters;
    // Define the smoother lag (in seconds)
    double lag = 2.0;
    double start_time;
    double last_update_timestamp;

    int step = 1;
    Key previousKey, currentKey;

    // Create containers to store the factors and linearization points that
    // will be sent to the smoothers
    NonlinearFactorGraph newFactors;
    Values newValues, result;
    FixedLagSmoother::KeyTimestampMap newTimestamps;

    // Declare smootherISAM2 as a unique_ptr, but do not instantiate yet
    std::unique_ptr<IncrementalFixedLagSmoother> smootherISAM2;

    Pose3 current_pose = Pose3(Rot3::RzRyRx(0, 0, 0), Point3(0, 0, 0));

public:
    GTSAMGraph()
    {
        parameters.relinearizeThreshold = 0.0; // Set the relin threshold to zero such that the batch estimate is recovered
        parameters.relinearizeSkip = 1; // Relinearize every time
        smootherISAM2 = std::make_unique<IncrementalFixedLagSmoother>(lag, parameters);
    }

    void insert_prior_pose(double timestamp, const gtsam::Pose3 &pose);
    void insert_landmark_prior( double timestamp, int landmark_id, const gtsam::Point3& position, bool is_new=false);
    void insert_odometry_pose(double timestamp, const gtsam::Pose3 &pose);
    void add_landmark_measurement(std::vector<std::tuple<int, double, Eigen::Vector3d, bool>> measured_corners);

    gtsam::Pose3 get_robot_pose();
    std::map<double, gtsam::Key> get_graph_timestamps();
    double get_last_update_timestamp();
    double get_start_time();

    void print_gtsam_graph();

    void clear_containers();

    void set_start_time(double start_time);

    void draw_graph_nodes(QGraphicsScene *pScene);
    void draw_landmark_measurements(QGraphicsScene *pScene, Pose3 pose, Point2 landmark, bool clear);

    //////////////////////////////// DEBUG ////////////////////////////////
    void print_current_key();
    };


#endif //GTSAM_AGENT_C_GTSAM_GRAPH_H
