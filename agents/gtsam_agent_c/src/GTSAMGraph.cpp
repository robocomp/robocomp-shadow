//
// Created by robolab on 11/11/24.
//

#include "GTSAMGraph.h"

void GTSAMGraph::insert_prior_pose(double timestamp, const gtsam::Pose3 &pose)
{
//    last_update_timestamp = getStartTime();
    start_time = timestamp;
    std::cout << "Start time: " << start_time << std::endl;
    last_update_timestamp = timestamp;
    Key priorKey = X(0);
    newFactors.addPrior(priorKey, pose, prior_noise);
    newValues.insert(priorKey, pose); // Initialize the first pose at the mean of the prior
    newTimestamps[priorKey] = 0.0; // Set the timestamp associated with this key to 0.0 seconds;
    smootherISAM2->update(newFactors, newValues, newTimestamps);
//    smootherISAM2->getFactors().print("\nFactor Graph Contents:\n");
//    smootherISAM2->getISAM2().print("\nISAM2 contents:\n");
//    result = smootherISAM2->calculateEstimate();
    std::cout << BLUE << "Prior pose inserted" << RESET << std::endl;
    clear_containers();
    previousKey = priorKey;

}

void GTSAMGraph::insert_landmark_prior(double timestamp, int landmark_id, const gtsam::Point3& position, bool is_new)
{
    std::cout << "Inserting landmark with timestamp: " << timestamp << std::endl;
    Key landmark_key = L(landmark_id);
    for (const auto& pair : newTimestamps) {
        std::cout << "Key: " << pair.first << ", Timestamp: " << pair.second << std::endl;
    }
//    std::cout << "Landmark key: " << "ID " << landmark_id << " " << landmark_key << std::endl;
    landmark_keys[landmark_id] = std::make_pair(landmark_key, position);
    newFactors.addPrior(landmark_key, position, prior_landmark_noise);
    newValues.insert(landmark_key, position);
    if(not is_new)
    {
        qInfo() << "Landmark " << landmark_id << " already exists. Not inserting prior";
        newTimestamps[landmark_key] = 0.0;
    }

    else
    {
        qInfo() << "Landmark " << landmark_id << " is new. Inserting prior";
        newTimestamps[landmark_key] = timestamp;
    }



    smootherISAM2->update(newFactors, newValues, newTimestamps);
//    smootherISAM2->getFactors().print("\nFactor Graph Contents:\n");
//    smootherISAM2->getISAM2().print("\nISAM2 contents:\n");
//    result = smootherISAM2->calculateEstimate();
    clear_containers();

    std::cout << "Landmark " << landmark_id << " inserted at "
              << position.transpose() << std::endl;
}

void GTSAMGraph::insert_odometry_pose(double timestamp, const gtsam::Pose3 &pose)
{
    double time_from_start = timestamp - get_start_time();
    currentKey = X(step);
    Pose3 prevPose = smootherISAM2->calculateEstimate<Pose3>(previousKey);
    Pose3 predicted_pose = prevPose.compose(pose);  // Estimate new pose
    newValues.insert(currentKey, predicted_pose);

//    std::cout << "Current key: " << currentKey << std::endl;
//    std::cout << "Previous key: " << previousKey << std::endl;
    BetweenFactor<Pose3> odometry_factor(previousKey, currentKey, pose, odometry_noise);

    newFactors.add(odometry_factor);
    newTimestamps[currentKey] = time_from_start;

    // Run update
    smootherISAM2->update(newFactors, newValues, newTimestamps);
    clear_containers();
    previousKey = currentKey;
    step++;

    last_update_timestamp = timestamp;
}

void GTSAMGraph::add_landmark_measurement(std::vector<std::tuple<int, double, Eigen::Vector3d, bool>> measured_corners)
{
    gtsam::Values current_estimates = smootherISAM2->calculateEstimate();
    auto gtsam_timestamps = smootherISAM2->timestamps();
    auto act_start_time = get_start_time();

    for (const auto &corner: measured_corners)
    {
        int landmark_id = std::get<0>(corner);
        double timestamp = std::get<1>(corner);
        Eigen::Vector3d corner_position = std::get<2>(corner);
        bool corner_valid = std::get<3>(corner);



        if (corner_valid)
        {
//            std::cout << "Adding landmark measurement for corner ID: " << landmark_id
//                      << " at timestamp: " << timestamp << std::endl;
            if (!landmark_keys.count(landmark_id))
            {
                std::cerr << "Landmark " << landmark_id << " not initialized" << std::endl;
                return;
            }

            // Check if the landmark key exists in the estimate
            if (not current_estimates.exists(landmark_keys[landmark_id].first))
            {
                double time_from_start = timestamp - act_start_time;
//                std::cerr << "Landmark " << landmark_id << " not found in estimates. Inserting prior landmark" << std::endl;
                std::cout << "Landmark timestamp: " << timestamp << " Start time " << act_start_time << " Time from start " << time_from_start << std::endl;
                if(time_from_start < 0)
                {
                    std::cerr << "Landmark timestamp is negative. Not inserting prior landmark" << std::endl;
                    continue;
                }
                insert_landmark_prior(time_from_start, landmark_id, landmark_keys[landmark_id].second, true);
            }

            gtsam::Unit3 bearing(corner_position);

            // calculate norm and angle of the measurement
            double measured_range = corner_position.norm();

            // Parameters for time matching
            const double maxTimeDiff = 0.032; // Maximum allowed time difference (seconds)

            Key observedPoseKey;
            double smallestDiff = std::numeric_limits<double>::max();
            bool found = false;

            for (const auto &kv: gtsam_timestamps) {
//                std::cout << "Key: " << kv.first << ", Timestamp: " << start_time + kv.second << std::endl;
                // Check if this is a pose key (starts with 'x')
                if (Symbol(kv.first).chr() == 'x') {
                    double timeDiff = std::abs(act_start_time + kv.second - timestamp);
//                    std::cout << "Time difference: " << timeDiff << std::endl;
                    // Check if within acceptable range and closer than previous candidates
                    if (timeDiff < maxTimeDiff && timeDiff < smallestDiff) {
                        smallestDiff = timeDiff;
//                        std::cout << "Smallest diff: " << smallestDiff << std::endl;
                        observedPoseKey = kv.first;
                        found = true;
                    }
                }
            }

            if (found)
            {
//                std::cout << "Found pose " << observedPoseKey
//                          << " with time difference " << smallestDiff << " seconds" << std::endl;

                BearingRangeFactor<Pose3, Point3> range_factor(observedPoseKey,
                                                               landmark_keys[landmark_id].first, bearing, measured_range,
                                                               landmark_noise);
//                std::cout << "Adding range factor for landmark " << landmark_id
//                          << " with timestamp: " << timestamp << std::endl;
                newFactors.add(range_factor);
                newTimestamps[landmark_keys[landmark_id].first] = timestamp - act_start_time;
            }
        }
        smootherISAM2->update(newFactors, newValues, newTimestamps);
        clear_containers();
    }
}

gtsam::Pose3 GTSAMGraph::get_robot_pose()
{
//    std::cout << "Required key: " << previousKey << std::endl;
    return smootherISAM2->calculateEstimate<Pose3>(previousKey);
}

std::map<double, gtsam::Key> GTSAMGraph::get_graph_timestamps()
{
    //  Create map to be filled
    std::map<double, gtsam::Key> timestamps;
    // Get the smoother's keys and timestamps
    auto timestamps_graph = smootherISAM2->timestamps();

    auto act_start_time = get_start_time();

    for (const auto& kv : timestamps_graph)
    {
        // Convert timestamp to double and append to map
        timestamps[Symbol(kv.first).chr()] = act_start_time + (kv.second / 1.0);
//        std::cout << "Key: " << kv.first << ", Timestamp: " << start_time + (kv.second / 1.0) << std::endl;
    }
    return timestamps;
}

double GTSAMGraph::get_last_update_timestamp()
{
    return last_update_timestamp;
}

void GTSAMGraph::print_gtsam_graph()
{
    std::cout << "GTSAM Graph:" << std::endl;
    std::cout << "Landmark keys: ";
    for (const auto& pair : landmark_keys) {
        std::cout << pair.first << " ";
    }
    std::cout << std::endl;

    smootherISAM2->getFactors().print("\nFactor Graph Contents:\n");
    smootherISAM2->getISAM2().print("\nISAM2 contents:\n");
}

void GTSAMGraph::clear_containers()
{
    // Clear contains for the next iteration
    newTimestamps.clear();
    newValues.clear();
    newFactors.resize(0);
}

double GTSAMGraph::get_start_time()
{
    return start_time;
}


void GTSAMGraph::print_current_key()
{
    std::cout << "Current key: " << step << std::endl;
    step += 1;
}

void GTSAMGraph::set_start_time(double start_time)
{
    this->start_time = start_time;
}

void GTSAMGraph::reset_graph()
{
    smootherISAM2 = std::make_unique<gtsam::IncrementalFixedLagSmoother>(lag, parameters);
}

void GTSAMGraph::draw_graph_nodes(QGraphicsScene *pScene) {
    static std::vector<QGraphicsItem*> pose_items, landmark_items;
    static std::vector<QGraphicsEllipseItem*> covariance_items;

    // Clear previous items
    for (auto& item : pose_items) { pScene->removeItem(item); delete item; }
    for (auto& item : landmark_items) { pScene->removeItem(item); delete item; }
    for (auto& item : covariance_items) { pScene->removeItem(item); delete item; }
    pose_items.clear();
    landmark_items.clear();
    covariance_items.clear();

    // Get current estimates and ISAM2 instance
    gtsam::Values current_estimates = smootherISAM2->calculateEstimate();
    gtsam::ISAM2 isam = smootherISAM2->getISAM2();

    // Find latest pose
    int max_pose_index = -1;
    gtsam::Key latest_pose_key;
    gtsam::Pose3 latest_pose;

    // First pass: draw landmarks and find latest pose
    for (const auto& key_value : current_estimates) {
        const gtsam::Key& key = key_value.key;
        gtsam::Symbol symbol(key);

        if (symbol.chr() == 'x') { // Pose node
            try {
                int current_index = symbol.index();
                if (current_index > max_pose_index) {
                    max_pose_index = current_index;
                    latest_pose_key = key;
                    latest_pose = current_estimates.at<gtsam::Pose3>(key);
                }
            } catch (...) {}
        }
        else if (symbol.chr() == 'l') { // Landmark node
            try {
                gtsam::Point3 landmark_point = current_estimates.at<gtsam::Point3>(key);

                // Draw landmark
                int size = 150;
                auto item = pScene->addEllipse(-size/2, -size/2, size, size,
                                               QPen(Qt::yellow, 1), QBrush(Qt::darkYellow));
                item->setPos(landmark_point.x() * 1000.0, landmark_point.y() * 1000.0);
                landmark_items.push_back(item);

                // Draw landmark covariance
                try {
                    gtsam::Matrix covariance = isam.marginalCovariance(key);
                    gtsam::Matrix position_cov = covariance.block<2,2>(0,0);

                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(position_cov);
                    if (eigensolver.info() == Eigen::Success) {
                        Eigen::Vector2d eigenvalues = eigensolver.eigenvalues();
                        Eigen::Matrix2d eigenvectors = eigensolver.eigenvectors();

                        double angle = atan2(eigenvectors(1,0), eigenvectors(0,0));
                        double width = 2.0 * sqrt(eigenvalues(0)) * 1000.0 * 3.0;
                        double height = 2.0 * sqrt(eigenvalues(1)) * 1000.0 * 3.0;

                        auto cov_item = pScene->addEllipse(-width/2, -height/2, width, height,
                                                           QPen(QColor(0, 255, 0, 150), 10));
                        cov_item->setPos(landmark_point.x() * 1000.0, landmark_point.y() * 1000.0);
                        cov_item->setRotation(qRadiansToDegrees(angle));
                        cov_item->setZValue(-1);
                        covariance_items.push_back(cov_item);
                    }
                } catch (...) {}
            } catch (...) {}
        }
    }

    // Draw latest pose and its covariance
    if (max_pose_index >= 0) {
        try {
            const gtsam::Point3 translation = latest_pose.translation();
            const gtsam::Rot3 rotation = latest_pose.rotation();

            // Draw pose
            QPolygonF triangle;
            triangle << QPointF(0, 150) << QPointF(-150, -150) << QPointF(150, -150) << QPointF(0, 150);
            auto item = pScene->addPolygon(triangle, QPen(Qt::darkBlue, 25));
            item->setPos(translation.x() * 1000.0, translation.y() * 1000.0);
            item->setRotation(qRadiansToDegrees(rotation.yaw()));
            pose_items.push_back(item);

            // Draw pose covariance
            try {
                gtsam::Matrix covariance = isam.marginalCovariance(latest_pose_key);
                gtsam::Matrix position_cov = covariance.block<2,2>(0,0);

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(position_cov);
                if (eigensolver.info() == Eigen::Success) {
                    Eigen::Vector2d eigenvalues = eigensolver.eigenvalues();
                    Eigen::Matrix2d eigenvectors = eigensolver.eigenvectors();

                    double angle = atan2(eigenvectors(1,0), eigenvectors(0,0));
                    double width = 2.0 * sqrt(eigenvalues(0)) * 1000.0 * 3.0;
                    double height = 2.0 * sqrt(eigenvalues(1)) * 1000.0 * 3.0;

                    auto cov_item = pScene->addEllipse(-width/2, -height/2, width, height,
                                                       QPen(QColor(255, 0, 0, 150), 10));
                    cov_item->setPos(translation.x() * 1000.0, translation.y() * 1000.0);
                    cov_item->setRotation(qRadiansToDegrees(angle));
                    cov_item->setZValue(-1);
                    covariance_items.push_back(cov_item);
                }
            } catch (...) {}
        } catch (...) {}
    }
}

void GTSAMGraph::draw_landmark_measurements(QGraphicsScene *pScene, Pose3 pose, Point2 landmark, bool clear)
{
    static std::vector<QGraphicsItem*> observation_items;

    // Clear previous items
    if(clear)
    {
        for(auto &item : observation_items) { pScene->removeItem(item); delete item; }
        observation_items.clear();
    }
    else
    {
        // Generate lines from pose to landmark
        QLineF line(QPointF(pose.translation().x() * 1000.0, pose.translation().y() * 1000.0),
                    QPointF(landmark.x() * 1000.0, landmark.y() * 1000.0));
        QGraphicsLineItem *line_item = pScene->addLine(line, QPen(Qt::red, 10));
        observation_items.push_back(line_item);
    }
}

