/*
 *    Copyright (C) 2025 by YOUR NAME HERE
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
 *    adouble with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
	\brief
	@author authorname
*/



#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <genericworker.h>
#include "dsr/api/dsr_api.h"
#include "dsr/gui/dsr_gui.h"
#include "GTSAMGraph.h"
#include <doublebuffer/DoubleBuffer.h>
#include "fixedsizedeque.h"
#include "custom_widget.h"
#include "ui_localUI.h"
#include <boost/circular_buffer.hpp>
#include <mutex>
#include <clipper2/clipper.h>

#include <chrono>

// If you want to reduce the period automatically due to lack of use, you must uncomment the following line
//#define HIBERNATION_ENABLED

using namespace Clipper2Lib;
constexpr double SCALE = 1e6;

/**
 * \brief Class SpecificWorker implements the core functionality of the component.
 */
class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    /**
     * \brief Constructor for SpecificWorker.
     * \param configLoader Configuration loader for the component.
     * \param tprx Tuple of proxies required for the component.
     * \param startup_check Indicates whether to perform startup checks.
     */
	SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);

	/**
     * \brief Destructor for SpecificWorker.
     */
	~SpecificWorker();

	void FullPoseEstimationPub_newFullPose(RoboCompFullPoseEstimation::FullPoseEuler pose);


public slots:

	/**
	 * \brief Initializes the worker one time.
	 */
	void initialize();

	/**
	 * \brief Main compute loop of the worker.
	 */
	void compute();

	/**
	 * \brief Handles the emergency state loop.
	 */
	void emergency();

	/**
	 * \brief Restores the component from an emergency state.
	 */
	void restore();

    /**
     * \brief Performs startup checks for the component.
     * \return An integer representing the result of the checks.
     */
	int startup_check();

	void modify_node_slot(std::uint64_t id, const std::string &type);
	void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names);
	void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type);
	void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
	void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag);
	void del_node_slot(std::uint64_t from){};     
private:

	/**
     * \brief Flag indicating whether startup checks are enabled.
     */
	bool startup_check_flag;
    DSR::QScene2dViewer* widget_2d;
    CustomWidget *room_widget;

    struct Params
    {
        std::string robot_name = "Shadow";
        double scale = 1.0; // Scale factor for measurements
    };
    Params params;

    //Odometry
    int odometry_node_id = -1;
    Eigen::Vector4d last_odometry = Eigen::Vector4d::Zero();
    double last_odometry_timestamp = 0;
    float odometry_noise_std_dev = 1.0f; // Standard deviation for odometry noise
    float odometry_noise_angle_std_dev = 1.0f; // Standard deviation for odometry angle noise
    float measurement_noise_std_dev = 1.0f; // Standard deviation for measurement noise
    boost::circular_buffer<std::tuple<double, Eigen::Vector3d, Eigen::Vector3d>> odometry_queue{100};
    DoubleBuffer<std::tuple<double, Eigen::Vector3d, Eigen::Vector3d>, std::tuple<double, Eigen::Vector3d, Eigen::Vector3d>> odom_buffer;

    std::mutex odom_mutex;

    //Room
    uint64_t last_room_id = -1;
    uint64_t actual_room_id = -1;
    std::string actual_room_name = "";
    bool room_initialized = false;
    int iterations = 0;

    //Graph
    int state = 0;
    bool reset = false;
    bool current_edge_set = false;
    bool first_rt_set = false;
    GTSAMGraph gtsam_graph;

    QPolygonF room_polygon;        // Empty polygon initially
    QPolygonF security_polygon;    // Empty polygon initially

    //Timers
    std::chrono::system_clock::time_point rt_set_last_time;
    float rt_time_min = 1.0;

    std::chrono::system_clock::time_point last_update_with_corners;
    std::chrono::system_clock::time_point elapsed;

    // RT Values
    std::vector<float> translation_to_set = {0.0, 0.0, 0.0};
    std::vector<float> rotation_to_set = {0.0, 0.0, 0.0};

    Eigen::Affine3d robot_pose;
    double last_timestamp = 0;

    // Corners
    std::map<int, uint64_t> corners_last_update_timestamp;
    std::vector<Eigen::Vector2d> safe_polygon;

    // DSR Values
    uint64_t robot_id = 0, room_id = 0;
    std::vector<std::string > attr_reading_variables {"robot_current_advance_speed", "robot_current_side_speed", "robot_current_angular_speed", "timestamp_alivetime"};

    // RT APi
    std::unique_ptr<DSR::RT_API> rt_api;
    std::unique_ptr<DSR::InnerEigenAPI> inner_api;

    void draw_robot_in_room(QGraphicsScene *pScene);
    void draw_nominal_corners(QGraphicsScene *pScene, const std::vector<std::tuple<int, double, Eigen::Vector3d>> &nominal_corners, const std::vector<Eigen::Vector2d> &security_polygon, bool erase=false);
    void draw_measured_corners(QGraphicsScene *pScene, const std::vector<std::tuple<int, double, Eigen::Vector3d, bool>> &measured_corners);

    bool initialize_graph();
    std::vector<std::tuple<int, double, Eigen::Vector3d>> get_nominal_corners();
    std::vector<Eigen::Vector2d> shrink_polygon_to_safe_zone(const std::vector<Eigen::Vector2d>& corners2D, double margin_meters = 0.9);
    std::vector<Eigen::Vector2d> fromClipper2Path(const Path64& path);
    Path64 toClipper2Path(const std::vector<Eigen::Vector2d>& poly);
    std::optional<std::pair<double, gtsam::Pose3>> get_dsr_robot_pose();
    std::optional<std::pair<std::uint64_t, Eigen::Vector3d>>  transform_point(     const std::string& from_node_name,
                                                                   const std::string& to_node_name,
                                                                    std::uint64_t timestamp);
    void update_robot_odometry_data_in_DSR(const std::tuple<double, Eigen::Vector3d, Eigen::Vector3d> &odom_value);
    std::vector<std::tuple<double, Eigen::Vector3d, Eigen::Vector3d>> copy_odometry_buffer(
            boost::circular_buffer<std::tuple<double, Eigen::Vector3d, Eigen::Vector3d>>& buffer,
            std::mutex& buffer_mutex,
            double min_timestamp = 0);
    std::tuple<double, Eigen::Affine3d> integrate_odometry(
            double current_time, Eigen::Vector3d translation, Eigen::Vector3d rotation);
    bool is_pose_inside_polygon(const Eigen::Vector2d& point, const std::vector<Eigen::Vector2d>& polygon);
    std::vector<std::tuple<int, double, Eigen::Vector3d, bool>> get_measured_corners(double timestamp);


    void update_robot_dsr_pose(double x, double y, double ang, double timestamp);

signals:
	//void customSignal();
};

#endif
