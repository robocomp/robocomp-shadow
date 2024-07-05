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
#define DEBUG 0

#include <genericworker.h>
#include <webots/Robot.hpp>
#include <webots/Lidar.hpp>
#include <webots/Camera.hpp>
#include <webots/RangeFinder.hpp>
#include <webots/Motor.hpp>
#include <webots/PositionSensor.hpp>
#include <webots/Node.hpp>
#include <webots/Supervisor.hpp>
#include <webots/Keyboard.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "dsr/api/dsr_api.h"
#include "dsr/gui/dsr_gui.h"
#include <doublebuffer/DoubleBuffer.h>
#include <fps/fps.h>
#include <fixedsizedeque.h>
#include <random>

using namespace Eigen;
#define TIME_STEP 33
// robot geometry
#define WHEEL_RADIUS 0.08
#define LX 0.135  // longitudinal distance from robot's COM to wheel [m].
#define LY 0.237  // lateral distance from robot's COM to wheel [m].

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
	SpecificWorker(TuplePrx tprx, bool startup_check);
	~SpecificWorker();
	bool setParams(RoboCompCommonBehavior::ParameterList params);

	RoboCompCameraRGBDSimple::TRGBD CameraRGBDSimple_getAll(std::string camera);
	RoboCompCameraRGBDSimple::TDepth CameraRGBDSimple_getDepth(std::string camera);
	RoboCompCameraRGBDSimple::TImage CameraRGBDSimple_getImage(std::string camera);
	RoboCompCameraRGBDSimple::TPoints CameraRGBDSimple_getPoints(std::string camera);

	RoboCompLaser::TLaserData Laser_getLaserAndBStateData(RoboCompGenericBase::TBaseState &bState);
	RoboCompLaser::LaserConfData Laser_getLaserConfData();
	RoboCompLaser::TLaserData Laser_getLaserData();

	RoboCompLidar3D::TData Lidar3D_getLidarData(std::string name, float start, float len, int decimationDegreeFactor);
	RoboCompLidar3D::TData Lidar3D_getLidarDataWithThreshold2d(std::string name, float distance, int decimationDegreeFactor);
    RoboCompLidar3D::TData Lidar3D_getLidarDataProyectedInImage(std::string name){ return RoboCompLidar3D::TData();};
	RoboCompLidar3D::TDataImage Lidar3D_getLidarDataArrayProyectedInImage(std::string name);

	RoboCompCamera360RGB::TImage Camera360RGB_getROI(int cx, int cy, int sx, int sy, int roiwidth, int roiheight);

	void OmniRobot_correctOdometer(int x, int z, float alpha);
	void OmniRobot_getBasePose(int &x, int &z, float &alpha);
	void OmniRobot_getBaseState(RoboCompGenericBase::TBaseState &state);
	void OmniRobot_resetOdometer();
	void OmniRobot_setOdometer(RoboCompGenericBase::TBaseState state);
	void OmniRobot_setOdometerPose(int x, int z, float alpha);
	void OmniRobot_setSpeedBase(float advx, float advz, float rot);
	void OmniRobot_stopBase();

	RoboCompVisualElements::TObjects VisualElements_getVisualObjects(RoboCompVisualElements::TObjects objects);
	void VisualElements_setVisualObjects(RoboCompVisualElements::TObjects objects);

    void Webots2Robocomp_resetWebots();
	void Webots2Robocomp_setPathToHuman(int humanId, RoboCompGridder::TPath path);

	void JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data);

public slots:
	void compute();
	int startup_check();

	void initialize(int period);
	void modify_node_slot(std::uint64_t id, const std::string &type);
	void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
	void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
	void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
	void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
	void del_node_slot(std::uint64_t from){};  
private:
	// DSR graph
	std::shared_ptr<DSR::DSRGraph> G;

	//DSR params
	std::string agent_name;
	int agent_id;

	bool tree_view;
	bool graph_view;
	bool qscene_2d_view;
	bool osg_3d_view;

	// DSR graph viewer
	std::unique_ptr<DSR::DSRViewer> graph_viewer;
	QHBoxLayout mainLayout;
	bool startup_check_flag;

    FPSCounter fps;

    webots::Supervisor* robot;
    webots::Lidar* lidar_helios;
    webots::Lidar* lidar_pearl;
    webots::Camera* camera;
    webots::RangeFinder* range_finder;
    webots::Camera* camera360_1;
    webots::Camera* camera360_2;
    webots::Motor *motors[4];
    webots::PositionSensor *ps[4];

    // Keyboard
    webots::Keyboard* keyboard;
    int previousKey = 0;

    void receiving_lidarData(string name, webots::Lidar* _lidar, DoubleBuffer<RoboCompLidar3D::TData, RoboCompLidar3D::TData>& lidar_doubleBuffer, FixedSizeDeque<RoboCompLidar3D::TData>& delay_queue);
    void receiving_cameraRGBData(webots::Camera* _camera);
    void receiving_depthImageData(webots::RangeFinder* _rangeFinder);
    void receiving_camera360Data(webots::Camera* _camera1, webots::Camera* _camera2);

    //DSR
    void insert_robot_speed_dsr();

    // Laser
    RoboCompLaser::TLaserData laserData;
    RoboCompLaser::LaserConfData laserDataConf;

    // Lidar3d
    //    RoboCompLidar3D::TData lidar3dData_helios;
    //    RoboCompLidar3D::TData lidar3dData_pearl;

    // Camera RGBD simple
    RoboCompCameraRGBDSimple::TDepth depthImage;
    RoboCompCameraRGBDSimple::TImage cameraImage;

    // Camera 360
    RoboCompCamera360RGB::TImage camera360Image;

    // Human Tracking
    struct WebotsHuman{
        webots::Node *node;
        RoboCompGridder::TPath path;
        RoboCompGridder::TPoint currentTarget;
        std::vector<double> startPosition;
        std::vector<double> startOrientation;

        // initialize a vector of doubles with size of 5
    };

    std::map<int, WebotsHuman> humanObjects;
//    void parseHumanObjects();
    void parseHumanObjects(bool firstTime=false);

    // Auxiliar functions
    void printNotImplementedWarningMessage(string functionName);

    // Webots2RoboComp interface
    void moveHumanToNextTarget(int humanId);
    void humansMovement();

    struct PARAMS
    {
        bool delay = false;
        bool do_joystick = false;
    };
    PARAMS pars;

    FixedSizeDeque<RoboCompCamera360RGB::TImage> camera_queue{10};
    //Is it necessary to use two lidar queues? One for each lidaR?
    FixedSizeDeque<RoboCompLidar3D::TData> pearl_delay_queue{10};
    FixedSizeDeque<RoboCompLidar3D::TData> helios_delay_queue{10};

    // Double buffer
    DoubleBuffer<RoboCompCamera360RGB::TImage, RoboCompCamera360RGB::TImage> double_buffer_360;

    //Lidar3D doublebuffer
    DoubleBuffer<RoboCompLidar3D::TData, RoboCompLidar3D::TData> double_buffer_helios;
    DoubleBuffer<RoboCompLidar3D::TData, RoboCompLidar3D::TData> double_buffer_pearl;

    Matrix4d create_affine_matrix(double a, double b, double c, Vector3d trans);

    // Vector for obstacle position
    std::vector<float> obstacleStartPosition{3};
    void setObstacleStartPosition();
    void setElementsToStartPosition();
    std::pair<double, double> getRandomPointInRadius(double centerX, double centerY, double radius);

    std::vector<float> robotStartPosition{3};
    std::vector<float> robotStartOrientation{3};

    bool webots_initializated = false;
    float max_person_speed = 0.7;

    // Structure for store two points that delimits a line
    struct PoseLine{
        RoboCompGridder::TPoint p1;
        RoboCompGridder::TPoint p2;
    };
    RoboCompGridder::TPoint getRandomPointInLine(PoseLine pl);
    // Pose line for elements in IROS experiment
    PoseLine personPoseLine{.p1 = RoboCompGridder::TPoint{.x=3.05462, .y=1.19488}, .p2 = {.x=3.09135, .y=-0.420693}};
    PoseLine obstaclePoseLine{.p1 = RoboCompGridder::TPoint{.x=-1.45976, .y=-0.0863835}, .p2 = {.x=-0.915218, .y=-1.45976}};
    PoseLine robotPoseLine1{.p1 = RoboCompGridder::TPoint{.x=-0.258, .y=2.85}, .p2 = {.x=-3.38, .y=0.884}};
    PoseLine robotPoseLine2{.p1 = RoboCompGridder::TPoint{.x=2.45, .y=-1.85}, .p2 = {.x=0.3, .y=-3.0}};

    //METRICS
    std::ofstream file;
    int experiment_id = 0;
    // Tuple for storing a TUPLE OF METRICS
    std::tuple<int, int, Eigen::Vector2f> calculate_collision_metrics();
    //webots metric data vector
    std::vector<std::vector<double>> getPositions();

    bool reset = false;
    void reset_sim();

    double generarRuido(double stddev);
};

#endif
