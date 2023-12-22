/*
 *    Copyright (C) 2023 by YOUR NAME HERE
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
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/action_node.h>
#include <chrono>
#include "behaviortree_cpp/behavior_tree.h"
#include "door_detector.h"
#include <functional>
//#include <behaviortree_cpp/dummy_nodes.h>

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
	SpecificWorker(TuplePrx tprx, bool startup_check);
	~SpecificWorker();
	bool setParams(RoboCompCommonBehavior::ParameterList params);

	void GridPlanner_setPlan(RoboCompGridPlanner::TPlan plan);
	void OmniRobot_correctOdometer(int x, int z, float alpha);
	void OmniRobot_getBasePose(int &x, int &z, float &alpha);
	void OmniRobot_getBaseState(RoboCompGenericBase::TBaseState &state);
	void OmniRobot_resetOdometer();
	void OmniRobot_setOdometer(RoboCompGenericBase::TBaseState state);
	void OmniRobot_setOdometerPose(int x, int z, float alpha);
	void OmniRobot_setSpeedBase(float advx, float advz, float rot);
	void OmniRobot_stopBase();

    struct Data{

        //Door variables
        std::vector<DoorDetector::Door> detected_doors;
        DoorDetector::Door target_door;
        float advx_point;
        float advy_point;
        float rot_point;
        bool chosen_door;
        std::chrono::high_resolution_clock::time_point startThroughDoor;

        //Plan variables
        bool valid;
        float subtarget_x = 0.f;
        float subtarget_y = 0.f;
    };

public slots:
	void compute();
	int startup_check();
	void initialize(int period);

private:

    //	std::shared_ptr < InnerModel > innerModel;
	bool startup_check_flag;
    struct Constants
    {
        std::string lidar_name = "helios";
        std::vector<std::pair<float, float>> ranges_list = {{1000, 2500}};
    };
    Constants consts;

    // Crear un objeto de la clase DoorDetector
    DoorDetector detector;
    std::vector<DoorDetector::Door> detected_doors;
    AbstractGraphicViewer *viewer;

    //Alias
    using Line = std::vector<Eigen::Vector2f>;
    using Lines = std::vector<Line>;

    void draw_floor_line(const vector<Eigen::Vector2f> &lines);
    std::vector<Line> extract_lines(const RoboCompLidar3D::TPoints &points, const vector<std::pair<float, float>> &ranges);

    float min_distance;
    int current_phi;
    RoboCompLidar3D::TPoint aux_point;
    BT::BehaviorTreeFactory factory;
    BT::Tree tree;

    // Thread
    void robot_comunication();
    std::thread robot_comunication_th;

    std::shared_ptr<Data> dataPtr;
};


//Blocking nodes they do not return RUNNING, only SUCCESS or FAILURE
class LookForNewDoor : public BT::SyncActionNode
{
public:
    LookForNewDoor(const std::string &name, const BT::NodeConfig &config, std::shared_ptr<SpecificWorker::Data> data) :
            BT::SyncActionNode(name, config), _data(data)  //const BT::NodeConfiguration &config BT::SyncActionNode(name, config)
    {
    }

    // This example doesn't require any port
    static BT::PortsList providedPorts() { return { }; } //BT::OutputPort<Door_detector::Door>("selected")}

    // You must override the virtual function tick()
    BT::NodeStatus tick() override
    {
        std::cout << this->name() << std::endl;
        bool found = false;
        // Sending plan to remote interface

        if(_data->detected_doors.size() > 0) //and !_data->chosen_door
        {
            std::cout << "DoorSize" << _data->detected_doors.size() << std::endl;

            if( this->_data->chosen_door )
            {
                for (DoorDetector::Door &d : this->_data->detected_doors)
                {
                    if( d == this->_data->target_door )
                    {
                        found = true;
                        this->_data->target_door = d;
                    }
                }

                if(not found)
                    this->_data->target_door = (_data->detected_doors)[0];
            }
            else
                this->_data->target_door = (_data->detected_doors)[0];
            //Check if not the last targeted door

            std::cout << "Puerta seleccionada: " << this->_data->target_door.middle << std::endl;

//            this->_data->rot_point = 0.;
            this->_data->chosen_door = true;
            return BT::NodeStatus::SUCCESS;
        }
        else
        {
            this->_data->rot_point = 0.3;
            this->_data->advx_point = 0.0;
            this->_data->chosen_door = false;

            std::cout << "Turning looking for door" << std::endl;
            return BT::NodeStatus::FAILURE;
        }
    }

private:
    std::shared_ptr<SpecificWorker::Data> _data;
};

//Blocking nodes they do not return RUNNING, only SUCCESS or FAILURE
class InFrontDoor : public BT::SyncActionNode
{
public:
    InFrontDoor(const std::string &name, const BT::NodeConfig& config, std::shared_ptr<SpecificWorker::Data> data) :
            BT::SyncActionNode(name, config), _data(data)
    {
    }

//    static BT::PortsList providedPorts()
//    {
//        // Optionally, a port can have a human readable description
//        const char*  description = "Door selected to go through";
//        return { BT::InputPort<Door_detector::Door>("target", description) };
//    }

    static BT::PortsList providedPorts() { return { }; }

    // You must override the virtual function tick()
    BT::NodeStatus tick() override
    {
        std::pair<Eigen::Vector2f, Eigen::Vector2f> perpendicular_points = this->_data->target_door.point_perpendicular_to_door_at();

        Eigen::Vector2f pre_middle_point = perpendicular_points.first;
        Eigen::Vector2f post_middle_point = perpendicular_points.second;

        std::cout << this->name() << " rot:" << atan2( pre_middle_point.x(), pre_middle_point.y())  << std::endl;

        if (pre_middle_point.norm() > 200)
        {
            this->_data->rot_point = atan2(pre_middle_point.x(), pre_middle_point.y()) ;  //*1.3 dumps rotation for small resultant force;
            this->_data->advx_point = pre_middle_point.x() / 2000;
            this->_data->advy_point = pre_middle_point.y() / 2000;
            std::cout << "going" << this->_data->advx_point << this->_data->advx_point << std::endl;
            return BT::NodeStatus::FAILURE;
        }
        else
        {
            std::cout << "rotating" << std::endl;
            //Check
            if( -0.1 < atan2( post_middle_point.x(), post_middle_point.y()) and atan2( post_middle_point.x(), post_middle_point.y()) < 0.1)
            {
                this->_data->startThroughDoor = std::chrono::high_resolution_clock::now();
                return BT::NodeStatus::SUCCESS;
            }
            else
            {
                this->_data->advx_point = 0.0;
                this->_data->advy_point = 0.0;
                this->_data->rot_point = atan2( post_middle_point.x(), post_middle_point.y());
                return BT::NodeStatus::FAILURE;
            }
        }
    }

private:

    //Variables
    std::shared_ptr<SpecificWorker::Data> _data;
    float speed = 0.0 , rot = 0.0;
    float dist_threshold = 300.0;
};

//Blocking nodes they do not return RUNNING, only SUCCESS or FAILURE
class GoThroughDoor : public BT::SyncActionNode
{
public:
    GoThroughDoor(const std::string &name, const BT::NodeConfig& config, std::shared_ptr<SpecificWorker::Data> data) :
            BT::SyncActionNode(name, config), _data(data)
    {
    }

//    static BT::PortsList providedPorts()
//    {
//        // Optionally, a port can have a human readable description
//        const char*  description = "Door selected to go through";
//        return { BT::InputPort<Door_detector::Door>("target", description) };
//    }

    static BT::PortsList providedPorts() { return { }; }

    // You must override the virtual function tick()
    BT::NodeStatus tick() override
    {
        std::cout << this->name() << " Dist:" << this->_data->target_door.middle.norm()  << std::endl;

        std::pair<Eigen::Vector2f, Eigen::Vector2f> perpendicular_points = this->_data->target_door.point_perpendicular_to_door_at();

        Eigen::Vector2f post_middle_point = perpendicular_points.second;

//        std::cout << this->name() << " rot:" << atan2( post_middle_point.x(), post_middle_point.y())  << std::endl;

//        std::cout << "Time:" << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - this->_data->startThroughDoor).count();

        //Limbo
        if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - this->_data->startThroughDoor).count() < 6) {
            this->_data->rot_point = 0.0;  //*1.3 dumps rotation for small resultant force;
            this->_data->advx_point = post_middle_point.x() / 2000;
            this->_data->advy_point = post_middle_point.y() / 2000;

            return BT::NodeStatus::FAILURE;
        }
        else
        {
            this->_data->advx_point = 0.0;
            this->_data->advy_point = 0.0;
            this->_data->rot_point = 0.0;
            this->_data->chosen_door = false;

            return BT::NodeStatus::SUCCESS;
        }
    }

private:

    //Variables
    std::shared_ptr<SpecificWorker::Data> _data;
    float speed = 0.0 , rot = 0.0;
    float dist_threshold = 300.0;
};

//Blocking nodes they do not return RUNNING, only SUCCESS or FAILURE
class GoMiddleOfTheRoom : public BT::SyncActionNode
{
public:
    GoMiddleOfTheRoom(const std::string &name, const BT::NodeConfig& config) :
            BT::SyncActionNode(name, config)
    {
    }

    static BT::PortsList providedPorts() { return {}; }

    // You must override the virtual function tick()
    BT::NodeStatus tick() override
    {
        std::cout << this->name() << std::endl;

        return BT::NodeStatus::FAILURE;
    }
};

////Blocking nodes they do not return RUNNING, only SUCCESS or FAILURE
//class GoMiddleOfTheRoom : public BT::SyncActionNode
//{
//public:
//    GoMiddleOfTheRoom(const std::string &name, const BT::NodeConfig& config, SpecificWorker* component, const std::function<RoboCompLidar3D::TData()>& getLidar) :
//    BT::SyncActionNode(name, config), _component(component), _getLidar(getLidar)
//    {
//    }
//
//    static BT::PortsList providedPorts() { return {}; }
//
//    // You must override the virtual function tick()
//    BT::NodeStatus tick() override
//    {
//        std::cout << "GoMiddleOfTheRoom: " << _component->lidar3d_proxy->getLidarDataWithThreshold2d("helios",8000).points.size() << std::endl;
//        return BT::NodeStatus::SUCCESS;
//    }
//
//private:
//    std::function<RoboCompLidar3D::TData()> _getLidar;
//    SpecificWorker* _component;
//};

#endif
