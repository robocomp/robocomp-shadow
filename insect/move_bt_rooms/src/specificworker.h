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
#include <door_detector_alej.h>
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/action_node.h>
#include "behaviortree_cpp/behavior_tree.h"
#include <functional>
//#include <behaviortree_cpp/dummy_nodes.h>

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
	SpecificWorker(TuplePrx tprx, bool startup_check);
	~SpecificWorker();
	bool setParams(RoboCompCommonBehavior::ParameterList params);

	void OmniRobot_correctOdometer(int x, int z, float alpha);
	void OmniRobot_getBasePose(int &x, int &z, float &alpha);
	void OmniRobot_getBaseState(RoboCompGenericBase::TBaseState &state);
	void OmniRobot_resetOdometer();
	void OmniRobot_setOdometer(RoboCompGenericBase::TBaseState state);
	void OmniRobot_setOdometerPose(int x, int z, float alpha);
	void OmniRobot_setSpeedBase(float advx, float advz, float rot);
	void OmniRobot_stopBase();

    void draw_floor_line(const vector<Eigen::Vector2f> &lines);

    struct Data{
        std::vector<Door_detector::Door> detected_doors;
        Door_detector::Door target_door;
        float advx_point;
        float advy_point;
        float rot_point;
        bool chosen_door;
    };

public slots:
	void compute();
	int startup_check();
	void initialize(int period);

private:
//	std::shared_ptr < InnerModel > innerModel;
	bool startup_check_flag;

    // Crear un objeto de la clase DoorDetector
//    DoorDetector detector;
    Door_detector detector;
    std::vector<Door_detector::Door> detected_doors;
    AbstractGraphicViewer *viewer;

    float min_distance;
    int current_phi;
    RoboCompLidar3D::TPoint aux_point;
    BT::BehaviorTreeFactory factory;
    BT::Tree tree;

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

        if(_data->detected_doors.size() > 0) //and !_data->chosen_door
        {
            std::srand(static_cast<unsigned int>(std::time(nullptr)));
            std::cout << "DoorSize" << _data->detected_doors.size() << std::endl;

            // Obtener un índice aleatorio
//            std::size_t randomIndex = std::rand() % _data->detected_doors.size();

            // Acceder al elemento aleatorio
            this->_data->target_door = (_data->detected_doors)[0];
//            this->_data->target_door.dist_pmedio = std::sqrt(std::pow(this->_data->target_door.punto_medio.x(), 2) + std::pow(this->_data->target_door.punto_medio.y(), 2));

            std::cout << "Puerta seleccionada: " << this->_data->target_door.punto_medio << std::endl;

            this->_data->rot_point = 0.;
//            _data->chosen_door = true;
            return BT::NodeStatus::SUCCESS;
        }
        else
        {
//            this->_data->rot_point = 0.3;
//            this->_data->advx_point = 0.0;

            std::cout << "Turning looking for door" << std::endl;
            return BT::NodeStatus::FAILURE;
        }
    }

private:
    std::shared_ptr<SpecificWorker::Data> _data;
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
        std::cout << this->name() << " Dist:" << this->_data->target_door.dist_pmedio  << std::endl;

        if( this->_data->target_door.punto_medio.norm() > this->dist_threshold ) //and _data->chosen_door
        {

            if( _data->target_door.pre_middle_point.norm() > 400)
            {
                this->_data->rot_point = atan2( this->_data->target_door.pre_middle_point.x(), this->_data->target_door.pre_middle_point.y()) * 1.3;  // dumps rotation for small resultant force;
                this->_data->advx_point =  this->_data->target_door.pre_middle_point.x()/1000;
                this->_data->advy_point = this->_data->target_door.pre_middle_point.y()/1000;
                std::cout << "Speedx:" << this->_data->advx_point << "Speedy:" << this->_data->advy_point << " rot:" << this->_data->rot_point << std::endl;

                return BT::NodeStatus::FAILURE;
            }
            
            if( _data->target_door.punto_medio.norm() > 400)
            {
                this->_data->rot_point = atan2( this->_data->target_door.punto_medio.x(), this->_data->target_door.punto_medio.y()) * 1.3;  // dumps rotation for small resultant force;
                this->_data->advx_point =  this->_data->target_door.punto_medio.x()/1000;
                this->_data->advy_point = this->_data->target_door.punto_medio.y()/1000;
                std::cout << "Speedx:" << this->_data->advx_point << "Speedy:" << this->_data->advy_point << " rot:" << this->_data->rot_point << std::endl;

                return BT::NodeStatus::FAILURE;
            }

            if( _data->target_door.post_middle_point.norm() > 400)
            {
                this->_data->rot_point = atan2( this->_data->target_door.post_middle_point.x(), this->_data->target_door.post_middle_point.y()) * 1.3;  // dumps rotation for small resultant force;
                this->_data->advx_point =  this->_data->target_door.post_middle_point.x()/1000;
                this->_data->advy_point = this->_data->target_door.post_middle_point.y()/1000;
                std::cout << "Speedx:" << this->_data->advx_point << "Speedy:" << this->_data->advy_point << " rot:" << this->_data->rot_point << std::endl;

                return BT::NodeStatus::FAILURE;
            }
        }
        else
        {
            std::cout << this->name() << "else:" << std::endl;
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
