/*
 *    Copyright (C) 2021 by YOUR NAME HERE
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
#include "specificworker.h"
#include <cppitertools/zip.hpp>
#include <cppitertools/sliding_window.hpp>
#include <cppitertools/enumerate.hpp>
#include <chrono>

/**
* \brief Default constructor
*/
SpecificWorker::SpecificWorker(TuplePrx tprx, bool startup_check) : GenericWorker(tprx)
{
	this->startup_check_flag = startup_check;
    QLoggingCategory::setFilterRules("*.debug=false\n");
}
/**
* \brief Default destructor
*/
SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
	G->write_to_json_file("./"+agent_name+".json");
	G.reset();
}
bool SpecificWorker::setParams(RoboCompCommonBehavior::ParameterList params)
{
    try
    {
        agent_name = params.at("agent_name").value;
        agent_id = stoi(params.at("agent_id").value);
        tree_view = params.at("tree_view").value == "true";
        graph_view = params.at("graph_view").value == "true";
        qscene_2d_view = params.at("2d_view").value == "true";
        osg_3d_view = params.at("3d_view").value == "true";
    }
    catch(const std::exception &e) {std::cout << e.what() << std::endl;}
	return true;
}
void SpecificWorker::initialize(int period)
{
	std::cout << "Initialize worker" << std::endl;

	if(this->startup_check_flag)
		this->startup_check();
	else
	{
        // create graph
        G = std::make_shared<DSR::DSRGraph>(0, agent_name, agent_id, ""); // Init nodes
        std::cout<< __FUNCTION__ << "Graph loaded" << std::endl;

        //dsr update signals
        connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::add_or_assign_node_slot);
        //connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
        //connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_attrs_slot);
        //connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
        //connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

        // Graph viewer
        using opts = DSR::DSRViewer::view;
        int current_opts = 0;
        opts main = opts::none;
        if(tree_view)
            current_opts = current_opts | opts::tree;
        if(graph_view)
        {
            current_opts = current_opts | opts::graph;
            main = opts::graph;
        }
        if(qscene_2d_view)
            current_opts = current_opts | opts::scene;
        if(osg_3d_view)
            current_opts = current_opts | opts::osg;
        graph_viewer = std::make_unique<DSR::DSRViewer>(this, G, current_opts, main);
        setWindowTitle(QString::fromStdString(agent_name + "-") + QString::number(agent_id));

        //Inner Api
        inner_eigen = G->get_inner_eigen_api();
        agent_info_api = std::make_unique<DSR::AgentInfoAPI>(G.get());


//        try {
////            float x = 3.85;
//            float x = 0.0;
////            float y = -22.387;
//            float y = 0.0;
//            float z = 0.0;
//            float rx = 0;
//            float ry = 0;
//            float rz = 180;
//            fullposeestimation_proxy->setInitialPose(x, y, z, rx, ry, rz);
//        }
//        catch (const Ice::Exception &e) { std::cout << e.what() << std::endl; };

        convertion_matrix <<  1.f, 0.f, 0.f,
                0.f , 1.f, 0.f,
                0.f, 0.f, 1.f;

        // Setting robot state to 0
        if(auto robot_o = G->get_node(robot_name); robot_o.has_value())
        {
            auto robot = robot_o.value();
            if (auto parent_o = G->get_parent_node(robot); parent_o.has_value())
            {
                auto parent = parent_o.value();
                if(auto edge_o = rt->get_edge_RT(parent, robot.id()); edge_o.has_value())
                {
                    auto edge = edge_o.value();
                    G->modify_attrib_local<rt_rotation_euler_xyz_att>(edge, std::vector<float>{0.0, 0.0, 0.0});
                    G->modify_attrib_local<rt_translation_att>(edge, std::vector<float>{0.0, 0.0, 0.0});
                    G->modify_attrib_local<rt_translation_velocity_att>(edge, std::vector<float>{0.0, 0.0, 0.0});
                    G->modify_attrib_local<rt_rotation_euler_xyz_velocity_att>(edge, std::vector<float>{0.0, 0.0, 0.0});
                    G->insert_or_assign_edge(edge);
                }
                 else
                {
                    qWarning() << "No RT edge between robot and parent found. Terminate";
                    std::terminate();
                }
            } else
            {
                qWarning() << "No robot's parent node found. Terminate";
                std::terminate();
            }
        }
        else
        {
            qWarning() << "No robot node found. Terminate";
            std::terminate();
        }



        this->Period = 50;
		timer.start(Period);
	}
}
void SpecificWorker::compute()
{
    auto t_start = std::chrono::high_resolution_clock::now();

    //update_robot_localization();
    update_cameras();
    //read_battery();
    //update_servo_position();
    //auto laser = read_laser_from_robot();
    //update_laser(laser);

    //auto t_end = std::chrono::high_resolution_clock::now();
    //double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    // std::cout << elapsed_time_ms << std::endl;

    fps.print("FPS: ", [this](auto x){ graph_viewer->set_external_hz(x);});

}

/////////////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::update_cameras()
{
    try
    {
        auto rgbd = camerargbdsimple_proxy->getAll("/Shadow/camera_top");
        auto &rgb = rgbd.image;
        auto &depth = rgbd.depth;
        if( auto node = G->get_node(shadow_camera_head_name); node.has_value())
        {
            cv::Mat rgb_frame (cv::Size(rgb.width, rgb.height), CV_8UC3, &rgb.image[0], cv::Mat::AUTO_STEP);
            std::vector<uint8_t> rgb_vector; rgb_vector.assign(rgb_frame.data, rgb_frame.data + rgb_frame.total()*rgb_frame.channels());
            G->add_or_modify_attrib_local<cam_rgb_att>(node.value(),  rgb_vector);
            G->add_or_modify_attrib_local<cam_rgb_width_att>(node.value(), rgb_frame.cols);
            G->add_or_modify_attrib_local<cam_rgb_height_att>(node.value(), rgb_frame.rows);
            G->add_or_modify_attrib_local<cam_rgb_depth_att>(node.value(), rgb_frame.depth());
            G->add_or_modify_attrib_local<cam_rgb_cameraID_att>(node.value(), 3);
            G->add_or_modify_attrib_local<cam_rgb_focalx_att>(node.value(), rgb.focalx);
            G->add_or_modify_attrib_local<cam_rgb_focaly_att>(node.value(), rgb.focaly);
            G->add_or_modify_attrib_local<cam_rgb_alivetime_att>(node.value(), static_cast<std::uint64_t>(rgb.alivetime));

            // CAMBIAR POR UNA ASIGNACION DIRECTA DESDE CameraRGBDSimple
            cv::Mat depth_frame(cv::Size(depth.width, depth.height), CV_32FC1, &depth.depth[0], cv::Mat::AUTO_STEP);
            std::vector<uint8_t> depth_vector; depth_vector.assign(depth_frame.data, depth_frame.data + depth_frame.total()* sizeof(float));
            G->add_or_modify_attrib_local<cam_depth_att>(node.value(),  depth_vector);
            G->add_or_modify_attrib_local<cam_depth_width_att>(node.value(), depth_frame.cols);
            G->add_or_modify_attrib_local<cam_depth_height_att>(node.value(), depth_frame.rows);
            G->add_or_modify_attrib_local<cam_depth_cameraID_att>(node.value(), 3);
            G->add_or_modify_attrib_local<cam_depth_focalx_att>(node.value(), depth.focalx);
            G->add_or_modify_attrib_local<cam_depth_focaly_att>(node.value(), depth.focaly);
            G->add_or_modify_attrib_local<cam_depth_alivetime_att>(node.value(), static_cast<std::uint64_t>(depth.alivetime));
            G->update_node(node.value());
        }
        else
                qWarning() << __FUNCTION__ << "Node " << QString::fromStdString(shadow_camera_head_name) << " not found";
    }
    catch (const Ice::Exception &e){std::cout << e.what() << " reading camera-top" << std::endl;}

    try
    {
        auto omni_rgb = camerargbdsimple_proxy->getImage("/Shadow/omnicamera/sensorRGB");
        auto omni_depth = camerargbdsimple_proxy->getImage("/Shadow/omnicamera/sensorDepth");
        cv::Mat omni_rgbd_frame (cv::Size(omni_rgb.width, omni_rgb.height), CV_8UC3, &omni_rgb.image[0], cv::Mat::AUTO_STEP);
        cv::Mat omni_depth_frame (cv::Size(omni_depth.width, omni_depth.height), CV_8UC3, &omni_depth.image[0], cv::Mat::AUTO_STEP);
        cv::cvtColor(omni_depth_frame, omni_depth_frame, cv::COLOR_RGB2GRAY);
        cv::Mat omni_depth_float;
        omni_depth_frame.convertTo(omni_depth_float, CV_32FC1);
//        cv::Mat omni_color;
//        omni_depth_frame.convertTo(omni_color, CV_8UC3, 255. / 10, 0);
//        applyColorMap(omni_color, omni_color, cv::COLORMAP_RAINBOW); //COLORMAP_HSV tb
//        cv::imshow("Depth image ", omni_color);

        if( auto node = G->get_node(shadow_omni_camera_name); node.has_value())
        {
            std::vector<uint8_t> rgb;
            rgb.assign(omni_rgbd_frame.data, omni_rgbd_frame.data + omni_rgbd_frame.total() * omni_rgbd_frame.channels());
            G->add_or_modify_attrib_local<cam_rgb_att>(node.value(), rgb);
            G->add_or_modify_attrib_local<cam_rgb_width_att>(node.value(), omni_rgbd_frame.cols);
            G->add_or_modify_attrib_local<cam_rgb_height_att>(node.value(), omni_rgbd_frame.rows);
            G->add_or_modify_attrib_local<cam_rgb_depth_att>(node.value(), omni_rgbd_frame.depth());
            G->add_or_modify_attrib_local<cam_rgb_cameraID_att>(node.value(), 2);
            G->add_or_modify_attrib_local<cam_rgb_focalx_att>(node.value(), omni_rgb.focalx);
            G->add_or_modify_attrib_local<cam_rgb_focaly_att>(node.value(), omni_rgb.focaly);
            G->add_or_modify_attrib_local<cam_rgb_alivetime_att>(node.value(), static_cast<std::uint64_t>(omni_rgb.alivetime));

            std::vector<uint8_t> depth;
            depth.assign(omni_depth_float.data, omni_depth_float.data + omni_depth_float.total() * sizeof(float));
            G->add_or_modify_attrib_local<cam_depth_att>(node.value(),  depth);
            G->add_or_modify_attrib_local<cam_depth_width_att>(node.value(), omni_depth.width);
            G->add_or_modify_attrib_local<cam_depth_height_att>(node.value(), omni_depth.height);
            G->add_or_modify_attrib_local<cam_depth_cameraID_att>(node.value(), 2);
            G->add_or_modify_attrib_local<cam_depth_focalx_att>(node.value(), omni_depth.focalx);
            G->add_or_modify_attrib_local<cam_depth_focaly_att>(node.value(), omni_depth.focaly);
            G->add_or_modify_attrib_local<cam_depth_alivetime_att>(node.value(), static_cast<std::uint64_t>(omni_depth.alivetime));
            G->update_node(node.value());
        }
        else
            qWarning() << __FUNCTION__ << "Node " << QString::fromStdString(shadow_omni_camera_name) << " not found";
    }
    catch (const Ice::Exception &e)
    {std::cout << e.what() << " reading omnicamera rgb" << std::endl; return;}
}
void SpecificWorker::read_battery()
{
    try
    {
        auto battery_level = batterystatus_proxy->getBatteryState();
        if( auto battery = G->get_node(battery_name); battery.has_value())
        {
            G->add_or_modify_attrib_local<battery_load_att>(battery.value(), (100*battery_level.percentage));
            G->update_node(battery.value());
        }
    }
    catch(const Ice::Exception &e) { /*std::cout << e.what() << std::endl;*/}
}
void SpecificWorker::update_servo_position()
{
    try
    {
        auto servo_data = this->jointmotorsimple_proxy->getMotorState("camera_pan_joint");
        float servo_position = (float)servo_data.pos;
        float servo_vel = (float)servo_data.vel;
        bool moving = servo_data.isMoving;
        if( auto servo = G->get_node("camera_pan_joint"); servo.has_value())
        {
            G->add_or_modify_attrib_local<servo_pos_att>(servo.value(), servo_position);
            G->add_or_modify_attrib_local<servo_speed_att>(servo.value(), servo_vel);
            G->add_or_modify_attrib_local<servo_moving_att>(servo.value(), moving);
            G->update_node(servo.value());
        }
    }
    catch(const Ice::Exception &e){ /*std::cout << e.what() <<  __FUNCTION__ << std::endl;*/};
}
void SpecificWorker::update_robot_localization()
{
    static RoboCompFullPoseEstimation::FullPoseEuler last_state;
    RoboCompFullPoseEstimation::FullPoseEuler pose;
    try
    {
        pose = fullposeestimation_proxy->getFullPoseEuler();
        //qInfo() << "X:" << pose.x  << "// Y:" << pose.y << "// Z:" << pose.z << "// RX:" << pose.rx << "// RY:" << pose.ry << "// RZ:" << pose.rz;
    }
    catch(const Ice::Exception &e){ std::cout << e.what() <<  __FUNCTION__ << std::endl;};

    if( auto robot = G->get_node(robot_name); robot.has_value())
    {
        if( auto parent = G->get_parent_node(robot.value()); parent.has_value())
        {
            if (are_different(std::vector < float > {pose.x, pose.y, pose.rz},
                          std::vector < float > {last_state.x, last_state.y, last_state.rz},
                          std::vector < float > {1, 1, 0.05}))
            {
                auto edge = rt->get_edge_RT(parent.value(), robot->id()).value();
                Eigen::Vector2f converted_pos = (convertion_matrix * Eigen::Vector3f(pose.x, pose.y, 1.f)).head(2);
                G->modify_attrib_local<rt_rotation_euler_xyz_att>(edge, std::vector < float > {0.0, 0.0, pose.rz});
                G->modify_attrib_local<rt_translation_att>(edge, std::vector < float > {converted_pos.x(), converted_pos.y(), 0.0});
                // TODO: maybe it could be changed too
                G->modify_attrib_local<rt_translation_velocity_att>(edge, std::vector<float>{pose.vx, pose.vy, pose.vz});
                G->modify_attrib_local<rt_rotation_euler_xyz_velocity_att>(edge, std::vector<float>{pose.vrx, pose.vry, pose.vrz});
                // linear velocities are WRT world axes, so local speed has to be computed WRT to the robot's moving frame
                float side_velocity = -sin(pose.rz) * pose.vx + cos(pose.rz) * pose.vy;
                float adv_velocity = -cos(pose.rz) * pose.vx + sin(pose.rz) * pose.vy;
                G->insert_or_assign_edge(edge);
//                std::cout << "VELOCITIES: " << adv_velocity << " " << pose.vrz << std::endl;


//                G->add_or_modify_attrib_local<robot_local_linear_velocity_att>(robot.value(), std::vector<float>{adv_velocity, side_velocity, pose.rz});
//                G->add_or_modify_attrib_local<robot_ref_adv_speed_att>(robot.value(), -adv_velocity);
//                G->add_or_modify_attrib_local<robot_ref_rot_speed_att>(robot.value(), pose.vrz);
//
//                G->update_node(robot.value());
                last_state = pose;
            }
        }
        else  qWarning() << __FUNCTION__ << " No parent found for node " << QString::fromStdString(robot_name);
    }
    else    qWarning() << __FUNCTION__ << " No node " << QString::fromStdString(robot_name);
}
cv::Mat SpecificWorker::compute_camera_simple_frame()
{
    RoboCompCameraSimple::TImage cdata_camera_simple;
    cv::Mat camera_simple_frame;
    try
    {
        cdata_camera_simple = camerasimple_proxy->getImage();
        //this->focalx = cdata_camera_simple.focalx;
        //this->focaly = cdata_camera_simple.focaly;
        if( not cdata_camera_simple.image.empty())
        {
            if(cdata_camera_simple.compressed)
                camera_simple_frame = cv::imdecode(cdata_camera_simple.image, -1 );
            else
                camera_simple_frame = cv::Mat(cv::Size(cdata_camera_simple.width, cdata_camera_simple.height), CV_8UC3, &cdata_camera_simple.image[0], cv::Mat::AUTO_STEP);
            cv::cvtColor(camera_simple_frame, camera_simple_frame, cv::COLOR_BGR2RGB);
        }
    }
    catch (const Ice::Exception &e){ /*std::cout << e.what() <<  " In compute_camera_simple_frame" << std::endl;*/}
    return camera_simple_frame;
}
void SpecificWorker::update_camera_simple(std::string camera_name, const cv::Mat &v_image)
{

    if( auto node = G->get_node(camera_name); node.has_value())
    {
        std::vector<uint8_t> rgb; rgb.assign(v_image.data, v_image.data + v_image.total()*v_image.channels());
        G->add_or_modify_attrib_local<cam_rgb_att>(node.value(),  rgb);
        G->add_or_modify_attrib_local<cam_rgb_width_att>(node.value(), v_image.cols);
        G->add_or_modify_attrib_local<cam_rgb_height_att>(node.value(), v_image.rows);
        G->add_or_modify_attrib_local<cam_rgb_depth_att>(node.value(), v_image.depth());
        G->add_or_modify_attrib_local<cam_rgb_cameraID_att>(node.value(), 3);
        G->add_or_modify_attrib_local<cam_rgb_alivetime_att>(node.value(), static_cast<std::uint64_t>(0));  // CAMBIAR

        G->update_node(node.value());
    }
    else
        qWarning() << __FUNCTION__ << "Node camera_simple not found";
}

//////////////////// AUX ///////////////////////////////////////////////
bool SpecificWorker::are_different(const std::vector<float> &a, const std::vector<float> &b, const std::vector<float> &epsilon)
{
    for(auto &&[aa, bb, e] : iter::zip(a, b, epsilon))
        if (fabs(aa - bb) > e)
            return true;
    return false;
};
Eigen::Vector2f SpecificWorker::from_world_to_robot(const Eigen::Vector2f &p,
                                                    const RoboCompFullPoseEstimation::FullPoseEuler &r_state)
{
    Eigen::Matrix2f matrix;
    matrix << cos(r_state.rz) , -sin(r_state.rz) , sin(r_state.rz) , cos(r_state.rz);
    return (matrix.transpose() * (p - Eigen::Vector2f(r_state.x, r_state.y)));
}

//////////////////////////////// TEMP GRID ////////////////////////////////
Eigen::Matrix3f SpecificWorker::get_new_grid_matrix(Eigen::Vector3d robot_position, Eigen::Vector3d robot_rotation)
{
    // build the matrix to transform from robot to local_grid, knowing robot and grid pose in world
    Eigen::Matrix3f r2w;
    qInfo() << __FUNCTION__;
    r2w <<  cos((float)robot_rotation.z()), -sin((float)robot_rotation.z()), (float)robot_position.x(),
            sin((float)robot_rotation.z()) , cos((float)robot_rotation.z()), (float)robot_position.y(),
            0.f, 0.f, 1.f;
//    Eigen::Matrix2f w2g_2d_matrix;
//    w2g_2d_matrix <<  cos(grid_world_pose.get_ang()), sin(grid_world_pose.get_ang()),
//            -sin(grid_world_pose.get_ang()), cos(grid_world_pose.get_ang());
//    auto tr = w2g_2d_matrix * grid_world_pose.get_pos();
//    Eigen::Matrix3f w2g;
//    w2g << cos(grid_world_pose.get_ang()), sin(grid_world_pose.get_ang()), -tr.x(),
//            -sin(grid_world_pose.get_ang()), cos(grid_world_pose.get_ang()), -tr.y(),
//            0.f, 0.f, 1.f;
//    Eigen::Matrix3f r2g = w2g * r2w;  // from r to world and then from world to grid
    return r2w;
}

///////////////////////////////////////////////////////////////////
/// Asynchronous changes on G nodes from G signals
///////////////////////////////////////////////////////////////////
void SpecificWorker::add_or_assign_node_slot(const std::uint64_t id, const std::string &type)
{
    if (type == "grid")
    {
        qInfo() << __FUNCTION__;
        if(auto grid_node = G->get_node(id); grid_node.has_value())
        {
            qInfo() << __FUNCTION__;
            if(auto robot_node = G->get_node("robot"); robot_node.has_value())
            {
                qInfo() << __FUNCTION__;
                if(auto robot_pose = inner_eigen->transform(world_name, robot_name); robot_pose.has_value())
                {
                    qInfo() << __FUNCTION__;
//                    if(auto grid_rotation = inner_eigen->transform(world_name, robot_node.value().name()); grid_rotation.has_value())
//                    {
//                        qInfo() << __FUNCTION__;
                        if(auto robot_rotation_3d = inner_eigen->get_euler_xyz_angles(world_name, robot_name); robot_rotation_3d.has_value())
                        {
                            qInfo() << __FUNCTION__;
                            convertion_matrix = get_new_grid_matrix(robot_pose.value(), robot_rotation_3d.value());
                            qInfo() << __FUNCTION__;
                            std::cout << "////////////////////////// CONVERTION MATRIX //////////////////////////" << std::endl;
                            std::cout << convertion_matrix << std::endl;
                        }
//                    }
                }
            }
        }
    }

    if (type == "servo")
    {
        if (auto servo = G->get_node("servo"); servo.has_value())
        {
//            std::cout << "ENTERING IN SERVO" << std::endl;
            if(auto servo_send_pos = G->get_attrib_by_name<servo_ref_pos_att>(servo.value()); servo_send_pos.has_value())
            {
                float servo_pos = servo_send_pos.value();
                if(auto servo_send_speed = G->get_attrib_by_name<servo_ref_speed_att>(servo.value()); servo_send_speed.has_value())
                {
                    float servo_speed = servo_send_speed.value();
                    servo_pos_anterior = servo_pos;
                    servo_speed_anterior =  servo_speed;

                    try {
//                        std::cout << "SENDING POSITION" << std::endl;
                        RoboCompJointMotorSimple::MotorGoalPosition goal;
                        goal.maxSpeed = (float)servo_speed;
                        goal.position = (float)servo_pos;
                        this->jointmotorsimple_proxy->setPosition("eye_motor", goal);
                    }
                    catch (const RoboCompGenericBase::HardwareFailedException &re) {
//                        std::cout << __FUNCTION__ << "Exception setting base speed " << re << '\n';
//                        std::cout << __FUNCTION__ << "Exception setting base speed " << re << '\n';
                    }
                    catch (const Ice::Exception &e) {
                        //std::cout << e.what() << '\n';
                    }
                }
            }

        }
    }

    if (type == differentialrobot_type_name)   // pasar al SLOT the change attrib
    {
////        qInfo() << __FUNCTION__  << " Dentro " << id << QString::fromStdString(type);
//        if (auto robot = G->get_node(robot_name); robot.has_value())
//        {
//            // speed
//            auto ref_adv_speed = G->get_attrib_by_name<robot_ref_adv_speed_att>(robot.value());
//            auto ref_rot_speed = G->get_attrib_by_name<robot_ref_rot_speed_att>(robot.value());
////            qInfo() << __FUNCTION__ << ref_adv_speed.has_value() << ref_rot_speed.has_value();
//            if (ref_adv_speed.has_value() and ref_rot_speed.has_value())
//            {
//                //comprobar si la velocidad ha cambiado y el cambio es mayor de 10mm o algo así, entonces entra y tiene que salir estos mensajes
//                std::cout << __FUNCTION__  <<endl;
//                // Check de values are within robot's accepted range. Read them from config
//                //const float lowerA = -10, upperA = 10, lowerR = -10, upperR = 5, lowerS = -10, upperS = 10;
//                //std::clamp(ref_adv_speed.value(), lowerA, upperA);
//                float adv = ref_adv_speed.value();
//                float rot = ref_rot_speed.value();
//                //float inc = 10.0;
////                cout << __FUNCTION__ << "adv " << adv << " rot " << rot << endl;
//                if ( adv != av_anterior or rot != rot_anterior)
//                {
//                    std::cout<< "..................................."<<endl;
////                    std::cout << __FUNCTION__ << " " << ref_adv_speed.value() << " " << ref_rot_speed.value()
////                              << std::endl;
//                    av_anterior = adv;
//                    rot_anterior = rot;
//                    try {
//                        omnirobot_proxy->setSpeedBase(ref_adv_speed.value(), ref_rot_speed.value(), 0);
//                        std::cout << "VELOCIDADES: " << ref_adv_speed.value() << " " << ref_rot_speed.value() << std::endl;
//                    }
//                    catch (const RoboCompGenericBase::HardwareFailedException &re) {
////                        std::cout << __FUNCTION__ << "Exception setting base speed " << re << '\n';
////                        std::cout << __FUNCTION__ << "Exception setting base speed " << re << '\n';
//                    }
//                    catch (const Ice::Exception &e) {
//                        //std::cout << e.what() << '\n';
//                    }
//                }
//            }
//        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////


/**************************************/
// From the RoboCompBatteryStatus you can call this methods:
// this->batterystatus_proxy->getBatteryState(...)

/**************************************/
// From the RoboCompBatteryStatus you can use this types:
// RoboCompBatteryStatus::TBattery

/**************************************/
// From the RoboCompCameraRGBDSimple you can call this methods:
// this->camerargbdsimple_proxy->getAll(...)
// this->camerargbdsimple_proxy->getDepth(...)
// this->camerargbdsimple_proxy->getImage(...)

/**************************************/
// From the RoboCompCameraRGBDSimple you can use this types:
// RoboCompCameraRGBDSimple::TImage
// RoboCompCameraRGBDSimple::TDepth
// RoboCompCameraRGBDSimple::TRGBD

/**************************************/
// From the RoboCompCameraSimple you can call this methods:
// this->camerasimple_proxy->getImage(...)

/**************************************/
// From the RoboCompCameraSimple you can use this types:
// RoboCompCameraSimple::TImage

/**************************************/
// From the RoboCompCameraSimple you can call this methods:
// this->camerasimple1_proxy->getImage(...)

/**************************************/
// From the RoboCompCameraSimple you can use this types:
// RoboCompCameraSimple::TImage

/**************************************/
// From the RoboCompDifferentialRobot you can call this methods:
// this->differentialrobot_proxy->correctOdometer(...)
// this->differentialrobot_proxy->getBasePose(...)
// this->differentialrobot_proxy->getBaseState(...)
// this->differentialrobot_proxy->resetOdometer(...)
// this->differentialrobot_proxy->setOdometer(...)
// this->differentialrobot_proxy->setOdometerPose(...)
// this->differentialrobot_proxy->setSpeedBase(...)
// this->differentialrobot_proxy->stopBase(...)

/**************************************/
// From the RoboCompDifferentialRobot you can use this types:
// RoboCompDifferentialRobot::TMechParams

/**************************************/
// From the RoboCompFullPoseEstimation you can call this methods:
// this->fullposeestimation_proxy->getFullPoseEuler(...)
// this->fullposeestimation_proxy->getFullPoseMatrix(...)
// this->fullposeestimation_proxy->setInitialPose(...)

/**************************************/
// From the RoboCompFullPoseEstimation you can use this types:
// RoboCompFullPoseEstimation::FullPoseMatrix
// RoboCompFullPoseEstimation::FullPoseEuler

/**************************************/
// From the RoboCompGiraff you can call this methods:
// this->giraff_proxy->decTilt(...)
// this->giraff_proxy->getBotonesState(...)
// this->giraff_proxy->getTilt(...)
// this->giraff_proxy->incTilt(...)
// this->giraff_proxy->setTilt(...)

/**************************************/
// From the RoboCompGiraff you can use this types:
// RoboCompGiraff::Botones

/**************************************/
// From the RoboCompLaser you can call this methods:
// this->laser_proxy->getLaserAndBStateData(...)
// this->laser_proxy->getLaserConfData(...)
// this->laser_proxy->getLaserData(...)

/**************************************/
// From the RoboCompLaser you can use this types:
// RoboCompLaser::LaserConfData
// RoboCompLaser::TData

/**************************************/
// From the RoboCompRealSenseFaceID you can call this methods:
// this->realsensefaceid_proxy->authenticate(...)
// this->realsensefaceid_proxy->enroll(...)
// this->realsensefaceid_proxy->eraseAll(...)
// this->realsensefaceid_proxy->eraseUser(...)
// this->realsensefaceid_proxy->getQueryUsers(...)
// this->realsensefaceid_proxy->startPreview(...)
// this->realsensefaceid_proxy->stopPreview(...)

/**************************************/
// From the RoboCompRealSenseFaceID you can use this types:
// RoboCompRealSenseFaceID::UserData


// FOR REAL ROBOT
//    auto rgbd = camerargbdsimple_proxy->getImage("");
//    cv::Mat rgbd_frame (cv::Size(rgbd.height, rgbd.width), CV_8UC3, &rgbd.image[0]);
//    cv::Mat rgbd_frame_rotated (cv::Size(rgbd.width, rgbd.height), CV_8UC3);
//    // Rotation process
//    cv::Point2f src_center(rgbd_frame.cols/2.0F, rgbd_frame.rows/2.0F);
//    cv::Mat rot_matrix = getRotationMatrix2D(src_center, 90, 1.0);
//    warpAffine(rgbd_frame, rgbd_frame_rotated, rot_matrix, rgbd_frame.size());
//
//    cv::cvtColor(rgbd_frame_rotated, rgbd_frame_rotated, 4);

////////////////////////////////////////////////////////////////////////
int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, qApp, SLOT(quit()));
    return 0;
}