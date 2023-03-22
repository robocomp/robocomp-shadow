/*
 *    Copyright (C) 2022 by YOUR NAME HERE
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

/**
* \brief Default constructor
*/
SpecificWorker::SpecificWorker(TuplePrx tprx, bool startup_check) : GenericWorker(tprx)
{
	this->startup_check_flag = startup_check;
	// Uncomment if there's too many debug messages
	// but it removes the possibility to see the messages
	// shown in the console with qDebug()
//	QLoggingCategory::setFilterRules("*.debug=false\n");
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
//	THE FOLLOWING IS JUST AN EXAMPLE
//	To use innerModelPath parameter you should uncomment specificmonitor.cpp readConfig method content
//	try
//	{
//		RoboCompCommonBehavior::Parameter par = params.at("InnerModelPath");
//		std::string innermodel_path = par.value;
//		innerModel = std::make_shared(innermodel_path);
//	}
//	catch(const std::exception &e) { qFatal("Error reading config params"); }


	try
	{
		agent_name = params["agent_name"].value;
		agent_id = stoi(params["agent_id"].value);

		tree_view = params["tree_view"].value == "true";
		graph_view = params["graph_view"].value == "true";
		qscene_2d_view = params["2d_view"].value == "true";
		osg_3d_view = params["3d_view"].value == "true";

		consts.max_adv_speed = stof(params.at("max_advance_speed").value);
        consts.max_rot_speed = stof(params.at("max_rotation_speed").value);
        consts.max_side_speed = stof(params.at("max_side_speed").value);
        consts.robot_length = stof(params.at("robot_length").value);
        consts.robot_width = stof(params.at("robot_width").value);
        consts.robot_radius = stof(params.at("robot_radius").value);
        consts.lateral_correction_gain = stof(params.at("lateral_correction_gain").value);
        consts.lateral_correction_for_side_velocity = stof(params.at("lateral_correction_for_side_velocity").value);
        consts.rotation_gain = std::stof(params.at("rotation_gain").value);
        consts.times_final_distance_to_target_before_zero_rotation = stof(params.at("times_final_distance_to_target_before_zero_rotation").value);
        consts.advance_gaussian_cut_x = stof(params.at("advance_gaussian_out_x").value);
        consts.advance_gaussian_cut_y = stof(params.at("advance_gaussian_out_y").value);
        consts.final_distance_to_target = stof(params.at("final_distance_to_target").value); // mm
	}
    catch (const std::exception &e)
    {
        std::cout << __FUNCTION__ << " Problem reading params" << std::endl;
        std::terminate();
    };

    //Creating robot limits
    robot_limits.push_back(std::make_tuple(Eigen::ParametrizedLine< float, 2 >::Through( upper_left_robot, upper_right_robot ), upper_left_robot, upper_right_robot));
    robot_limits.push_back(std::make_tuple(Eigen::ParametrizedLine< float, 2 >::Through( upper_left_robot, bottom_left_robot ),upper_left_robot, bottom_left_robot));
    robot_limits.push_back(std::make_tuple(Eigen::ParametrizedLine< float, 2 >::Through( bottom_left_robot, bottom_right_robot ), bottom_left_robot, bottom_right_robot));
    robot_limits.push_back(std::make_tuple(Eigen::ParametrizedLine< float, 2 >::Through( upper_right_robot, bottom_right_robot ), upper_right_robot, bottom_right_robot));
    qInfo() << "ROBOT LIMITS SIZE:" << robot_limits.size();
	return true;
}

void SpecificWorker::initialize(int period)
{
	std::cout << "Initialize worker" << std::endl;
	this->Period = period;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
		timer.start(Period);
		// create graph
		G = std::make_shared<DSR::DSRGraph>(0, agent_name, agent_id, ""); // Init nodes
		std::cout<< __FUNCTION__ << "Graph loaded" << std::endl;  

		//dsr update signals
		connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
		connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
		//connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_attrs_slot);
		connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
		//connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);
        inner_eigen = G->get_inner_eigen_api();
		// Graph viewer
		using opts = DSR::DSRViewer::view;
		int current_opts = 0;
		opts main = opts::none;
		if(tree_view)
		{
		    current_opts = current_opts | opts::tree;
		}
		if(graph_view)
		{
		    current_opts = current_opts | opts::graph;
		    main = opts::graph;
		}
		if(qscene_2d_view)
		{
		    current_opts = current_opts | opts::scene;
		}
		if(osg_3d_view)
		{
		    current_opts = current_opts | opts::osg;
		}
        graph_viewer = std::make_unique<DSR::DSRViewer>(this, G, current_opts, main);
        // 2D widget
        widget_2d = qobject_cast<DSR::QScene2dViewer *>(graph_viewer->get_widget(opts::scene));
        if (widget_2d != nullptr)
        {
            widget_2d->set_draw_laser(true);
        }
        c_start = std::clock();
		this->Period = period;
		timer.start(Period);
        std::vector<QGraphicsItem*> robot_limit_draw;
        for ( const auto &r_limit : robot_limits){

            auto limit_draw = widget_2d->scene.addLine(std::get<1>(r_limit).x(), std::get<1>(r_limit).y(), std::get<2>(r_limit).x(),std::get<2>(r_limit).y(), QPen(QColor("yellow"), 20));
            limit_draw->setZValue(20);
            robot_limit_draw.push_back(limit_draw);

        }
        rt = G->get_rt_api();

//        hide();
	}
}

void SpecificWorker::compute()
{
//    qInfo() << "JOYSTICK SPEEDS (ADV, SIDE; ROT):" << std::get<0>(joystick_speeds) << std::get<1>(joystick_speeds) << std::get<2>(joystick_speeds);
    std::clock_t c_act = std::clock();
//    std::cout << "TIME PASSED: " << 10.0 *(c_act - c_start)/ CLOCKS_PER_SEC << std::endl;
    if(10.0 *(c_act - c_start)/ CLOCKS_PER_SEC > 1)
    {
        joystick_speeds = std::make_tuple(0.0, 0.0, 0.0);
    }

    // Read laser data
    auto[laser_data, laser_data_dec] = read_laser();
    auto laser_gain = 96;
    // Compute force field coefficients
    auto force_vector = compute_laser_forces(laser_data_dec);
    // Get path from buffer
    if (auto path_o = path_buffer.try_get(); path_o.has_value()) // NEW PATH!
    {
        path.clear();
        path = path_o.value();
    }
    // if there is a path in G
    if(auto node_path = G->get_node(current_path_name); node_path.has_value())
    {
        set_target_pose(node_path.value());
        if(target.get_pos().x() == 0 && target.get_pos().y() == 0) return;
        else
        {

            set_robot_poses();
            // Get speeds to the next point
            auto speeds = update();
            auto combined_speeds = combine_forces_and_speed(speeds, force_vector, laser_gain);
            // Set speed in graph
            auto[adv, side, rot] =  send_command_to_robot(combined_speeds);
            // Combine them
            return;

        }
    }
    else // stop controlling
    {
//        qDebug() << __FUNCTION__ << "No path_node found in G. Stopping the robot";
    }
    auto combined_speeds = combine_forces_and_speed(joystick_speeds, force_vector, laser_gain);
    // Set speed in graph
    auto[adv, side, rot] =  send_command_to_robot(combined_speeds);
//    print_current_state(adv, side, rot);
}

///////////////////// POINTS /////////////////////
void SpecificWorker::set_target_pose(DSR::Node node_path)
{
    Eigen::Vector2f zero_point = Eigen::Vector2f{0.0, 0.0};
    if(auto x_point = G->get_attrib_by_name<path_target_x_att>(node_path); x_point.has_value())
    {
        if(auto y_point = G->get_attrib_by_name<path_target_y_att>(node_path); y_point.has_value())
        {
            target.set_pos(Eigen::Vector2f{x_point.value(), y_point.value()});
        }
        else
            target.set_pos(zero_point);
    }
    else
        target.set_pos(zero_point);
}
void SpecificWorker::set_robot_poses()
{
    auto nose_3d = inner_eigen->transform(world_name, Mat::Vector3d(0, 500, 0), robot_name).value();
    auto robot_pose_3d = inner_eigen->transform(world_name, robot_name).value();
    qInfo() << "ROBOT POSE:" << robot_pose_3d.x() << robot_pose_3d.y();
    robot_nose.set_pos(Eigen::Vector2f(nose_3d.x(), nose_3d.y()));
    robot.set_pos(Eigen::Vector2f(robot_pose_3d.x(), robot_pose_3d.y()));
}
///////////////////// PATH /////////////////////
float SpecificWorker::dist_along_path(const std::vector<Eigen::Vector2f> &path)
{
    float len = 0.0;
    for(auto &&p : iter::sliding_window(path, 2))
        len += (p[1]-p[0]).norm();
    return len;
};
///////////////////// SPEEDS /////////////////////
std::tuple<float, float, float> SpecificWorker::update()
{
    // now y is forward direction and x is pointing rightwards
    float advVel = 0.f, sideVel = 0.f, rotVel = 0.f;
    // Compute euclidean distance to target
//    float euc_dist_to_target = (robot.get_pos() - target).norm();
//    float euc_dist_to_target = sqrt(pow(target.get_pos().x(), 2) + pow(target.get_pos().y(), 2));
    float euc_dist_to_target = target.get_pos().norm();
    float angle = atan2(target.get_pos().x(), target.get_pos().y());
    // qInfo() << "ANGLE:" << angle;
    // rot speed gain
    rotVel = consts.rotation_gain * angle;
    // limit angular  values to physicapathl limits
    // qInfo() << "rotVel:" << rotVel;
    rotVel = std::clamp(rotVel, -consts.max_rot_speed, consts.max_rot_speed);
    // qInfo() << "euc_dist_to_target:" << euc_dist_to_target;
    if(auto listened_person = G->get_node("listened_person"); listened_person.has_value())
        if(auto robot_node = G->get_node(robot_name); robot_node.has_value())
            if(auto grid_node = G->get_node(grid_type_name); grid_node.has_value())
                if(auto looking_for_edge = G->get_edge(robot_node.value().id(), listened_person.value().id(), looking_for_type_name); looking_for_edge.has_value())
                {
                    auto edge = rt->get_edge_RT(grid_node.value(), robot_node.value().id()).value();
                    
                    if(auto rotation_values = G->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge); rotation_values.has_value())
                    {
                        qInfo() << "ROTATION DIFFERENCE:" << abs(abs(angle) -abs(rotation_values.value().get()[2]));
            		//int direction_sign = (abs(angle) -abs(rotation_values.value().get()[2]))/abs(abs(angle) -abs(rotation_values.value().get()[2]));
            		int direction_sign = angle / abs(angle);
                        if(abs(abs(angle) -abs(rotation_values.value().get()[2])) < 10 * M_PI / 180)
                        {
                            
                            if(auto path_d = G->get_node(current_path_name); path_d.has_value())
                                G->delete_node(path_d.value().id());
                            qInfo() << "MACHACO 0 1";
                            return std::make_tuple(0,0,0);  //adv, side, rot
                        }
                        else
                        {
                            rotVel = 0.5*direction_sign;
                            qInfo() << "ROT VEL" << rotVel;
                            return std::make_tuple(0,0,0.6*direction_sign);
                        }
                    }
                }
    if(euc_dist_to_target < consts.final_distance_to_target)
    {
        if(abs(angle) > 0.3)
        {
            qInfo() << "MACHACO 0 2";
            return std::make_tuple(0,0,rotVel);  //adv, side, rot
        }
        qInfo() << __FUNCTION__ << " -------------- Target achieved -----------------";
        advVel = 0;  sideVel= 0; rotVel = 0;
        return std::make_tuple(0,0,0);  //adv, side, rot
    }

    // cancel final rotation

//    if(euc_dist_to_target < consts.times_final_distance_to_target_before_zero_rotation * consts.final_distance_to_target)
//        rotVel = 0.f;

    /// Compute advance speed
    advVel = std::min(consts.max_adv_speed * exponentialFunction(rotVel, consts.advance_gaussian_cut_x, consts.advance_gaussian_cut_y, 0),
                      target.get_pos().y());
    if(abs(target.get_pos().x()) < 500)
        sideVel = 0;
    else
    {
        sideVel =std::min(consts.max_side_speed * exponentialFunction(rotVel, consts.advance_gaussian_cut_x, consts.advance_gaussian_cut_y, 0),
                          target.get_pos().x());
        sideVel = std::clamp(sideVel, -consts.max_side_speed, consts.max_side_speed);
    }
    if (target_draw_speed != nullptr) delete target_draw_speed;
    target_draw_speed = widget_2d->scene.addLine(0, 0, sideVel,advVel, QPen(QColor("blue"), 40));
    target_draw_speed->setZValue(15);

//    qInfo() << __FUNCTION__ << "VELOCIDAD LATERAL QUE SALE:" <<  sideVel;
//    return std::make_tuple(0, 0, 0);
    return std::make_tuple(advVel, sideVel, rotVel);
}
std::tuple<float, float, float> SpecificWorker::send_command_to_robot(const std::tuple<float, float, float> &speeds){
    auto &[adv_, side_, rot_] = speeds;

    if(auto robot_node = G->get_node(robot_name); robot_node.has_value())
    {
       if(only_rotation)
       {
            G->add_or_modify_attrib_local<robot_ref_adv_speed_att>(robot_node.value(), (float) 0);
            G->add_or_modify_attrib_local<robot_ref_side_speed_att>(robot_node.value(), (float) 0);
       }
           
       else
       {
            G->add_or_modify_attrib_local<robot_ref_side_speed_att>(robot_node.value(), (float) (side_));
            G->add_or_modify_attrib_local<robot_ref_adv_speed_att>(robot_node.value(), (float) (adv_));
       }
           
//        G->add_or_modify_attrib_local<robot_ref_rot_speed_att>(robot_node.value(), (float) (lc_speed_coefficient * rot_));
//        G->add_or_modify_attrib_local<robot_ref_side_speed_att>(robot_node.value(), (float) (lc_speed_coefficient * side_));
        
        G->add_or_modify_attrib_local<robot_ref_rot_speed_att>(robot_node.value(), (float) (rot_));
        G->update_node(robot_node.value());
    }

    else qWarning() << __FUNCTION__ << "No robot node found";
    return std::make_tuple(adv_, side_, rot_);
}
std::tuple<float, float, float> SpecificWorker::combine_forces_and_speed(const std::tuple<float, float, float> &speeds, Eigen::Vector2f force_vector, float gain)
{
//    qInfo() << "JOYSTICK SPEED:" << std::get<0>(speeds) << std::get<1>(speeds) << std::get<2>(speeds);
//    qInfo() << "FORCE SPEED:" << force_vector.y()*force_gain << force_vector.x()*force_gain;
    auto modifyed_adv = std::get<0>(speeds) + std::clamp(force_vector.y()*force_gain, -consts.max_adv_speed, consts.max_adv_speed); auto modifyed_side = std::get<1>(speeds) + std::clamp(force_vector.x()*force_gain, -consts.max_adv_speed, consts.max_adv_speed);
    auto combined_speeds = std::make_tuple(std::clamp(modifyed_adv, -consts.max_adv_speed, consts.max_adv_speed),std::clamp(modifyed_side,-consts.max_side_speed, consts.max_side_speed),std::get<2>(speeds));
//    qInfo() << "COMBINED SPEED:" << std::get<0>(combined_speeds) << std::get<1>(combined_speeds) << std::get<2>(combined_speeds);
    if (target_draw_comb != nullptr) delete target_draw_comb;
    target_draw_comb = widget_2d->scene.addLine(0, 0, std::clamp(modifyed_side,-consts.max_side_speed, consts.max_side_speed), std::clamp(modifyed_adv, -consts.max_adv_speed, consts.max_adv_speed), QPen(QColor("green"), 40));
    target_draw_comb->setZValue(20);
    return combined_speeds;
}


///////////////////// LASER /////////////////////

//Eigen::Vector2f SpecificWorker::compute_laser_forces(const RoboCompLaser::TLaserData &laser_data)
//{
//    static std::vector<QGraphicsItem*> laser_points_draw;
////    int local_force_distance = force_distance;
////    auto res = (std::accumulate(laser_data.begin(), laser_data.end(), Eigen::Vector2f{0.f, 0.f},
////                                [local_force_distance](auto res, auto b)->Eigen::Vector2f{ Eigen::Vector2f p = {b.dist*sin(b.angle), b.dist*cos(b.angle)}; if (p.norm() < local_force_distance && p.norm() > 0) {return res - p.normalized()/p.norm();} else {return res;}}));
////    RoboCompLaser::TData min; min.dist=5000;
////    for (const auto &laser_point : laser_data)
////        if (laser_point.dist < min.dist)
////            min = laser_point;
////    if (min.dist < 600)
////    {
////        res =-Eigen::Vector2f{min.dist*sin(min.angle), min.dist*cos(min.angle)};
////        res = res.normalized()/(res.norm()/1000);
////    }
////    else
////        res={0.0,0.0};
////    std::cout<< "MIN: " << min.dist << " " << min.angle <<std::endl;
////    std::cout << "FORCE VECTOR: " << res<< std::endl;
//
//    //Recorrer laser
//    //Crear linea (p_laser,centro_robot)
//    //Ver intersecciones (seleccionar min)
//    //crear fuerza (p_laser,p_interserccion)
//    for(const auto &i : laser_points_draw)
//        widget_2d->scene.removeItem(i);
//    laser_points_draw.clear();
//
//    Eigen::Vector2f total_force{0.0, 0.0};
//    QRectF robot_polygon = QRectF(-consts.robot_width/2, -consts.robot_length/2, consts.robot_width, consts.robot_length);
//    auto inside =[robot_polygon](const Eigen::Vector2f &p, const std::tuple<Eigen::ParametrizedLine< float, 2 >, Eigen::Vector2f, Eigen::Vector2f> &line)
//            {
//
//        QPointF point (p.x(), p.y());
//        qInfo() << "POINT:" << p.x() << p.y() << "INSIDE:" << robot_polygon.contains(point);
//        return robot_polygon.contains(point);
////                auto [l, p1, p2] = line;
//////                return (p.x() >= p1.x() and p.y() >= p1.y() and p2.x() >= p.x() and p2.y() >= p.y()) or
//////                        (p.x() <= p1.x() and p.y() <= p1.y() and p2.x() <= p.x() and p2.y() <= p.y());
////                return (p.x() >= p1.x() and p.y() >= p1.y() and p2.x() >= p.x() and p2.y() >= p.y()) or
////                       (p.x() <= p1.x() and p.y() <= p1.y() and p2.x() <= p.x() and p2.y() <= p.y());
//            };
//    for (const auto &p_laser: laser_data)
//    {
//
//        Eigen::Vector2f point {p_laser.dist*sin(p_laser.angle), p_laser.dist*cos(p_laser.angle)};
//        if(point.norm()<force_distance && point.norm() > 200)
//        {
//            auto laser_points_draw_item = widget_2d->scene.addEllipse(- 100, - 100, 200,
//                                                            200, QPen(QColor("magenta")), QBrush(QColor("magenta")));
//            laser_points_draw_item->setPos(point.x(), point.y());
//            laser_points_draw_item->setZValue(20);
//            laser_points_draw.push_back(laser_points_draw_item);
//
//            Eigen::Vector2f nearest{999999,999999};
//            Eigen::Vector2f center {0.0 , 0.0};
//            Eigen::Hyperplane<float,2> point_plane = Eigen::Hyperplane< float, 2 >::Through( center, point );
//            Eigen::Vector2f intersection_point = get_intersection_point(p_laser);
////            if(point.x() == 0 or point.y() == 0)
////            {
////                if(point.x() == 0)
//////                {
//////                    if(point.y() < abs(consts.robot_length/2))
//////                }
////                    intersection_point = {0.0, point.y() - consts.robot_length/2 * (point.y()/abs(point.y()))};
////
////                if(point.y() == 0)
////                    intersection_point = {point.x() - consts.robot_width/2 * (point.x()/abs(point.x())), 0.0};
////            }
////            else
////            {
////                auto closest_point_to_robot = std::ranges::min_element(robot_limits,[point, point_plane, this, &inside](const auto &line_data1, const auto &line_data2){
////                    auto [l1, p1_a, p1_b] = line_data1;
////                    auto [l2, p2_a, p2_b] = line_data2;
////                    Eigen::Vector2f p1 = l1.intersectionPoint(point_plane); Eigen::Vector2f p2 = l2.intersectionPoint(point_plane);
////
//////                    std::cout << "P1: " << p1 << std::endl;
//////                    std::cout << "P2: " << p2 << std::endl;
//////                    std::cout << "P1 norm: " << p1.norm() << " " << "P2 norm: " << p2.norm() << std::endl;
////                    bool i1 = inside(p1, line_data1); bool i2 = inside(p2, line_data2);
////                    if(i1 and i2) return (point - p1).norm()  < (point - p2).norm();
////                    else if(i1) return i1;
////                    else return i2;
////                });
////                auto closest_point_to_robot = std::ranges::min_element(robot_limits,[point, point_plane, this, &inside](const auto &line_data1, const auto &line_data2){
////                    auto [l1, p1_a, p1_b] = line_data1;
////                    auto [l2, p2_a, p2_b] = line_data2;
////                    Eigen::Vector2f p1 = l1.intersectionPoint(point_plane); Eigen::Vector2f p2 = l2.intersectionPoint(point_plane);
////
//////                    std::cout << "P1: " << p1 << std::endl;
//////                    std::cout << "P2: " << p2 << std::endl;
//////                    std::cout << "P1 norm: " << p1.norm() << " " << "P2 norm: " << p2.norm() << std::endl;
////                    bool i1 = inside(p1, line_data1); bool i2 = inside(p2, line_data2);
////                    if(i1 and i2) return (point - p1).norm()  < (point - p2).norm();
////                    else if(i1) return i1;
////                    else return i2;
////                });
////                if(closest_point_to_robot == robot_limits.end()) continue;
////                intersection_point = std::get<0>(*closest_point_to_robot).intersectionPoint(point_plane);
////            }
//            Eigen::Vector2f act_force = intersection_point- point;
//
////            if(act_force.norm() > 1000)
////            {
////            qInfo() << "";
////            qInfo() << "LASER POINT:" << point.x() << point.y();
////            qInfo()<< "ACT FORCE:" << act_force.x() << act_force.y();
//            qInfo()<< "INTERSECTION POINT:" << intersection_point.x() << intersection_point.y();
////            qInfo() << "";
////            }
//
//            total_force += act_force.normalized()/(act_force.norm()/force_distance);
//        }
////        qInfo() << "TOTAL FORCE:" << total_force.x() << total_force.y();
//    }
//
//    auto target_draw_line = widget_2d->scene.addLine(0, 0, total_force.x(),total_force.y(), QPen(QColor("yellow"), 40));
//    target_draw_line->setZValue(20);
//    laser_points_draw.push_back(target_draw_line);
//    return total_force;
//}

Eigen::Vector2f SpecificWorker::compute_laser_forces(const RoboCompLaser::TLaserData &laser_data)
{
    static std::vector<QGraphicsItem*> laser_points_draw;
    for(const auto &i : laser_points_draw)
        widget_2d->scene.removeItem(i);
    laser_points_draw.clear();

    Eigen::Vector2f total_force{0.0, 0.0};

    for (const auto &p_laser: laser_data)
    {

        Eigen::Vector2f point {p_laser.dist*sin(p_laser.angle), p_laser.dist*cos(p_laser.angle)};
        auto dec_laser_points_draw_item = widget_2d->scene.addEllipse(- 50, - 50, 100,
                                                                  100, QPen(QColor("red")), QBrush(QColor("red")));
        dec_laser_points_draw_item->setPos(point.x(), point.y());
        dec_laser_points_draw_item->setZValue(20);
        laser_points_draw.push_back(dec_laser_points_draw_item);

//        qInfo()<< "POINT:" << point.x() << point.y() << point.norm();
        if(point.norm()<force_distance)
        {
            auto laser_points_draw_item = widget_2d->scene.addEllipse(- 100, - 100, 200,
                                                                      200, QPen(QColor("magenta")), QBrush(QColor("magenta")));
            laser_points_draw_item->setPos(point.x(), point.y());
            laser_points_draw_item->setZValue(20);
            laser_points_draw.push_back(laser_points_draw_item);

            Eigen::Vector2f intersection_point = get_intersection_point(point);
            QRectF robot_polygon = QRectF(-consts.robot_width/2, -consts.robot_length/2, consts.robot_width, consts.robot_length);
            Eigen::Vector2f act_force = intersection_point - point;
            if(robot_polygon.contains(QPointF(point.x(), point.y())))
                act_force = -act_force;
//            qInfo()<< "INTERSECTION POINT:" << intersection_point.x() << intersection_point.y();

            // if(cos(atan2(point.x(), point.y())) > 0)
            //     act_force.y() = cos(atan2(point.x(), point.y())) * act_force.y(); 
            total_force += act_force.normalized()/(act_force.norm()/force_distance);

        }
        if(point.norm()>force_distance)
        {
            get_predicted_force_vector(point);
        }
//        qInfo() << "TOTAL FORCE:" << total_force.x() << total_force.y();
    }
    auto target_draw_line = widget_2d->scene.addLine(0, 0, total_force.x(),total_force.y(), QPen(QColor("yellow"), 40));
    target_draw_line->setZValue(20);
    laser_points_draw.push_back(target_draw_line);
    return total_force;
}

Eigen::Vector2f SpecificWorker::get_predicted_force_vector(Eigen::Vector2f laser_point)
{
    Eigen::ParametrizedLine< float, 2 > from_robot_to_target = Eigen::ParametrizedLine< float, 2 >::Through( Eigen::Vector2f{0.0, 0.0}, target.get_pos());
    auto projected_point = from_robot_to_target.projection(laser_point);
//    qInfo() << "PROJECTED POINT:" << projected_point.x() << projected_point.y();
    auto projected_to_laser_point_distance = (projected_point - laser_point).norm();
//    qInfo() << "PROPJECTION DISTANCE:" << projected_to_laser_point_distance;
    return projected_point;

}

Eigen::Vector2f SpecificWorker::get_intersection_point(Eigen::Vector2f laser_point)
{
    QRectF robot_polygon = QRectF(-consts.robot_width/2, -consts.robot_length/2, consts.robot_width, consts.robot_length);
//    QPolygonF robot_polygon = QPolygonF(robot_rect);
    Eigen::Hyperplane<float, 2> point_plane = Eigen::Hyperplane<float, 2>::Through(Eigen::Vector2f{0.0 , 0.0}, laser_point);
    Eigen::Vector2f intersection_point{99999, 99999};

    Eigen::Vector2f inter_point;
    if (laser_point.x() == 0 or laser_point.y() == 0)
    {
        if (laser_point.x() == 0)
            intersection_point = {0.0, laser_point.y() - consts.robot_length / 2 * (laser_point.y() / abs(laser_point.y()))};
        if (laser_point.y() == 0)
            intersection_point = {laser_point.x() - consts.robot_width / 2 * (laser_point.x() / abs(laser_point.x())), 0.0};
    }
    else
    {
        for (const auto &p_line: robot_limits)
        {
            inter_point = std::get<0>(p_line).intersectionPoint(point_plane);
            QPointF point(inter_point.x(), inter_point.y());
            if(robot_polygon.contains(point) and ((inter_point - laser_point).norm() < (intersection_point- laser_point).norm())) {
                intersection_point = inter_point;
            }
        }
    }
    return intersection_point;
}

std::tuple<RoboCompLaser::TLaserData, RoboCompLaser::TLaserData> SpecificWorker::read_laser()
{
    RoboCompLaser::TLaserData ldata;
    RoboCompLaser::TLaserData ldata_dec;
    int dec_factor = 120;
    QRectF robot_polygon = QRectF(-consts.robot_width/2, -consts.robot_length/2, consts.robot_width, consts.robot_length);

    try
    {
        if(auto laser_node = G->get_node("laser"); laser_node.has_value())
        {
            if (auto laser_dist = G->get_attrib_by_name<laser_dists_att>(laser_node.value()); laser_dist.has_value() && laser_dist.value().get().size() > 0)
            {
                auto dists = laser_dist.value().get();
                int dec_window = dists.size() / dec_factor;
//                qInfo() << "tamaños " << dec_window << dists.size() << dec_factor;
                if (auto laser_angle = G->get_attrib_by_name<laser_angles_att>(laser_node.value()); laser_angle.has_value() && laser_angle.value().get().size() > 0)
                {
                    auto angles = laser_angle.value().get();
                    // Build raw polygon
                    for (std::size_t i = 0; i < dists.size(); ++i)
                    {
                        RoboCompLaser::TData act_data;
                        act_data.dist = dists[i]; act_data.angle = angles[i];
//                        poly_robot << QPointF(act_data.dist * sin(act_data.angle), act_data.dist * cos(act_data.angle));

                        ldata.push_back(act_data);
                    }
                    int chunck = 0;

//                    if (laser_draw_min != nullptr) delete laser_draw_min;
                    for (auto&& laser_sec : iter::chunked(ldata,dec_factor))
                    {
                        RoboCompLaser::TData min_laser_point{.dist = 9999};
                        for (int i = 0; i < laser_sec.size(); i++)
                        {
                            if(laser_sec[i].dist < min_laser_point.dist)
                                if(i > 0 and i < laser_sec.size() - 1)
                                    if(abs(laser_sec[i-1].dist -laser_sec[i].dist) < 150 and abs(laser_sec[i+1].dist -laser_sec[i].dist) < 150)
                                        min_laser_point = laser_sec[i];
                        }
//                        if(min_laser_point.dist < 2000 && min_laser_point.dist > 100)
//                        {
////                            qInfo() << "MIN LASER: " << min_laser_point.dist << min_laser_point.angle;
//
////                            laser_draw_min = widget_2d->scene.addLine(0, 0, min_laser_point.dist * sin(min_laser_point.angle), min_laser_point.dist * cos(min_laser_point.angle), QPen(QColor("orange"), 40));
////                            laser_draw_min->setZValue(20);
//                        }
//                        if(!robot_polygon.contains(QPointF(min_laser_point.dist*sin(min_laser_point.angle), min_laser_point.dist*cos(min_laser_point.angle))))
                            ldata_dec.push_back(min_laser_point);
                    }
                    return std::make_tuple(ldata, ldata_dec);
                }
            }
        }
    }
    catch(const Ice::Exception &e){ std::cout << e.what() << std::endl;}
    return std::make_tuple(ldata, ldata_dec);
}
void SpecificWorker::change_force_parameters(string button_id)
{
    if(button_id == "increase_force")
        if(force_distance < 2 * consts.robot_length)
            force_distance += 10;
    if(button_id == "decrease_force")
        if(force_distance > 100)
            force_distance -= 10;
    if(button_id == "increase_gain")
        if(force_gain < 200)
            force_gain += 1;
    if(button_id == "decrease_gain")
        if(force_gain > 0)
            force_gain -= 1;
    qInfo() << "FORCE DISTANCE:" << force_distance;
    qInfo() << "FORCE GAIN:" << force_gain;
}
///////////////////// SLOTS /////////////////////
void SpecificWorker::modify_node_slot(std::uint64_t id, const std::string &type)
{
    if (type == path_to_target_type_name)
    {
        if( auto node = G->get_node(id); node.has_value())
        {
            auto x_values = G->get_attrib_by_name<path_x_values_att>(node.value());
            auto y_values = G->get_attrib_by_name<path_y_values_att>(node.value());
            if(x_values.has_value() and y_values.has_value())
            {
                auto x = x_values.value().get();
                auto y = y_values.value().get();
                std::vector<Eigen::Vector2f> path; path.reserve(x.size());
                for (auto &&[x, y] : iter::zip(x, y))
                {
                    path.push_back(Eigen::Vector2f(x, y));
                }
                path_buffer.put(std::move(path));
            }
        }
    }
}

void SpecificWorker::modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type)
{
    if(type == interacting_type_name or type == talking_type_name) only_rotation = true;
    if(type == following_action_type_name)
    {
        only_rotation = false;

    }
     
}

void SpecificWorker::del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag)
{
    if(edge_tag == interacting_type_name)
    {
        qInfo() << "#################################################";
        qInfo() << "#################################################";
        qInfo() << "#################################################";
        qInfo() << "#################################################";
        qInfo() << "#################################################";
        qInfo() << "#################################################";
        qInfo() << "#################################################";
        qInfo() << "#################################################";
        only_rotation = false;
    } 
    if(edge_tag == following_action_type_name)
    {
        send_command_to_robot(std::make_tuple(0,0,0));
    } 
}

///////////////////// OTHER /////////////////////
// compute max de gauss(value) where gauss(x)=y  y min
float SpecificWorker::exponentialFunction(float value, float xValue, float yValue, float min)
{
    if (yValue <= 0)
        return 1.f;
    float landa = -fabs(xValue) / log(yValue);
    float res = exp(-fabs(value) / landa);
    return std::max(res, min);
}
void SpecificWorker::print_current_state(float adv, float side, float rot)
{
    std::cout << "---------------------------" << std::endl;
    std::cout << "Target position: " << std::endl;
    std::cout << "\t " << target.get_pos().x() << ", " << target.get_pos().y() << std::endl;
    std::cout << "Ref speeds:  " << std::endl;
    std::cout << "\t Advance-> " << adv << std::endl;
    std::cout << "\t Side -> " << side << std::endl;
    std::cout << "\t Rotate -> " << rot << std::endl;
}
void SpecificWorker::JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data)
{
    std::get<2>(joystick_speeds) = data.axes[0].value*consts.max_rot_speed;
    std::get<0>(joystick_speeds) = data.axes[1].value*consts.max_adv_speed;
    std::get<1>(joystick_speeds) = data.axes[3].value*consts.max_side_speed;
    for(const auto &button : data.buttons)
        if(button.step == 1)
            change_force_parameters(button.name);

    c_start = std::clock();
}
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}




