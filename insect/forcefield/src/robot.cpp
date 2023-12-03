//
// Created by pbustos on 11/11/22.
//

#include "robot.h"
#include <cppitertools/range.hpp>
#include <ranges>
#include <cppitertools/enumerate.hpp>

namespace rc
{
    void Robot::initialize(RoboCompOmniRobot::OmniRobotPrxPtr omnirobot_proxy_,
                           AbstractGraphicViewer *viewer_)
    {
        this->omnirobot_proxy = omnirobot_proxy_;
        this->viewer = viewer_;
    }
//    Eigen::Vector3f Robot::get_robot_target_coordinates()
//    {
//        if(get_pure_rotation() != 0)
//            return Eigen::Vector3f(0.f, 0.f, pure_rotation);
//        if(not has_target_flag)
//            return Eigen::Vector3f{0.f, 0.f, 0.f};
//        Eigen::Transform<float, 3, Eigen::Affine> tf = get_tf_cam_to_base();
//        Eigen::Vector3f target = tf * current_target.get_camera_coordinates();
//        target[2] = 0.f;  // dismiss pure rotation here
//        target = target.normalized() * (target.norm() - min_distance_to_target);  // set target coordinates 1m shorter than real target
//        return target;
//    }

    //////// CONTROL ///////////////
    void Robot::goto_target(const std::vector<Eigen::Vector2f> &current_line)
    {
        float side=0.f, adv=0.f, rot=0.f;
        if(get_pure_rotation() != 0 and not is_stopped)
        {
            side=0.f; adv=0.f; rot=pure_rotation;
        }
        else if(not has_target_flag)
        {
            side=0.f; adv=0.f; rot = 0.f;
        }
        else if(not is_stopped)
        {
            // reduce target distance by min_distance_to_target
            //robot_target_coordinates = robot_target_coordinates.normalized() * (robot_target_coordinates.norm() - min_distance_to_target);

            auto [side_, adv_, rot_] = dwa.update(current_target.get_robot_coordinates(),
                                               current_line,
                                               get_current_advance_speed(),
                                               get_current_side_speed(),
                                               get_current_rot_speed(),
                                               get_bumper(),
                                               viewer);
            side = side_; adv = adv_; rot = rot_;
        }

        //qInfo() << __FUNCTION__ << side <<  adv << rot;
        //    if(not repulsion.isZero())
        //    { side += repulsion.x(); adv += repulsion.y();}
        try
        {
            omnirobot_proxy->setSpeedBase(side, adv, rot);
            current_adv_speed = adv; current_side_speed = side; current_rot_speed = rot;
        }
        catch(const Ice::Exception &e){ std::cout << e.what() << "Error connecting to omnirobot" << std::endl; }
    }

    void Robot::stop()
    {
        is_stopped = true;
        pure_rotation = 0.f;
        qInfo() << __FUNCTION__ << "STOPPING the robot";
        try{ omnirobot_proxy->setSpeedBase(0.f, 0.f, 0.f);}
        catch(const Ice::Exception &e){ std::cout << e.what() << "Error connecting to omnirobot in STOP" << std::endl; }
    }
    void Robot::just_rotate(float rot)
    {
        is_stopped = false;
        pure_rotation =  std::clamp(rot, -max_rot_speed, max_rot_speed);
    }
    void Robot::set_current_target(const rc::PreObject &target)  // store in robot coordinates, so doors don't have to be changed
    {
        current_target = target;
        has_target_flag = true;
        is_stopped = false;
    }

    ///// SETTERS  ////
    void Robot::set_has_target(bool val)
    {
        has_target_flag = val;
    }
    bool Robot::has_target() const
    {
        return has_target_flag;
    }
    float Robot::get_current_advance_speed() const
    {
        return current_adv_speed;
    }
    float Robot::get_current_side_speed() const
    {
        return current_side_speed;
    }
    float Robot::get_current_rot_speed() const
    {
        return current_rot_speed;
    }
    float Robot::get_target_angle_in_frame() const
    {
        return atan2(current_target.x, current_target.y);
    }
    float Robot::get_current_pan_angle() const
    {
        return camera_pan_angle;
    }
    rc::PreObject Robot::get_current_target() const
    {
        return current_target;
    }
    void Robot::set_desired_distance_to_target(float dist)
    {
        min_distance_to_target = dist;
    }
    float Robot::get_distance_to_target()
    {
        return current_target.get_robot_coordinates().head(2).norm();  // without the Z
    }
    Eigen::Transform<float, 3, Eigen::Affine> Robot::get_tf_cam_to_base()
    {
        Eigen::Transform<float, 3, Eigen::Affine> tf = Eigen::Translation3f(Eigen::Vector3f{0.f, 0.f, top_camera_height}) *
                                                       Eigen::AngleAxisf(camera_tilt_angle, Eigen::Vector3f::UnitX()) *
                                                       Eigen::AngleAxisf(camera_pan_angle, Eigen::Vector3f::UnitZ());
//      for(const auto &a: axes)
//            if(a == "z")
//                tf.linear() = (Eigen::AngleAxisf(camera_tilt_angle, Eigen::Vector3f::UnitX()) *
//                               Eigen::AngleAxisf(camera_pan_angle, Eigen::Vector3f::UnitZ())).toRotationMatrix();

        return tf;
    }
    Eigen::Transform<float, 3, Eigen::Affine> Robot::get_tf_base_to_cam()
    {
        return get_tf_cam_to_base().inverse();
    }
    void Robot::print()
    {
        std::cout << "---- Robot Current State ----" << std::endl;
        std::cout << "  Advance speed: " << current_adv_speed << std::endl;
        std::cout << "  Rotation speed: " << current_rot_speed << std::endl;
        std::cout << "  Pan angle: " << camera_pan_angle << std::endl;

        if(has_target_flag)
        {
            std::cout << "  Target:" << std::endl;
            std::cout << "      Type: " << current_target.type << std::endl;
            std::cout << "      Cam coor: [" << current_target.get_robot_coordinates().x() << ", " << current_target.get_robot_coordinates().y() << "]" << std::endl;
            std::cout << "      Robot coor: [" << current_target.get_robot_coordinates().x() << ", "
                            << current_target.get_robot_coordinates().y() << "]" << std::endl;
            std::cout << "      Distance: " << get_distance_to_target() << std::endl;
        }
    }

    std::vector<std::tuple<float, float, bool>> Robot::create_bumper()
    {
        // create bumper
        // sample the robot's contour at equally separated 360 angular bins
        //  get rectangle internal angles
        float s1  = atan2(semi_height+security_offset, semi_width+security_offset);
        float s2 = atan2(semi_width+security_offset, semi_height+security_offset);
        int dsample = 6;

        // compute angle samples grouped by the rectangle's internal angles, starting from -PI and all the way to PI
        sector1 = Eigen::ArrayXf::LinSpaced(Eigen::Sequential, (int)qRadiansToDegrees(s1)/dsample, -M_PI, -M_PI+s1);
        sector2 = Eigen::ArrayXf::LinSpaced(Eigen::Sequential, (int)qRadiansToDegrees(2*s2)/dsample, -M_PI+s1, -s1);
        sector3 = Eigen::ArrayXf::LinSpaced(Eigen::Sequential, (int)qRadiansToDegrees(2*s1)/dsample, -s1, s1);
        sector4 = Eigen::ArrayXf::LinSpaced(Eigen::Sequential, (int)qRadiansToDegrees(2*s2)/dsample, s1, s1+2*s2);
        sector5 = Eigen::ArrayXf::LinSpaced(Eigen::Sequential, (int)qRadiansToDegrees(s1)/dsample, s1+2*s2, M_PI);

        // compute the distances to the border of the rectangle
        float semi_offset = security_offset/2;
        std::map<float, float> bumper_points_rotated;
        for(auto &&i : iter::range(sector1.rows()))
            bumper_points_rotated.insert(std::pair(sector1(i), fabs((semi_width+semi_offset)/cos(sector1(i)))));
        for(auto &&i : iter::range(sector2.rows()))
            bumper_points_rotated.insert(std::pair(sector2(i), fabs((semi_height+semi_offset)/sin(sector2(i)))));
        for(auto &&i : iter::range(sector3.rows()))
            bumper_points_rotated.insert(std::pair(sector3(i), fabs((semi_width+semi_offset)/cos(sector3(i)))));
        for(auto &&i : iter::range(sector4.rows()))
            bumper_points_rotated.insert(std::pair(sector4(i), fabs((semi_height+semi_offset)/sin(sector4(i)))));
        for(auto &&i : iter::range(sector5.rows()))
            bumper_points_rotated.insert(std::pair(sector5(i), fabs((semi_height+semi_offset)/cos(sector5(i)))));

        // the vector runs from -pi to p1 in N steps, but 0 points towards the X axis.
        // we need to add pi/2 to the first part and 3pi/2 to the rest, so it is "rotated"  ccw 90 degrees and zero is the robot's nose
        for(const auto &[k, v]:  bumper_points_rotated)
            if(k <= M_PI_2)
                bumper.emplace_back(std::make_tuple(k+M_PI_2, v, false));
            else
                bumper.emplace_back(std::make_tuple(k-3.f*M_PI_2, v, false));

        draw_bumper();
        return bumper;
    }
    void Robot::recompute_bumper(float robot_advance_speed)
    {
        //static std::map<float, float> initial_bumper(bumper);
        static std::vector<std::tuple<float, float, bool>> initial_bumper(bumper);
        // if speed == 0 -> dynamic_offset=extended_security_offset;
        // if speed == max_advance_speed -> dynamic_offset=semi_width
        extended_security_offset = fabs(robot_advance_speed*(semi_width-100)/max_advance_speed);
        for(auto &&[i, tup]: bumper | iter::enumerate)
        {
            auto &[ang, dist, _] = tup;
            dist = std::get<1>(initial_bumper[i]) + extended_security_offset;
        }
        draw_bumper();
    }

//    void Robot::add_camera(const Eigen::Transform<float, 3, Eigen::Affine> &tf_,
//                           const std::vector<std::string> &axes_,
//                           RoboCompJointMotorSimple::JointMotorSimplePrxPtr jointmotorsimple_proxy_)
//    {
//        this->tf = tf_;
//        this->axes = axes_;
//        this->jointmotorsimple_proxy = jointmotorsimple_proxy_;
//        std::ranges::sort(this->axes);  // to get x, y z
//
//        RoboCompJointMotorSimple::MotorState servo_state;
//        qInfo() << __FUNCTION__ << "Setting servo position to zero";
//        zero_servo();
////        while(true) // TODO: add time limit
////            try
////            {
////                servo_state = jointmotorsimple_proxy->getMotorState("camera_pan_joint");
////                if( fabs(servo_state.pos)  < 0.03)  break;
////                jointmotorsimple_proxy->setPosition("camera_pan_joint", RoboCompJointMotorSimple::MotorGoalPosition{0.f, 1.f});
////                qInfo() << __FUNCTION__ << "moving eye" << servo_state.pos;
////                usleep(100000);
////            }
////            catch(const Ice::Exception &e){ std::cout << e.what() << std::endl; return;}
//
//        update_joints();
//    }
//    void Robot::zero_servo()
//    {
//        RoboCompJointMotorSimple::MotorState servo_state;
//        qInfo() << __FUNCTION__ << "Setting servo position to zero"  << get_current_pan_angle() << get_current_pan_speed();
//        while(true) // TODO: add time limit
//            try
//            {
//                servo_state = jointmotorsimple_proxy->getMotorState("camera_pan_joint");
//                if( fabs(servo_state.pos)  < 0.03)
//                {
//                    jointmotorsimple_proxy->setVelocity("camera_pan_joint", RoboCompJointMotorSimple::MotorGoalVelocity{0.f, 0.f});
//                    break;
//                }
//                jointmotorsimple_proxy->setPosition("camera_pan_joint", RoboCompJointMotorSimple::MotorGoalPosition{0.f, 1.f});
//                usleep(10000);
//            }
//            catch(const Ice::Exception &e){ std::cout << e.what() << std::endl; return;}
//    }
//    void Robot::update_joints()
//    {
//        try
//        {
//            if(auto state = jointmotorsimple_proxy->getMotorState("camera_pan_joint"); not std::isnan(state.pos))
//            {
//                camera_pan_angle = state.pos;
//                camera_pan_speed = state.vel;
//            }
//            else
//            {
//                qWarning() << "NAN value in servo position";
//                return;
//            }
//        }
//        catch (const Ice::Exception &e)
//        {
//            std::cout << e.what() << " Warning: error connecting with jointmotorsimple" << std::endl;
//            return;
//        }
//    }
    void Robot::update_speed()
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::normal_distribution<float> dist(0, 10); // add normal noise to velocity measurements

        RoboCompGenericBase::TBaseState bState;
        try
        {
            omnirobot_proxy->getBaseState(bState);
            bState.advVz += dist(gen);
            bState.advVx += dist(gen);
            current_side_speed = bState.advVx;
            current_adv_speed = bState.advVz;
            current_rot_speed = bState.rotV;
            recompute_bumper(bState.advVz);
        }
        catch (const Ice::Exception &e)
        { std::cout << e.what() << " Error reading omnirobot_proxy::getBaseSpeed" << std::endl; }
    }
    std::vector<std::tuple<float, float, bool>> Robot::get_bumper()
    {
        return bumper;
    }
    Eigen::Vector2f Robot::compute_repulsion_forces(std::vector<Eigen::Vector2f> &floor_line)
    {
        static QGraphicsItem *item = nullptr;
        if(item != nullptr) viewer->scene.removeItem(item);
        delete item;

        // we want to detect any bumpers' circle intersecting with the distance-line
        QPolygonF poly;
        for(const auto &p: floor_line)
            if(not p.isZero())
                poly << QPointF(p.x(), p.y());

        std::vector<std::pair<float, float>> contacts;
        Eigen::Vector2f res = {0.f, 0.f};
        float offset_2 = (security_offset + extended_security_offset)/2;
        for(auto &[angle, dist, occ]: bumper)
        {
            float x = (dist+offset_2)*sin(angle); float y = (dist+offset_2)*cos(angle);
            if (not poly.containsPoint(QPointF(x, y), Qt::OddEvenFill))
            {
                //const float &dist = (ray/1000.f).norm();
                res -= Eigen::Vector2f {x,y}/dist;  //unit vectors
                occ = true;
            }
            else occ = false;
        }
        // item = viewer->scene.addPolygon(poly, QPen(QColor("green"),  10));
        Eigen::Vector2f final = res * security_offset*1.5;
        draw_repulsion(final);
        return final; // mm
    }

    void Robot::set_current_pan_speed(float  vel)
    {
        camera_pan_speed = vel;
    }
    float Robot::get_current_pan_speed() const
    {
        return camera_pan_speed;
    }

    /////////////// DRAW //////////////////////////////////////////////7
    void Robot::draw_repulsion(const Eigen::Vector2f &repulsion)
    {
        static QGraphicsItem *item = nullptr;
        if(item != nullptr) viewer->scene.removeItem(item);
        delete item;
        item = viewer->scene.addLine(0.f, 0.f, repulsion.x(), repulsion.y(),  QPen(QColor("green"), 15));
    }
    void Robot::draw_optical_ray()
    {
        // draws a line from the robot to the intersection point with the floor
        static std::vector<QGraphicsItem *> items;
        for(const auto &i: items)
            viewer->scene.removeItem(i);
        items.clear();

        // plane
        Eigen::Vector3d x1{0.0, 0.0, 0.0};
        Eigen::Vector3d x2{1000.f, 0.0, 0.0};
        Eigen::Vector3d x3{0.0, 1000.f, 0.0};
        Eigen::Hyperplane<double, 3> floor = Eigen::Hyperplane<double, 3>::Through(x1, x2, x3);
        // line
        Eigen::Transform<float, 3, Eigen::Affine> tf = get_tf_cam_to_base();
        Eigen::Vector3f x4{top_camera_x_offset, top_camera_y_offset, top_camera_height};
        Eigen::Vector3f x5 = tf * Eigen::Vector3f(0.f, 1000.f, 0.f);  // vector pointing to a point of the optic ray
        auto ray = Eigen::Hyperplane<double, 3>::Through(Eigen::Vector3d(0.0,0.0, top_camera_height), Eigen::Vector3d());
        // compute intersection according to https://mathworld.wolfram.com/Line-PlaneIntersection.html
        Eigen::Matrix4f numerator;
        numerator << 1.f, 1.f, 1.f, 1.f,
                x1.x(), x2.x(), x3.x(), x4.x(),
                x1.y(), x2.y(), x3.y(), x4.y(),
                x1.z(), x2.z(), x3.z(), x4.z();
        Eigen::Matrix4f denominator;
        denominator << 1.f, 1.f, 1.f, 0.f,
                x1.x(), x2.x(), x3.x(), x5.x()-x4.x(),
                x1.y(), x2.y(), x3.y(), x5.y()-x4.y(),
                x1.z(), x2.z(), x3.z(), x5.z()-x4.z();
        float k = numerator.determinant()/denominator.determinant();
        float x = x4.x() + (x5.x()-x4.x())*k;
        float y = x4.y() + (x5.y()-x4.y())*k;
        //float z = x4.z() + (x5.z()-x4.z())*k;
        items.push_back(viewer->scene.addLine(0, 0, -x, -y, QPen(QColor("darkgrey"), 20)));
    }
    void Robot::draw_bumper()
    {
        if(viewer == nullptr) return;

        static std::vector<QGraphicsItem *> items;
        for(auto &&i: items)
            viewer->scene.removeItem(i);
        items.clear();

        QPen pen_free(QColor("orange"), 5);
        QPen pen_contact(QColor("red"), 5);
        QPen pen; QBrush brush;
        QBrush brush_contact(QColor("red"));
        float offset = security_offset + extended_security_offset;
        if(viewer != nullptr)
            for(const auto &[ang, dist, occ] : bumper)
            {
                if(occ) { pen = pen_contact; brush = brush_contact;} else { pen = pen_free; brush = QBrush();}
                auto item = viewer->scene.addEllipse(-offset/2, -offset/2, offset, offset, pen, brush);
                item->setPos(dist*sin(ang), dist*cos(ang));
                items.push_back(item);
            }
    }
    float Robot::get_pure_rotation() const
    {
        return pure_rotation;
    }

} // rc

//
//try
//{
//if(float angle = joint_proxy->getMotorState("camera_pan_joint").pos; not std::isnan(angle))
//set_current_pan_angle(angle);
//else
//{
//qWarning() << "NAN value in servo position";
//return;
//}
//}
//catch (const Ice::Exception &e)
//{
//std::cout << e.what() << " Warning: error connecting with jointmotorsimple" << std::endl;
//return;
//}