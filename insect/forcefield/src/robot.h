//
// Created by pbustos on 11/11/22.
//

#ifndef FORCEFIELD_ROBOT_H
#define FORCEFIELD_ROBOT_H

#include <Eigen/Dense>
#include <YoloObjects.h>
#include <QtCore>
#include <QColor>
#include <abstract_graphic_viewer/abstract_graphic_viewer.h>
#include "camera.h"
#include <OmniRobot.h>
#include <random>
#include <dynamic_window.h>
#include "preobject.h"

namespace rc
{
    class Robot
    {
        public:
                Robot() = default;

                // configuration
                void initialize(RoboCompOmniRobot::OmniRobotPrxPtr omnirobot_proxy_,
                                AbstractGraphicViewer *viewer_ = nullptr);
//                void add_camera(const Eigen::Transform<float, 3, Eigen::Affine> &tf,
//                                const std::vector<std::string> &axes_,
//                                RoboCompJointMotorSimple::JointMotorSimplePrxPtr joint_proxy);
                void update_speed();
                //void update_joints();
                std::vector<std::tuple<float, float, bool>> create_bumper();
                Eigen::Vector2f compute_repulsion_forces(std::vector<Eigen::Vector2f> &floor_line);
                void draw_bumper();
                void draw_repulsion(const Eigen::Vector2f &repulsion);
                //Eigen::Vector3f get_robot_target_coordinates();
                float get_current_advance_speed() const;
                float get_current_side_speed()   const;
                float get_current_rot_speed() const;
                float get_target_angle_in_frame() const;
                float get_current_pan_angle() const;
                float get_current_pan_speed() const;
                float get_pure_rotation() const;
                rc::PreObject get_current_target() const;
                std::vector<std::tuple<float, float, bool>> get_bumper();
                float get_distance_to_target();
                Eigen::Transform<float, 3, Eigen::Affine> get_tf_cam_to_base();
                Eigen::Transform<float, 3, Eigen::Affine> get_tf_base_to_cam();
                //void zero_servo();
                void print();

                // robot control
                void goto_target(const std::vector<Eigen::Vector2f> &current_line);
                void stop();
                void just_rotate(float rot);
                void set_current_pan_speed(float vel);  // sends speed commands to robot
                void set_current_target(const rc::PreObject &target);

                // target
                void set_has_target(bool val);
                bool has_target() const;
                void set_desired_distance_to_target(float dist); //mm

                const float width = 450;
                const float length = 450;
                const float semi_width = width / 2;
                const float semi_height = length / 2;
                float security_offset = 100; //  mm
                float extended_security_offset = 0.f; //  mm

                // top camera
                const float top_camera_height = 1755; //mm 1555
                const float top_camera_y_offset = -40; //mm
                const float top_camera_x_offset = 0; //mm
                const float dist_between_bumper_points = semi_width / 8;
                const float camera_tilt_angle = -0.30;  //  20º -0.35
                float max_advance_speed = 1000;  // mm/sg
                float max_rot_speed = 1;  // rad/sg
                float max_pan_angle = M_PI_2; // rad
                float min_pan_angle = -M_PI_2; // rad
                void draw_optical_ray();

                // proxies
                //RoboCompJointMotorSimple::JointMotorSimplePrxPtr jointmotorsimple_proxy;
                RoboCompOmniRobot::OmniRobotPrxPtr omnirobot_proxy;

            private:
                AbstractGraphicViewer *viewer;
                float current_adv_speed = 0.f;
                float  current_side_speed = 0.f;
                float current_rot_speed = 0.f;
                float camera_pan_angle = 0.f;
                float camera_pan_speed = 0.f;
                float min_distance_to_target = 800.f;
                Eigen::Transform<float, 3, Eigen::Affine> tf;
                std::vector<std::string> axes;
                rc::PreObject current_target;

                bool has_target_flag = false;
                bool is_stopped = true;

                bool eye_tracking = false;
                float pure_rotation = 0.f;
                //std::map<float, float> bumper;
                std::vector<std::tuple<float, float, bool>> bumper;
                Eigen::ArrayXf sector1, sector2, sector3,  sector4, sector5;

                void recompute_bumper(float dynamic_offset);

                // DWA
                Dynamic_Window dwa;
    };
} // rc

#endif //FORCEFIELD_ROBOT_H
