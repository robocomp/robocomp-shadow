//
// Created by pbustos on 1/12/22.
//

#include "eye_tracker.h"

namespace rc
{
    float Eye_Tracker::track(rc::Robot &robot)
    {
        if (robot.get_pure_rotation() != 0)
        {
            robot.jointmotorsimple_proxy->setVelocity("camera_pan_joint", RoboCompJointMotorSimple::MotorGoalVelocity{0.f, 0.f});
            return 0.f;
        }
        float hor_angle = robot.get_target_angle_in_frame();  // angle wrt camera origin
        if (robot.get_current_target().type != -1)
        {
            if (fabs(hor_angle) > consts.max_hor_angle_error)  // saccade
            {
                try
                {
                    float error = 0.4 * (robot.get_current_pan_angle() - hor_angle);
                    float new_angle = std::clamp(error, -1.f, 1.f);  // dumping
                    //qInfo() << __FUNCTION__ << "SACCADE" << hor_angle << "error" << error << "saccade to: " << new_angle;
                    robot.jointmotorsimple_proxy->setPosition("camera_pan_joint", RoboCompJointMotorSimple::MotorGoalPosition{new_angle, 1});
                }
                catch (const Ice::Exception &e)
                {
                    std::cout << e.what() << " Error connecting to MotorGoalPosition" << std::endl;
                    return 0.f;
                }
            } else    // smooth pursuit
            {
                try
                {
                    float new_vel = -1.2 * hor_angle;
                    //new_vel = std::clamp(new_vel, -1.f, 1.f);  // dumping
                    new_vel -= 0.4 * robot.get_current_rot_speed();  // compensate with current base rotation speed
                    //qInfo() << __FUNCTION__ << "SMOOTH" << hor_angle << "smooth vel: " << new_vel << "robot rot speed" << robot.get_current_rot_speed();
                    robot.jointmotorsimple_proxy->setVelocity("camera_pan_joint", RoboCompJointMotorSimple::MotorGoalVelocity{new_vel, 1});
                    //qInfo() << __FUNCTION__ << "smooth" << hor_angle << current_servo_angle << new_vel;
                }
                catch (const Ice::Exception &e)
                {
                    std::cout << e.what() << " Error connecting to MotorGoalPosition" << std::endl;
                    return 0.f;
                }
            }
        } else  // inhibition of return
            try
            { robot.jointmotorsimple_proxy->setPosition("camera_pan_joint", RoboCompJointMotorSimple::MotorGoalPosition{0.f, 1.f}); }
            catch (const Ice::Exception &e)
            {
                std::cout << e.what() << std::endl;
                return 0.f;
            }

        return hor_angle;
    }
}