//
// Created by pbustos on 18/12/22.
//

#ifndef FORCEFIELD_PREOBJECT_H
#define FORCEFIELD_PREOBJECT_H

#include <vector>
#include "door_detector.h"
#include <YoloObjects.h>
#include <Eigen/Dense>

namespace rc
{
    class PreObject
    {
        public:
            PreObject() = default;
            PreObject(const DoorDetector::Door &d);
            PreObject(const RoboCompYoloObjects::TBox &box, const  Eigen::Transform<float, 3, Eigen::Affine> &tf);
            Eigen::Vector3f get_robot_coordinates() const;
            Eigen::Vector3f get_local_coordinates() const;
            void print() const;
            static std::vector<PreObject> add_doors(const std::vector<DoorDetector::Door> &doors);
            static std::vector<PreObject> add_yolo(const std::vector<RoboCompYoloObjects::TBox> &boxes, const  Eigen::Transform<float, 3, Eigen::Affine> &tf);

            int id = 0;         // 0:person     81:door
            std::string label;
            int type = -1;
            int left;
            int top;
            int right;
            int bot;
            float score;  // 0-1
            float depth;  // distance to bbox center in mm
            // local coordinates, as seen from de device where it was detected
            float x;      // roi center coordinates in camera CS
            float y;
            float z;
            // robot coordinates, as seen from the base of the robot
            float rx;
            float ry;
            float rz;
    };
} // rc

#endif //FORCEFIELD_PREOBJECT_H
