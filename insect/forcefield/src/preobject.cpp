//
// Created by pbustos on 18/12/22.
//

#include "preobject.h"

namespace rc
{
    PreObject::PreObject(const RoboCompYoloObjects::TBox &box, const  Eigen::Transform<float, 3, Eigen::Affine> &tf) :
        id(box.id), type(box.type), left(box.left), top(box.top),
        right(box.right), bot(box.bot), score(box.score), depth(box.depth),
        x(box.x), y(box.y), z(box.z)
    {
        Eigen::Vector3f rc = tf * Eigen::Vector3f(box.x, box.y, box.z);
        rx = rc.x();
        ry = rc.y();
        rz = rc.z();
        label = std::to_string(box.type);
    }
    PreObject::PreObject(const DoorDetector::Door &d)
    {
        type = 80;  // doors
        label = "door";
        rx = d.center_floor.x();
        ry = d.center_floor.y();
        rz = 0.f;
        x = d.center_floor.x();
        y = d.center_floor.y();
        z = 0.f;
    }
    std::vector<PreObject> PreObject::add_yolo(const std::vector<RoboCompYoloObjects::TBox> &boxes, const  Eigen::Transform<float, 3, Eigen::Affine> &tf )
    {
        std::vector<PreObject> bobjs;
        for(const auto &b: boxes)
            bobjs.emplace_back(PreObject(b, tf));
        return bobjs;
    }
    std::vector<PreObject> PreObject::add_doors(const std::vector<DoorDetector::Door> &doors)
    {
        std::vector<PreObject> bdoors;
        for(const auto &d: doors)
            bdoors.emplace_back(PreObject(d));
        return bdoors;
    }
    Eigen::Vector3f PreObject::get_local_coordinates() const
    {
        return Eigen::Vector3f(x, y, z);
    }
    Eigen::Vector3f PreObject::get_robot_coordinates() const
    {
        return Eigen::Vector3f(rx, ry, rz);
    }
    void PreObject::print() const
    {
        std::cout << "Object: " << std::endl;
        std::cout << "  type: " << type << std::endl;
        std::cout << "  rx: " << rx << std::endl;
        std::cout << "  ry: " << ry << std::endl;
        std::cout << "  rz: " << rz << std::endl;
        std::cout << "--------------"<< std::endl;
    }
} // rc