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

#include <genericworker.h>
#include "dsr/api/dsr_api.h"
#include "dsr/gui/dsr_gui.h"
#include <doublebuffer/DoubleBuffer.h>
#include <personcone.h>

//class Pyramid
//{
//public:
//    Eigen::Vector3f p1, p2, p3, p4, p5;
//    // Generate a 5 vertex pyramid given height from square center and square size
//    Pyramid(const Eigen::Vector3f& topVertex,
//            const Eigen::Vector3f& baseCenter,
//            float baseSize)
//    {
//
//
//
//
////        // Calcular los vértices de la base de la pirámide
////        Eigen::Vector3f xDir = Eigen::Vector3f::UnitX() * baseSize / 2;
////        Eigen::Vector3f zDir = Eigen::Vector3f::UnitZ() * baseSize / 2;
////        // Print xDir and zDir
////         std::cout << "xDir: " << xDir.transpose() << std::endl;
////         std::cout << "zDir: " << zDir.transpose() << std::endl;
////        Eigen::Vector3f topToCenter = baseCenter - topVertex;
////        Eigen::Vector3f normalVector = topToCenter.normalized();
////        this->p1 = baseCenter - xDir - zDir;
////        this->p2 = baseCenter + xDir - zDir;
////        this->p3 = baseCenter + xDir + zDir;
////        this->p4 = baseCenter - xDir + zDir;
////        this->p1 = baseCenter + normalVector * (this->p1 - baseCenter).dot(normalVector);
////        this->p2 = baseCenter + normalVector * (this->p2 - baseCenter).dot(normalVector);
////        this->p3 = baseCenter + normalVector * (this->p3 - baseCenter).dot(normalVector);
////        this->p4 = baseCenter + normalVector * (this->p4 - baseCenter).dot(normalVector);
////        this->p5 = topVertex;
//    }
//    float area(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, const Eigen::Vector3f& p3) const
//    {
//        return 0.5 * ((p2 - p1).cross(p3 - p1)).norm();
//    }
//    bool is_inside(const Eigen::Vector3f &point) const
//    {
//        // Calcular el área total de la pirámide
//        float totalArea = area(p1, p2, p5) +
//                          area(p2, p3, p5) +
//                          area(p3, p4, p5) +
//                          area(p4, p1, p5);
//
//        // Calcular el área de los triángulos formados por el punto y cada cara de la pirámide
//        float area1 = area(point, p2, p3);
//        float area2 = area(point, p3, p4);
//        float area3 = area(point, p4, p1);
//        float area4 = area(point, p1, p2);
//        // Si la suma de las áreas de los triángulos es igual al área total de la pirámide, el punto está dentro
//        return (totalArea == area1 + area2 + area3 + area4);
//    }
//    void print_pyramid_values()
//    {
//        std::cout << "Inf iz: " << p1.transpose() << std::endl;
//        std::cout << "Inf der: " << p2.transpose() << std::endl;
//        std::cout << "Up der: " << p3.transpose() << std::endl;
//        std::cout << "Up iz: " << p4.transpose() << std::endl;
//        std::cout << "topVertex: " << p5.transpose() << std::endl;
//    }
//};

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
	SpecificWorker(TuplePrx tprx, bool startup_check);
	~SpecificWorker();
	bool setParams(RoboCompCommonBehavior::ParameterList params);



public slots:
	void compute();
	int startup_check();
	void initialize(int period);
	void modify_node_slot(std::uint64_t, const std::string &type){};
	void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
	void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
	void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
	void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
	void del_node_slot(std::uint64_t from){};
private:
	// DSR graph
	std::shared_ptr<DSR::DSRGraph> G;
    std::unique_ptr<DSR::RT_API> rt;

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

    //local widget
    DSR::QScene2dViewer* widget_2d;

    // Pilar cone parameters
    float cone_radius = 3000;
    float cone_angle = 1;   // rads

    // Person cones
    std::vector<PersonCone> person_cones;
    bool element_inside_cone(const Eigen::Vector3f& point,
                             const Eigen::Vector3f& basePoint,
                             const Eigen::Vector3f& apexPoint,
                             double radius);
    std::optional<std::tuple<float, float, float, float>> get_rt_data(const DSR::Node &n, uint64_t to);
    void delete_edge(uint64_t from, uint64_t to, const std::string &edge_tag);
    void insert_edge(uint64_t from, uint64_t to, const std::string &edge_tag);

};

#endif
