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
//	void modify_node_slot(std::uint64_t, const std::string &type){};
//	void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
//	void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
//	void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
//	void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
//	void del_node_slot(std::uint64_t from){};

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

    //Functions
    std::optional<std::tuple<float, float, float, float>> get_rt_data(const DSR::Node &n, uint64_t to);

    // Pilar cone parameters
    float cone_radius = 6000;
    float cone_angle = 0.7;   // rads

    //Person cones
    std::vector<PersonCone> person_cones;

    bool element_inside_cone(const Eigen::Vector3f& point,
                             const Eigen::Vector3f& basePoint,
                             const Eigen::Vector3f& apexPoint,
                             double radius);

    bool allow_prediction = true;
    
    void insert_edge(uint64_t from, uint64_t to, const string &edge_tag);
};

#endif
