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

/**
	\brief
	@author authorname
*/



#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <genericworker.h>
#include "dsr/api/dsr_api.h"
#include "dsr/gui/dsr_gui.h"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <doublebuffer/DoubleBuffer.h>
#include "/home/robocomp/robocomp/components/robocomp-shadow/etc/plan.h"
#include "/home/robocomp/robocomp/components/robocomp-shadow/etc/graph_names.h"

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
private:
	// DSR graph
	std::shared_ptr<DSR::DSRGraph> G;

	//DSR params
	std::string agent_name;
	int agent_id;

	bool tree_view;
	bool graph_view;
	bool qscene_2d_view;
	bool osg_3d_view;

    DoubleBuffer<uint64_t, uint64_t> person_buffer;

    std::optional<RoboCompRealSenseFaceID::ROIdata> detectAndDraw( cv::Mat& img, cv::CascadeClassifier& cascade, double scale );
    std::optional<cv::Mat> get_face_ID_image();
    std::optional<cv::Mat> get_person_ROI_from_node(DSR::Node person_node);
    std::optional<cv::Point2i> get_max_correlation_point(cv::Mat face_person_roi, cv::Mat person_roi);
    bool check_if_max_correlation_in_face(cv::Mat person_roi, cv::Point2i max_corr_point);



    int try_counter_with_match = 0;
    int try_counter_without_match = 0;
    double scale=1;

	// DSR graph viewer
	std::unique_ptr<DSR::DSRViewer> graph_viewer;
	QHBoxLayout mainLayout;
	void modify_node_slot(std::uint64_t id, const std::string &type){};
	void modify_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names);
	void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type);

	void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
	void del_node_slot(std::uint64_t from){};     
	bool startup_check_flag;

};

#endif
