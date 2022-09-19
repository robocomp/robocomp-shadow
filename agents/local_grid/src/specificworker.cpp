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
#include <cppitertools/enumerate.hpp>
#include <cppitertools/range.hpp>
#include <cppitertools/zip.hpp>

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
	agent_name = params["agent_name"].value;
	agent_id = stoi(params["agent_id"].value);
    if(agent_id == 0)
    { qWarning() << "Agents cannot have 0 id. Please change it in the config file"; std::terminate();}
	tree_view = params["tree_view"].value == "true";
	graph_view = params["graph_view"].value == "true";
	qscene_2d_view = params["2d_view"].value == "true";
	osg_3d_view = params["3d_view"].value == "true";

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
		// create graph
		G = std::make_shared<DSR::DSRGraph>(0, agent_name, agent_id, ""); // Init nodes
		std::cout<< __FUNCTION__ << "Graph loaded" << std::endl;  

		//dsr update signals
//		connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
//		connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
//		connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_attrs_slot);
//		connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
//		connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

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
        widget_2d = qobject_cast<DSR::QScene2dViewer*> (graph_viewer->get_widget(opts::scene));

        // Apis
        inner_eigen = G->get_inner_eigen_api();
        if(auto camera_node = G->get_node(shadow_omni_camera_name); camera_node.has_value())
            cam_api = G->get_camera_api(camera_node.value());
        else {qWarning() << "No " << QString::fromStdString(shadow_omni_camera_name) << "found in G. Aborting"; std::terminate();}
        agent_info_api = std::make_unique<DSR::AgentInfoAPI>(G.get());

        // localgrid
        points = std::make_shared<std::vector<std::tuple<float, float, float>>>();
        colors = std::make_shared<std::vector<std::tuple<float, float, float>>>();

		this->Period = 50;
		timer.start(Period);
	}
}

void SpecificWorker::compute()
{
    points->clear(); colors->clear();
    cv::Mat depth_frame;
    if(auto depth_o = cam_api->get_depth_image(); depth_o.has_value())
    {
        auto depth = depth_o.value();
        depth_frame = cv::Mat(cv::Size(cam_api->get_depth_width(), cam_api->get_depth_height()), CV_32FC1, &depth[0]);
    }

    cv::Mat rgb_frame;
    if(auto rgb_o = cam_api->get_rgb_image(); rgb_o.has_value())
    {
        auto &frame = rgb_o.value();
        rgb_frame = cv::Mat(cv::Size(cam_api->get_rgb_width(), cam_api->get_rgb_height()), CV_8UC3, &frame[0]);
        //cv::imshow("rgb", rgb);
        //cv::waitKey(5);
    }

    get_omni_3d_points(depth_frame, rgb_frame);

    // Group 3D points by angular sectors and sorted by minimum distance
    sets.clear();
    sets = group_by_angular_sectors(true);

//    // update local_grid
//    local_grid.update_map_from_3D_points(points);
//
    // compute floor line
    auto floor_line = compute_floor_line(sets);

    static QGraphicsPolygonItem *poly_draw=nullptr;
    if(poly_draw != nullptr)
        widget_2d->scene.removeItem(poly_draw);
    QPolygonF poly;
    for(const auto &l: floor_line)
        poly << QPointF(l.x()*1000, l.y()*1000);
    poly_draw = widget_2d->scene.addPolygon(poly, QPen(QColor("Orange"),50));
    poly_draw->setZValue(200);

    fps.print("FPS:");
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////77
void SpecificWorker::get_omni_3d_points(const cv::Mat &depth_frame, const cv::Mat &rgb_frame)   // in meters
{
     // Let's assume that each column corresponds to a polar coordinate: ang_step = 360/image.width
     // and along the column we have radius
     // hor ang = 2PI/512 * i

    std::size_t i = points->size();
    points->resize(depth_frame.rows * depth_frame.cols);
    colors->resize(points->size());
    int semi_height = depth_frame.rows/2;
    float hor_ang, dist, x, y, z, proy, ang_slope = 2*M_PI/depth_frame.cols;
    for(int u=50; u<depth_frame.rows-20; u=u+1)
        for(int v=0; v<depth_frame.cols; v++)
        {
            hor_ang = ang_slope * v - M_PI; // cols to radians
            dist = (float)depth_frame.ptr<uchar>(u)[v] * 19.f;  // pixel to dist scaling factor
            if(dist > consts.max_camera_depth_range) continue;
            if(dist < consts.min_camera_depth_range) continue;
            dist /= 1000.f; // to meters
            x = -dist * sin(hor_ang);
            y = dist * cos(hor_ang);
            proy = dist * cos( atan2((semi_height - u), 128.f));
            z = (semi_height - u)/128.f * proy; // 128 focal as PI fov angle for 256 pixels
            z += consts.omni_camera_height_meters;
            points->operator[](i) = std::make_tuple(x,y,z);
            auto rgb = rgb_frame.ptr<cv::Vec3b>(u)[v];
            colors->operator[](i++) = std::make_tuple(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0);
        };
}

SpecificWorker::SetsType SpecificWorker::group_by_angular_sectors(bool draw)
{
    int num_ang_bins = 360;
    const double ang_bin = 2.0*M_PI/num_ang_bins;
    SetsType sets(num_ang_bins);
    for(const auto &[point, color] : iter::zip(*points, *colors))
    {
        const auto &[x,y,z] = point;
        int ang_index = floor((M_PI + atan2(x, y))/ang_bin);
        float dist = sqrt(x*x+y*y+z*z);
        if(dist < 0.1 or dist > 4) continue;
        if(fabs(z) < 0.4 ) continue;
        sets[ang_index].emplace(std::make_tuple(Eigen::Vector3f(x,y,z), color));
    }

    // regenerate points and colors
    points->clear(); colors->clear();
    if(draw)
    {
        for (const auto &s: sets)
            if (not s.empty())
                for (auto init = s.cbegin(); init != s.cend(); ++init)
                {
                    const auto &[point, color] = *init;
                    for (auto &&t: iter::range(0.f, point.z(), 0.02f))
                    {
                        points->emplace_back(std::make_tuple(point.x(), point.y(), t));
                        colors->emplace_back(color);
                    }
                }
    }
    return sets;
}
vector<Eigen::Vector2f> SpecificWorker::compute_floor_line(const SpecificWorker::SetsType &sets, bool draw)  // meters
{
    vector<Eigen::Vector2f> floor_line;
    for(const auto &s: sets)
    {
        Eigen::Vector3f p = get<Eigen::Vector3f>(*s.cbegin());
        floor_line.emplace_back(Eigen::Vector2f(p.x(), p.y()));
        if(draw)
        {
            points->emplace_back(make_tuple(p.x(), p.y(), p.z()));
            colors->push_back({1.0, 1.0, 0.0});
        }
    }
    return floor_line;
}
/////////////////////////////////////////////////////////////////////////////////////////777
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}




