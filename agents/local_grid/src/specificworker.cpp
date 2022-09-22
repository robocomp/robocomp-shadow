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

        // Custom widget
        graph_viewer->add_custom_widget_to_dock("Local grid", &grid_widget);
        grid_viewer = new AbstractGraphicViewer(&grid_widget, QRectF{-3000, -3000, 6000, 6000});

        // Apis
        inner_eigen = G->get_inner_eigen_api();
        if(auto camera_node = G->get_node(shadow_omni_camera_name); camera_node.has_value())
            cam_omni_api = G->get_camera_api(camera_node.value());
        else {qWarning() << "No " << QString::fromStdString(shadow_omni_camera_name) << "found in G. Aborting"; std::terminate();}
        if(auto camera_node = G->get_node(shadow_camera_head_name); camera_node.has_value())
            cam_head_api = G->get_camera_api(camera_node.value());
        else {qWarning() << "No " << QString::fromStdString(shadow_omni_camera_name) << "found in G. Aborting"; std::terminate();}
        agent_info_api = std::make_unique<DSR::AgentInfoAPI>(G.get());

        // localgrid
        Local_Grid::Ranges angle_dim{ .init=0, .end=360, .step=5}, radius_dim{.init=0.f, .end=4000.f, .step=200};
        local_grid.initialize( angle_dim, radius_dim, &grid_viewer->scene);
        points = std::make_shared<std::vector<std::tuple<float, float, float>>>();
        colors = std::make_shared<std::vector<std::tuple<float, float, float>>>();

        // YOLO get object names and joints_list
        try
        {
            yolo_object_names = yoloobjects_proxy->getYoloObjectNames();
            qInfo() << "Yolo objects names successfully read";
        }
        catch(const Ice::Exception &e) {std::cout << e.what() << " Error connecting with YoloObjects interface to retrieve names" << std::endl;}

        this->Period = 50;
		timer.start(Period);
	}
}
void SpecificWorker::compute()
{
    points->clear(); colors->clear();
    cv::Mat depth_frame;
    //    if(auto depth_o = cam_omni_api->get_depth_image(); depth_o.has_value())
    //    {
    //        auto &depth = depth_o.value();
    //        depth_frame = cv::Mat(cv::Size(cam_api->get_depth_width(), cam_api->get_depth_height()), CV_32FC1, &depth[0]);
    //    }
    //    depth_frame = read_depth_omni();
    std::vector<float> depth_buffer;
    auto cam_node = G->get_node(shadow_omni_camera_name);
    if (auto g_depth = G->get_attrib_by_name<cam_depth_att>(cam_node.value()); g_depth.has_value())
    {
        float *depth_array = (float *)g_depth.value().get().data();
        depth_buffer = std::vector<float>{depth_array, depth_array + g_depth.value().get().size() / sizeof(float)};
        depth_frame = cv::Mat(cv::Size(cam_omni_api->get_depth_width(), cam_omni_api->get_depth_height()), CV_32FC1, &depth_buffer[0], cv::Mat::AUTO_STEP);
    }

    cv::Mat rgb_head;
    if(auto rgb_o = cam_head_api->get_rgb_image(); rgb_o.has_value())
    {
        auto &frame = rgb_o.value();
        rgb_head = cv::Mat(cv::Size(cam_head_api->get_rgb_width(), cam_head_api->get_rgb_height()), CV_8UC3, &frame[0]);
        //cv::imshow("rgb", rgb_head);
        //cv::waitKey(5);
    }

    get_omni_3d_points(depth_frame, rgb_head);  //float y uchar

    // Group 3D points by angular sectors and sorted by minimum distance
    auto sets = group_by_angular_sectors(false);

    // compute floor line
    auto floor_line = compute_floor_line(sets, false);

    // yolo
    auto yolo_objects = get_yolo_objects(rgb_head);

    // update local_grid
    local_grid.update_map_from_polar_data(floor_line, 5000);
    //local_grid.update_semantic_layer(yolo_objects);

    // draw on 2D
    draw_on_2D_tab(floor_line);
    grid_viewer->viewport()->repaint();

    fps.print("FPS: ", [this](auto x){ graph_viewer->set_external_hz(x);});

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////77
RoboCompYoloObjects::TObjects SpecificWorker::get_yolo_objects(cv::Mat frame)
{
    try
    {
        auto yolo_data = yoloobjects_proxy->getYoloObjects();
        //qInfo() << __FUNCTION__ <<  "YOLO" << yolo_data.objects.size();
        draw_yolo_objects(yolo_data.objects, frame);
        return yolo_data.objects;
    }
    catch(const Ice::Exception &e){ std::cerr << e.what() << ". No response from YoloServer" << std::endl;};
    return RoboCompYoloObjects::TObjects();
}
void SpecificWorker::draw_yolo_objects(const RoboCompYoloObjects::TObjects &objects, cv::Mat img)
{
    // Plots one bounding box on image img
    //qInfo() << __FUNCTION__ << QString::fromStdString(box.name) << box.left << box.top << box.right << box.bot << box.prob;
    for(const auto &box: objects)
    {
        auto tl = round(0.002 * (img.cols + img.rows) / 2) + 1; // line / fontthickness
        cv::Scalar color(0, 0, 200); // box color
        cv::Point c1(box.left, box.top);
        cv::Point c2(box.right, box.bot);
        cv::rectangle(img, c1, c2, color, tl, cv::LINE_AA);
        int tf = (int) std::max(tl - 1, 1.0);  // font thickness
        int baseline = 0;
        std::string label = yolo_object_names.at(box.type) + " " + std::to_string((int) (box.prob * 100)) + "%";
        auto t_size = cv::getTextSize(label, 0, tl / 3.f, tf, &baseline);
        c2 = {c1.x + t_size.width, c1.y - t_size.height - 3};
        cv::rectangle(img, c1, c2, color, -1, cv::LINE_AA);  // filled
        cv::putText(img, label, cv::Size(c1.x, c1.y - 2), 0, tl / 3, cv::Scalar(225, 255, 255), tf, cv::LINE_AA);
    }
    cv::imshow("central camera", img);
    cv::waitKey(5);
}
void SpecificWorker::get_omni_3d_points(const cv::Mat &depth_frame, const cv::Mat &rgb_frame)   // in meters
{
     // Let's assume that each column corresponds to a polar coordinate: ang_step = 360/image.width
     // and along the column we have radius
     // hor ang = 2PI/512 * i

    //std::size_t i = points->size();  // to continue inserting
    std::size_t i = 0;  // to continue inserting
    points->resize(depth_frame.rows * depth_frame.cols);
    colors->resize(points->size());
    int semi_height = depth_frame.rows/2;
    float hor_ang, dist, x, y, z, proy, ang_slope = 2*M_PI/depth_frame.cols;
    //int nancount=0;
    for(int u=50; u<depth_frame.rows-50; u=u+1)
        for(int v=0; v<depth_frame.cols-1; ++v)
        {
            hor_ang = ang_slope * v - M_PI; // cols to radians
            //qInfo() << __FUNCTION__ <<  u << v << depth_frame.cols;
            dist = depth_frame.ptr<float>(u)[v] * 19.f;  // pixel to dist scaling factor  -> to mm
            //if(std::isnan(dist)) {nancount++; points->clear(); goto the_end;};
            if(dist > consts.max_camera_depth_range) continue;
            if(dist < consts.min_camera_depth_range) continue;
            dist /= 1000.f; // to meters
            x = -dist * sin(hor_ang);
            y = dist * cos(hor_ang);
            proy = dist * cos( atan2((semi_height - u), 128.f));
            z = (semi_height - u)/128.f * proy; // 128 focal as PI fov angle for 256 pixels
            z += consts.omni_camera_height_meters;

            // filter out floor
            if(z < 0.4 ) continue;
            points->operator[](i) = std::make_tuple(x, y, z);
            //auto rgb = rgb_frame.ptr<cv::Vec3b>(u)[v];
            //colors->operator[](i) = std::make_tuple(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0);
            i += 1;
        };
    //the_end:
    //std::cout << "---------------------------------------------------------" << std::endl;

    //qInfo() << __FUNCTION__ << nancount << "Points size" << points->size();
}
SpecificWorker::SetsType SpecificWorker::group_by_angular_sectors(bool draw)
{
    int num_ang_bins = 360;
    const double ang_bin = 2.0*M_PI/num_ang_bins;
    SetsType sets(num_ang_bins);
    for(const auto &point :*points)
    {
        const auto &[x,y,z] = point;
        //if(fabs(x)<0.1 and fabs(y)<0.1) continue;
        int ang_index = floor((M_PI + atan2(x, y))/ang_bin);
        float dist = sqrt(x*x+y*y+z*z);
        //if(dist < 0.1 or dist > 5) continue;
        //if(z < 0.4 ) continue;
        sets[ang_index].emplace(std::make_tuple(Eigen::Vector3f(x,y,z), std::make_tuple(0,0,0)));  // remove color
    }
    //qInfo() << __FUNCTION__ << "--------------------------------------";


    // regenerate points and colors
//    points->clear(); colors->clear();
//    if(draw)
//    {
//        for (const auto &s: sets)
//            if (not s.empty())
//                for (auto init = s.cbegin(); init != s.cend(); ++init)
//                {
//                    const auto &[point, color] = *init;
//                    for (auto &&t: iter::range(0.f, point.z(), 0.02f))
//                    {
//                        points->emplace_back(std::make_tuple(point.x(), point.y(), t));
//                        colors->emplace_back(color);
//                    }
//                }
//    }
    return sets;
}
vector<Eigen::Vector2f> SpecificWorker::compute_floor_line(const SpecificWorker::SetsType &sets, bool draw)  // meters
{
    vector<Eigen::Vector2f> floor_line;  //polar
    const double ang_bin = 2.0*M_PI/360;
    float ang= -ang_bin;
    for(const auto &s: sets)
    {
        ang += ang_bin;
        if(s.empty()) continue;
        Eigen::Vector3f p = get<Eigen::Vector3f>(*s.cbegin());  // first element is the smallest distance
        floor_line.emplace_back(Eigen::Vector2f(ang-M_PI, p.head(2).norm()*1000.0));  // to undo the +M_PI in group_by_angular_sectors
    }
    //qInfo() << __FUNCTION__ << "--------------------------------------";
    return floor_line;
}
void SpecificWorker::draw_on_2D_tab(const std::vector<Eigen::Vector2f> &points)
{
    static std::vector<QGraphicsRectItem*> dot_vec;
    static QGraphicsPolygonItem* poly_draw;
    if(not dot_vec.empty())
    {
        for (auto p: dot_vec)
        {
            widget_2d->scene.removeItem(p);
            delete p;
        }
        dot_vec.clear();
    }
    if(poly_draw != nullptr) widget_2d->scene.removeItem(poly_draw);

    QPolygonF poly;
    for(const auto &l: points)
    {
        //poly << QPointF(l.x()*1000, l.y()*1000);
        auto p = widget_2d->scene.addRect(-15, -15, 30, 30, QPen(QColor("Green"),10));
        //poly_draw = widget_2d->scene.addPolygon(poly, QPen(QColor("Blue"),10));
        dot_vec.push_back(p);
        p->setPos(l(1)*sin(l(0)), l(1)*cos(l(0)));
        //qInfo() << __FUNCTION__ << l(1)*sin(l(0)) << l(1)*cos(l(0));
        //p->setPos(QPointF(l.x()*1000, l.y()*1000));
    }
}
/////////////////////////////////////////////////////////////////////////////////////////777
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}




