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
#include <cppitertools/combinations.hpp>

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
        connect(grid_viewer, &AbstractGraphicViewer::new_mouse_coordinates, [](auto p){ qInfo() << p;});

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
        //points = std::make_shared<std::vector<std::tuple<float, float, float>>>();
        //colors = std::make_shared<std::vector<std::tuple<float, float, float>>>();

        // YOLO get object names and joints_list
        try
        {
            yolo_object_names = yoloobjects_proxy->getYoloObjectNames();
            qInfo() << "Yolo objects names successfully read";
            // excluded names
            excluded_yolo_types.push_back(std::distance(yolo_object_names.begin(), std::ranges::find(yolo_object_names, "bench")));
            excluded_yolo_types.push_back(std::distance(yolo_object_names.begin(), std::ranges::find(yolo_object_names, "toilet")));
            excluded_yolo_types.push_back(std::distance(yolo_object_names.begin(), std::ranges::find(yolo_object_names, "surfboard")));
        }
        catch(const Ice::Exception &e) {std::cout << e.what() << " Error connecting with YoloObjects interface to retrieve names" << std::endl;}

        // pixmaps
        object_pixmaps[std::distance(yolo_object_names.begin(), std::ranges::find(yolo_object_names, "person"))] = QPixmap("human.png").scaled(800,800);
        object_pixmaps[std::distance(yolo_object_names.begin(), std::ranges::find(yolo_object_names, "chair"))] = QPixmap("chair.png").scaled(800,800);
        object_pixmaps[std::distance(yolo_object_names.begin(), std::ranges::find(yolo_object_names, "potted plant"))] = QPixmap("plant.png").scaled(800,800);

        // RANSAC
        Estimator.Initialize(20, 100); // Threshold, iterations

        this->Period = 50;
		timer.start(Period);
	}
}
void SpecificWorker::compute()
{
    //points->clear(); colors->clear();

    //get omni_depth
    std::optional<std::vector<float>> depth_omni_o;
    if(depth_omni_o = cam_omni_api->get_depth_image(); not depth_omni_o.has_value())
    { qWarning() << "No rgb attribute in node " << cam_head_api->get_id(); return;}
    cv::Mat depth_frame(cv::Size(cam_omni_api->get_depth_width(), cam_omni_api->get_depth_height()), CV_32FC1, &depth_omni_o.value()[0]);

    // get head_rgb
    std::optional<std::vector<uint8_t>> rgb_head_o;
    if(rgb_head_o = cam_head_api->get_rgb_image(); not rgb_head_o.has_value())
    { qWarning() << "No rgb attribute in node " << cam_head_api->get_id(); return;}
    cv::Mat rgb_head(cv::Size(cam_head_api->get_rgb_width(), cam_head_api->get_rgb_height()), CV_8UC3, &rgb_head_o.value()[0]);

    // get head_depth
    std::optional<std::vector<float>> depth_top_o;
    if(depth_top_o = cam_head_api->get_depth_image(); not depth_top_o.has_value())
     { qWarning() << "No depth attribute in node " << cam_head_api->get_id(); return;}
    cv::Mat depth_head(cv::Size(cam_head_api->get_depth_width(), cam_head_api->get_depth_height()), CV_32FC1, &depth_top_o.value()[0]);

    ////////////////////////////////////////////////////////////////////////////////////////

    auto points = get_omni_3d_points(depth_frame, rgb_head);  // compute 3D points form omni_depth

    // Group 3D points by angular sectors and sorted by minimum distance
    auto sets = group_by_angular_sectors(points, false);    // group them in 360 set sectors sorted by distance

    // compute floor line
    auto floor_line = compute_floor_line(sets, false);      // extract the first element of eachs set
    std::vector<Eigen::Vector2f> floor_line_cart;
    for(const auto &p : floor_line)
        floor_line_cart.emplace_back(Eigen::Vector2f(p(1)*sin(p(0)), p(1)*cos(p(0))));

    // yolo
    auto yolo_objects = get_yolo_objects();
    draw_yolo_objects(yolo_objects, rgb_head);

    // update local_grid
    //local_grid.update_map_from_polar_data(floor_line, 5000);
    //local_grid.update_semantic_layer(atan2(o.x, o.y), o.depth, o.id, o.type);  // rads -PI,PI and mm

    // compute mean point
    Eigen::Vector2f mean{0.0, 0.0};
    mean = std::accumulate(floor_line_cart.begin(), floor_line_cart.end(), mean) / (float)floor_line_cart.size();
    //qInfo() << "Center " << mean(0) << mean(1);

    // estimate size
    Eigen::MatrixX2f zero_mean_points(floor_line_cart.size(), 2);
    for(const auto &[i, p] : iter::enumerate(floor_line_cart))
        zero_mean_points.row(i) = p - mean;
    auto max = zero_mean_points.colwise().maxCoeff();
    auto min = zero_mean_points.colwise().minCoeff();
    //std::cout << "max " << max << "min " << min << std::endl;

    // compute lines
    std::vector<cv::Vec2f> floor_line_cv;
    for(const auto &p : floor_line_cart)
        floor_line_cv.emplace_back(cv::Vec2f(p.x(), p.y()));
    double rhoMin = 0.0f, rhoMax = 5000.0f, rhoStep = 10;
    double thetaMin = 0.0f, thetaMax = 2.0*CV_PI, thetaStep = CV_PI / 180.0f;
    cv::Mat lines;
    HoughLinesPointSet(floor_line_cv, lines, 20, 1, rhoMin, rhoMax, rhoStep, thetaMin, thetaMax, thetaStep);
    std::vector<cv::Vec3d> lines3d;
    lines.copyTo(lines3d);
    draw_on_2D_tab(lines3d);

    // compute intersections
    std::vector<QLineF> elines;
    //std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> line_points;
    for(auto &l: std::ranges::filter_view(lines3d, [](auto a){return a[0]>10;}))
    {
        float rho = l[1], theta = l[2];
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        QPointF p1(x0 + 5000 * (-b), y0 + 5000 * (a));
        QPointF p2(x0 - 5000 * (-b), y0 - 5000 * (a));
        elines.emplace_back(QLineF(p1, p2));
        //line_points.emplace_back(std::make_pair(p2, p2));
    }
    std::vector<Eigen::Vector2f> corners;
    for(auto &&comb: iter::combinations(elines, 2))
    {
        auto &line1= comb[0]; auto &line2 = comb[1];
        auto angle = qDegreesToRadians(line1.angle(line2));
        float delta = 0.1;
        QPointF intersection;
        if(fabs(angle) < M_PI/2+delta and fabs(angle) > M_PI/2-delta  and line1.intersect(line2, &intersection) == 1)
            corners.emplace_back(Eigen::Vector2f(intersection.x(), intersection.y()));
    }
    draw_on_2D_tab(corners, "blue");

    // draw on 2D
    draw_on_2D_tab(floor_line_cart, "green", false);
    draw_on_2D_tab(yolo_objects);
    grid_viewer->viewport()->repaint();
    cv::imshow("Top_camera", rgb_head);
    cv::waitKey(5);

    fps.print("FPS: ", [this](auto x){ graph_viewer->set_external_hz(x);});
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////77
RoboCompYoloObjects::TObjects SpecificWorker::get_yolo_objects()
{
    try
    {
        auto yolo_data = yoloobjects_proxy->getYoloObjects();
        return yolo_data.objects;
    }
    catch(const Ice::Exception &e){ std::cerr << e.what() << ". No response from YoloServer" << std::endl;};
    return RoboCompYoloObjects::TObjects();
}
void SpecificWorker::draw_on_2D_tab(const RoboCompYoloObjects::TObjects &objects)
{
    static std::vector<QGraphicsItem*> pixmap_ptrs;
    for(auto p: pixmap_ptrs)
    {
        widget_2d->scene.removeItem(p);
        delete p;
    }
    pixmap_ptrs.clear();

    for(auto &o: objects)
        if(o.score > 0.5 and std::ranges::find(excluded_yolo_types, o.type) == excluded_yolo_types.end())
        {
            auto p = widget_2d->scene.addPixmap(object_pixmaps[o.type]);
            p->setPos(o.x-object_pixmaps[o.type].width()/2, o.y-object_pixmaps[o.type].height()/2);
            pixmap_ptrs.push_back(p);
        }
}
void SpecificWorker::draw_on_2D_tab(const std::vector<Eigen::Vector2f> &points, QString color, bool clean)
{
    static std::vector<QGraphicsRectItem*> dot_vec;
    if(clean)
        if(not dot_vec.empty())
        {
            for (auto p: dot_vec)
            {
                widget_2d->scene.removeItem(p);
                delete p;
            }
            dot_vec.clear();
        }
    for(const auto &l: points)
    {
        auto p = widget_2d->scene.addRect(-15, -15, 30, 30, QPen(QColor(color),20));
        dot_vec.push_back(p);
        p->setPos(l.x(), l.y());
    }
}
void SpecificWorker::draw_on_2D_tab(const std::vector<cv::Vec3d> &lines)
{
    static std::vector<QGraphicsItem*> lines_vec;
    for (auto l: lines_vec)
    {
        widget_2d->scene.removeItem(l);
        delete l;
    }
    lines_vec.clear();
    for(auto &l: std::ranges::filter_view(lines, [](auto a){return a[0]>10;}))
    {
        float rho = l[1], theta = l[2];
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        QLineF line(x0 + 5000*(-b), y0 + 5000*(a), x0 - 5000*(-b), y0 - 5000*(a));
        auto p = widget_2d->scene.addLine(line, QPen(QColor("orange"), 40));
        lines_vec.push_back(p);
    }
}
cv::Mat SpecificWorker::draw_yolo_objects(const RoboCompYoloObjects::TObjects &objects, cv::Mat img)
{
    // Plots one bounding box on image img
    cv::Mat cimg;
    for(const auto &box: objects)
    {
        auto tl = round(0.002 * (img.cols + img.rows) / 2) + 1; // line / fontthickness
        cv::Scalar color(0, 0, 200); // box color
        cv::Point c1(box.left, box.top);
        cv::Point c2(box.right, box.bot);
        cv::rectangle(img, c1, c2, color, tl, cv::LINE_AA);
        int tf = (int) std::max(tl - 1, 1.0);  // font thickness
        int baseline = 0;
        std::string label = yolo_object_names.at(box.type) + " " + std::to_string((int) (box.score * 100)) + "%";
        auto t_size = cv::getTextSize(label, 0, tl / 3.f, tf, &baseline);
        c2 = {c1.x + t_size.width, c1.y - t_size.height - 3};
        cv::rectangle(img, c1, c2, color, -1, cv::LINE_AA);  // filled
        cv::putText(img, label, cv::Size(c1.x, c1.y - 2), 0, tl / 3, cv::Scalar(225, 255, 255), tf, cv::LINE_AA);
    }
    return cimg;
}
std::vector<Point3f> SpecificWorker::get_omni_3d_points(const cv::Mat &depth_frame, const cv::Mat &rgb_frame)   // in meters
{
     // Let's assume that each column corresponds to a polar coordinate: ang_step = 360/image.width
     // and along the column we have radius
     // hor ang = 2PI/512 * i

    //std::size_t i = points->size();  // to continue inserting
    std::vector<Point3f> points(depth_frame.rows * depth_frame.cols);
    //points->resize(depth_frame.rows * depth_frame.cols);
    //colors->resize(points->size());
    int semi_height = depth_frame.rows/2;
    float hor_ang, dist, x, y, z, proy;
    float ang_slope = 2*M_PI/depth_frame.cols;
    //int nancount=0;
    std::size_t i = 0;  // direct access to points
    for(int u=0; u<depth_frame.rows; u=u+1)
        for(int v=0; v<depth_frame.cols; ++v)
        {
            hor_ang = ang_slope * v - M_PI; // cols to radians
            //qInfo() << __FUNCTION__ <<  u << v << depth_frame.cols;
            dist = depth_frame.ptr<float>(u)[v] * consts.scaling_factor;  // pixel to dist scaling factor  -> to mm
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
            points[i] = std::make_tuple(x, y, z);
            //auto rgb = rgb_frame.ptr<cv::Vec3b>(u)[v];
            //colors->operator[](i) = std::make_tuple(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0);
            i += 1;
        };
    return points;
}
SpecificWorker::SetsType SpecificWorker::group_by_angular_sectors(const std::vector<Point3f> &points, bool draw)
{
    const double ang_bin = 2.0*M_PI/consts.num_angular_bins;
    SetsType sets(consts.num_angular_bins);
    for(const auto &[x, y, z] : points)
    {
        //const auto &[x,y,z] = point;
        //if(fabs(x)<0.1 and fabs(y)<0.1) continue;
        int ang_index = floor((M_PI + atan2(x, y))/ang_bin);
        //float dist = sqrt(x*x+y*y+z*z);
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
        float dist = p.head(2).norm();
        if(fabs(dist) > consts.max_camera_depth_range/1000) continue;
        floor_line.emplace_back(Eigen::Vector2f(ang-M_PI, p.head(2).norm()*1000.0));  // to undo the +M_PI in group_by_angular_sectors
    }
    //qInfo() << __FUNCTION__ << "--------------------------------------";
    return floor_line;
}

/////////////////////////////////////////////////////////////////////////////////////////777
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}



//std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> CandPoints;
//for(const auto &p: floor_line)
//{
//std::shared_ptr<GRANSAC::AbstractParameter> CandPt = std::make_shared<Point2D>(p.x(), p.y());
//CandPoints.push_back(CandPt);
//}
//int64_t start = cv::getTickCount();
//Estimator.Estimate(CandPoints);
//int64_t end = cv::getTickCount();
////std::cout << "RANSAC took: " << GRANSAC::VPFloat(end - start) / GRANSAC::VPFloat(cv::getTickFrequency()) * 1000.0 << " ms." << std::endl;
//auto BestLine = Estimator.GetBestModel();
//if (BestLine)
//{
//auto BestLinePt1 = std::dynamic_pointer_cast<Point2D>(BestLine->GetModelParams()[0]);
//auto BestLinePt2 = std::dynamic_pointer_cast<Point2D>(BestLine->GetModelParams()[1]);
//static QGraphicsItem* temp;
//if(temp != nullptr) widget_2d->scene.removeItem(temp);
//if (BestLinePt1 && BestLinePt2)
//{
//cv::Point Pt1(BestLinePt1->m_Point2D[0], BestLinePt1->m_Point2D[1]);
//cv::Point Pt2(BestLinePt2->m_Point2D[0], BestLinePt2->m_Point2D[1]);
//temp = widget_2d->scene.addLine(QLineF( BestLinePt1->m_Point2D[0], BestLinePt1->m_Point2D[1],
//                                        BestLinePt2->m_Point2D[0], BestLinePt2->m_Point2D[1]), QPen(QColor("orange"), 40));
//}
//}


//cv::Mat edges = cv::Mat::zeros(200, 200, CV_8UC1);
//for(const auto &p: floor_line)
//{
//int x = p(1) * sin(p(0)) * (200 / 9000.) + 100;   // scale to 0--200
//int y = p(1) * cos(p(0)) * (200 / 9000.) + 100;   // scale to 0--200
////qInfo() << __FUNCTION__ << p(1)*sin(p(0)) << p(1)*cos(p(0)) << x << y;
//if(x>= 0 and x<200 and y>=0 and y<200)
//edges.at<uchar>(x, y) = 255;
//}
//cv::Mat cedges;
//cv::cvtColor(edges, cedges, cv::COLOR_GRAY2BGR);
//cv::Mat lines_img = cedges.clone();
//
//std::vector<cv::Vec4i> lines;
//HoughLinesP(edges, lines, 1, CV_PI/180, 5, 10, 5 );
//qInfo() << __FUNCTION__ << lines.size();
//draw_on_2D_tab(lines);
//cv::imshow("sdf", edges);
//cv::waitKey(1);
