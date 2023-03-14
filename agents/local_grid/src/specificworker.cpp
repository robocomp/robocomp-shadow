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
#include <cppitertools/combinations_with_replacement.hpp>
#include <cppitertools/range.hpp>

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
        //Estimator.Initialize(20, 100); // Threshold, iterations

        // cube
        //Eigen::Tensor<double, 5> cube(10, 10, 10, 10, 10);
        //stimer.tick();
        Eigen::TensorFixedSize<float, Eigen::Sizes<15, 15, 15, 15, 15>> cube;
        cube.setZero();
        //stimer.tock();
        //std::cout << stimer.duration() << " " << m <<  " " << cube.size() << std::endl;

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

    // get level lines from camera depth image
    auto ml_points = get_multi_level_3d_points(depth_frame, rgb_head);  // compute 3D points form omni_depth
    std::vector<Eigen::Vector2f> floor_line_cart;

    // extract floor_line
    for(const auto &p: ml_points[3])
        if(p.norm() < consts.max_camera_depth_range)
            floor_line_cart.emplace_back(p);

    // get yolo detections
    auto yolo_objects = get_yolo_objects();
    draw_yolo_objects(yolo_objects, rgb_head);

    // update local_grid
    //local_grid.update_map_from_polar_data(floor_line, 5000);
    //local_grid.update_semantic_layer(atan2(o.x, o.y), o.depth, o.id, o.type);  // rads -PI,PI and mm

    // compute mean point
    Eigen::Vector2f room_center = get_mean(floor_line_cart);

    // estimate size
    float estimated_size = get_size(room_center, floor_line_cart);
    //create_image_for_learning(floor_line_cart, estimated_size);

    room.update(QSizeF(estimated_size, estimated_size), room_center);

    // compute lines
    std::vector<pair<int, QLineF>> elines = get_hough_lines(floor_line_cart);

    // compute parallel lines of minimum length and separation
    std::vector<std::pair<QLineF, QLineF>> par_lines = get_parallel_lines(elines, estimated_size);

    // compute corners
    std::vector<QLineF> corners = get_corners(elines);

    // compute triple corners (double corner and the third one is inferred)
    //vector<tuple<QLineF, QLineF, QLineF>> double_corners;
    vector<tuple<QLineF, QLineF, QLineF>> double_corners = get_double_corners(estimated_size, corners);

    // compute room. Start with more complex features and go backwards
    if(not double_corners.empty())
    {
        const auto &[c1, c2, c3] = double_corners.front();
        QSizeF size(points_dist(c1.p1(), c3.p1()), points_dist(c3.p1(), c2.p1()));
        QPointF center = c1.p1() + (c2.p1()-c1.p1())/2.0;
        QLineF tmp(c1); tmp.setAngle(tmp.angle()-45);
        float rot = QLineF(QPointF(0.0, 0.0), QPointF(0.0, 500.0)).angleTo(tmp);
        room.update(size, Eigen::Vector2f(center.x(), center.y()), rot);
        room.draw_on_2D_tab(&widget_2d->scene, {});
    }
    else if(not corners.empty())
    {
        float total_torque = 0.f;
        std::vector<std::tuple<QPointF, QPointF, int>> temp;
        for(const auto &c : corners)
        {
            // get closest corner in model c*
            auto const &corner = Eigen::Vector2f(c.p1().x(), c.p1().y());
            auto closest = room.get_closest_corner(corner);
            // project p on line joining c* with center of model
            auto line = Eigen::ParametrizedLine<float, 2>::Through(room.center, closest);
            auto a = room.to_local_coor(closest);
            auto b = room.to_local_coor(corner);
            float angle = atan2( a.x()*b.y() - a.y()*b.x(), a.x()*b.x() + a.y()*b.y() );  // sin / cos
            int signo;
            if(angle >= 0) signo = 1; else signo = -1;
            auto proj = line.projection(corner);
            // compute torque as distance to center times projection length
            total_torque += signo */* ((room_center - proj).norm() +*/ (proj - corner).norm();
            temp.emplace_back(std::make_tuple(QPointF(corner.x(), corner.y()), QPointF(proj.x(), proj.y()), signo));
        }

        // rotate the model
        float inertia = 10.f;
        float inc = total_torque / 1000.f / inertia;
        room.rotate(inc);
        // if there is still error
        // translate the model
        room.draw_on_2D_tab(&widget_2d->scene, temp);
    }
    else if(not par_lines.empty())
    {
        // get closest lines to model l1* y l2*
        for(const auto &[l1, l2] : par_lines)
        {
            QLineF closest = room.get_closest_side(l1);
            // compute angle between model lines and l1,l2
            float ang = closest.angleTo(l1);
            // rotate the models
            room.rotate(ang/10.f);
            // compute distance to lines
            // translate the model
        }
    }
    else if(not elines.empty())
    {
        // get closest line un model l*
        // compute angle between line and l*
        // compute torque as angle difference
        // rotate the model
        // compute distance to line
        // translate the model
    }

    // draw on 2D
    draw_on_2D_tab(elines);
    //draw_on_2D_tab(corners, "blue");
    draw_on_2D_tab(double_corners, "cyan");
    //draw_on_2D_tab(ml_points, "green", 60, true);
    //draw_on_2D_tab(std::vector<Eigen::Vector2f>{room_center}, "red", 140, true);
    //draw_on_2D_tab(yolo_objects);

    grid_viewer->viewport()->repaint();
    cv::imshow("Top_camera", rgb_head);
    cv::waitKey(5);

    fps.print("FPS: ", [this](auto x){ graph_viewer->set_external_hz(x);});
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::create_image_for_learning(const std::vector<Eigen::Vector2f> &floor_line_cart, float estimated_size) const
{
    // create image for function synthesis
    static int frame_counter = 0;
    cv::Mat synth(cv::Size(512,512), CV_8UC1, 0);
    for(const auto &p : floor_line_cart)
    {
        // trans p to image coordinates
        int row = (int)(255.0/estimated_size*p.y() + 128);
        int col = (int)(p.y()*255.0/estimated_size + 128);
        synth.at<uchar>(row, col) = 255;
    }
    cv::imwrite("frame_" + to_string(frame_counter) + ".png", synth);
}
vector<tuple<QLineF, QLineF, QLineF>> SpecificWorker::get_double_corners(float estimated_size, const vector<QLineF> &corners)
{
    vector<tuple<QLineF, QLineF, QLineF>> double_corners;
    // if in opposite directions and separation within room_size estimations
    for(auto &&comb: iter::combinations(corners, 2))
    {
        const auto &c1 = comb[0];
        const auto &c2 = comb[1];
        float ang = fabs(qDegreesToRadians(c1.angleTo(c2)));
        if( ang > M_PI-0.2 and ang < M_PI+0.2 and points_dist(c1.p1(), c2.p2()) > estimated_size / 2.0)
        {
            auto c11 = c1; c11.setAngle(c11.angle()+45);
            auto c22 = c2; c22.setAngle(c22.angle()-45);
            QPointF intersection;
            c11.intersect(c22, &intersection);
            // rep corner as the vector pointing outward through the bisectriz
            auto pl = get_distant(intersection, c11.p1(), c11.p2());
            auto pr = get_distant(intersection, c22.p1(), c22.p2());
            auto tip = QLineF(intersection, pl).pointAt(0.1) + (QLineF(intersection, pr).pointAt(0.1) - QLineF(intersection, pl).pointAt(0.1))/2.0;
            double_corners.push_back(make_tuple(c1, c2, QLineF(intersection, tip)));
        }
    }
    return double_corners;
}
std::vector<QLineF> SpecificWorker::get_corners(vector<pair<int, QLineF>> &elines)
{
    std::vector<QLineF> corners;
    for(auto &&comb: iter::combinations_with_replacement(elines, 2))
    {
        auto &[votes_1, line1] = comb[0];
        auto &[votes_2, line2] = comb[1];
        float angle = fabs(qDegreesToRadians(line1.angleTo(line2)));
        if(angle> M_PI) angle -= M_PI;
        float delta = 0.2;
        QPointF intersection;
        if(angle < M_PI/2+delta and angle > M_PI/2-delta  and line1.intersect(line2, &intersection) == 1)
        {
            // rep corner as the vector pointing outward through the bisectriz
            auto pl = get_distant(intersection, line1.p1(), line1.p2());
            auto pr = get_distant(intersection, line2.p1(), line2.p2());
            auto tip = QLineF(intersection, pl).pointAt(0.1) + (QLineF(intersection, pr).pointAt(0.1) - QLineF(intersection, pl).pointAt(0.1))/2.0;
            corners.emplace_back(QLineF(intersection, tip));
        }
    }
    return corners;
}
std::vector<std::pair<QLineF, QLineF>> SpecificWorker::get_parallel_lines(const  std::vector<pair<int, QLineF>> &lines, float estimated_size)
{
    std::vector<std::pair<QLineF, QLineF>> par_lines;
    for(auto &&line_pairs : iter::combinations(lines, 2))
    {
        const auto &[v1, line1] = line_pairs[0];
        const auto &[v2, line2] = line_pairs[1];
        float ang = fabs(line1.angleTo(line2));
        const float delta = 10;  //degrees
        if((ang > -delta and ang < delta) or ( ang > 180-delta and ang < 180+delta))
            if( points_dist(line1.center(), line2.center()) > estimated_size/2.0)
                par_lines.push_back(std::make_pair(line1,  line2));
    }
    return par_lines;
}
std::vector<pair<int, QLineF>> SpecificWorker::get_hough_lines(vector<Eigen::Vector2f> &floor_line_cart) const
{
    vector<cv::Vec2f> floor_line_cv;
    for(const auto &p : floor_line_cart)
        floor_line_cv.emplace_back(cv::Vec2f(p.x(), p.y()));
    double rhoMin = 0.0f, rhoMax = 5000.0f, rhoStep = 5;
    double thetaMin = -CV_PI, thetaMax = CV_PI, thetaStep = CV_PI / 180.0f;
    cv::Mat lines;
    HoughLinesPointSet(floor_line_cv, lines, 10, 1, rhoMin, rhoMax, rhoStep, thetaMin, thetaMax, thetaStep);
    vector<cv::Vec3d> lines3d;
    lines.copyTo(lines3d);

    // compute lines from Hough params
    vector<pair<int, QLineF>> elines;
    for(auto &l: ranges::filter_view(lines3d, [](auto a){return a[0]>10;}))
    {
        float rho = l[1], theta = l[2];
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        QPointF p1(x0 + 5000 * (-b), y0 + 5000 * (a));
        QPointF p2(x0 - 5000 * (-b), y0 - 5000 * (a));
        elines.emplace_back(make_pair(l[0], QLineF(p1, p2)));
    }

    // Non-Maximum Suppression of close parallel lines
    vector<QLineF> to_delete;
    for(auto &&comb: iter::combinations(elines, 2))
    {
        auto &[votes_1, line1] = comb[0];
        auto &[votes_2, line2] = comb[1];
        float angle = qDegreesToRadians(line1.angle(line2));
        float dist = (line1.center() - line2.center()).manhattanLength();
        float delta = 0.3;
        if( fabs(angle) < delta and dist < 700)
        {
            if (votes_1 >= votes_2) to_delete.push_back(line2);
            else to_delete.push_back(line1);
        }
    }
    elines.erase(remove_if(elines.begin(), elines.end(), [to_delete](auto l){ return ranges::find(to_delete, std::get<1>(l)) != to_delete.end();}), elines.end());
    return elines;
}
float SpecificWorker::get_size(const Eigen::Vector2f &room_center, vector<Eigen::Vector2f> &floor_line_cart) const
{
    Eigen::MatrixX2f zero_mean_points(floor_line_cart.size(), 2);
    for(const auto &[i, p] : iter::enumerate(floor_line_cart))
        zero_mean_points.row(i) = p - room_center;
    auto max_room_size = zero_mean_points.colwise().maxCoeff();
    auto min_room_size = zero_mean_points.colwise().minCoeff();
    return max_room_size[0] - min_room_size[0];
}
Eigen::Vector2f SpecificWorker::get_mean(vector<Eigen::Vector2f> &floor_line_cart) const
{
    Eigen::Vector2f room_center{0.0, 0.0};
    room_center = accumulate(floor_line_cart.begin(), floor_line_cart.end(), room_center) / (float)floor_line_cart.size();
    return room_center;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////77
RoboCompYoloObjects::TObjects SpecificWorker::get_yolo_objects()
{
    try
    {
        auto yolo_data = yoloobjects_proxy->getYoloObjects();/**/
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
            auto cxi = o.left + ((o.right-o.left)/2);
            auto cyi = o.top + ((o.bot-o.top)/2);
            auto cx = cxi - cam_head_api->get_depth_width()/2;
            auto cy = cyi - cam_head_api->get_depth_height()/2;
            auto x = cx * o.depth / cam_head_api->get_depth_focal_x();
            auto z = cy * o.depth / cam_head_api->get_depth_focal_y();
            auto y = o.depth;
//            if depth is distance along ray
//            auto x = cx * o.depth / sqrt(cx*cx + cam_head_api->get_depth_focal_x()*cam_head_api->get_depth_focal_x());
//            auto z = cy * o.depth / sqrt(cy*cy + cam_head_api->get_depth_focal_y()*cam_head_api->get_depth_focal_y());
//            auto proy = sqrt(o.depth*o.depth-z*z);
//            auto y = sqrt(x*x+proy*proy);
            if(o.type==0)
                qInfo() << x << y ;

            if(auto object_in_robot = inner_eigen->transform(robot_name, Eigen::Vector3d(x, y, z) , shadow_camera_head_name); object_in_robot.has_value())
            {
                //auto p = widget_2d->scene.addPixmap(object_pixmaps[o.type]);
                auto p = widget_2d->scene.addRect(-200, -200, 400, 400, QPen(QColor("blue"), 30));
                //p->setPos(object_in_robot.value().x() - object_pixmaps[o.type].width() / 2, object_in_robot.value().y() - object_pixmaps[o.type].height() / 2);

                p->setPos(object_in_robot.value().x(), object_in_robot.value().y());
                //p->setPos(o.x - object_pixmaps[o.type].width() / 2, o.y - object_pixmaps[o.type].height() / 2);
                pixmap_ptrs.push_back(p);
            }
        }
}
void SpecificWorker::draw_on_2D_tab(const std::vector<Eigen::Vector2f> &lines, QString color, int size, bool clean)
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
    for(const auto &point: lines)
    {
        auto p = widget_2d->scene.addRect(-size/2, -size/2, size, size, QPen(QColor(color), 20));
        dot_vec.push_back(p);
        p->setPos(point.x(), point.y());
    }
}
void SpecificWorker::draw_on_2D_tab(const std::vector<std::vector<Eigen::Vector2f>> &lines, QString color, int size, bool clean)
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
    static QStringList my_color = {"red", "orange", "blue", "magenta", "black", "yellow", "brown", "cyan"};
    for(auto &&[k, line]: lines | iter::enumerate)
        for(const auto &points: line)
        {
            auto p = widget_2d->scene.addRect(-size/2, -size/2, size, size, QPen(QColor(my_color.at(k)), 20));
            dot_vec.push_back(p);
            p->setPos(points.x(), points.y());
        }
}
void SpecificWorker::draw_on_2D_tab(const std::vector<std::pair<int, QLineF>> &lines)
{
    static std::vector<QGraphicsItem*> lines_vec;
    for (auto l: lines_vec)
    {
        widget_2d->scene.removeItem(l);
        delete l;
    }
    lines_vec.clear();

    for(const auto &l : lines)
    {
        auto p = widget_2d->scene.addLine(l.second, QPen(QColor("orange"), 40));
        lines_vec.push_back(p);
    }
}
void SpecificWorker::draw_on_2D_tab(const std::vector<QLineF> &corners, QString color, int size, bool clean)
{
    static std::vector<QGraphicsItem*> lines_vec;
    for (auto l: lines_vec)
    {
        widget_2d->scene.removeItem(l);
        delete l;
    }
    lines_vec.clear();

    for(const auto &l : corners)
    {
        auto up = l;
        up.setAngle(up.angle()+45);
        auto down = l;
        down.setAngle(down.angle()-45);
        auto upv = widget_2d->scene.addLine(up, QPen(QColor("magenta"), 30));
        auto downv = widget_2d->scene.addLine(down, QPen(QColor("magenta"), 30));
        lines_vec.push_back(upv);
        lines_vec.push_back(downv);
        auto p = widget_2d->scene.addLine(l, QPen(QColor("magenta"), 40));
        lines_vec.push_back(p);
    }
}
void SpecificWorker::draw_on_2D_tab(const std::vector<std::tuple<QLineF, QLineF, QLineF>> &double_corners, QString color, int size, bool clean)
{
    static std::vector<QGraphicsItem*> lines_vec;
    for (auto l: lines_vec)
    {
        widget_2d->scene.removeItem(l);
        delete l;
    }
    lines_vec.clear();

    for(const auto &[first, second, third] : double_corners)
    {
        auto p = widget_2d->scene.addLine(QLineF(first.p1(), second.p1()), QPen(QColor(color), 60));
        auto pp = widget_2d->scene.addEllipse(-100, -100, 200, 200, QPen(QColor(color), 60), QBrush(QColor(color)));
        pp->setPos(third.p1());
        lines_vec.push_back(pp);
        lines_vec.push_back(p);
        break;
    }
}
void SpecificWorker::draw_on_2D_tab(const Room &room, QString color)
{
    static QGraphicsItem* item;
    widget_2d->scene.removeItem(item);
    delete item;
    QColor col(color);
    col.setAlpha(30);
    auto size = room.rsize;
    item = widget_2d->scene.addRect(-size.height()/2, -size.width()/2, size.height(), size.width(), QPen(QColor(col), 60), QBrush(QColor(col)));
    item->setPos(room.center.x(), room.center.y());
    item->setRotation(-room.rot);
    item->setZValue(1);
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


std::vector<std::vector<Eigen::Vector2f>> SpecificWorker::get_multi_level_3d_points(const cv::Mat &depth_frame, const cv::Mat &rgb_frame)   // in meters
/**/{
    std::vector<std::vector<Eigen::Vector2f>> points(int((2000-400)/200));  //height steps
    for(auto &p: points)
        p.resize(360, Eigen::Vector2f(consts.max_camera_depth_range, consts.max_camera_depth_range));   // angular resolution

    int semi_height = depth_frame.rows/2;
    float hor_ang, dist, x, y, z, proy;
    float ang_slope = 2*M_PI/depth_frame.cols;
    const float ang_bin = 2.0*M_PI/consts.num_angular_bins;

    for(int u=0; u<depth_frame.rows; u++)
        for(int v=0; v<depth_frame.cols; v++)
        {
            hor_ang = ang_slope * v - M_PI; // cols to radians
            dist = depth_frame.ptr<float>(u)[v] * consts.scaling_factor;  // pixel to dist scaling factor  -> to mm
            if(dist > consts.max_camera_depth_range) continue;
            if(dist < consts.min_camera_depth_range) continue;
            //dist /= 1000.f; // to meters
            x = -dist * sin(hor_ang);
            y = dist * cos(hor_ang);
            proy = dist * cos( atan2((semi_height - u), 128.f));
            z = (semi_height - u)/128.f * proy; // 128 focal as PI fov angle for 256 pixels
            z += consts.omni_camera_height; // get from DSR
            // add Y axis displacement
            if(z < 400) continue; // filter out floor

            //auto less_than = [](auto a, auto b){ auto &[x, y, z]=a; auto &[i, j, k]=b; return x*x+y*y+z*z < i*i+j*j+k*k;};

            for(auto &&[level, step] : iter::range(400, 2000, 200) | iter::enumerate)
            if(z > step and z < step+200)
            {
                int ang_index = floor((M_PI + atan2(x, y)) / ang_bin);
                Eigen::Vector2f new_point(x, y);
                if(new_point.norm() <  points[level][ang_index].norm())
                    points[level][ang_index] = new_point;
            }
        };

    return points;
}
std::vector<Point3f> SpecificWorker::get_omni_3d_points(const cv::Mat &depth_frame, const cv::Mat &rgb_frame)   // in meters
{
     // Let's assume that each column corresponds to a polar coordinate: ang_step = 360/image.width
     // and along the column we have radius
     // hor ang = 2PI/512 * i

    std::vector<Point3f> points(depth_frame.rows * depth_frame.cols);
    int semi_height = depth_frame.rows/2;
    float hor_ang, dist, x, y, z, proy;
    float ang_slope = 2*M_PI/depth_frame.cols;

    std::size_t i = 0;  // direct access to points
    for(int u=0; u<depth_frame.rows; u++)
        for(int v=0; v<depth_frame.cols; v++)
        {
            hor_ang = ang_slope * v - M_PI; // cols to radians
            dist = depth_frame.ptr<float>(u)[v] * consts.scaling_factor;  // pixel to dist scaling factor  -> to mm
            if(dist > consts.max_camera_depth_range) continue;
            if(dist < consts.min_camera_depth_range) continue;
            dist /= 1000.f; // to meters
            x = -dist * sin(hor_ang);
            y = dist * cos(hor_ang);
            proy = dist * cos( atan2((semi_height - u), 128.f));
            z = (semi_height - u)/128.f * proy; // 128 focal as PI fov angle for 256 pixels
            z += consts.omni_camera_height_meters; // get from DSR
            if(z < 0.4) continue; // filter out floor
            points[i] = std::make_tuple(x, y, z);
            //auto rgb = rgb_frame.ptr<cv::Vec3b>(u)[v];
            //colors->operator[](i) = std::make_tuple(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0);
            i += 1;/**/
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
vector<Eigen::Vector2f> SpecificWorker::compute_floor_line(const std::vector<Eigen::Vector2f> &line, bool draw)  // meters
{
    vector<Eigen::Vector2f> floor_line;  //polar
    const double ang_bin = 2.0*M_PI/360;
    float ang= -ang_bin;
    for(const auto &p: line)
    {
        ang += ang_bin;
        float dist = p.norm();
        if(fabs(dist) > consts.max_camera_depth_range/1000) continue;
        floor_line.emplace_back(Eigen::Vector2f(ang-M_PI, p.norm()*1000.0));  // to undo the +M_PI in group_by_angular_sectors
    }
    //qInfo() << __FUNCTION__ << "--------------------------------------";
    return floor_line;
}
//vector<Eigen::Vector2f> SpecificWorker::compute_floor_line(const SpecificWorker::SetsType &sets, bool draw)  // meters
//{
//    vector<Eigen::Vector2f> floor_line;  //polar
//    const double ang_bin = 2.0*M_PI/360;
//    float ang= -ang_bin;
//    for(const auto &s: sets)
//    {
//        ang += ang_bin;
//        if(s.empty()) continue;
//        Eigen::Vector3f p = get<Eigen::Vector3f>(*s.cbegin());  // first element is the smallest distance
//        float dist = p.head(2).norm();
//        if(fabs(dist) > consts.max_camera_depth_range/1000) continue;
//        floor_line.emplace_back(Eigen::Vector2f(ang-M_PI, p.head(2).norm()*1000.0));  // to undo the +M_PI in group_by_angular_sectors
//    }
//    //qInfo() << __FUNCTION__ << "--------------------------------------";
//    return floor_line;
//}

/////////////////////////////////////////////////////////////////////////////////////////777
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}

double SpecificWorker::points_dist(const QPointF &p1, const QPointF &p2)
{
    return sqrt((p1.x()-p2.x())*(p1.x()-p2.x())+(p1.y()-p2.y())*(p1.y()-p2.y()));
}

QPointF SpecificWorker::get_distant(const QPointF &p, const QPointF &p1, const QPointF &p2)
{
    if( (p-p1).manhattanLength() < (p-p2).manhattanLength()) return p2; else return p1;
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
