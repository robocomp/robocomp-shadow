/*
 *    Copyright (C) 2021 by YOUR NAME HERE
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
#include <cppitertools/zip.hpp>
#include <cppitertools/range.hpp>
#include <cppitertools/sliding_window.hpp>
#include <unordered_map>

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
	tree_view = params["tree_view"].value == "true";
	graph_view = params["graph_view"].value == "true";
	qscene_2d_view = params["2d_view"].value == "true";
	osg_3d_view = params["3d_view"].value == "true";

	return true;
}

void SpecificWorker::initialize(int period)
{
	std::cout << "Initialize worker" << std::endl;
	this->Period = 80;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
		timer.start(Period);

        // create graph
		G = std::make_shared<DSR::DSRGraph>(0, agent_name, agent_id, ""); // Init nodes
		std::cout<< __FUNCTION__ << "Graph loaded" << std::endl;

		//dsr update signals
		connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::add_or_assign_node_slot);
		connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::add_or_assign_edge_slot);
//		connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_attrs_slot);
//		connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
		connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

		// Graph viewer
		using opts = DSR::DSRViewer::view;
		int current_opts = 0;
		opts main = opts::none;
		if(tree_view)
		    current_opts = current_opts | opts::tree;
		if(graph_view)
		{
		    current_opts = current_opts | opts::graph;
		    //main = opts::graph;
		}
		if(qscene_2d_view)
		    current_opts = current_opts | opts::scene;
		if(osg_3d_view)
		    current_opts = current_opts | opts::osg;

        main = opts::graph;
		graph_viewer = std::make_unique<DSR::DSRViewer>(this, G, current_opts, main);
        setWindowTitle(QString::fromStdString(agent_name + "-") + QString::number(agent_id));

        // 2D widget
        widget_2d = qobject_cast<DSR::QScene2dViewer *>(graph_viewer->get_widget(opts::scene));
        widget_2d->set_draw_laser(true);

        // custom widget
        graph_viewer->add_custom_widget_to_dock("Giraff Plan Controller", &custom_widget);
        connect(custom_widget.pushButton_start_mission, SIGNAL(clicked()), this, SLOT(slot_start_mission()));
        connect(custom_widget.pushButton_stop_mission, SIGNAL(clicked()), this, SLOT(slot_stop_mission()));
        connect(custom_widget.pushButton_cancel_mission, SIGNAL(clicked()), this, SLOT(slot_cancel_mission()));
        custom_widget.layout()->addWidget(widget_2d);
        custom_widget.raise();

        // List of missions
        custom_widget.list_plan->addItem("Select a mission");
        custom_widget.list_plan->addItem("Goto x");
        custom_widget.list_plan->addItem("Path p"); //cargar un plan con diferentes puntos y más adelante, más planes
        custom_widget.list_plan->addItem("Bouncer");
        custom_widget.list_plan->addItem("Follow people");
        connect(custom_widget.list_plan, SIGNAL(currentIndexChanged(int)), this , SLOT(slot_change_mission_selector(int)));

       // List of people
       auto people_nodes = G->get_nodes_by_type("person");
       for (auto person : people_nodes)
       {
           QString str = QString::fromUtf8( person.name().data(), person.name().size() );
           custom_widget.object_list->addItem(str);
       }


        // stacked widgets
        point_dialog.setupUi(custom_widget.stacked_widget->widget(0));
        pathfollow_dialog.setupUi(custom_widget.stacked_widget->widget(1));
        auto empty_widget = new QWidget();
        custom_widget.stacked_widget->addWidget(empty_widget);
        custom_widget.stacked_widget->setCurrentIndex(2);
        QRectF dimensions;
        auto world_node = G->get_node(world_name).value();
        dimensions.setLeft(G->get_attrib_by_name<OuterRegionLeft_att>(world_node).value());
        dimensions.setTop(G->get_attrib_by_name<OuterRegionTop_att>(world_node).value());
        dimensions.setRight(G->get_attrib_by_name<OuterRegionRight_att>(world_node).value());
        dimensions.setBottom(G->get_attrib_by_name<OuterRegionBottom_att>(world_node).value());
        pathfollow_draw_widget = new AbstractGraphicViewer(pathfollow_dialog.test_widget, dimensions);


        // get camera_api
        if(auto cam_node = G->get_node(giraff_camera_realsense_name); cam_node.has_value())
            cam_api = G->get_camera_api(cam_node.value());
        else
        {
            std::cout << "Controller-DSR terminate: could not find a camera node named " << giraff_camera_usb_name << std::endl;

            std::terminate();
        }

        // Inner Api
        inner_eigen = G->get_inner_eigen_api();

        // Robot polygon
        if(auto robot_body = G->get_node(robot_body_name); robot_body.has_value())
        {
            auto width = G->get_attrib_by_name<width_att>(robot_body.value());
            auto height = G->get_attrib_by_name<depth_att>(robot_body.value());
            cout << "Width " << width.value() << " height " << height.value() << endl;
            if (width.has_value() and height.has_value())
            {
                robot_polygon << QPointF(-width.value() / 2, -height.value() / 2)
                              << QPointF(-width.value() / 2, height.value() / 2)
                              << QPointF(width.value() / 2, height.value() / 2)
                              << QPointF(width.value() / 2, -height.value() / 2);
            } else
            {
                std::cout << __FUNCTION__ << " No robot body width or depth found. Terminating..." << std::endl;
                std::terminate();
            }
        }
        else
        {
            std::cout << __FUNCTION__ << " No robot body found. Terminating..." << std::endl;
            std::terminate();
        }

        // trail
        auto robot_pos = inner_eigen->transform(world_name, robot_name).value();
        last_point = QPointF(robot_pos.x(), robot_pos.y());
        connect( custom_widget.path_trail_button, SIGNAL(toggled(bool)), this, SLOT(trace_button_slot(bool)));

        // Eigen format
        OctaveFormat = Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
        CommaInitFmt = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
        hide();
        this->Period = 100;
		timer.start(Period);
	}
}

void SpecificWorker::compute()
{
    static std::chrono::steady_clock::time_point begin, lastPathStep;
    static QGraphicsEllipseItem *target_scene;

    people_checker();

    // check for existing missions
    if (auto plan_o = plan_buffer.try_get(); plan_o.has_value())
    {
        current_plan = plan_o.value();
        std::cout << __FUNCTION__  << " New plan arrived: " << std::endl;
        std::cout << current_plan.pprint() << std::endl;
        custom_widget.textedit_current_plan->appendPlainText("-> compute: initiating plan " + current_plan.get_action());
        current_plan.set_running();
    }

    auto robot_pos = inner_eigen->transform(world_name, robot_name).value();
    float coordX = point_dialog.goto_spinbox_coordX->value();
    float coordY = point_dialog.goto_spinbox_coordY->value();
    float umbral = 200.0;
    auto robot_node = G->get_node(robot_name);
    float adv = G->get_attrib_by_name<robot_ref_adv_speed_att>(robot_node.value()).value_or(0);
    float rot = G->get_attrib_by_name<robot_ref_rot_speed_att>(robot_node.value()).value_or(0);

    if (current_plan.is_running())
    {
        qInfo() << __FUNCTION__ << " Plan is running...";

        if (current_plan.get_action() == "GOTO")
        {
            float mod = sqrt(pow((coordX - robot_pos.x()), 2) + pow((coordY - robot_pos.y()), 2));
            if (mod < umbral && (adv == 0.0 && rot == 0.0))
            {
                current_plan.is_finished();
                slot_stop_mission();
                qInfo() << __FUNCTION__ << " Plan finished!";
            }
        }
        if(current_plan.get_action()=="FOLLOW_PEOPLE")
        {
//            auto start = std::chrono::steady_clock::now();
//            auto duration=(start-begin) / 1ms ;
//            // std::cout << "DURATION: " << duration << std::endl;
//            if(duration > 400){
//                if(auto intention= G->get_node(current_intention_name);intention.has_value()){
//                    if(auto path=G->get_node(current_path_name);path.has_value()) {
//                        auto x_values = G->get_attrib_by_name<path_x_values_att>(path.value());
//                        auto y_values = G->get_attrib_by_name<path_y_values_att>(path.value());
//                        if (x_values.has_value() and y_values.has_value()) {
//                            if (auto people = G->get_nodes_by_type(person_type_name);not people.empty())
//                            {
//                                for(auto person : people)
//                                {
//                                    if (auto followed_node = G->get_attrib_by_name<followed_att>(person); followed_node.has_value() && followed_node.value() == true)
//                                    {
////                                        auto last_person_pose = inner_eigen->transform(world_name,person.name(), );
//                                        auto person_pose = inner_eigen->transform(world_name,person.name()).value();
//
//                                        auto x = x_values.value().get();
//                                        auto y = y_values.value().get();
//                                        cout<<x.size()<<endl;
//                                        cout << "PERSON POSE: " << person_pose[0] << " " << person_pose[1] << endl;
//
//
//
//                                        if(sqrt(pow(x.back()-person_pose.x(),2)+ pow(y.back()-person_pose.y(),2))>500){
//
//                                            auto robot_pose = inner_eigen->transform(world_name, robot_name).value();
//                                            Mat::Vector2d pp(int(x.back()), int(y.back()));
//                                            Mat::Vector2d hp(int(person_pose.x()),int(person_pose.y()));
//
//                                            Mat::Vector2d point;
//
//                                            float segmento = (hp - pp).norm() / 150;
//                                            auto vector_dir = (hp - pp) / segmento;
//
//                                            for (int i = 1; i < int(segmento); ++i) {
//                                                auto point = pp + (vector_dir * i);
//                                                x.push_back(point.x());
//                                                y.push_back(point.y());
//                                            }
//
//                                            x.push_back(person_pose[0]);
//                                            y.push_back(person_pose[1]);
//
//                                            G->add_or_modify_attrib_local<path_x_values_att>(path.value(), x);
//                                            G->add_or_modify_attrib_local<path_target_x_att>(path.value(),float(person_pose[0]));
//                                            G->add_or_modify_attrib_local<path_y_values_att>(path.value(), y);
//                                            G->add_or_modify_attrib_local<path_target_y_att>(path.value(),float(person_pose[1]));
//                                            G->update_node(path.value());
//                                        }
//                                    }
//                                }
//                            }
//                        }
//                    }
////                        if (not temporary_plan.is_valid()) //resetea el valor de la y cuando pones la x
////                            temporary_plan.new_plan(Plan::Actions::FOLLOW_PEOPLE);
////                        temporary_plan.insert_attribute("x",int(person_pose[0]));
////                        temporary_plan.insert_attribute("y",int(person_pose[1]));
////                        temporary_plan.insert_attribute("destiny", "floor");
////                            if (target_scene != nullptr)
////                                widget_2d->scene.removeItem(target_scene);
////                            target_scene = widget_2d->scene.addEllipse(-50, -50, 100, 100, QPen(QColor("Orange")),
////                                                                       QBrush(QColor("Orange")));
////                            target_scene->setPos(person_pose[0], person_pose[1]);
////                            target_scene->setZValue(100);
//                    }
//                begin=std::chrono::steady_clock::now();
//                  }

        }

    }
    else
    { // there should be a plan after a few seconds
    }
//    read_camera();
    // path trail
//    if(custom_widget.path_trail_button->isChecked())
//    {
//        auto robot_pos = inner_eigen->transform(world_name, robot_name).value();
//        QLineF line(last_point.x(), last_point.y(), robot_pos.x(), robot_pos.y());
//        lines.push_back(widget_2d->scene.addLine(line, QPen(QColor("Blue"),40)));
//        last_point = QPointF(robot_pos.x(), robot_pos.y());
//    }
}

////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::read_camera()
{
    static QPixmap pix;
    if (auto vframe_t = virtual_camera_buffer.try_get(); vframe_t.has_value())
    {
        auto vframe = cv::Mat(cam_api->get_height(), cam_api->get_width(), CV_8UC3, vframe_t.value().data());
        pix = QPixmap::fromImage(QImage(vframe.data, vframe.cols, vframe.rows, QImage::Format_RGB888));
        // custom_widget.label_rgb->setPixmap(pix);
    }
//    else if (!custom_widget.image_onoff_button->isChecked())
//        custom_widget.label_rgb->clear();

}

////////////////////////////////////////////////////////////////////////////////////////////
/// Mission creation methods
////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::create_bouncer_mission()
{
    custom_widget.stacked_widget->setCurrentIndex(2);
    temporary_plan.new_plan(Plan::Actions::BOUNCE);
    custom_widget.textedit_current_plan->appendPlainText("-> New temporary plan: BOUNCE");
    custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));
}
void SpecificWorker::create_path_mission()
{
    custom_widget.stacked_widget->setCurrentIndex(1);
    temporary_plan.new_plan(Plan::Actions::FOLLOW_PATH);
    custom_widget.textedit_current_plan->appendPlainText("-> New temporary plan: FOLLOW_PATH");
    custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));

    auto draw_circle = [this]()
            {
                // remove current drawing
                std::vector<Eigen::Vector2f> fake_path;
                draw_path(fake_path, &pathfollow_draw_widget->scene, true);
                draw_path(fake_path, &widget_2d->scene, true);

                const float radio = pathfollow_dialog.circle_radius_slider->value();
                const float arco = 200;  // get from robot size
                auto robot_pose = inner_eigen->transform(world_name, robot_name).value();
                temporary_plan.x_path.clear();
                temporary_plan.y_path.clear();
                std::vector<Eigen::Vector2f> local_path;
                for(auto &&alfa : iter::range(0.0, 2*M_PI, (double)(arco/radio)))
                {
                    float x = radio * cos(alfa) + robot_pose.x();
                    float y = radio * sin(alfa) + robot_pose.y();
                    temporary_plan.x_path.push_back(x);
                    temporary_plan.y_path.push_back(y);
                    local_path.emplace_back(Eigen::Vector2f(x, y));
                }
                draw_path(local_path, &pathfollow_draw_widget->scene);
            };
    //draw_circle(); // it starts with the circle button selected
    auto draw_oval = [this]()
    {
        // remove curent drawing
        std::vector<Eigen::Vector2f> fake_path;
        draw_path(fake_path, &pathfollow_draw_widget->scene, true);
        draw_path(fake_path, &widget_2d->scene, true);
        auto robot_pose = inner_eigen->transform(world_name, robot_name).value();

        const float long_radio = pathfollow_dialog.oval_long_radius_slider->value();
        const float short_radio = pathfollow_dialog.oval_short_radius_slider->value();
        const float arco = 200;  // get from robot size
        temporary_plan.x_path.clear();
        temporary_plan.y_path.clear();
        std::vector<Eigen::Vector2f> local_path;  // for drawing
        // start drawing a line from center to the right
        Eigen::ParametrizedLine<float, 2> line(Eigen::Vector2f(0.f, short_radio), Eigen::Vector2f(1.f,0.f));
        for(auto &&t: iter::range(0.f, long_radio/2, arco))
        {
            auto p = line.pointAt(t);
            temporary_plan.x_path.push_back(p.x());
            temporary_plan.y_path.push_back(p.y());
            local_path.emplace_back(p);
        }
        for(auto &&alfa : iter::range(M_PI/2.0, -M_PI/2.0, -(double)(arco/short_radio)))
        {
            float x = short_radio * cos(alfa) + long_radio/2 + robot_pose.x();
            float y = short_radio * sin(alfa) + robot_pose.y();
            temporary_plan.x_path.push_back(x);
            temporary_plan.y_path.push_back(y);
            local_path.emplace_back(Eigen::Vector2f(x, y));
        }
        line = Eigen::ParametrizedLine<float, 2>(Eigen::Vector2f(long_radio/2, -short_radio), Eigen::Vector2f(-1.0, 0.0));
        for(auto &&t: iter::range(0.f, long_radio, arco))
        {
            auto p = line.pointAt(t);
            temporary_plan.x_path.push_back(p.x());
            temporary_plan.y_path.push_back(p.y());
            local_path.emplace_back(p);
        }
        for(auto &&alfa : iter::range(3*M_PI/2.0, M_PI/2, -(double)(arco/short_radio)))
        {
            float x = short_radio * cos(alfa) - long_radio/2;
            float y = short_radio * sin(alfa);
            temporary_plan.x_path.push_back(x);
            temporary_plan.y_path.push_back(y);
            local_path.emplace_back(Eigen::Vector2f(x, y));
        }
        line = Eigen::ParametrizedLine<float, 2>(Eigen::Vector2f(-long_radio/2, short_radio), Eigen::Vector2f(1.f, 0.0));
        for(auto &&t: iter::range(0.f, long_radio, arco))
        {
            auto p = line.pointAt(t);
            temporary_plan.x_path.push_back(p.x());
            temporary_plan.y_path.push_back(p.y());
            local_path.emplace_back(p);
        }
        draw_path(local_path, &pathfollow_draw_widget->scene);
    };
    //draw_oval();

    connect(pathfollow_dialog.button_group, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked),
            [this, draw_circle, draw_oval](QAbstractButton *button)
            {
                if(button == pathfollow_dialog.circle_radio_button)
                {
                    qInfo() << __FUNCTION__ << " circle selected";
                    draw_circle();
                }
                else if(button == pathfollow_dialog.oval_radio_button)
                {
                    qInfo() << __FUNCTION__ << " oval selected";
                    draw_oval();
                }
            });

    connect(pathfollow_dialog.circle_radius_slider, qOverload<int>(&QSlider::valueChanged),[draw_circle](int v)
    {
        draw_circle();
    });

    connect(pathfollow_dialog.oval_short_radius_slider, qOverload<int>(&QSlider::valueChanged),[draw_oval](int v)
    {
        draw_oval();
    });
    connect(pathfollow_dialog.oval_long_radius_slider, qOverload<int>(&QSlider::valueChanged),[draw_oval](int v)
    {
        draw_oval();
    });
}
void SpecificWorker::create_goto_mission() {
    static QGraphicsEllipseItem *target_scene;
    custom_widget.stacked_widget->setCurrentIndex(0);
    temporary_plan.new_plan(Plan::Actions::GOTO);
    custom_widget.textedit_current_plan->appendPlainText("-> New temporary plan: GOTO");
    custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));

    connect(point_dialog.goto_spinbox_coordX, qOverload<int>(&QSpinBox::valueChanged), [this](int v) {
        if (not temporary_plan.is_valid()) //resetea el valor de la y cuando pones la x
            temporary_plan.new_plan(Plan::Actions::GOTO);

        temporary_plan.insert_attribute("x", v);
        custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'x' in GOTO plan");
        custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));
        // redraw target
        if (target_scene != nullptr)
            widget_2d->scene.removeItem(target_scene);
        target_scene = widget_2d->scene.addEllipse(-50, -50, 100, 100, QPen(QColor("Orange")),
                                                   QBrush(QColor("Orange")));
        target_scene->setPos(v, point_dialog.goto_spinbox_coordY->value());
        target_scene->setZValue(100);
    });

    connect(point_dialog.goto_spinbox_coordY, qOverload<int>(&QSpinBox::valueChanged), [this](int v) {
//        if (not temporary_plan.is_valid())
//            temporary_plan.new_plan(Plan::Actions::GOTO);

        temporary_plan.insert_attribute("y", v);
        custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'y' in GOTO plan");
        custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));
        //redraw target
        if (target_scene != nullptr)
            widget_2d->scene.removeItem(target_scene);
        target_scene = widget_2d->scene.addEllipse(-50, -50, 100, 100, QPen(QColor("Orange")),
                                                   QBrush(QColor("Orange")));
        target_scene->setPos(point_dialog.goto_spinbox_coordX->value(), v);
        target_scene->setZValue(100);
    });

    connect(point_dialog.goto_spinbox_angle, qOverload<int>(&QSpinBox::valueChanged), [this](int v) {
        temporary_plan.insert_attribute("destiny", v);
        custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'destiny' in GOTO plan");
        custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));
    });

    connect(widget_2d, &DSR::QScene2dViewer::mouse_right_click, [this](int x, int y, std::uint64_t obj) {
        if (not temporary_plan.is_valid())
            temporary_plan.new_plan(Plan::Actions::GOTO);

        if (auto node = G->get_node(obj); node.has_value())
            temporary_plan.insert_attribute("destiny", QString::fromStdString(node.value().name()));

        point_dialog.goto_spinbox_coordX->setValue(x);
        point_dialog.goto_spinbox_coordY->setValue(y);

        // redraw target
        if (target_scene != nullptr)
            widget_2d->scene.removeItem(target_scene);
        target_scene = widget_2d->scene.addEllipse(-50, -50, 100, 100, QPen(QColor("Orange")),
                                                   QBrush(QColor("Orange")));
        target_scene->setPos(x, y);
        target_scene->setZValue(100);

    });
}

void SpecificWorker::create_searching_person_mission() 
{
        static QGraphicsEllipseItem *target_scene;
        temporary_plan.new_plan(Plan::Actions::SEARCHING_PERSON);
        custom_widget.textedit_current_plan->appendPlainText("-> New temporary plan: SEARCHING_PERSON");
        custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));

        if (not temporary_plan.is_valid()) //resetea el valor de la y cuando pones la x
            temporary_plan.new_plan(Plan::Actions::SEARCHING_PERSON);

        temporary_plan.insert_attribute("x", QString::fromStdString(to_string(last_person_pos.x())));
        temporary_plan.insert_attribute("y", QString::fromStdString(to_string(last_person_pos.y())));
        if (target_scene != nullptr)
            widget_2d->scene.removeItem(target_scene);
}

// void SpecificWorker::create_talking_people_mission(uint64_t person_id)
// {
//     if(person_id != 0)
//     {
//         static QGraphicsEllipseItem *target_scene;
//         temporary_plan.new_plan(Plan::Actions::TALKING_WITH_PEOPLE);
//         custom_widget.textedit_current_plan->appendPlainText("-> New temporary plan: TALKING_WITH_PEOPLE");
//         custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));

//         if (not temporary_plan.is_valid()) //resetea el valor de la y cuando pones la x
//             temporary_plan.new_plan(Plan::Actions::TALKING_WITH_PEOPLE);
//         auto talk_person_node = G->get_node(person_id);
//         auto talk_person_node_id = talk_person_node->id();

//         std::ostringstream oss;
//         oss << talk_person_node_id;
//         std::string intAsString(oss.str());

//         temporary_plan.insert_attribute("person_node_id",  QString::fromStdString(intAsString));
//         if(auto person_name = G->get_attrib_by_name<person_name_att>(talk_person_node.value()); person_name.has_value())
//         {
//             temporary_plan.insert_attribute("person_name",  QString::fromStdString(person_name.value()));
//         }
//         custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'person_name' in Follow People plan");
//         if (target_scene != nullptr)
//             widget_2d->scene.removeItem(target_scene);
//     }
//     else
//     {
//         if(custom_widget.object_list->selectedItems().size() > 0)
//         {
//             static QGraphicsEllipseItem *target_scene;
//             custom_widget.stacked_widget->setCurrentIndex(3);
//             temporary_plan.new_plan(Plan::Actions::TALKING_WITH_PEOPLE);
//             custom_widget.textedit_current_plan->appendPlainText("-> New temporary plan: FOLLOW PEOPLE");
//             custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));
// //    basic_string<char> person_name = custom_widget.object_list->currentItem()->text().toUtf8().constData();
//             auto person_name = custom_widget.object_list->currentItem()->text();

//             if (not temporary_plan.is_valid()) //resetea el valor de la y cuando pones la x
//                 temporary_plan.new_plan(Plan::Actions::TALKING_WITH_PEOPLE);
//             auto talk_person_node = G->get_node(person_name.toUtf8().constData());
//             auto talk_person_node_id = talk_person_node->id();

//             std::ostringstream oss;
//             oss << talk_person_node_id;
//             std::string intAsString(oss.str());

//             temporary_plan.insert_attribute("person_node_id",  QString::fromStdString(intAsString));
//             temporary_plan.insert_attribute("destiny", "floor");
//             custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'person_name' in Follow People plan");
//             if (target_scene != nullptr)
//                 widget_2d->scene.removeItem(target_scene);
//         }
//         else
//             custom_widget.textedit_current_plan->appendPlainText("-> Please, choose a person to follow first");
//         return;
//     }


//    target_scene = widget_2d->scene.addEllipse(-50, -50, 100, 100, QPen(QColor("Orange")),
//                                               QBrush(QColor("Orange")));
//    target_scene->setZValue(100);

//        temporary_plan.insert_attribute("x",int(person_pose[0]));
//        temporary_plan.insert_attribute("y",int(person_pose[1]));
//        temporary_plan.insert_attribute("destiny", "floor");
//        cout<<person_pose[0]<<"   "<<person_pose[1]<< endl;
//        custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'x' in Follow People plan");
//        custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'y' in Follow People plan");
//        if (target_scene != nullptr)
//            widget_2d->scene.removeItem(target_scene);
//        target_scene = widget_2d->scene.addEllipse(-50, -50, 100, 100, QPen(QColor("Orange")),
//                                                   QBrush(QColor("Orange")));
//        target_scene->setPos(person_pose[0], person_pose[1]);
//        target_scene->setZValue(100);

// }
void SpecificWorker::create_follow_people_mission(uint64_t person_id)
{
    if(person_id != 0)
    {
        static QGraphicsEllipseItem *target_scene;
        temporary_plan.new_plan(Plan::Actions::FOLLOW_PEOPLE);
        custom_widget.textedit_current_plan->appendPlainText("-> New temporary plan: FOLLOW PEOPLE");
        custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));

        if (not temporary_plan.is_valid()) //resetea el valor de la y cuando pones la x
            temporary_plan.new_plan(Plan::Actions::FOLLOW_PEOPLE);
        auto followed_person_node = G->get_node(person_id);
        auto followed_person_node_id = followed_person_node->id();

        std::ostringstream oss;
        oss << followed_person_node_id;
        std::string intAsString(oss.str());

        temporary_plan.insert_attribute("person_node_id",  QString::fromStdString(intAsString));
        if(auto person_name = G->get_attrib_by_name<person_name_att>(followed_person_node.value()); person_name.has_value())
        {
            temporary_plan.insert_attribute("person_name",  QString::fromStdString(person_name.value()));
        }
    
        custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'person_name' in Follow People plan");
        if (target_scene != nullptr)
            widget_2d->scene.removeItem(target_scene);
    }
    else
    {
        if(custom_widget.object_list->selectedItems().size() > 0)
        {
            static QGraphicsEllipseItem *target_scene;
            custom_widget.stacked_widget->setCurrentIndex(3);
            temporary_plan.new_plan(Plan::Actions::FOLLOW_PEOPLE);
            custom_widget.textedit_current_plan->appendPlainText("-> New temporary plan: FOLLOW PEOPLE");
            custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));
//    basic_string<char> person_name = custom_widget.object_list->currentItem()->text().toUtf8().constData();
            auto person_name = custom_widget.object_list->currentItem()->text();

            if (not temporary_plan.is_valid()) //resetea el valor de la y cuando pones la x
                temporary_plan.new_plan(Plan::Actions::FOLLOW_PEOPLE);
            auto followed_person_node = G->get_node(person_name.toUtf8().constData());
            auto followed_person_node_id = followed_person_node->id();

            std::ostringstream oss;
            oss << followed_person_node_id;
            std::string intAsString(oss.str());

            temporary_plan.insert_attribute("person_node_id",  QString::fromStdString(intAsString));
            temporary_plan.insert_attribute("destiny", "floor");
            custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'person_name' in Follow People plan");
            if (target_scene != nullptr)
                widget_2d->scene.removeItem(target_scene);
        }
        else
            custom_widget.textedit_current_plan->appendPlainText("-> Please, choose a person to follow first");
        return;
    }


//    target_scene = widget_2d->scene.addEllipse(-50, -50, 100, 100, QPen(QColor("Orange")),
//                                               QBrush(QColor("Orange")));
//    target_scene->setZValue(100);

//        temporary_plan.insert_attribute("x",int(person_pose[0]));
//        temporary_plan.insert_attribute("y",int(person_pose[1]));
//        temporary_plan.insert_attribute("destiny", "floor");
//        cout<<person_pose[0]<<"   "<<person_pose[1]<< endl;
//        custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'x' in Follow People plan");
//        custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'y' in Follow People plan");
//        if (target_scene != nullptr)
//            widget_2d->scene.removeItem(target_scene);
//        target_scene = widget_2d->scene.addEllipse(-50, -50, 100, 100, QPen(QColor("Orange")),
//                                                   QBrush(QColor("Orange")));
//        target_scene->setPos(person_pose[0], person_pose[1]);
//        target_scene->setZValue(100);

}
void SpecificWorker::create_recognize_people_mission(uint64_t person_id)
{
    interacting_person_id = person_id;
    temporary_plan.new_plan(Plan::Actions::RECOGNIZE_PEOPLE);
    custom_widget.textedit_current_plan->appendPlainText("-> New temporary plan: RECOGNIZE PEOPLE");
    custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));

    if (not temporary_plan.is_valid()) //resetea el valor de la y cuando pones la x
        temporary_plan.new_plan(Plan::Actions::RECOGNIZE_PEOPLE);
    auto to_recognize_person_node = G->get_node(person_id);
    auto to_recognize_person_node_id = to_recognize_person_node->id();

    std::ostringstream oss;
    oss << to_recognize_person_node_id;
    std::string intAsString(oss.str());

    temporary_plan.insert_attribute("person_node_id",  QString::fromStdString(intAsString));
    temporary_plan.insert_attribute("destiny", "floor");
    custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'person_name' in Recognize People plan");
}
void SpecificWorker::create_talking_people_mission(uint64_t person_id)
{
    temporary_plan.new_plan(Plan::Actions::TALKING_WITH_PEOPLE);
    custom_widget.textedit_current_plan->appendPlainText("-> New temporary plan: TALKING WITH PEOPLE");
    custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));

    if (not temporary_plan.is_valid()) //resetea el valor de la y cuando pones la x
        temporary_plan.new_plan(Plan::Actions::TALKING_WITH_PEOPLE);
    if(auto to_recognize_person_node = G->get_node(person_id); to_recognize_person_node.has_value())
    {
        auto to_recognize_person_node_id = to_recognize_person_node->id();
        if(auto person_name = G->get_attrib_by_name<person_name_att>(to_recognize_person_node.value()); person_name.has_value())
        {
            std::ostringstream oss;
            oss << to_recognize_person_node_id;
            std::string intAsString(oss.str());

            temporary_plan.insert_attribute("person_node_id",  QString::fromStdString(intAsString));
            temporary_plan.insert_attribute("person_name",  QString::fromStdString(person_name.value()));
            temporary_plan.insert_attribute("destiny", "floor");
            custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'person_name' in Recognize People plan");
        }
    }
}

//    connect(point_dialog.goto_spinbox_coordX, qOverload<int>(&QSpinBox::valueChanged), [this](int v) {
//        if (not temporary_plan.is_valid()) //resetea el valor de la y cuando pones la x
//            temporary_plan.new_plan(Plan::Actions::GOTO);
//
//        temporary_plan.insert_attribute("x", v);
//        custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'x' in GOTO plan");
//        custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));
//        // redraw target
//        if (target_scene != nullptr)
//            widget_2d->scene.removeItem(target_scene);
//        target_scene = widget_2d->scene.addEllipse(-50, -50, 100, 100, QPen(QColor("Orange")),
//                                                   QBrush(QColor("Orange")));
//        target_scene->setPos(v, point_dialog.goto_spinbox_coordY->value());
//        target_scene->setZValue(100);
//    });
//
//    connect(point_dialog.goto_spinbox_coordY, qOverload<int>(&QSpinBox::valueChanged), [this](int v) {
////        if (not temporary_plan.is_valid())
////            temporary_plan.new_plan(Plan::Actions::GOTO);
//
//        temporary_plan.insert_attribute("y", v);
//        custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'y' in GOTO plan");
//        custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));
//        //redraw target
//        if (target_scene != nullptr)
//            widget_2d->scene.removeItem(target_scene);
//        target_scene = widget_2d->scene.addEllipse(-50, -50, 100, 100, QPen(QColor("Orange")),
//                                                   QBrush(QColor("Orange")));
//        target_scene->setPos(point_dialog.goto_spinbox_coordX->value(), v);
//        target_scene->setZValue(100);
//    });
//
//    connect(point_dialog.goto_spinbox_angle, qOverload<int>(&QSpinBox::valueChanged), [this](int v) {
//        temporary_plan.insert_attribute("destiny", v);
//        custom_widget.textedit_current_plan->appendPlainText("-> New attribute 'destiny' in GOTO plan");
//        custom_widget.textedit_current_plan->appendPlainText(QString::fromStdString(temporary_plan.pprint()));
//    });
//
//    connect(widget_2d, &DSR::QScene2dViewer::mouse_right_click, [this](int x, int y, std::uint64_t obj) {
//        if (not temporary_plan.is_valid())
//            temporary_plan.new_plan(Plan::Actions::GOTO);
//
//        if (auto node = G->get_node(obj); node.has_value())
//            temporary_plan.insert_attribute("destiny", QString::fromStdString(node.value().name()));
//
//        point_dialog.goto_spinbox_coordX->setValue(x);
//        point_dialog.goto_spinbox_coordY->setValue(y);
//
//        // redraw target
//        if (target_scene != nullptr)
//            widget_2d->scene.removeItem(target_scene);
//        target_scene = widget_2d->scene.addEllipse(-50, -50, 100, 100, QPen(QColor("Orange")),
//                                                   QBrush(QColor("Orange")));
//        target_scene->setPos(x, y);
//        target_scene->setZValue(100);

//    });
//}

void SpecificWorker::people_checker()
{
    // List of people

    auto people_nodes = G->get_nodes_by_type("person");

    if(custom_widget.object_list->count() != people_nodes.size())
    {
//    for(int i = 0; i < custom_widget.object_list->count(); ++i)
//    {
//        QListWidgetItem* item = custom_widget.object_list->item(i);
//        for (auto person : people_nodes)
//        {
//            QString str = QString::fromUtf8( person.name().data(), person.name().size());
//            if(item->text() == str)
//                break;
//        }
//        custom_widget.object_list->removeItemWidget(item);
//    }
//
//    for (auto person : people_nodes)
//    {
//        QString str = QString::fromUtf8( person.name().data(), person.name().size());
//
//        for(int i = 0; i < custom_widget.object_list->count(); ++i)
//        {
//            QListWidgetItem* item = custom_widget.object_list->item(i);
//            if(item->text() == str)
//                break;
//        }
//        custom_widget.object_list->addItem(str);
//        if(custom_widget.object_list->findItems(str, 0).size() > 0)
//            continue;
//        else
//            custom_widget.object_list->addItem(str);
//    }


        custom_widget.object_list->clear();
        for (auto person : people_nodes)
        {
            QString str = QString::fromUtf8( person.name().data(), person.name().size());
            custom_widget.object_list->addItem(str);
        }
    }
}

uint64_t SpecificWorker::node_string2id(Plan currentPlan)
{
    auto person_id = currentPlan.get_attribute("person_node_id");
    uint64_t value;
    std::istringstream iss(person_id.toString().toUtf8().constData());
    iss >> value;
    return value;
}

////////////////////////////////////////////////////////////////////////////////////////////
/// Asynchronous changes on G nodes from G signals
////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::add_or_assign_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type)
{
    if(from == 200 and (to == followed_person_id or to == talkative_person_id) and type == "RT")
    {
        if(auto person = inner_eigen->transform(G->get_node(from).value().name(), G->get_node(to).value().name()); person.has_value())
        {
            last_person_pos = person.value();
            qInfo() << "LAST PERSON POS:" << last_person_pos.x() << last_person_pos.y();
        }
    }

    // if(type == lost_type_name)
    // {
    //     slot_stop_mission();
    //     create_searching_person_mission();
    //     if(temporary_plan.is_complete())
    //     {
    //         qInfo() << "COMPLETED";
    //         insert_intention_node(temporary_plan);
    //         auto temp_plan = temporary_plan;
    //         plan_buffer.put(std::move(temp_plan));
    //         custom_widget.textedit_current_plan->verticalScrollBar()->setValue(custom_widget.textedit_current_plan->verticalScrollBar()->maximum());
    //     }
    // }
}
void SpecificWorker::del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag)
{
    // if(edge_tag == "interacting")
    // {
    //     if(auto intention = G->get_node(robot_current_intention_name); intention.has_value())
    //     {
    //         slot_stop_mission();
    //     }
    // }
}
void SpecificWorker::del_node_slot(std::uint64_t from)
{
    if(current_plan.get_action()=="FOLLOW_PEOPLE" || current_plan.get_action()=="TALKING_WITH_PEOPLE")
    {
        auto person_id = current_plan.get_attribute("person_node_id");
        auto person_id_int = from_variant_to_uint64(person_id);
        if(from == person_id_int)
        {
            DSR::Node new_virtual_node = DSR::Node::create<virtual_person_node_type>("lost_person");
            G->add_or_modify_attrib_local<person_name_att>(new_virtual_node, current_plan.get_attribute("person_name").toString().toStdString());
            G->add_or_modify_attrib_local<person_mission_att>(new_virtual_node, current_plan.get_action().toStdString());
            G->insert_node(new_virtual_node);
            G->update_node(new_virtual_node);
            DSR::Edge lost_edge = DSR::Edge::create<lost_edge_type>(G->get_node("robot").value().id(), new_virtual_node.id());
            if (G->insert_or_assign_edge(lost_edge))
            {
                std::cout << __FUNCTION__ << " Edge succcessfully inserted: " << G->get_node("robot").value().id() << "->" << new_virtual_node.id()
                          << " type: lost" << std::endl;
            }
            else
            {
                std::cout << __FUNCTION__ << ": Fatal error inserting new edge: " << G->get_node("robot").value().id() << "->" << new_virtual_node.id()
                          << " type: lost" << std::endl;
            }
            DSR::Edge has_edge = DSR::Edge::create<has_edge_type>(G->get_node(robot_mind_name).value().id(), new_virtual_node.id());
            if (G->insert_or_assign_edge(has_edge))
            {
                std::cout << __FUNCTION__ << " Edge successfully inserted: " << G->get_node(robot_mind_name).value().id() << "->" << new_virtual_node.id()
                          << " type: has" << std::endl;
            }
            else
            {
                std::cout << __FUNCTION__ << ": Fatal error inserting new edge: " << G->get_node(robot_mind_name).value().id() << "->" << new_virtual_node.id()
                          << " type: has" << std::endl;
            }

            slot_stop_mission();

            // CODE TO MAKE ROBOT GO TO THE LAST PERSON POSITION
            // create_searching_person_mission();
            // if(temporary_plan.is_complete())
            // {
            //     qInfo() << "COMPLETED";
            //     insert_intention_node(temporary_plan);
            //     auto temp_plan = temporary_plan;
            //     plan_buffer.put(std::move(temp_plan));
            //     custom_widget.textedit_current_plan->verticalScrollBar()->setValue(custom_widget.textedit_current_plan->verticalScrollBar()->maximum());
            // }
            // if(auto grid_node = G->get_node(grid_type_name); grid_node.has_value())
            // {
            //     if(auto grid_activated_attr = G->get_attrib_by_name<grid_activated_att>(grid_node.value()); grid_activated_attr.has_value())
            //     {
            //         G->add_or_modify_attrib_local<grid_activated_att>(grid_node.value(), true);
            //         G->update_node(grid_node.value());
            //     }
            // }
        }

    }

}
void SpecificWorker::add_or_assign_node_slot(const std::uint64_t id, const std::string &type)
{
//    if(type == rgbd_type_name and id == cam_api->get_id())
//    {
//        if (auto cam_node = G->get_node(id); cam_node.has_value())
//            if (const auto g_image = G->get_attrib_by_name<cam_rgb_att>(cam_node.value()); g_image.has_value())
//                virtual_camera_buffer.put(std::vector<uint8_t>(g_image.value().get().begin(), g_image.value().get().end()));
//    }

    if (type == "intention")    // Laser node updated
    {
        if(auto current_intention_node = G->get_node(id); current_intention_node.has_value())
        {
            std::optional <std::string> plan = G->get_attrib_by_name<current_intention_att>(current_intention_node.value());
            if (plan.has_value())
            {
                Plan received_plan = plan.value();
                auto action_name = received_plan.get_action();
                auto follow_action_name = QString::fromStdString("ASK_FOR_FOLLOWING");
                auto stop_follow_action_name = QString::fromStdString("ASK_FOR_STOP_FOLLOWING");
                auto talking_action_name = QString::fromStdString("TALKING_WITH_ROBOT");
                auto stop_talking_action_name = QString::fromStdString("ASK_FOR_STOP_TALKING");

                if (action_name == follow_action_name)
                {
                    
                    if(auto lost_person_in_mind = G->get_node("lost_person"); lost_person_in_mind.has_value())
                    {
                        if (auto has_edge = G->get_edge(G->get_node(robot_mind_name).value().id(),lost_person_in_mind.value().id(),"has");has_edge.has_value())
                            G->delete_edge(G->get_node(robot_mind_name).value().id(),lost_person_in_mind.value().id(),"has");
                        if (auto lost_edge = G->get_edge(G->get_node(robot_name).value().id(),lost_person_in_mind.value().id(),"lost");lost_edge.has_value())
                            G->delete_edge(G->get_node(robot_name).value().id(),lost_person_in_mind.value().id(),"lost");
                        G->delete_node(lost_person_in_mind.value().id());
                    }
                    followed_person_id = node_string2id(received_plan);
                    create_follow_people_mission(followed_person_id);
                    if(temporary_plan.is_complete())
                    {
                        insert_intention_node(temporary_plan);
                        if(temporary_plan.is_action(Plan::Actions::FOLLOW_PATH))
                            follow_path_copy_path_to_graph(temporary_plan.x_path, temporary_plan.y_path);

                        auto temp_plan = temporary_plan;
                        plan_buffer.put(std::move(temp_plan));
                        custom_widget.textedit_current_plan->verticalScrollBar()->setValue(custom_widget.textedit_current_plan->verticalScrollBar()->maximum());
                    }
                    else
                        qWarning() << __FUNCTION__ << "Plan is not complete. Mission cannot be created";
                }
                else if (action_name == stop_follow_action_name || action_name == stop_talking_action_name)
                {
                    slot_stop_mission();
                }
                else if (action_name == talking_action_name)
                {
                if(auto lost_person_in_mind = G->get_node("lost_person"); lost_person_in_mind.has_value())
                    {
                        if (auto has_edge = G->get_edge(G->get_node(robot_mind_name).value().id(),lost_person_in_mind.value().id(),"has");has_edge.has_value())
                            G->delete_edge(G->get_node(robot_mind_name).value().id(),lost_person_in_mind.value().id(),"has");
                        if (auto lost_edge = G->get_edge(G->get_node(robot_name).value().id(),lost_person_in_mind.value().id(),"lost");lost_edge.has_value())
                            G->delete_edge(G->get_node(robot_name).value().id(),lost_person_in_mind.value().id(),"lost");
                        G->delete_node(lost_person_in_mind.value().id());
                    }
                    talkative_person_id = node_string2id(received_plan);
                    create_talking_people_mission(talkative_person_id);
                    if(temporary_plan.is_complete())
                    {
                        insert_intention_node(temporary_plan);
                        auto temp_plan = temporary_plan;
                        plan_buffer.put(std::move(temp_plan));
                        custom_widget.textedit_current_plan->verticalScrollBar()->setValue(custom_widget.textedit_current_plan->verticalScrollBar()->maximum());
                    }
                    else
                        qWarning() << __FUNCTION__ << "Plan is not complete. Mission cannot be created";
                }
            }
        }
    }

    if (type == laser_type_name)    // Laser node updated
    {
        //qInfo() << __FUNCTION__ << " laser node change";
        if( auto node = G->get_node(id); node.has_value())
        {
            auto angles = G->get_attrib_by_name<laser_angles_att>(node.value());
            auto dists = G->get_attrib_by_name<laser_dists_att>(node.value());
            if(dists.has_value() and angles.has_value())
            {
                if(dists.value().get().empty() or angles.value().get().empty()) return;
                laser_buffer.put(std::move(std::make_tuple(angles.value().get(), dists.value().get())),
                                 [this](const LaserData &in, std::tuple<std::vector<float>, std::vector<float>, QPolygonF,std::vector<QPointF>> &out) {
                                     QPolygonF laser_poly_local;
                                     //laser_poly_local << QPointF(0.f, 200.f);
                                     std::vector<QPointF> laser_cart_world;
                                     const auto &[angles, dists] = in;
                                     for (const auto &[angle, dist] : iter::zip(angles, dists))
                                     {
                                         //convert laser polar coordinates to cartesian in the robot's coordinate frame
                                         float x = dist * sin(angle);
                                         float y = dist * cos(angle);
                                         Mat::Vector3d laserWorld = inner_eigen->transform(world_name,Mat::Vector3d(x, y, 0), laser_name).value();
                                         laser_poly_local << QPointF(x, y);
                                         laser_cart_world.emplace_back(QPointF(laserWorld.x(), laserWorld.y()));
                                     }
                                     //laser_poly_local << QPointF(0.f, 200.f);
                                     out = std::make_tuple(angles, dists, laser_poly_local, laser_cart_world);
                                 });
            }
        }
    }
    else if (type == path_to_target_type_name)
    {
        if( auto path_to_target_node = G->get_node(id); path_to_target_node.has_value())
        {
            auto x_values_o = G->get_attrib_by_name<path_x_values_att>(path_to_target_node.value());
            auto y_values_o = G->get_attrib_by_name<path_y_values_att >(path_to_target_node.value());
            if(x_values_o.has_value() and y_values_o.has_value())
            {
                path.clear();
                auto &x_values = x_values_o.value().get();
                auto &y_values = y_values_o.value().get();
                path.reserve(x_values.size());
                for(auto &&[p, q] : iter::zip(x_values,y_values))
                    path.emplace_back(Eigen::Vector2f(p, q));
                draw_path(path, &widget_2d->scene);
            }
        }
    }
}
void SpecificWorker::modify_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names)
{

    // if(interacting_person_id == id)
    // {
    //     qInfo() << "2";
    //     if (std::count(att_names.begin(), att_names.end(), "checked_face"))
    //     {
    //         qInfo() << "3";
    //         if(auto checked_face_value = G->get_attrib_by_name<checked_face_att>(G->get_node(interacting_person_id).value()); checked_face_value.has_value() && checked_face_value.value() == true)
    //         {
    //             qInfo() << "4";
    //             slot_stop_mission();
    //             create_talking_people_mission(interacting_person_id);
    //             if(temporary_plan.is_complete())
    //             {
    //                 insert_intention_node(temporary_plan);
    //                 auto temp_plan = temporary_plan;
    //                 plan_buffer.put(std::move(temp_plan));
    //                 custom_widget.textedit_current_plan->verticalScrollBar()->setValue(custom_widget.textedit_current_plan->verticalScrollBar()->maximum());
    //             }
    //             else
    //                 qWarning() << __FUNCTION__ << "Plan is not complete. Mission cannot be created";
    //         }
    //     }
    // }
}
void SpecificWorker::insert_intention_node(const Plan &plan)
{
    // Check if there is not 'intention' node yet in G
    if(auto mind = G->get_node(robot_mind_name); mind.has_value())
    {
        if (auto intention = G->get_node(robot_current_intention_name); not intention.has_value())
        {
            DSR::Node intention_node = DSR::Node::create<intention_node_type>(robot_current_intention_name);
            G->add_or_modify_attrib_local<parent_att>(intention_node, mind.value().id());
            G->add_or_modify_attrib_local<level_att>(intention_node, G->get_node_level(mind.value()).value() + 1);
            G->add_or_modify_attrib_local<pos_x_att>(intention_node, (float) -290);
            G->add_or_modify_attrib_local<pos_y_att>(intention_node, (float) -474);
            G->add_or_modify_attrib_local<current_intention_att>(intention_node, plan.to_json());
            std::cout << "DATOS DEL PALN: " << plan.to_json() << std::endl;
            if (std::optional<int> intention_node_id = G->insert_node(intention_node); intention_node_id.has_value())
            {
                DSR::Edge edge = DSR::Edge::create<has_edge_type>(mind.value().id(), intention_node.id());
                if (G->insert_or_assign_edge(edge))
                {
                    std::cout << __FUNCTION__ << " Edge successfully inserted: " << mind.value().id() << "->" << intention_node.id()
                              << " type: has" << std::endl;
                    G->add_or_modify_attrib_local<current_intention_att>(intention_node, plan.to_json());
                    G->update_node(intention_node);
                }
                else
                {
                    std::cout << __FUNCTION__ << ": Fatal error inserting new edge: " << mind.value().id() << "->" << intention_node_id.value()
                              << " type: has" << std::endl;
                    std::terminate();

                }
                if(plan.get_action()=="FOLLOW_PEOPLE")
                {
                    auto follow_plan = plan;
                    auto person_id_str = follow_plan.get_attribute("person_node_id");
                    auto person_id = from_variant_to_uint64(person_id_str);
                }

            } else
            {
                std::cout << __FUNCTION__ << ": Fatal error inserting_new 'intention' node" << std::endl;
                std::terminate();
            }
        }
        else // there is one intention node
        {
            std::cout << __FUNCTION__ << ": Updating existing intention node with Id: " << intention.value().id() << std::endl;
            G->add_or_modify_attrib_local<current_intention_att>(intention.value(), plan.to_json());
            G->update_node(intention.value());
            std::cout << "INSERT: " << plan.to_json() << std::endl;
        }
    }
    else
    {
        std::cout  << __FUNCTION__ << " No node " << robot_mind_name << " found in G" << std::endl;
        std::terminate();
    }
}
void SpecificWorker::follow_path_copy_path_to_graph(const std::vector<float> &x_values, const std::vector<float> &y_values)
{
    if (auto path = G->get_node(current_path_name); path.has_value())
    {
        auto path_to_target_node = path.value();
        G->add_or_modify_attrib_local<path_x_values_att>(path_to_target_node, x_values);
        G->add_or_modify_attrib_local<path_y_values_att>(path_to_target_node, y_values);
        G->add_or_modify_attrib_local<path_is_cyclic_att>(path_to_target_node, true);
        G->update_node(path_to_target_node);
    }
    else // create path_to_target_node with the solution path
    {
        if(auto intention = G->get_node(robot_current_intention_name); intention.has_value())
        {
            auto path_to_target_node = DSR::Node::create<path_to_target_node_type>(current_path_name);
            G->add_or_modify_attrib_local<path_x_values_att>(path_to_target_node, x_values);
            G->add_or_modify_attrib_local<path_y_values_att>(path_to_target_node, y_values);
            G->add_or_modify_attrib_local<path_is_cyclic_att>(path_to_target_node, true);
            G->add_or_modify_attrib_local<pos_x_att>(path_to_target_node, (float) -542);
            G->add_or_modify_attrib_local<pos_y_att>(path_to_target_node, (float) 106);
            G->add_or_modify_attrib_local<parent_att>(path_to_target_node, intention.value().id());
            G->add_or_modify_attrib_local<level_att>(path_to_target_node, 3);
            auto id = G->insert_node(path_to_target_node);
            DSR::Edge edge_to_intention = DSR::Edge::create<thinks_edge_type>(id.value(), intention.value().id());
            G->insert_or_assign_edge(edge_to_intention);
        }
        else qWarning() << __FUNCTION__ << "No intention node found. Can't create path node";
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// UI
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SpecificWorker::slot_start_mission()
{
    if(not temporary_plan.is_valid())
    {
        slot_change_mission_selector(custom_widget.list_plan->currentIndex());
    }


    if(temporary_plan.is_complete())
    {
        insert_intention_node(temporary_plan);
        if(temporary_plan.is_action(Plan::Actions::FOLLOW_PATH))
            follow_path_copy_path_to_graph(temporary_plan.x_path, temporary_plan.y_path);

        auto temp_plan = temporary_plan;
        plan_buffer.put(std::move(temp_plan));
        custom_widget.textedit_current_plan->verticalScrollBar()->setValue(custom_widget.textedit_current_plan->verticalScrollBar()->maximum());
    }
    else
        qWarning() << __FUNCTION__ << "Plan is not complete. Mission cannot be created";
}

void SpecificWorker::slot_stop_mission()
{
    qInfo() << "A CERO";
    send_command_to_robot(std::make_tuple(0.f,0.f,0.f));   //adv, rot, side
    if(auto intention = G->get_node(robot_current_intention_name); intention.has_value())
    {

        if(auto path = G->get_node(current_path_name); path.has_value())
        {
            if (auto edge_thinks = G->get_edge(path.value().id(),intention.value().id(),"thinks");edge_thinks.has_value())
                G->delete_edge(path.value().id(),intention.value().id(), "thinks");


            G->delete_node(path.value().id());
        }

        auto mind = G->get_node(robot_mind_name);

        if (auto has_mind_intention = G->get_edge(mind.value().id(),intention.value().id(),"has");has_mind_intention.has_value())
            G->delete_edge(mind.value().id(),intention.value().id(), "has");

        G->delete_node(intention.value().id());
        this->eyecontrol_proxy->setFollowedPerson(0);

    }
    else
        qWarning() << __FUNCTION__ << "No intention node found";
    if(temporary_plan.is_valid())
        custom_widget.textedit_current_plan->appendPlainText("-> mission " + temporary_plan.get_action() + " cancelled");
    if(current_plan.get_action()=="FOLLOW_PEOPLE")
    {
        auto person_id = current_plan.get_attribute("person_node_id");
        auto id_value = from_variant_to_uint64(person_id);
        if (auto followed_person_node = G->get_node(id_value); followed_person_node.has_value())
        {
            G->delete_edge(G->get_node("robot").value().id(), followed_person_node.value().id(), following_action_type_name);
        }
    }

    temporary_plan.reset();
    current_plan.reset();
    // remove path form drawing
//    std::vector<Eigen::Vector2f> fake_path;
//    draw_path(fake_path, &widget_2d->scene, true); // just remove
//    draw_path(fake_path, &pathfollow_draw_widget->scene, true); // just remove
}

void SpecificWorker::slot_cancel_mission()
{
    slot_stop_mission();
}

void SpecificWorker::slot_change_mission_selector(int index)
{
    // remove current filling_plan and create a new one
    slot_stop_mission();
    // createa new current filling_plan of the index typw
    switch(index)
    {
        case 1:
            create_goto_mission(); // Goto_XYA
            break;
        case 2:
            create_path_mission();
            break;
        case 3:
            create_bouncer_mission();
            break;
        case 4:
            create_follow_people_mission();
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////
/// Auxiliary methods
/////////////////////////////////////////////////////////////////////////////////////////////
uint64_t SpecificWorker::from_variant_to_uint64(QVariant value)
{
    uint64_t aux;
    std::istringstream iss(value.toString().toUtf8().constData());
    iss >> aux;
    return aux;
}


void SpecificWorker::draw_path(std::vector<Eigen::Vector2f> &path, QGraphicsScene* viewer_2d, bool remove)
{
    //static std::vector<QGraphicsLineItem *> scene_road_points;
    static std::unordered_map<QGraphicsScene *, std::vector<QGraphicsLineItem *> *> scene_road_points_map;
    std::vector<QGraphicsLineItem *> *scene_road_points;

    if (scene_road_points_map.contains(viewer_2d))
        scene_road_points = scene_road_points_map[viewer_2d];
    else
        scene_road_points = new std::vector<QGraphicsLineItem *>();

    scene_road_points_map[viewer_2d] = scene_road_points;

    //clear previous points
    //for (QGraphicsLineItem* item : scene_road_points)
    for (QGraphicsLineItem* item : *scene_road_points)
    {
        viewer_2d->removeItem((QGraphicsItem *) item);
        delete item;
    }

    //scene_road_points.clear();
    scene_road_points->clear();

    if(remove) return;      // Just clear the path

    /// Draw all points
    QGraphicsLineItem *line1, *line2;
    std::string color;
    for (unsigned int i = 1; i < path.size(); i++)
        for(auto &&p_pair : iter::sliding_window(path, 2))
        {
            if(p_pair.size() < 2)
                continue;
            Mat::Vector2d a_point(p_pair[0].x(), p_pair[0].y());
            Mat::Vector2d b_point(p_pair[1].x(), p_pair[1].y());
            Mat::Vector2d dir = a_point - b_point;
            Mat::Vector2d dir_perp = dir.unitOrthogonal();
            Eigen::ParametrizedLine segment = Eigen::ParametrizedLine<double, 2>::Through(a_point, b_point);
            Eigen::ParametrizedLine<double, 2> segment_perp((a_point+b_point)/2, dir_perp);
            auto left = segment_perp.pointAt(50);
            auto right = segment_perp.pointAt(-50);
            QLineF qsegment(QPointF(a_point.x(), a_point.y()), QPointF(b_point.x(), b_point.y()));
            QLineF qsegment_perp(QPointF(left.x(), left.y()), QPointF(right.x(), right.y()));

            if(i == 1 or i == path.size()-1)
                color = "#00FF00"; //Green

            line1 = viewer_2d->addLine(qsegment, QPen(QBrush(QColor(QString::fromStdString(color))), 20));
            line2 = viewer_2d->addLine(qsegment_perp, QPen(QBrush(QColor(QString::fromStdString("#F0FF00"))), 20));
            line1->setZValue(2000);
            line2->setZValue(2000);
//            scene_road_points.push_back(line1);
//            scene_road_points.push_back(line2);
            scene_road_points->push_back(line1);
            scene_road_points->push_back(line2);
        }
    std::cout << "scene_road_points size: " << scene_road_points->size() << std::endl;
}
void SpecificWorker::send_command_to_robot(const std::tuple<float, float, float> &speeds)   //adv, rot, side
{
    auto &[adv_, rot_, side_] = speeds;
    auto robot_node = G->get_node(robot_name);
    G->add_or_modify_attrib_local<robot_ref_adv_speed_att>(robot_node.value(),  (float)adv_);
    G->add_or_modify_attrib_local<robot_ref_rot_speed_att>(robot_node.value(), (float)rot_);
    G->add_or_modify_attrib_local<robot_ref_side_speed_att>(robot_node.value(), (float)side_);
    G->update_node(robot_node.value());
}
void SpecificWorker::trace_button_slot(bool checked)
{
    if(not checked)
    {
        qInfo() << "Should delete lines";
        for(auto &l : lines)
            widget_2d->scene.removeItem(l);
        lines.clear();
    }
    else
        try
        {
            auto robot_pos = inner_eigen->transform(world_name, robot_name).value();
            last_point = QPointF(robot_pos.x(), robot_pos.y());
        }
        catch(const Ice::Exception &e) { std::cout << e.what() << std::endl;}
}
//////////////////////////////////////////////////////////////////////////////////

int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, qApp, SLOT(quit()));
    return 0;
}

//    using namespace std::placeholders;
//    if (auto target_node = G->get_node(id); target_node.has_value())
//    {
//        const std::string location =
//                "[" + std::to_json(pos_x) + "," + std::to_json(pos_y) + "," + std::to_json(0) + "]";
//        const std::string plan =
//                "{\"plan\":[{\"action\":\"goto\",\"params\":{\"location\":" + location + ",\"object\":\"" +
//                target_node.value().name() + "\"}}]}";
//        std::cout << plan << std::endl;
//        plan_buffer.put(plan, std::bind(&SpecificWorker::json_to_plan, this, _1, _2));
//    } else
//        qWarning() << __FILE__ << __FUNCTION__ << " No target node  " << QString::number(id) << " found";

///////////////////////////////////



//            if(not primera_vez)
//            {
//                primera_vez = true;
//                begin = lastPathStep = std::chrono::steady_clock::now();
//                last_path_size = path.value().size();
//                auto dist = 0;
//                for (uint i = 0; i < path.value().size() - 1; ++i)
//                    dist = dist + (path.value()[i] - path.value()[i + 1]).norm();
//
//                qInfo() << path.value().size();
//                qInfo() << __FUNCTION__ << "Inicio: " << path.value()[0].x() << "," << path.value()[0].y();
//                qInfo() << __FUNCTION__ << "Final: " << path.value()[path.value().size() - 1].x() << ","
//                        << path.value()[path.value().size() - 1].y();
//                qInfo() << __FUNCTION__ << "Distancia: " << dist;
//            }
//            auto robot_pose = inner_eigen->transform(world_name, robot_name).value();
//            if (last_path_size != path.value().size())
//                lastPathStep = std::chrono::steady_clock::now();
//            float dist = (robot_pose - plan.get_target_trans()).norm();
//            auto now = std::chrono::steady_clock::now();
//            auto time_delta_s = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPathStep).count();
//            qInfo() << __FUNCTION__ << "Tiempo desde el último paso: " << time_delta_s << "s";
//            auto acceptable_distance = 1 + uint(K * time_delta_s);
//            if (path.value().size() <= acceptable_distance or dist < 200)
//            {
//                plan.set_active(false);
//                send_command_to_robot(std::make_tuple(0.f, 0.f, 0.f));
//                slot_stop_mission();
//                primera_vez = false;
//                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//                auto d=std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
//                qInfo() << __FUNCTION__ << " Tiempo: " << d << " s";
//            }
//            else  qInfo() << __FUNCTION__ << "Path size: " << path.value().size();




//DSR::Node SpecificWorker::get_robot_node()
//{
//    if (auto robot_o = G->get_node(robot_name); robot_o.has_value())
//        return robot_o.value();
//    else
//    {
//        std::cout << __FUNCTION__ << " Terminating due to missing robot_node: " << robot_name << std::endl;
//        std::terminate();
//    }
//}

//void SpecificWorker::read_index()
//{
//    int index = custom_widget.list_plan->currentIndex();
//    switch (index)
//    {
//        case 1: //misión punto
//            custom_widget.empty_widget->show();
//        break;
//        case 2: //misión secuencia puntos (se carga un plan de diferentes puntos preestablecidos)
//        break;
//        case 3: //misión chocachoca (creo plan que en vez de "goto" sea "chocachoca")
//            create_bouncer_mission();
//        break;
//    }
//}


//void SpecificWorker::new_target_from_mouse(int pos_x, int pos_y, std::uint64_t id)
//{
//    qInfo() << __FUNCTION__ << " Creating GOTO mission to " << pos_x << pos_y;
//    qInfo() << __FUNCTION__ << "[" << pos_x << " " << pos_y << "] Id:" << id;
//
//    //create_goto_mission(QPointF(pos_x, pos_y), id);
//
//    // we get the id of the object clicked from the 2D representation
////    if (auto target_node = G->get_node(id); target_node.has_value())
////    {
////        std::stringstream location;
////        location <<"[" << std::to_json(pos_x) << "," << std::to_json(pos_y) << "," + std::to_json(0) << "]";
////        std::string plan_string = R"({"plan":[{"action":"goto","params":{"location":)" + location.str() + R"(,"object":")" + target_node.value().name() + "\"}}]}";
////        Plan plan(plan_string);
////        plan.print();
////        insert_intention_node(plan);
////        plan_buffer.put(std::move(plan));
////    }
////    else
////        qWarning() << __FUNCTION__ << " No target node  " << QString::number(id) << " found";
//}


// Check if there is intention node in G
//    if(auto intention = G->get_node(current_intention_name); intention.has_value())
//    {
//        // check if there is a path_node and delete it
//        if (auto path = G->get_node(current_path_name); path.has_value())
//        {
//            if (G->delete_node(path.value().id()))
//                qInfo() << __FUNCTION__ << "Node " << QString::fromStdString(current_path_name) << " deleted ";
//            else
//                qInfo() << __FUNCTION__ << "Error deleting node " << QString::fromStdString(current_path_name);
//        }
//        // check if there is a current_intention attribute in intention node
//        else if (std::optional<std::string> plan = G->get_attrib_by_name<current_intention_att>(intention.value()); plan.has_value())
//        {
//            Plan plan_o = Plan(plan.value());
//            if (plan_o.action == Plan::Actions::BOUNCER)
//            {
//                G->delete_node(intention.value().id());
//                auto t= G->get_node_edges_by_type(intention.value(), "chocachoca_action");
//                for(const auto &tr_edge : t)
//                    G->delete_edge(tr_edge.from(), tr_edge.to(), "goto_action");
//
//                if (auto robot_node = G->get_node(robot_name); robot_node.has_value())
//                {
//                    G->add_or_modify_attrib_local<robot_ref_adv_speed_att>(robot_node.value(), (float) 0.0);
//                    G->add_or_modify_attrib_local<robot_ref_rot_speed_att>(robot_node.value(), (float) 0.0);
//                    G->update_node(robot_node.value());
//                }
//                else
//                {
//                    qWarning() << __FUNCTION__ << "No robot node found";
//                }
//            }
//        }
//
//        /*if( auto target_room_edges = G->get_node_edges_by_type(intention.value(), "goto_action"); not target_room_edges.empty())
//        {
//            cout<<"EDGE"<<endl;
//            for (const auto &tr_edge : target_room_edges)
//                G->delete_edge(tr_edge.from(), tr_edge.to(), "goto_action");
//        }
//        if (G->delete_node(intention.value().id()))
//            qInfo() << __FUNCTION__ << "Node " << QString::fromStdString(current_intention_name) << " deleted ";
//        else
//            qInfo() << __FUNCTION__ << "Error deleting node " << QString::fromStdString(current_intention_name);
//        send_command_to_robot(std::make_tuple(0.f,0.f,0.f));   //adv, side, rot*/
//    }
