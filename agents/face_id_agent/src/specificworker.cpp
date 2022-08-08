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

/**
* \brief Default constructor
*/
SpecificWorker::SpecificWorker(TuplePrx tprx, bool startup_check) : GenericWorker(tprx)
{
	this->startup_check_flag = startup_check;
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
//	THE FOLLOWING IS JUST AN EXAMPLE
//	To use innerModelPath parameter you should uncomment specificmonitor.cpp readConfig method content
//	try
//	{
//		RoboCompCommonBehavior::Parameter par = params.at("InnerModelPath");
//		std::string innermodel_path = par.value;
//		innerModel = std::make_shared(innermodel_path);
//	}
//	catch(const std::exception &e) { qFatal("Error reading config params"); }





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
	this->Period = period;
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
		connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
//		connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
//		connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_attrs_slot);
//		connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
//		connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

		// Graph viewer
		using opts = DSR::DSRViewer::view;
		int current_opts = 0;
		opts main = opts::none;
		if(tree_view)
		{
		    current_opts = current_opts | opts::tree;
		}
		if(graph_view)
		{
		    current_opts = current_opts | opts::graph;
		    main = opts::graph;
		}
		if(qscene_2d_view)
		{
		    current_opts = current_opts | opts::scene;
		}
		if(osg_3d_view)
		{
		    current_opts = current_opts | opts::osg;
		}
		graph_viewer = std::make_unique<DSR::DSRViewer>(this, G, current_opts, main);
		setWindowTitle(QString::fromStdString(agent_name + "-") + QString::number(agent_id));

		this->Period = period;
		timer.start(Period);
        hide();
	}
}

void SpecificWorker::compute()
{
    // check if somebody to recognize exists
    if (auto person_o = person_buffer.try_get(); person_o.has_value())
    {
            std::cout << "PERSON ID:" << person_o.value() << std::endl;
            if(auto person_node = G->get_node(person_o.value()); person_node.has_value())
            {
                while(found_person == false)
                {
                    if(auto person_gray_roi = get_person_ROI_from_node(person_node.value()); person_gray_roi.has_value())
                    {
                        // Get Face ID image from G
                        if(auto face_id_image = get_face_ID_image(); face_id_image.has_value())
                        {
                            // Get recognised people from Face ID
                            auto recognised_people = this->realsensefaceid_proxy->authenticate();
                            // If somebody was recognised, read each one to associate the name to people from G
                            if (recognised_people.size() > 0)
                            {
                                for (auto identified_person: recognised_people)
                                {
                                    // Get face image with Face ID image and rectangle data given by Face ID component
                                    auto personROI = face_id_image.value()(cv::Rect(identified_person.faceROI.x, identified_person.faceROI.y, identified_person.faceROI.width, identified_person.faceROI.height));
                                    // Generate gray ROI to avoid matching problems at calculating correlation, due to image color differences between Face ID and depth cameras
                                    cv::Mat LAB_personROI, LAB_personROI_CH_L, LAB_personROI_CH_A, LAB_personROI_CH_B;
                                    cv::cvtColor(personROI, LAB_personROI, cv::COLOR_RGB2Lab);
                                    extractChannel(LAB_personROI, LAB_personROI_CH_L, 0);
                                    cv::imwrite("LAB_personROI.png", LAB_personROI_CH_L);
                                    // Get max correlation point in image
                                    auto max_correlation_point = get_max_correlation_point(LAB_personROI_CH_L, person_gray_roi.value());
                                    auto is_point_in_face = check_if_max_correlation_in_face(person_gray_roi.value(), max_correlation_point);
                                    cv::circle(person_gray_roi.value(), max_correlation_point, 10, cv::Scalar(255, 0, 0), 2);
                                    cv::imwrite("person_LAB_face_roi.png", person_gray_roi.value());
                                    if(is_point_in_face)
                                    {
//                                        if(auto person_name = G->get_attrib_by_name<person_name_att>(person_node.value()); !person_name.has_value())
//                                        {
                                            found_person = true;
                                            try_counter_with_match = 0;
                                            G->add_or_modify_attrib_local<person_name_att>(person_node.value(), identified_person.name);
                                            G->update_node(person_node.value());
                                            return;
//                                        }
//                                    else
//                                    {
//                                        if(person_name.value() !=)
//                                    }
                                    }

                                }
                                qInfo() << "NO SE HA DETECTADO A LA PERSONA";
                                try_counter_with_match += 1;
                                if(try_counter_with_match == 5)
                                {
                                    try_counter_with_match = 0;
                                    return;
                                }
                            }
                            else
                            {
                                qInfo() << "PERSONA DESCONOCIDA";
                                try_counter_without_match += 1;
                                if(try_counter_without_match == 5)
                                {
                                    try_counter_with_match = 0;
                                    return;
                                }
                            }
                        } else return;
                    } else return;
                }
            } else return;
    } else return;
}

std::optional<cv::Mat> SpecificWorker::get_person_ROI_from_node(DSR::Node person_node)
{
    if (auto person_ROI_data_att = G->get_attrib_by_name<person_image_att>(
                person_node); person_ROI_data_att.has_value())
    {
        if (auto person_ROI_width_att = G->get_attrib_by_name<person_image_width_att>(
                    person_node); person_ROI_width_att.has_value())
        {
            if (auto person_ROI_height_att = G->get_attrib_by_name<person_image_height_att>(
                        person_node); person_ROI_height_att.has_value())
            {
                auto leader_ROI_data = person_ROI_data_att.value().get();

                cv::Mat node_person_roi = cv::Mat(person_ROI_width_att.value(),
                                                  person_ROI_height_att.value(), CV_8UC3,
                                                  &leader_ROI_data[0]);
                cv::Mat person_LAB_face_roi, person_LAB_face_roi_CH_L;
                cv::cvtColor(node_person_roi, person_LAB_face_roi, cv::COLOR_RGB2Lab);
                extractChannel(person_LAB_face_roi, person_LAB_face_roi_CH_L, 0);
                return person_LAB_face_roi_CH_L;
            }
            else return{};
        }
        else return{};
    }
    else return{};
}

std::optional<cv::Mat> SpecificWorker::get_face_ID_image()
{
    if(auto face_id_camera = G->get_node("giraff_camera_face_id"); face_id_camera.has_value())
    {
        if(auto id_image_height = G->get_attrib_by_name<cam_rgb_height_att>(face_id_camera.value()); id_image_height.has_value())
        {
            if (auto id_image_width = G->get_attrib_by_name<cam_rgb_width_att>(
                        face_id_camera.value()); id_image_width.has_value())
            {
                if (auto id_image_data = G->get_attrib_by_name<cam_rgb_att>(face_id_camera.value()); id_image_data.has_value())
                {
                    auto id_image_data_at = id_image_data.value().get();
                    return cv::Mat(id_image_height.value(), id_image_width.value(), CV_8UC3,
                                   &id_image_data_at[0]);
                }
                else return{};
            }
            else return{};
        }
        else return{};
    }
    else return{};
}

cv::Point2i SpecificWorker::get_max_correlation_point(cv::Mat face_person_roi, cv::Mat person_roi)
{
    cv::Mat result;
    cv::matchTemplate(person_roi, face_person_roi, result, cv::TM_CCOEFF);
    cv::Point2i max_point_correlated, min_point_correlated, roi_center;
    double max_value, min_value;
    cv::minMaxLoc(result, &min_value, &max_value, &min_point_correlated, &max_point_correlated, cv::Mat());
    return cv::Point2i (max_point_correlated.x + face_person_roi.cols / 2, max_point_correlated.y + face_person_roi.rows / 2);
}

bool SpecificWorker::check_if_max_correlation_in_face(cv::Mat person_roi, cv::Point2i max_corr_point)
{
    cv::CascadeClassifier cascade;
    std::string cascadePath = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml";
    cascade.load(cascadePath);
    if (auto detected_roi = detectAndDraw(person_roi, cascade, scale); detected_roi.has_value())
    {
        if (detected_roi.value().x < max_corr_point.x && max_corr_point.x < detected_roi.value().x + detected_roi.value().width)
        {
            if (detected_roi.value().y < max_corr_point.y && max_corr_point.y < detected_roi.value().y + detected_roi.value().height)
            {
                qInfo() << "################ COINCIDE CON LA CARA ################";
                return true;
            }
            else return false;
        }
        else return false;
    }
    else return false;
}

std::optional<RoboCompRealSenseFaceID::ROIdata> SpecificWorker::detectAndDraw( cv::Mat& img, cv::CascadeClassifier& cascade, double scale)
{
    std::vector <cv::Rect> faces;
    cv::Mat smallImg;
    double fx = 1 / scale;
    // Resize the Grayscale Image
    cv::resize(img, smallImg, cv::Size(), fx, fx, cv::INTER_LINEAR);
    cv::equalizeHist(smallImg, smallImg);
    // Detect faces of different sizes using cascade classifier
    cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
    // Draw circles around the faces
    if(faces.size() > 0)
    {
        std::vector<cv::Mat> vector_faces;
        cv::Rect r = faces[0];
        RoboCompRealSenseFaceID::ROIdata face_data;
        face_data.x = r.x*scale;
        face_data.y = r.y*scale;
        face_data.width = r.width;
        face_data.height = r.height;
        return face_data;
    }
    else return{};

//        int radius;
//        std::string face_id = to_string(i);
//        cv::imwrite( face_id + ".png", img(cv::Rect(cvRound(r.x*scale), cvRound(r.y*scale), r.width, r.height)) );

//        double aspect_ratio = (double)r.width/r.height;
//        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
//        {
//            center.x = cvRound((r.x + r.width*0.5)*scale);
//            center.y = cvRound((r.y + r.height*0.5)*scale);
//            radius = cvRound((r.width + r.height)*0.25*scale);
//            circle( img, center, radius, color, 3, 8, 0 );
//        }
//        else
//        cv::rectangle( img, cv::Point(cvRound(r.x*scale), cvRound(r.y*scale)),
//                       cv::Point(cvRound((r.x + r.width-1)*scale),
//                                   cvRound((r.y + r.height-1)*scale)), color, 3, 8, 0);
//        if( nestedCascade.empty() )
//            continue;
//        smallImgROI = smallImg( r );
//        // Detection of eyes int the input image
//        nestedCascade.detectMultiScale( smallImgROI, nestedObjects, 1.1, 2,
//                                        0|cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30) );
//        // Draw circles around eyes
//        for ( size_t j = 0; j < nestedObjects.size(); j++ )
//        {
//            cv::Rect nr = nestedObjects[j];
//            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
//            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
//            radius = cvRound((nr.width + nr.height)*0.25*scale);
//            cv::circle( img, center, radius, color, 3, 8, 0 );
//        }
//    }

    // Show Processed Image with detected faces
}

void SpecificWorker::modify_node_slot(std::uint64_t id, const std::string &type)
{
    if(type=="intention")
    {
        qInfo() << "################ SE ENTERA ################";
        if (auto intention = G->get_node(id); intention.has_value())
        {
            std::optional<std::string> plan = G->get_attrib_by_name<current_intention_att>(intention.value());
            if (plan.has_value())
            {
                Plan my_plan(plan.value());
                if(my_plan.get_action() == "RECOGNIZE_PEOPLE")
                {
                    auto person_id = my_plan.get_attribute("person_node_id");
                    uint64_t value;
                    std::istringstream iss(person_id.toString().toUtf8().constData());
                    iss >> value;
                    person_buffer.put(std::move(value));
                }
            }
        }
    }
}

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}

/**************************************/
// From the RoboCompCameraSimple you can call this methods:
// this->camerasimple_proxy->getImage(...)

/**************************************/
// From the RoboCompCameraSimple you can use this types:
// RoboCompCameraSimple::TImage

/**************************************/
// From the RoboCompRealSenseFaceID you can call this methods:
// this->realsensefaceid_proxy->authenticate(...)
// this->realsensefaceid_proxy->enroll(...)
// this->realsensefaceid_proxy->eraseAll(...)
// this->realsensefaceid_proxy->eraseUser(...)
// this->realsensefaceid_proxy->getQueryUsers(...)
// this->realsensefaceid_proxy->startPreview(...)
// this->realsensefaceid_proxy->stopPreview(...)

/**************************************/
// From the RoboCompRealSenseFaceID you can use this types:
// RoboCompRealSenseFaceID::UserData

