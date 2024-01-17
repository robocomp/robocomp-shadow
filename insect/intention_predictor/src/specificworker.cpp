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
#include "specificworker.h"
#include <cppitertools/enumerate.hpp>

/**
* \brief Default constructor
*/
SpecificWorker::SpecificWorker(TuplePrx tprx, bool startup_check) : GenericWorker(tprx)
{
	this->startup_check_flag = startup_check;
	// Uncomment if there's too many debug messages
	// but it removes the possibility to see the messages
	// shown in the console with qDebug()
//	QLoggingCategory::setFilterRules("*.debug=false\n");
}
/**
* \brief Default destructor
*/
SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
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

	return true;
}
void SpecificWorker::initialize(int period)
{
	std::cout << "Initialize worker" << std::endl;
	this->Period = 50;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
        // Viewer
        viewer = new AbstractGraphicViewer(this->frame, params.GRID_MAX_DIM);
        //QRectF(params.xMin, params.yMin, params.grid_width, params.grid_length));
        viewer->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, QColor("Blue"));
        viewer->show();

        // Lidar thread is created
        read_lidar_th = std::thread(&SpecificWorker::read_lidar,this);
        std::cout << __FUNCTION__ << " Started lidar reader" << std::endl;

//        wanted_person = Person(gridder_proxy);
//        wanted_person.init_item(&viewer->scene, 0.f, 1000.f, 0.f);

        // mouse
        connect(viewer, &AbstractGraphicViewer::new_mouse_coordinates, [this](QPointF p)
        {
            qInfo() << "[MOUSE] New click left arrived:" << p;
            return;
        });
        //Right click
        connect(viewer, &AbstractGraphicViewer::right_click, [this](QPointF p)
        {
            qInfo() << "[MOUSE] New click right arrived:" << p;
        });
        timer.start(Period);
	}
}
void SpecificWorker::compute()
{
    /// read LiDAR
    auto res_ = buffer_lidar_data.try_get();
    if (not res_.has_value())  {   /*qWarning() << "No data Lidar";*/ return; }
    auto points = res_.value();
    draw_lidar(points, params.LIDAR_LOW_DECIMATION_FACTOR);

    // Read visual elements from buffer_visual_elements
    auto ve_ = buffer_visual_elements.try_get();
    if (not ve_.has_value())  {   /*qWarning() << "No data VisualElements";*/ return; }
    auto ve = ve_.value();

    // Process visual elements
    process_visual_elements(ve);

}

//////////////////////////////// SpecificWorker /////////////////////////////////////////////////
void SpecificWorker::process_visual_elements(const RoboCompVisualElementsPub::TData &data)
{
    // Check if there is data in data.objects
    if (data.objects.empty()) {qWarning() << "No VE data"; return;}
    // Create a vector for storing updated people ids
    std::vector<int> updated_people_ids;

    // Iterate over the objects
    for (const auto &o : data.objects)
    {
        // Print element id
        std::cout << "Element id: " << o.id << std::endl;
        // Check if the object is a person
        if (o.type == 0)
        {
            // Check if the person is the wanted person
            if (o.id == wanted_person.get_id())
            {
                // Update the attributes of the person
                wanted_person.update_attributes(data.objects);
                // Check if the person is inside the pilar cone
                wanted_person.is_inside_pilar_cone(data.objects);
                // Draw the paths
                wanted_person.draw_paths(&viewer->scene, false, true);
            }
            else
            {
                // Check if the person is already in the people vector and store the iterator
                auto it = std::find_if(people.begin(), people.end(), [&o](const Person &p) { return p.get_id() == o.id; });
                // Check if the person is in the people vector and add it to the vector of iterators
                if (it != people.end())
                {
                    // Update the attributes of the person
                    it->update_attributes(data.objects);
                    // Check if the person is inside the pilar cone
                    it->is_inside_pilar_cone(data.objects);
                    // Draw the paths
                    it->draw_paths(&viewer->scene, false, false);
                    // Print the id of the person
                    std::cout << "Person id: " << it->get_id() << std::endl;
                    // Add the iterator to the vector of iterators
//                    updated_people_ids.push_back(it->get_id());
                }
                else
                {
                    // Create a new person
                    Person new_person(gridder_proxy);
                    // Initialize the item
                    new_person.init_item(&viewer->scene, std::stof(o.attributes.at("x_pos")), std::stof(o.attributes.at("y_pos")), std::stof(o.attributes.at("orientation")));
                    // Update the attributes of the person
                    new_person.update_attributes(data.objects);
                    // Check if the person is inside the pilar cone
                    new_person.is_inside_pilar_cone(data.objects);
                    // Draw the paths
                    new_person.draw_paths(&viewer->scene, false, false);
                    // Add the person to the people vector
                    people.push_back(new_person);
                    // Print new person inserted
                    std::cout << "New person inserted" << std::endl;
                }
            }
        }
    }
    for (const auto &id : updated_people_ids)
    {
        qInfo() << id << " ";
    }
    // Delete people that were not updated
//    people.erase(std::remove_if(people.begin(), people.end(), [&updated_people_ids](const Person &p) { return std::find(updated_people_ids.begin(), updated_people_ids.end(), p.get_id()) == updated_people_ids.end(); }), people.end());
}
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}
void SpecificWorker::read_lidar()
{
    auto wait_period = std::chrono::milliseconds (this->Period);
    while(true)
    {
        try
        {
            auto data = lidar3d_proxy->getLidarDataWithThreshold2d(params.LIDAR_NAME_LOW,
                                                                   params.MAX_LIDAR_LOW_RANGE,
                                                                   params.LIDAR_LOW_DECIMATION_FACTOR);
            // compute the period to read the lidar based on the current difference with the lidar period. Use a hysteresis of 2ms
            if (wait_period > std::chrono::milliseconds((long) data.period + 2)) wait_period--;
            else if (wait_period < std::chrono::milliseconds((long) data.period - 2)) wait_period++;
            std::vector<Eigen::Vector3f> eig_data(data.points.size());
            for (const auto &[i, p]: data.points | iter::enumerate)
                eig_data[i] = {p.x, p.y, p.z};
            buffer_lidar_data.put(std::move(eig_data));
        }
        catch (const Ice::Exception &e)
        { std::cout << "Error reading from Lidar3D" << e << std::endl; }
        std::this_thread::sleep_for(wait_period);
    }
} // Thread to read the lidar

//////////////////////////////// Draw ///////////////////////////////////////////////////////
void SpecificWorker::draw_lidar(const std::vector<Eigen::Vector3f> &points, int decimate)
{
    static std::vector<QGraphicsItem *> draw_points;
    for (const auto &p: draw_points)
    {
        viewer->scene.removeItem(p);
        delete p;
    }
    draw_points.clear();

    QPen pen = QPen(params.LIDAR_COLOR);
    QBrush brush = QBrush(params.LIDAR_COLOR);
    for (const auto &[i, p]: points |iter::enumerate)
    {
        // skip 2 out of 3 points
        if(i % decimate == 0)
        {
            auto o = viewer->scene.addRect(-20, 20, 40, 40, pen, brush);
            o->setPos(p.x(), p.y());
            draw_points.push_back(o);
        }
    }
}

//////////////////////////////// Interfaces /////////////////////////////////////////////////
void SpecificWorker::VisualElementsPub_setVisualObjects(RoboCompVisualElementsPub::TData data)
{
//    qInfo() << __FUNCTION__ << " New visual objects arrived";
    buffer_visual_elements.put(std::move(data));
}
