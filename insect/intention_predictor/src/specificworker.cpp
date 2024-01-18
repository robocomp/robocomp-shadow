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
    cone_radius = std::stof(params["cone_radius"].value);
    cone_angle = std::stof(params["cone_angle"].value);
    qInfo() << "Cone radius: " << cone_radius;
    qInfo() << "Cone angle: " << cone_angle;
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

        wanted_person = Person(gridder_proxy);
//        wanted_person.set_target_element(true);
//        wanted_person.init_item(&viewer->scene, 0.f, 1000.f, 0.f);

        // mouse
        connect(viewer, &AbstractGraphicViewer::new_mouse_coordinates, [this](QPointF p)
        {
            qInfo() << "[MOUSE] New click left arrived:" << p;
//            QGraphicsItem *item = viewer->scene.itemAt(p, QTransform());
            // Check item at position corresponding to person representation (person can't be selected if cone layes is superposed)
            QList<QGraphicsItem *> itemsAtPosition = viewer->scene.items(p, Qt::IntersectsItemShape, Qt::DescendingOrder, QTransform());
            QGraphicsItem *selectedItem = nullptr;
            for (QGraphicsItem *item : itemsAtPosition) {
                if (item->pos() != QPointF(0, 0)) {
                    selectedItem = item;
                    break;
                }
            }
            if (selectedItem == nullptr) return;
            // check person with the same item
            for (auto &person: people)
            {
                if (person.get_item() == selectedItem)
                {
                    qInfo() << "Target clicked";
                    person.set_target_element(true);
                    try
                    {
                        segmentatortrackingpub_pubproxy->setTrack(person.get_target());
                    }
                    catch (const Ice::Exception &e)
                    { std::cout << "Error setting target" << e << std::endl; }
                }
                else
                {
                    person.set_target_element(false);
                }
            }
        });
        //Right click
        connect(viewer, &AbstractGraphicViewer::right_click, [this](QPointF p)
        {
            qInfo() << "[MOUSE] New click right arrived:" << p;
            for (auto &person: people)
            {
                person.set_target_element(false);
            }
            try
            {
                segmentatortrackingpub_pubproxy->setTrack(RoboCompVisualElementsPub::TObject{.id = -1});
            }
            catch (const Ice::Exception &e)
            { std::cout << "Error setting target" << e << std::endl; }
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

    // Read room elements from buffer_room_elements
    auto re_ = buffer_room_elements.try_get();
    if (not re_.has_value())  {   /*qWarning() << "No data VisualElements";*/ return; }
    auto re = re_.value();

    process_room_elements(re);
}

//////////////////////////////// SpecificWorker /////////////////////////////////////////////////
void SpecificWorker::process_visual_elements(const RoboCompVisualElementsPub::TData &data)
{
    std::vector<RoboCompVisualElementsPub::TObject> remaining_objects;
    // Check if there is data in data.objects
    if (data.objects.empty()) {qWarning() << "No VE data"; return;}
    // Print update phase
//    qInfo() << "Update phase";
    for (const auto &object: data.objects)
    {
        // Check if the object is a person
        if (object.type != 0) continue;
        // Check if the person is already in the people vector
        if(auto it = std::ranges::find_if(people, [&object](const Person &p) { return p.get_id() == object.id; }); it != people.end())
        {
            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - it->get_insertion_time()).count() > 1)
            {
                if(it->get_item() == nullptr)
                {
                    // Initialize the item
                    it->init_item(&viewer->scene, std::stof(object.attributes.at("x_pos")), std::stof(object.attributes.at("y_pos")), std::stof(object.attributes.at("orientation")), cone_radius, cone_angle);
                }

                else
                {
                    // Print update
                    // qInfo() << "Update person with id: " << object.id;
                    // Update the attributes of the person
                    it->update_attributes(data.objects);
                    // Check if the person is inside the pilar cone
                    it->is_inside_pilar_cone(data.objects);
                    // Draw the paths
                    if(it->is_target_element())
                        it->draw_paths(&viewer->scene, false, true);
                    else
                        it->draw_paths(&viewer->scene, false, false);
                }
            }
            it->update_last_update_time();
        }
        // Emplace the object in the remaining objects vector
        else
        {
            remaining_objects.push_back(object);
        }
    }
    for (const auto &object: remaining_objects)
    {
        // Create a new person
        Person new_person(gridder_proxy);
        new_person.set_person_data(object);
        new_person.set_insertion_time();
        new_person.update_last_update_time();
        // Add the person to the people vector
        people.emplace_back(new_person);
    }
    // Print insert phase
//    qInfo() << "Remove phase";
    // Check if last time a person was updated is more than 2 seconds
    for (auto &person: people)
    {
        if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - person.get_last_update_time()).count() > 1)
        {
//            qInfo() << "Person with id: " << person.get_id() << " has been removed";
            person.draw_paths(&viewer->scene, true, false);
            person.remove_item(&viewer->scene);
            people.erase(std::ranges::find_if(people, [&person](const Person &p) { return p.get_id() == person.get_id(); }));
        }
    }
}
void SpecificWorker::process_room_elements(const RoboCompVisualElementsPub::TData &data) {
    // Check if there is data in data.objects
    if (data.objects.empty())
    {
        qWarning() << "No rooms data";
        return;
    }
    //iterate over the objects and draw the rooms
    for (const auto &o : data.objects)
        if(o.attributes.at("name") == "room")
            draw_room(o);
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
void SpecificWorker::draw_room(const RoboCompVisualElementsPub::TObject &obj)
{
    //check if obj.attributes.contains the key name if it does print the value
    if(obj.attributes.contains("name"))
    {
        if (obj.attributes.at("name") == "room")
        {
            //save the attributes of the room width, depth,height,center_x,center_y,rotation
            float width = std::stof(obj.attributes.at("width"));
            float depth = std::stof(obj.attributes.at("depth"));
            float height = std::stof(obj.attributes.at("height"));
            float center_x = std::stof(obj.attributes.at("center_x"));
            float center_y = std::stof(obj.attributes.at("center_y"));
            float rotation = std::stof(obj.attributes.at("rotation"));

            static QGraphicsRectItem *item = nullptr;

            if (item != nullptr)
                viewer->scene.removeItem(item);

            item = viewer->scene.addRect(-width / 2, -depth / 2, width, depth, QPen(QColor("black"),50));
            item->setPos(QPointF(center_x, center_y));
            item->setRotation(qRadiansToDegrees(rotation));
        }
        else
            qWarning() << "The object by parameter is not a room";
    }
    else
        qWarning() << "The object does not contain the key name";

}
//////////////////////////////// Interfaces /////////////////////////////////////////////////
void SpecificWorker::VisualElementsPub_setVisualObjects(RoboCompVisualElementsPub::TData data)
{
    if (data.objects.empty())
        return;
    if(data.publisher == "forcefield")
        buffer_room_elements.put(std::move(data));
    else
        buffer_visual_elements.put(std::move(data));
}
