//
// Created by robolab on 1/16/24.
//

#include "personcone.h"

#include <utility>
#include "genericworker.h"


PersonCone::PersonCone(RoboCompGridder::GridderPrxPtr g_proxy)
{
    gridder_proxy = g_proxy;
}
PersonCone::PersonCone()
{
    gridder_proxy = nullptr;
}
void PersonCone::init_item(QGraphicsScene *scene, float x, float y, float angle, float cone_radius, float cone_angle, const std::string& name) // Radius in mm, cone_angle in rad
{
    item = scene->addEllipse(-200, -200, 400, 400, QPen(QColor("black"), 20), QBrush("orange"));
    auto line = scene->addLine(0, 0, 0, 200, QPen(QColor("black"), 10, Qt::SolidLine, Qt::RoundCap));
    line->setParentItem(item);
    item->setPos(x, y);
    item->setRotation(qRadiansToDegrees(angle));
    item->setZValue(100);  // set it higher than the grid

    // Create pilar cone as an isosceles triangle with base at item origin and bisector orientation aligned with item orientation
    //qInfo() << "cone_radius: " << cone_radius << " cone_angle: " << cone_angle;
    float x_cone = cone_radius * sin(cone_angle / 2.f);
    float y_cone = cone_radius * cos(cone_angle / 2.f);
    // Set pilar cone up and down to item

    pilar_cone << QPointF(0, 0) << QPointF(x_cone, y_cone) << QPointF(-x_cone, y_cone);
    auto pilar_cone_item = scene->addPolygon(this->pilar_cone, QPen(QColor("green"), 20, Qt::SolidLine, Qt::RoundCap));
    pilar_cone_item->setParentItem(item);

    // add a text item with the id
//    auto text = scene->addText(QString::number(dsr_id));
    auto text = scene->addText(QString::fromStdString(name));
    text->setParentItem(item);
    text->setPos(-text->boundingRect().width()*5, text->boundingRect().height()*17
    );
    text->setDefaultTextColor(QColor("black"));
    text->setScale(10);
    // Create a QTransform for vertical reflection
    QTransform transform; transform.scale(1, -1);
    text->setTransform(transform);
}
// Method to check if there is an object in a TObjects list with the same id
void PersonCone::update_attributes(float x, float y, float angle)
{
    item->setPos(x, y);
    item->setRotation(qRadiansToDegrees(angle - atan2(x, y)) + 180);
    // Map Pilar cone to parent rotation and translation
//    auto pilar_cone_without_conv = this->pilar_cone;
//    pilar_cone = item->mapToParent(pilar_cone_without_conv);
}
// Method to check if there are TObjects inside the pilar cone
bool PersonCone::is_inside_pilar_cone(const QPolygonF &pilar_cone, float x, float y)
{
    if (pilar_cone.containsPoint(QPointF(x, y), Qt::OddEvenFill))
    {
        return true;
    }
    return false;
}
void PersonCone::set_dsr_id(long int id)
{
    dsr_id = id;
}
long int PersonCone::get_dsr_id() const
{
    return dsr_id;
}
QGraphicsItem* PersonCone::get_item() const
{
    return item;
}
QPolygonF PersonCone::get_pilar_cone() const
{
    return pilar_cone;
}
// Method for getting intentions size
int PersonCone::get_intentions_size() const
{
    return intentions.size();
}

//////////////////////////////// Draw ///////////////////////////////////////////////////////
// Method to remove item from scene
void PersonCone::remove_item(QGraphicsScene *scene)
{
    scene->removeItem(item);
    delete item; item = nullptr;
}
void PersonCone::set_intention(const std::tuple<float, float, float, float> &person_point, const std::tuple<float, float, float, float> &element_point, const std::string &target_name)
{
    Intention i {.target_name = target_name};
    intentions.emplace_back(i);
}
//void PersonCone::set_intention(const std::tuple<float, float, float, float> &person_point, const std::tuple<float, float, float, float> &element_point, const std::string &target_name)
//{
//    auto [x_person, y_person, z_person, ang_person] = person_point;
//    auto [x_element, y_element, z_element, ang_element] = element_point;
//    try
//    {
//        auto result = gridder_proxy->getPaths(RoboCompGridder::TPoint{.x=x_person, .y=y_person, .radius=500},
//                                              RoboCompGridder::TPoint{.x=x_element, .y=y_element, .radius=500},
//                                              1, true, true, false);
//        if (result.valid and not result.paths.empty())
//        {
//            std::vector<std::vector<Eigen::Vector2f>> paths;
//            for(const auto &p: result.paths)
//            {
//                std::vector<Eigen::Vector2f> path;
//                for (const auto &point: p)
//                    path.emplace_back(point.x, point.y);
//                paths.emplace_back(path);
//            }
//
//            Intention i {.target_name = target_name, .paths = paths};
//            intentions.emplace_back(i);
//        }
//    }
//    catch (const Ice::Exception &e)
//    { std::cout << __FUNCTION__ << " Error reading plans from Gridder. " << e << std::endl; }
//}
// Method for getting intention pointer
std::optional<PersonCone::Intention*> PersonCone::get_intention(const std::string &target_name)
{
    auto it = std::find_if(intentions.begin(), intentions.end(), [&target_name](const Intention &i){ return i.target_name == target_name; });
    if(it != intentions.end())
        return &(*it);
    else
        return {};
}
//Method for updating intention path
std::optional<std::vector<std::vector<Eigen::Vector2f>>> PersonCone::get_paths(const std::tuple<float, float, float, float> &person_point, const std::tuple<float, float, float, float> &element_point, const std::string &target_name)
{
    auto [x_person, y_person, z_person, ang_person] = person_point;
    auto [x_element, y_element, z_element, ang_element] = element_point;
    try
    {
        auto result = gridder_proxy->getPaths(RoboCompGridder::TPoint{.x=x_person, .y=y_person, .radius=500},
                                              RoboCompGridder::TPoint{.x=x_element, .y=y_element, .radius=500},
                                              1, true, true, false);
        if (result.valid and not result.paths.empty())
        {
            std::vector<std::vector<Eigen::Vector2f>> paths;
            for(const auto &p: result.paths)
            {
                std::vector<Eigen::Vector2f> path;
                for (const auto &point: p)
                    path.emplace_back(point.x, point.y);
                paths.emplace_back(path);
            }
            return paths;
        }
        else
            return {};
    }
    catch (const Ice::Exception &e)
    { std::cout << __FUNCTION__ << " Error reading plans from Gridder. " << e << std::endl; return {};}
}
void PersonCone::remove_intention(const std::string &target_name)
{
    intentions.erase(std::remove_if(intentions.begin(), intentions.end(), [&target_name](const Intention &i){ return i.target_name == target_name; }), intentions.end());
}
// Method to draw path
void PersonCone::draw_paths(QGraphicsScene *scene, bool erase_only)
{
    for(auto p : points)
    {scene->removeItem(p); delete p; }
    points.clear();
    if(erase_only) return;
    float s = 100;
    QColor color;
    for(const auto &i: intentions)
        for(const auto &path: i.paths)
        {
            // check if path is in the last vector position
            if(i.paths.back() == path)
                color = QColor("red");
            else
                color = QColor("green");
            for(const auto &p: path)
            {
                auto ptr = scene->addEllipse(-s/2, -s/2, s, s, QPen(color), QBrush(color));
                ptr->setPos(QPointF(p.x(), p.y()));
                points.push_back(ptr);
            }
        }
}
