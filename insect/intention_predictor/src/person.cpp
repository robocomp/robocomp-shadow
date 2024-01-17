//
// Created by robolab on 1/16/24.
//

#include "person.h"
#include "genericworker.h"

Person::Person(RoboCompGridder::GridderPrxPtr g_proxy)
{
    gridder_proxy = g_proxy;
}
Person::Person()
{
    gridder_proxy = nullptr;
}
void Person::init_item(QGraphicsScene *scene, float x, float y, float angle)
{
    item = scene->addEllipse(-200, -200, 400, 400, QPen("orange"), QBrush("orange"));
    auto line = scene->addLine(0, 0, 0, 200, QPen(QColor("black"), 10, Qt::SolidLine, Qt::RoundCap));
    line->setParentItem(item);
    item->setPos(x, y);
    item->setRotation(angle);

    // Create pilar cone as a isosceles triangle with base at item origin and bisector orientation aligned with item orientation

    pilar_cone << QPointF(0, 0) << QPointF(2000, 4000) << QPointF(-2000, 4000);
    auto pilar_cone_item = scene->addPolygon(this->pilar_cone, QPen(QColor("magenta"), 10, Qt::SolidLine, Qt::RoundCap));
    pilar_cone_item->setParentItem(item);
}
// Method to check if there is an object in a TObjects list with the same id
void Person::update_attributes(const RoboCompVisualElementsPub::TObjects &list)
{
    // Check if TAttributes in target is empty
    if (target.attributes.empty())
    {
        if (auto r = std::ranges::find_if(list, [this](const auto &o) { return o.type == 0; }); r != list.end())
        {
            target = *r;
            return;
        }
        else
        {
            qWarning() << __FUNCTION__ << "No person found to fill empty target";
            return;
        }
    }
    // Check if target is in list and update
    if(auto r = std::ranges::find_if(list, [this](const auto &o) { return o.id == target.id; }); r!= list.end())
    {
        if (r->attributes.contains("x_pos") and
            r->attributes.contains("y_pos") and
            r->attributes.contains("orientation"))
        {
            target.attributes["x_pos"] = r->attributes.at("x_pos");
            target.attributes["y_pos"] = r->attributes.at("y_pos");
            target.attributes["orientation"] = r->attributes.at("orientation");

            item->setPos(std::stof(r->attributes.at("x_pos")), std::stof(r->attributes.at("y_pos")));
            item->setRotation(qRadiansToDegrees(std::stof(r->attributes.at("orientation")) - atan2(std::stof(r->attributes.at("x_pos")), std::stof(r->attributes.at("y_pos"))))+180);
        }
        else
        {
            qWarning() << __FUNCTION__ << "No x_pos or y_pos in target attributes";
            return;
        }
    }
}
// Method to check if there are TObjects inside the pilar cone
void Person::is_inside_pilar_cone(const RoboCompVisualElementsPub::TObjects &list)
{
    // Check if list is empty
    if (list.empty())
    {
        qWarning() << __FUNCTION__ << "List is empty";
        return;
    }
    // Map Pilar cone to parent rotation and translation
    auto pilar_cone_ = this->pilar_cone;
    auto pilar_cone_conv = item->mapToParent(pilar_cone_);
    // Print pilar cone points
//    for (const auto &p : pilar_cone_conv)
//    {
//        qInfo() << p.x() << " " << p.y();
//    }
    // Clear paths vector
    paths.clear();
    // Check if there are objects inside the pilar cone
    for (const auto &o : list)
    {
        if (o.attributes.contains("x_pos") and
            o.attributes.contains("y_pos") and
            o.attributes.contains("orientation") and
            o.id != target.id)
        {
            if (pilar_cone_conv.containsPoint(QPointF(std::stof(o.attributes.at("x_pos")), std::stof(o.attributes.at("y_pos"))), Qt::OddEvenFill))
            {
//                qInfo() << o.type << "TRUE" << std::stof(o.attributes.at("x_pos")) << " " << std::stof(o.attributes.at("y_pos"));
                if (auto path = order_paths(o); path.has_value())
                {
                    paths.push_back(path.value());
                }
            }
//            else
//            {
//                qInfo() << "FALSE";
//            }
        }
    }
}
std::optional<std::pair<int, std::vector<Eigen::Vector2f>>> Person::order_paths(const RoboCompVisualElementsPub::TObject &object)
{
    try
    {
        auto result = gridder_proxy->getPaths(RoboCompGridder::TPoint{std::stof(target.attributes.at("x_pos")), std::stof(target.attributes.at("y_pos"))},
                                              RoboCompGridder::TPoint{std::stof(object.attributes.at("x_pos")), std::stof(object.attributes.at("y_pos"))},
                                              1,
                                              true,
                                              true);
        if(not result.valid or result.paths.empty())   //TODO: try a few times
        {
            qWarning() << __FUNCTION__ << "No path found while initializing current_path";
            return {};
        }
        std::vector<Eigen::Vector2f> path;
        for(const auto &p: result.paths.front())
            path.emplace_back(p.x, p.y);
        return std::make_pair(object.id, path);
    }
    catch (const Ice::Exception &e)
    {
        std::cout << "Error reading plans from Gridder" << e << std::endl;
        return {};
    }
}
void Person::set_target_element(bool value)
{
    is_target = value;
}
int Person::get_id() const
{
    return target.id;
}
//////////////////////////////// Draw ///////////////////////////////////////////////////////
void Person::draw_paths(QGraphicsScene *scene, bool erase_only, bool wanted_person)
{
    static std::vector<QGraphicsEllipseItem*> points;
    for(auto p : points)
    { scene->removeItem(p); delete p; }
    points.clear();

    if(erase_only) return;
    QPen pen;
    QBrush brush;
    if(wanted_person)
    {
        pen = QPen("red");
        brush = QBrush("red");
    }
    else
    {
        pen = QPen("black");
        brush = QBrush("black");
    }
    float s = 100;
    for(const auto &p: paths)
    {
        for(const auto &pp: p.second)
        {
            auto ptr = scene->addEllipse(-s/2, -s/2, s, s, pen, brush);
            ptr->setPos(QPointF(pp.x(), pp.y()));
            points.push_back(ptr);
        }
    }
}
// Method to remove item from scene
void Person::remove_item(QGraphicsScene *scene)
{
    scene->removeItem(item);
    delete item;
}