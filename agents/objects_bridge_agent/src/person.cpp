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
void Person::init_item(QGraphicsScene *scene, float x, float y, float angle, float cone_radius, float cone_angle) // Radius in mm, cone_angle in rad
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
    pilar_cone << QPointF(0, 0) << QPointF(x_cone, y_cone) << QPointF(-x_cone, y_cone);
    auto pilar_cone_item = scene->addPolygon(this->pilar_cone, QPen(QColor("green"), 20, Qt::SolidLine, Qt::RoundCap));
    pilar_cone_item->setParentItem(item);

    // add a text item with the id
    auto text = scene->addText(QString::number(target.id));
    text->setParentItem(item);
    text->setPos(-text->boundingRect().width()*5, text->boundingRect().height()*17
    );
    text->setDefaultTextColor(QColor("black"));
    text->setScale(10);
    // Create a QTransform for vertical reflection
    QTransform transform; transform.scale(1, -1);
    text->setTransform(transform);

}
void Person::set_person_data(RoboCompVisualElementsPub::TObject person)
{
    target = person;
}
// Method to check if there is an object in a TObjects list with the same id
void Person::update_attributes(const RoboCompVisualElementsPub::TObjects &list)
{
    // Check if TAttributes in target is empty
//    if (target.attributes.empty())
//    {
//        if (auto r = std::ranges::find_if(list, [this](const auto &o) { return o.type == 0; }); r != list.end())
//        {
//            target = *r;
//            return;
//        }
//        else
//        {
//            qWarning() << __FUNCTION__ << "No person found to fill empty target";
//            return;
//        }
//    }
    qInfo() << "Updating target attributes 1";
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
        }
        else
        {
            qWarning() << __FUNCTION__ << "No x_pos or y_pos in target attributes";
            return;
        }
    }
}
std::optional<float> Person::get_attribute(const string &attribute_name) const
{
    if (target.attributes.contains(attribute_name))
        return std::stof(target.attributes.at(attribute_name));
    else
        return {};
}
bool Person::set_attribute(const std::string &attribute_name, float value)
{
    if (target.attributes.contains(attribute_name))
        target.attributes.at(attribute_name) = std::to_string(value);
    else
        return false;
    return true;
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
                //if (auto path = search_for_paths(o); path.has_value())
                //{
                //    paths.push_back(path.value());
                //}
            }
        }
    }
}
std::optional<std::pair<int, std::vector<Eigen::Vector2f>>> Person::search_for_paths(const RoboCompVisualElementsPub::TObject &object)
{
    try
    {
        auto result = gridder_proxy->getPaths(RoboCompGridder::TPoint{std::stof(target.attributes.at("x_pos")), std::stof(target.attributes.at("y_pos"))},
                                              RoboCompGridder::TPoint{std::stof(object.attributes.at("x_pos")), std::stof(object.attributes.at("y_pos"))},
                                              1,
                                              true,
                                              true);
        qInfo() << __FUNCTION__ << "result: " << result.error_msg.c_str() << " to " << object.id;
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
    if(value)
        item->setBrush(QBrush("red"));
    else
        item->setBrush(QBrush("orange"));
}
void Person::update_last_update_time()
{
    auto now = std::chrono::high_resolution_clock::now();
    last_update_time = std::chrono::time_point_cast<std::chrono::seconds>(now).time_since_epoch().count();
}
uint64_t Person::get_last_update_time() const
{
    return last_update_time;
}
int Person::get_id() const
{
    return target.id;
}
QGraphicsItem* Person::get_item() const
{
    return item;
}
RoboCompVisualElementsPub::TObject Person::get_target() const
{
    RoboCompVisualElementsPub::TObject aux = this->target;
    if (aux.attributes.contains("x_pos") and aux.attributes.contains("y_pos"))
    {
        auto x = std::stof(aux.attributes.at("x_pos"));
        auto y = std::stof(aux.attributes.at("y_pos"));
        // compute a new point closer to the robot by 500mm
        Eigen::Vector2f t{x, y};
        if(t.norm() > 500.f)    // TODO: move to params
        {
            t = t.normalized() * (t.norm() - 500.f);
            aux.attributes["x_pos"] = std::to_string(t.x());
            aux.attributes["y_pos"] = std::to_string(t.y());
            return aux;
        }
    }
    return target;
}
bool Person::is_target_element() const
{
    return is_target;
}
// Method to set the insertion time
void Person::set_insertion_time()
{
    // Convert std::chrono::high_resolution_clock::now() to uint64_t
    auto now = std::chrono::high_resolution_clock::now();
    insertion_time = std::chrono::time_point_cast<std::chrono::seconds>(now).time_since_epoch().count();
}
// Method to get the insertion time
uint64_t Person::get_insertion_time() const
{
    return insertion_time;
}
void Person::set_dsr_id(long int id)
{
    dsr_id = id;
}
long int Person::get_dsr_id() const
{
    return dsr_id;
}

//////////////////////////////// Draw ///////////////////////////////////////////////////////
void Person::draw_paths(QGraphicsScene *scene, bool erase_only, bool wanted_person) const
{
    static std::vector<QGraphicsPolygonItem*> points;
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

    const double arrowSize = 100;

    for (const auto &p : paths) {
        for (auto it = p.second.begin(); it != p.second.end(); ++it) {
            if (std::next(it) != p.second.end()) {
                // Puntos actual y siguiente
                Eigen::Vector2f p1 = *it;
                Eigen::Vector2f p2 = *std::next(it);

                // Convertir Eigen::Vector2f a QPointF
                QPointF qp1(p1.x(), p1.y());
                QPointF qp2(p2.x(), p2.y());

                // Calcular la línea entre los puntos actual y siguiente
                QLineF line(qp1, qp2);

                // Crear una flecha como un polígono
                QPolygonF arrowHead;
                arrowHead << line.p2()
                          << line.p2() - QPointF(sin(line.angle() * M_PI / 180 + M_PI / 3) * arrowSize,
                                                 cos(line.angle() * M_PI / 180 + M_PI / 3) * arrowSize)
                          << line.p2() - QPointF(sin(line.angle() * M_PI / 180 + M_PI - M_PI / 3) * arrowSize,
                                                 cos(line.angle() * M_PI / 180 + M_PI - M_PI / 3) * arrowSize);

                // Agregar la línea al escenario
//                scene->addLine(line, pen);

                // Agregar la cabeza de la flecha (el polígono) al escenario
                auto ptr = scene->addPolygon(arrowHead, pen, brush);
                points.push_back(ptr);
            }
        }
    }
//    for(const auto &p: paths)
//    {
//        for(const auto &pp: p.second)
//        {
//            auto ptr = scene->addEllipse(-s/2, -s/2, s, s, pen, brush);
//            ptr->setPos(QPointF(pp.x(), pp.y()));
//            points.push_back(ptr);
//        }
//    }
}
// Method to remove item from scene
void Person::remove_item(QGraphicsScene *scene)
{
    scene->removeItem(item);
    delete item; item = nullptr;
}

void Person::print() const
{
    qInfo() << "Person id: " << target.id;
    qInfo() << "    x_pos: " << std::stof(target.attributes.at("x_pos"));
    qInfo() << "    y_pos: " << std::stof(target.attributes.at("y_pos"));
    qInfo() << "    orientation: " << std::stof(target.attributes.at("orientation"));
}

