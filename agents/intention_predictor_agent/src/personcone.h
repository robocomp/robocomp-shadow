//
// Created by robolab on 1/16/24.
//

#ifndef PEOPLE_PATH_PREDICTOR_PERSON_H
#define PEOPLE_PATH_PREDICTOR_PERSON_H

#include "abstract_graphic_viewer/abstract_graphic_viewer.h"
#include <Eigen/Eigen>
#include "dsr/api/dsr_api.h"
#include "dsr/gui/dsr_gui.h"
#include <Gridder.h>
class PersonCone
{
    struct Intention
    {
        std::string target_name;
        std::vector<std::vector<Eigen::Vector2f>> paths;
        QGraphicsRectItem *item = nullptr;
        void init_item(QGraphicsScene *scene, float x, float y, float width, float height, bool obstacle=false)
        {
            qInfo() << "Creating intention item";
            if(obstacle)
                item = scene->addRect(-width / 2.f, -height / 2.f, width, height,
                                      QPen(QColor("red")), QBrush(QColor("red")));
            else
                item = scene->addRect(-width / 2.f, -height / 2.f, width, height,
                                      QPen(QColor("green")), QBrush(QColor("green")));
            item->setPos(x, y);
            // add a text item with the id
            auto text = scene->addText(QString::fromStdString(target_name));
            text->setParentItem(item);
            text->setPos(-text->boundingRect().width() * 5, text->boundingRect().height() * 17);
            text->setDefaultTextColor(QColor("black"));
            text->setScale(10);
            QTransform transform; transform.scale(1, -1);
            text->setTransform(transform);
        }
        void update_attributes(const float &x, const float &y)
        {
            item->setPos(x, y);
        };
        void update_color(bool obstacle)
        {
            if(!obstacle)
            {
                qInfo() << "Setting intention color to blue";
                item->setPen(QPen(QColor("blue")));
                item->setBrush(QBrush(QColor("blue")));
            }
            else
            {
                qInfo() << "Setting intention color to red";
                item->setPen(QPen(QColor("red")));
                item->setBrush(QBrush(QColor("red")));
            }
        };
        void remove_item(QGraphicsScene *scene) { scene->removeItem(item); delete item; item = nullptr;};
        void update_path(const std::vector<std::vector<Eigen::Vector2f>> &new_paths)
        {
            paths = new_paths;
        }
    };
    private:
        QGraphicsEllipseItem *item = nullptr;
        long int dsr_id = -1;
        std::vector<QGraphicsEllipseItem*> points;

        // Pilar Cone
        QPolygonF pilar_cone;
        std::vector<Intention> intentions;

        // Gridder proxy pointer
        RoboCompGridder::GridderPrxPtr gridder_proxy;

    public:
        PersonCone(RoboCompGridder::GridderPrxPtr g_proxy);
        PersonCone();
        void init_item(QGraphicsScene *scene, float x, float y, float angle, float cone_radius, float cone_angle, const std::string& name);
        void update_attributes(float x, float y, float angle);
        bool is_inside_pilar_cone(const QPolygonF &pilar_cone, float x, float y);
        void set_dsr_id(long int id);
        long int get_dsr_id() const;
        QGraphicsItem* get_item() const;
        QPolygonF get_pilar_cone() const;
        int get_intentions_size() const;
        void remove_item(QGraphicsScene *scene);
        void remove_intentions(QGraphicsScene *scene, std::vector<DSR::Node> object_nodes);
        void set_intention(const std::tuple<float, float, float, float> &person_point, const std::tuple<float, float, float, float> &element_point, const std::string &target_name);
        // Method for getting intention
        std::optional<PersonCone::Intention*> get_intention(const std::string &target_name);
        std::optional<std::vector<std::vector<Eigen::Vector2f>>> get_paths(const std::tuple<float, float, float, float> &person_point, const std::tuple<float, float, float, float> &element_point, const std::string &target_name);
        void remove_intention(const std::string &target_name);
        // Method to draw path
        void draw_paths(QGraphicsScene *scene, bool erase_only, RoboCompGridder::TPath hallucinogen_path);
};

#endif //PEOPLE_PATH_PREDICTOR_PERSON_H
