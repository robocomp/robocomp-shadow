//
// Created by robolab on 1/16/24.
//

#ifndef PEOPLE_PATH_PREDICTOR_PERSON_H
#define PEOPLE_PATH_PREDICTOR_PERSON_H

#include "abstract_graphic_viewer/abstract_graphic_viewer.h"
#include <VisualElementsPub.h>
#include <Gridder.h>
#include <Eigen/Eigen>

class Person
{
    private:
        RoboCompVisualElementsPub::TObject target;
        QGraphicsEllipseItem *item = nullptr;
        bool is_target = false;
        std::chrono::high_resolution_clock::time_point insertion_time, last_update_time;
        long int dsr_id = -1;

        // Pilar Cone
        QPolygonF pilar_cone;
        RoboCompVisualElementsPub::TObjects objects_inside_pilar_cone;

        // Gridder proxy pointer
        RoboCompGridder::GridderPrxPtr gridder_proxy;

        // Paths to visual elements
        std::vector<std::pair<int, std::vector<Eigen::Vector2f>>> paths;

    public:
        Person(RoboCompGridder::GridderPrxPtr g_proxy);
        Person();
        void init_item(QGraphicsScene *scene, float x, float y, float angle, float cone_radius, float cone_angle);
        void set_person_data(RoboCompVisualElementsPub::TObject person);
        void update_attributes(const RoboCompVisualElementsPub::TObjects &list);
        std::optional<float> get_attribute(const std::string &attribute_name) const;
        bool set_attribute(const std::string &attribute_name, float value) ;
        void is_inside_pilar_cone(const RoboCompVisualElementsPub::TObjects &list);
        std::optional<std::pair<int, std::vector<Eigen::Vector2f>>> search_for_paths(const RoboCompVisualElementsPub::TObject &object);
        void set_target_element(bool value);
        int get_id() const;
        QGraphicsItem* get_item() const;
        RoboCompVisualElementsPub::TObject get_target() const;
        bool is_target_element() const;
        void set_updated(bool value);
        void update_last_update_time();
        std::chrono::high_resolution_clock::time_point get_last_update_time() const;
        void set_insertion_time();
        std::chrono::high_resolution_clock::time_point get_insertion_time() const;
        void set_dsr_id(long int id);
        long int get_dsr_id() const;
        void draw_paths(QGraphicsScene *scene, bool erase_only, bool wanted_person) const;
        void remove_item(QGraphicsScene *scene);
        void print() const;
};

#endif //PEOPLE_PATH_PREDICTOR_PERSON_H
