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
    QGraphicsItem *item = nullptr;
    bool is_target = false;
    std::chrono::high_resolution_clock::time_point insertion_time, last_update_time;

    // Pilar cone
    QPolygonF pilar_cone;
    RoboCompVisualElementsPub::TObjects objects_inside_pilar_cone;

    // Gridder proxy pointer
    RoboCompGridder::GridderPrxPtr gridder_proxy;

    // Paths to visual elements
    std::vector<std::pair<int, std::vector<Eigen::Vector2f>>> paths;
    std::vector<QGraphicsPolygonItem*> points;

    public:
        Person(RoboCompGridder::GridderPrxPtr g_proxy);
        Person();
        // Method to initialize item from scene
        void init_item(QGraphicsScene *scene, float x, float y, float angle, float cone_radius, float cone_angle);
        void set_person_data(RoboCompVisualElementsPub::TObject person);
        // Method to check if there is an object in a TObjects list with the same id
        void update_attributes(const RoboCompVisualElementsPub::TObjects &list);
        // Method to check if there are TObjects inside the pilar cone
        void is_inside_pilar_cone(const RoboCompVisualElementsPub::TObjects &list);
        // Method to order paths
        std::optional<std::pair<int, std::vector<Eigen::Vector2f>>> order_paths(const RoboCompVisualElementsPub::TObject &object);
        // Set if the element is the robot target
        void set_target_element(bool value);
        // Method to get the id of the person
        int get_id() const;
        QGraphicsItem* get_item() const;
        RoboCompVisualElementsPub::TObject get_target() const;
        bool is_target_element() const;
        // method to set updated
        void set_updated(bool value);
        // Method to update the last update time
        void update_last_update_time();
        // Method to get the last update time
        std::chrono::high_resolution_clock::time_point get_last_update_time() const;
        // Method to set the insertion time
        void set_insertion_time();
    // Method to get the insertion time
        std::chrono::high_resolution_clock::time_point get_insertion_time() const;
        // Method to draw paths
        void draw_paths(QGraphicsScene *scene, bool erase_only, bool wanted_person);
        // Method to remove item from scene
        void remove_item(QGraphicsScene *scene);
};

#endif //PEOPLE_PATH_PREDICTOR_PERSON_H
