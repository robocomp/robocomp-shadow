//
// Created by pbustos on 17/11/22.
//

#ifndef FORCEFIELD_SM_SEARCH_AND_APPROACH_H
#define FORCEFIELD_SM_SEARCH_AND_APPROACH_H

#include <Eigen/Dense>
#include "robot.h"
#include <timer/timer.h>
#include "door_detector.h"
#include "preobject.h"
#include "room.h"
#include "graph.h"
#include <QFrame>

class SM_search_and_approach
{
    public:
        enum class State {IDLE, SEARCH, APPROACH, CROSSING, LOST};
        void init(QFrame *graph_frame);
        State update(rc::Robot &robot, const std::vector<rc::PreObject> &preobjects,
                               const  std::vector<std::string> &yolo_names);

    private:
        State state = State::SEARCH;
        State search_state(rc::Robot &robot, const std::vector<rc::PreObject> &preobjects,
                           const std::vector<std::string> &yolo_names);
        State approach_state(rc::Robot &robot, const std::vector<rc::PreObject> &preobjects,
                             const std::vector<std::string> &yolo_names);
        State crossing_state();
        State lost_state(rc::Robot &robot, const std::vector<rc::PreObject> &preobjects);

        float min_dist_to_target = 1000;  //mm
        int current_room;

        // graph
        AbstractGraphicViewer *graph_viewer;
        rc::Graph graph;
};


#endif //FORCEFIELD_SM_SEARCH_AND_APPROACH_H
