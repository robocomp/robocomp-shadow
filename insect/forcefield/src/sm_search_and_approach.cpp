//
// Created by pbustos on 17/11/22.
//

#include "sm_search_and_approach.h"

void SM_search_and_approach::init(QFrame *graph_frame)
{
    graph_viewer = new AbstractGraphicViewer(graph_frame,  QRectF(0, 0, 2000, 300));
    graph.init(graph_viewer);
    current_room = graph.add_node();
    graph.draw();
}
SM_search_and_approach::State SM_search_and_approach::update(rc::Robot &robot, const std::vector<rc::PreObject> &preobjects,
                                               const std::vector<std::string> &yolo_names)
{
    switch (state)
    {
        case State::IDLE:
            qInfo() << "IDLE";
            robot.stop();
            break;
        case State::SEARCH:
            qInfo() << "SEARCH";
            state = search_state(robot, preobjects, yolo_names); // turns around until it finds a valid target
            break;
        case State::APPROACH:
            qInfo() << "APPROACH";
            state = approach_state(robot, preobjects,  yolo_names);
            break;
        case State::CROSSING:
            crossing_state();
            break;
        case State::LOST:
            lost_state(robot, preobjects);
            break;
    };
    return state;
}
SM_search_and_approach::State SM_search_and_approach::search_state(rc::Robot &robot, const std::vector<rc::PreObject> &prebojects,
                                                                   const std::vector<std::string> &yolo_names)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> dist(0, 1);
    static float rot_speed;
    static bool first_time=true;
    static const float nominal_rot_speed = 0.2;

    if(robot.get_current_target().type != -1)
        qInfo() << __FUNCTION__ << QString::fromStdString(yolo_names[robot.get_current_target().type]);
    //  get first object different from current robot.target and stick with it (be persistent)
    for (const auto &o: prebojects)
    {
        if(o.type == 80)  // door
        {
            robot.set_current_target(o);
            robot.stop();
            first_time = true;
            qInfo() << __FUNCTION__ << "Target selected" << QString::fromStdString(yolo_names[o.type]);
            return State::APPROACH;
        }
    }
    if(first_time)
    {
        if (dist(gen) == 0)
            rot_speed = nominal_rot_speed;
        else
            rot_speed = -nominal_rot_speed;
        first_time = false;
    }
    robot.just_rotate(rot_speed);
    return State::SEARCH;
}
SM_search_and_approach::State SM_search_and_approach::approach_state(rc::Robot &robot, const std::vector<rc::PreObject> &preobjects,const std::vector<std::string> &yolo_names)
{
    static int lost_counter;

    // keep and refresh visual target while approaching
    qInfo() << __FUNCTION__ << " current target" << QString::fromStdString(yolo_names[robot.get_current_target().type]);

    //  match current target with new objects
    if (auto it = std::find_if(preobjects.begin(), preobjects.end(), [r = robot](auto &a) { return a.type == r.get_current_target().type; }); it !=
                                                                                                                                              preobjects.end())
    {
        robot.set_current_target(*it);
        lost_counter = 0;
        qInfo() << __FUNCTION__ << "Target FOUND";
    }
    else
    {
        lost_counter++;
        qInfo() << __FUNCTION__ << "Target LOST" << lost_counter;
    //        state = State::LOST;
    //        std::cout << __FUNCTION__  << " Lost " << robot.get_distance_to_target() << std::endl;
    //        robot.set_eye_track(false);
    }

    if(lost_counter > 20)
    {
        robot.stop();
        lost_counter = 0;
        return State::SEARCH;   //TODO: move to the center of the room in LOST state
    }

    robot.get_current_target().print();
    qInfo() << __FUNCTION__ << "distance to target" << robot.get_distance_to_target();
    if(robot.get_distance_to_target() < min_dist_to_target) // arrival condition
    {
        std::cout << __FUNCTION__  << " ARRIVED to target " << robot.get_distance_to_target() << std::endl;
        return State::CROSSING;
    }
    else
    {
        graph.add_tags(current_room, preobjects);
        return State::APPROACH;
    }

}
SM_search_and_approach::State SM_search_and_approach::crossing_state()
{
    static rc::Timer clock;
    static const uint  TIME_INTERVAL = 4000; //ms
    static bool first_time = true;
    if (first_time)
    {
        clock.tick();
        first_time = false;
    }
    else
    {
        qInfo() << __FUNCTION__ << "CROSSING THE DOOR";
        // what must happen when crossing the door?
        clock.tock();
        if(clock.duration() >= TIME_INTERVAL)
        {
            current_room = graph.add_node(current_room);
            graph.draw();
            state = State::SEARCH;
            first_time = true;
        }
    }
    return State::CROSSING;
}
SM_search_and_approach::State SM_search_and_approach::lost_state(rc::Robot &robot, const std::vector<rc::PreObject> &preobjects)
{
    qInfo() << __FUNCTION__  << "distance to target" << robot.get_distance_to_target();
    return State::LOST;
}


//    static rc::Timer clock;
//    static const uint TIME_INTERVAL = 600; //ms
//    static bool first_time = true;
//    if (first_time)
//    {
//        clock.tick();
//        first_time = false;
//        qInfo() << __FUNCTION__ << "Lost";
//        robot.just_rotate(0.f);
//    } else
//    {
//        if (auto it = std::find_if(preobjects.begin(), preobjects.end(), [r = robot](auto &a) { return a.type == r.get_current_target().type; }); it != preobjects.end())
//        {
//            state = State::APPROACH;
//            robot.just_rotate(0.f);
//            first_time = true;
//        } else
//        {
//            clock.tock();
//            if (clock.duration() >= TIME_INTERVAL)
//            {
//                state = State::SEARCH;
//                first_time = true;
//            }
//            else
//                robot.just_rotate(0.3);
//        }
//    }