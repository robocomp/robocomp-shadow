/*
 *    Copyright (C) 2023 by YOUR NAME HERE
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
#include <cppitertools/filter.hpp>
#include <cppitertools/enumerate.hpp>
#include <cppitertools/sliding_window.hpp>


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


    return true;
}

void SpecificWorker::initialize(int period)
{
    std::cout << "Initialize worker" << std::endl;
    this->Period = 50;

    if (this->startup_check_flag) {
        this->startup_check();
    } else {
        viewer = new AbstractGraphicViewer(this->frame, QRectF(xMin, yMin, grid_widht, grid_length));

//        QRectF dim{-5000, -5000, 10000, 10000};
        QRectF dim{xMin, yMin, static_cast<qreal>(grid_widht), static_cast<qreal>(grid_length)};

        viewer->draw_contour();
        viewer->add_robot(500, 500, 0, 0, QColor("Blue"));
        grid.initialize(dim, tile_size, &viewer->scene, false);

        // mouse
        // connect(viewer, SIGNAL(new_mouse_coordinates(QPointF)), this, SLOT());
        connect(viewer, &AbstractGraphicViewer::new_mouse_coordinates, [this](QPointF p){ qInfo() << "Target:" << p; target.set(p);});
        t.tick();
        timer.start(Period);
    }
}

void SpecificWorker::compute()
{
    clock.tick();
    if (auto target = target_buffer.try_get(); target.has_value())
    {
//        if(target.active)
//        {
        grid.clear();
        auto points = get_lidar_data();
        grid.update_map(points, Eigen::Vector2f{0.0, 0.0}, 3500);
        grid.update_costs(true);
        QPointF Qtarget(target->x(),target->y());

        if (los_path(Qtarget))
        {
            cout << "PATH BLOCKED" << endl;
            auto path = grid.compute_path(QPointF(0, 0), Qtarget);
            if(not path.empty() and path.size() > 0)
            {
                auto subtarget = send_path(path, 750, M_PI_4 / 4);

                draw_path(path, &viewer->scene);
                draw_subtarget(Eigen::Vector2f(subtarget.x, subtarget.y), &viewer->scene);

                //Tplan (path, timestamps, Tpoint subtarget,bool valid)
                RoboCompGridPlanner::TPlan returning_plan;
                returning_plan.subtarget = subtarget;
                returning_plan.valid = true;
                try
                {
                    gridplanner_proxy->setPlan(returning_plan);
                }
                catch (const Ice::Exception &e) { std::cout << "Error setting valid plan" << e << std::endl; }

            }
            else
            {
                draw_subtarget(Eigen::Vector2f {0.0, 0.0}, &viewer->scene);
                RoboCompGridPlanner::TPlan returning_plan;
                returning_plan.valid = false;
                try
                {
                    gridplanner_proxy->setPlan(returning_plan);
                }
                catch (const Ice::Exception &e) { std::cout << "Error setting valid plan" << e << std::endl; }
            }
        }
        else
        {
            t.tick();
            RoboCompGridPlanner::TPlan returning_plan;
            returning_plan.valid = false;
            draw_subtarget(Eigen::Vector2f {0.0, 0.0}, &viewer->scene);
            try
            {
                gridplanner_proxy->setPlan(returning_plan);
            }
            catch (const Ice::Exception &e) { std::cout << "Error setting empty plan" << e << std::endl; }
        }

        viewer->update();


//        }
    }
    else //NO TARGET
    {

    }
    fps.print("FPS:");
//    qInfo()<< "Duration_end" << clock.duration();
}

////////////////////////////////////////////////////////////////////////////////////////////
std::vector<Eigen::Vector3f> SpecificWorker::get_lidar_data()
{
    std::vector <Eigen::Vector3f> points;
    try {
    	string lidar_name = "bpearl";
        auto ldata = lidar3d_proxy->getLidarData(lidar_name, 0, 360, 1);
        //HELIOS
//        for (auto &&[i, p]: iter::filter([z = z_lidar_height](auto p)
//        {
//            float dist = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
//            return p.z < 300
//                   and p.z > -900
//                   and dist < 5000
//                   and dist > 550;
//        }, ldata) | iter::enumerate)
//            points.emplace_back(Eigen::Vector3f{p.x, p.y, p.z});

        // BPEARL

        for (auto &&[i, p]: iter::filter([z = z_lidar_height](auto p)
                                         {
                                             float dist = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
                                             return p.z < 300
                                                    and p.z > -200
                                                    and dist < 5000
                                                    and dist > 250;
                                         }, ldata) | iter::enumerate)
            points.emplace_back(Eigen::Vector3f{p.x, p.y, p.z});
        return points;
    }
    catch (const Ice::Exception &e) { std::cout << "Error reading from Lidar3D" << e << std::endl; }
    return points;
}

RoboCompGridPlanner::TPoint SpecificWorker::send_path(const std::vector<Eigen::Vector2f> &path, float threshold_dist, float threshold_angle)
{
    RoboCompGridPlanner::TPoint subtarget;

    static float h = sin(M_PI_2 + (M_PI - threshold_angle)/ 2.0);
    float len = 0.0;

    if (path.empty())
        return RoboCompGridPlanner::TPoint{.x=0.f, .y=0.f};

    if(path.size() < 3)
    {
        subtarget.x = path.back().x();
        subtarget.y = path.back().y();
    }

    auto local_path = path;
    local_path.erase(local_path.begin());

    for(auto &&p : iter::sliding_window(local_path, 3))
    {
        float d0d1 = (p[1]-p[0]).norm();
        float d1d2 = (p[2]-p[1]).norm();
        float d0d2 = (p[2]-p[0]).norm();
        len += d0d1;
//        qInfo()<< len << (d0d2 - (d1d2 + d0d1)) << (d0d1 * h) << (((d1d2 + d0d1) - d0d2) > (d0d1 * h));
        if (len > threshold_dist or (abs((d1d2 + d0d1) - d0d2) > (d0d1 * h)))
        {
            subtarget.x = p[1].x();
            subtarget.y = p[1].y();
            break;
        }
        subtarget.x = p[2].x();
        subtarget.y = p[2].y();
    }
//    qInfo()<< "t.duration"<<t.duration();
//    if((Eigen::Vector2f{last_subtarget.x,last_subtarget.y}-Eigen::Vector2f{subtarget.x,subtarget.y}).norm()< 600 or t.duration() > 500)
//    {
//        t.tick();
//        last_subtarget=subtarget;
//        return subtarget;
//    }
//    else
//        return last_subtarget;
    return subtarget;
}

void SpecificWorker::draw_path(const std::vector<Eigen::Vector2f> &path, QGraphicsScene *scene)
{
    static std::vector<QGraphicsEllipseItem*> points;
    for(auto p : points)
        scene->removeItem(p);
    points.clear();

    int s = 80;
    for(const auto &p: path)
    {
        auto ptr = scene->addEllipse(-s/2, -s/2, s, s, QPen(QColor("green")), QBrush(QColor("green")));
        ptr->setPos(QPointF(p.x(), p.y()));
        points.push_back(ptr);
    }
}



void SpecificWorker::draw_subtarget(const Eigen::Vector2f &point, QGraphicsScene *scene)
{
    //CLEAN

    static QGraphicsEllipseItem* subtarget;
    scene->removeItem(subtarget);

    int s = 120;
    auto ptr = scene->addEllipse(-s/2, -s/2, s, s, QPen(QColor("red")), QBrush(QColor("red")));
    ptr->setPos(QPointF(point.x(), point.y()));
    subtarget = ptr;
}

///////////////////////////////////////////////////////////////////////////////////////////
int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, qApp, SLOT(quit()));
    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////// Inrterface
///////////////////////////////////////////////////////////////////////////////////////////////


//SUBSCRIPTION to setTrack method from SegmentatorTrackingPub interface
void SpecificWorker::SegmentatorTrackingPub_setTrack (RoboCompVisualElements::TObject target)
{
    qInfo()<< "TARGET " << target.x << target.y;
    if (target.x < xMax and target.x > xMin and target.y > yMin and target.y < yMax)
        target_buffer.put(Eigen::Vector2f{target.x, target.y});
    else{
        target_buffer.put(border_subtarget(target));
        qInfo()<<"TARGET OUT OF GRID" << border_subtarget(target).x() << border_subtarget(target).y();
    }
}


Eigen::Vector2f SpecificWorker::border_subtarget(RoboCompVisualElements::TObject target)
{
    Eigen::Vector2f target2f {target.x,target.y};

    float dist = target2f.norm();

    Eigen::Vector2f corner_left_top {xMin, yMax};
    Eigen::Vector2f corner_right_bottom {xMax, yMin};

    //Vertical
    if (target2f.x() == 0){
        target2f.y() = (target.y > 0) ? corner_left_top.y() : corner_right_bottom.y();
        return target2f;
    }

    double m = target2f.y() / target2f.x();  // Pendiente de la línea


    // Calculamos las intersecciones con los lados del rectángulo
    Eigen::Vector2f interseccionIzquierda(xMin, m * xMin);
    Eigen::Vector2f interseccionDerecha(xMax, m * xMax);
    Eigen::Vector2f interseccionSuperior(xMax / m, yMax);
    Eigen::Vector2f interseccionInferior(xMin / m, yMin);

    // Comprobamos si las intersecciones están dentro del rectángulo
    Eigen::Vector2f intersecciones[4] = { interseccionIzquierda, interseccionDerecha, interseccionSuperior, interseccionInferior };
    Eigen::Vector2f resultado;

    for (int i = 0; i < 4; ++i) {
        float x = intersecciones[i].x();
        float y = intersecciones[i].y();
        if (xMin <= x && x <= xMax && yMin <= y && y <= yMax) {
            if((intersecciones[i]-target2f).norm() < dist)
            {
                resultado = intersecciones[i];
                break;
            }
        }
    }
    return resultado;
}

bool SpecificWorker::los_path(QPointF f) {
    int tile_size = 200;
    std::vector<Eigen::Vector2f> path;
//    cout << "f X" << f.x() << endl;
//    cout << "f y" << f.y() << endl;
    Eigen::Vector2f target(f.x(),f.y());
    Eigen::Vector2f origin(0.0,0.0);
    float steps = (target - origin).norm() / tile_size;
    Eigen::Vector2f step((target-origin)/steps);

//    cout << "STEPS " << steps << endl;
//    cout << "STEP X" << step.x() << endl;
//    cout << "STEP y" << step.y() << endl;

    for ( int i = 0 ; i <= steps-3; ++i)
    {
        path.push_back(origin + i*step);
    }
    draw_path(path, &viewer->scene);
    return  grid.is_path_blocked(path);
}

/**************************************/
// From the RoboCompGridPlanner you can call this methods:
// this->gridplanner_proxy->setPlan(...)

/**************************************/
// From the RoboCompGridPlanner you can use this types:
// RoboCompGridPlanner::TPoint
// RoboCompGridPlanner::TPlan

/**************************************/
// From the RoboCompLidar3D you can call this methods:
// this->lidar3d_proxy->getLidarData(...)

/**************************************/
// From the RoboCompLidar3D you can use this types:
// RoboCompLidar3D::TPoint



