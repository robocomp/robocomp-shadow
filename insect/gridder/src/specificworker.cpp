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
#include <cppitertools/filter.hpp>
#include <cppitertools/enumerate.hpp>
#include <cppitertools/sliding_window.hpp>
#include <cppitertools/range.hpp>

/**
* \brief Default constructor
*/
SpecificWorker::SpecificWorker(TuplePrx tprx, bool startup_check) : GenericWorker(tprx)
{
	this->startup_check_flag = startup_check;
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
	this->Period = period;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
        // Viewer
        viewer = new AbstractGraphicViewer(this->frame, params.GRID_MAX_DIM);
        viewer->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, QColor("Blue"));
        viewer->show();

        // Grid
        grid.initialize(params.GRID_MAX_DIM, static_cast<int>(params.TILE_SIZE), &viewer->scene);

        // Lidar thread is created
        read_lidar_th = std::thread(&SpecificWorker::read_lidar,this);
        std::cout << __FUNCTION__ << " Started lidar reader" << std::endl;

        // mouse
        connect(viewer, &AbstractGraphicViewer::new_mouse_coordinates, [this](QPointF p)
        {
            qInfo() << "[MOUSE] New global target arrived:" << p;
            auto paths = grid.compute_k_paths(Eigen::Vector2f::Zero(), Eigen::Vector2f{p.x(), p.y()},
                                         params.NUM_PATHS_TO_SEARCH, params.MIN_DISTANCE_BETWEEN_PATHS);
            draw_path(paths.front(), &viewer->scene);
            this->lcdNumber_length->display((int)paths.front().size());
        });
        connect(viewer, &AbstractGraphicViewer::right_click, [this](QPointF p)
        {
            qInfo() <<  "RIGHT CLICK. Cancelling target";
            cancel_from_mouse = true;
        });

		timer.start(params.PERIOD);
	}
}

void SpecificWorker::compute()
{
    /// read LiDAR
    auto res_ = buffer_lidar_data.try_get();
    if (not res_.has_value())  {   /*qWarning() << "No data Lidar";*/ return; }
    auto points = res_.value();

    /// clear grid and update it
    mutex_path.lock();
        grid.clear();  // sets all cells to initial values
        grid.update_map(points, Eigen::Vector2f{0.0, 0.0}, params.MAX_LIDAR_RANGE);
        grid.update_costs( params.ROBOT_SEMI_WIDTH, true);     // not color all cells
    mutex_path.unlock();

    this->hz = fps.print("FPS:", 3000);
    this->lcdNumber_hz->display(this->hz);
}

///////////////////////////////////////////////////////////////////////////////////////////////
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
            auto data_helios = lidar3d1_proxy->getLidarDataWithThreshold2d(params.LIDAR_NAME_HIGH,
                                                                           params.MAX_LIDAR_HIGH_RANGE,
                                                                           params.LIDAR_HIGH_DECIMATION_FACTOR);
            // concatenate both lidars
            data.points.insert(data.points.end(), data_helios.points.begin(), data_helios.points.end());
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
void SpecificWorker::draw_path(const std::vector<Eigen::Vector2f> &path, QGraphicsScene *scene, bool erase_only)
{
    static std::vector<QGraphicsEllipseItem*> points;
    for(auto p : points)
        scene->removeItem(p);
    points.clear();

    if(erase_only) return;

    float s = 100;
    auto color = QColor("green");
    for(const auto &p: path)
    {
        auto ptr = scene->addEllipse(-s/2, -s/2, s, s, QPen(color), QBrush(color));
        ptr->setPos(QPointF(p.x(), p.y()));
        points.push_back(ptr);
    }
}
void SpecificWorker::draw_paths(const std::vector<std::vector<Eigen::Vector2f>> &paths, QGraphicsScene *scene, bool erase_only)
{
    static std::vector<QGraphicsEllipseItem*> points;
    static QColor colors[] = {QColor("cyan"), QColor("blue"), QColor("red"), QColor("orange"), QColor("magenta"), QColor("cyan")};
    for(auto p : points)
        scene->removeItem(p);
    points.clear();

    if(erase_only) return;

    float s = 80;
    for(const auto &[i, path]: paths | iter::enumerate)
    {
        // pick a consecutive color
        auto color = colors[i];
        for(const auto &p: path)
        {
            auto ptr = scene->addEllipse(-s/2.f, -s/2.f, s, s, QPen(color), QBrush(color));
            ptr->setPos(QPointF(p.x(), p.y()));
            points.push_back(ptr);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
RoboCompGridder::Result SpecificWorker::Gridder_getPaths(RoboCompGridder::TPoint source,
                                                         RoboCompGridder::TPoint target,
                                                         int max_paths,
                                                         bool try_closest_free_point,
                                                         bool target_is_human)
{
    //TODO: improve this method to try to find a path even if the target is not free by using the closest free point
    //TODO: if target is human, set safe area around as free

    auto begin = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    qInfo() << __FUNCTION__ << " New plan request: source [" << source.x << source.y << "], target [" << target.x << target.y << "]";
    mutex_path.lock();
        auto paths = grid.compute_k_paths(Eigen::Vector2f{source.x, source.y}, Eigen::Vector2f{target.x, target.y},
                                          params.NUM_PATHS_TO_SEARCH, params.MIN_DISTANCE_BETWEEN_PATHS);
    mutex_path.unlock();

    if(not paths.empty())
    {
        draw_paths(paths, &viewer->scene);
        this->lcdNumber_length->display((int)paths.front().size());
    }

    // fill Result with data
    RoboCompGridder::Result result;
    result.paths.resize(paths.size());
    for(const auto &[i, path]: paths | iter::enumerate)
    {
        result.paths[i].resize(path.size());
        for (const auto &[j, point]: path | iter::enumerate)
        {
            result.paths[i][j].x = point.x(); result.paths[i][j].y = point.y();
        }
    }
    result.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    qInfo() << __FUNCTION__ << " " << paths.size() << " paths computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - begin << " ms";
    return result;
}
bool SpecificWorker::Gridder_LineOfSightToTarget(RoboCompGridder::TPoint source, RoboCompGridder::TPoint target, float robot_radius)
{
    std::lock_guard<std::mutex> lock(mutex_path);
        bool res = grid.is_line_of_sigth_to_target_free(Eigen::Vector2f{source.x, source.y}, Eigen::Vector2f{target.x, target.y},
                                                    robot_radius);
        if(res)
            qInfo() << __FUNCTION__ <<  "Line of sight from [" << source.x << source.y << "] to [" << target.x << target.y << "] is FREE";
        else
            qInfo() << __FUNCTION__ <<  "Line of sight from [" << source.x << source.y << "] to [" << target.x << target.y << "] is BLOCKED";
        return res;
}
RoboCompGridder::TPoint SpecificWorker::Gridder_getClosestFreePoint(RoboCompGridder::TPoint source)
{
    std::lock_guard<std::mutex> lock(mutex_path);
    if(const auto &p = grid.closest_free({source.x, source.y}); p.has_value())
        return {static_cast<float>(p->x()), static_cast<float>(p->y())};
    else
        return {0, 0};  // non valid closest point
}
RoboCompGridder::TDimensions SpecificWorker::Gridder_getDimensions()
{
    return {static_cast<float>(params.GRID_MAX_DIM.x()),
            static_cast<float>(params.GRID_MAX_DIM.y()),
            static_cast<float>(params.GRID_MAX_DIM.width()),
            static_cast<float>(params.GRID_MAX_DIM.height())};
}
bool SpecificWorker::Gridder_IsPathBlocked(RoboCompGridder::TPath path)
{
    std::lock_guard<std::mutex> lock(mutex_path);
    std::vector<Eigen::Vector2f> path_;
    for(const auto &p: path)
        path_.emplace_back(p.x, p.y);
    return grid.is_path_blocked(path_);
}

//////////////////////////////////////////////////////////////////////////////////////////////
int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, qApp, SLOT(quit()));
    return 0;
}