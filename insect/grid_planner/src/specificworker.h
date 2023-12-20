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

/**
	\brief
	@author authorname
*/



#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <genericworker.h>
#include "grid.h"
#include "doublebuffer/DoubleBuffer.h"
#include <Eigen/Eigen>
#include "abstract_graphic_viewer/abstract_graphic_viewer.h"
#include <fps/fps.h>
#include <timer/timer.h>
#include "fastgicp.h"
#include <pcl/common/io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
        SpecificWorker(TuplePrx tprx, bool startup_check);
        ~SpecificWorker();
        bool setParams(RoboCompCommonBehavior::ParameterList params);
        void SegmentatorTrackingPub_setTrack (RoboCompVisualElements::TObject target);

    public slots:
        void compute();
        int startup_check();
        void initialize(int period);

    private:
        bool startup_check_flag;

        //Graphics
        AbstractGraphicViewer *viewer;
        std::vector<Eigen::Vector3f> get_lidar_data();

        //GRID
        int z_lidar_height = 0;
        Grid grid;
        float grid_width = 8000;
        float grid_length = 8000;
        float back_distance = 3000;
        float tile_size = 80;

        float xMin = -grid_width / 2;
        float xMax = grid_width / 2;
        float yMin = -back_distance;
        float yMax = grid_length - back_distance;

        // FPS
        FPSCounter fps;

        // Timer
        rc::Timer<> clock;

        // Lidar Thread
        DoubleBuffer<RoboCompLidar3D::TData, RoboCompLidar3D::TData> buffer_lidar_data;
        std::thread read_lidar_th;
        void read_lidar();

        // Target
        struct Target
        {
            bool active = false;
            bool global = false;  // defined in global coordinates
            Eigen::Vector2f point = Eigen::Vector2f(0, 0);
            QPointF qpoint;
            void set(const QPointF &p, bool global_ = false)
            {
                point = Eigen::Vector2f(p.x(), p.y());
                qpoint = p;
                global = global_;
                active = true;
            }
            void set(const Eigen::Vector2f &p, bool global_ = false)
            {
                point = p;
                global = global_;
                qpoint = QPointF(p.x(), p.y());
                active = true;
            }
            Eigen::Vector2f pos_eigen() const { return point; }
            QPointF pos_qt() const { return qpoint; }
            void unset() { active = false; }
            void print() const
            {
                qInfo() << "Target: ";
                qInfo() << "    point: " << point.x() << " " << point.y();
                qInfo() << "    qpoint: " << qpoint.x() << " " << qpoint.y();
                qInfo() << "    global: " << global;
                qInfo() << "    active: " << active;
            }
        };
        //Target target;
        //DoubleBuffer<std::tuple<Eigen::Vector2f, bool> , std::tuple<Eigen::Vector2f, bool>> target_buffer;
        DoubleBuffer<Target, Target> target_buffer;

    // Lidar odometry
        FastGICP fastgicp;

        // Path
        void draw_path(const vector<Eigen::Vector2f> &path, QGraphicsScene *scene);
        std::vector<Eigen::Vector2f> last_path;

        // Frechet distance calculus
        float euclideanDistance(const Eigen::Vector2f& a, const Eigen::Vector2f& b);
        float frechetDistanceUtil(const std::vector<Eigen::Vector2f>& path1, const std::vector<Eigen::Vector2f>& path2, int i, int j, std::vector<std::vector<float>>& dp);
        float frechetDistance(const std::vector<Eigen::Vector2f>& path1, const std::vector<Eigen::Vector2f>& path2);
        RoboCompGridPlanner::TPoint send_path(const vector<Eigen::Vector2f> &path,
                                              float threshold_dist, float threshold_angle);
        std::optional<Eigen::Vector2f> closest_point_to_target(const QPointF &p);
        bool not_line_of_sight_path(const QPointF &f);
        Eigen::Vector2f  border_subtarget(RoboCompVisualElements::TObject target);
        void draw_lidar(const RoboCompLidar3D::TPoints &points, int decimate=1);
        void draw_subtarget(const Eigen::Vector2f &point, QGraphicsScene *scene);
        void draw_global_target(const Eigen::Vector2f &point, QGraphicsScene *scene);
};

#endif
