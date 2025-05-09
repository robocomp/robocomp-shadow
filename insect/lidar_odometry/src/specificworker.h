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
#include "doublebuffer/DoubleBuffer.h"
#include "fastgicp.h"
#include <pcl/common/io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <fps/fps.h>
#include "abstract_graphic_viewer/abstract_graphic_viewer.h"


class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
        SpecificWorker(TuplePrx tprx, bool startup_check);
        ~SpecificWorker();
        bool setParams(RoboCompCommonBehavior::ParameterList params);

        RoboCompFullPoseEstimation::FullPoseEuler LidarOdometry_getFullPoseEuler();
        RoboCompFullPoseEstimation::FullPoseMatrix LidarOdometry_getFullPoseMatrix();
        RoboCompLidarOdometry::PoseAndChange LidarOdometry_getPoseAndChange();

        void LidarOdometry_reset();

    public slots:
        void compute();
        int startup_check();
        void initialize(int period);

    private:
        bool startup_check_flag;

        // config
        bool display = false;
        std::string lidar_name = "helios";
        QRectF dim = QRectF(-4000, -4000, 8000, 8000);
        int MAX_INACTIVE_TIME = 5;  // secs after which the component is paused. It reactivates with a new reset

        // Lidar and Thread
        DoubleBuffer<RoboCompLidar3D::TData, RoboCompLidar3D::TData> buffer_lidar_data;
        std::thread read_lidar_th;
        void read_lidar_thread();
        std::optional<pcl::PointCloud<pcl::PointXYZ>::Ptr> read_lidar();

        // Lidar odometry
        FastGICP fastgicp;
        DoubleBuffer<std::pair<Eigen::Transform<double, 3, 1>, Eigen::Matrix<double, 4, 4>>, RoboCompLidarOdometry::PoseAndChange> buffer_odometry;

        // Graphics
        AbstractGraphicViewer *viewer;
        QGraphicsPolygonItem *robot_polygon;

        // draw
        void draw_robot(const Eigen::Isometry3d &robot_pose);
        void draw_lidar(const RoboCompLidar3D::TPoints &points, int decimate=1);
        void draw_path(bool only_clean=false);

        // Timer
        FPSCounter fps;
        std::atomic<std::chrono::high_resolution_clock::time_point> last_read;

        // Path
        std::vector<Eigen::Vector2f> path;



};

#endif
