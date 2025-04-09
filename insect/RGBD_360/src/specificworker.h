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
#include <opencv2/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include "cppitertools/zip.hpp"
#include"cppitertools/enumerate.hpp"
#include<opencv2/highgui/highgui.hpp>
#include<fixedsizedeque.h>
#include <deque>
#include <doublebuffer/DoubleBuffer.h>
#include <fps/fps.h>
#include <boost/circular_buffer.hpp>
#include <multibuffer_sync/multibuffer_sync.h>

using namespace std::chrono;

class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
        SpecificWorker(TuplePrx tprx, bool startup_check);
        ~SpecificWorker();
        bool setParams(RoboCompCommonBehavior::ParameterList params);
        RoboCompCamera360RGBD::TRGBD Camera360RGBD_getROI(int cx, int cy, int sx, int sy, int roiwidth, int roiheight);

    public slots:
        void compute();
        int startup_check();
        void initialize(int period);

    private:
        struct Params
        {
            bool DISPLAY = false;
        };
        Params params;

        FixedSizeDeque<RoboCompCamera360RGB::TImage> camera_queue{20};
        FixedSizeDeque<RoboCompLidar3D::TDataImage> lidar_queue{20};
        boost::circular_buffer<RoboCompCamera360RGB::TImage> b_camera_queue{20};
        boost::circular_buffer<RoboCompLidar3D::TDataImage> b_lidar_queue{20};
        SyncBuffer<std::pair<RoboCompLidar3D::TDataImage, RoboCompLidar3D::TDataImage>,
                   std::pair<RoboCompCamera360RGB::TImage, RoboCompCamera360RGB::TImage>> sync_buffer{5 /* buffer capacity */,
                                                                                  10000.0 /* max allowed timestamp spread */,
                                                                                                      20000.0 /* timeout */};

        cv::Mat cut_image(cv::Mat image, int cx, int cy, int sx, int sy, int roiwidth, int roiheight);
        bool startup_check_flag;

        int MAX_WIDTH, MAX_HEIGHT;
        bool enabled_camera = false;
        bool enabled_lidar = false;
        long long capture_time;

        // fps
        FPSCounter fps;
        std::atomic<std::chrono::high_resolution_clock::time_point> last_read;
        int MAX_INACTIVE_TIME = 5;  // secs after which the component is paused. It reactivates with a new reset

        // camera buffers
        cv::Mat rgb_frame_write, depth_frame_write;

        mutable std::mutex swap_mutex;
    };



#endif
