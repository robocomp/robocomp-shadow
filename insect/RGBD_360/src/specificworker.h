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
    FixedSizeDeque<RoboCompCamera360RGB::TImage> camera_queue{20};
    FixedSizeDeque<RoboCompLidar3D::TDataImage> lidar_queue{20};
    cv::Mat cut_image(cv::Mat image, int cx, int cy, int sx, int sy, int roiwidth, int roiheight);
	bool startup_check_flag;
    DoubleBuffer<cv::Mat, cv::Mat> buffer_rgb_image;
    DoubleBuffer<cv::Mat, cv::Mat> buffer_depth_image;
    int MAX_WIDTH, MAX_HEIGHT;
    bool enabled_camera = false;
    bool enabled_lidar = false;
    long long capture_time;
    FPSCounter fps;

    cv::Mat rgb_frame_write, depth_frame_write;
    mutable std::mutex swap_mutex;
};



#endif