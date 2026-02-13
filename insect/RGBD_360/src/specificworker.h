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

// If you want reduce compute period automaticaly for lack of use
//#define HIBERNATION_ENABLED

#include <genericworker.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cppitertools/zip.hpp"
#include "cppitertools/enumerate.hpp"
#include "fixedsizedeque.h"
#include <deque>
#include <optional>
#include <doublebuffer/DoubleBuffer.h>
#include <fps/fps.h>
#include <boost/circular_buffer.hpp>
#include <multibuffer_sync/multibuffer_sync.h>
#include <shared_mutex>

using namespace std::chrono;

class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
       /**
     * \brief Constructor for SpecificWorker.
     * \param configLoader Configuration loader for the component.
     * \param tprx Tuple of proxies required for the component.
     * \param startup_check Indicates whether to perform startup checks.
     */
	SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);

	/**
     * \brief Destructor for SpecificWorker.
     */
	~SpecificWorker();

	RoboCompCamera360RGBD::TRGBD Camera360RGBD_getROI(int cx, int cy, int sx, int sy, int roiwidth, int roiheight);

	RoboCompLidar3D::TColorCloudData Lidar3D_getColorCloudData();
	RoboCompLidar3D::TData Lidar3D_getLidarData(std::string name, float start, float len, int decimationDegreeFactor);
	RoboCompLidar3D::TDataImage Lidar3D_getLidarDataArrayProyectedInImage(std::string name);
	RoboCompLidar3D::TDataCategory Lidar3D_getLidarDataByCategory(RoboCompLidar3D::TCategories categories, Ice::Long timestamp);
	RoboCompLidar3D::TData Lidar3D_getLidarDataProyectedInImage(std::string name);
	RoboCompLidar3D::TData Lidar3D_getLidarDataWithThreshold2d(std::string name, float distance, int decimationDegreeFactor);


    public slots:

    /**
         * \brief Initializes the worker one time.
         */
        void initialize();

        /**
         * \brief Main compute loop of the worker.
         */
        void compute();

        /**
         * \brief Handles the emergency state loop.
         */
        void emergency();

        /**
         * \brief Restores the component from an emergency state.
         */
        void restore();

    /**
     * \brief Performs startup checks for the component.
     * \return An integer representing the result of the checks.
     */
        int startup_check();

    private:
        struct Params
        {
            bool DISPLAY = false;
        };
        Params params;

        SyncBuffer<std::pair<RoboCompLidar3D::TDataImage, RoboCompLidar3D::TDataImage>,
                   std::pair<RoboCompCamera360RGB::TImage, RoboCompCamera360RGB::TImage>> sync_buffer{5 /* buffer capacity */,
                                                                                  10000.0 /* max allowed timestamp spread */,
                                                                                                      20000.0 /* timeout */};
        boost::circular_buffer<RoboCompCamera360RGB::TImage> b_camera_queue{3};
        boost::circular_buffer<RoboCompLidar3D::TDataImage> b_lidar_queue{1};

        // Helper functions
        bool initialize_lidar();
        bool initialize_camera();
        bool get_sensor_data(RoboCompLidar3D::TDataImage &lidar_data, RoboCompCamera360RGB::TImage &cam_data);
        std::optional<std::pair<size_t, size_t>> find_best_timestamp_match();
        void process_matched_data(const RoboCompLidar3D::TDataImage &lidar_data,
                                  const RoboCompCamera360RGB::TImage &rgb_data,
                                  RoboCompLidar3D::TColorCloudData &cloud);

        // ROI processing helpers
        void normalize_roi_parameters(int &cx, int &cy, int &sx, int &sy, int &roiwidth, int &roiheight);
        void adjust_roi_for_boundaries(int &cx, int &cy, int &sx, int &sy);
        void extract_roi_images(const cv::Mat &rgb_img, const cv::Mat &depth_img,
                               int cx, int cy, int sx, int sy,
                               cv::Mat &dst_rgb, cv::Mat &dst_depth);
        cv::Mat resize_depth_image(const cv::Mat &src_depth, int target_width, int target_height);

        cv::Mat cut_image(cv::Mat image, int cx, int cy, int sx, int sy, int roiwidth, int roiheight);
        /**
         * \brief Flag indicating whether startup checks are enabled.
         */
        bool startup_check_flag;

        int MAX_WIDTH, MAX_HEIGHT;
        bool enabled_camera = false;
        bool enabled_lidar = false;
        long long capture_time;

        long long last_fused_time = 0;
        long long last_lidar_stamp = 0;
        long long last_camera_stamp = 0;

        // fps
        FPSCounter fps;
        std::atomic<std::chrono::high_resolution_clock::time_point> last_read;
        int MAX_INACTIVE_TIME = 5;  // secs after which the component is paused. It reactivates with a new reset

        // camera buffers
        cv::Mat rgb_frame_write, depth_frame_write;
        RoboCompLidar3D::TColorCloudData pointCloud;

        mutable std::shared_mutex swap_mutex;
        std::string lidarName;

        signals:
	//void customSignal();
    };

#endif
