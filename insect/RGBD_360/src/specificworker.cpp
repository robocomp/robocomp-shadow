/*
 *    Copyright (C) 2025 by YOUR NAME HERE
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

SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check) : GenericWorker(configLoader, tprx)
{
this->startup_check_flag = startup_check;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
		#ifdef HIBERNATION_ENABLED
			hibernationChecker.start(500);
		#endif

		statemachine.setChildMode(QState::ExclusiveStates);
		statemachine.start();

		auto error = statemachine.errorString();
		if (error.length() > 0){
			qWarning() << error;
			throw error;
		}
		
	}
}

SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
}

void SpecificWorker::initialize()
{
    std::cout << "Initialize worker" << std::endl;

    if(this->startup_check_flag)
    {
        this->startup_check();
        return;
    }

    // Load configuration parameters
    lidarName = this->configLoader.get<std::string>("lidar");
    params.DISPLAY = this->configLoader.get<bool>("display");
    last_read.store(std::chrono::high_resolution_clock::now());
    capture_time = -1;

    // Initialize LiDAR connection
    enabled_lidar = initialize_lidar();

    // Initialize Camera connection
    enabled_camera = initialize_camera();
}

bool SpecificWorker::initialize_lidar()
{
    std::cout << "Initializing LiDAR connection..." << std::endl;
    while(true)
    {
        try
        {
            RoboCompLidar3D::TDataImage lidar_data = this->lidar3d_proxy->getLidarDataArrayProyectedInImage(lidarName);
            std::cout << "LiDAR connection established successfully" << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cout << "Waiting for LiDAR connection: " << e.what() << std::endl;
            sleep(1);
        }
    }
}

bool SpecificWorker::initialize_camera()
{
    std::cout << "Initializing Camera connection..." << std::endl;
    while(true)
    {
        try
        {
            RoboCompCamera360RGB::TImage cam_data = this->camera360rgb_proxy->getROI(-1, -1, -1, -1, -1, -1);
            MAX_WIDTH = cam_data.width;
            MAX_HEIGHT = cam_data.height;
            std::cout << "Camera connection established successfully. Size: " << MAX_WIDTH << "x" << MAX_HEIGHT << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cout << "Waiting for Camera connection: " << e.what() << std::endl;
            sleep(1);
        }
    }
}

void SpecificWorker::compute()
{
    RoboCompLidar3D::TDataImage lidar_data;
    RoboCompCamera360RGB::TImage cam_data;
    RoboCompLidar3D::TColorCloudData cloud;

    // Get sensor data
    if (!get_sensor_data(lidar_data, cam_data))
        return;

    // Find best timestamp match
    auto match_result = find_best_timestamp_match();
    if (!match_result.has_value())
        return;

    auto [chosen_rgb, chosen_lidar] = match_result.value();

    RoboCompLidar3D::TDataImage chosen_lidar_data = b_lidar_queue.at(chosen_lidar);
    RoboCompCamera360RGB::TImage chosen_rgb_data = b_camera_queue.at(chosen_rgb);

    // Check if already processed
    if (last_fused_time == chosen_lidar_data.timestamp)
        return;

    // Process matched data
    process_matched_data(chosen_lidar_data, chosen_rgb_data, cloud);

    fps.print("Timestamps: Lidar " + std::to_string(chosen_lidar_data.timestamp) +
              " RGB " + std::to_string(chosen_rgb_data.timestamp) +
              " diff:" + std::to_string(chosen_lidar_data.timestamp - chosen_rgb_data.timestamp), 3000);
}

bool SpecificWorker::get_sensor_data(RoboCompLidar3D::TDataImage &lidar_data, RoboCompCamera360RGB::TImage &cam_data)
{
    // Get lidar data
    try
    {
        lidar_data = this->lidar3d_proxy->getLidarDataArrayProyectedInImage(lidarName);
        if (lidar_data.timestamp != last_lidar_stamp) 
        {
            b_lidar_queue.push_back(lidar_data);
            last_lidar_stamp = lidar_data.timestamp;
        }
    }
    catch (const Ice::Exception &e)
    {
        std::cout << "Error getting LiDAR data: " << e.what() << std::endl;
        return false;
    }

    // Get camera data
    try
    {
        cam_data = this->camera360rgb_proxy->getROI(-1, -1, -1, -1, -1, -1);
        if (cam_data.timestamp != last_camera_stamp) 
        {
            b_camera_queue.push_back(cam_data);
            last_camera_stamp = cam_data.timestamp;
        }
    }
    catch (const Ice::Exception &e)
    {
        std::cout << "Error getting camera data: " << e.what() << std::endl;
        return false;
    }

    return true;
}

std::optional<std::pair<size_t, size_t>> SpecificWorker::find_best_timestamp_match()
{
    int timestamp_diff = std::numeric_limits<int>::max();
    size_t chosen_rgb = 0, chosen_lidar = 0;
    bool exists_data = false;
    
    constexpr int MAX_TIMESTAMP_DIFF = 500; // microseconds

    // Iterate through lidar queue in reverse
    for (auto it_lidar = b_lidar_queue.rbegin(); it_lidar != b_lidar_queue.rend(); ++it_lidar)
    {
        const auto& lidar = *it_lidar;
        const auto j = std::distance(it_lidar, b_lidar_queue.rend()) - 1;

        // Iterate through camera queue in reverse
        for (auto it_rgb = b_camera_queue.rbegin(); it_rgb != b_camera_queue.rend(); ++it_rgb)
        {
            const auto& rgb = *it_rgb;
            const auto i = std::distance(it_rgb, b_camera_queue.rend()) - 1;

            int act_timestamp_diff = std::abs(rgb.timestamp - lidar.timestamp);
            
            if (act_timestamp_diff < timestamp_diff && act_timestamp_diff < MAX_TIMESTAMP_DIFF)
            {
                timestamp_diff = act_timestamp_diff;
                chosen_rgb = i;
                chosen_lidar = j;
                exists_data = true;

                if (timestamp_diff == 0) break;
            }
        }
        if (timestamp_diff == 0) break;
    }

    if (exists_data)
        return std::make_pair(chosen_rgb, chosen_lidar);

    return std::nullopt;
}

void SpecificWorker::process_matched_data(const RoboCompLidar3D::TDataImage &chosen_lidar_data,
                                           const RoboCompCamera360RGB::TImage &chosen_rgb_data,
                                           RoboCompLidar3D::TColorCloudData &cloud)
{
    // Generate rgb image
    cv::Mat rgb_image(cv::Size(chosen_rgb_data.width, chosen_rgb_data.height),
                      CV_8UC3, const_cast<unsigned char*>(&chosen_rgb_data.image[0]));

    // Generate depth image
    cv::Mat depth_image(cv::Size(chosen_rgb_data.width, chosen_rgb_data.height),
                        CV_32FC3, cv::Scalar(0,0,0));

    cv::Vec3f* depth_ptr = depth_image.ptr<cv::Vec3f>();
    cv::Vec3b* rgb_ptr = rgb_image.ptr<cv::Vec3b>();

    // Prepare cloud data
    cloud.timestamp = chosen_lidar_data.timestamp;
    cloud.compressed = false;

    const size_t N = chosen_lidar_data.XArray.size();
    cloud.X.resize(N);
    cloud.Y.resize(N);
    cloud.Z.resize(N);
    cloud.R.resize(N);
    cloud.G.resize(N);
    cloud.B.resize(N);

    constexpr float maxShort = 32767.0f;

    // Fill depth image and cloud data
    for (size_t i = 0; i < N; ++i)
    {
        const int px = chosen_lidar_data.XPixel[i];
        const int py = chosen_lidar_data.YPixel[i];
        const float x = chosen_lidar_data.XArray[i];
        const float y = chosen_lidar_data.YArray[i];
        const float z = chosen_lidar_data.ZArray[i];

        const int index = py * chosen_rgb_data.width + px;

        // Depth image
        cv::Vec3f& depth_pixel = depth_ptr[index];
        depth_pixel[0] = x;
        depth_pixel[1] = y;
        depth_pixel[2] = z;

        // RGB data
        const cv::Vec3b& rgb_pixel = rgb_ptr[index];

        // Cloud data
        cloud.X[i] = static_cast<int16_t>(std::clamp(x, -maxShort, maxShort));
        cloud.Y[i] = static_cast<int16_t>(std::clamp(y, -maxShort, maxShort));
        cloud.Z[i] = static_cast<int16_t>(std::clamp(z, -maxShort, maxShort));
        cloud.R[i] = rgb_pixel[2];
        cloud.G[i] = rgb_pixel[1];
        cloud.B[i] = rgb_pixel[0];
    }
    cloud.numberPoints = N;

    // Display if enabled
    if (params.DISPLAY)
    {
        cv::imshow("rgb_image", rgb_image);
        cv::imshow("depth_image", depth_image);
        cv::waitKey(1);
    }

    // Update shared data
    {
        std::unique_lock<std::shared_mutex> lock(swap_mutex);
        rgb_image.copyTo(rgb_frame_write);
        depth_image.copyTo(depth_frame_write);
        capture_time = chosen_lidar_data.timestamp;
        last_fused_time = capture_time;
        pointCloud = std::move(cloud);
    }
}

void SpecificWorker::normalize_roi_parameters(int &cx, int &cy, int &sx, int &sy, int &roiwidth, int &roiheight)
{
    if(sx == 0 || sy == 0)
    {
        std::cout << "No size. Sending complete image" << std::endl;
        sx = MAX_WIDTH;
        sy = MAX_HEIGHT;
        cx = MAX_WIDTH / 2;
        cy = MAX_HEIGHT / 2;
    }
    if(sx == -1) sx = MAX_WIDTH;
    if(sy == -1) sy = MAX_HEIGHT;
    if(cx == -1) cx = MAX_WIDTH / 2;
    if(cy == -1) cy = MAX_HEIGHT / 2;
    if(roiwidth == -1) roiwidth = MAX_WIDTH;
    if(roiheight == -1) roiheight = MAX_HEIGHT;
}

void SpecificWorker::adjust_roi_for_boundaries(int &cx, int &cy, int &sx, int &sy)
{
    // Check if y is out of range. Get max or min values in that case
    if((cy - sy / 2) < 0)
    {
        sx = static_cast<int>(static_cast<float>(sx) / static_cast<float>(sy) * 2 * cy);
        sy = 2 * cy;
    }
    else if((cy + sy / 2) >= MAX_HEIGHT)
    {
        sx = static_cast<int>(static_cast<float>(sx) / static_cast<float>(sy) * 2 * (MAX_HEIGHT - cy));
        sy = 2 * (MAX_HEIGHT - cy);
    }
}

void SpecificWorker::extract_roi_images(const cv::Mat &rgb_img, const cv::Mat &depth_img,
                                        int cx, int cy, int sx, int sy,
                                        cv::Mat &dst_rgb, cv::Mat &dst_depth)
{
    cv::Mat x_out_image_left_rgb, x_out_image_right_rgb;
    cv::Mat x_out_image_left_depth, x_out_image_right_depth;

    // Check if x is out of range. Add proportional image section in that case
    if((cx - sx / 2) < 0)
    {
        int left_width = std::abs(cx - sx / 2);
        rgb_img(cv::Rect(MAX_WIDTH - 1 - left_width, cy - sy / 2, left_width, sy)).copyTo(x_out_image_left_rgb);
        rgb_img(cv::Rect(0, cy - sy / 2, cx + sx / 2, sy)).copyTo(x_out_image_right_rgb);
        cv::hconcat(x_out_image_left_rgb, x_out_image_right_rgb, dst_rgb);

        depth_img(cv::Rect(MAX_WIDTH - 1 - left_width, cy - sy / 2, left_width, sy)).copyTo(x_out_image_left_depth);
        depth_img(cv::Rect(0, cy - sy / 2, cx + sx / 2, sy)).copyTo(x_out_image_right_depth);
        cv::hconcat(x_out_image_left_depth, x_out_image_right_depth, dst_depth);
    }
    else if((cx + sx / 2) > MAX_WIDTH)
    {
        int right_width = cx + sx / 2 - MAX_WIDTH;
        rgb_img(cv::Rect(cx - sx / 2 - 1, cy - sy / 2, MAX_WIDTH - cx + sx / 2, sy)).copyTo(x_out_image_left_rgb);
        rgb_img(cv::Rect(0, cy - sy / 2, right_width, sy)).copyTo(x_out_image_right_rgb);
        cv::hconcat(x_out_image_left_rgb, x_out_image_right_rgb, dst_rgb);

        depth_img(cv::Rect(cx - sx / 2 - 1, cy - sy / 2, MAX_WIDTH - cx + sx / 2, sy)).copyTo(x_out_image_left_depth);
        depth_img(cv::Rect(0, cy - sy / 2, right_width, sy)).copyTo(x_out_image_right_depth);
        cv::hconcat(x_out_image_left_depth, x_out_image_right_depth, dst_depth);
    }
    else
    {
        sx--;
        rgb_img(cv::Rect(cx - sx / 2, cy - sy / 2, sx, sy)).copyTo(dst_rgb);
        depth_img(cv::Rect(cx - sx / 2, cy - sy / 2, sx, sy)).copyTo(dst_depth);
    }
}

cv::Mat SpecificWorker::resize_depth_image(const cv::Mat &src_depth, int target_width, int target_height)
{
    float x_ratio = static_cast<float>(src_depth.cols) / target_width;
    float y_ratio = static_cast<float>(src_depth.rows) / target_height;

    cv::Mat dst = cv::Mat::zeros(target_height, target_width, src_depth.type());

    // Scale and assign values
    for (int y = 0; y < src_depth.rows; ++y)
    {
        for (int x = 0; x < src_depth.cols; ++x)
        {
            int newX = cvRound(x / x_ratio);
            int newY = cvRound(y / y_ratio);

            cv::Vec3f value = src_depth.at<cv::Vec3f>(y, x);
            if (cv::norm(value) != 0)
            {
                dst.at<cv::Vec3f>(newY, newX) = value;
            }
        }
    }

    return dst;
}


void SpecificWorker::emergency()
{
    std::cout << "Emergency worker" << std::endl;
	//computeCODE
	//
	//if (SUCCESSFUL)
    //  emmit goToRestore()
}

//Execute one when exiting to emergencyState
void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
	//computeCODE
	//Restore emergency component

}

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}


RoboCompCamera360RGBD::TRGBD SpecificWorker::Camera360RGBD_getROI(int cx, int cy, int sx, int sy, int roiwidth, int roiheight)
{
    RoboCompCamera360RGBD::TRGBD res;
    std::shared_lock<std::shared_mutex> lock(swap_mutex);

    if (!enabled_camera || !enabled_lidar)
        return res;

    last_read.store(std::chrono::high_resolution_clock::now());

    // Normalize ROI parameters
    normalize_roi_parameters(cx, cy, sx, sy, roiwidth, roiheight);

    // Get images from buffer
    auto depth_img = depth_frame_write;
    auto rgb_img = rgb_frame_write;

    // Adjust ROI for boundary conditions
    adjust_roi_for_boundaries(cx, cy, sx, sy);

    // Extract ROI from images
    cv::Mat dst_rgb, dst_depth;
    extract_roi_images(rgb_img, depth_img, cx, cy, sx, sy, dst_rgb, dst_depth);

    // Resize RGB image
    cv::Mat rdst_rgb;
    cv::resize(dst_rgb, rdst_rgb, cv::Size(roiwidth, roiheight), cv::INTER_LINEAR);

    // Resize depth image preserving 3D coordinates
    cv::Mat resized_depth = resize_depth_image(dst_depth, roiwidth, roiheight);

    // Fill result structure
    res.rgb.assign(rdst_rgb.data, rdst_rgb.data + (rdst_rgb.total() * rdst_rgb.elemSize()));
    res.depth.assign(resized_depth.data, resized_depth.data + (resized_depth.total() * resized_depth.elemSize()));
    res.rgbcompressed = false;
    res.depthcompressed = false;
    res.period = fps.get_period();
    res.alivetime = capture_time;
    res.rgbchannels = rdst_rgb.channels();
    res.depthchannels = resized_depth.channels();
    res.height = rdst_rgb.rows;
    res.width = rdst_rgb.cols;
    res.roi = RoboCompCamera360RGBD::TRoi{
        .xcenter=cx,
        .ycenter=cy,
        .xsize=sx,
        .ysize=sy,
        .finalxsize=res.width,
        .finalysize=res.height
    };

    return res;

}

RoboCompLidar3D::TColorCloudData SpecificWorker::Lidar3D_getColorCloudData()
{
	RoboCompLidar3D::TColorCloudData ret{};
    std::shared_lock<std::shared_mutex> lock(swap_mutex);
	return pointCloud;
}

RoboCompLidar3D::TData SpecificWorker::Lidar3D_getLidarData(std::string name, float start, float len, int decimationDegreeFactor)
{
	RoboCompLidar3D::TData ret{};
	//implementCODE

	return ret;
}

RoboCompLidar3D::TDataImage SpecificWorker::Lidar3D_getLidarDataArrayProyectedInImage(std::string name)
{
	RoboCompLidar3D::TDataImage ret{};
	//implementCODE

	return ret;
}

RoboCompLidar3D::TDataCategory SpecificWorker::Lidar3D_getLidarDataByCategory(RoboCompLidar3D::TCategories categories, Ice::Long timestamp)
{
	RoboCompLidar3D::TDataCategory ret{};
	//implementCODE

	return ret;
}

RoboCompLidar3D::TData SpecificWorker::Lidar3D_getLidarDataProyectedInImage(std::string name)
{
	RoboCompLidar3D::TData ret{};
	//implementCODE

	return ret;
}

RoboCompLidar3D::TData SpecificWorker::Lidar3D_getLidarDataWithThreshold2d(std::string name, float distance, int decimationDegreeFactor)
{
	RoboCompLidar3D::TData ret{};
	//implementCODE

	return ret;
}


/**************************************/
// From the RoboCompCamera360RGB you can call this methods:
// RoboCompCamera360RGB::TImage this->camera360rgb_proxy->getROI(int cx, int cy, int sx, int sy, int roiwidth, int roiheight)

/**************************************/
// From the RoboCompCamera360RGB you can use this types:
// RoboCompCamera360RGB::TRoi
// RoboCompCamera360RGB::TImage

/**************************************/
// From the RoboCompLidar3D you can call this methods:
// RoboCompLidar3D::TColorCloudData this->lidar3d_proxy->getColorCloudData()
// RoboCompLidar3D::TData this->lidar3d_proxy->getLidarData(std::string name, float start, float len, int decimationDegreeFactor)
// RoboCompLidar3D::TDataImage this->lidar3d_proxy->getLidarDataArrayProyectedInImage(std::string name)
// RoboCompLidar3D::TDataCategory this->lidar3d_proxy->getLidarDataByCategory(TCategories categories, long timestamp)
// RoboCompLidar3D::TData this->lidar3d_proxy->getLidarDataProyectedInImage(std::string name)
// RoboCompLidar3D::TData this->lidar3d_proxy->getLidarDataWithThreshold2d(std::string name, float distance, int decimationDegreeFactor)

/**************************************/
// From the RoboCompLidar3D you can use this types:
// RoboCompLidar3D::TPoint
// RoboCompLidar3D::TDataImage
// RoboCompLidar3D::TData
// RoboCompLidar3D::TDataCategory
// RoboCompLidar3D::TColorCloudData

/**************************************/
// From the RoboCompCamera360RGBD you can use this types:
// RoboCompCamera360RGBD::TRoi
// RoboCompCamera360RGBD::TRGBD

/**************************************/
// From the RoboCompLidar3D you can use this types:
// RoboCompLidar3D::TPoint
// RoboCompLidar3D::TDataImage
// RoboCompLidar3D::TData
// RoboCompLidar3D::TDataCategory
// RoboCompLidar3D::TColorCloudData

