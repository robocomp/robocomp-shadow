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
    this->Period = std::stoi(params.at("period").value);
	return true;
}

void SpecificWorker::initialize(int period)
{
	std::cout << "Initialize worker" << std::endl;
	// Period is set in the setParams method

	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
        // pause variable
        last_read.store(std::chrono::high_resolution_clock::now());

        capture_time = -1;
        while(!enabled_lidar)
        {
            try
            {
                RoboCompLidar3D::TDataImage lidar_data = this->lidar3d_proxy->getLidarDataArrayProyectedInImage("helios");
                enabled_lidar = true;
            }
            catch (const std::exception &e){std::cout << __FUNCTION__ << " " << e.what() << std::endl; return;}
        }
        while(!enabled_camera)
        {
            try
            {
                RoboCompCamera360RGB::TImage cam_data = this->camera360rgb_proxy->getROI(-1, -1, -1, -1, -1, -1);
                MAX_WIDTH = cam_data.width;
                MAX_HEIGHT = cam_data.height;
                enabled_camera = true;
            }
            catch (const std::exception &e){std::cout << __FUNCTION__ << " " << e.what() << std::endl; return;}
        }
		timer.start(Period);
	}
}

void SpecificWorker::compute()
{
    /// check idle time
    if(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - last_read.load()).count() > MAX_INACTIVE_TIME)
    {
        fps.print("No requests in the last 5 seconds. Pausing. Comp wil continue in next call", 3000);
        return;
    }

    RoboCompLidar3D::TDataImage lidar_data;
    // Get lidar data
    try
    {
        lidar_data = this->lidar3d_proxy->getLidarDataArrayProyectedInImage("helios");
        //lidar_queue.push(lidar_data);
        buffer_sync.put<1>(std::move(lidar_data), lidar_data.timestamp);

    }
    catch (const Ice::Exception &e){std::cout << __FUNCTION__ << " " <<  e.what() << std::endl; return;}

    // Get camera data
    RoboCompCamera360RGB::TImage cam_data;
    try
    {
        cam_data = this->camera360rgb_proxy->getROI(-1, -1, -1, -1, -1, -1);
        //camera_queue.push(cam_data);
        buffer_sync.put<0>(std::move(cam_data), cam_data.alivetime);
    }
    catch (const Ice::Exception &e){std::cout << __FUNCTION__ << " " <<  e.what() << std::endl; return;}

    // capture_time = duration_cast< milliseconds >(system_clock::now().time_since_epoch()).count();
//    int timestamp_diff = 999999999;
//    int chosen_rgb, chosen_lidar;
//    bool exists_data = false;
//    for(const auto &[i, rgb] : camera_queue | iter::enumerate)
//    {
//        for(const auto &[j, lidar] : lidar_queue | iter::enumerate)
//        {
//            int act_timestamp_diff = abs(rgb.alivetime - lidar.timestamp);
//
//            // std::cout << "timestamp diff: " << act_timestamp_diff << std::endl;
//            if(act_timestamp_diff < timestamp_diff and act_timestamp_diff < 300000)
//            {
//                timestamp_diff = act_timestamp_diff;
//                chosen_rgb = i;
//                chosen_lidar = j;
//                exists_data = true;
//                break;
//            }
//        }
//        if(exists_data){break;}
//    }


    auto [cam, laser] = buffer_sync.read_last();
    if(not laser or not cam)
    { qWarning() << __FUNCTION__ << " No data in buffer"; return;}
    qDebug() << "Diff: " << cam.value().alivetime - laser.value().timestamp;

    RoboCompLidar3D::TDataImage chosen_lidar_data = laser.value();
    RoboCompCamera360RGB::TImage chosen_rgb_data = cam.value();
    //
    // RoboCompLidar3D::TDataImage chosen_lidar_data = lidar_queue.at(chosen_lidar);
    // RoboCompCamera360RGB::TImage chosen_rgb_data = camera_queue.at(chosen_rgb);
    // qDebug() << "Chosen data: " << chosen_rgb_data.alivetime << " " << chosen_lidar_data.timestamp;

    // Generate rgb image
    cv::Mat rgb_image(cv::Size(chosen_rgb_data.width, chosen_rgb_data.height), CV_8UC3, &chosen_rgb_data.image[0]);
    // Generate depth image
    cv::Mat depth_image(cv::Size(chosen_rgb_data.width, chosen_rgb_data.height), CV_32FC3, cv::Scalar(0,0,0));
   // get pointer to the image data
    cv::Vec3f* ptr = depth_image.ptr<cv::Vec3f>();

    for (auto&& [px, py, x, y, z] : iter::zip(chosen_lidar_data.XPixel, chosen_lidar_data.YPixel, chosen_lidar_data.XArray, chosen_lidar_data.YArray, chosen_lidar_data.ZArray))
    {
        cv::Vec3f& pixel = ptr[py * chosen_rgb_data.width + px];
        pixel[0] = x;
        pixel[1] = y;
        pixel[2] = z;
    }

    if(params.DISPLAY)
    {
        cv::imshow("rgb_image", rgb_image);
        cv::imshow("depth_image", depth_image);
        cv::waitKey(1);
    }

    auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    cv::Mat rgb_image_copy, depth_image_copy;
    rgb_image.copyTo(rgb_image_copy);
    depth_image.copyTo(depth_image_copy);
    buffer_transfer.put<0>(std::move(rgb_image_copy), ts);
    buffer_transfer.put<1>(std::move(depth_image_copy), ts);

//        swap_mutex.lock();
//            rgb_image.copyTo(rgb_frame_write);
//            depth_image.copyTo(depth_frame_write);
//            // capture_time = chosen_rgb_data.alivetime;
//        swap_mutex.unlock();

    //lidar_queue.clean_old(chosen_lidar);
    //camera_queue.clean_old(chosen_rgb);

    fps.print("FPS:", 3000);
}

//////// AUXILIARY FUNCTIONS ////////////////////////////////////////////////////////////////////////////
int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// INTERFACE FUNCTIONS
//////////////////////////////////////////////////////////////////////////////////////////////////////////

RoboCompCamera360RGBD::TRGBD SpecificWorker::Camera360RGBD_getROI(int cx, int cy, int sx, int sy, int roiwidth, int roiheight)
{
    RoboCompCamera360RGBD::TRGBD res;
    if (not enabled_camera or not enabled_lidar)
    return res;

    last_read.store(std::chrono::high_resolution_clock::now());

    //const std::lock_guard<std::mutex> lg(swap_mutex);

    if(sx == 0 or sy == 0)
    {
        std::cout << __FUNCTION__ << " No size. Sending complete image" << std::endl;
        sx = MAX_WIDTH; sy = MAX_HEIGHT;
        cx = (int)(MAX_WIDTH/2); cy = int(MAX_HEIGHT/2);
    }
    if(sx == -1) sx = MAX_WIDTH;
    if(sy == -1) sy = MAX_HEIGHT;
    if(cx == -1) cx = (int)(MAX_WIDTH/2);
    if(cy == -1) cy = int(MAX_HEIGHT/2);
    if(roiwidth == -1) roiwidth = MAX_WIDTH;
    if(roiheight == -1) roiheight = MAX_HEIGHT;

    // Get image and lidar from buffer_sync
    const auto &[rgb_img_, depth_img_] = buffer_transfer.read_first();
    if(not rgb_img_ or not depth_img_) { qWarning() << __FUNCTION__ << " No data in buffer"; return res;}

    auto rgb_img = rgb_img_.value();
    auto depth_img = depth_img_.value();

    // auto rgb_img = buffer_rgb_image.get_idemp();

    //auto depth_img = depth_frame_write;
    //auto rgb_img = rgb_frame_write;


    // std::cout << "RGB SIZE " <<  rgb_img.rows << " " << rgb_img.cols << std::endl;
    // std::cout << "DEPTH SIZE " <<  depth_img.rows << " " << depth_img.cols << std::endl;

    // Check if y is out of range. Get max or min values in that case
    if((cy - (int) (sy / 2)) < 0)
    {
        sx = (int) ((float) sx / (float) sy * 2 * cy );
        sy = 2*cy;
    }
    else if((cy + (int) (sy / 2)) >= MAX_HEIGHT)
    {
        sx = (int) ((float) sx / (float) sy * 2 * (MAX_HEIGHT - cy) );
        sy = 2 * (MAX_HEIGHT - cy);
    }

    // Check if x is out of range. Add proportional image section in that case
    cv::Mat x_out_image_left_rgb, x_out_image_right_rgb, dst_rgb, rdst_rgb;
    cv::Mat x_out_image_left_depth, x_out_image_right_depth, dst_depth, rdst_depth;
    if((cx - (int) (sx / 2)) < 0)
    {
        rgb_img(cv::Rect(MAX_WIDTH - 1 - abs (cx - (int) (sx / 2)), cy - (int) (sy / 2), abs (cx - (int) (sx / 2)), sy)).copyTo(x_out_image_left_rgb);
        rgb_img(cv::Rect(0, cy - (int) (sy / 2),  cx + (int) (sx / 2), sy)).copyTo(x_out_image_right_rgb);
        cv::hconcat(x_out_image_left_rgb, x_out_image_right_rgb, dst_rgb);

        depth_img(cv::Rect(MAX_WIDTH - 1 - abs (cx - (int) (sx / 2)), cy - (int) (sy / 2), abs (cx - (int) (sx / 2)), sy)).copyTo(x_out_image_left_depth);
        depth_img(cv::Rect(0, cy - (int) (sy / 2),  cx + (int) (sx / 2), sy)).copyTo(x_out_image_right_depth);
        cv::hconcat(x_out_image_left_depth, x_out_image_right_depth, dst_depth);
    }
    else if((cx + (int) (sx / 2)) > MAX_WIDTH)
    {
        rgb_img(cv::Rect(cx - (int) (sx / 2) - 1, cy - (int) (sy / 2), MAX_WIDTH - cx + (int) (sx / 2), sy)).copyTo(x_out_image_left_rgb);
        rgb_img(cv::Rect(0, cy - (int) (sy / 2), cx + (int) (sx / 2) - MAX_WIDTH, sy)).copyTo(x_out_image_right_rgb);
        cv::hconcat(x_out_image_left_rgb, x_out_image_right_rgb, dst_rgb);
        depth_img(cv::Rect(cx - (int) (sx / 2) - 1, cy - (int) (sy / 2), MAX_WIDTH - cx + (int) (sx / 2), sy)).copyTo(x_out_image_left_depth);
        depth_img(cv::Rect(0, cy - (int) (sy / 2), cx + (int) (sx / 2) - MAX_WIDTH, sy)).copyTo(x_out_image_right_depth);
        cv::hconcat(x_out_image_left_depth, x_out_image_right_depth, dst_depth);
    }
    else
    {
        sx--;
        rgb_img(cv::Rect(cx - (int) (sx / 2), cy - (int) (sy / 2), sx, sy)).copyTo(dst_rgb);
        depth_img(cv::Rect(cx - (int) (sx / 2), cy - (int) (sy / 2), sx, sy)).copyTo(dst_depth);
    }

    cv::resize(dst_rgb, rdst_rgb, cv::Size(roiwidth, roiheight), cv::INTER_LINEAR);
    float x_ratio = float(dst_depth.cols) / roiwidth;
    float y_ratio = float(dst_depth.rows) / roiheight;

    // Crear una imagen de destino con las nuevas dimensiones
    cv::Mat dst = cv::Mat::zeros(roiheight, roiwidth, dst_depth.type());

    // Escalar y asignar los valores
    for (int y = 0; y < dst_depth.rows; ++y)
        for (int x = 0; x < dst_depth.cols; ++x)
        {
            int newX = cvRound(x / x_ratio);
            int newY = cvRound(y / y_ratio);

            // Asumiendo que la imagen es de tres canales (color)
            cv::Vec3f value = dst_depth.at<cv::Vec3f>(y, x);
            if (cv::norm(value) != 0) { // Si el vector del pixel de la fuente es diferente de cero, copiamos el valor
                dst.at<cv::Vec3f>(newY, newX) = value;
            }
        }

    res.rgb.assign(rdst_rgb.data, rdst_rgb.data + (rdst_rgb.total() * rdst_rgb.elemSize()));
    res.depth.assign(dst.data, dst.data + (dst.total() * dst.elemSize()));
    res.rgbcompressed = false;
    res.depthcompressed = false;
    res.period = fps.get_period();
    res.alivetime = duration_cast< milliseconds >(system_clock::now().time_since_epoch()).count();

    res.rgbchannels = rdst_rgb.channels();
    res.depthchannels = dst.channels();
    res.height = rdst_rgb.rows;
    res.width = rdst_rgb.cols;
    res.roi = RoboCompCamera360RGBD::TRoi{.xcenter=cx, .ycenter=cy, .xsize=sx, .ysize=sy, .finalxsize=res.width, .finalysize=res.height};
    return res;
}

/**************************************/
// From the RoboCompCamera360RGB you can call these methods:
// this->camera360rgb_proxy->getROI(...)

/**************************************/
// From the RoboCompCamera360RGB you can use this types:
// RoboCompCamera360RGB::TRoi
// RoboCompCamera360RGB::TImage

/**************************************/
// From the RoboCompLidar3D you can call this methods:
// this->lidar3d_proxy->getLidarData(...)
// this->lidar3d_proxy->getLidarDataArrayProyectedInImage(...)
// this->lidar3d_proxy->getLidarDataProyectedInImage(...)
// this->lidar3d_proxy->getLidarDataWithThreshold2d(...)

/**************************************/
// From the RoboCompLidar3D you can use this types:
// RoboCompLidar3D::TPoint
// RoboCompLidar3D::TDataImage
// RoboCompLidar3D::TData

/**************************************/
// From the RoboCompCamera360RGBD you can use this types:
// RoboCompCamera360RGBD::TRoi
// RoboCompCamera360RGBD::TRGBD

// std::cout << "For time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clean).count() << " microseconds" << std::endl<<std::flush;

//        if (pars.compressed)
//        {
//            std::vector<uchar> buffer;
//            cv::imencode(".jpg", rdst, buffer, compression_params);
//            res.image = buffer;
//            res.compressed = true;
//        } else
//        {
//            res.image.assign(rdst.data, rdst.data + (rdst.total() * rdst.elemSize()));
//            res.compressed = false;
//        }