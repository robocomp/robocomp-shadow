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
	return true;
}

void SpecificWorker::initialize(int period)
{
	std::cout << "Initialize worker" << std::endl;

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
            catch (const std::exception &e){std::cout << " In Initialize getting LiDAR " << e.what() << std::endl; return;}
            sleep(1);
        }
        while(!enabled_camera)
        {
            try
            {
                RoboCompCamera360RGB::TImage cam_data = this->camera360rgb_proxy->getROI(-1, -1, -1, -1, -1, -1);
                MAX_WIDTH = cam_data.width;
                MAX_HEIGHT = cam_data.height;
                enabled_camera = true;
                sleep(1);
            }
            catch (const std::exception &e){std::cout << " In Initialize getting Camera " <<e.what() << std::endl; return;}
        }
	    this->Period = 1;
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
    RoboCompCamera360RGB::TImage cam_data;


    // Get lidar data
    try
    {
        lidar_data = this->lidar3d_proxy->getLidarDataArrayProyectedInImage("helios");
        if (lidar_data.timestamp != last_lidar_stamp) 
        {
            b_lidar_queue.push_back(lidar_data);
            last_lidar_stamp = lidar_data.timestamp;
            // qInfo() << "Lidar timestamp" << lidar_data.timestamp;
        } 
        else 
        {
            // qInfo() << "Lidar data with timestamp" << lidar_data.timestamp << "already exists in buffer. Stopping compute";
        }
    }
    catch (const Ice::Exception &e){std::cout << " In getting LiDAR data " << e.what() << std::endl; return;}

    // Get camera data
    try
    {
        cam_data = this->camera360rgb_proxy->getROI(-1, -1, -1, -1, -1, -1);
        if (cam_data.timestamp != last_camera_stamp) 
        {
            b_camera_queue.push_back(cam_data);
            last_camera_stamp = cam_data.timestamp;
            // qInfo() << "RGB timestamp" << cam_data.timestamp;
        } 
        else {
            // qInfo() << "Camera data with timestamp" << cam_data.timestamp << "already exists in buffer";
        }
    }
    catch (const Ice::Exception &e){std::cout << " In getting LiDAR data " << e.what() << std::endl; return;}

    // capture_time = duration_cast< milliseconds >(system_clock::now().time_since_epoch()).count();
    // int timestamp_diff = 999999999;
    // int chosen_rgb, chosen_lidar;
    // bool exists_data = false;
    // for(const auto &[i, rgb] : b_camera_queue | iter::enumerate)
    // {
    //     for(const auto &[j, lidar] : b_lidar_queue | iter::enumerate)
    //     {
    //         int act_timestamp_diff = abs(rgb.alivetime - lidar.timestamp);
    //
    //         //std::cout << "timestamp diff: " << act_timestamp_diff << " Queue sizes:" << b_camera_queue.size() << "," << b_lidar_queue.size() <<  std::endl;
    //         if(act_timestamp_diff < timestamp_diff and act_timestamp_diff < 300000)
    //         {
    //             timestamp_diff = act_timestamp_diff;
    //             chosen_rgb = i;
    //             chosen_lidar = j;
    //             exists_data = true;
    //             break;
    //         }
    //     }
    //     if(exists_data){break;}
    // }

    int timestamp_diff = std::numeric_limits<int>::max();
    size_t chosen_rgb = 0, chosen_lidar = 0;
    bool exists_data = false;
    


    // Iterate through lidar queue in reverse
    for (auto it_lidar = b_lidar_queue.rbegin(); it_lidar != b_lidar_queue.rend(); ++it_lidar)
    {
        const auto& lidar = *it_lidar;
        const auto j = std::distance(it_lidar, b_lidar_queue.rend()) - 1; // Calculate reverse index
        // qInfo() << "LIDAR Timestamp loop" << lidar.timestamp;
        // Iterate through camera queue in reverse
        for (auto it_rgb = b_camera_queue.rbegin(); it_rgb != b_camera_queue.rend(); ++it_rgb)
        {
            const auto& rgb = *it_rgb;
            const auto i = std::distance(it_rgb, b_camera_queue.rend()) - 1; // Calculate reverse index
            // qInfo() << "RGB Timestamp loop" << rgb.timestamp;
            int act_timestamp_diff = std::abs(rgb.timestamp - lidar.timestamp);
            
            if (act_timestamp_diff < timestamp_diff && act_timestamp_diff < 500)
            {
                timestamp_diff = act_timestamp_diff;
                chosen_rgb = i;
                chosen_lidar = j;
                exists_data = true;

                if(timestamp_diff == 0) break;
            }
        }
        if(timestamp_diff == 0) break;
    }

    if(exists_data)
    {        
        RoboCompLidar3D::TDataImage chosen_lidar_data = b_lidar_queue.at(chosen_lidar);
        RoboCompCamera360RGB::TImage chosen_rgb_data = b_camera_queue.at(chosen_rgb);

        std::cout << "Timestamps " <<  chosen_lidar_data.timestamp << " " << chosen_rgb_data.timestamp <<  std::endl;
        std::cout << "timestamp diff: " << chosen_lidar_data.timestamp - chosen_rgb_data.timestamp <<  std::endl;
        
        if(last_fused_time == chosen_lidar_data.timestamp) 
        {
            // qInfo() << "Same lidar matched. Returning"; 
            return;
        }
        // auto &[chosen_lidar_data, chosen_rgb_data] = opt_tuple.value();


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

        swap_mutex.lock();
            rgb_image.copyTo(rgb_frame_write);
            depth_image.copyTo(depth_frame_write);
            capture_time = chosen_lidar_data.timestamp;
            last_fused_time = capture_time;
        swap_mutex.unlock();

        //lidar_queue.clean_old(chosen_lidar);
        //camera_queue.clean_old(chosen_rgb);
    }

    fps.print("FPS:", 3000);
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
    const std::lock_guard<std::mutex> lg(swap_mutex);
    if (enabled_camera and enabled_lidar)
    {
        last_read.store(std::chrono::high_resolution_clock::now());

        if(sx == 0 || sy == 0)
        {
            std::cout << "No size. Sending complete image" << std::endl;
            sx = MAX_WIDTH; sy = MAX_HEIGHT;
            cx = (int)(MAX_WIDTH/2); cy = int(MAX_HEIGHT/2);
        }
        if(sx == -1)
            sx = MAX_WIDTH;
        if(sy == -1)
            sy = MAX_HEIGHT;
        if(cx == -1)
            cx = (int)(MAX_WIDTH/2);
        if(cy == -1)
            cy = int(MAX_HEIGHT/2);
        if(roiwidth == -1)
            roiwidth = MAX_WIDTH;
        if(roiheight == -1)
            roiheight = MAX_HEIGHT;

        // Get image from doublebuffer
        // auto depth_img = buffer_depth_image.get_idemp();
        // auto rgb_img = buffer_rgb_image.get_idemp();

        auto depth_img = depth_frame_write;
        auto rgb_img = rgb_frame_write;

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
        for (int y = 0; y < dst_depth.rows; ++y) {
            for (int x = 0; x < dst_depth.cols; ++x) {
                int newX = cvRound(x / x_ratio);
                int newY = cvRound(y / y_ratio);
                
                // Asumiendo que la imagen es de tres canales (color)
                cv::Vec3f value = dst_depth.at<cv::Vec3f>(y, x);
                if (cv::norm(value) != 0) { // Si el vector del pixel de la fuente es diferente de cero, copiamos el valor
                    dst.at<cv::Vec3f>(newY, newX) = value;
                }
            }
        }

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
        res.rgb.assign(rdst_rgb.data, rdst_rgb.data + (rdst_rgb.total() * rdst_rgb.elemSize()));
        res.depth.assign(dst.data, dst.data + (dst.total() * dst.elemSize()));
        res.rgbcompressed = false;
        res.depthcompressed = false;
        res.period = fps.get_period();
        res.alivetime = capture_time;

        res.rgbchannels = rdst_rgb.channels();
        res.depthchannels = dst.channels();
        res.height = rdst_rgb.rows;
        res.width = rdst_rgb.cols;
        res.roi = RoboCompCamera360RGBD::TRoi{.xcenter=cx, .ycenter=cy, .xsize=sx, .ysize=sy, .finalxsize=res.width, .finalysize=res.height};
    }
    
    return res;

}



/**************************************/
// From the RoboCompCamera360RGB you can call this methods:
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
