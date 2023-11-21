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
//	THE FOLLOWING IS JUST AN EXAMPLE
//	To use innerModelPath parameter you should uncomment specificmonitor.cpp readConfig method content
//	try
//	{
//		RoboCompCommonBehavior::Parameter par = params.at("InnerModelPath");
//		std::string innermodel_path = par.value;
//		innerModel = std::make_shared(innermodel_path);
//	}
//	catch(const std::exception &e) { qFatal("Error reading config params"); }






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
		timer.start(Period);
	}
    capture_time = -1;
    while(!enabled_lidar)
    {
        try
        {
            RoboCompLidar3D::TDataImage lidar_data = this->lidar3d_proxy->getLidarDataArrayProyectedInImage("helios");
            enabled_lidar = true;
        }
        catch (const std::exception &e){std::cout << e.what() << std::endl; return;}
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
        catch (const std::exception &e){std::cout << e.what() << std::endl; return;}
    }
}

void SpecificWorker::compute()
{

    auto cstart = std::chrono::high_resolution_clock::now();
    RoboCompLidar3D::TDataImage lidar_data;
    RoboCompCamera360RGB::TImage cam_data;
        // std::cout << "1" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clean).count() << " microseconds" << std::endl<<std::flush;

    // Get lidar data
    try
    {
        lidar_data = this->lidar3d_proxy->getLidarDataArrayProyectedInImage("helios");
        lidar_queue.push(lidar_data);
    }
    catch (const std::exception &e){std::cout << e.what() << std::endl; return;}

    // Get camera data
    try
    {
        cam_data = this->camera360rgb_proxy->getROI(-1, -1, -1, -1, -1, -1);
        camera_queue.push(cam_data);
    }
    catch (const std::exception &e){std::cout << e.what() << std::endl; return;}

    // std::cout << "2" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clean).count() << " microseconds" << std::endl<<std::flush;


    // capture_time = duration_cast< milliseconds >(system_clock::now().time_since_epoch()).count();
    int timestamp_diff = 999999999;
    int chosen_rgb, chosen_lidar;
    bool exists_data = false;
    auto cstart2 = std::chrono::high_resolution_clock::now();
    // std::cout << "3" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clean).count() << " microseconds" << std::endl<<std::flush;

//    std::cout << "CAMERA QUEUE SIZE " << camera_queue.size() << std::endl;
//    std::cout << "LIDAR QUEUE SIZE " << lidar_queue.size() << std::endl;
    for(const auto &[i, rgb] : camera_queue | iter::enumerate)
    {
        for(const auto &[j, lidar] : lidar_queue | iter::enumerate)
        {
            int act_timestamp_diff = abs(rgb.alivetime - lidar.timestamp);

//            std::cout << "timestamp diff: " << act_timestamp_diff << std::endl;
            if(act_timestamp_diff < timestamp_diff and act_timestamp_diff < 30000)
            {
                timestamp_diff = act_timestamp_diff;
                chosen_rgb = i;
                chosen_lidar = j;
                exists_data = true;
                break;
            }
        }
        if(exists_data){break;}
    }
//    std::cout << "Time for " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - cstart2).count() << " milli" << std::endl<<std::flush;

    if(exists_data)
    {


//        std::cout << "Min timestamp: " << timestamp_diff << " " << chosen_rgb << " " << chosen_lidar << std::endl;
        RoboCompLidar3D::TDataImage chosen_lidar_data = lidar_queue.at(chosen_lidar);
        RoboCompCamera360RGB::TImage chosen_rgb_data = camera_queue.at(chosen_rgb);
        // std::cout << "4 " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clean).count() << " microseconds" << std::endl<<std::flush;

        // std::cout << "5" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clean).count() << " microseconds" << std::endl<<std::flush;

        // Generate rgb image
        cv::Mat rgb_image(cv::Size(chosen_rgb_data.width, chosen_rgb_data.height), CV_8UC3, &chosen_rgb_data.image[0]);
        // Generate depth image
        cv::Mat depth_image(cv::Size(chosen_rgb_data.width, chosen_rgb_data.height), CV_32FC3, cv::Scalar(0,0,0));
        // Obtener puntero al inicio de la matriz
        auto start_clean = std::chrono::high_resolution_clock::now();
        cv::Vec3f* ptr = depth_image.ptr<cv::Vec3f>();

        for (auto&& [px, py, x, y, z] : iter::zip(chosen_lidar_data.XPixel, chosen_lidar_data.YPixel, chosen_lidar_data.XArray, chosen_lidar_data.YArray, chosen_lidar_data.ZArray))
        {
            cv::Vec3f& pixel = ptr[py * chosen_rgb_data.width + px];
            pixel[0] = x;
            pixel[1] = y;
            pixel[2] = z;
        }
        // std::cout << "6" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clean).count() << " microseconds" << std::endl<<std::flush;

    //    cv::imshow("rgb_image", rgb_image);
    //    cv::imshow("depth_image", depth_image);
    //    cv::waitKey(1);
        // buffer_rgb_image.put(std::move(rgb_image));
        // buffer_depth_image.put(std::move(depth_image));

        swap_mutex.lock();
            rgb_image.copyTo(rgb_frame_write);
            depth_image.copyTo(depth_frame_write);
            capture_time = chosen_rgb_data.alivetime;
        swap_mutex.unlock();

        lidar_queue.clean_old(chosen_lidar);
        camera_queue.clean_old(chosen_rgb);
        // std::cout << "7" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clean).count() << " microseconds" << std::endl<<std::flush;
    }
    std::cout << "Compute time " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - cstart).count() << " milli" << std::endl<<std::flush;
}

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}


RoboCompCamera360RGBD::TRGBD SpecificWorker::Camera360RGBD_getROI(int cx, int cy, int sx, int sy, int roiwidth, int roiheight)
{
    auto start = std::chrono::high_resolution_clock::now();
    RoboCompCamera360RGBD::TRGBD res;
    const std::lock_guard<std::mutex> lg(swap_mutex);
    if (enabled_camera and enabled_lidar)
    {
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
        // cv::resize(dst_depth, rdst_depth, cv::Size(roiwidth, roiheight), cv::INTER_NEAREST);
        auto start_clean = std::chrono::high_resolution_clock::now();

        float x_ratio = float(dst_depth.cols) / roiwidth;
        float y_ratio = float(dst_depth.rows) / roiheight;
        
        // Crear una imagen de destino con las nuevas dimensiones
        cv::Mat dst = cv::Mat::zeros(roiheight, roiwidth, dst_depth.type());
        
        // Escalar y asignar los valores
        for (int y = 0; y < dst_depth.rows; ++y) {
            for (int x = 0; x < dst_depth.cols; ++x) {
                int newX = cvRound(x * x_ratio);
                int newY = cvRound(y * y_ratio);
                
                // Asumiendo que la imagen es de tres canales (color)
                cv::Vec3f value = dst_depth.at<cv::Vec3f>(y, x);
                if (cv::norm(value) != 0) { // Si el vector del pixel de la fuente es diferente de cero, copiamos el valor
                    dst.at<cv::Vec3f>(newY, newX) = value;
                }
            }
        }

        std::cout << "For time " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_clean).count() << " microseconds" << std::endl<<std::flush;

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
    std::cout << "Time expended " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " milli" << std::endl<<std::flush;

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

