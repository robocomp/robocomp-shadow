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
#include <thread>
#include "Eigen/Dense"

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
		
		
		
		// Example statemachine:
		/***
		//Your definition for the statesmachine (if you dont want use a execute function, use nullptr)
		states["CustomState"] = std::make_unique<GRAFCETStep>("CustomState", period, 
															std::bind(&SpecificWorker::customLoop, this),  // Cyclic function
															std::bind(&SpecificWorker::customEnter, this), // On-enter function
															std::bind(&SpecificWorker::customExit, this)); // On-exit function

		//Add your definition of transitions (addTransition(originOfSignal, signal, dstState))
		states["CustomState"]->addTransition(states["CustomState"].get(), SIGNAL(entered()), states["OtherState"].get());
		states["Compute"]->addTransition(this, SIGNAL(customSignal()), states["CustomState"].get()); //Define your signal in the .h file under the "Signals" section.

		//Add your custom state
		statemachine.addState(states["CustomState"].get());
		***/

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
    running = false;
    zed.close();
}

void SpecificWorker::initialize()
{   
    simulated = configLoader.get<bool>("Config.Simulated");

    display = configLoader.get<bool>("Config.Display");

    if (!simulated){
        // INIT PARAMETERS
        init_parameters.camera_resolution = sl::RESOLUTION::HD720; // Use HD720 opr HD1200 video mode, depending on camera type.
        init_parameters.camera_fps = 30; // Set fps at 30
        #ifdef POSE
            init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL_PLUS; // Use ULTRA depth mode
        #else
            init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL_PLUS; // Use ULTRA depth mode
        #endif
        init_parameters.coordinate_units = sl::UNIT::MILLIMETER; // Use millimeter units (for depth measurements)
        init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP; // Use a right-handed Y-up coordinate system
        init_parameters.depth_minimum_distance = 650 ;
        init_parameters.enable_image_enhancement = true; // Mejora visual en condiciones complejas
        //init_parameters.async_grab_camera_recovery = false; // Evita bloqueos si hay pérdida temporal de conexión
        //init_parameters.camera_disable_self_calib = false; // Habilitar autocalibración mejora tracking en largo plazo

        // Abrir cámara
        returned_state = zed.open(init_parameters);
        if (returned_state != sl::ERROR_CODE::SUCCESS) {
            std::cout << "Error " << returned_state << ", exit program." << std::endl;
            exit(0);
        }

        // TRACKING PARAMETERS
        #ifndef POSE
            //init_parameters.camera_fps = 60; // 30 FPS balancea precisión y estabilidad

            tracking_parameters.enable_imu_fusion = true; // Combina visual e IMU
            tracking_parameters.enable_pose_smoothing = true; // Estabiliza la pose (algo más lento)
            tracking_parameters.set_as_static = false; // Asegura que use movimiento real
            tracking_parameters.mode = sl::POSITIONAL_TRACKING_MODE::GEN_2; // Último modo de seguimiento (mejor IMU)
            tracking_parameters.enable_area_memory = true;

        #endif

        auto err = zed.enablePositionalTracking(tracking_parameters);
        if (err != sl::ERROR_CODE::SUCCESS)
            exit(-1);

        sl::Transform reset_transform;
        zed.resetPositionalTracking(reset_transform);

        // Get the distance between the center of the camera and the left eye
        translation_left_to_center = zed.getCameraInformation().camera_configuration.calibration_parameters.stereo_transform;
        // Lanzar hilo
        running = true;

    //IMU, barometer and magnetometer
    // sensor_thread = std::thread(&SpecificWorker::sensorsLoop, this);
    }
}

void SpecificWorker::compute()
{
    if (simulated){
        process_RGBD_data();
    }
    else{
        auto start = std::chrono::high_resolution_clock::now();
        returned_state = zed.grab();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        // std::cout << "Elapsed grab: " << elapsed << " microseconds" << std::endl;
        // A new image is available if grab() returns ERROR_CODE::SUCCESS
        if (returned_state == sl::ERROR_CODE::SUCCESS)
        {
            process_RGBD_data();
            process_pose_data();
        }
    }
}


void SpecificWorker::emergency()
{
    std::cout << "Emergency worker" << std::endl;
    //emergencyCODE
    //
    //if (SUCCESSFUL) //The componet is safe for continue
    //  emmit goToRestore()
}

//Execute one when exiting to emergencyState
void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
    //restoreCODE
    //Restore emergency component

}

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
	return 0;
}

void SpecificWorker::process_RGBD_data()
{   
    RoboCompCameraRGBDSimple::TRGBD rgbd;
    if (!simulated){
        // Retrieve left image
        zed.retrieveImage(image, sl::VIEW::LEFT);
        // Retrieve colored point cloud. Point cloud is aligned on the left image.
        zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZ);

        cv::Mat cv_image = cv::Mat(image.getHeight(), image.getWidth(), CV_8UC4, image.getPtr<sl::uchar1>(sl::MEM::CPU));
        // Wrap the sl::Mat in a cv::Mat without copying
        cv::Mat pointcloud_mat = cv::Mat(point_cloud.getHeight(),
                            point_cloud.getWidth(),
                            CV_32FC4,
                            point_cloud.getPtr<sl::float1>(sl::MEM::CPU));

        RoboCompCameraRGBDSimple::TImage rgb_image;
        rgb_image.width = cv_image.cols;
        rgb_image.height = cv_image.rows;
    //    rgb_image.cameraID = 0;
    //    depth.focalx = depth_intr.fx;
    //    depth.focaly = depth_intr.fy;
        rgb_image.alivetime = image.timestamp.getNanoseconds();
    //    depth.period = fps.get_period();
        rgb_image.compressed = false;
        rgb_image.image.assign(cv_image.data, cv_image.data + (cv_image.total() * cv_image.elemSize()));

        RoboCompCameraRGBDSimple::TDepth depth_image;
        depth_image.width = pointcloud_mat.cols;
        depth_image.height = pointcloud_mat.rows;
    //    rgb_image.cameraID = 0;
    //    depth.focalx = depth_intr.fx;
    //    depth.focaly = depth_intr.fy;
        depth_image.alivetime = rgb_image.alivetime;
    //    depth.period = fps.get_period();
        depth_image.compressed = false;
        depth_image.depth.assign(pointcloud_mat.data, pointcloud_mat.data + (pointcloud_mat.total() * pointcloud_mat.elemSize()));
    //    qInfo() << "Publishing";
        rgbd.image = rgb_image;    
        rgbd.depth = depth_image;
    }
    else{
        RoboCompCameraRGBDSimple::TImage rgb_image = this->camerargbdsimple_proxy->getImage("");
        RoboCompCameraRGBDSimple::TDepth depth_image = this->camerargbdsimple_proxy->getDepth("");
        RoboCompCameraRGBDSimple::TPoints points;

        cv::Mat depthMat(cv::Size(depth_image.width, depth_image.height), CV_32FC1, &depth_image.depth[0], cv::Mat::AUTO_STEP);
        
        int width = depth_image.width;
        int height = depth_image.height;
        float fx = depth_image.focalx;
        float fy = depth_image.focaly;
        float cx = width / 2.0;
        float cy = height / 2.0;

        // -------------------- Tablas precalculadas --------------------
        static std::vector<float> xTable, yTable;
        if (xTable.size() != (size_t)width || yTable.size() != (size_t)height)
        {
            xTable.resize(width);
            yTable.resize(height);

            for (int u = 0; u < width; u++)
                xTable[u] = (u - cx) / fx;

            for (int v = 0; v < height; v++)
                yTable[v] = (v - cy) / fy;
        }

        std::vector<RoboCompCameraRGBDSimple::Point3D> cloud;
        cloud.reserve(width * height);

        // -------------------- Bucle paralelizado --------------------
        #pragma omp parallel
        {
            std::vector<RoboCompCameraRGBDSimple::Point3D> localCloud;
            localCloud.reserve(width * height / omp_get_num_threads());

            #pragma omp for collapse(2)
            for (int v = 0; v < height; v++) {
                for (int u = 0; u < width; u++) {
                    float d = depthMat.at<float>(v, u);
                    if (d <= 0.0f || d == INFINITY) {
                        continue;
                    }

                    float X = xTable[u] * d;
                    float Y = yTable[v] * d;
                    float Z = d;

                    localCloud.push_back({X, Y, Z});
                }
            }

            #pragma omp critical
            cloud.insert(cloud.end(), localCloud.begin(), localCloud.end());
        }

        // -------------------- Salida --------------------
        points.points = std::move(cloud);

        rgbd.image = rgb_image;
        rgbd.depth = depth_image;
        rgbd.points = points;
    }
    if(display){
        cv::Mat rgb_frame, depth_frame;
        if(simulated){
            rgb_frame = cv::Mat(cv::Size(rgbd.image.width, rgbd.image.height), CV_8UC3, &rgbd.image.image[0], cv::Mat::AUTO_STEP);
            cv::cvtColor(rgb_frame, rgb_frame, cv::COLOR_RGBA2BGR);

            depth_frame = cv::Mat(cv::Size(rgbd.depth.width, rgbd.depth.height), CV_32FC1, &rgbd.depth.depth[0], cv::Mat::AUTO_STEP);
            depth_frame.convertTo(depth_frame, CV_8UC3, 255. / 10, 0);
            applyColorMap(depth_frame, depth_frame, cv::COLORMAP_RAINBOW); //COLORMAP_HSV tb    

            cv::imshow("rgb", rgb_frame);
            cv::imshow("depth", depth_frame);
            cv::waitKey(1);
        }
        else{
            // TODO: Imprimir zed real

            // rgb_frame = cv::Mat(cv::Size(rgbd.image.width, rgbd.image.height), CV_8UC4, &rgbd.image.image[0], cv::Mat::AUTO_STEP);

            // depth_frame = cv::Mat(cv::Size(rgbd.depth.width, rgbd.depth.height), CV_32FC4, &rgbd.depth.depth[0], cv::Mat::AUTO_STEP);
            
            // depth_frame.convertTo(depth_frame, CV_8UC3, 255. / 10, 0);
            // applyColorMap(depth_frame, depth_frame, cv::COLORMAP_RAINBOW); //COLORMAP_HSV tb
        }

    }

    camerargbdsimplepub_pubproxy->pushRGBD(rgbd);
}

void SpecificWorker::process_pose_data()
{
    static long last_timestamp = 0;


    // 1) ZED pose
    #ifdef POSE
        zed.getPosition(zed_pose, sl::REFERENCE_FRAME::WORLD);
    #else
        zed.getPosition(zed_pose, sl::REFERENCE_FRAME::CAMERA);
    #endif
    // 2) traslación y orientación
    auto t_cam_sl = zed_pose.getTranslation();
    auto r_cam_sl = zed_pose.getEulerAngles(true);
    Eigen::Vector3f t_cam(t_cam_sl.tx, t_cam_sl.ty, t_cam_sl.tz);
    Eigen::AngleAxisf aa_x(r_cam_sl.x, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf aa_y(r_cam_sl.y, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf aa_z(r_cam_sl.z, Eigen::Vector3f::UnitZ());
    Eigen::Quaternionf q_cam = aa_z * aa_y * aa_x;  // orden ZYX
    std::cout<<std::setprecision(3)<<"\rX:"<<q_cam.x()<<"|Y:"<<q_cam.y()<<"|Z:"<< q_cam.z()<<"|W:"<< q_cam.w();

    // 3) Transformación fija cámara?robot
    static const Eigen::Vector3f cam_to_robot_t(60.0f, 76.6f, 0.0f);
    static const Eigen::Quaternionf cam_to_robot_q = Eigen::Quaternionf::Identity();

    Eigen::Transform<float,3,Eigen::Affine> T_cam_to_robot;
    T_cam_to_robot = Eigen::Translation3f(cam_to_robot_t) * cam_to_robot_q;

    // 4) Pose del robot en el mundo: T_world_robot = T_world_cam * T_cam_to_robot
    Eigen::Transform<float,3,Eigen::Affine> T_world_cam(q_cam);
    T_world_cam.pretranslate(t_cam);
    Eigen::Transform<float,3,Eigen::Affine> T_world_robot = T_world_cam * T_cam_to_robot;
    Eigen::Vector3f t_robot = T_world_robot.translation();
    Eigen::Matrix3f R_robot = T_world_robot.rotation();
    Eigen::Vector3f euler_robot = R_robot.eulerAngles(0, 1, 2);  // roll, pitch, yaw

    // 5) Velocidades (diferencial)
    long timestamp = zed_pose.timestamp.getNanoseconds();
    float dt = (timestamp - last_timestamp) * 1e-9f;  // segundos
    last_timestamp = timestamp;

    auto pose_confidence = zed_pose.pose_confidence;
    float* cov = zed_pose.pose_covariance;
    float* twist = zed_pose.twist;

    // Print all values
    //std::cout << "Pose confidence: " << pose_confidence << std::endl;
    //std::cout << "Pose covariance: ";
    //for (int i = 0; i < 36; ++i) {
    //    std::cout << cov[i] << " ";
    //    if ((i + 1) % 6 == 0) std::cout << std::endl;
    //}
    RoboCompFullPoseEstimation::CovMatrix pose_cov_matrix{
        .m00 = cov[0], .m01 = cov[1], .m02 = cov[2], .m03 = cov[3], .m04 = cov[4], .m05 = cov[5],
        .m10 = cov[6], .m11 = cov[7], .m12 = cov[8], .m13 = cov[9], .m14 = cov[10], .m15 = cov[11],
        .m20 = cov[12], .m21 = cov[13], .m22 = cov[14], .m23 = cov[15], .m24 = cov[16], .m25 = cov[17],
        .m30 = cov[18], .m31 = cov[19], .m32 = cov[20], .m33 = cov[21], .m34 = cov[22], .m35 = cov[23],
        .m40 = cov[24], .m41 = cov[25], .m42 = cov[26], .m43 = cov[27], .m44 = cov[28], .m45 = cov[29],
        .m50 = cov[30], .m51 = cov[31], .m52 = cov[32], .m53 = cov[33], .m54 = cov[34], .m55 = cov[35]
    };

    Eigen::Vector3f v_cam = t_cam / dt;
    Eigen::Vector3f omega_cam(r_cam_sl.x / dt, r_cam_sl.y / dt, r_cam_sl.z / dt);

    // 6) Corrección de velocidad lineal: v_robot = v_cam + ?_cam × (lever arm)
    Eigen::Vector3f lever = q_cam * cam_to_robot_t;
    Eigen::Vector3f v_robot = v_cam + omega_cam.cross(lever);
    Eigen::Vector3f omega_robot = omega_cam;

    // 7) Publicación
    RoboCompFullPoseEstimation::FullPoseEuler pose;
    pose.x  = t_cam_sl.x;
    pose.y  = t_cam_sl.y;
    pose.z  = t_cam_sl.z;
    pose.rx = r_cam_sl.x;
    pose.ry = r_cam_sl.y;
    pose.rz = r_cam_sl.z;
    pose.vx  = v_robot.x();
    pose.vy  = v_robot.y();
    pose.vz  = v_robot.z();
    pose.vrx = omega_robot.x();
    pose.vry = omega_robot.y();
    pose.vrz = omega_robot.z();
    pose.poseCov = pose_cov_matrix;
    pose.timestamp = zed_pose.timestamp.getMilliseconds();

    try
    {
        fullposeestimationpub_pubproxy->newFullPose(pose);
    }
    catch (const Ice::Exception &e)
    {
        std::cerr << "Error publishing pose: " << e.what() << std::endl;
    }

    //std::cout<<std::setprecision(3)<<"\rX:"<<pose.x<<"|Y:"<<pose.y<<"|Z:"<< pose.z<<
        // "| RX:"<< pose.rx << "|RY:"<<pose.ry<< "|RZ:"<< pose.rz<<"| VX:"<<pose.vx<<"|VY:"<<pose.vy<<"|VZ:"<< pose.vz<<
        // "| VRX:"<< pose.vrx << "|VRY:"<<pose.vry<< "|VRZ:"<< pose.vrz<<"| Time:"<<pose.timestamp<<"              ";    

}

void SpecificWorker::sensorsLoop()
{
    const int N = 16;                   // Downsampling factor (publicarás a 100 Hz si N=4)
    int sample_count = 0;
    // Acumuladores para aceleración y giro
    sl::float3 acc_sum{0,0,0}, gyr_sum{0,0,0};
    while (running)
    {
        // Get start time
        auto start = std::chrono::high_resolution_clock::now();
        zed.getSensorsData(sensors_data, sl::TIME_REFERENCE::CURRENT);

        // Solo consideramos IMU para downsampling
        if (ts.isNew(sensors_data.imu))
        {
            // Acumular muestras
            const auto &imu = sensors_data.imu;
            acc_sum.x += imu.linear_acceleration.x;
            acc_sum.y += imu.linear_acceleration.y;
            acc_sum.z += imu.linear_acceleration.z;
            gyr_sum.x += imu.angular_velocity.x;
            gyr_sum.y += imu.angular_velocity.y;
            gyr_sum.z += imu.angular_velocity.z;
            sample_count++;

            if (sample_count >= N)
            {
                // Calcular promedio
                sl::float3 acc_avg{ acc_sum.x / float(N),
                                    acc_sum.y / float(N),
                                    acc_sum.z / float(N) };
                sl::float3 gyr_avg{ gyr_sum.x / float(N),
                                    gyr_sum.y / float(N),
                                    gyr_sum.z / float(N) };

                // Publicar por proxy
                RoboCompIMU::DataImu imu_data;
                RoboCompIMU::Gyroscope gyro_data;
                gyro_data.XGyr = gyr_avg.x;
                gyro_data.YGyr = gyr_avg.y;
                gyro_data.ZGyr = gyr_avg.z;
                gyro_data.timestamp = sensors_data.imu.timestamp.getNanoseconds();
                RoboCompIMU::Acceleration accel_data;
                accel_data.XAcc = acc_avg.x;
                accel_data.YAcc = acc_avg.y;
                accel_data.ZAcc = acc_avg.z;
                accel_data.timestamp = sensors_data.imu.timestamp.getNanoseconds();
                imu_data.acc  = accel_data;
                imu_data.gyro = gyro_data;

                this->imupub_pubproxy->publish(imu_data);

                // Reiniciar acumuladores
                sample_count = 0;
                acc_sum = {0,0,0};
                gyr_sum = {0,0,0};
            }
        }
        // Publicar magnetómetro y barómetro
        if (ts.isNew(sensors_data.magnetometer))
        {
//            std::cout << " - Magnetometer\n\t Magnetic Field: {"
//                      << sensors_data.magnetometer.magnetic_field_calibrated
//                      << "} [uT]\n";
        }
        if (ts.isNew(sensors_data.barometer))
        {
//            std::cout << " - Barometer\n\t Atmospheric pressure: "
//                      << sensors_data.barometer.pressure << " [hPa]\n";
        }

        // Get end time
        auto end = std::chrono::high_resolution_clock::now();
        // Calculate elapsed time
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        // Sleep until the next iteration taking into account a required loop time
//        std::cout << "Elapsed time: " << elapsed << " microseconds" << std::endl;
//        std::cout << "Sleeping for: " << 2500 - elapsed << " microseconds" << std::endl;
        std::this_thread::sleep_for(std::chrono::microseconds(2500 - elapsed));
        auto end_2 = std::chrono::high_resolution_clock::now();
        // Calculate elapsed time
        auto elapsed_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start).count();
//        std::cout << "Elapsed 2 time: " << elapsed_2 << " microseconds" << std::endl;
    }
}

float SpecificWorker::wrapToPi(float angle_rad) {
    angle_rad = std::fmod(angle_rad + M_PI, 2.0f * M_PI);
    if (angle_rad < 0)
        angle_rad += 2.0f * M_PI;
    return angle_rad - M_PI;
}
//**************************************/AUX/**************************************/
void SpecificWorker::transformPose(sl::Transform &pose, sl::Transform transform) {
    // Pose(new reference frame) = Pose (camera frame) * M, where M is the transform between two frames
    pose = pose * transform;
}

/**************************************/
// From the RoboCompCameraRGBDSimple you can call this methods:
// RoboCompCameraRGBDSimple::TRGBD this->camerargbdsimple_proxy->getAll(string camera)
// RoboCompCameraRGBDSimple::TDepth this->camerargbdsimple_proxy->getDepth(string camera)
// RoboCompCameraRGBDSimple::TImage this->camerargbdsimple_proxy->getImage(string camera)
// RoboCompCameraRGBDSimple::TPoints this->camerargbdsimple_proxy->getPoints(string camera)

/**************************************/
// From the RoboCompCameraRGBDSimple you can use this types:
// RoboCompCameraRGBDSimple::Point3D
// RoboCompCameraRGBDSimple::TPoints
// RoboCompCameraRGBDSimple::TImage
// RoboCompCameraRGBDSimple::TDepth
// RoboCompCameraRGBDSimple::TRGBD

/**************************************/
// From the RoboCompCameraRGBDSimplePub you can publish calling this methods:
// RoboCompCameraRGBDSimplePub::void this->camerargbdsimplepub_pubproxy->pushRGBD(RoboCompCameraRGBDSimple::TRGBD rgbd)

/**************************************/
// From the RoboCompFullPoseEstimationPub you can publish calling this methods:
// RoboCompFullPoseEstimationPub::void this->fullposeestimationpub_pubproxy->newFullPose(RoboCompFullPoseEstimation::FullPoseEuler pose)

/**************************************/
// From the RoboCompIMUPub you can publish calling this methods:
// RoboCompIMUPub::void this->imupub_pubproxy->publish(RoboCompIMU::DataImu imu)


