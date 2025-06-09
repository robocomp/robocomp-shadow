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
    // std::cout << "initialize worker" << std::endl;
    init_parameters.camera_resolution = sl::RESOLUTION::HD720; // Use HD720 opr HD1200 video mode, depending on camera type.
    init_parameters.camera_fps = 30; // Set fps at 30
    init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL_PLUS; // Use ULTRA depth mode
    init_parameters.coordinate_units = sl::UNIT::MILLIMETER; // Use millimeter units (for depth measurements)
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP; // Use a right-handed Y-up coordinate system
    init_parameters.depth_minimum_distance = 650 ;
    //
    // // Open the camera
    // returned_state = zed.open(init_parameters);
    // if (returned_state != sl::ERROR_CODE::SUCCESS) {
    //     std::cout << "Error " << returned_state << ", exit program." << std::endl;
    //     return;
    // }
    //
    // // Enable positional tracking with default parameters
    tracking_parameters.enable_imu_fusion = true;
    tracking_parameters.enable_pose_smoothing = true;

    tracking_parameters.mode = sl::POSITIONAL_TRACKING_MODE::GEN_3;
    // tracking_parameters.enable_area_memory = false;
    //tracking_parameters.enable_light_computation_mode = true;
    //
    //
    // auto err = zed.enablePositionalTracking(tracking_parameters);
    // if (err != sl::ERROR_CODE::SUCCESS)
    //     exit(-1);


    // INIT PARAMETERS
    //init_parameters.camera_resolution = sl::RESOLUTION::HD720; // Máxima resolución compatible
    //init_parameters.camera_fps = 60; // 30 FPS balancea precisión y estabilidad
    //init_parameters.depth_mode = sl::DEPTH_MODE::PERFORMANCE; // Máxima calidad de profundidad
    //init_parameters.coordinate_units = sl::UNIT::MILLIMETER; // Para mayor precisión de pose
    //init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP; // Consistente con robótica

    //init_parameters.depth_minimum_distance = 0.6; // Reducido para mejorar seguimiento cercano
    //init_parameters.enable_image_enhancement = true; // Mejora visual en condiciones complejas
    //init_parameters.async_grab_camera_recovery = false; // Evita bloqueos si hay pérdida temporal de conexión
    //init_parameters.camera_disable_self_calib = false; // Habilitar autocalibración mejora tracking en largo plazo

    // Abrir cámara
    returned_state = zed.open(init_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Error " << returned_state << ", exit program." << std::endl;
        exit(0);
    }



    // TRACKING PARAMETERS
    //tracking_parameters.enable_imu_fusion = true; // Combina visual e IMU
    //tracking_parameters.enable_pose_smoothing = true; // Estabiliza la pose (algo más lento)
    //tracking_parameters.set_as_static = false; // Asegura que use movimiento real
    //tracking_parameters.mode = sl::POSITIONAL_TRACKING_MODE::GEN_2; // Último modo de seguimiento (mejor IMU)

    auto err = zed.enablePositionalTracking(tracking_parameters);
    if (err != sl::ERROR_CODE::SUCCESS)
        exit(-1);



    // Get the distance between the center of the camera and the left eye
    translation_left_to_center = zed.getCameraInformation().camera_configuration.calibration_parameters.stereo_transform;
    // Lanzar hilo
    running = true;

    //IMU, barometer and magnetometer
    // sensor_thread = std::thread(&SpecificWorker::sensorsLoop, this);

}

void SpecificWorker::compute()
{
    auto start = std::chrono::high_resolution_clock::now();
    returned_state = zed.grab();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // std::cout << "Elapsed grab: " << elapsed << " microseconds" << std::endl;
    // A new image is available if grab() returns ERROR_CODE::SUCCESS
    if (returned_state == sl::ERROR_CODE::SUCCESS)
    {
        // process_RGBD_data();
        process_pose_data();
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
    camerargbdsimplepub_pubproxy->pushRGBD(rgb_image, depth_image);
}

void SpecificWorker::process_pose_data()
{
    static long last_timestamp = 0;


    // 1) ZED pose
    zed.getPosition(zed_pose, sl::REFERENCE_FRAME::CAMERA);
    // 2) traslación y orientación
    auto t_cam_sl = zed_pose.getTranslation();
    auto r_cam_sl = zed_pose.getEulerAngles(true);
    Eigen::Vector3f t_cam(t_cam_sl.tx, t_cam_sl.ty, t_cam_sl.tz);
    Eigen::AngleAxisf aa_x(r_cam_sl.x, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf aa_y(r_cam_sl.y, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf aa_z(r_cam_sl.z, Eigen::Vector3f::UnitZ());
    Eigen::Quaternionf q_cam = aa_z * aa_y * aa_x;  // orden ZYX

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
    pose.x  = t_robot.x();
    pose.y  = t_robot.y();
    pose.z  = t_robot.z();
    pose.rx = euler_robot.x();
    pose.ry = euler_robot.y();
    pose.rz = euler_robot.z();
    pose.vx  = v_robot.x();
    pose.vy  = v_robot.y();
    pose.vz  = v_robot.z();
    pose.vrx = omega_robot.x();
    pose.vry = omega_robot.y();
    pose.vrz = omega_robot.z();
    pose.pose_cov = pose_cov_matrix;
    pose.timestamp = zed_pose.timestamp.getMilliseconds();

    //Try except publish
    try
    {
        fullposeestimationpub_pubproxy->newFullPose(pose);
    }
    catch (const Ice::Exception &e)
    {
        std::cerr << "Error publishing pose: " << e.what() << std::endl;
    }

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
// From the RoboCompCameraRGBDSimplePub you can publish calling this methods:
// RoboCompCameraRGBDSimplePub::void this->camerargbdsimplepub_pubproxy->pushRGBD(RoboCompCameraRGBDSimple::TImage im, RoboCompCameraRGBDSimple::TDepth dep)

/**************************************/
// From the RoboCompFullPoseEstimationPub you can publish calling this methods:
// RoboCompFullPoseEstimationPub::void this->fullposeestimationpub_pubproxy->newFullPose(RoboCompFullPoseEstimation::FullPoseEuler pose)

/**************************************/
// From the RoboCompIMUPub you can publish calling this methods:
// RoboCompIMUPub::void this->imupub_pubproxy->publish(RoboCompIMU::DataImu imu)


