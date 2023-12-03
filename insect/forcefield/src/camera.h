//
// Created by pbustos on 14/11/22.
//

#ifndef FORCEFIELD_CAMERA_H
#define FORCEFIELD_CAMERA_H

#include <Eigen/Dense>
#include <QtCore>
#include <CameraRGBDSimple.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <abstract_graphic_viewer/abstract_graphic_viewer.h>

namespace rc
{
    class Camera
    {
        public:
            Camera() = default;
            void initialize(const std::string &name_, RoboCompCameraRGBDSimple::CameraRGBDSimplePrxPtr proxy_);
            cv::Mat capture_rgb();
            cv::Mat capture_rgbd();
            cv::Mat capture_depth();
            std::vector<std::vector<Eigen::Vector2f>> capture_depth_lines();
            std::vector<std::vector<Eigen::Vector2f>> capture_depth_line(int i);
            float get_depth_focalx() const;
            float get_depth_focaly() const;
            std::vector<std::vector<Eigen::Vector2f>> get_depth_lines_in_robot(float min_height, float max_height, float step_size,
                                                                               const Eigen::Transform<float, 3, Eigen::Affine> &tf);
            Eigen::Vector2f project_point3d(const Eigen::Vector3f &p, const Eigen::Transform<float, 3, Eigen::Affine> &tf);
            std::pair <Eigen::Vector2f, Eigen::Vector2f>
            project_line3d(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2, const Eigen::Transform<float, 3, Eigen::Affine> &tf, cv::Mat frame,
                           const cv::Scalar &color);
            void project_polygon_3d(const std::vector<Eigen::Vector3f> &points,
                                    const Eigen::Transform<float, 3, Eigen::Affine> &tf,
                                    cv::Mat frame, const cv::Scalar &color, const std::string &label);
            void project_walls(const std::vector <Eigen::Vector3f> &points, const Eigen::Transform<float, 3, Eigen::Affine> &tf,
                               cv::Mat frame, const cv::Scalar &color);
            void project_floor(const std::vector <Eigen::Vector3f> &points, const Eigen::Transform<float, 3, Eigen::Affine> &tf, cv::Mat frame, const cv::Scalar &color);

    private:
            std::string name;
            RoboCompCameraRGBDSimple::CameraRGBDSimplePrxPtr proxy;

            int rgb_width;
            int rgb_height;
            int rgb_depth;
            int rgb_cameraID;
            float rgb_focalx;
            float rgb_focaly;
            bool rgb_compressed;
            int depth_width;
            int depth_height;
            int depth_depth;
            int depth_cameraID;
            float depth_focalx;
            float depth_focaly;
            bool depth_compressed;

            int num_angular_bins =  360;
            float max_camera_depth_range = 5000; // mm
            float min_camera_depth_range = 300; // mm

    };

} // rc

#endif //FORCEFIELD_CAMERA_H


/// draw depth image code
//cv::Mat nor; cv::normalize(top_depth_frame, nor, 0, 255, cv::NORM_MINMAX);
//nor.convertTo(nor, CV_8U); cv::imshow("depth", nor); cv::waitKey(5);