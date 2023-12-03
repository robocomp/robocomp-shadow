//
// Created by pbustos on 14/11/22.
//

#include "camera.h"
#include <cppitertools/range.hpp>
#include <cppitertools/enumerate.hpp>
#include <cppitertools/sliding_window.hpp>
namespace rc
{
    void Camera::initialize(const std::string &name_, RoboCompCameraRGBDSimple::CameraRGBDSimplePrxPtr proxy_)
    {
        std::cout << __FUNCTION__ << "Testing camera connection..." << std::endl;
        name = name_;
        proxy = proxy_;
        RoboCompCameraRGBDSimple::TImage rgb;
        RoboCompCameraRGBDSimple::TDepth depth;
        RoboCompCameraRGBDSimple::TRGBD rgbd;
        try
        {
            rgb = proxy->getImage(name);
            if (not rgb.image.empty() and rgb.image.size() == (size_t) (rgb.width * rgb.height * rgb.depth))
            {
                rgb_width = rgb.width;
                rgb_height = rgb.height;
                rgb_depth = rgb.depth;
                rgb_cameraID = rgb.cameraID;
                rgb_focalx = rgb.focalx;
                rgb_focaly = rgb.focaly;
                rgb_compressed = rgb.compressed;
            } else
            {
                std::cout << __FUNCTION__ << "Warning: Image read is empty -" << rgb.image.size() << "- or dimensions don't agree: rows = "
                          << rgb.height << " cols = " << rgb.width << "  depth = " << rgb.depth << std::endl;
                throw std::runtime_error(std::string("Warning: Image read is empty -" + std::to_string(rgb.image.size())
                                                     + "- or dimensions don't agree: rows = "
                                                     + std::to_string(rgb.height) + " cols = " + std::to_string(rgb.width)
                                                     + "  depth = " + std::to_string(rgb.depth)));
            }
        }
        catch (const Ice::Exception &e)
        {
            std::cout << e.what() << std::endl;
            throw std::runtime_error("Warning: Error reading camerargbdsimple_proxy::getImage");
        }

        try
        {
            depth = proxy->getDepth(name);
            if (not depth.depth.empty() and depth.depth.size() == depth.width * depth.height * sizeof(float))
            {
                depth_width = depth.width;
                depth_height = depth.height;
                depth_cameraID = depth.cameraID;
                depth_focalx = depth.focalx;
                depth_focaly = depth.focaly;
                depth_compressed = depth.compressed;
            } else
                throw std::runtime_error(std::string("Warning: Depth read is empty -" + std::to_string(rgb.image.size())
                                                     + "- or dimensions don't agree: rows = "
                                                     + std::to_string(rgb.height) + " cols = " + std::to_string(rgb.width)));
        }
        catch (const Ice::Exception &e)
        {
            std::cout << e.what() << std::endl;
            throw std::runtime_error("Warning: Error reading camerargbdsimple_proxy::depthImage");
        }

        std::cout << __FUNCTION__ << "Camera " << name << " tested and operational" << std::endl;
    };

    cv::Mat Camera::capture_rgb()
    {
        RoboCompCameraRGBDSimple::TImage rgb;
        cv::Mat rgb_frame;
        try
        {
            rgb = proxy->getImage(name);
            if (not rgb.image.empty())
                rgb_frame = cv::Mat(cv::Size(rgb.width, rgb.height), CV_8UC3, &rgb.image[0], cv::Mat::AUTO_STEP);
        }
        catch (const Ice::Exception &e)
        { std::cout << e.what() << " Error reading camerargbdsimple_proxy::getImage" << std::endl; }
        return rgb_frame;  // MIRAR SI SE PUEDE QUITAR
    }

    cv::Mat Camera::capture_depth()
    {
        RoboCompCameraRGBDSimple::TDepth depth;
        cv::Mat depth_frame;
        try
        {
            depth = proxy->getDepth(name);
            if (not depth.depth.empty())
                depth_frame = cv::Mat(cv::Size(depth.width, depth.height), CV_32FC1, &depth.depth[0], cv::Mat::AUTO_STEP);

        }
        catch (const Ice::Exception &e)
        { std::cout << e.what() << " Error reading camerargbdsimple_proxy::getDepth" << std::endl; }
        return depth_frame;
    }

    float Camera::get_depth_focalx() const
    {
        return depth_focalx;
    }

    float Camera::get_depth_focaly() const
    {
        return depth_focaly;
    }

    std::vector<std::vector<Eigen::Vector2f>> Camera::get_depth_lines_in_robot(float min_height, float max_height, float step_size,
                                                                               const Eigen::Transform<float, 3, Eigen::Affine> &tf)
    {
        //const float min_height = 0, max_height = 1600, step_size = 50;
        cv::Mat depth_frame = this->capture_depth();
        std::vector<std::vector<Eigen::Vector2f>> points(int((max_height - min_height) / step_size));  //height steps
        for (auto &p: points)
            p.resize(num_angular_bins, Eigen::Vector2f(max_camera_depth_range * 2, max_camera_depth_range * 2));   // angular resolution

        float dist, x, y, z;
        const float ang_bin = 2.f * M_PI / num_angular_bins;
        for (int u = 0; u < depth_frame.rows; u++)
            for (int v = 0; v < depth_frame.cols; v++)
            {
                dist = depth_frame.ptr<float>(u)[v] * 1000.f;  //  -> to mm
                if (std::isnan(dist))
                { qWarning() << " Distance value un depth frame coors " << u << v << "is nan:"; };
                if (dist > max_camera_depth_range or dist < min_camera_depth_range) continue;
                // compute axis coordinates according to the camera's coordinate system (Y outwards and Z up). Dist 0 Y
                y = dist;
                x = (v - (depth_frame.cols / 2.f)) * y / depth_focalx;
                z = -(u - (depth_frame.rows / 2.f)) * y / depth_focaly;

                // convert to robot's CS
                Eigen::Vector3f robot_p = tf * Eigen::Vector3f(x, y, z);

                // add only to its bin if less than current value
                for (auto &&[level, step]: iter::range(min_height, max_height, step_size) | iter::enumerate)
                    if (robot_p.z() > step and robot_p.z() < step + step_size and level < points.size())
                    {
                        int ang_index = floor((M_PI + atan2(robot_p.x(), robot_p.y())) / ang_bin);  //all positive starting at zero
                        Eigen::Vector2f &p = points[level][ang_index];
                        if (robot_p.head(2).norm() < p.norm() /*and new_point.norm() > 400*/)
                            p = robot_p.head(2);
                    }
            };
        // remove all bins not hit by a points
        for (auto &level: points)
            level.erase(std::remove_if(level.begin(), level.end(), [d = max_camera_depth_range](auto p) { return p.x() == d * 2 and p.y() == d * 2; }),
                        level.end());
        return points;
    }

    Eigen::Vector2f Camera::project_point3d(const Eigen::Vector3f &p,
                                            const Eigen::Transform<float, 3, Eigen::Affine> &tf)
    {
        Eigen::Vector3f tp = tf * p;
        if (qFuzzyIsNull(p.y()))
        {
            //qInfo() << __FUNCTION__ << "Exiting with NULL" << tp.y();  // TODO:: fix this
            return Eigen::Vector2f(0.f, 0.f);
        }
        return Eigen::Vector2f{ rgb_focaly * tp.x() / tp.y() + rgb_width / 2,
                               -rgb_focalx * tp.z() / tp.y() + rgb_height / 2};  // Y grows dowwards in the image plane
    }

    std::pair<Eigen::Vector2f, Eigen::Vector2f> Camera::project_line3d(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2,
                                                                       const Eigen::Transform<float, 3, Eigen::Affine> &tf,
                                                                       cv::Mat frame, const cv::Scalar &color)
    {
        //qInfo() << __FUNCTION__ << "Points:" << p1.x() << p1.y() << p1.z() << p2.x() << p2.y() << p2.z();
        if (p1.y() < 0 and p2.y() < 0)
        {
            //qInfo() << __FUNCTION__ << "Both with y-. REJECTED" << p1.x() << p1.y() << p2.x() << p2.y();
            return std::make_pair(Eigen::Vector2f(0.f,0.f), Eigen::Vector2f(0.f,0.f));
        }
        if (p1.y() >= 0 and p2.y() >= 0)
        {
            auto ph2d1 = project_point3d(p1, tf);
            auto ph2d2 = project_point3d(p2, tf);
            if (not frame.empty())
                cv::line(frame, cv::Point(ph2d1.x(), ph2d1.y()), cv::Point(ph2d2.x(), ph2d2.y()), color, 10);
            //qInfo() << __FUNCTION__ << "Both ACCEPTED y pos:" << ph2d1.x() << ph2d1.y() << ph2d2.x() << ph2d2.y();
            return std::make_pair(ph2d1, ph2d2);
        }

        // transform to camera coordinate system
        Eigen::Vector3f pp1 = tf * p1;
        Eigen::Vector3f pp2 = tf * p2;
        Eigen::Vector3f p;
        Eigen::Vector3f v;

        // line p' = p + n.v
        if (pp1.y() >= 0)
        {
            p = pp1;
            v = (pp2-pp1).normalized();
        }
        else
        {
            p = pp2;
            v = (pp1-pp2).normalized();
        };

        if (qFuzzyIsNull(p.y()) or qFuzzyIsNull(v.y()))
        {
            qInfo() << __FUNCTION__ << "Exiting with NULL" << p.x() << p.y() << p.z() << v.x() << v.y() << v.z();
            return std::make_pair(Eigen::Vector2f(), Eigen::Vector2f());
        }
        Eigen::Vector2f ph2d{ rgb_focaly * p.x() / p.y() + rgb_width / 2,           // TODO: Change focals in pyrep
                              -rgb_focalx * p.z() / p.y() + rgb_height / 2};

        Eigen::Vector2f v2d{ rgb_focaly * v.x() / v.y() + rgb_width / 2,
                            -rgb_focalx * v.z() / v.y() + rgb_height / 2};
        v2d -= ph2d;    // Pilar results

        //qInfo() << __FUNCTION__ << "2D proj" << ph2d.x() << ph2d.y() <<  v2d.x() << v2d.y();
        // compute intersections with image borders
        std::vector<float> inters;
        if(auto r = -ph2d.y() / v2d.y(); r>0)
            inters.emplace_back(r);                 // for y=0  ph2d.y + k*v2d.y = 0
        if(auto r = (rgb_height - ph2d.y()) / v2d.y(); r>0)
            inters.emplace_back(r);                 // for y=height ph2d.y + k*v2d.y = height
        if(auto r = -ph2d.x() / v2d.x(); r>0)
                inters.emplace_back(r);             // for x=0 ph2d.x + k*v2d.x = 0
        if(auto r = (rgb_width - ph2d.x()) / v2d.x(); r>0)
                inters.emplace_back(r);             // for x= width ph2d.x + k*v2d.x = width
        std::ranges::sort(inters);

        Eigen::Vector2f initial{0.f, 0.f}, final{0.f, 0.f};
        if(ph2d.x()<0 or ph2d.x()>rgb_width or ph2d.y()<0 or ph2d.y()>rgb_height)     // out of image plane
        {
            //qInfo() << __FUNCTION__ << "p outside " << ph2d.x() << ph2d.y();
            initial = ph2d - inters[0]*v2d;
            final = ph2d - inters[1]*v2d;
        }
        else
        {
            //qInfo() << __FUNCTION__ << "p inside " << ph2d.x() << ph2d.y();
            initial = ph2d;
            final = ph2d - inters[0] * v2d;
        }

        //qInfo() << __FUNCTION__ << "Line:" << initial.x() << initial.y() << final.x() << final.y();
        if (not frame.empty())
            cv::line(frame, cv::Point(initial.x(), initial.y()), cv::Point(final.x(), final.y()), color, 10);
        //qInfo() << __FUNCTION__ << " BEFORE " << initial.x() << initial.y() << final.x() << final.y();
        return std::make_pair(initial, final);
    }

    void Camera::project_polygon_3d(const std::vector<Eigen::Vector3f> &points,
                                    const Eigen::Transform<float, 3, Eigen::Affine> &tf,
                                    cv::Mat frame, const cv::Scalar &color, const std::string &label)
    {
        auto cpoints(points);
        cpoints.push_back(points.front());
        for(auto &&pts: cpoints | iter::sliding_window(2))
            project_line3d(pts[0], pts[1], tf, frame, color);

    }

    void Camera::project_walls(const std::vector<Eigen::Vector3f> &points,
                               const Eigen::Transform<float, 3, Eigen::Affine> &tf,
                               cv::Mat frame, const cv::Scalar &color)
    {
        std::vector<Eigen::Vector3f> cpoints(points);
        cpoints.push_back(points.front());
        for(const auto &p: points | iter::sliding_window(2))
            project_floor({p[0], p[1], Eigen::Vector3f{p[1].x(), p[1].y(), 1000.f}, Eigen::Vector3f{p[0].x(), p[0].y(), 1000.f}}, tf, frame, color);
    }

    void Camera::project_floor(const std::vector<Eigen::Vector3f> &points,
                               const Eigen::Transform<float, 3, Eigen::Affine> &tf,
                               cv::Mat frame, const cv::Scalar &color)
    {
        std::vector<cv::Point> cvpts;
        std::vector<Eigen::Vector3f> cpoints(points);
        cpoints.push_back(points.front());
        for(auto &&pts: cpoints | iter::sliding_window(2))
        {
            const auto [p1, p2] = project_line3d(pts[0], pts[1], tf, cv::Mat(), color);
            if(p1.isZero() and p2.isZero()) continue;
            auto pp1 = p1.cast<int>(); auto pp2 = p2.cast<int>();
            cvpts.emplace_back(cv::Point{pp1.x(), pp1.y()});
            cvpts.emplace_back(cv::Point{pp2.x(), pp2.y()});
        }
        if(not cvpts.empty())
        {
            cv::Mat overlay(frame.clone());
            fillConvexPoly(frame, cvpts, cv::Scalar(200, 255, 200));
            cv::addWeighted(overlay, 0.6, frame, 1 - 0.6, 0, frame);
        }
    }
}
// rc