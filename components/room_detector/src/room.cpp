//
// Created by pbustos on 9/10/22.
//

#include "room.h"
#include "pch.h"

namespace rc
{
    /**
     * @brief Initializes the room with given corners and robot pose.
     *
     * This function calculates the center, width, height, and orientation of the room based on the provided corners.
     * It also computes the transformation from the room frame to the robot frame.
     * It checks that the computed transform matches nominal and measured corners and detects the 90º symmetry.
     * @param corners A vector of QPointF representing the corners of the room in Origin frame
     * @param robot_pose The pose of the robot as an Eigen::Affine2d transformation in Origin frame
     * @return A pair containing:
     *         - cv::RotatedRect: The nominal room rectangle.  *
     *         - Eigen::Affine2d: The transformation from the room frame to the robot frame.
     */
    std::pair<cv::RotatedRect, Eigen::Affine2d> Room::initialize(const std::vector<QPointF>& corners,
                                                                 const Eigen::Affine2d& robot_pose)
    {
        auto toE = [](const QPointF &p) { return Eigen::Vector2d(p.x(), p.y()); };
        auto qnorm = [](const QPointF &p) { return std::hypot(p.x(), p.y()); };

        // 1. Calculate the center point
        const auto center = std::accumulate(corners.begin(), corners.end(), QPointF{0.f, 0.f}) / static_cast<double>(corners.size());
        //qDebug() << __FUNCTION__ << "Room Center in global" << center.x() << center.y();

        // 2. Calculate the sides of the rectangle
        const auto width = qnorm(corners[1] - corners[0]);  // Assuming corners are in order
        const auto height = qnorm(corners[2] - corners[1]); // Assuming corners are in order

        // 3. Calculate the angle of each side with the Y-axis of the robot and pick the smallest one
        const QLineF line1(corners[0], corners[1]);
        const QLineF line2(corners[1], corners[2]);
        const QLineF yxis(0.f, 0.f, 0.f, 1000.f);

        // Keep values between -PI and PI
        const double ang1 = keep_angle_between_minus_pi_and_pi(qDegreesToRadians(static_cast<double>(yxis.angleTo(line1))));
        const double ang2 = keep_angle_between_minus_pi_and_pi(qDegreesToRadians(static_cast<double>(yxis.angleTo(line2))));

        //qDebug() << __FUNCTION__ << "Angle to line1" << ang1;
        //qDebug() << __FUNCTION__ << "Angle to line2" << ang2;

        // Select the line with the closest angle to 0
        const auto min_angle = std::abs(ang1) < std::abs(ang2) ? ang1 : ang2;
        //qDebug() << __FUNCTION__ << "Min angle" << min_angle;

        // compute room frame wrt to origin
        const Eigen::Affine2d room_frame = Eigen::Translation2d(toE(center)) * Eigen::Rotation2Dd(-min_angle);

        // Print robot pose
        Eigen::Rotation2Df rot(room_frame.rotation());
        //qDebug() << __FUNCTION__ << "Room respect to origin" << room_frame.translation().x() << room_frame.translation().y() << rot.angle();


        // 7. Construct the nominal room at 0,0 and no rotation
        rect = cv::RotatedRect(cv::Point2d(0.0, 0.0), cv::Size2d(width, height), 0.f);

        // 8. Check that the computed transform matches nominal and measured corners and detect the 90º symmetry
        // A corner in origin frame (accumulated polygon centers) must match a corner in the room frame
        // transform original corners to projective adding one row = 1.f
        std::vector<Eigen::Vector3d> original_corners_3d;
        std::ranges::transform(corners, std::back_inserter(original_corners_3d), [](const QPointF &p)
        { return Eigen::Vector3d(p.x(), p.y(), 1.f); });

        // nominal corners (in room frame) have to be transformed to the origin frame
        std::vector<Eigen::Vector3d> nominal_corners_in_room_frame_1;
        std::ranges::transform(get_corners_3d(), std::back_inserter(nominal_corners_in_room_frame_1), [room_frame](const auto &p)
        { return room_frame * Eigen::Vector3d(p.x(), p.y(), 1.f); });

        // match
        const auto affine = Eigen::Affine2d::Identity();
        const auto matches_1 = rc::hungarian(original_corners_3d, nominal_corners_in_room_frame_1, affine);
        const auto suma_1 = std::accumulate(matches_1.begin(), matches_1.end(), 0.0f, [](double acc, const auto &m)
                { return acc + std::get<4>(m);});

        // rotate the nominal room 90º and compute the matching error
        rect = cv::RotatedRect(cv::Point2d(0.0, 0.0), cv::Size2d(height, width), 0.f);

        // nominal corners (in room frame) have to be transformed to the origin frame
        std::vector<Eigen::Vector3d> nominal_corners_in_room_frame_2;
        std::ranges::transform(get_corners_3d(), std::back_inserter(nominal_corners_in_room_frame_2), [room_frame](const auto &p)
        { return room_frame * Eigen::Vector3d(p.x(), p.y(), 1.f); });

        // match
        const auto matches_2 = rc::hungarian(original_corners_3d, nominal_corners_in_room_frame_2, affine);
        const auto suma_2 = std::accumulate(matches_2.begin(), matches_2.end(), 0.0f, [](double acc, const auto &m)
                { return acc + std::get<4>(m);});

        // check and select the best orientation
        if (suma_1 < suma_2)
            rect = cv::RotatedRect(cv::Point2d(0.0, 0.0), cv::Size2d(height, width), 90.f);
        //else
            //qDebug() << __FUNCTION__ << "Room has been ROTATED";

        // 9. Transform the room to the robot frame
        Eigen::Affine2d room_to_robot = room_frame.inverse() * robot_pose;
        set_valid(true);

        return std::make_pair(rect, room_to_robot);
    }
    double Room::keep_angle_between_minus_pi_and_pi(double angle)
    {
        while (angle > M_PI)
            angle -= 2 * M_PI;
        while (angle < -M_PI)
            angle += 2 * M_PI;
        return angle;
    }
    double Room::size_dist(const QSizeF &p1, const QSizeF &p2) const
    {
        return sqrt((p1.width() - p2.width()) * (p1.width() - p2.width()) + (p1.height() - p2.height()) * (p1.height() - p2.height()));
    }
    // std::vector<Eigen::Vector3d> Room::get_3d_corners_in_robot_coor()
    // {
    //     cv::Point2f pts[4];
    //     rect.points(pts);
    //     std::vector<Eigen::Vector3d> rcorners{Eigen::Vector3d(pts[0].x, pts[0].y, 0.f),
    //                                           Eigen::Vector3d(pts[1].x, pts[1].y, 0.f),
    //                                           Eigen::Vector3d(pts[2].x, pts[2].y, 0.f),
    //                                           Eigen::Vector3d(pts[3].x, pts[3].y, 0.f)};
    //     return rcorners;
    // }
    std::vector<Eigen::Vector3d> Room::get_corners_3d() const
    {
        cv::Point2f pts[4];
        rect.points(pts);
        std::vector<Eigen::Vector3d> rcorners{Eigen::Vector3d(pts[0].x, pts[0].y, 1.f),
                                              Eigen::Vector3d(pts[1].x, pts[1].y, 1.f),
                                              Eigen::Vector3d(pts[2].x, pts[2].y, 1.f),
                                              Eigen::Vector3d(pts[3].x, pts[3].y, 1.f)};
        return rcorners;
    }
    Eigen::Matrix<double, 4, 2> Room::get_corners_mat() const
    {
        cv::Point2f pts[4];
        rect.points(pts);
        Eigen::Matrix<double, 4, 2> rcorners;
        rcorners.row(0) = Eigen::Vector2d(pts[0].x, pts[0].y);
        rcorners.row(1) = Eigen::Vector2d(pts[1].x, pts[1].y);
        rcorners.row(2) = Eigen::Vector2d(pts[2].x, pts[2].y);
        rcorners.row(3) = Eigen::Vector2d(pts[3].x, pts[3].y);
        return rcorners;
    }
    std::vector<Eigen::Vector2d> Room::get_corners() const
    {
        cv::Point2f pts[4];
        rect.points(pts);
        std::vector<Eigen::Vector2d> corners(4);
        for (auto &&i: iter::range(4))
            corners[i] = Eigen::Vector2d(pts[i].x, pts[i].y);
        return corners;
    }
    double Room::get_minX() const
    {
        cv::Point2f corners[4];
        rect.points(corners);
        auto min_x_corner = std::ranges::min_element(corners, [](const cv::Point2d &a, const cv::Point2d &b)
                                            { return a.x < b.x; });
        return min_x_corner->x;
    }
    double Room::get_minY() const
    {
        cv::Point2f corners[4];
        rect.points(corners);
        auto min_x_corner = std::ranges::min_element(corners, [](const cv::Point2d &a, const cv::Point2d &b)
        { return a.y < b.y; });
        return min_x_corner->y;
    }
    double Room::get_maxX() const
    {
        cv::Point2f corners[4];
        rect.points(corners);
        auto max_x_corner = std::ranges::min_element(corners, [](const cv::Point2d &a, const cv::Point2d &b)
        { return a.x > b.x; });
        return max_x_corner->x;
    }
    double Room::get_maxY() const
    {
        cv::Point2f corners[4];
        rect.points(corners);
        auto max_x_corner = std::ranges::min_element(corners, [](const cv::Point2d &a, const cv::Point2d &b)
        { return a.y > b.y; });
        return max_x_corner->y;
    }
    Eigen::Vector2d Room::get_closest_corner_in_robot_coor2(const Eigen::Vector2d &c) const
    {
        cv::Point2f pts[4];
        rect.points(pts);
        std::vector<Eigen::Vector2d> cs;
        for (auto &&i: iter::range(4))
            cs.emplace_back(Eigen::Vector2d(pts[i].x, pts[i].y));
        return *std::ranges::min_element(cs, [c](auto &a, auto &b) { return (a - c).norm() < (b - c).norm(); });
    }
    Eigen::Vector2d Room::get_closest_corner_in_robot_coor(const Eigen::Vector2d &c)
    {
        auto corners = get_corners_mat();
        Eigen::Index index;
        (corners.rowwise() - c.transpose()).rowwise().squaredNorm().minCoeff(&index);
        return corners.row(index);
    }
    Eigen::Vector2d Room::to_room_coor(const Eigen::Vector2d &p) const
    {
        const Eigen::Rotation2Dd rt_from_robot(rot);
        //Eigen::Matrix<double, 2, 2> rt_from_robot;
        //rt_from_robot << cos(rot), -sin(rot) , sin(rot), cos(rot);
        Eigen::Vector2d res = rt_from_robot.inverse() * (p - center);
        return res;
    }
    Eigen::Vector2d Room::get_closest_corner(const Eigen::Vector2d &c)
    {
        // c should be in rooms's coordinates
        auto corners = get_corners_mat();
        Eigen::Index index;
        (corners.rowwise() - c.transpose()).rowwise().squaredNorm().minCoeff(&index);
        return corners.row(index);
    }
    std::pair<double, QLineF> Room::get_closest_side(const QLineF &l)
    {
        //auto sides = get_sides_in_robot_coor();
        std::vector<std::pair<double, QLineF>> distances;
        cv::Point2f pts[4];
        rect.points(pts);
        double signed_distance;
        for (auto &&p: iter::range(4) | iter::sliding_window(2))
        {
            QLineF line(pts[p[0]].x, pts[p[0]].y, pts[p[1]].x, pts[p[1]].y);
            // if tested line is further than model line, distance is positive
            if (euc_distance_between_points(rect.center, line.center()) < euc_distance_between_points(rect.center, l.center()))
                signed_distance = euc_distance_between_points(line.center(), l.center());
            else
                signed_distance = -euc_distance_between_points(line.center(), l.center());
            distances.emplace_back(std::make_pair(signed_distance, l));
        }
        // last with first
        QLineF line(pts[3].x, pts[3].y, pts[0].x, pts[0].y);
        if (euc_distance_between_points(rect.center, line.center()) < euc_distance_between_points(rect.center, l.center()))
            signed_distance = euc_distance_between_points(line.center(), l.center());
        else
            signed_distance = -euc_distance_between_points(line.center(), l.center());
        distances.emplace_back(std::make_pair(euc_distance_between_points(line.center(), l.center()), l));

        return *std::ranges::min_element(distances, [](auto &a, auto &b) { return fabs(std::get<double>(a)) < fabs(std::get<double>(b)); });
    }
    void Room::print()
    {
        if(is_initialized)
        {
            std::cout << "Room: " << std::endl;
            std::cout << "  center: " << rect.center << std::endl;
            std::cout << "  size: " << rect.size << std::endl;
            std::cout << "  rot: " << rect.angle << "º, " << qDegreesToRadians(rect.angle) << " rads" << std::endl;
            cv::Point2f vertices[4]; rect.points(vertices);
            for(const auto &[i, c]: vertices | iter::enumerate)
                std::cout << "  corner " << i << ": [" << c.x << ", " << c.y << "]" << std::endl;
            std::cout << std::endl;
        }
        else
            std::cout << __FUNCTION__ <<  "PRINT: Room not initialized" << std::endl;
    }
    Eigen::Vector2d Room::to_local_coor(const Eigen::Vector2d &p)
    {
        //Eigen::Rotation2Df rt_from_robot(rot);
        Eigen::Matrix<double, 2, 2> rt_from_robot;
        rt_from_robot << cos(rot), -sin(rot), sin(rot), cos(rot);
        return rt_from_robot.transpose() * (p - center);
    }
    double Room::euc_distance_between_points(const QPointF &p1, const QPointF &p2) const
    {
        return sqrt((p1.x() - p2.x()) * (p1.x() - p2.x()) + (p1.y() - p2.y()) * (p1.y() - p2.y()));
    }
    double Room::euc_distance_between_points(const cv::Point2d &p1, const QPointF &p2) const
    {
        return sqrt((p1.x - p2.x()) * (p1.x - p2.x()) + (p1.y - p2.y()) * (p1.y - p2.y()));
    }
    QPointF Room::to_qpoint(const cv::Point2d &p) const
    {
        return QPointF{p.x, p.y};
    }
    void Room::rotate(double delta)
    {
        delta = std::clamp(delta, -5.0, 5.0);
        rot += delta;
        //qInfo() << __FUNCTION__ << delta << rot;
    }
    void Room::draw_2D(const QString &color, QGraphicsScene *scene) const
    {
        static std::vector<QGraphicsItem *> items;
        for (const auto i: items)
        { scene->removeItem(i); delete i;}
        items.clear();

        QColor col("green");
        col.setAlpha(30);
        const auto size = rect.size;
        const auto item = scene->addRect(-size.height / 2, -size.width / 2, size.height, size.width, QPen(QColor("orange"), 60), QBrush(QColor(col)));
        item->setPos(rect.center.x, rect.center.y);
        item->setRotation(rect.angle -90 ); // -90
        items.push_back(item);

    }
    std::vector<QLineF> Room::get_room_lines_qt() const
    {
        std::vector<QLineF> lines;
        cv::Point2f pts[4];
        rect.points(pts);
        lines.emplace_back(QLineF(pts[0].x, pts[0].y, pts[1].x, pts[1].y));
        lines.emplace_back(QLineF(pts[1].x, pts[1].y, pts[2].x, pts[2].y));
        lines.emplace_back(QLineF(pts[2].x, pts[2].y, pts[3].x, pts[3].y));
        lines.emplace_back(QLineF(pts[3].x, pts[3].y, pts[0].x, pts[0].y));
        return lines;
    }
    std::vector<Eigen::ParametrizedLine<double, 2>> Room::get_room_lines_eigen() const
    {
        std::vector<Eigen::ParametrizedLine<double, 2>> lines;
        cv::Point2f pts[4];
        rect.points(pts);
        lines.emplace_back(Eigen::ParametrizedLine<double, 2>::Through(Eigen::Vector2d {pts[0].x, pts[0].y}, Eigen::Vector2d{pts[1].x, pts[1].y}));
        lines.emplace_back(Eigen::ParametrizedLine<double, 2>::Through(Eigen::Vector2d {pts[1].x, pts[1].y}, Eigen::Vector2d{pts[2].x, pts[2].y}));
        lines.emplace_back(Eigen::ParametrizedLine<double, 2>::Through(Eigen::Vector2d {pts[2].x, pts[2].y}, Eigen::Vector2d{pts[3].x, pts[3].y}));
        lines.emplace_back(Eigen::ParametrizedLine<double, 2>::Through(Eigen::Vector2d {pts[3].x, pts[3].y}, Eigen::Vector2d{pts[0].x, pts[0].y}));
        return lines;
    }
    QPolygonF Room::get_qt_polygon() const
    {
        QPolygonF poly;
        cv::Point2f pts[4];
        rect.points(pts);
        for(const auto &p: pts)
            poly << QPointF(p.x, p.y);
        return poly;
    }
    double Room::get_width() const
    {
        return rect.size.width;
    }
    double Room::get_width_meters() const
    {
        return rect.size.width / 1000.0;
    }
    double Room::get_depth() const
    {
        return rect.size.height;
    }
    double Room::get_depth_meters() const
    {
        return rect.size.width / 1000.0;
    }
    double Room::get_largest_dim() const
    {
        return std::max(rect.size.width, rect.size.height);
    }
    double Room::get_smallest_dim() const
    {
        return std::min(rect.size.width, rect.size.height);
    }
    double Room::get_height() const
    {
        return 2000.f;  // mm TODO: to be estimated
    }
    Eigen::Vector2d Room::get_center() const
    {
        return Eigen::Vector2d(rect.center.x, rect.center.y);
    }
    double Room::get_center_x() const
    {
        return rect.center.x;
    }
    double Room::get_center_y() const
    {
        return rect.center.y;
    }
    double Room::get_rotation() const
    {
//        return qDegreesToRadians(rect.angle);
        return rect.angle;
    }
    std::vector<QPolygonF> Room::get_walls_as_polygons(const std::vector<QPolygonF> &obstacles, double robot_width) const
    {
        std::vector<QPolygonF> obs(obstacles);
        cv::Point2f pts[4];
        rect.points(pts);
        std::vector<cv::Point2d> points{pts[0], pts[1], pts[2], pts[3], pts[0]};
        for(auto &&pp: iter::sliding_window(points, 2))
        {
            // create line
            QLineF line{pp[0].x, pp[0].y, pp[1].x, pp[1].y};

            // Calculate the direction vector of the line+
            QPointF direction = line.p2() - line.p1();

            // Normalize the direction vector
            double length = std::sqrt(direction.x() * direction.x() + direction.y() * direction.y());
            QPointF unitDirection = direction / length;

            // Calculate the perpendicular vector
            QPointF perpendicular(-unitDirection.y() * robot_width / 2, unitDirection.x() * robot_width / 2);

            // Create the polygon points
            QPointF p1 = line.p1() + perpendicular;
            QPointF p2 = line.p1() - perpendicular;
            QPointF p3 = line.p2() - perpendicular;
            QPointF p4 = line.p2() + perpendicular;

            // Create and return the polygon
            QPolygonF polygon;
            polygon << p1 << p2 << p3 << p4;
            obs.push_back(polygon);
        }
        return obs;
    }
} //rc