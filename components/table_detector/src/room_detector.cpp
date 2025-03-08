//
// Created by pbustos on 2/12/22.
//

#include "room_detector.h"
#include "pch.h"

namespace rc
{
    // par lines and rooms are commented. Lines have to be at least of half-room size
    Features Room_Detector::compute_features(const std::vector<Eigen::Vector2d> &line, QGraphicsScene *scene)
    {
        std::vector<Eigen::Vector2d> floor_line_cart = line;

        // compute mean point
        Eigen::Vector2d room_center = Eigen::Vector2d::Zero();
        room_center = accumulate(floor_line_cart.begin(), floor_line_cart.end(), room_center) / (double)floor_line_cart.size();
        // std::cout << "Center " << room_center << std::endl;

        // estimate size
        Eigen::Vector3d estimated_size = estimate_room_sizes(room_center, floor_line_cart);
        // std::cout << "Size " << estimated_size.x() << " " << estimated_size.y() << std::endl;

        // compute lines
        Lines elines = get_hough_lines(floor_line_cart, estimated_size.head(2));
        if (scene != nullptr) draw_lines_on_2D_tab(elines, scene);

        // compute parallel lines of minimum length and separation
        //Par_lines par_lines = get_parallel_lines(elines, estimated_size.head(2));

        // compute corners
        Corners corners = get_corners(elines);
        if (scene != nullptr) draw_corners_on_2D_tab(corners, {Eigen::Vector2d{0,0}}, scene);

        // compute room candidates by finding triplets of corners in opposite directions and separation within room_size estimations
        //All_Corners all_corners = get_rooms(estimated_size.head(2), corners);

        // draw
        //if (scene != nullptr) draw_lines_on_2D_tab(elines, scene);
        //draw_triple_corners_on_2D_tab(all_corners, "green", scene);
        //        qInfo() << "Room detector: ";
        //        qInfo() << "    num. points:" << floor_line_cart.size();
        //        qInfo() << "    center: [" << room_center.x() << "," << room_center.y() << "]";
        //        qInfo() << "    size: [" << estimated_size.x() << "," << estimated_size.y() << "]";
        //        qInfo() << "    num. lines:" << elines.size();
        //        qInfo() << "    num. parallel lines:" << par_lines.size();
        //        qInfo() << "    num. corners:" << corners.size();
        //        qInfo() << "    num. triple corners:" << all_corners.size();

        //return std::make_tuple(elines, par_lines, corners, all_corners);
        return std::make_tuple(elines, Par_lines(), corners, All_Corners());
    }
    Features Room_Detector::compute_features(const std::vector<Eigen::Vector3d> &line, QGraphicsScene *scene)
    {
        std::vector<Eigen::Vector2d> line2d;
        std::ranges::transform(line, std::back_inserter(line2d), [](const auto &p){return p.head(2);});
        return compute_features(line2d, scene);
    }
    std::optional<std::tuple<Room, std::vector<QPointF>, Eigen::Affine2d>>
        Room_Detector::sample_room_concept(const std::vector<QPolygonF> &polys,
                                           const std::vector<unsigned long> &votes,
                                           const Eigen::Affine2d &robot_pose)
    {
        // Seed with a real random value, if available
        static std::random_device rd;
        // Initialize a random number generator
        static std::mt19937 gen(rd());

        auto euclidean_distance = [](const auto &p1, const auto &p2) { return std::hypot(p1.x() - p2.x(), p1.y() - p2.y()); };
        auto forms_a_rectangle = [euclidean_distance](const auto &centers)  // lambda to check if a combination of 4 points forms a rectangle
            {
                // 1. Calculate side lengths
                const double d12 = euclidean_distance(centers[0].first, centers[1].first);
                const double d23 = euclidean_distance(centers[1].first, centers[2].first);
                const double d34 = euclidean_distance(centers[2].first, centers[3].first);
                const double d41 = euclidean_distance(centers[3].first, centers[0].first);

                // 2. Calculate diagonal lengths
                const double d13 = euclidean_distance(centers[0].first, centers[2].first);
                const double d24 = euclidean_distance(centers[1].first, centers[3].first);

                // 3. Check for equality of opposite sides and diagonals
                //if (qFuzzyCompare(d12, d34) and qFuzzyCompare(d23, d41) and qFuzzyCompare(d13, d24))
                //    return true;
                constexpr double min_dist = 200.f;
                if (std::fabs(d12 - d34)<min_dist and std::fabs(d23 - d41)<min_dist and std::fabs(d13 - d24)<min_dist)
                return true;

                return false;
            };

        // We want to find a set of four polygon centers that form a rectangle
        // Of all 4-points configurations that form a rectangle, we pick with higher probability the one with the highest number of votes
        // compute the center of each polygon
        std::vector<Center> centers;
        for(const auto &[i, p]: polys | iter::enumerate)
            centers.emplace_back(std::accumulate(p.begin(), p.end(), QPointF{0.f, 0.f}) / static_cast<double>(p.size()), votes[i]);

        // make sure points are ordered in a counter-clockwise fashion
        centers = reorder_points_CCW(centers);

        // get all combinations of 4 centers
        std::vector<std::tuple<Center, Center, Center, Center>> valid_rooms;

        for (const auto &comb: iter::combinations(centers, 4))
            if (forms_a_rectangle(comb))
            {
                // Generate QPolygonF from centers
                QPolygonF poly;
                for (const auto &c: comb)
                    poly << c.first;
                // Check if (0, 0) is inside the polygon

                if (poly.containsPoint(QPointF(0.f, 0.f), Qt::OddEvenFill))
                    valid_rooms.emplace_back(comb[0], comb[1], comb[2], comb[3]);
            }
//                valid_rooms.emplace_back(comb[0], comb[1], comb[2], comb[3]);

        if (valid_rooms.empty()) { /*qDebug() << __FUNCTION__ << "No rooms sampled"*/; return{};};

        // from all elements in valid_rooms, select one according to a probability distribution
        // compute the probability of each room
        std::vector<double> weights;
        for (const auto &room: valid_rooms)
        {
            const auto &[c1, c2, c3, c4] = room;
            int weight = std::get<1>(c1) + std::get<1>(c2) + std::get<1>(c3) + std::get<1>(c4);
            weights.emplace_back(static_cast<double>(weight));
        }
        // select a room
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        const auto selected_room = valid_rooms[dist(gen)];
        const auto &[c0, c1, c2, c3] = selected_room;

        // compute the room
        const std::vector<QPointF> corners{c0.first, c1.first, c2.first, c3.first};
        Room room;
        //std::pair<cv::RotatedRect, Eigen::Affine2d>
        const auto &[rect, r_pose] = room.initialize(corners, robot_pose);

        return std::make_tuple(room, corners, r_pose);
    }

     ////////////////////////////////////////////////
    Eigen::Vector3d Room_Detector::estimate_room_sizes(const Eigen::Vector2d &room_center, std::vector<Eigen::Vector2d> &floor_line_cart)
    {
        Eigen::MatrixX2d zero_mean_points(floor_line_cart.size(), 2);
        for(const auto &[i, p] : iter::enumerate(floor_line_cart))
            zero_mean_points.row(i) = p - room_center;

        Eigen::Matrix2d cov = (zero_mean_points.adjoint() * zero_mean_points) / double(zero_mean_points.rows() - 1);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver;
        eigensolver.compute(cov);
        Eigen::Vector2d values = eigensolver.eigenvalues().real().cwiseSqrt()*2.f;
        Eigen::Index i;
        values.maxCoeff(&i);
        Eigen::Vector2d max_vector = eigensolver.eigenvectors().real().col(i);
        return Eigen::Vector3d(values.x(), values.y(), atan2(max_vector.x(), max_vector.y()));
    }
    std::vector<std::pair<int, QLineF>> Room_Detector::get_hough_lines(std::vector<Eigen::Vector2d> &floor_line_cart, const Eigen::Vector2d &estimated_size)
    {
        constexpr double rhoMin = -7000.0, rhoMax = 7000.0, rhoStep = 20;
        constexpr double thetaMin = 0, thetaMax = CV_PI, thetaStep = CV_PI / 180.0f;
        constexpr double max_norm = 8000.f;  // max norm of a point to be considered
        constexpr int max_lines = 10;   // max number of lines to detect
        constexpr int threshold = 25;
        constexpr double delta = 0.3;  // min value for non-maximum supression degrees
        constexpr double min_dist = 400;  // min value for non_maximun supression mm

        if (floor_line_cart.empty()) return {};

        std::vector<cv::Vec2f> floor_line_cv;
        for(const auto &p : floor_line_cart)
            if(not p.isZero() and p.norm() < max_norm)
                floor_line_cv.emplace_back(p.x(), p.y());

        cv::Mat lines;
        HoughLinesPointSet(floor_line_cv, lines, max_lines, threshold, rhoMin,
                           rhoMax, rhoStep, thetaMin, thetaMax, thetaStep);
        if(lines.empty())
            return  {};

        std::vector<cv::Vec3d> lines3d;
        lines.copyTo(lines3d);  // copy-convert to cv::Vec3d  (votes, rho, theta)

        // compute lines from Hough params
        std::vector<std::pair<int, QLineF>> elines;
        for(const auto &l: lines3d)
        {
            // compute QlineF from rho, theta
            const double rho = l[1];
            const double theta = l[2];
            const double a = cos(theta);
            const double b = sin(theta);
            const double x0 = a * rho;
            const double y0 = b * rho;
            const QPointF p1(x0 + max_norm * (-b), y0 + max_norm * (a));
            const QPointF p2(x0 - max_norm * (-b), y0 - max_norm* (a));

            /// compute the supported length of the line and filter lines with low density of points
            auto eig_line = Eigen::ParametrizedLine<double, 2>::Through(to_eigen(p1), to_eigen(p2));
            // select points on the line
            std::vector<Eigen::Vector2d> selected;
            std::ranges::copy_if(floor_line_cart, std::back_inserter(selected), [eig_line](auto &p){ return eig_line.distance(p) < 80;});  //TODO: this is a magic number
            // get the pair of points in selected that are most distant between them.
            double t_min = std::numeric_limits<double>::max();
            double t_max = std::numeric_limits<double>::lowest();
            for (const auto& point : selected)
            {
                Eigen::Vector2d diff = point - eig_line.origin();
                double t = diff.dot(eig_line.direction());
                t_min = std::min(t_min, t);
                t_max = std::max(t_max, t);
            }
            // segment length
            const float length = t_max-t_min;
            // we want to discard lines that are not dense enough be computing the ratio of the length of the line to the number of points
            float ratio = selected.size() / length;
            if (ratio > 0.05) // TODO: this is a magic number (1 point every 20mm)
                elines.emplace_back(l[0], QLineF(p1, p2));    // votes, line
        }

        // Non-Maximum Suppression of close parallel lines
        std::vector<QLineF> to_delete;
        for(auto &&comb: iter::combinations(elines, 2))
        {
            auto &[votes_1, line1] = comb[0];
            auto &[votes_2, line2] = comb[1];
            const double angle = qDegreesToRadians(qMin(line1.angleTo(line2), line2.angleTo(line1)));
            QPointF diff = line1.center() - line2.center();
            const double dist = std::hypot(diff.x(), diff.y());
            if((fabs(angle)<delta or (fabs(angle)-M_PI)< delta) and dist < min_dist)
            {
                if (votes_1 >= votes_2) to_delete.push_back(line2);
                else to_delete.push_back(line1);
            }
        }
        // erase lines that are parallel and too close
        elines.erase(remove_if(elines.begin(), elines.end(), [to_delete](auto l)
            { return std::ranges::find(to_delete, std::get<1>(l)) != to_delete.end();}), elines.end());
        return elines;
    }
    void Room_Detector::filter_lines_by_length(const Lines &lines, std::vector<Eigen::Vector2d> &floor_line_cart )
    {
        Eigen::ParametrizedLine<double, 2> eline;
        const double max_distance_to_line = 100;
        std::vector<std::vector<std::pair<double, Eigen::Vector2d>>> distances(lines.size());
        for(const auto &[i, line]: lines | iter::enumerate)
        {
            const auto &[vote, l] = line;
            eline = Eigen::ParametrizedLine<double, 2>::Through(to_eigen(l.p1()), to_eigen(l.p2()));
            for (const auto &p: floor_line_cart)
            {
                double d = eline.distance(p);
                if(d < max_distance_to_line)
                    distances[i].emplace_back(std::make_pair((p- to_eigen(l.p1())).norm(), p));
            }
        }
        for(const auto &[i, line]: distances | iter::enumerate)
        {
            auto [min, max] = std::ranges::minmax_element(line, [](auto &a, auto &b){ return a.first < b.first;});
            qInfo() << __FUNCTION__ << "Line" << lines[i].second << "min" << min->second.x() << min->second.x()
                                    << "max" << max->second.x() << max->second.y() << "Dist" << (max->second - min->second).norm();
        }
        qInfo() << __FUNCTION__ << "--------------------";
    }
    std::vector<std::pair<QLineF, QLineF>> Room_Detector::get_parallel_lines(const  std::vector<std::pair<int, QLineF>> &lines,
                                                                             const Eigen::Vector2d &estimated_size)
    {
        std::vector<std::pair<QLineF, QLineF>> par_lines;
        for(auto &&line_pairs : iter::combinations(lines, 2))
        {
            const auto &[v1, line1] = line_pairs[0];
            const auto &[v2, line2] = line_pairs[1];
            double ang = fabs(line1.angleTo(line2));
            const double delta = 10;  //degrees
            if((ang > -delta and ang < delta) or ( ang > 180-delta and ang < 180+delta))
            if( euc_distance_between_points(line1.center(), line2.center()) > estimated_size.minCoeff()/2.0)
            par_lines.emplace_back(line1,  line2);
        }
    return par_lines;
    }
    Corners Room_Detector::get_corners(Lines &elines)
    {
        Corners corners;
        for(auto &&comb: iter::combinations(elines, 2))
        {
            auto &[votes_1, line1] = comb[0];
            auto &[votes_2, line2] = comb[1];
            double angle = fabs(qDegreesToRadians(line1.angleTo(line2)));
            if(angle> M_PI) angle -= M_PI;
            if(angle< -M_PI) angle += M_PI;
            double delta = 0.2;
            QPointF intersection;
            if(angle < M_PI/2+delta and angle > M_PI/2-delta  and line1.intersects(line2, &intersection) == QLineF::BoundedIntersection)
                corners.emplace_back(std::make_tuple(std::min(votes_1, votes_2), intersection, 0));
        }
        // sort by votes
        std::sort(corners.begin(), corners.end(), [](auto &a, auto &b){ return std::get<0>(a) > std::get<0>(b);});
        // NM suppression
        const double min_distance_among_corners = 200;
        Corners filtered_corners;
        for(auto &&c: corners | iter::combinations(2))
            if(euc_distance_between_points(std::get<1>(c[0]), std::get<1>(c[1])) < min_distance_among_corners and
               std::ranges::find_if_not(filtered_corners, [p=std::get<1>(c[0])](auto &a){ return std::get<1>(a) == p;}) == filtered_corners.end())
                filtered_corners.push_back(c[0]);
        return corners;
    }
    All_Corners Room_Detector::get_rooms(const Eigen::Vector2d &estimated_size, const Corners &corners)
    {
        // This method has to be improved with:
        // 1, Two corners with opposite angles suffice to define a rectangle and compute the other two corners
        // 2. The distance between corners should be proportional to the room size

        All_Corners all_corners;
        // find pairs of corners in opposite directions and separation within room_size estimations
        double min_dist = estimated_size.minCoeff() / 2; // threshold for distance between corners based on room half size
        for(auto &&comb: iter::combinations(corners, 3))
        {
            const auto &[votes1, p1, _1] = comb[0];
            const auto &[votes2, p2, _2] = comb[1];
            const auto &[votes3, p3, _3] = comb[2];
            std::vector<double> ds{euc_distance_between_points(p1, p2), euc_distance_between_points(p1, p3), euc_distance_between_points(p2, p3)};
            // if all pairs meet the distance criteria of being more than min_dist apart
            if (std::ranges::all_of(ds, [min_dist](auto &a) { return a > min_dist; }))
            {
                auto res = std::ranges::max_element(ds); // find the longest distance
                long pos = std::distance(ds.begin(), res);  //
                QPointF p4;
                // compute the fourth point
                if (pos == 0){ auto l = QLineF(p3, (p1 + p2) / 2).unitVector(); l.setLength(euc_distance_between_points(p1, p2)); p4 = l.pointAt(1);};
                if (pos == 1){ auto l = QLineF(p2, (p1 + p3) / 2).unitVector(); l.setLength(euc_distance_between_points(p1, p3)); p4 = l.pointAt(1);};
                if (pos == 2){ auto l = QLineF(p1, (p2 + p3) / 2).unitVector(); l.setLength(euc_distance_between_points(p2, p3)); p4 = l.pointAt(1);};
                all_corners.emplace_back(p1, p2, p3, p4);
            }
        }
        return all_corners;
    }

     ////////// DRAW
    void Room_Detector::draw_par_lines_on_2D_tab(const Par_lines &par_lines, QGraphicsScene *scene)
    {
        static std::vector<QGraphicsItem*> items;
        for(const auto &item: items)
        {
            scene->removeItem(item);
            delete item;
        }
        items.clear();

        for(const auto &[l1, l2]: par_lines)
        {
            auto i1 = scene->addLine(l1, QPen(QColor("brown"), 30));
            auto i2 = scene->addLine(l2, QPen(QColor("brown"), 30));
            items.push_back(i1); items.push_back(i2);
        }
    }
    void Room_Detector::draw_lines_on_2D_tab(const std::vector<std::pair<int, QLineF>> &lines, QGraphicsScene *scene)
    {
        static std::vector<QGraphicsItem*> lines_vec;
        for (auto l: lines_vec)
        {
            scene->removeItem(l);
            delete l;
        }
        lines_vec.clear();

        for(const auto &l : lines)
        {
            auto p = scene->addLine(l.second, QPen(QColor("orange"), 20));
            lines_vec.push_back(p);
        }
    }
    void Room_Detector::draw_corners_on_2D_tab(const Corners &corners, const std::vector<Eigen::Vector2d> &model_corners,
                                               QGraphicsScene *scene)
    {
        static std::vector<QGraphicsItem*> items;
        for (const auto &i: items)
        {
            scene->removeItem(i);
            delete i;
        }
        items.clear();

        QColor color("lightgreen");
        for(const auto &[votes, p, _] : corners)
        {
            auto i = scene->addEllipse(-100, -100, 200, 200, QPen(color), QBrush(color));
            i->setPos(p.x(), p.y());
            items.push_back(i);
        }
        //model corners
        QColor ccolor("cyan");
        for(const auto &[m, c] : iter::zip(model_corners, corners))
        {
            auto p = scene->addEllipse(-100, -100, 200, 200, QPen(ccolor), QBrush(ccolor));
            p->setPos(m.x(), m.y());
            items.push_back(p);
        }
    }
    void Room_Detector::draw_triple_corners_on_2D_tab(const All_Corners &all_corners, QString color,
                                                      QGraphicsScene *scene)
    {
        static std::vector<QGraphicsItem*> lines_vec;
        for (auto l: lines_vec)
        {
            scene->removeItem(l);
            delete l;
        }
        lines_vec.clear();

        for(const auto &[first, second, third, forth] : all_corners)
        {
            QPolygonF poly;
            poly << first << second << third;
            QColor col(color); col.setAlpha(30);
            auto p = scene->addPolygon(poly, QPen(col), QBrush(col));
            lines_vec.push_back(p);
            break;  // loops ONLY ONCE
        }
    }

    ////////// AUX
    QPointF Room_Detector::get_most_distant_point(const QPointF &p, const QPointF &p1, const QPointF &p2)
    {
        if( (p-p1).manhattanLength() < (p-p2).manhattanLength()) return p2; else return p1;
    }
    Eigen::Vector2d Room_Detector::to_eigen(const QPointF &p)
    {
        return Eigen::Vector2d{p.x(), p.y()};
    }
    Eigen::Vector2d Room_Detector::to_eigen(const cv::Point2d  &p)
    {
        return Eigen::Vector2d{p.x, p.y};
    }
    QPointF Room_Detector::to_qpointf(const cv::Point2d  &p)
    {
        return QPointF{p.x, p.y};
    }
    double Room_Detector::euc_distance_between_points(const QPointF &p1, const QPointF &p2)
    {
        return sqrt((p1.x()-p2.x())*(p1.x()-p2.x())+(p1.y()-p2.y())*(p1.y()-p2.y()));
    }
    std::vector<Center> Room_Detector::reorder_points_CCW(const std::vector<Center>& points)
    {
        // Reorders four points that form a rectangle so that lines connecting  adjacent points do not cross.

        // 2. Calculate angles relative to the center
        std::vector<double> angles;
        for (const auto &p: points | std::views::keys)
            angles.push_back(std::atan2(p.y(), p.x()));

        // 3. Sort points by angle using indices
        std::vector<size_t> indices(points.size());
        std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...
        std::ranges::sort(indices, [&angles](const size_t i1, const size_t i2){ return angles[i1] < angles[i2];});

        // 4. Reorder points
        std::vector<Center> reordered_points;
        for (const auto i : indices)
            reordered_points.push_back(points[i]);
        return reordered_points;
    }

} // rc
