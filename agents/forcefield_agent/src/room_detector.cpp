//
// Created by pbustos on 2/12/22.
//

#include "room_detector.h"
#include <cppitertools/enumerate.hpp>
#include <cppitertools/combinations.hpp>
#include <cppitertools/zip.hpp>
#include <cppitertools/range.hpp>

namespace rc
{
    Room Room_Detector::detect(const std::vector<std::vector<Eigen::Vector2f>> &lines, QGraphicsScene *scene, bool draw_lines)
    {
        // compute features
        const auto &[elines, par_lines, corners, all_corners] =  compute_features(lines);

        // Start with more complex features and go downwards
        if (not all_corners.empty())    // if there are room candidates, take the first room // TODO:: order by votes
        {
            const auto &[c1, c2, c3, c4] = all_corners.front();
            std::vector<cv::Point2f> poly{cv::Point2f(c1.x(), c1.y()), cv::Point2f(c2.x(), c2.y()),
                                          cv::Point2f(c3.x(), c3.y()), cv::Point2f(c4.x(), c4.y())};
            current_room.rect = cv::minAreaRect(poly);
            current_room.is_initialized = true;
        }

        // print sizes of corners, par_lines and all_corners
        //        std::cout << "Corners " << corners.size() << " Par_lines " << par_lines.size() << " All_corners " << all_corners.size() << std::endl;
        //        for(const auto &l: elines)
        //            qInfo() << "Line" << l.second << "Votes" << l.first;
        //        qInfo() << "----------------";

        if(draw_lines)
        {
            draw_corners_on_2D_tab(corners, std::vector<Eigen::Vector2f>(), scene);
            draw_par_lines_on_2D_tab(par_lines, scene);
        }
        return  current_room;
    }

    Room_Detector::Features Room_Detector::compute_features(const std::vector<std::vector<Eigen::Vector2f>> &lines)
    {
        std::vector<Eigen::Vector2f> floor_line_cart = lines[0];

        // compute mean (center) point for x and y coordinates
        Eigen::Vector2f room_center = Eigen::Vector2f::Zero();
        room_center = accumulate(floor_line_cart.begin(), floor_line_cart.end(), room_center) / (float)floor_line_cart.size();
        // std::cout << "Center " << room_center << std::endl;

        // estimate room size
        Eigen::Vector3f estimated_size = estimate_room_sizes(room_center, floor_line_cart);
        // std::cout << "Size " << estimated_size.x() << " " << estimated_size.y() << std::endl;

        // compute lines
        Lines elines = get_hough_lines(floor_line_cart);

        // filter parallel lines of minimum length and separation: GOOD PLACE TO INTRODUCE TOP_DOWN MODEL
        Par_lines par_lines = get_parallel_lines(elines, estimated_size.head(2));

        // compute corners
        Corners corners = get_corners(elines);

        // compute room candidates by finding triplets of corners in opposite directions and separation within room_size estimations
        All_Corners all_corners = get_rooms(estimated_size.head(2), corners);

        return std::make_tuple(elines, par_lines, corners, all_corners);
    }
     ////////////////////////////////////////////////
    Eigen::Vector3f Room_Detector::estimate_room_sizes(const Eigen::Vector2f &room_center, std::vector<Eigen::Vector2f> &floor_line_cart) const
    {
        Eigen::MatrixX2f zero_mean_points(floor_line_cart.size(), 2);
        for(const auto &[i, p] : iter::enumerate(floor_line_cart))
            zero_mean_points.row(i) = p - room_center;

        Eigen::Matrix2f cov = (zero_mean_points.adjoint() * zero_mean_points) / float(zero_mean_points.rows() - 1);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eigensolver;
        eigensolver.compute(cov);
        Eigen::Vector2f values = eigensolver.eigenvalues().real().cwiseSqrt()*2.f;
        Eigen::Index i;
        values.maxCoeff(&i);
        Eigen::Vector2f max_vector = eigensolver.eigenvectors().real().col(i);
        return Eigen::Vector3f(values.x(), values.y(), atan2(max_vector.x(), max_vector.y()));
    }
    std::vector<std::pair<int, QLineF>> Room_Detector::get_hough_lines(std::vector<Eigen::Vector2f> &floor_line_cart) const
    {
        std::vector<cv::Vec2f> floor_line_cv;
        for(const auto &p : floor_line_cart)
                floor_line_cv.emplace_back(p.x(), p.y());

        cv::Mat lines;
        HoughLinesPointSet(floor_line_cv, lines,
                           params.LINES_MAX, params.LINE_THRESHOLD,
                           params.rhoMin, params.rhoMax, params.rhoStep,
                           params.thetaMin, params.thetaMax, params.thetaStep);
        std::vector<cv::Vec3d> lines3d;
        if(lines.empty())
            return  {};

        lines.copyTo(lines3d);  // copy-convert to cv::Vec3d  (votes, rho, theta)

        /// compute lines from Hough params
        std::vector<std::pair<int, QLineF>> elines;
        // filter lines with more than params.MIN_VOTES votes
        for(auto &l: std::ranges::filter_view(lines3d, [min = params.MIN_VOTES](auto &l){return l[0] > min;}))
        {
            // compute QlineF from rho, theta
            double rho = l[1], theta = l[2];
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            QPointF p1(x0 + params.MAX_LIDAR_DISTANCE * (-b), y0 + params.MAX_LIDAR_DISTANCE * (a));
            QPointF p2(x0 - params.MAX_LIDAR_DISTANCE * (-b), y0 - params.MAX_LIDAR_DISTANCE * (a));
            elines.emplace_back(l[0], QLineF(p1, p2));    // votes, line
        }

        // TODO: filter out lines shorter than wall estimated length

        // Non-Maximum Suppression of close parallel lines
        std::vector<QLineF> to_delete;
        for(auto &&comb: iter::combinations(elines, 2))
        {
            auto &[votes_1, line1] = comb[0];
            auto &[votes_2, line2] = comb[1];
            double angle = qDegreesToRadians(qMin(line1.angleTo(line2), line2.angleTo(line1)));
            double dist = (line1.center() - line2.center()).manhattanLength();
            if((fabs(angle)<params.NMS_DELTA or (fabs(angle)-M_PI)< params.NMS_DELTA) and dist < params.NMS_DIST)
            {
                if (votes_1 >= votes_2) to_delete.push_back(line2);
                else to_delete.push_back(line1);
            }
        }

        // erase lines that are parallel and too close
        elines.erase(remove_if(elines.begin(), elines.end(), [to_delete](auto l){ return std::ranges::find(to_delete, std::get<1>(l)) != to_delete.end();}), elines.end());
        return elines;

    }
    void Room_Detector::filter_lines_by_length(const Lines &lines, std::vector<Eigen::Vector2f> &floor_line_cart )
    {
        Eigen::ParametrizedLine<float, 2> eline;
        const float max_distance_to_line = 100;
        std::vector<std::vector<std::pair<float, Eigen::Vector2f>>> distances(lines.size());
        for(const auto &[i, line]: lines | iter::enumerate)
        {
            const auto &[vote, l] = line;
            eline = Eigen::ParametrizedLine<float, 2>::Through(to_eigen(l.p1()), to_eigen(l.p2()));
            for (const auto &p: floor_line_cart)
            {
                float d = eline.distance(p);
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
                                                                             const Eigen::Vector2f &estimated_size)
    {
        std::vector<std::pair<QLineF, QLineF>> par_lines;
        for(auto &&line_pairs : iter::combinations(lines, 2))
        {
            const auto &[v1, line1] = line_pairs[0];
            const auto &[v2, line2] = line_pairs[1];
            double ang = fabs(line1.angleTo(line2));
            if((ang > - params.PAR_LINES_ANG_THRESHOLD and ang < params.PAR_LINES_ANG_THRESHOLD)
                or ( ang > 180 - params.PAR_LINES_ANG_THRESHOLD and ang < 180 + params.PAR_LINES_ANG_THRESHOLD))
            if( euc_distance_between_points(line1.center(), line2.center()) > estimated_size.minCoeff()/2.0)
            par_lines.emplace_back(line1,  line2);
        }
    return par_lines;
    }
    Room_Detector::Corners Room_Detector::get_corners(Lines &elines)
    {
        Corners corners;
        for(auto &&comb: iter::combinations(elines, 2))
        {
            auto &[votes_1, line1] = comb[0];
            auto &[votes_2, line2] = comb[1];
            double angle = fabs(qDegreesToRadians(line1.angleTo(line2)));
            if(angle> M_PI) angle -= M_PI;
            if(angle< -M_PI) angle += M_PI;
            QPointF intersection;
            if(angle < M_PI/2 + params.CORNERS_PERP_ANGLE_THRESHOLD and angle > M_PI/2 - params.CORNERS_PERP_ANGLE_THRESHOLD
                and line1.intersects(line2, &intersection) == QLineF::BoundedIntersection)
                corners.emplace_back(std::make_tuple(std::min(votes_1, votes_2), intersection));
        }
        // sort by votes
        std::sort(corners.begin(), corners.end(), [](auto &a, auto &b){ return std::get<0>(a) > std::get<0>(b);});

        // Non Maximum suppression
        Corners filtered_corners;
        for(auto &&c: corners | iter::combinations(2))
            if(euc_distance_between_points(std::get<1>(c[0]), std::get<1>(c[1])) < params.NMS_MIN_DIST_AMONG_CORNERS and
               std::ranges::find_if_not(filtered_corners, [p=std::get<1>(c[0])](auto &a){ return std::get<1>(a) == p;}) == filtered_corners.end())
                filtered_corners.push_back(c[0]);
        return corners;
    }
    Room_Detector::All_Corners Room_Detector::get_rooms(const Eigen::Vector2f &estimated_size, const Corners &corners)
    {
        All_Corners all_corners;

        // find pairs of corners in opposite directions and separation within room_size estimations
        float min_dist = estimated_size.minCoeff() / 2; // threshold for distance between corners based on room half size
        for(auto &&comb: iter::combinations(corners, 3))
        {
            const auto &[votes1, p1] = comb[0];
            const auto &[votes2, p2] = comb[1];
            const auto &[votes3, p3] = comb[2];
            std::vector<float> ds{euc_distance_between_points(p1, p2), euc_distance_between_points(p1, p3), euc_distance_between_points(p2, p3)};

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

     ////////// DRAW  //////////////////////////////////////////////////////////////////////////////////////
    void Room_Detector::draw_par_lines_on_2D_tab(const Par_lines &par_lines, QGraphicsScene *scene, QColor color)
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
            auto i1 = scene->addLine(l1, QPen( color, 30));
            auto i2 = scene->addLine(l2, QPen( color, 30));
            items.push_back(i1); items.push_back(i2);
        }
    }
    void Room_Detector::draw_corners_on_2D_tab(const Corners &corners, const std::vector<Eigen::Vector2f> &model_corners,
                                               QGraphicsScene *scene, QColor color)
    {
        static std::vector<QGraphicsItem*> items;
        for (const auto &i: items)
        {
            scene->removeItem(i);
            delete i;
        }
        items.clear();

        for(const auto &[votes, p] : corners)
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
            auto &[v, pc] = c;
            auto l = scene->addLine(m.x(), m.y(), pc.x(), pc.y(), QPen(ccolor, 25));
            items.push_back(l);
        }
    }

    ////////// AUX  ////////////////////////////////////////////////////////////////////////////////////////////7
    QPointF Room_Detector::get_most_distant_point(const QPointF &p, const QPointF &p1, const QPointF &p2) const
    {
        if( (p-p1).manhattanLength() < (p-p2).manhattanLength()) return p2; else return p1;
    }
    Eigen::Vector2f Room_Detector::to_eigen(const QPointF &p) const
    {
        return Eigen::Vector2f{p.x(), p.y()};
    }
    Eigen::Vector2f Room_Detector::to_eigen(const cv::Point2f  &p) const
    {
        return Eigen::Vector2f{p.x, p.y};
    }
    QPointF Room_Detector::to_qpointf(const cv::Point2f  &p) const
    {
        return QPointF{p.x, p.y};
    }
    float Room_Detector::euc_distance_between_points(const QPointF &p1, const QPointF &p2) const
    {
        return sqrt((p1.x()-p2.x())*(p1.x()-p2.x())+(p1.y()-p2.y())*(p1.y()-p2.y()));
    }

} // rc
