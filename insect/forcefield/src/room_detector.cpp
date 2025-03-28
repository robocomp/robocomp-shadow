//
// Created by pbustos on 2/12/22.
//

#include "room_detector.h"
#include <cppitertools/enumerate.hpp>
#include <cppitertools/combinations.hpp>
#include <cppitertools/combinations_with_replacement.hpp>
#include <cppitertools/zip.hpp>
#include <cppitertools/range.hpp>

namespace rc
{
    void Room_Detector::init(QCustomPlot *customplot_)
    {
        custom_plot = customplot_;
        triple_sig = custom_plot->addGraph();
        triple_sig->setPen(QColor("blue"));
        corners_sig = custom_plot->addGraph();
        corners_sig->setPen(QColor("orange"));
        par_lines_sig = custom_plot->addGraph();
        par_lines_sig->setPen(QColor("green"));
    }
    Room Room_Detector::detect(const std::vector<std::vector<Eigen::Vector2f>> &lines, AbstractGraphicViewer *viewer, bool draw_lines)
    {
        // compute features
        const auto &[_, par_lines, corners, all_corners] =  compute_features(lines, viewer);

        // Start with more complex features and go downwards
        float triple_val=0, corners_val=0, par_lines_val=0;   // for qcustomplot
        if (not all_corners.empty())    // if there are room candidates, take the first room // TODO:: order by votes
        {
            const auto &[c1, c2, c3, c4] = all_corners.front();
            std::vector<cv::Point2f> poly{cv::Point2f(c1.x(), c1.y()), cv::Point2f(c2.x(), c2.y()),
                                          cv::Point2f(c3.x(), c3.y()), cv::Point2f(c4.x(), c4.y())};
            current_room.rect = cv::minAreaRect(poly);
            current_room.is_initialized = true;
            //triple_val = 300; corners_val = 0; par_lines_val = 0;
        }
        // if there is a room and detected corners that match some of the model, compute a resulting force and torque to rotate the model
        else if (not corners.empty() and corners.size() > 1 and current_room.is_initialized)
        {
            double torque = 0.0;
            std::vector<Eigen::Vector2f> model_corners;
            for(const auto &[i, c]: corners | iter::enumerate)
            {
                auto &[votes, p] = c;
                Eigen::Vector2f corner = to_eigen(p);
                Eigen::Vector2f model = current_room.get_closest_corner_in_robot_coor2(corner);
                // compute torque as the perp signed distance of corner on Line(robot_center, model);
                auto line = Eigen::ParametrizedLine<float, 2>::Through(to_eigen(current_room.rect.center), model);
                Eigen::Hyperplane<float, 2> hline(line);
                float sdist = hline.signedDistance(corner);
                model_corners.push_back(model);
                torque += sdist;
            }
            // rotate de rectangle proportional to resulting torque
            double k = 0.01;
            double ang = std::clamp(k*torque, -10.0, 10.0);
            current_room.rect.angle += (float)ang;
            if(fabs(ang) < 2)   // translate
            {
                Eigen::Vector2f tr = to_eigen(std::get<1>(corners[0]))-model_corners[0];
                current_room.rect.center += cv::Point2f(cv::Point2f(tr.x(), tr.y()));
            }
            draw_corners_on_2D_tab(corners, model_corners, viewer);
            triple_val = 0; corners_val = 300; par_lines_val = 0;
        }

        current_room.draw_on_2D_tab(current_room, "yellow", viewer);
        if(draw_lines)
            draw_timeseries(triple_val, corners_val, par_lines_val);
        return  current_room;
    }
    Room_Detector::Features Room_Detector::compute_features(const std::vector<std::vector<Eigen::Vector2f>> &lines,
                                                            AbstractGraphicViewer *viewer)
    {
        std::vector<Eigen::Vector2f> floor_line_cart = lines[0];

        // compute mean point
        Eigen::Vector2f room_center = Eigen::Vector2f::Zero();
        room_center = accumulate(floor_line_cart.begin(), floor_line_cart.end(), room_center) / (float)floor_line_cart.size();
        // std::cout << "Center " << room_center << std::endl;

        // estimate size
        Eigen::Vector3f estimated_size = estimate_room_sizes(room_center, floor_line_cart);
        // std::cout << "Size " << estimated_size.x() << " " << estimated_size.y() << std::endl;

        // compute lines
        //std::vector<std::pair<int, QLineF>> elines = hough_transform(floor_line_cart);
        Lines elines = get_hough_lines(floor_line_cart);
        draw_lines_on_2D_tab(elines, viewer);

        // compute parallel lines of minimum length and separation
        Par_lines par_lines = get_parallel_lines(elines, estimated_size.head(2));

        // compute corners
        Corners corners = get_corners(elines);
        draw_corners_on_2D_tab(corners, {Eigen::Vector2f{0,0}}, viewer);

        // compute room candidates by finding triplets of corners in opposite directions and separation within room_size estimations
        All_Corners all_corners = get_rooms(estimated_size.head(2), corners);

        // draw
        draw_lines_on_2D_tab(elines, viewer);
        draw_triple_corners_on_2D_tab(all_corners, "green", viewer);
//        qInfo() << "Room detector: ";
//        qInfo() << "    num. points:" << floor_line_cart.size();
//        qInfo() << "    center: [" << room_center.x() << "," << room_center.y() << "]";
//        qInfo() << "    size: [" << estimated_size.x() << "," << estimated_size.y() << "]";
//        qInfo() << "    num. lines:" << elines.size();
//        qInfo() << "    num. parallel lines:" << par_lines.size();
//        qInfo() << "    num. corners:" << corners.size();
//        qInfo() << "    num. triple corners:" << all_corners.size();

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
            if(not p.isZero() and p.norm() < 8000.f)
                floor_line_cv.emplace_back(p.x(), p.y());

        double rhoMin = -5000.0, rhoMax = 5000.0, rhoStep = 20;
        double thetaMin = 0, thetaMax = CV_PI, thetaStep = CV_PI / 180.0f;
        cv::Mat lines;
        HoughLinesPointSet(floor_line_cv, lines, 10, 25, rhoMin, rhoMax, rhoStep, thetaMin, thetaMax, thetaStep);
        std::vector<cv::Vec3d> lines3d;
        if(lines.empty())
            return  {};

        lines.copyTo(lines3d);  // copy-convert to cv::Vec3d  (votes, rho, theta)

        // compute lines from Hough params
        std::vector<std::pair<int, QLineF>> elines;
        // filter lines with more than 10 votes
        const double min_votes = 10;  //TODO: take this cons to a CONSTS member
        for(auto &l: std::ranges::filter_view(lines3d, [min = min_votes](auto &l){return l[0]>min;}))
        {
            // compute QlineF from rho, theta
            double rho = l[1], theta = l[2];
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            QPointF p1(x0 + 5000 * (-b), y0 + 5000 * (a));
            QPointF p2(x0 - 5000 * (-b), y0 - 5000 * (a));
            elines.emplace_back(l[0], QLineF(p1, p2));    // votes, line
        }

        // Non-Maximum Suppression of close parallel lines
        const double delta = 0.3;  // degrees
        const double min_dist = 300;  // mm
        std::vector<QLineF> to_delete;
        for(auto &&comb: iter::combinations(elines, 2))
        {
            auto &[votes_1, line1] = comb[0];
            auto &[votes_2, line2] = comb[1];
            double angle = qDegreesToRadians(qMin(line1.angleTo(line2), line2.angleTo(line1)));
            double dist = (line1.center() - line2.center()).manhattanLength();
            if((fabs(angle)<delta or (fabs(angle)-M_PI)< delta) and dist < min_dist)
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
            const float delta = 10;  //degrees
            if((ang > -delta and ang < delta) or ( ang > 180-delta and ang < 180+delta))
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
            float delta = 0.2;
            QPointF intersection;
            if(angle < M_PI/2+delta and angle > M_PI/2-delta  and line1.intersects(line2, &intersection) == QLineF::BoundedIntersection)
                corners.emplace_back(std::make_tuple(std::min(votes_1, votes_2), intersection));
        }
        // sort by votes
        std::sort(corners.begin(), corners.end(), [](auto &a, auto &b){ return std::get<0>(a) > std::get<0>(b);});
        // NM suppression
        const float min_distance_among_corners = 200;
        Corners filtered_corners;
        for(auto &&c: corners | iter::combinations(2))
            if(euc_distance_between_points(std::get<1>(c[0]), std::get<1>(c[1])) < min_distance_among_corners and
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

     ////////// DRAW
    void Room_Detector::draw_par_lines_on_2D_tab(const Par_lines &par_lines, AbstractGraphicViewer *viewer)
    {
        static std::vector<QGraphicsItem*> items;
        for(const auto &item: items)
        {
            viewer->scene.removeItem(item);
            delete item;
        }
        items.clear();

        for(const auto &[l1, l2]: par_lines)
        {
            auto i1 = viewer->scene.addLine(l1, QPen(QColor("brown"), 30));
            auto i2 = viewer->scene.addLine(l2, QPen(QColor("brown"), 30));
            items.push_back(i1); items.push_back(i2);
        }
    }
    void Room_Detector::draw_lines_on_2D_tab(const std::vector<std::pair<int, QLineF>> &lines, AbstractGraphicViewer *viewer)
    {
        static std::vector<QGraphicsItem*> lines_vec;
        for (auto l: lines_vec)
        {
            viewer->scene.removeItem(l);
            delete l;
        }
        lines_vec.clear();

        for(const auto &l : lines)
        {
            auto p = viewer->scene.addLine(l.second, QPen(QColor("orange"), 20));
            lines_vec.push_back(p);
        }
    }
    void Room_Detector::draw_corners_on_2D_tab(const Corners &corners, const std::vector<Eigen::Vector2f> &model_corners,
                                               AbstractGraphicViewer *viewer)
    {
        static std::vector<QGraphicsItem*> items;
        for (const auto &i: items)
        {
            viewer->scene.removeItem(i);
            delete i;
        }
        items.clear();

        QColor color("lightgreen");
        for(const auto &[votes, p] : corners)
        {
            auto i = viewer->scene.addEllipse(-100, -100, 200, 200, QPen(color), QBrush(color));
            i->setPos(p.x(), p.y());
            items.push_back(i);
        }
        //model corners
        QColor ccolor("cyan");
        for(const auto &[m, c] : iter::zip(model_corners, corners))
        {
            auto p = viewer->scene.addEllipse(-100, -100, 200, 200, QPen(ccolor), QBrush(ccolor));
            p->setPos(m.x(), m.y());
            items.push_back(p);
            auto &[v, pc] = c;
            auto l = viewer->scene.addLine(m.x(), m.y(), pc.x(), pc.y(), QPen(ccolor, 25));
            items.push_back(l);
        }
    }
    void Room_Detector::draw_triple_corners_on_2D_tab(const All_Corners &all_corners, QString color,
                                                      AbstractGraphicViewer *viewer)
    {
        static std::vector<QGraphicsItem*> lines_vec;
        for (auto l: lines_vec)
        {
            viewer->scene.removeItem(l);
            delete l;
        }
        lines_vec.clear();

        for(const auto &[first, second, third, forth] : all_corners)
        {
            QPolygonF poly;
            poly << first << second << third;
            QColor col(color); col.setAlpha(30);
            auto p = viewer->scene.addPolygon(poly, QPen(col), QBrush(col));
            lines_vec.push_back(p);
            break;  // loops ONLY ONCE
        }
    }

    ////////// AUX
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

    ///////// QCustomPlot
    void Room_Detector::draw_timeseries(float triple_val, float corners_val, float par_lines_val)
    {
        static int cont = 0;
        triple_sig->addData(cont, triple_val);
        corners_sig->addData(cont, corners_val);
        par_lines_sig->addData(cont++, par_lines_val);
        custom_plot->xAxis->setRange(cont++, 200, Qt::AlignRight);
        custom_plot->replot();
    }

} // rc



//std::vector<std::pair<int, QLineF>> Room_Detector::hough_transform(const std::vector<Eigen::Vector2f> &floor_line_cart)
//{
//    const int line_threshold = 30;
//    Eigen::Vector2f max_elem = std::ranges::max(floor_line_cart, [](auto &a, auto &b){return a.norm() < b.norm();});
//    int max_distance = hypot(max_elem.x(), max_elem.y());
//    int rhoMin = 0.0, rhoMax = max_distance+1, rhoStep = 1;
//    int thetaMin = 0, thetaMax = 180, thetaStep = 1;
//    std::vector<std::vector<int>> votes(2*(max_distance+1), std::vector<int>((int)((thetaMax-thetaMin)/thetaStep), 0));
//    //qInfo() << __FUNCTION__ << votes.size();
//    double rho;
//    // votes
//    for (const auto &p: floor_line_cart)
//        for (int theta = thetaMin; theta <= thetaMax; theta += thetaStep)
//        {
//            double angle = qDegreesToRadians((double )(theta-90));
//            rho = round(p.x()*cos(angle) + p.y()*sin(angle) + max_distance+1);
//            if(rho < 0 or rho > max_distance*2)
//                qInfo() << __FUNCTION__ << "asd" << rho;
//            else
//                votes[rho][theta]++;
//        }
//    //qInfo() << __FUNCTION__ << "ok";
//    // find peaks
//    std::vector<std::pair<int, QLineF>> lines;
//    for (const auto &[i, vote]: votes | iter::enumerate)
//        for (const auto &[j, v]: vote | iter::enumerate)
//            if (v >= line_threshold)
//            {
//                //qInfo() << __FUNCTION__ << v;
//                rho = i - max_distance-1;
//                int theta = qDegreesToRadians((double)(j - 90));
//                double x0 = rho * cos(theta);
//                double y0 = rho * sin(theta);
//                QPointF p1{x0 + 1000 * (-sin(theta)), y0 + 1000 * (cos(theta))};
//                QPointF p2{x0 - 1000 * (-sin(theta)), y0 - 1000 * (cos(theta))};
//                lines.emplace_back(std::make_pair(v, QLineF(p1, p2)));
//            }
//
//    qInfo() << __FUNCTION__ << lines.size();
//    return lines;
//}


//    else if(not corners.empty())
//{
//    float total_torque = 0.f;
//    std::vector<std::tuple<QPointF, QPointF, int>> temp;
//    for(const auto &c : corners)
//{
//    // get the closest corner in model c*
//    auto const &corner = Eigen::Vector2f(c.p1().x(), c.p1().y());
//    auto closest = current_room.get_closest_corner(corner);
//    // project p on line joining c* with center of model
//    auto line = Eigen::ParametrizedLine<float, 2>::Through(current_room.center, closest);
//    auto a = current_room.to_local_coor(closest);
//    auto b = current_room.to_local_coor(corner);
//    float angle = atan2( a.x()*b.y() - a.y()*b.x(), a.x()*b.x() + a.y()*b.y() );  // sin / cos
//    int signo;
//    if(angle >= 0) signo = 1; else signo = -1;
//    auto proj = line.projection(corner);
//    // compute torque as distance to center times projection length
//    total_torque += signo */* ((room_center - proj).norm() +*/ (proj - corner).norm();
//    temp.emplace_back(std::make_tuple(QPointF(corner.x(), corner.y()), QPointF(proj.x(), proj.y()), signo));
//}
//
//            // rotate the model
//            float inertia = 10.f;
//            float inc = total_torque / 1000.f / inertia;
//            current_room.rotate(inc);
//            // if there is still error
//            // translate the model
//            //room.draw_on_2D_tab(&viewer->scene, temp);
//        }

//        else if(not corners.empty() and corners.size() > 1 and current_room.is_initialized)
//        {
//            // associate to the closest corners in the model (replace by Hungarian)
//            Eigen::MatrixX2f model_corners(corners.size(), 2);
//            Eigen::MatrixX2f scene_corners(corners.size(), 2);
//            for(const auto &[i, c]: corners | iter::enumerate)
//            {
//                // both matrices in room coordinates
//                auto &[votes, p] = c;
//                Eigen::Vector2f c2r = to_eigen(p);
//                model_corners.row(i) = current_room.get_closest_corner_in_robot_coor2(c2r);
//                scene_corners.row(i) = c2r;
//            }
//            // compute the cross products
//            Eigen::MatrixX2f model_centered = model_corners.rowwise() - model_corners.colwise().mean();
//            Eigen::MatrixX2f scene_centered = scene_corners.rowwise() - scene_corners.colwise().mean();
//            Eigen::MatrixX2f data = (model_centered.adjoint() * scene_centered);
//            Eigen::JacobiSVD<Eigen::MatrixXf> svd(data, Eigen::ComputeThinU | Eigen::ComputeThinV);  // gives U * S.asDiagonal() * V.transpose() == C
//            Eigen::Rotation2Df rot;
//            Eigen::Matrix2f r = svd.matrixU() * svd.matrixV().transpose();
//            rot.fromRotationMatrix(r);
//            qInfo() << __FUNCTION__  << "Angle from corners" << rot.smallestAngle();
//            current_room.rect.angle = qRadiansToDegrees(rot.angle());
//            draw_corners_on_2D_tab(corners, model_corners, viewer);
//        }

/////////  code to move the room pulling from parallel lines
//        else if(not par_lines.empty()  and current_room.is_initialized)
//        {
//            std::vector<float> ang_dist, min_angles;
//            for (const auto &[l1, l2]: par_lines)
//            {
//                // compute angle between model lines and l1,l2
//                for (auto model_lines = current_room.get_room_lines_qt(); auto &ml: model_lines)
//                    ang_dist.emplace_back(fabs(ml.angleTo(l1)));
//                min_angles.emplace_back(std::ranges::min(ang_dist));
//            }
//            // rotate the model
//            float min_ang = std::ranges::min(min_angles);
//            float k = 0.01;
//            float ang = std::clamp(k * min_ang, -10.f, 10.f);
//            current_room.rect.angle += ang;
//            triple_val = 0;
//            corners_val = 0;
//            par_lines_val = 300;
//        }