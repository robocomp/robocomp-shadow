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
#include <cppitertools/enumerate.hpp>
#include <cppitertools/filter.hpp>
#include <cppitertools/range.hpp>
#include <cppitertools/sliding_window.hpp>
#include <cppitertools/slice.hpp>

/**
* \brief Default constructor
*/
SpecificWorker::SpecificWorker(TuplePrx tprx, bool startup_check) : GenericWorker(tprx)
{
	this->startup_check_flag = startup_check;
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
		// Viewer
 		viewer = new AbstractGraphicViewer(this->frame, QRectF(-3000, -3000, 6000, 6000), false);
        viewer->draw_contour();
        viewer->add_robot(460, 480, 0, 100, QColor("Blue"));
        std::cout << "Started viewer" << std::endl;

        // create map from degrees (0..360)  -> edge distances
        // int robot_width = 460;
        // int robot_heigth = 480;
        robot_contour << QPointF(-230,240) << QPointF(230, 240) << QPointF(230, -240) << QPointF(-230, -240);
        robot_safe_band << QPointF(-230-BAND_WIDTH, 240+BAND_WIDTH) <<
                           QPointF(230+BAND_WIDTH, 240+BAND_WIDTH) <<
                           QPointF(230+BAND_WIDTH, -240-BAND_WIDTH) <<
                           QPointF(-230-BAND_WIDTH, -240-BAND_WIDTH);

		map_of_points = create_map_of_points();
        draw_ring(map_of_points, &viewer->scene);

        float x1 = -band.left_distance;
        float y1 = -band.frontal_distance;
        float width = band.left_distance + band.right_distance;
        float height = band.frontal_distance + band.back_distance;

        rectItem = viewer->scene.addRect(QRectF(x1, y1, width, height), QColor("Red"));

        // Create Pen used in the QrectItem
        QPen rect_pen;
        rect_pen.setColor(QColor("Red"));
        rect_pen.setWidth(20);
        rectItem->setPen(rect_pen);

        std::cout << "Started robot draw" << std::endl;

        // A thread is created 
        read_lidar_th = std::move(std::thread(&SpecificWorker::read_lidar,this));
        std::cout << "Started lidar reader" << std::endl;

        timer.start(50);
	}
}

void SpecificWorker::compute()
{
    //std::cout << "Speeds:" << robot_speed.adv_speed << " " << robot_speed.side_speed << " "  << robot_speed.rot_speed << std::endl;

    // Calculate band width using last robot_speed
    //band = adjustSafetyZone(Eigen::Vector3f(robot_speed.adv_speed,robot_speed.side_speed,robot_speed.rot_speed));

    #if DEBUG
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    auto res_ = buffer_lidar_data.try_get(); res_.has_value();

    if(res_.has_value() == false)
    { qWarning() << "No data Lidar"; return; }

    auto ldata = res_.value();

    // filter out floor points and inbody points
    RoboCompLidar3D::TPoints above_floor_points;
    for(const auto &p: ldata.points)    // TODO: Check this
        if(p.r > 100 and p.z < 1000)
            above_floor_points.emplace_back(p);

    if(above_floor_points.empty())
    { qInfo() << "empty vector"; return;}

    //auto ldata = filterPointsInRectangle(ldata_raw);
    auto discr_points = discretize_lidar(above_floor_points);
    auto enlarged_points = configuration_space(discr_points);
    //qInfo() << discr_points.size() << enlarged_points.size();

#if DEBUG
    qInfo() << "Post get_lidar_data" << (std::chrono::duration<double, std::milli> (std::chrono::high_resolution_clock::now() - start)).count();
    start = std::chrono::high_resolution_clock::now();
    #endif

    // check for a repulsion force
    Eigen::Vector2f result{0.f, 0.f};
    std::vector<QPointF> draw_points;

    for(const auto &p: ldata.points)
	{
		float ang = atan2(p.x, p.y);
        int index  = qRadiansToDegrees(ang);

        if(index < 0) index += 360;

        float norma = std::sqrt(p.x * p.x + p.y * p.y);
        float diff = map_of_points.at(index) - norma;// negative vals are inside belt
        
        //Check that p is greater than robot body and smaller than robot body + BAND_WIDTH
        if(diff < 0 and diff > -BAND_WIDTH)
        {
			// something is inside the perimeter
            // pseudo sigmoid
            auto diff_normalized = fabs(diff)*(1.f/BAND_WIDTH);
            //std::cout << "Diff_normalized:" << diff_normalized << std::endl;
            //float modulus = std::clamp(fabs(diff)*(1.f/BAND_WIDTH), 0.f, 1.f);  
            float modulus = (-1/(1+ exp((diff_normalized-0.5)*12)))+1;
            //std::cout << "Modulus:" << modulus << std::endl;
            //std::cout << "Distance:" << p.head(2).normalized() << std::endl;
            //result -= p.head(2).normalized() * modulus;     // opposite direction and normalized modulus
            result -= Eigen::Vector2f{p.x/norma, p.y/norma} * modulus;     // opposite direction and normalized modulus

            //qInfo() << "diff" << diff << modulus << result.x() << result.y();
            draw_points.emplace_back(p.x, p.y);
		}
	}

    #if DEBUG
    qInfo() << "Post result" << (std::chrono::duration<double, std::milli> (std::chrono::high_resolution_clock::now() - start)).count();
    start = std::chrono::high_resolution_clock::now();
    #endif

    //qInfo() << "Result norm from last point" << result.norm();
    // TODO: Modify to one try catch calling setSpeedBase and only modify adv and side variables 
//    if(result.norm() > 3)    // a clean bumper should be zero
//    {
//        // Use std::clamp to ensure that the value of side is within the range [-value, value]
//        robot_speed.adv_speed = std::clamp(result.x() * x_gain,-max_adv,max_adv);
//        robot_speed.side_speed = std::clamp(result.y() * y_gain,-max_side,max_side);
//        robot_speed.rot_speed = 0.0f;
//
//        robot_stop = false;
//    }
//    else
    if(const auto res = buffer_dwa.try_get(); res.has_value())
    {
        const auto &[side, adv, rot] = res.value();
        robot_speed.adv_speed = adv;
        robot_speed.side_speed = side;
        robot_speed.rot_speed = rot;
    }
    else
        if(not robot_stop)
        {
            robot_speed.adv_speed = 0.0f;
            robot_speed.side_speed = 0.0f;
            robot_speed.rot_speed = 0.0f;
            robot_stop = true;
        }
    try
    {
        omnirobot_proxy->setSpeedBase(robot_speed.adv_speed, 0, robot_speed.rot_speed);

        // Draw repulsion line
        static QGraphicsItem * line = nullptr;
        if( line != nullptr)
            viewer->scene.removeItem(line);

        line =  viewer->scene.addLine(0, 0, robot_speed.adv_speed*5, robot_speed.side_speed*5, QPen(QColor("Green"), 10));

    }
    catch (const Ice::Exception &e) {}//std::cout << "Error talking to OmniRobot " << e.what() << std::endl; }

#if DEBUG
    qInfo() << "Post sending adv, side, rot" << (std::chrono::duration<double, std::milli> (std::chrono::high_resolution_clock::now() - start)).count();
    start = std::chrono::high_resolution_clock::now();
#endif

//     if(display)
//         draw_ring_points(draw_points, result, &viewer->scene);

    draw_all_points(above_floor_points, discr_points, enlarged_points, result, &viewer->scene);
    draw_band_width(&viewer->scene);

    #if DEBUG
        qInfo() << "Post draw_all_points" << (std::chrono::duration<double, std::milli> (std::chrono::high_resolution_clock::now() - start)).count();
        qInfo() << "";
    #endif
    fps.print("FPS:");
}

std::vector<std::tuple<float, float>> SpecificWorker::discretize_lidar(const RoboCompLidar3D::TPoints &ldata)
{
    std::vector<std::tuple<float, float>> polar_points;
    const float delta_phi = 2*(M_PI*2/360); // number of degrees
    //float running_angle = ldata[0].phi +  delta_phi;
    float running_angle = -M_PI + delta_phi;
    float running_min = std::numeric_limits<float>::max();


    // Group points by discrete angle bins and compute the min dist of the set
    for (const auto& p : ldata)
    {
        if(p.phi <= running_angle)
        {
            if (p.r < running_min)
                running_min = p.r;
        }
        else
        {
            if (p.phi <= running_angle + delta_phi)
            {
                polar_points.emplace_back(running_angle + delta_phi / 2.f, std::min(running_min, p.r));
                running_angle += delta_phi;
            }
            else
                while(p.phi > running_angle + delta_phi)
                {
                    polar_points.emplace_back(running_angle + delta_phi / 2.f, 3000);
                    running_angle += delta_phi;
                    if(running_angle > M_PI) running_angle -= 2*M_PI;
                }
            running_min = std::numeric_limits<float>::max();
        }
    }
    // complete the circle
    while(running_angle< M_PI)
    {
        polar_points.emplace_back(running_angle + delta_phi / 2.f, 3000);
        running_angle += delta_phi;
    }
    return polar_points;
}
// compute new polar vector extending the obstacles by half the radius of the robot: paper
std::vector<std::tuple<float, float>> SpecificWorker::configuration_space(const std::vector<std::tuple<float, float>> &points)
{
    std::vector<std::tuple<float, float>> conf_space;
    const float R = 250;
    auto angle_diff = [](auto a, auto b){ return atan2(sin(a - b), cos(a - b));};
    //auto angle_diff = [](auto a, auto b){ float diff = std::abs(a-b); if(diff > 180.0) diff = 2*M_PI - diff; return diff;};

    for (const auto &[ang_i, dist_i] : points)
    {
        std::vector<float> dij_vec;
        for(auto &&[ang_j, dist_j] :
                iter::filter([ang_i, angle_diff](auto pj){return fabs(angle_diff(ang_i, std::get<0>(pj))) <= M_PI/4;},  points))
        {
            float diff = fabs(angle_diff(ang_i, ang_j));
            float sij = dist_j * sin(diff);
            if(sij > R or dist_j*cos(diff) > dist_i)
                dij_vec.emplace_back(dist_i);
            else
                dij_vec.emplace_back(dist_j*cos(diff));
        }
        conf_space.emplace_back(ang_i, std::ranges::min(dij_vec) - R);
    }
    return conf_space;
}

//////////////////////////////////////////////////////////////////////////////
SpecificWorker::Band SpecificWorker::adjustSafetyZone(Eigen::Vector3f velocity)
{
    Band adjusted;

    // Si x es positivo, aumentamos la distancia frontal y disminuimos la trasera.
    // Si x es negativo, hacemos lo contrario.
    if (velocity.x() >= 0.0f)
    {
        adjusted.right_distance = std::max(velocity.x()*5,600.0f);
        adjusted.left_distance = -600.0f;
    } else //if(velocity.x() <= -600.0f)
    {
        adjusted.right_distance = 600.0f;
        adjusted.left_distance = std::min(velocity.x()*5,-600.0f);
    }
//    else
//    {
//        adjusted.right_distance = 600.0f;
//        adjusted.left_distance = 600.0f;
//    }

    // Si y es positivo, aumentamos la distancia derecha y disminuimos la izquierda.
    // Si y es negativo, hacemos lo contrario.
    if (velocity.y() >= 0.0f)
    {
        adjusted.frontal_distance = std::max(velocity.y()*5,600.0f);
        adjusted.back_distance = -600.0f;
    } else //if(velocity.y() <= -600.0f)
    {
        adjusted.frontal_distance = 600.0f;
        adjusted.back_distance = std::min(velocity.y()*5,-600.0f);
    }
//    else
//    {
//        adjusted.frontal_distance = 600.0f;
//        adjusted.back_distance = 600.0f;
//    }

    std::cout << adjusted.frontal_distance << " " << adjusted.back_distance << " " << adjusted.right_distance << " " << adjusted.left_distance << std::endl;
    return adjusted;
}

void SpecificWorker::draw_band_width(QGraphicsScene *scene)
{
    // Suponiendo que el punto central es (0, 0)
    float x1 = band.left_distance;   // Coordenada x de la esquina superior izquierda
    float y1 = band.frontal_distance; // Coordenada y de la esquina superior izquierda
    float x2 = band.right_distance;   // Coordenada x de la esquina inferior derecha
    float y2 = band.back_distance;   // Coordenada y de la esquina inferior derecha

    rectItem->setRect(QRectF ( x1, y1, x2 - x1, y2 - y1));
}

std::vector<Eigen::Vector3f> SpecificWorker::filterPointsInRectangle(const std::vector<Eigen::Vector3f>& points)
{
    std::vector<Eigen::Vector3f> filteredPoints;

    for (const auto& p : points)
    {
        if (p.x() >= band.left_distance && p.x() <= band.right_distance &&
            p.y() >= band.back_distance && p.y() <= band.frontal_distance)
        {
            filteredPoints.push_back(p);
        }
    }

    return filteredPoints;
}

//void SpecificWorker::draw_speed_vector(Eigen::Vector3f &velocity, Eigen::Vector3f &colour, QGraphicsScene *scene)
//{
//
//}

// Thread to read the lidar
void SpecificWorker::read_lidar()
{
    //auto start = std::chrono::high_resolution_clock::now();
    while(true)
    {
        // qInfo() << "While beginning" << (std::chrono::duration<double, std::milli> (std::chrono::high_resolution_clock::now() - start)).count();
        // start = std::chrono::high_resolution_clock::now();
        try
        {
            // Use with simulated lidar in webots using "pearl" name
            // auto data = lidar3d_proxy->getLidarData("pearl", 0, 360, 8);

            auto data = lidar3d_proxy->getLidarDataWithThreshold2d("bpearl", 3000);
            // std::cout << data.points.size() << std::endl;
            std::ranges::sort(data.points, {}, &RoboCompLidar3D::TPoint::phi);
            buffer_lidar_data.put(std::move(data));
        }
        catch (const Ice::Exception &e) { std::cout << "Error reading from Lidar3D" << e << std::endl; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));}
}
std::vector<float> SpecificWorker::create_map_of_points()
{
	std::vector<float> dists;
	for(auto &&i: iter::range(DEGREES_NUMBER))
	{
		// get the corresponding angle: 0..360 -> -pi, +pi
		float alf = qDegreesToRadians(static_cast<float>(i));
		bool found = false;
        // iter from 0 to OUTER_RIG_DISTANCE until the point falls outside the polygon
		for(const int r : iter::range(OUTER_RIG_DISTANCE))
		{
			float x = r * sin(alf);
			float y = r * cos(alf);
            if( not robot_contour.containsPoint(QPointF(x, y), Qt::OddEvenFill))
			{
				dists.push_back(Eigen::Vector2f(x, y).norm() - 1);
				found = true;
				break;
			}
		}
		if(not found) { qFatal("ERROR: Could not find limit for angle ");	}
	}
    return dists;
}
//            return p.z < 100		// uppper limit
//                   and p.z > -600	// floor limit
//                   and dist < 1000	// range limit
//                   and dist > 100;	// body out limit. This should be computed using the robot's contour

///////////////////////////////////////////////////////////////////////////
void SpecificWorker::draw_ring(const std::vector<float> &dists, QGraphicsScene *scene) {
    static std::vector<QGraphicsItem *> draw_points;
    for (const auto &p: draw_points) {
        scene->removeItem(p);
        delete p;
    }
    draw_points.clear();

    QPolygonF poly;
    for (const auto &[i, p]: dists | iter::enumerate)
        poly << QPointF(p * cos(qDegreesToRadians(static_cast<float>(i))), p * sin(qDegreesToRadians(static_cast<float>(i))));
    auto o = scene->addPolygon(poly, QPen(QColor("DarkBlue"), 10));
    draw_points.push_back(o);

    // draw external ring
    o = scene->addPolygon(robot_safe_band, QPen(QColor("DarkBlue"), 10));
    draw_points.push_back(o);
}

void SpecificWorker::draw_ring_points(const std::vector<QPointF> &points, const Eigen::Vector2f &result, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem *> draw_points;
    for(const auto &p : draw_points) {
        scene->removeItem(p);
        delete p;
    }
     draw_points.clear();

     for(const auto p: points)
     {
         auto o = scene->addRect(-10, 10, 20, 20, QPen(QColor("green")), QBrush(QColor("green")));
         o->setPos(p);
         draw_points.push_back(o);
     }

     float scl = 10;
     draw_points.push_back(scene->addLine(0.f, 0.f, result.x()*scl, result.y()*scl, QPen(QColor("orange"), 15)));
     auto ball = scene->addEllipse(-20, -20, 40, 40, QPen(QColor("orange"), 15));
     ball->setPos(result.x()*scl, result.y()*scl);
     draw_points.push_back(ball);
}

void SpecificWorker::draw_all_points(const RoboCompLidar3D::TPoints &points,
                                     const std::vector<std::tuple<float, float>> &discr_points,
                                     const std::vector<std::tuple<float, float>> &enlarged_points,
                                     const Eigen::Vector2f &result,  QGraphicsScene *scene)
{
    // draw draw points
    static std::vector<QGraphicsItem *> draw_points;
    for(const auto &p : draw_points)
    {
        scene->removeItem(p);
        delete p;
    }
    draw_points.clear();

    //qInfo() << __FUNCTION__ << "points" << points.size();
    for(const auto &p: points)
    {
        auto o = scene->addRect(-10, -10, 20, 20, QPen(QColor("blue")), QBrush(QColor("blue")));
        o->setPos(p.x, p.y);
        draw_points.push_back(o);
    }

    ///////////////////////////////
    /// draw discretized
    ///////////////////////////////
    static std::vector<QGraphicsItem *> draw_discr_points;
    for(const auto &p : draw_discr_points)
    {
        scene->removeItem(p);
        delete p;
    }
    draw_discr_points.clear();

    for(const auto &[ang, dist]: discr_points)
    {
        auto o = scene->addRect(-15, -15, 30, 30, QPen(QColor("green")), QBrush(QColor("green")));
        QPointF pos{-dist*sin(ang), dist*cos(ang)}; //TODO: Por qué hay que cambiar el signo de x?
        o->setPos(pos);
        draw_discr_points.push_back(o);
    }

    // draw polar contour
    static std::vector<QGraphicsItem *> draw_discr_lines;
    for(const auto &p : draw_discr_lines)
    {
        scene->removeItem(p);
        delete p;
    }
    draw_discr_lines.clear();

    for(auto &&ps  : discr_points | iter::sliding_window(2))
    {
        auto &[ang1, dist1] = ps[0];
        auto &[ang2, dist2] = ps[1];
        QLineF l{QPointF{-dist1*sin(ang1), dist1*cos(ang1)}, QPointF{-dist2*sin(ang2), dist2*cos(ang2)}};
        auto o = scene->addLine(l, QPen(QColor("green"), 5));
        draw_discr_lines.push_back(o);
    }
    // join first and last points
    {
        auto &[ang1, dist1] = discr_points.front();
        auto &[ang2, dist2] = discr_points.back();
        QLineF l{QPointF{-dist1 * sin(ang1), dist1 * cos(ang1)}, QPointF{-dist2 * sin(ang2), dist2 * cos(ang2)}};
        auto o = scene->addLine(l, QPen(QColor("green"), 5));
        draw_discr_lines.push_back(o);
    }

    ////////////////////////
    /// draw enlarged contour
    ////////////////////////
    static std::vector<QGraphicsItem *> draw_enlarged_lines;
    for(const auto &p : draw_enlarged_lines)
    {
        scene->removeItem(p);
        delete p;
    }
    draw_enlarged_lines.clear();

    for(auto &&ps  : enlarged_points | iter::sliding_window(2))
    {
        auto &[ang1, dist1] = ps[0];
        auto &[ang2, dist2] = ps[1];
        QLineF l{QPointF{-dist1*sin(ang1), dist1*cos(ang1)}, QPointF{-dist2*sin(ang2), dist2*cos(ang2)}};
        auto o = scene->addLine(l, QPen(QColor("magenta"), 5));
        draw_enlarged_lines.push_back(o);
    }
    // join first and last points
    {
        auto &[ang1, dist1] = enlarged_points.front();
        auto &[ang2, dist2] = enlarged_points.back();
        QLineF l{QPointF{-dist1 * sin(ang1), dist1 * cos(ang1)}, QPointF{-dist2 * sin(ang2), dist2 * cos(ang2)}};
        auto o = scene->addLine(l, QPen(QColor("magenta"), 5));
        draw_enlarged_lines.push_back(o);
    }

    /////////////////////////
    /// draw resultant
    /////////////////////////
    static QGraphicsItem * line = nullptr;
    if( line != nullptr)
        scene->removeItem(line);
    
    line = scene->addLine(0, 0, result.x()*50, result.y()*50, QPen(QColor("red"), 10));
}

//////////////////////////////////////////////////////////////////////////////

int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}
void SpecificWorker::self_adjust_period(int new_period)
{
    if(abs(new_period - this->Period) < 2 || new_period < 1)      // do it only if period changes
        return;

    if(new_period > this->Period)
    {
        this->Period += 1;
        timer.setInterval(this->Period);
    } else
    {
        this->Period -= 1;
        this->timer.setInterval(this->Period);
    }
}

//////////////////////////// Interfaces //////////////////////////////////////////

void SpecificWorker::OmniRobot_correctOdometer(int x, int z, float alpha)
{
//implementCODE

}
void SpecificWorker::OmniRobot_getBasePose(int &x, int &z, float &alpha)
{
//implementCODE

}
void SpecificWorker::OmniRobot_getBaseState(RoboCompGenericBase::TBaseState &state)
{
//implementCODE

}
void SpecificWorker::OmniRobot_resetOdometer()
{
//implementCODE

}
void SpecificWorker::OmniRobot_setOdometer(RoboCompGenericBase::TBaseState state)
{
//implementCODE

}
void SpecificWorker::OmniRobot_setOdometerPose(int x, int z, float alpha)
{
//implementCODE

}
void SpecificWorker::OmniRobot_setSpeedBase(float advx, float advz, float rot)
{

//     Todo: Check range of input values.
    buffer_dwa.put(std::make_tuple(advx, advz, rot));
}
void SpecificWorker::OmniRobot_stopBase()
{
//implementCODE

}

//SUBSCRIPTION to setTrack method from SegmentatorTrackingPub interface
void SpecificWorker::SegmentatorTrackingPub_setTrack(RoboCompVisualElements::TObject target)
{
//subscribesToCODE

}

/**************************************/
// From the RoboCompLidar3D you can call this methods:
// this->lidar3d_proxy->getLidarData(...)
// this->lidar3d_proxy->getLidarDataWithThreshold2d(...)

/**************************************/
// From the RoboCompLidar3D you can use this types:
// RoboCompLidar3D::TPoint
// RoboCompLidar3D::TData

/**************************************/
// From the RoboCompOmniRobot you can call this methods:
// this->omnirobot_proxy->correctOdometer(...)
// this->omnirobot_proxy->getBasePose(...)
// this->omnirobot_proxy->getBaseState(...)
// this->omnirobot_proxy->resetOdometer(...)
// this->omnirobot_proxy->setOdometer(...)
// this->omnirobot_proxy->setOdometerPose(...)
// this->omnirobot_proxy->setSpeedBase(...)
// this->omnirobot_proxy->stopBase(...)

/**************************************/
// From the RoboCompOmniRobot you can use this types:
// RoboCompOmniRobot::TMechParams

/**************************************/
// From the RoboCompOmniRobot you can use this types:
// RoboCompOmniRobot::TMechParams

////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//  TODO: Divide into methods draw_polar_graph and draw_cartesian_graph

//  Draw the Gpoints vector in cartesian coordinates angles/distance
//    for(const auto &[ang, dist] : gpoints)
//    {
//        cv::Point pt1(ang * scaleXcartesian, height - dist * scaleYcartesian);
//        cv::circle(graphCartesian, pt1, 2, cv::Scalar(0, 0, 255),-8);
//    }
// Show cartesian graph
//const float scaleXcartesian = width / 360.0;
//const float scaleYcartesian = height / max_distance;
//cv::namedWindow("GraphCartesian", cv::WINDOW_AUTOSIZE);
//cv::imshow("GraphCartesian", graphCartesian);

//void SpecificWorker::draw_histogram(const RoboCompLidar3D::TPoints &ldata)
//{
//    // Draw a white image one for each graph
//    const int width = 400, height = 400;
//    cv::Mat graphPolar = cv::Mat::zeros(height, width, CV_8UC3) + cv::Scalar(255, 255, 255);
//
//    // Max_distance of the laser used for scaling
//    const float max_distance = 4000.0; //mm
//
//    // scale factors
//
//    const float scaleX = width / (2 * max_distance);  // Asumiendo que el ángulo varía de 0 a 360
//    const float scaleY = height / (2 * max_distance);
//
//    for(auto &ps : ldata )
//    {
//        cv::Point p{static_cast<int>(width/2.0 - (ps.r * scaleX)), static_cast<int>(height/2.0 - (ps.r * scaleY))};
//        cv::circle(graphPolar, p, 3, cv::Scalar(255,0,0));
//    }
//    cv::line(graphPolar, cv::Point{0, height/2}, cv::Point{width, height/2}, cv::Scalar(0, 255, 0), 1);
//    cv::line(graphPolar, cv::Point{width/2, 0}, cv::Point{width/2, height}, cv::Scalar(0, 255, 0), 1);
//    cv::namedWindow("GraphPolar", cv::WINDOW_AUTOSIZE);
//    cv::imshow("GraphPolar", graphPolar);
//    cv::waitKey(1);
//}
//void SpecificWorker::draw_histogram(const std::vector<std::tuple<float, float>> &pdata)
//{
//    // Draw a white image one for each graph
//    const int width = 400, height = 400;
//    cv::Mat graphPolar = cv::Mat::zeros(height, width, CV_8UC3) + cv::Scalar(255, 255, 255);
//
//    // Max_distance of the laser used for scaling
//    const float max_distance = 1500.0; //mm
//
//    // scale factors
//
//    const float scaleXpolar = width / (2 * max_distance);  // Asumiendo que el ángulo varía de 0 a 360
//    const float scaleYpolar = height / (2 * max_distance);
//
//    //  Draw the Gpoints vector in polar coordinates
//    for(auto &&ps : pdata | iter::sliding_window(2))
//    {
//        const auto &[d1, ang1] = ps[0];
//        cv::Point p1{ static_cast<int>(width/2.0 - (d1 * scaleXpolar * std::sin(qDegreesToRadians((float)ang1)))),
//                      static_cast<int>(height/2.0 - (d1 * scaleYpolar * std::cos(qDegreesToRadians((float)ang1))))};
//        const auto &[d2, ang2] = ps[1];
//        cv::Point p2{ static_cast<int>(width/2.0 - (d2 * scaleXpolar * std::sin(qDegreesToRadians((float)ang2)))),
//                      static_cast<int>(height/2.0 - (d2 * scaleYpolar * std::cos(qDegreesToRadians((float)ang2))))};
//
//        cv::line(graphPolar, p1, p2, cv::Scalar(255, 0, 0), 2);
//    }
//    cv::line(graphPolar, cv::Point{0, height/2}, cv::Point{width, height/2}, cv::Scalar(0, 255, 0), 1);
//    cv::line(graphPolar, cv::Point{width/2, 0}, cv::Point{width/2, height}, cv::Scalar(0, 255, 0), 1);
//    cv::namedWindow("GraphPolar", cv::WINDOW_AUTOSIZE);
//    cv::imshow("GraphPolar", graphPolar);
//    cv::waitKey(1);
//}
