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
        //viewer->draw_contour();
        viewer->add_robot(460, 480, 0, 100, QColor("Blue"));
        viewer->show();
        std::cout << "Started viewer" << std::endl;

        // create map from degrees (0..360)  -> edge distances
         int robot_semi_width = 230;
         int robot_semi_height = 240;
        robot_contour << QPointF(-robot_semi_width,robot_semi_height) << QPointF(robot_semi_width, robot_semi_height) << QPointF(robot_semi_width, -robot_semi_height) << QPointF(-robot_semi_width, -robot_semi_height);
        robot_safe_band << QPointF(-robot_semi_width-BAND_WIDTH, robot_semi_height+BAND_WIDTH) <<
                           QPointF(robot_semi_width+BAND_WIDTH, robot_semi_height+BAND_WIDTH) <<
                           QPointF(robot_semi_width+BAND_WIDTH, -robot_semi_height-BAND_WIDTH) <<
                           QPointF(-robot_semi_width-BAND_WIDTH, -robot_semi_height-BAND_WIDTH);

		edge_points = create_edge_points();
        draw_ring(edge_points, &viewer->scene);

//        float x1 = -band.left_distance;
//        float y1 = -band.frontal_distance;
//        float width = band.left_distance + band.right_distance;
//        float height = band.frontal_distance + band.back_distance;

        //rectItem = viewer->scene.addRect(QRectF(x1, y1, width, height), QPen(QColor("Red"), 5));
        std::cout << "Robot is drawn" << std::endl;

        // A thread is created 
        read_lidar_th = std::move(std::thread(&SpecificWorker::read_lidar,this));
        std::cout << "Started lidar reader" << std::endl;

        connect(viewer, SIGNAL(new_mouse_coordinates(QPointF)), this, SLOT(new_mouse_coordinates(QPointF)));

        timer.start(50);
	}
}

void SpecificWorker::compute()
{
    /// read LiDAR
    auto res_ = buffer_lidar_data.try_get();
    if (res_.has_value() == false) {   /*qWarning() << "No data Lidar";*/ return; }
    auto ldata = res_.value();
    qInfo() << ldata.points.size();

    /// filter out floor points and inbody points
    RoboCompLidar3D::TPoints above_floor_points;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &[i, p]: ldata.points | iter::enumerate)
    {   // TODO: Check this
//        if (abs(p.x) > 300 and abs(p.y) > 300 and p.z < 1000)
            above_floor_points.emplace_back(p);
        pcl_cloud_source->push_back(pcl::PointXYZ{p.x/1000.f, p.y/1000.f, p.z/1000.f});
    }
    if (above_floor_points.empty())
    {  qInfo() << "empty vector"; return; }

    /// compute odometry
    auto robot_pose = fastgicp.align(pcl_cloud_source);
    //QPointF robot_pos(traj->back().x*1000, traj->back().y*1000);
    QPointF robot_tr(robot_pose(0, 3)*1000, robot_pose(1, 3)*1000);

    /// compute free space polar representation
    auto discr_points = discretize_lidar(above_floor_points);
    auto enlarged_points = configuration_space(discr_points);
    //auto blocks_and_tagged_points = get_blocks(enlarged_points);
    //auto sblocks = set_blocks_symbol(blocks_and_tagged_points.first);

    /// Check for new external target
    if(const auto ext = buffer_dwa.try_get(); ext.has_value())
    {
        const auto &[side, adv, rot, debug] = ext.value();
        target_ext.set(side, adv, rot, true);
//        target_ext = get_closest_point_inside(enlarged_points, target_ext);
        fastgicp.reset();
        target_original = target_ext;
        draw_target_original(target_ext);
    }

    /// Check for debug target (later for LOST target)
    if(target_ext.active and target_ext.debug)
    {
        // Update odometry
        Eigen::Vector4d target_in_robot =
                robot_pose.inverse().matrix() * Eigen::Vector4d(target_original.x / 1000.f, target_original.y / 1000.f, 0.f, 1.f) * 1000;
        target_ext.set(target_in_robot(0)/2, target_in_robot(1)/2, 0, true);

        // Check if the robot is at target
        qInfo() << __FUNCTION__ << "Dist to target:" << target_in_robot.norm();
        if (target_in_robot.norm() < 600)
        {
            stop_robot("Robot arrived to target");
            target_ext.debug = false;
            target_ext.active = false;
            draw_target(target, true);
        }
    }

    /// Check bumper for a security breach
    std::vector<Eigen::Vector2f> displacements = check_safety(enlarged_points);
    //std::cout << "displacement_size" << displacements.size() << std::endl;

    draw_displacements(displacements,&viewer->scene);
    bool security_breach = not displacements.empty();
//    security_breach = false;
    if(not security_breach) draw_target_breach(target, true);

    //////////////////////////////////
    /// We have now four possibilities
    //////////////////////////////////
    if(target_ext.active and security_breach)    // target active. Choose displacement best aligned with target
    {
        std::cout << "1"<<std::endl;
        if (not displacements.empty())
        {
            std::cout << "2"<<std::endl;
            auto res = std::ranges::max(displacements, [t = target_ext](auto &a, auto &b)
            {
                auto tv = t.eigen().transpose();
                return tv * a < tv * b; //return closest displacement to target using scalar product. (-1,1) 1=target aligned
            });
            target.set(res.x(), res.y(), 0.f);
        }
    }
    if(not security_breach and not target_ext.active)
    {
       stop_robot("No target, no breach");
    }
    if(not target_ext.active and security_breach) // choose displacement that maximizes sum of distances to obstacles
    {
        std::cout << "3"<<std::endl;
       if (not displacements.empty())
       {
            std::cout << "4"<<std::endl;
//           auto res = std::ranges::max(displacements,
//                                       [](auto &a, auto &b) { return a.second < b.second; });

            target.set(displacements[0].x(), displacements[0].y(), 0.f);
            draw_target_breach(target);
       } else  {  stop_robot("Collision but no solution found");  }
    }
    if(target_ext.active and not security_breach)
    {

        std::cout << "5"<<std::endl;
       target = target_ext;
    }

    if(target.active)
    {
        std::cout << "6"<<std::endl;
        draw_target(target_ext);
        target.print("FINAL");
        float adv = target.y;
        float side = target.x;
        float rot = atan2( target.x, target.y);  // dumps rotation for small resultant force;
        robot_current_speed = {adv, side};
        try
        {
            std::cout << "7"<<std::endl;
//            omnirobot_proxy->setSpeedBase(adv , -side , -rot);
            robot_stopped = false;
            target.active = false;
        }
        catch (const Ice::Exception &e)
        { std::cout << "Error talking to OmniRobot " << e.what() << std::endl; }
    }

    //////////////////////////// draw
    draw_discr_points(discr_points,&viewer->scene);
    draw_enlarged_points(enlarged_points,&viewer->scene);
    // fps.print("FPS:");
}

std::vector<std::tuple<float, float>> SpecificWorker::discretize_lidar(const RoboCompLidar3D::TPoints &ldata)
{
    std::vector<std::tuple<float, float>> polar_points;
    const float delta_phi = 2*(M_PI*2/360); // number of degrees
    float running_angle = -M_PI + delta_phi;
    float running_min = std::numeric_limits<float>::max();

    // Group points by discrete angle bins and compute the min dist of the set
    for (const auto& p : ldata)
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

    // complete the circle
    while(running_angle < M_PI)
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
        conf_space.emplace_back(ang_i, std::ranges::min(dij_vec) - R*2);
    }
    return conf_space;
}
std::pair<std::vector<SpecificWorker::Block>, std::vector<SpecificWorker::LPoint>>
SpecificWorker::get_blocks(const std::vector<std::tuple<float, float>> &enlarged_points)
{
    auto angle_diff = [](auto a, auto b){ return atan2(sin(a - b), cos(a - b));};
    Block block{std::get<0>(enlarged_points[0]),std::get<1>(enlarged_points[0])};
    std::vector<Block> blocks;
    std::vector<LPoint> tagged_enlarged_points;
    std::vector<std::tuple<float, float>> aux(enlarged_points.begin()+1,enlarged_points.end());
    aux.push_back(enlarged_points.back());

    int i=0;
    for (auto &&p: aux | iter::sliding_window(2))
    {
        auto &[ang,dist] = p[0];
        auto &[ang1,dist1] = p[1];
        float beta = angle_diff(ang,block.A_ang);
        float diff_dist = std::sqrt(block.A_dist * block.A_dist + dist1 * dist1 - 2 * cos(beta) * block.A_dist * dist1);

        if( diff_dist < R * 1.1)
        {
            block.B_ang = ang1;
            block.B_dist = dist1;
            tagged_enlarged_points.emplace_back(LPoint{.ang=ang1, .dist=dist1, .block=i});
        }
        else
        {
            blocks.push_back(block);
            tagged_enlarged_points.emplace_back(LPoint{.ang=ang1, .dist=dist1, .block=++i});
            block.A_ang = ang1;
            block.A_dist = dist1;
            block.B_ang = ang1;
            block.B_dist = dist1;
        }
    }

    // Check if last point of the last block if connected to the first option of that same block

    //qInfo() << blocks.size() << aux.size();
    return {blocks, tagged_enlarged_points};
}
vector<SpecificWorker::Block> SpecificWorker::set_blocks_symbol(const std::vector<Block> &blocks)
{
    std::vector<Block> sblocks(blocks);
    auto gt_than = [](auto &b0, auto &b1, auto &b2)
            { return b1.A_dist >= b0.B_dist and b1.B_dist >= b2.A_dist;};

    for(auto &sb : sblocks | iter::sliding_window(3))
        if( gt_than(sb[0], sb[1], sb[2]) )
            sb[1].concave = false;

    if( gt_than(sblocks[sblocks.size()-2],sblocks.back(), sblocks.front()))
        sblocks.back().concave = false;

    if( gt_than(sblocks.back(),sblocks.front(),  sblocks[2]))
        sblocks.front().concave = false;

    return sblocks;
}

SpecificWorker::Target SpecificWorker::get_closest_point_inside(const std::vector<std::tuple<float, float>> &points, const Target &target)
{
    Target t = target;
    if(auto r =  std::upper_bound(points.cbegin(), points.cend(), target.ang,
                     [](float value, std::tuple<float, float> p){ return std::get<0>(p) > value;}); r != points.end())
        if(std::get<1>(*r) < target.dist)
            t.set(std::get<0>(*r), std::get<1>(*r));
    return t;

}
//////////////////////////////////////////////////////////////////////////////

std::vector<Eigen::Vector2f> SpecificWorker::check_safety(const std::vector<std::tuple<float, float>> &points)
{
  //compute reachable positions at max acc that sets the robot free. Choose the one closest to target or of minimum length if not target.
  // get lidar points closer than safety band. If none return
  // TODO: AVOID FORCE INVERSION
  std::vector<Eigen::Vector2f> close_points;
  std::vector<Eigen::Vector2f> displacements, final_displacement;

  // lambda to compute if a point is inside the belt
  auto point_in_belt = [this](auto ang, auto dist)
          {
              auto r = std::upper_bound(edge_points.cbegin(), edge_points.cend(), ang,
                                        [](float value, std::tuple<float, float> p) { return std::get<0>(p) >= value; });
              if (r != edge_points.end() and dist < std::get<1>(*r) and dist > (std::get<1>(*r) - this->BAND_WIDTH) ) return true;
              else return false;
          };

    // gather all points inside safety belt
    for(const auto &[ang, dist]: points)
      if(point_in_belt(ang, dist)) close_points.emplace_back(dist*sin(ang), dist*cos(ang));

  // max dist to reach from current situation
  const float delta_t = 1; // 0.050; //200ms
  const float MAX_ACC = 200; // mm/sg2
//  double dist = robot_current_speed.norm() * delta_t + (MAX_ACC * delta_t * delta_t);
  double dist = (MAX_ACC * delta_t * delta_t);

    //qInfo() << "Dist" << dist << robot_current_speed.norm();

  if(not close_points.empty())
  {
      for (const double ang: iter::range(-M_PI, M_PI, BELT_ANGULAR_STEP))
      {
          // compute coor of conflict points after moving the robot to new pos (d*sin(ang), d*cos(ang))
          Eigen::Vector2f t{dist * sin(ang), dist * cos(ang)};
          bool free = true;
          for (const auto &p: close_points)
          {
              float dist_p = (p - t).norm();
              qInfo() << "COORDS" << dist << ang << t.x() << t.y() << dist_p ;
              if(point_in_belt(atan2((p-t).x(),(p-t).y()), dist_p))
              {
                  free = false;
                  break;
              }
          }
          if (free)
              displacements.emplace_back(t); // add displacement to list
      }

      for (const auto &d : displacements)
      {
          bool success = true;
          for(const auto &[ang, dist]: points)
          {
              Eigen::Vector2f p_cart{dist*sin(ang), dist*cos(ang)};
              auto traslated = p_cart - d;

              if(point_in_belt(atan2(traslated.x(),traslated.y()), traslated.norm())){
                  success = false;
                  break;
              }
          }
          if (success)
              final_displacement.emplace_back(d);
      }
      return final_displacement;
  }
  else return {};
}


bool SpecificWorker::inside_contour(const Target &target, const std::vector<std::tuple<float, float>> &contour)
{
    auto r =  std::upper_bound(contour.cbegin(), contour.cend(), target.ang,
                             [](float value, std::tuple<float, float> p){ return std::get<0>(p) > value;});
    //qInfo()<< (*r).ang << (*r).dist << target.ang << target.dist;
    if(r != contour.end() and std::get<1>(*r) > target.dist)
        return true;
    else
        return false;
}
void SpecificWorker::repulsion_force(const RoboCompLidar3D::TData &ldata)
{
//    Eigen::Vector2f result{0.f, 0.f};
//    std::vector<QPointF> draw_points;
//
//    for (const auto &p: ldata.points)
//    {
//        float ang = atan2(p.x, p.y);
//        int index = qRadiansToDegrees(ang);
//
//        if (index < 0) index += 360;
//
//        float norma = std::sqrt(p.x * p.x + p.y * p.y);
//        float diff = edge_points.at(index) - norma;// negative vals are inside belt
//
//        //Check that p is greater than robot body and smaller than robot body + BAND_WIDTH
//        if (diff < 0 and diff > -BAND_WIDTH)
//        {
//            // something is inside the perimeter
//            // pseudo sigmoid
//            auto diff_normalized = fabs(diff) * (1.f / BAND_WIDTH);
//            //std::cout << "Diff_normalized:" << diff_normalized << std::endl;
//            //float modulus = std::clamp(fabs(diff)*(1.f/BAND_WIDTH), 0.f, 1.f);
//            float modulus = (-1 / (1 + exp((diff_normalized - 0.5) * 12))) + 1;
//            //std::cout << "Modulus:" << modulus << std::endl;
//            //std::cout << "Distance:" << p.head(2).normalized() << std::endl;
//            //result -= p.head(2).normalized() * modulus;     // opposite direction and normalized modulus
//            result -= Eigen::Vector2f{p.x / norma, p.y / norma} *
//                      modulus;     // opposite direction and normalized modulus
//
//            //qInfo() << "diff" << diff << modulus << result.x() << result.y();
//            draw_points.emplace_back(p.x, p.y);
//        }
//    }
}
SpecificWorker::Band SpecificWorker::adjustSafetyZone(Eigen::Vector3f velocity)
{
    Band adjusted;
    float safety_distance = 200;

    // Si x es positivo, aumentamos la distancia frontal y disminuimos la trasera.
    // Si x es negativo, hacemos lo contrario.
    if (velocity.x() >= 0.0f)
    {
        adjusted.right_distance = std::max(velocity.x() / 1000,safety_distance);
        adjusted.left_distance = -safety_distance;
    } else //if(velocity.x() <= -600.0f)
    {
        adjusted.right_distance = safety_distance;
        adjusted.left_distance = std::min(velocity.x() / 1000,-safety_distance);
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
        adjusted.frontal_distance = std::max(velocity.y() / 1000,safety_distance);
        adjusted.back_distance = -safety_distance;
    } else //if(velocity.y() <= -600.0f)
    {
        adjusted.frontal_distance = safety_distance;
        adjusted.back_distance = std::min(velocity.y() / 1000,-safety_distance);
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
            filteredPoints.push_back(p);
    }

    return filteredPoints;
}
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

//            auto data = lidar3d_proxy->getLidarDataWithThreshold2d("bpearl", 10000);
            auto data = lidar3d_proxy->getLidarData("bpearl", -90, 360, 1);
            //std::cout << data.points.size() << " " << data.period << std::endl;
            //std::ranges::for_each(data.points, [](auto& c){c.phi = -c.phi; return c;});
            std::ranges::sort(data.points, {}, &RoboCompLidar3D::TPoint::phi);
            buffer_lidar_data.put(std::move(data));
        }
        catch (const Ice::Exception &e) { std::cout << "Error reading from Lidar3D" << e << std::endl; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));}
}
std::vector<std::tuple<float,float>> SpecificWorker::create_edge_points()
{
	std::vector<std::tuple<float, float>> dists;
	for (const double ang: iter::range(-M_PI, M_PI, BELT_ANGULAR_STEP))
        {
		bool found = false;
        // iter from 0 to OUTER_RIG_DISTANCE until the point falls outside the polygon
		for(const int r : iter::range(OUTER_RIG_DISTANCE))
		{
			double x = r * sin(ang);
			double y = r * cos(ang);
            if( not robot_safe_band.containsPoint(QPointF(x, y), Qt::OddEvenFill))
			{
				dists.emplace_back(ang, Eigen::Vector2f(x, y).norm());
				found = true;
				break;
			}
		}
		if(not found) { qFatal("ERROR: Could not find limit for angle ");	}
	}
    return dists;
}
void SpecificWorker::stop_robot(const std::string_view txt)
{
    if(not robot_stopped)
    {
        target.active = false;
        try
        {
            omnirobot_proxy->setSpeedBase(0, 0, 0);
            qInfo() << __FUNCTION__ << "Robot stopped";
            draw_target(target, true);
            robot_current_speed = {0.f, 0.f};
        }
        catch (const Ice::Exception &e)
        { std::cout << "Error talking to OmniRobot " << e.what() << std::endl; }
        robot_stopped = true;
        std::cout << "Robot stopped due to " << txt << std::endl;
    }
};
///////////////////////////////////////////////////////////////////////////
void SpecificWorker::draw_ring(const std::vector<std::tuple<float, float>> &dists, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem *> draw_points;
    for (const auto &p: draw_points) {
        scene->removeItem(p);
        delete p;
    }
    draw_points.clear();

    QPolygonF poly;
    for (const auto &[ang, dist]: dists)
        poly << QPointF(dist * sin(ang), dist * cos(ang));

    auto o = scene->addPolygon(poly, QPen(QColor("DarkBlue"), 10));
    draw_points.push_back(o);

    // draw external ring
    //    o = scene->addPolygon(robot_safe_band, QPen(QColor("DarkBlue"), 10));
    //    draw_points.push_back(o);
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
void SpecificWorker::draw_discr_points(const std::vector<std::tuple<float, float>> &discr_points, QGraphicsScene *scene)
{
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
        QPointF pos{dist*sin(ang), dist*cos(ang)};
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
        QLineF l{QPointF{dist1*sin(ang1), dist1*cos(ang1)}, QPointF{dist2*sin(ang2), dist2*cos(ang2)}};
        auto o = scene->addLine(l, QPen(QColor("green"), 5));
        draw_discr_lines.push_back(o);
    }
    // join first and last points
    {
        auto &[ang1, dist1] = discr_points.front();
        auto &[ang2, dist2] = discr_points.back();
        QLineF l{QPointF{dist1 * sin(ang1), dist1 * cos(ang1)}, QPointF{dist2 * sin(ang2), dist2 * cos(ang2)}};
        auto o = scene->addLine(l, QPen(QColor("green"), 10));
        draw_discr_lines.push_back(o);
    }
}
void SpecificWorker::draw_enlarged_points(const std::vector<std::tuple<float, float>> &enlarged_points, QGraphicsScene *scene)
{
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

        QLineF l{QPointF{dist1*sin(ang1), dist1*cos(ang1)}, QPointF{dist2*sin(ang2), dist2*cos(ang2)}};
        auto o = scene->addLine(l, QPen(QColor("magenta"), 5));
        draw_enlarged_lines.push_back(o);
    }
    // join first and last points
    {
        auto &[ang1, dist1] = enlarged_points.front();
        auto &[ang2, dist2] = enlarged_points.back();
        QLineF l{QPointF{dist1 * sin(ang1), dist1 * cos(ang1)}, QPointF{dist2 * sin(ang2), dist2 * cos(ang2)}};
        auto o = scene->addLine(l, QPen(QColor("magenta"), 15));
        draw_enlarged_lines.push_back(o);
    }
}
void SpecificWorker::draw_blocks(const std::vector<Block> &blocks, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem *> draw_blocks_lines;
    for(const auto &p : draw_blocks_lines)
    {
        scene->removeItem(p);
        delete p;
    }
    draw_blocks_lines.clear();

    QColor concave_color("orange");
    QColor convex_color("black");
    for( auto &&b : blocks)
    {
        QLineF l{QPointF{b.A_dist * sin(b.A_ang), b.A_dist * cos(b.A_ang)}, QPointF{b.B_dist * sin(b.B_ang), b.B_dist * cos(b.B_ang)}};
        QGraphicsLineItem *o;
        if( b.concave)
            o = scene->addLine(l, QPen(concave_color, 20));
        else
            o = scene->addLine(l, QPen(convex_color, 20));
        draw_blocks_lines.push_back(o);
    }
}
void SpecificWorker::draw_result(const LPoint &res)
{
    static QGraphicsItem * o = nullptr;
    static QGraphicsItem * l = nullptr;
    if( o != nullptr) viewer->scene.removeItem(o);
    if( l != nullptr) viewer->scene.removeItem(l);
    l =  viewer->scene.addLine(0, 0, res.dist*sin(res.ang), res.dist*cos(res.ang), QPen(QColor("red"), 15));
    o = viewer->scene.addRect(-60, -60, 120, 120, QPen(QColor("red")), QBrush(QColor("red")));
    QPointF pos{res.dist*sin(res.ang), res.dist*cos(res.ang)};
    o->setPos(pos);
}
void SpecificWorker::draw_target(const Target &t, bool erase)
{
    static QGraphicsItem * line = nullptr;
    static QGraphicsItem * ball = nullptr;
    if( line != nullptr) { viewer->scene.removeItem(line); line = nullptr; }
    if( ball != nullptr) { viewer->scene.removeItem(ball); ball = nullptr; }
    if(not erase)
    {
        line = viewer->scene.addLine(0, 0, t.x, t.y, QPen(QColor("green"), 20));
        ball = viewer->scene.addEllipse(-20, -20, 40, 40, QPen(QColor("green"), 20));
        ball->setPos(t.x, t.y);
    }
}
void SpecificWorker::draw_target(double x, double y, bool erase)
{
    static QGraphicsItem * line = nullptr;
    static QGraphicsItem * ball = nullptr;
    if( line != nullptr) { viewer->scene.removeItem(line); line = nullptr;}
    if( ball != nullptr) { viewer->scene.removeItem(ball); ball = nullptr;}
    if(not erase)
    {
        line = viewer->scene.addLine(0, 0, x, y, QPen(QColor("green"), 20));
        ball = viewer->scene.addEllipse(-20, -20, 40, 40, QPen(QColor("green"), 20));
        ball->setPos(x, y);
    }
}
void SpecificWorker::draw_target_original(const Target &t, bool erase)
{
    static QGraphicsItem * line = nullptr;
    static QGraphicsItem * ball = nullptr;
    if( line != nullptr) { viewer->scene.removeItem(line); line = nullptr;}
    if( ball != nullptr) { viewer->scene.removeItem(ball); ball = nullptr;}
    if(not erase)
    {
        //qInfo() << __FUNCTION__ << t.x << t.y;
        line = viewer->scene.addLine(0, 0, t.x, t.y, QPen(QColor("blue"), 20));
        ball = viewer->scene.addEllipse(-20, -20, 40, 40, QPen(QColor("blue"), 20));
        ball->setPos(t.x, t.y);
    }
}
void SpecificWorker::draw_target_breach(const Target &t, bool erase)
{
    static QGraphicsItem *line=nullptr, *ball=nullptr;
    if( line != nullptr) { viewer->scene.removeItem(line); line = nullptr; }
    if( ball != nullptr) { viewer->scene.removeItem(ball); ball = nullptr; }
    if(not erase)
    {
        line = viewer->scene.addLine(0, 0, t.x, t.y, QPen(QColor("magenta"), 20));
        ball = viewer->scene.addEllipse(-30, -30, 60, 60, QPen(QColor("magenta"), 20));;
        ball->setPos(t.x, t.y);
    }
}

void SpecificWorker::draw_displacements(std::vector<Eigen::Matrix<float, 2, 1>> displacement_points, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem *> draw_displacements;
    for(const auto &p : draw_displacements)
    {
        scene->removeItem(p);
        delete p;
    }
    draw_displacements.clear();

    for(auto &d  : displacement_points)
    {
        d *= 5;
        QLineF l{0,0, d.x(),d.y()};
        auto o = scene->addLine(l, QPen(QColor("blue"), 8));
        auto o_p = scene->addEllipse(-50,-50,100,100 , QPen(QColor("blue"), 8));
        o_p->setPos(d.x(),d.y());
        draw_displacements.push_back(o);
        draw_displacements.push_back(o_p);
    }

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
void SpecificWorker::GridPlanner_setPlan(RoboCompGridPlanner::TPlan plan)
{
    if(plan.valid) // Possible failure variable
    {
        buffer_dwa.put(std::make_tuple(plan.subtarget.x, plan.subtarget.y, 0, false));
    }
}
void SpecificWorker::new_mouse_coordinates(QPointF p)
{
    buffer_dwa.put(std::make_tuple(p.x(), p.y(), 0, true)); // for debug
}
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

//#if DEBUG
//    auto start = std::chrono::high_resolution_clock::now();
//#endif
//#if DEBUG
//        qInfo() << "Post get_lidar_data" << (std::chrono::duration<double, std::milli> (std::chrono::high_resolution_clock::now() - start)).count();
//        start = std::chrono::high_resolution_clock::now();
//    #endif
//    #if DEBUG
//        qInfo() << "Post result" << (std::chrono::duration<double, std::milli> (std::chrono::high_resolution_clock::now() - start)).count();
//        start = std::chrono::high_resolution_clock::now();
//    #endif
//    #if DEBUG
//        qInfo() << "Post sending adv, side, rot" << (std::chrono::duration<double, std::milli> (std::chrono::high_resolution_clock::now() - start)).count();
//        start = std::chrono::high_resolution_clock::now();
//    #endif
//    #if DEBUG
//        qInfo() << "Post draw_all_points" << (std::chrono::duration<double, std::milli> (std::chrono::high_resolution_clock::now() - start)).count();
//        qInfo() << "";
//    #endif

//std::tuple<float, float> SpecificWorker::cost_function(const std::vector<std::tuple<float, float>> &points, const Target &target)
//{
//    std::vector<std::tuple<float, float>> p_costs(points);
//    auto angle_diff = [](auto a, auto b){ return atan2(sin(a - b), cos(a - b));};
//    const float k1 = 10.f; const float k2 = 1.f; const float k3 = 0.001;
//
//    for(auto &[ang, dist] : p_costs)
//    {
//        float hg = angle_diff(p.ang, target.ang);
//        float ho = fabs(p.ang);
//        //Eigen::Vector2f d{p.dist*sin(p.ang), p.dist*cos(p.ang)};
//        //float proy = (robot_current_speed.transpose() * d ) ;
//        //proy = proy / d.norm();
//        //if(proy < d.norm())
//        //if(not blocks[p.block].concave)
//        {
//            //p.coste = (blocks[p.block].dist() / ( k1 * hg + k2 * ho + k3));
//            p.coste = (p.dist / ( k1 * hg + k2 * ho + k3));
//        }
//    }
//    LPoint max_point = std::ranges::max_element(p_costs, [](auto &a, auto &b){ return a.coste > b.coste;}).operator*();
//    max_point.dist *= 0.8;  // to avoid being on the border
//    return  max_point;
//}
