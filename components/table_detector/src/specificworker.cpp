/*
 *    Copyright (C) 2024 by YOUR NAME HERE
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
#include "pch.h"
#include "dbscan.h"

#include <atomic>

std::atomic<bool> SpecificWorker::stop_flag{false};  // will stop all instances of actionable_thing

/**
* \brief Default constructor
*/
SpecificWorker::SpecificWorker(TuplePrx tprx, bool startup_check) : GenericWorker(tprx)
{
    std::locale::global(std::locale("C"));
    this->startup_check_flag = startup_check;
    // Register signal handler
    std::signal(SIGINT, SpecificWorker::signal_handler);
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
    return true;
}
void SpecificWorker::initialize(int period)
{
    std::cout << "Initialize worker" << std::endl;
    if(this->startup_check_flag)
    {
        this->startup_check();
    }
    else
    {
        // Viewer
        viewer = new AbstractGraphicViewer(this->frame_local, params.GRID_MAX_DIM);
        auto [r, e] = viewer->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, QColor("Blue"));
        robot_draw = r;
        viewer_global = new AbstractGraphicViewer(this->frame_global, params.GRID_MAX_DIM);
        auto [rg, eg] = viewer_global->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, QColor("Blue"));
        robot_draw_global = rg;
        this->resize(900, 600);
        viewer->show(); viewer_global->show();

        // Actionables
        initializer_actionable.initialize(rc::ActionableOrigin(), std::make_pair(rg, eg));

        inner_model = std::make_shared<rc::ActionablesData>();

        // 3D
        viewer3D = new rc::Viewer3D(this->frame_3d, inner_model);
        viewer3D->show();
        connect(this, SIGNAL(updateRobotTransform(Eigen::Affine2d)), viewer3D, SLOT(updateRobotTransform(Eigen::Affine2d)));
        connect(this, SIGNAL(createRoomTransform()), viewer3D, SLOT(createRoomTransform()));
        connect(this, SIGNAL(createTableTransform()), viewer3D, SLOT(createTableTransform()));
        connect(this, SIGNAL(updateTableTransform()), viewer3D, SLOT(updateTableTransform()));

        //plot
        plot = new QCustomPlot(frame_plot);
        // Adapt the plot to the frame
        plot->resize(frame_plot->size());
        plot->addGraph();
        plot->graph(0)->setPen(QPen(QColor(0, 0, 255)));
        plot->addGraph();
        plot->graph(1)->setPen(QPen(QColor(255, 0, 0)));
        plot->xAxis->setLabel("Time");
        plot->yAxis->setLabel("Error/Trace");
        plot->xAxis->setRange(0, params.MAX_DIST_POINTS_TO_SHOW);
        plot->yAxis->setRange(-1, 2);
        plot->replot();
        plot->show();

        // reset program. Connect pushButton_stop to restart the application
        connect(pushButton_restart, &QPushButton::clicked, []()
                {
                    QProcess::startDetached(QApplication::applicationFilePath(), QStringList());
                    QApplication::quit();
                });


        // UI connections
        connect(slider_points_cost, &QSlider::valueChanged, [this](auto v){ lcdNumber_points->display((static_cast<float>(v/params.div_value_slider)));});
        slider_points_cost->setValue(params.div_value_slider * rc::ActionableThing::params.POINTS_SIGMA);

        connect(slider_cx_prior, &QSlider::valueChanged, [this](auto v){ lcdNumber_cx_prior->display((static_cast<float>(v/params.div_value_slider)));});
        slider_cx_prior->setValue(params.div_value_slider * rc::ActionableThing::params.PRIOR_CX_SIGMA);
        connect(slider_cy_prior, &QSlider::valueChanged, [this](auto v){ lcdNumber_cy_prior->display((static_cast<float>(v/params.div_value_slider)));});
        slider_cy_prior->setValue(params.div_value_slider * rc::ActionableThing::params.PRIOR_CY_SIGMA);
        connect(slider_alpha_prior, &QSlider::valueChanged, [this](auto v){ lcdNumber_alpha_prior->display((static_cast<float>(v/params.div_value_slider)));});
        slider_alpha_prior->setValue(params.div_value_slider * rc::ActionableThing::params.PRIOR_ALPHA_SIGMA);
        connect(slider_width_prior, &QSlider::valueChanged, [this](auto v){ lcdNumber_width_prior->display((static_cast<float>(v/params.div_value_slider)));});
        slider_width_prior->setValue(params.div_value_slider * rc::ActionableThing::params.PRIOR_WIDTH_SIGMA);
        connect(slider_depth_prior, &QSlider::valueChanged, [this](auto v){ lcdNumber_depth_prior->display((static_cast<float>(v/params.div_value_slider)));});
        slider_depth_prior->setValue(params.div_value_slider * rc::ActionableThing::params.PRIOR_DEPTH_SIGMA);
        connect(slider_height_prior, &QSlider::valueChanged, [this](auto v){ lcdNumber_height_prior->display((static_cast<float>(v/params.div_value_slider)));});
        slider_depth_prior->setValue(params.div_value_slider * rc::ActionableThing::params.PRIOR_HEIGHT_SIGMA);

        try{ lidarodometry_proxy->reset();}
        catch (const Ice::Exception &e){std::cout << e << " No connection to Odometry" << std::endl; return;};

        this->setPeriod( 100);
        //this->setPeriod(STATES::Emergency, 500);
    }
}
void SpecificWorker::compute()
{
    const auto [lidar_timestamp, helios_points] = read_lidar_helios();
    if(helios_points.empty()) { qWarning() << __FUNCTION__ << "Empty helios lidar data"; return; };
    draw_lidar(helios_points, &viewer->scene);



        actionable_units(helios_points, lidar_timestamp - 100);


    // thread alternative: the actionables communicate through the (thread safe) internal model
    // init room_actionable thread
        // initialise until first room is detected
        // project corners and update robot pose
    // init fridge_actionable thread (waits until the room is available)
        // initialise from residuals
            // activate affordances
            // if confirmed, update internal model
        // project existing instances and contribute factors to robot pose


    /// check for affordances

    fps.print("");

}

//////////////////////////////////////////////////////////////////
/// MAIN LOGIC
//////////////////////////////////////////////////////////////////
void SpecificWorker::actionable_units(const LidarPoints &points, long lidar_timestamp)
{
    /// All this, until thin_actionables can go in a separate threaded class

    // Get measured corners
    const auto &[_, __, corners, ___] = rc::Room_Detector::compute_features(points, &viewer->scene);
    Target target{0.0, 0.0, 0.0};

    // Update actionable_origin
    if (room_actionables.empty())
    {
        // get robot pose from lidar odometry
        Eigen::Affine2d robot_pose_in_origin;
        try { robot_pose_in_origin = affine3d_to_2d(lidarodometry_proxy->getPoseAndChange().pose);}
        catch (const Ice::Exception &e){std::cout << e << " No connection to Odometry" << std::endl; return;};

        initializer_actionable.update_robot_pose(robot_pose_in_origin);
        target = initializer_actionable.update_target(points);
        initializer_actionable.project_corners(corners, lidar_timestamp);
        if ( auto valid_candidate = check_new_candidate(initializer_actionable); valid_candidate.has_value())
        {
            inner_model->room = std::make_shared<rc::Room>(valid_candidate.value().room);  // to test internal_model
            room_actionables.emplace_back(valid_candidate.value());
            robot_draw_global->hide();
            qInfo() << __FUNCTION__ << "Emitted createRoomTransform";
            emit createRoomTransform();
        }
        return;
    }

    /////////////////////////////
    // process room actionables
    ////////////////////////////
    for (const auto &[i, act]: room_actionables | iter::enumerate)
        //for (const auto &[i, par]: internal_model | iter::filter([](auto &a){return a.first == State::ROOM;}) | iter::enumerate)
    {
        //auto act= par.second;
        auto match = act.project_corners(corners, lidar_timestamp);
        //act.compute_prediction_fitness(); // 1 -> perfect prediction

        // draw measured corners projected on room frame
        std::vector<Eigen::Vector3d> corners_projected_on_room;
        std::ranges::transform(match, std::back_inserter(corners_projected_on_room), [&](const auto &p)
                { return std::get<1>(p);});
        draw_corners(corners_projected_on_room ,&viewer_global->scene, act.get_actionable_color(), act.get_value(), i);
        draw_room(act.get_corners_3d(), act.get_actionable_color(), &viewer_global->scene, 100, i);
        draw_robots_global(act.get_robot_draw(), act.get_robot_pose());
        emit updateRobotTransform(act.get_robot_pose());

        //qDebug() << "Room data: " << i << act.get_room().get_depth() << act.get_room().get_width() << act.get_room().get_rotation();

        // procreation
        // if (act.get_value() < 0.3)
        //     if ( const auto valid_candidate = check_new_candidate(act); valid_candidate.has_value())
        //         actionables.emplace_back(valid_candidate.value());
    }

    /// Select best actionable based on energy
    const auto best_actionable = std::ranges::max_element(room_actionables,[](const auto &a, const auto &b)
                            //{ return a.get_prediction_fitness() < b.get_prediction_fitness(); });
                            { return a.get_value() < b.get_value(); });

    // plot
    //lcdNumber_best_fitness->display(best_actionable->get_value());
    //plot_multiple_free_energy(actionables);

    // remove actionables with low fitness
    // std::erase_if(actionables, [this](const auto &act)
    //     {   static int i=-1; i++;
    //         if (act.get_value() < 0.1)
    //         {
    //             viewer_global->scene.removeItem(act.get_robot_draw());
    //             delete act.get_robot_draw();
    //             plot->clearGraphs();
    //             plot->xAxis->rescale(true);
    //             plot->replot();
    //             return true;
    //         }
    //         return false; ;
    //     });


    if(room_actionables.size() <= 0) // if no room, return.
        return;

    // Filter by model from room
    std::vector<Eigen::Vector3d> room_filtered_points;
    std::ranges::copy_if(points, std::back_inserter(room_filtered_points), [best_actionable, MD = params.MIN_DIST_TO_WALL](auto &p)
        {
            auto room_polygon = best_actionable->get_room().get_qt_polygon();
            // check if the point is inside the room
//            if(not room_polygon.containsPoint(QPointF(p.x(), p.y()), Qt::FillRule::OddEvenFill))
//                return false;
            const auto lines = best_actionable->get_room().get_room_lines_eigen();
            for(const auto &l : lines)
                if(l.distance((best_actionable->get_robot_pose_3d() * p).head(2)) < MD)
                    return false;
            return true;
        });


    // transform room_filtered_points to the room frame
    std::vector<Eigen::Vector3d> room_filtered_points_in_room_frame;
    std::ranges::transform(room_filtered_points, std::back_inserter(room_filtered_points_in_room_frame), [best_actionable](auto &p)
        {
          auto t_pt =  best_actionable->get_robot_pose_3d() * p;
          return Eigen::Vector3d(t_pt.x() * 0.001, t_pt.y() * 0.001, t_pt.z() * 0.001);
        });


    /////////////////////////////////////////////////////////////////////////////////
    // Process fridge actionables. We need from above: room_filtered_points and current room (the parent actionable)
    //////////////////////////////////////////////////////////////////////////////////
    LidarPoints thing_filtered_points(room_filtered_points_in_room_frame);

    // cluster points with dbscan and filter out the biggest cluster, for now. They come sorted by size
    const auto clusters = rc::dbscan_get_point_clusters(thing_filtered_points, 0.1, 20); //meters
    if(clusters.empty())
    { qWarning() << __FUNCTION__ << "No clusters found for thing_filtered_points" ; return;}
    thing_filtered_points = clusters.begin()->second;

    static bool first_time = true;

    if(first_time){


    // process the actionables making them project the points
    for (const auto &[i, act]: thing_actionables | iter::enumerate)
    {
        thing_filtered_points = act->project(thing_filtered_points, *best_actionable, &viewer_global->scene);
        this->reset_optimiser = false;  // reset button
        this->clean_buffer = false; // ui button
        emit updateTableTransform();  // signal a change in the fridge. Should depend on being real changes.
        plot_free_energy(act->get_error(), act->get_traza()*params.TRACE_SCALING_FACTOR);
        //std::cout << "Param in SpecificWorker: " << act->get_table().transpose() << std::endl;
        //std::cout << "Param in SpecificWorker: " << act->get_table().transpose() << std::endl;
        lcdNumber_cx_current->display(act->get_table().x());
        lcdNumber_cy_current->display(act->get_table().y());
        lcdNumber_cz_current->display(act->get_table().z());
        lcdNumber_angle_current->display(act->get_table()(5));
        lcdNumber_width_current->display(act->get_table()(6));
        lcdNumber_depth_current->display(act->get_table()(7));
        lcdNumber_height_current->display(act->get_table()(8));

    }

        first_time = true;
    }

    if(thing_actionables.empty() and thing_filtered_points.size() > 20)
    {



        // Add point x=0, y=0, z

        auto thing = new rc::ActionableThing(inner_model);
        if(thing->initialize( *best_actionable, thing_filtered_points, &viewer_global->scene))
        {
            // if a reasonable low error is reached, add the new actionable
            thing_actionables.append(thing);

        // Connect pushButton_reset and pushButton_clean_buffer to the reset_optimiser and clean_buffer variables
        connect(pushButton_reset, &QPushButton::clicked, thing, &rc::ActionableThing::reset_optimiser_slot);
        connect(pushButton_clean_buffer, &QPushButton::clicked, thing, &rc::ActionableThing::clean_buffer_slot);
        connect(slider_beta_cost, &QSlider::valueChanged, [this](auto v){ lcdNumber_beta->display(v);});
        connect(slider_beta_cost, &QSlider::valueChanged, thing, &rc::ActionableThing::change_beta_slot);
        slider_beta_cost->setValue(rc::ActionableThing::params.INIT_BETA_SOFTMAX_VALUE);
        connect(slider_gamma_cost, &QSlider::valueChanged, [this](auto v){ lcdNumber_gamma->display(v);});
        connect(slider_gamma_cost, &QSlider::valueChanged, thing, &rc::ActionableThing::change_gamma_slot);
        connect(slider_width_prior, &QSlider::valueChanged, thing, &rc::ActionableThing::change_width_prior);

        emit createTableTransform();
        }
    }
    draw_residuals(thing_filtered_points, &viewer->scene);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::competition_dynamics_lateral_inhibition(std::vector<rc::ActionableRoom> &actionables,
                                                             double alpha,
                                                             double beta,
                                                             int iterations)
{
    // // Function to perform the Winner-Take-All algorithm
    // //const double alpha = 0.1;
    // //const double beta = 0.05;
    // constexpr double gamma = 0.0; // Set to non-zero if using self-inhibition
    // constexpr double E_total = 1.0;
    //
    // const size_t N = actionables.size();
    // std::vector<double> values(N); // Current signal values
    // std::vector<double> energies(N);
    // for ( const auto &[i, act] : iter::enumerate(actionables))
    // {
    //     values[i] = act.get_value(); // Current signal values
    //     energies[i] = act.get_energy();      // Energies
    // }
    //
    // // Initialize energies proportional to initial signal values
    // double sum_S_initial = std::accumulate(values.begin(), values.end(), 0.0);
    // if (sum_S_initial == 0.0)
    // {
    //     // If all initial signals are zero, distribute energy equally
    //     std::ranges::fill(energies, E_total / static_cast<double>(N));
    // } else
    //     for (size_t i = 0; i < N; ++i)
    //         energies[i] = E_total * (values[i] / sum_S_initial);
    //
    // // Compute gains and competition for each signal
    // std::vector<double> gain(N, 0.0);
    // std::vector<double> competition(N, 0.0);
    //
    // const double sum_S = std::accumulate(values.begin(), values.end(), 0.0);
    // for (size_t i = 0; i < N; ++i) {
    //     gain[i] = alpha * energies[i] * values[i];
    //     competition[i] = beta * energies[i] * (sum_S - values[i]);
    //     if (gamma > 0.0) {
    //         competition[i] += gamma * energies[i];
    //     }
    // }
    //
    // // Update energies
    // for (size_t i = 0; i < N; ++i) {
    //     energies[i] += (gain[i] - competition[i]);
    //     // Prevent negative energies
    //     if (energies[i] < 0.0)
    //         energies[i] = 0.0;
    // }
    //
    // // Normalize energies to maintain E_total
    // double sum_E = std::accumulate(energies.begin(), energies.end(), 0.0);
    // if (sum_E > 0.0)
    //     for (size_t i = 0; i < N; ++i)
    //         energies[i] = (energies[i] / sum_E) * E_total;
    //
    // else
    //     // If total energy is zero, distribute equally
    //     std::ranges::fill(energies, E_total / static_cast<double>(N));
    //
    // // set updated energies in actionables
    // for (auto [i, act]: iter::enumerate(actionables))
    //     act.set_energy(energies[i]);
    //
    // // Check for termination condition
    // // double max_E = *std::ranges::max_element(energies);
    // // if (max_E >= threshold * E_total)
    // // {
    // //     const size_t winner = std::distance(energies.begin(), std::ranges::max_element(energies));
    // //     std::cout << "Winner is signal " << winner << " at time " << t
    // //               << " with energy " << std::fixed << std::setprecision(4) << energies[winner] << "\n";
    // //     // Optionally, break here if you want to stop after finding the winner
    // // }
}

// void SpecificWorker::competition_dynamics_lateral_inhibition_soft_max(std::vector<rc::ActionableRoom> &actionables,
//                                                                       double alpha,
//                                                                       double beta,
//                                                                       int iterations)
// {
//     std::vector<double> compete_softmax_with_inhibition(const std::vector<double>& errors, double total_energy, double tau, double gamma, const std::vector<std::vector<int>>& neighbors, std::vector<double> previous_energies) {
//         int num_actionables = errors.size();
//         std::vector<double> fitnesses(num_actionables);
//         double alpha = 1.0; // Fitness parameter (you can adjust this)
//
//         // Calculate fitnesses
//         std::transform(errors.begin(), errors.end(), fitnesses.begin(),
//                        [&alpha](double e) { return std::exp(-alpha * e); });
//
//         std::vector<double> energies(num_actionables, 0.0); // Initialize energies to 0
//
//         for (int i = 0; i < num_actionables; ++i) {
//             double denominator = 0.0;
//             // Sum over neighbors and self
//             for (int j : neighbors[i]) {
//                 double inhibition = std::exp(-gamma * std::abs(previous_energies[j] - previous_energies[i]));
//                 denominator += std::exp(fitnesses[j] / tau) * inhibition;
//
//             }
//             denominator += std::exp(fitnesses[i] / tau);
//
//             double probability = std::exp(fitnesses[i] / tau) / denominator;
//             energies[i] =
// }

std::optional<rc::ActionableRoom> SpecificWorker::check_new_candidate(const rc::ActionableOrigin &parent)
{
    rc::ActionableRoom candidate;
    const auto actionable_color = generate_random_color();
    const auto rd = viewer_global->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, actionable_color);
    if (candidate.initialize(parent, room_actionables, std::get<0>(rd)))
    {
        //        draw_room(candidate.get_corners_3d(), candidate.get_actionable_color(), &viewer_global->scene, 1);
        return candidate;
    }
    viewer_global->scene.removeItem(std::get<0>(rd));
    return {};
};

std::optional<rc::ActionableRoom> SpecificWorker::check_new_candidate(const rc::ActionableRoom &parent)
{
    rc::ActionableRoom candidate;
    const auto actionable_color = generate_random_color();
    const auto rd = viewer_global->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, actionable_color);
    if (candidate.initialize(parent, room_actionables, std::get<0>(rd)))
    {
//        draw_room(candidate.get_corners_3d(), candidate.get_actionable_color(), &viewer_global->scene, 1);
        qDebug() << __FUNCTION__ << "---New candidate from room---";
        return candidate;
    }
    else {viewer_global->scene.removeItem(std::get<0>(rd));};
    return {};
};

std::tuple<Match, Target> SpecificWorker::evaluate(rc::ActionableRoom &act,
                                                   const LidarPoints &points,
                                                   const Eigen::Affine2d &robot_pose)
{
    return {};
}

/////////////   HELPER FUNCTIONS   //////////////////////////////////////
std::pair<long, LidarPoints> SpecificWorker::read_lidar_helios() const
{
    try
    {
        auto ldata =  lidar3d_proxy->getLidarData("helios", 0, 2*M_PI, 2);
        // filter points according to height and distance
        LidarPoints p_filter;
        for(const auto &a: ldata.points)
        {
            if(a.z > 100 and a.distance2d > 200)
                p_filter.emplace_back(a.x, a.y, a.z);
        }
        return std::make_pair(ldata.timestamp, p_filter);
    }
    catch(const Ice::Exception &e){std::cout << e << std::endl;}
    return {};
}

Eigen::Vector2d SpecificWorker::select_new_target()
{
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> xdist(params.GRID_MAX_DIM.left(), params.GRID_MAX_DIM.right());
    static std::uniform_int_distribution<> ydist(params.GRID_MAX_DIM.left(), params.GRID_MAX_DIM.right());

    // Generate a random number within the specified range
    return {xdist(gen), ydist(gen)};
}
RetVal SpecificWorker::move_to_random_point(const std::vector<Eigen::Vector2d> &points, const Eigen::Vector2d &target)
{
    // if at target point, change to SELECT_NEW_POINT
    const double angle = atan2(target.x(), target.y());
    const double distance = target.norm();

    if (distance < params.MIN_DIST_TO_CENTER)
    { qDebug() << "At target. Change to SELECT_NEW_POINT"; return {STATE::SELECT_NEW_TARGET, 0.0, 0.0};}

    // compute velocities
    double rot = std::clamp(0.8*angle, -params.MAX_ROT_SPEED, params.MAX_ROT_SPEED);
    const double adv_brake = std::clamp(distance * 1.0/params.ROBOT_WIDTH - params.ROBOT_WIDTH, 0.0, 1.0);
    double adv = std::clamp(params.MAX_ADV_SPEED * gaussian(rot) * adv_brake, 0.0, params.MAX_ADV_SPEED);
    return {STATE::MOVE_TO_CENTER, adv, rot};
}
RetVal SpecificWorker::move_to_center(const std::vector<Eigen::Vector2d> &points)
{
    // compute the center of mass of the points
    Eigen::Vector2d center = std::accumulate(points.begin(), points.end(), Eigen::Vector2d{0.0,0.0}) / points.size();
    draw_room_center(center, &viewer->scene);

    // compute the angle to the center
    const double angle = atan2(center.x(), center.y());
    // compute the distance to the center
    double distance = center.norm();

    if (distance < params.MIN_DIST_TO_CENTER)
    { qDebug() << __FUNCTION__ << " At center, distance:" << distance << " Change to AT_CENTER"; return {STATE::AT_CENTER, 0.0, 0.0};};

    // compute velocities
    //const double adv_brake = std::clamp(distance * 1.f/params.ROBOT_WIDTH - params.ROBOT_WIDTH, 0.f, 1.f);
    const double rot = std::clamp(0.8*angle, -params.MAX_ROT_SPEED, params.MAX_ROT_SPEED);
    const double adv = std::clamp(params.MAX_ADV_SPEED * gaussian(rot) , 0.0, params.MAX_ADV_SPEED);
    return {STATE::MOVE_TO_CENTER, adv, rot};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
void SpecificWorker::print_clusters(const std::vector<QPolygonF> &polys, const std::vector<unsigned long> &votes,
                                    const std::vector<unsigned long> &centroids, const std::vector<unsigned long> &assignments) const
{
    qDebug() << "---------- End of clustering -------- -";
    qDebug() << "   Polygons:" << "" << polys.size();
    qDebug() << "   items per polygon";
    for (const auto &v: votes)
        std::cout << "   " << v;
    std::cout << std::endl;
    qDebug() << "   Clustering of votes";
    qDebug() << "       Number of centroids: " << centroids.size();
    std::cout << "       mean values: ";
    for (const auto &c: centroids)
        std::cout << c << " ";
    std::cout << std::endl;
    std::cout << "       number of items per cluster: ";
    for (const auto &a: assignments)
        std::cout << a << " ";
    std::cout << std::endl;
}
Eigen::Affine2d SpecificWorker::affine3d_to_2d(const RoboCompFullPoseEstimation::FullPoseMatrix& pose)
{
    // Extract the 2D rotation and translation components
    Eigen::Affine3d affine3d;
    affine3d.matrix() << pose.m00, pose.m01, pose.m02, pose.m03*1000,
            pose.m10, pose.m11, pose.m12, pose.m13*1000,
            pose.m20, pose.m21, pose.m22, pose.m23*1000,
            pose.m30, pose.m31, pose.m32, pose.m33;
    const double theta = std::atan2(affine3d.linear()(1, 0), affine3d.linear()(0, 0));
    const Eigen::Vector2d translation(affine3d.translation().x(), affine3d.translation().y());

    // Build the 2D affine transformation
    Eigen::Affine2d affine2d = Eigen::Translation2d(translation) * Eigen::Rotation2Dd(theta);

    return affine2d;
}
std::vector<Eigen::Vector2d> SpecificWorker::get_and_accumulate_corners(const Eigen::Affine2d &pose,
                                                                        const std::vector<Eigen::Vector2d> &points,
                                                                        const std::vector<Eigen::Vector2d> &corners)
{
    std::vector<Eigen::Vector2d> n_corners(corners);
    const auto &[_, __, rcorners, ___] = rc::Room_Detector::compute_features(points, &viewer->scene);
    // transform corners to the global frame given the current robot pose wrt the global frame (pose
    for(const auto &[v, c, _]: rcorners)
    {
        Eigen::Vector3d cc;
        cc << Eigen::Vector2d{c.x(), c.y()}, 1.0;
        auto t_cc = pose.matrix() * cc;  // Rx + T  (from robot RS to global RS)
        n_corners.emplace_back(t_cc.head(2));
    }
    return n_corners;
}
double SpecificWorker::gaussian(const double x) const
{
    // variance of the gaussian function is set by the user giving a point xset where the function must be yset, and solving for s
    constexpr double xset = 0.5;
    constexpr double yset = 0.4;
    constexpr double s = -xset * xset / log(yset);
    return static_cast<double>(exp(-x * x / s));
}
void SpecificWorker::move_robot(const double adv, const double rot) const
{
    try
    { omnirobot_proxy->setSpeedBase(0.f, adv, rot); }
    catch (const Ice::Exception &e)
    { std::cout << e << std::endl; }
}
void SpecificWorker::stop_robot() const
{
    try
    { omnirobot_proxy->setSpeedBase(0.f, 0.f, 0.f); }
    catch (const Ice::Exception &e)
    { std::cout << e << std::endl; }
}
void SpecificWorker::turn_robot(double angle)
{
    try
    { omnirobot_proxy->setSpeedBase(0.f, 0.f, angle); }
    catch (const Ice::Exception &e)
    { std::cout << e << std::endl; }
}

bool SpecificWorker::move_robot_to_target(const Eigen::Vector2d &target) const
{
    static bool first_time = true;

    const double distance = target.norm();
    //qDebug() << __FUNCTION__ << "Distance to target: " << distance;
    if (distance < params.MIN_DIST_TO_CENTER)   // if at target, stop
    {
        if (first_time)
        {
            qDebug() << __FUNCTION__ << "At target. Stop robot";
            first_time = false;
            stop_robot();
        }
        return true;
    }
    // compute the angle to the target
    const double angle = atan2(target.x(), target.y());
    const double rot = std::clamp(0.6*angle, -params.MAX_ROT_SPEED, params.MAX_ROT_SPEED);
    const double adv = std::clamp(params.MAX_ADV_SPEED * gaussian(rot) , 0.0, params.MAX_ADV_SPEED);
    move_robot(adv, rot);
    first_time = true;
    return false;
}
//////////////////////////////////////////////////////////////////
/// STATE  MACHINE
//////////////////////////////////////////////////////////////////

QColor SpecificWorker::generate_random_color()
{
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 255);
    return QColor(dis(gen), dis(gen), dis(gen));
}

/**
 * Draws LIDAR points onto a QGraphicsScene.
 *
 * This method clears any existing graphical items from the scene, then iterates over the filtered
 * LIDAR points to add new items. Each LIDAR point is represented as a colored rectangle. The point
 * with the minimum distance is highlighted in red, while the other points are drawn in green.
 *
 * @param filtered_points A collection of filtered points to be drawn, each containing the coordinates
 *                        and distance.
 * @param scene A pointer to the QGraphicsScene where the points will be drawn.
 */
void SpecificWorker::draw_lidar(auto &filtered_points, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem*> items;   // store items so they can be shown between iteratio
    for(const auto i: items)
    { scene->removeItem(i); delete i; }
    items.clear();

    const auto color = QColor(Qt::darkGreen);
    const auto brush = QBrush(QColor(Qt::darkGreen));
    for(const auto &p : filtered_points)
    {
        const auto item = scene->addRect(-25, -25, 50, 50, color, brush);
        item->setPos(p.x(), p.y());
        items.push_back(item);
    }
}

void SpecificWorker::draw_room_center(const Eigen::Vector2d &center, QGraphicsScene *scene, bool erase)
{
    static std::vector<QGraphicsItem*> items;   // store items so they can be shown between iteratio
    for(const auto i: items)
    { scene->removeItem(i); delete i; }
    items.clear();
    if (erase) return;

    const auto color = QColor(Qt::magenta);
    const auto brush = QBrush(QColor(Qt::magenta));
    items.push_back(scene->addLine(center.x(), center.y(), center.x(), center.y() + 200, QPen(QColor("green"), 20)));
    items.push_back(scene->addLine(center.x(), center.y(), center.x() + 200, center.y(), QPen(QColor("red"), 20)));
    const auto item = scene->addEllipse(-100, -100, 200, 200, color, brush);
    item->setPos(center.x(), center.y());
    items.push_back(item);
}
void SpecificWorker::draw_corners(const auto &corner, QGraphicsScene *scene, QColor color, float opacity, int vector_position) const
{
    static std::vector<QGraphicsItem*> items;
    if(vector_position == 0) // Ñapa for cleaning cornert at the beginning of the loop
    {
        for(const auto i: items)
        {
            scene->removeItem(i);
            delete i;
        }
        items.clear();
    }

    for (size_t i =0; i < corner.size(); i++)  // con range loop no funciona
    {
        auto item = scene->addEllipse(-100, -100, 200, 200, QPen(color, 20), QBrush(color));
        if constexpr (std::is_same_v<decltype(corner[i]), const Corner&>)
        {
            auto &[votes, p] = corner[i];
            item->setPos(p.x(), p.y());
        }
        else
        {
            item->setPos(corner[i].x(), corner[i].y());
        }
        item->setOpacity(opacity);
        items.emplace_back(item);
    }
}
void SpecificWorker::draw_corners_local(const std::vector<Eigen::Vector3d> &corner, QGraphicsScene *scene) const
{
    static std::vector<QGraphicsItem*> items;
    for(const auto i: items)
    {
        scene->removeItem(i);
        delete i;
    }
    items.clear();

    for (const auto &c: corner)
    {
        auto item = scene->addEllipse(-100, -100, 200, 200, QPen(QColor("magenta"), 20));
        item->setPos(c.x(), c.y());
        items.emplace_back(item);
    }
}
void SpecificWorker::draw_polys(const std::vector<QPolygonF> &polys, QGraphicsScene *scene, const QColor &color)
{
    static std::vector<QGraphicsItem*> items;
    for(const auto i: items)
    {
        scene->removeItem(i);
        delete i;
    }
    items.clear();

    for(const auto &poly: polys)
    {
        auto item = scene->addPolygon(poly, QPen(color, 20));
        items.push_back(item);
    }
};
void SpecificWorker::draw_room(const auto &corners, const QColor &color, QGraphicsScene *scene, float opacity, int vector_position) const
{
    static std::vector<QGraphicsItem*> items;
    //if(vector_position == 0)
    {
        for(const auto i: items)
        {
            scene->removeItem(i);
            delete i;
        }
        items.clear();
    }
    auto c_corners = corners;
    c_corners.push_back(corners.front());
    for (const auto &pp: iter::sliding_window(c_corners, 2))
    {
        const auto item = scene->addLine(pp[0].x(), pp[0].y(), pp[1].x(), pp[1].y(), QPen(color, 20));
        item->setOpacity(opacity);
        items.push_back(item);
    }
}

void SpecificWorker::draw_matching_corners_local(
        const std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, double, double, double>> &map, QGraphicsScene *scene)
{
    static std::vector<QGraphicsItem*> items;
    for(const auto i: items)
    {
        scene->removeItem(i);
        delete i;
    }
    items.clear();

    const std::vector<std::pair<QColor, QColor>> colors = {{QColor("cyan"), QColor("blue")},
                                                           {QColor("yellow"), QColor("red")},
                                                           {QColor("green"), QColor("purple")},
                                                           {QColor("orange"), QColor("pink")}};

    const std::vector<std::pair<QColor, QColor>> brushes = {{QColor("cyan"), QColor("blue")},
                                                            {QColor("yellow"), QColor("red")},
                                                            {QColor("green"), QColor("purple")},
                                                            {QColor("orange"), QColor("pink")}};

    for (const auto &[i, m ]: map | iter::enumerate)
    {
        auto &[nominal, measurement, range, bearing, errpr] = m;
        // Transform nominal and measurement corners to the robot reference system
        auto robot_actual_pose = robot_current_pose;
        const auto nominal_robot = robot_actual_pose.inverse() * nominal;
        const auto measurement_robot = robot_actual_pose.inverse() * measurement;

        const auto item1 = scene->addEllipse(-150, -150, 300, 300, QPen(colors[i].first, 40), brushes[i].first);
        item1->setPos(nominal_robot.x(), nominal_robot.y());
        items.emplace_back(item1);
        const auto item2 = scene->addEllipse(-150, -150, 300, 300, QPen(colors[i].second, 40), brushes[i].second);;
        item2->setPos(measurement_robot.x()+100, measurement_robot.y());
        items.emplace_back(item2);
        //qDebug() << "Nominal: " << nominal.x() << nominal.y() << " Measurement: " << measurement.x() << measurement.y();
    }
}
void SpecificWorker::draw_robot_global(const Eigen::Affine2d &pose)
{
    robot_draw_global->setRotation(qRadiansToDegrees(Eigen::Rotation2Df(pose.rotation()).angle()));
    robot_draw_global->setPos(pose.translation().x(), pose.translation().y());
}
void SpecificWorker::draw_robots_global(QGraphicsItem* robot, const Eigen::Affine2d &pose)
{
    robot->setRotation(qRadiansToDegrees(Eigen::Rotation2Df(pose.rotation()).angle()));
    robot->setPos(pose.translation().x(), pose.translation().y());
}
void SpecificWorker::plot_free_energy(double distance, double traza)
{
    // add value to plot
    static int key = 0;
    plot->graph(0)->addData(key++, distance);
    plot->graph(1)->addData(key++, traza);
    // Remove data points if there are more than X
    if (plot->graph(0)->dataCount() > params.MAX_DIST_POINTS_TO_SHOW)
        plot->graph(0)->data()->removeBefore(key - params.MAX_DIST_POINTS_TO_SHOW);
    if (plot->graph(1)->dataCount() > params.MAX_DIST_POINTS_TO_SHOW)
        plot->graph(1)->data()->removeBefore(key - params.MAX_DIST_POINTS_TO_SHOW);
    plot->xAxis->rescale(true);
    plot->yAxis->rescale(true);
    plot->replot();
}
// Function to plot the free energy of every actionable in the color of each one
void SpecificWorker::plot_multiple_free_energy(const std::vector<rc::ActionableRoom> &actionables)
{
    static int key = 0; // Variable para el eje X

    // Asegúrate de que el número de gráficos en el plot sea igual al tamaño del vector distances
    while (plot->graphCount() < static_cast<int>(actionables.size()))
    {
        plot->addGraph(); // Agrega un nuevo gráfico
        const int graphIndex = plot->graphCount() - 1;

        // Configura un color único para cada gráfico
        plot->graph(graphIndex)->setPen(QPen(actionables[graphIndex].get_actionable_color()));

        // (Opcional) Estilo de línea o puntos
        plot->graph(graphIndex)->setLineStyle(QCPGraph::lsLine);
        plot->graph(graphIndex)->setScatterStyle(QCPScatterStyle::ssCircle);
    }

    // Añade datos a cada gráfico
    for (const auto &[i, act]: iter::enumerate(actionables))
    {
        // plot->graph(i)->addData(key, actionables[i].get_error());
        plot->graph(i)->addData(key, act.get_value());
        //qDebug() << "Energy: " << actionables[i].get_energy();

        // Elimina datos antiguos si exceden el límite
        if (plot->graph(i)->dataCount() > params.MAX_DIST_POINTS_TO_SHOW)
            plot->graph(i)->data()->removeBefore(key - params.MAX_DIST_POINTS_TO_SHOW);
    }

    ++key; // Incrementa la clave para el eje X

    // Reescala y actualiza
    plot->xAxis->rescale(true); // Ajusta el eje X automáticamente
    //plot->yAxis->rescale(true); // Ajusta el eje Y automáticamente
    if (frame_plot->size().width() > plot->width()) plot->resize(size().width(), plot->height());
    plot->replot();             // Actualiza el gráfico
}

void SpecificWorker::draw_circular_queue(const Boost_Circular_Buffer &buffer, QGraphicsScene *scene, bool erase)
{
    static std::vector<QGraphicsItem*> items;
    for(const auto i: items)
    { scene->removeItem(i); delete i; }
    items.clear();

    if (erase) return;

    for (const auto &mat: buffer)
        for (const auto i: iter::range(mat.cols()))
        {
            const auto item = scene->addEllipse(-25, -25, 50, 50, QPen(QColor("cyan"), 20));
            item->setPos(mat(0, i), mat(1, i));
            items.push_back(item);
        }
}

void SpecificWorker::draw_residuals_in_room_frame(std::vector<Eigen::Vector3d> points, QGraphicsScene *scene, bool erase)
{
    static std::vector<QGraphicsItem*> items;
    for(const auto i: items)
    { scene->removeItem(i); delete i; }
    items.clear();

    if (erase) return;

    const auto color = QColor(Qt::darkBlue);
    const auto brush = QBrush(QColor(Qt::darkBlue));
    for(const auto &p : points)
    {
        const auto item = scene->addRect(-25, -25, 50, 50, color, brush);
        item->setPos(p.x(), p.y());
        items.push_back(item);
    }
};

void SpecificWorker::draw_residuals(const auto &points, QGraphicsScene *scene, bool erase)
{
    static std::vector<QGraphicsItem*> items;
    for(const auto i: items)
    { scene->removeItem(i); delete i; }
    items.clear();

    if (erase) return;

    const auto color = QColor(Qt::darkBlue);
    const auto brush = QBrush(QColor(Qt::darkBlue));
    for(const auto &p : points)
    {
        const auto item = scene->addRect(-25, -25, 50, 50, color, brush);
        item->setPos(p.x(), p.y());
        items.push_back(item);
    }
}
//////////////////////////////////////////////////////////////////
/// AUXILIARY FUNCTIONS
//////////////////////////////////////////////////////////////////
void SpecificWorker::emergency()
{
    std::cout << "Emergency worker" << std::endl;
    //computeCODE
    //
    //if (SUCCESSFUL)
    //  emmit goToRestore()
}
//Execute one when exiting to emergencyState
void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
    //computeCODE
    //Restore emergency component

}
int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, qApp, SLOT(quit()));
    return 0;
}


// move the robot
// if(pushButton_stop->isChecked())
// {
//     double adv = 0;
//     double rot = 0;
//     //qDebug() << "Adv: " << adv << " Rot: " << rot << "Dist to person" << dist;
//     try
//     { omnirobot_proxy->setSpeedBase(0.f, adv, rot); }
//     catch (const Ice::Exception &e)
//     { std::cout << e << std::endl; }
//     lcdNumber_adv->display(adv);
//     lcdNumber_rot->display(rot);
// }



/**************************************/
// From the RoboCompLidar3D you can call these methods:
// this->lidar3d_proxy->getLidarData(...)
// this->lidar3d_proxy->getLidarDataArrayProyectedInImage(...)
// this->lidar3d_proxy->getLidarDataProyectedInImage(...)
// this->lidar3d_proxy->getLidarDataWithThreshold2d(...)

/**************************************/
// From the RoboCompLidar3D you can use this types:
// RoboCompLidar3D::TPoint
// RoboCompLidar3D::TDataImage
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
// From the RoboCompVisualElements you can call these methods:
// this->visualelements_proxy->getVisualObjects(...)
// this->visualelements_proxy->setVisualObjects(...)

/**************************************/
// From the RoboCompVisualElements you can use this types:
// RoboCompVisualElements::TRoi
// RoboCompVisualElements::TObject
// RoboCompVisualElements::TObjects

// Instantiate the random number generator and distribution
//    static std::mt19937 gen(rd());
//    static std::uniform_int_distribution<int> dist(0, 1);
//    static bool first_time = true;
//    static int sign = 1;

//        auto poly = room_model.get_qt_polygon();
//        // filter out point on wall and outside room
//        poly = shrink_polygon(poly, params.ROBOT_WIDTH);
//        std::ranges::copy_if(points, std::back_inserter(points_inside), [poly](auto &a)
//        { return poly.containsPoint(QPointF(a.x, a.y), Qt::FillRule::OddEvenFill); });
//        if(points_inside.empty()) { qWarning() << __FUNCTION__ << "Empty points inside"; return {}; }
// cluster remaining points
// auto list_poly = rc::dbscan(points_inside, params.ROBOT_WIDTH, 2);

// Find Pareto-optimal tasks using ranges and lambdas
// std::vector<Task> findParetoFront(const std::vector<Task>& tasks) {
//     // Define 'dominates' within the function:
//     // encapsulating it within the function is better than referenced.
//     auto dominates = [](const Task& a, const Task& b) {
//         return (a.time <= b.time && a.energy <= b.energy) &&
//                (a.time < b.time || a.energy < b.energy);
//     };
//
//     // Use ranges and views within the function
//     auto paretoView = tasks
//         | std::views::filter([&tasks, &dominates](const Task& task) {
//             return std::none_of(tasks.begin(), tasks.end(),
//                 [&task, &dominates](const Task& other) {
//                     return (&task != &other) && dominates(other, task);
//                 });
//         });

//        if (const auto final_corners = clusters_have_a_room(cluster_map, polys, votes); final_corners.has_value())
//            {
//                // Transform final corners to std::vector<Eigen::Vector3d>
//                std::vector<Eigen::Vector3d> final_corners_3d;
//                final_corners_3d.reserve(final_corners.value().size());
//                for (const auto &c: final_corners.value())
//                {
//                    Eigen::Vector3d rc; rc << c.x(), c.y(), 1.0f;     // add homogeneous coordinate
//                    final_corners_3d.emplace_back(rc);
//                }
//
//                draw_corners(final_corners_3d, &viewer_global->scene);
//                draw_room(final_corners.value(), &viewer_global->scene);
// update the room from size and orientation.
// try
// {
//     // get latest pose from odometry
//     robot_current_pose = affine3d_to_2d(lidarodometry_proxy->getPoseAndChange().pose);
//     qDebug() << __FUNCTION__ << "Robot pose AT_CENTER before initializing: " << robot_current_pose.translation().x() << robot_current_pose.translation().y();
//     // initialize the new room
//     const auto &[rect, pose] = current_room.initialize(final_corners.value(), robot_current_pose);
//     // robot pose now is relative to the room coordinate system, which is (0,0) at the center of the room
//     robot_current_pose = pose;
//     std::cout << __FUNCTION__ << "Robot pose AT_CENTER after room initialization:" <<  pose.rotation() << std::endl << pose.translation() << std::endl;
//     current_room.print();
//     current_room.draw_2D(current_room, "green", &viewer_global->scene);
// }
// catch(const Ice::Exception &e){std::cout << e << std::endl;}
//}

/// procreation
    // initializer_actionable.update_robot_pose(best_actionable->get_robot_pose());
    // const auto match_initializer = initializer_actionable.project_corners(corners);
    // if (initializer_actionable.ready_to_procreate() and best_actionable->get_buffer_error() > 1.5 /*1500*/)
    // {
    //     rc::ActionableRoom candidate;
    //     // Generate random QColor
    //     auto color = generate_random_color();
    //
    //     auto rd = viewer_global->add_robot(params.ROBOT_WIDTH, params.ROBOT_LENGTH, 0, 100, color);
    //     candidate.initialize(initializer_actionable, rd);
    //     candidate.define_actionable_color(color);
    //     if (candidate.check_viability())
    //     {
    //         // Check that new candidate room dimensions are different from every actionable room using a lambda function
    //         // In case the result is true, discard process
    //         if (std::ranges::any_of(actionables, [candidate](const auto &act)
    //         {
    //             if(candidate.get_room().get_rotation() == act.get_room().get_rotation())
    //                 return fabs(act.get_room().get_depth() - candidate.get_room().get_depth()) < 300 and
    //                        fabs(act.get_room().get_width() - candidate.get_room().get_width()) < 300;
    //             else
    //                 return fabs(act.get_room().get_depth() - candidate.get_room().get_width()) < 300 and
    //                        fabs(act.get_room().get_width() - candidate.get_room().get_depth()) < 300;
    //         }))
    //         {
    //             qDebug()  << __FUNCTION__<< "New candidate discarded";
    //             candidate.remove_robot_draw(&viewer_global->scene);
    //         } else
    //         {
    //             actionables.emplace_back(candidate);
    //             draw_room(candidate.get_corners_3d(), candidate.get_actionable_color(), &viewer_global->scene, false);
    //         }
    //     }   // no viable candidate
    //     else
    //         candidate.remove_robot_draw(&viewer_global->scene);
    // } // end procreation
