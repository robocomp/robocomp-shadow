//
// Created by pbustos on 9/12/24.
//

#include "actionable_thing.h"
#include <iosfwd>
#include <tuple>
#include <utility>
#include <vector>
#include "pch.h"
#include <gtsam/nonlinear/AdaptAutoDiff.h>
//#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/base/Vector.h>
#include <gtsam/slam/expressions.h>
//#include <gtsam/nonlinear/expressions.h>

namespace rc
{
    ActionableThing::Params ActionableThing::params;

    //ActionableThing::ActionableThing(ProtectedMap_shr innermodel, const std::atomic<bool> &stop_flag) : internal_model(innermodel), stop_flag(stop_flag)
    ActionableThing::ActionableThing(const std::shared_ptr<rc::ActionablesData> &innermodel_)
    {
        error_buffer.set_capacity(25);
        residuals_queue.set_capacity(500);
        inner_model = innermodel_; // do not move since we want that the sender retains ownership also
        inner_model->fridge = std::make_shared<rc::Fridge>();
    }

    bool ActionableThing::initialize(const rc::ActionableRoom &current_room,
                                     const std::vector<Eigen::Vector3d> residuals,  // residuals in room frame and in meters
                                     const Params &params_,
                                     QGraphicsScene *scene)
    {
        // copy SIGMA values
        params = params_;
        std::cout.precision(3);

        // Add the points to the circular buffer
        const auto points_memory = add_new_points(residuals);

        //qInfo() << __FILE__ << __FUNCTION__ << "In Initialize: points_memory set size: " << points_memory.size();

        Eigen::Vector3d mass_center = std::accumulate(points_memory.begin(), points_memory.end(), Eigen::Vector3d(0.0, 0.0, 0.0)) / static_cast<float>(points_memory.size());

        // Draw centroid point
        draw_point(mass_center, scene);

        // augment the state vector
        //this->fridge = {mass_center.x(), mass_center.y(), params.INIT_ANGLE_VALUE, params.INIT_WIDTH_VALUE, params.INIT_WIDTH_VALUE};
        inner_model->fridge = std::make_shared<rc::Fridge>();
        inner_model->fridge->means = {mass_center.x(), mass_center.y(), params.INIT_ANGLE_VALUE, params.INIT_WIDTH_VALUE, params.INIT_WIDTH_VALUE};
        //qInfo() << __FUNCTION__<< "Initial fridge: " << mass_center.x() << mass_center.y();

        // Then minimise the sum of residual distances to their closest fridge side
        //this->fridge = factor_graph_expr_points(points_memory, fridge, current_room, false, mass_center, scene);
        inner_model->fridge->means = factor_graph_expr_points(points_memory, inner_model->fridge->means, current_room, false, mass_center, scene);

        //draw_fridge(fridge, Qt::red, scene);
        draw_fridge(inner_model->fridge->means, Qt::red, scene);
        draw_residuals_in_room_frame(points_memory, scene);
        return true;
    }

    LidarPoints ActionableThing::project(const LidarPoints &residuals,  // residuals in room frame and in meters
                                         const rc::ActionableRoom &current_room,
                                         bool reset_optimiser,
                                         QGraphicsScene *scene)
    {
        // remove from points all the points that are inside the rectangle: TO BE DONE
        LidarPoints filtered_points;

        // add the points to the circular buffer and return a vector with the new points
        const auto points_memory = add_new_points(residuals);

        // compute the mass center of the points
        const Eigen::Vector3d mass_center = std::accumulate(points_memory.begin(), points_memory.end(), Eigen::Vector3d(0.0, 0.0, 0.0)) / static_cast<float>(points_memory.size());
        draw_point(mass_center, scene);

        // minimise the sum of residual distances to their closest fridge side
        //fridge = factor_graph_expr_points(points_memory, fridge, current_room, reset_optimiser, mass_center, scene);
        inner_model->fridge->means = factor_graph_expr_points(points_memory, inner_model->fridge->means, current_room, reset_optimiser, mass_center, scene);

        //draw_fridge(fridge, Qt::red, scene);
        draw_fridge(inner_model->fridge->means, Qt::red, scene);
        draw_residuals_in_room_frame(points_memory, scene);

        ///////  Affordances  and freezing //////////////
        const double erorr = get_error();
        const double traza_d1 = get_traza();
        //const double det = get_determinant();
        //const double proj = get_projection_error();

        // compute the derivative of the traza
        static double traza_d2 = 0.0, traza_d3 = 0.0;
        // compute derivative of traza_d2 using finite differences
        const double d_traza_d2 = (traza_d1 - traza_d3) / 2.0;
        traza_d3 = traza_d2;
        traza_d2 = traza_d1;

        // a combination of the above values is the urge to explore vs the urge to freeze
        if (traza_d1 > 0.1)
        {
            qInfo() << __FUNCTION__ << "Error: " << erorr << " Traza: " << traza_d1 << " dTraza: " << d_traza_d2;
        }
        
        // if current error is not too high, we propose an affordance. Otherwise, we wait we reset the optimiser
        // check which are the sides of the fridge that are self-occluded
        // given the current fridge, we want the face(s) that are occluded by itself
        // occlusion are computed by drawing rays from the robot to the fridge corners and side midpoints
        // If the ray intersects the fridge polygon, then the side is occluded
        // we need the robot pose in the room frame, the fridge object that has a square in it.
        // It should be a method of the fridge object that takes the robot pose and returns the occluded sides

        // compute a simple path from the robot towards the closest hidden side of the fridge
        // we need the robot pose in the room frame, the hidden side of the fridge as a target point, the fridge polygon and the room polygon (and other closeby obstacles)
        // A simple sample procedure may work. Define a circle around the fridge with radius = 1.5 * max(width, depth)
        // sample points in the outer circle and check if they are inside the room and not inside the fridge and not inside any other obstacle
        // check if the point is in LoS with the robot. If so set the affordance. If not, continue sampling in a new circle in the midpoint.
        // The path is wrapped into an affordance object aff_goto_x which is scheduled for execution upon approval.

        return filtered_points;
    }

    gtsam::Vector5 ActionableThing::factor_graph_expr_points( const std::vector<Eigen::Vector3d> &residual_points, // residuals in room frame and in meters
                                                              const gtsam::Vector5 &initial_fridge,
                                                              const rc::ActionableRoom &current_room,
                                                              bool reset_optimiser,
                                                              const Eigen::Vector3d &mass_center,
                                                              QGraphicsScene *scene)
    {
        //        std::cout << "---------- factor_graph_expr_points --------------------------------" << std::endl;
        //---- Create Variable with noise
        const gtsam::Symbol fridgeSym('f', 1);
        gtsam::Key fridgeKey = fridgeSym.key();

        // Create an expression for the fridge params
        gtsam::Expression<gtsam::Vector5> fridge_(fridgeKey);

        ///// Priors (x, y, alpha, w, d) //////////////////////////////
//        auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector5(1,1, 10, 1, 1));
        // priors using params.
        auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector5(params.PRIOR_CX_SIGMA,
                                                                              params.PRIOR_CY_SIGMA,
                                                                              params.PRIOR_ALPHA_SIGMA,
                                                                              params.PRIOR_WIDTH_SIGMA,
                                                                              params.PRIOR_DEPTH_SIGMA));
        //auto prior_noise = gtsam::noiseModel::Isotropic::Sigma(5, 0.1);

        //// Likelihoods /////////////////////////////

        /// Define the points likelihood function
        //auto denom_expression = gtsam::Vector1_(gtsam::Vector1(residual_points.size() / 2));

        //----------------------------------- LAMBDA USING EXPRESSIONS -----------------------------------
//                auto proj_points_ = [fridge_, denom_expression](const gtsam::Vector2 &p, const size_t num_points)
//        {
//            return
//                    gtsam::Double_(&factors::normalize,
//                                   denom_expression,
//                                   gtsam::Double_(&factors::softmin,
//                                                  gtsam::Double_(&factors::dist2side_sqr, gtsam::Point2_(&factors::fridge2side,
//                                                                                                         fridge_, gtsam::Double_(1.0),
//                                                                                                         gtsam::Point2_(&factors::room2fridge,
//                                                                                                                        fridge_,
//                                                                                                                        gtsam::Point2_(p)))),
//
//                                                  gtsam::Double_(&factors::dist2side_sqr, gtsam::Point2_(&factors::fridge2side,
//                                                                                                         fridge_,  gtsam::Double_(2.0),
//                                                                                                         gtsam::Point2_(&factors::room2fridge,
//                                                                                                                        fridge_,
//                                                                                                                        gtsam::Point2_(p)))),
//
//                                                  gtsam::Double_(&factors::dist2side_sqr, gtsam::Point2_(&factors::fridge2side,
//                                                                                                         fridge_,  gtsam::Double_(3.0),
//                                                                                                         gtsam::Point2_(&factors::room2fridge,
//                                                                                                                        fridge_,
//                                                                                                                        gtsam::Point2_(p)))),
//
//                                                  gtsam::Double_(&factors::dist2side_sqr,gtsam::Point2_(&factors::fridge2side,
//                                                                                                        fridge_,  gtsam::Double_(4.0),
//                                                                                                        gtsam::Point2_(&factors::room2fridge,
//                                                                                                                       fridge_, gtsam::Point2_(p))))
//                                   )
//                    );
//        };


        auto proj_points_ = [fridge_](const gtsam::Vector2 &p, const size_t num_points)
        {
            return
                  // gtsam::Double_(&factors::softmin,
                  //                gtsam::Double_(&factors::dist2seg_sqr, fridge_, gtsam::Double_(1.0), gtsam::Point2_(&factors::fridge2side,
                  //                                                                       fridge_, gtsam::Double_(1.0),
                  //                                                                       gtsam::Point2_(&factors::room2fridge,
                  //                                                                                      fridge_,
                  //                                                                                      gtsam::Point2_(p)))),
                  //
                  //                gtsam::Double_(&factors::dist2seg_sqr, fridge_ , gtsam::Double_(2.0) , gtsam::Point2_(&factors::fridge2side,
                  //                                                                       fridge_,  gtsam::Double_(2.0),
                  //                                                                       gtsam::Point2_(&factors::room2fridge,
                  //                                                                                      fridge_,
                  //                                                                                      gtsam::Point2_(p)))),
                  //
                  //                gtsam::Double_(&factors::dist2seg_sqr, fridge_ , gtsam::Double_(3.0) , gtsam::Point2_(&factors::fridge2side,
                  //                                                                       fridge_,  gtsam::Double_(3.0),
                  //                                                                       gtsam::Point2_(&factors::room2fridge,
                  //                                                                                      fridge_,
                  //                                                                                      gtsam::Point2_(p)))),
                  //
                  //                gtsam::Double_(&factors::dist2seg_sqr, fridge_ , gtsam::Double_(4.0) , gtsam::Point2_(&factors::fridge2side,
                  //                                                                      fridge_,  gtsam::Double_(4.0),
                  //                                                                      gtsam::Point2_(&factors::room2fridge,
                  //                                                                                     fridge_, gtsam::Point2_(p)))));
                    gtsam::Double_(&factors::softmin,
                                   gtsam::Double_(&factors::dist2seg, fridge_, gtsam::Double_(1.0),
                                                                               gtsam::Point2_(&factors::room2fridge,
                                                                                              fridge_, gtsam::Point2_(p))),

                                   gtsam::Double_(&factors::dist2seg,fridge_ , gtsam::Double_(2.0) ,
                                                                               gtsam::Point2_(&factors::room2fridge,
                                                                                           fridge_, gtsam::Point2_(p))),

                                   gtsam::Double_(&factors::dist2seg, fridge_ , gtsam::Double_(3.0) ,
                                                                              gtsam::Point2_(&factors::room2fridge,
                                                                                            fridge_, gtsam::Point2_(p))),

                                   gtsam::Double_(&factors::dist2seg, fridge_ , gtsam::Double_(4.0) ,
                                                                              gtsam::Point2_(&factors::room2fridge,
                                                                                            fridge_, gtsam::Point2_(p)))

                    );
        };



        // Define a noise model for the points loss (e.g., isotropic noise)
        auto noise_model_points = gtsam::noiseModel::Isotropic::Sigma(1, params.POINTS_SIGMA);

        ////////////////////////////////////////////////////////7
        /// Define the close_to_wall likelihood function
        ////////////////////////////////////////////////////////
        //  Transform the center of the fridge to each room wall frame, the compute the distance to the wall as the y abs of the y coordinate and get the minimum of the four distances.
        //auto sw = current_room.get_room().get_width_meters()/2.0;
        //auto sd = current_room.get_room().get_depth_meters()/2.0;
        auto sw = inner_model->room->get_width_meters()/2.0; // Now we get the room from the inner model
        auto sd = inner_model->room->get_depth_meters()/2.0;


        // auto proj_adj_ = gtsam::Double_(&factors::softmin,
        //         gtsam::Double_(&factors::dist_side2wall_sqr, fridge_, gtsam::Point2_(&factors::fridge2wall, fridge_, gtsam::Vector3_(gtsam::Vector3(1.0, sw, sd)))),
        //         gtsam::Double_(&factors::dist_side2wall_sqr, fridge_, gtsam::Point2_(&factors::fridge2wall, fridge_, gtsam::Vector3_(gtsam::Vector3(2.0, sw, sd)))),
        //         gtsam::Double_(&factors::dist_side2wall_sqr, fridge_, gtsam::Point2_(&factors::fridge2wall, fridge_, gtsam::Vector3_(gtsam::Vector3(3.0, sw, sd)))),
        //         gtsam::Double_(&factors::dist_side2wall_sqr, fridge_, gtsam::Point2_(&factors::fridge2wall, fridge_, gtsam::Vector3_(gtsam::Vector3(4.0, sw, sd)))));

        auto proj_adj_ = gtsam::Double_(&factors::softmin,
                gtsam::Double_(&factors::closest_fridge_side_to_wall, fridge_, gtsam::Vector3_(gtsam::Vector3(1.0, sw, sd))),
                gtsam::Double_(&factors::closest_fridge_side_to_wall, fridge_, gtsam::Vector3_(gtsam::Vector3(2.0, sw, sd))),
                gtsam::Double_(&factors::closest_fridge_side_to_wall, fridge_, gtsam::Vector3_(gtsam::Vector3(3.0, sw, sd))),
                gtsam::Double_(&factors::closest_fridge_side_to_wall, fridge_, gtsam::Vector3_(gtsam::Vector3(4.0, sw, sd))));

        // Define a noise model for the adj loss (e.g., isotropic noise)
        auto noise_model_adj = gtsam::noiseModel::Isotropic::Sigma(1, params.ADJ_SIGMA);

        ///////////////////////////////////////////////////////
        /// Define the align_to_wall likelihood function
        double closest_wall = 1.0;
        auto proj_align_ = gtsam::Double_(&factors::angle2wall_sqr, fridge_, gtsam::Double_(closest_wall));

        // Define a noise model for the points loss (e.g., isotropic noise)
        auto noise_model_align = gtsam::noiseModel::Isotropic::Sigma(1, params.ALIGNMENT_SIGMA);

        //////////////////////////////////////
        // Create a Factor Graph
        gtsam::ExpressionFactorGraph graph;

        // Add the fridge variable to the graph
        graph.addExpressionFactor(fridge_, initial_fridge, prior_noise);

        /// Add custom expression factor for each liDAR point. The total weight should be normalized by the size of residual_points
        double num_points = static_cast<double>(residual_points.size());
        for (const auto &pp: residual_points)
             graph.addExpressionFactor(proj_points_(pp.head(2), num_points), 0.0, noise_model_points);  // measurement is zero since we want to minimize the distance

        /// Add custom expression factor for closest wall adjacency
        graph.addExpressionFactor(proj_adj_, 0.0, noise_model_adj);  // measurement is zero since we want to minimize the distance

        /// Add custom expression factor for alignment
        graph.addExpressionFactor(proj_align_, 0.0, noise_model_align);  // measurement is zero since we want to minimize the distance

        //////////////////////////////////////////////////////////////////
        // Provide initial estimates for the fridge center
        ///////////////////////////////////////////////////////////////////
        gtsam::Values initialEstimate;
        auto initial_fridge_value = gtsam::Vector5(initial_fridge.x(), initial_fridge.y(), initial_fridge[2], initial_fridge[3], initial_fridge[4]);
        if (reset_optimiser)    // if the UI buttom is clicked, add random noise to the initialFridge vector
        {
            // add random noise to the initialFridge vector
            auto noise = gtsam::Vector5::Random() * 0.12;
            initial_fridge_value = gtsam::Vector5(mass_center.x() + noise.x(),
                                                  mass_center.y() + noise.y(),
                                                  params.INIT_ANGLE_VALUE + noise[2],
                                                  params.INIT_WIDTH_VALUE + noise[3],
                                                  params.INIT_DEPTH_VALUE + noise[4]);
        }
        initialEstimate.insert(fridgeKey, initial_fridge_value);

        double initialError = graph.error(initialEstimate);

        // graph.print("Factor Graph:\n");

        /// ---------------- OPTIMIZATION ----------------------
        gtsam::LevenbergMarquardtParams params;
        params.maxIterations = 2;  // Número máximo de iteraciones
        params.absoluteErrorTol = 1e-5;  // Tolerancia de error absoluto
        params.relativeErrorTol = 1e-5;  // Tolerancia de error relativo
        //params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::VALUES;  // Nivel de verbosidad
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
        gtsam::Values result = optimizer.optimize();

        /// Show optimization results
        this->error = graph.error(result);
        //std::cout << __FUNCTION__ << " Error initial: " << initialError << " - Final error: " << this->error << std::endl;
        //std::cout << __FUNCTION__ << "Optimized fridge center: " << result.at<gtsam::Vector5>(fridgeKey).transpose() << std::endl;

        // Compute the marginal covariance for the 5D variable
        try
        {
            gtsam::Marginals marginals(graph, result);
            this->covariance = marginals.marginalCovariance(fridgeKey);
            //std::cout << "covariance:\n " << covariance << std::endl;
            //Compute and print the covariance matrix determinant
//            std::cout << __FUNCTION__ << " Covariance determinant: " << covariance.determinant() << std::endl;
            // Compute and print matrix trace
//            std::cout << __FUNCTION__ << "Covariance trace: " << covariance.trace() << std::endl;
            plotUncertaintyEllipses(covariance, result.at<gtsam::Vector5>(fridgeKey), scene);
        }
        catch (const gtsam::IndeterminantLinearSystemException &e){ std::cout << e.what() << std::endl; };
        return result.at<gtsam::Vector5>(fridgeKey);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
     std::vector<Eigen::Vector3d> ActionableThing::add_new_points(const std::vector<Eigen::Vector3d> &points)
    {
        static std::vector<Eigen::Vector3d> points_memory;
        for (const auto &point: points)
        {
            // Generate a lambda function to calculate the distance between the point and every
            // point in aux vector and stops for the first point that is closer than 0.1 meters
            auto it = std::find_if(points_memory.begin(), points_memory.end(), [&point](const Eigen::Vector3d &p)
                { return (point - p).norm() < 0.05; });
            // If the point is not found, add it to the aux vector
            if (it == points_memory.end())
                points_memory.push_back(point);
        }

        //auto end_time = std::chrono::high_resolution_clock::now();
        //auto duration = std::chrono::duration_cast<std::chrono::microseconds >(end_time - start_time);
        // qDebug() << __FUNCTION__ << "Time to add new points: " << duration.count() << "micros";
        return points_memory;
    }
    //------------------------NOé VERSION ---------------------------------------------------
    std::vector<Eigen::Vector3d> ActionableThing::add_new_points_2(const std::vector<Eigen::Vector3d> &points)
    {
        //start chrono
        //auto start = std::chrono::high_resolution_clock::now();
        constexpr double search_radius = 0.05 * 0.05; // radius

        //Create std::vector<Eigen::Vector3d> from residual queue
        std::vector<Eigen::Vector3d> cloud(residuals_queue.begin(), residuals_queue.end());

        // Obtener el índice donde comienzan los nuevos puntos
        size_t index = cloud.size();
        //Insertar puntos
        cloud.insert(cloud.end(), points.begin(), points.end());

        // Crear vector de booleanos para marcar qué puntos son insertables
        std::vector<bool> insertable(cloud.size(), true);

        // Comparar cada nuevo punto con todos los puntos en el cloud
        for (size_t i = index; i < cloud.size(); i++)
        {
            if (!insertable[i])
                continue;

            for (size_t j = 0; j < cloud.size(); j++)
            {
                if (!insertable[j] or i == j)
                    continue;
                double dist_sq = (cloud[i] - cloud[j]).squaredNorm();
                if (dist_sq <= search_radius)
                {
                    if (j >= index)
                        insertable[j] = false;
                    else
                        insertable[i] = false;
                }
            }
        }

        // Limpiar residuals_queue y agregar solo los puntos insertables
        residuals_queue.clear();
        for (size_t i = 0; i < cloud.size(); i++)
        {
            if (insertable[i])
                residuals_queue.push_back(cloud[i]);
        }

        std::vector<Eigen::Vector3d> points_memory(residuals_queue.begin(), residuals_queue.end());

        // Imprimir tamaños de los vectores
        //qInfo() << __FUNCTION__ << "Size NOE" << residuals_queue.size() << " points in points";
        //print chrono - start
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> elapsed = end - start;
        //std::cout << "Elapsed time: " << elapsed.count() << " s\n";
        return points_memory;
    }

    void ActionableThing::plotUncertaintyEllipses(const Eigen::MatrixXd& covariance, const Eigen::Matrix<double, 5, 1> &params, QGraphicsScene* scene)
    {
        // Compute the eigenvalues and eigenvectors of the covariance matrix
        // const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance);
        // Eigen::VectorXd eigenvalues = solver.eigenvalues();
        // Eigen::MatrixXd eigenvectors = solver.eigenvectors();

        static std::vector<QGraphicsItem*> items;
        for(const auto i: items)
        {  scene->removeItem(i); delete i; }
        items.clear();

        // Scale factor for the ellipse (e.g., 95% confidence interval)
        const double scale = std::sqrt(5.991); // Chi-squared distribution for 2 degrees of freedom

        const std::vector<QColor> colors = {QColor("yellow"), QColor("cyan"), QColor("orange"), QColor("blue"), QColor("green")};
        // Draw the eigenvectors with origin in (params(0), params(1)) and length proportional to the eigenvalues.
        // for (int i = 0; i < eigenvalues.size(); ++i)
        // {
        //     const double length = 5000 * scale * std::sqrt(eigenvalues(i));
        //     const Eigen::Vector2d end = Eigen::Vector2d(params(0) * 1000, params(1) * 1000) + length * eigenvectors.col(i).head(2);
        //     QGraphicsLineItem* line = scene->addLine(params(0) * 1000, params(1) * 1000, end.x(), end.y(), QPen(colors[i], 15));
        //     items.push_back(line);
        // }
        // draw a circle centered at params(0), params(1) with radius proportional to the covariance trace
        //qInfo() << __FUNCTION__ << "Covariance trace: " << covariance.trace() << "determinant: " << covariance.determinant();
        const double radius = 10000 * covariance.trace();     //TODO: magic number
        QGraphicsEllipseItem* ellipse = scene->addEllipse(-radius / 2, -radius / 2, radius, radius);
        ellipse->setPos(params(0) * 1000, params(1) * 1000); // Center of the ellipse
        ellipse->setBrush(Qt::transparent);
        ellipse->setPen(QPen(QColor("orange"), 15));
        items.push_back(ellipse);
        // draw a text with two decimal points only close to the ellipse showing the value of radius.
        QGraphicsSimpleTextItem* text = scene->addSimpleText(QString::number(radius, 'f', 2));
        text->setPos(params(0) * 1000 + radius, params(1) * 1000);
        QFont font = text->font(); font.setPointSize(130); text->setFont(font);
        text->setTransform(QTransform::fromScale(1, -1));
        items.push_back(text);


        // for (int i = 0; i < eigenvalues.size(); ++i)
        // {
        //     const double majorAxis = 2000 * scale * std::sqrt(eigenvalues(i));
        //     const double minorAxis = 2000 * scale * std::sqrt(eigenvalues(i));
        //     // Print the major and minor axes
        //     //std::cout << "Major axis: " << majorAxis << ", Minor axis: " << minorAxis << std::endl;
        //     // Compute the rotation angle (in degrees)
        //     const double angle = std::atan2(eigenvectors(1, i), eigenvectors(0, i)) * 180 / M_PI;
        //
        //     // Create an ellipse item
        //     QGraphicsEllipseItem* ellipse = scene->addEllipse(-majorAxis / 2, -minorAxis / 2, majorAxis, minorAxis);
        //     //ellipse->setPos(params(3) * 1000, params(4) * 1000); // Center of the ellipse
        //     ellipse->setPos(params(0) * 1000, params(1) * 1000); // Center of the ellipse
        //     ellipse->setRotation(angle);
        //     ellipse->setBrush(Qt::transparent);
        //     ellipse->setPen(QPen(colors[i], 15));
        //     items.push_back(ellipse);
        // }
    }

    ///////////// DRAW FUNCTIONS /////////////////
    void ActionableThing::draw_box(const std::vector<Eigen::AlignedBox2d> &boxes, QGraphicsScene *scene)
    {
        static std::vector<QGraphicsItem*> items;
        for(const auto i: items)
        {  scene->removeItem(i); delete i; }
        items.clear();

        // Draw the box on the scene
        for (const auto &box: boxes)
            items.emplace_back(scene->addRect( box.min().x(), box.min().y(), box.sizes().x(), box.sizes().y(),
                                               QPen(Qt::magenta, 20)));
    }

    void ActionableThing::draw_fridge(const auto &params, const QColor &color, QGraphicsScene *scene) const
    {
        static std::vector<QGraphicsItem*> items;
        for(const auto i: items)
        {scene->removeItem(i); delete i;}
        items.clear();

        // Extract params
        // const double width = params(0) * 1000;
        // const double depth = params(1) * 1000;
        // const double theta = params(2);
        // const double x = params(3) * 1000;
        // const double y = params(4) * 1000;
        const double width = params(3) * 1000;
        const double depth = params(4) * 1000;
        const double theta = params(2);
        const double x = params(0) * 1000;
        const double y = params(1) * 1000;
        //qDebug() << __FUNCTION__<< " Fridge params: " << width << depth << theta << x << y;

        // Generate QRect based on the fridge parameters
        const QRectF rect(-width/2, -depth/2, width, depth);
        const QPolygonF poly = QPolygonF(rect);
        // Rotate the fridge
        QTransform transform;
        transform.translate(x, y);
        transform.rotateRadians(theta);
        const QPolygonF rotated_poly = transform.map(poly);
        // Draw the fridge
        const auto item = scene->addPolygon(rotated_poly, QPen(color, 20));
        items.push_back(item);
    }

    void ActionableThing::draw_qrects(const std::vector<QRectF> &rects, QGraphicsScene *scene)
    {
        static std::vector<QGraphicsItem*> items;
        for(const auto i: items)
        {
            scene->removeItem(i);
            delete i;
        }
        items.clear();

        // Draw the box on the scene
        for (const auto &box: rects)
            items.push_back(scene->addRect( box.x(), box.y(),box.width(), box.height(), QPen(Qt::red, 20)));
    }

    void ActionableThing::draw_clusters(const std::vector<cv::RotatedRect> &rects, QGraphicsScene *scene)
    {
        static std::vector<QGraphicsItem*> items;
        for(const auto i: items)
        {
            scene->removeItem(i);
            delete i;
        }
        items.clear();

        // Draw the box on the scene
        for (const auto &box: rects)
        {
            auto box_draw = scene->addRect(-box.size.width/2, -box.size.height/2,
                                                             box.size.width, box.size.height,
                                                             QPen(Qt::red, 20));
            box_draw->setPos(box.center.x , box.center.y);
            box_draw->setRotation(-box.angle);
        }
    }

    double ActionableThing::compute_total_error(const Match &matches)
    {
        if (matches.empty()) {
            return std::numeric_limits<double>::max();
        }
        return std::accumulate(matches.begin(), matches.end(), 0.0,
                               [](double acc, const auto& m) { return acc + std::get<4>(m); });
    }

    void ActionableThing::remove_thing_draw(QGraphicsScene *scene)
    {
        if (thing_draw != nullptr)
         { scene->removeItem(thing_draw); delete thing_draw;}
    }

    void ActionableThing::set_thing_opacity(float opacity)
    {
        thing_draw->setOpacity(opacity);
    }

    void ActionableThing::define_actionable_color(const QColor &color)
    {
        this->color = color;
    }

    void ActionableThing::set_error(double error)
    {
        error_buffer.push_back(error);
        this->error = error;
    }

    QGraphicsItem * ActionableThing::get_thing_draw() const
    {
        return thing_draw;
    }

    QColor ActionableThing::get_actionable_color() const
    {
        return color;
    }

    double ActionableThing::get_error() const
    {
        return error;
    }

    double ActionableThing::get_buffer_error() const
    {
        if (error_buffer.empty()) return 0.0; // Si el buffer está vacío, devuelve 0
        return std::accumulate(error_buffer.begin(), error_buffer.end(), 0.0) / static_cast<float>(error_buffer.size());
    }

    double ActionableThing::get_prediction_fitness(double eps) const
    {
        return 1.0 / (1.0 + (eps * get_buffer_error()));
    }

    void ActionableThing::set_energy(double energy_)
    {
       energy = energy_;
    }

    bool ActionableThing::operator==(const ActionableThing &other) const
    {
        return true;
//         if(get_room().get_rotation() == other.get_room().get_rotation())
//            return fabs(other.get_room().get_depth() - get_room().get_depth()) < 300 and
//                   fabs(other.get_room().get_width() - get_room().get_width()) < 300;
//
//        return fabs(other.get_room().get_depth() - get_room().get_width()) < 300 and
//               fabs(other.get_room().get_width() - get_room().get_depth()) < 300;
    }

    //Function draw_point to draw one red point given Eigen::Vector3d, scene
    void ActionableThing::draw_point(Eigen::Vector3d point, QGraphicsScene *scene, bool erase)
    {
        static std::vector<QGraphicsItem*> items;
        for(const auto i: items)
        { scene->removeItem(i); delete i; }
        items.clear();

        if (erase) return;

        const auto color = QColor(Qt::darkRed);
        const auto brush = QBrush(QColor(Qt::darkRed));

        const auto item = scene->addRect(-50, -50, 100, 100, color, brush);
        item->setPos(point.x() * 1000, point.y() * 1000);
        items.push_back(item);
    };

    void ActionableThing::draw_residuals_in_room_frame(const std::vector<Eigen::Vector3d> &points, QGraphicsScene *scene, bool erase)
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
            item->setPos(p.x()*1000, p.y()*1000);
            items.push_back(item);
        }
    };


} // rc

 /**
     * \brief Computes the translation of the fridge to the mass center of the residual points.
     *
     * This function computes the mass center of the given residual points and creates a factor graph
     * to optimize the fridge's position (only cx, cy params) to this mass center.
     *
     * \param residual_points A vector of 3D points representing the residuals after the room detection.
     * \return The optimized 2D position of the fridge.
     */
//    gtsam::Vector2 ActionableThing::factor_graph_expr_translation(const std::vector<Eigen::Vector3d> &residual_points)
//    {
//        std::cout << "---------- TRANSLATION --------------------------------" << std::endl;
//        // Compute the mass center using accumulate
//        Eigen::Vector3d mass_center = std::accumulate(residual_points.begin(), residual_points.end(), Eigen::Vector3d(0.0, 0.0, 0.0)) / static_cast<float>(residual_points.size());
//        std::cout << "Mass center: " << mass_center.transpose() << std::endl;
//
//        // Create a constant expression for the initial nominal fridge parameters
//        gtsam::Vector2 measurement(mass_center.x(), mass_center.y());
//        gtsam::Expression<gtsam::Vector2> measurement_(measurement);
//
//        //---- Create Variable with noise
//        const gtsam::Symbol fridgeSym('f', 1);
//        gtsam::Key fridgeKey = fridgeSym.key();
//
//        // Define a noise model (e.g., isotropic noise)
//        auto noiseModel = gtsam::noiseModel::Isotropic::Sigma(2, 0.1);  // Adjust sigma as needed
//
//        // Create an expression for the fridge params
//        gtsam::Expression<gtsam::Vector2> fridge_(fridgeKey);
//
//        // Create an expression for the error (difference)
//        gtsam::Expression<gtsam::Vector2> proj_trans_(&factors::project_pose, fridge_);
//
//        //////////////////////////////////////
//        // Create a Factor Graph
//        gtsam::ExpressionFactorGraph graph;
//
//        // Provide initial estimates for the fridge center
//        gtsam::Values initialEstimate;
//
//        // Add a random offset to the initial estimate using std::random
//        static std::random_device rd;
//        static std::mt19937 gen(rd());
//        std::uniform_real_distribution<double> dis(-0.5, 0.5);
//        mass_center += Eigen::Vector3d(dis(gen), dis(gen), dis(gen));
//
//        // set initial estimate
//        auto initial_fridge = gtsam::Vector2(mass_center.x(), mass_center.y());
//        initialEstimate.insert(fridgeKey, initial_fridge);  // Initial guess (can be arbitrary)
//        std::cout << "Initial guess " << initialEstimate.at<gtsam::Vector2>(fridgeKey).transpose() << std::endl;
//
//        ///////////////////////////////
//        // Add custom expression factor for each liDAR point
//        //////////////////////////////
//        // Add the fridge variable to the graph
//        // auto noise_fridge_params = gtsam::noiseModel::Isotropic::Sigma(2, 0.1);
//        // graph.addExpressionFactor(fridge_, initial_fridge, noise_fridge_params);
//
//        // Add the likelihood to the graph
//        graph.addExpressionFactor(proj_trans_, measurement, noiseModel);  // measurement is zero sin we want to minimize the distance
//
//        // Calcular el error final del gráfico de factores
//        double initialError = graph.error(initialEstimate);
//        std::cout << "Error initial: " << initialError << std::endl;
//
//        /// ---------------- OPTIMIZATION ----------------------
//        gtsam::LevenbergMarquardtParams params;
//        params.maxIterations = 250;  // Número máximo de iteraciones
//        params.absoluteErrorTol = 1e-7;  // Tolerancia de error absoluto
//        params.relativeErrorTol = 1e-6;  // Tolerancia de error relativo
//        //params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::VALUES;  // Nivel de verbosidad
//        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
//        gtsam::Values result = optimizer.optimize();
//
//        /// Show optimization results
//        this->error = graph.error(result);
//        std::cout << "Final error: " << this->error << std::endl;
//        std::cout << "Optimized fridge center: " << result.at<gtsam::Vector2>(fridgeKey).transpose() << std::endl;
//
//        // Compute the marginal covariance for the 5D variable
//        //std::cout.precision(3);
//        //gtsam::Marginals marginals(graph, result);
//        //gtsam::Matrix covariance = marginals.marginalCovariance(fridgeKey);
//        //std::cout << "covariance:\n " << covariance << std::endl; //TODO: show the reduction in variance after optimization
//        // auto res = result.at<gtsam::Vector5>(fridgeKey);
//        // plotUncertaintyEllipses(covariance, res, scene);
//        return result.at<gtsam::Vector2>(fridgeKey);
//    }


//    std::vector<Eigen::Vector3d> ActionableThing::add_new_points(const std::vector<Eigen::Vector3d> &points)
//    {
//        // Create nanoflann structure from residuals_queue to filter close points
//        NanoFlannPointCloud cloud;
//
//        using RetMatches = std::vector<std::pair<unsigned, nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double,
//                NanoFlannPointCloud>,
//                NanoFlannPointCloud, 3>::DistanceType>>;
//
////        using RetMatches = std::vector<nanoflann::ResultItem<uint32_t, Eigen::Vector3d>>;
//        nanoflann::SearchParams params;
//        constexpr double search_radius = 0.025 * 0.025; // Squared radius
//
//        //Build cloud with residuals_queue and points append
//        for (const auto &pt: residuals_queue)
//            cloud.pts.emplace_back(pt.x(), pt.y(), pt.z());
//
//        //Get index at end off cloud size
//        int index = cloud.pts.size();
//        //Append points to cloud
//        for (const auto &pt: points)
//            cloud.pts.emplace_back(pt.x(), pt.y(), pt.z());
//
//        //Create bool vector = True with size == cloud
//        std::vector<bool> insertable(cloud.pts.size(), true);
//
//        //Create KDTree
//        KDTree tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
//        tree.buildIndex();
//
//
//        //for pt in cloud == index check if neighbour in cloud
//        //    if == 1 -> insertable = True
//        //    else -> neighbour = false
//        for (int i = index; i < cloud.pts.size(); i++)
//        {
//            if (insertable[i] == false)
//                continue;
//            RetMatches ret_matches;
//            auto result = tree.radiusSearch(cloud.pts[i].data(), search_radius, ret_matches, params);
//            if (result != 1)
//            {
//                for (const auto &match : ret_matches)
//                {
////                    printf("Matched point %d at distance %f\n", match.first, match.second, index);
//                    if (match.first > index)
//                        insertable[match.first] = false;
//                    else if (match.first < index)
//                        insertable[i] = false;
//                }
//            }
//        }
//        residuals_queue.clear();
//        //for i in insertable == true -> insert in residuals_queue
//        for (int i = 0; i < cloud.pts.size(); i++)
//        {
//            if (insertable[i])
//            {
//                residuals_queue.insert(residuals_queue.end(), Eigen::Vector3d(cloud.pts[i].x(), cloud.pts[i].y(), cloud.pts[i].z()));
//            }
//        }
//
//        std::vector<Eigen::Vector3d> points_memory (residuals_queue.begin(), residuals_queue.end());
//        // Print sizes of residuals_queue and points
//        qInfo() << __FUNCTION__ << "Size " << residuals_queue.size() << " points in points";
//        return points_memory;
//    }

// Define a point cloud structure
// struct NanoFlannPointCloud
// {
//     std::vector<Eigen::Vector3d> pts;
//
//     // Must return the number of data points
//     inline size_t kdtree_get_point_count() const { return pts.size(); }
//
//     // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class
//     //                inline double kdtree_distance(const double *p1, const size_t idx_p2, size_t size) const {
//     //                    const Eigen::Vector3d &p2 = pts[idx_p2];
//     //                    return (Eigen::Vector3d(p1) - p2).squaredNorm();
//     //                }
//
//     // Returns the dim'th component of the idx'th point in the class
//     inline double kdtree_get_pt(const size_t idx, int dim) const{
//         if (dim == 0)
//             return pts[idx].x();
//         else if (dim == 1)
//             return pts[idx].y();
//         else
//             return pts[idx].z();
//     }
//
//     // Optional bounding-box computation
//     template <class BBOX>
//     bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }
// };

    // Eigen::AlignedBox2d ActionableThing::fit_rectangle_to_lidar_points(const LidarPoints &points,
    //                                                                    const rc::ActionableRoom &current_room)
    // {
    //     // transform all residual points to the room frame
    //     std::vector<Eigen::Vector3d> points_room_frame; points_room_frame.reserve(points.size());
    //     auto rpose = current_room.get_robot_pose().matrix();
    //     for (const auto &point : points)
    //         points_room_frame.emplace_back(rpose * point);
    //
    //     // Compute the centroid of the lidar points
    //     auto centroid = std::accumulate(points_room_frame.begin(), points_room_frame.end(), Eigen::Vector3d(0.0, 0.0, 1.0)) /
    //                                      static_cast<float>(points_room_frame.size());
    //
    //     // Find the closest room model line
    //     auto lines = current_room.room.get_room_lines_eigen();
    //     const auto closest_line = std::ranges::min_element(lines, [&centroid](auto &line1, auto &line2)
    //         { return line1.distance(centroid.head(2)) < line2.distance(centroid.head(2)); });
    //
    //     // TODO:: get here an enum for left, right, top, bottom walls
    //
    //     if (closest_line == lines.end())
    //         return {};
    //
    //     // Compute the axis-aligned bounding rectangle in the rotated space
    //     Eigen::AlignedBox2d bounding_box;
    //     for (const auto &point : points_room_frame)
    //         bounding_box.extend(point.head(2));
    //
    //     // // Adjust the rectangle to touch the line
    //     const Eigen::Vector2d rectangleEdge = closest_line->origin() + closest_line->direction() *
    //                              (closest_line->direction().dot(bounding_box.center() - closest_line->origin()));
    //     Eigen::Vector2d translation = rectangleEdge - bounding_box.center();
    //
    //     // Adjust the translation to make the bounding box adjacent to the line
    //     const Eigen::Vector2d normal = Eigen::Vector2d(-closest_line->direction().y(), closest_line->direction().x()).normalized();
    //     translation += normal * (bounding_box.sizes().x() / 2.0);
    //     bounding_box.translate(translation);
    //
    //     // check that all points are inside
    //     for (const auto &point : points_room_frame)
    //         bounding_box.extend(point.head(2));
    //
    //     return bounding_box;
    // }
