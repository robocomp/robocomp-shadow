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
#include <tuple>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/base/Vector.h>
#include <gtsam/slam/expressions.h>
//#include <gtsam/nonlinear/expressions.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

namespace rc
{
    ActionableThing::Params ActionableThing::params;

    ActionableThing::ActionableThing(const std::shared_ptr<rc::ActionablesData> &innermodel_, QObject* parent) : QObject(parent), inner_model(innermodel_)
    {
        //error_buffer.set_capacity(25);
        residuals_queue.set_capacity(500);
        //inner_model = innermodel_; // do not move since we want that the sender retains ownership also
        inner_model->table = std::make_shared<rc::Table>();
    }

    bool ActionableThing::initialize(const rc::ActionableRoom &current_room,
                                     const std::vector<Eigen::Vector3d> residuals,  // residuals in room frame and in meters
                                     QGraphicsScene *scene)
    {
        //TODO: Solve initialize function to pass actionable_type

        std::cout.precision(3);

        // Add the points to the circular buffer
        const auto points_memory = add_new_points(residuals);

        //qInfo() << __FILE__ << __FUNCTION__ << "In Initialize: points_memory set size: " << points_memory.size();

        Eigen::Vector3d mass_center = std::accumulate(points_memory.begin(), points_memory.end(), Eigen::Vector3d(0.0, 0.0, 0.0)) / static_cast<float>(points_memory.size());
        draw_point(mass_center, scene);

        // Add noise to the mass center z == rand uniform 0,25
        //mass_center.z() += 0.25 * (2.0 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 1.0);

        //print mass_center values x, y, z
        //std::cout << __FUNCTION__ << " Mass center: " << mass_center.x() << " " << mass_center.y() << " " << mass_center.z()  << std::endl;


        // augment the state vector
        inner_model->table = std::make_shared<rc::Table>();
         inner_model->table->means = {mass_center.x(), mass_center.y(), mass_center.z()/2,
                                         params.INIT_ALPHA_VALUE, params.INIT_BETA_VALUE, params.INIT_GAMMA_VALUE,
                                         params.INIT_WIDTH_VALUE, params.INIT_WIDTH_VALUE, params.INIT_HEIGHT_VALUE};
        // inner_model->table->means = {0.5, 0.5, mass_center.z()/2,
        //                                        params.INIT_ALPHA_VALUE, params.INIT_BETA_VALUE, params.INIT_GAMMA_VALUE,
        //                                        params.INIT_WIDTH_VALUE, params.INIT_WIDTH_VALUE, params.INIT_HEIGHT_VALUE};

        // Then minimise the sum of residual distances to their closest fridge side
        inner_model->table->means = factor_graph_expr_points_table_top(points_memory, inner_model->table->means, current_room, mass_center, scene);

        draw_table(inner_model->table->means, Qt::red, scene);
        draw_residuals_in_room_frame(points_memory, scene);
        return true;
    }

    LidarPoints ActionableThing::project(const LidarPoints &residuals,  // residuals in room frame and in meters
                                         const rc::ActionableRoom &current_room,
                                         QGraphicsScene *scene)
    {
        //qInfo() << __FUNCTION__ << "------------------ project ---------------------";
        // remove from points all the points that are inside the rectangle: TO BE DONE
        LidarPoints filtered_points;

        if (clean_buffer)
        {
            this->points_memory_buffer.clear();
            clean_buffer = false;
        }

        //Create std::vector<Eigen::Vector3d> points_memory_fake
        std::vector<Eigen::Vector3d> points_memory_fake;
        //Add x,y,z points to points_memory_fake
        points_memory_fake.emplace_back(Eigen::Vector3d(-1, 1, 0.7));
        points_memory_fake.emplace_back(Eigen::Vector3d(1, 1, 0.7));
        //points_memory_fake.emplace_back(Eigen::Vector3d(-0.5, 1, 0.7));
        //points_memory_fake.emplace_back(Eigen::Vector3d(0.5, 1, 0.7));
        points_memory_fake.emplace_back(Eigen::Vector3d(-1, -1, 0.7));
        points_memory_fake.emplace_back(Eigen::Vector3d(1, -1, 0.7));
        // points_memory_fake.emplace_back(Eigen::Vector3d(-0.5, -1, 0.7));
        // points_memory_fake.emplace_back(Eigen::Vector3d(0.5, -1, 0.7));
        //points_memory_fake.emplace_back(Eigen::Vector3d(0, 0, 0.7));


        // add the points to the circular buffer and return a vector with the new points
        const auto points_memory = add_new_points(residuals);
        //const auto points_memory = points_memory_fake;
        //const auto points_memory = generatePerimeterPoints(1, 1.5 ,0.7, 0., 0., 0.1, 0.5);

        // compute the mass center of the points
        const Eigen::Vector3d mass_center = std::accumulate(points_memory.begin(), points_memory.end(), Eigen::Vector3d(0.0, 0.0, 0.0)) / static_cast<float>(points_memory.size());
        //draw_point(mass_center, scene);
        //Print mass_center x, y, z
        //std::cout << __FUNCTION__ << " Mass center: " << mass_center.x() << " " << mass_center.y() << " " << mass_center.z() << std::endl;
        // minimise the sum of residual distances to their closest table side

        static bool first_time = true;
        if (first_time or reset_optimiser)
        {
            first_time = true;
            inner_model->table->means = factor_graph_expr_points_table_top(points_memory, inner_model->table->means, current_room, mass_center, scene);
            draw_table(inner_model->table->means, Qt::red, scene);
            draw_residuals_in_room_frame(points_memory, scene);
        }


        return filtered_points;
    }

    gtsam::Vector9 ActionableThing::factor_graph_expr_points_table_top( const std::vector<Eigen::Vector3d> &residual_points, // residuals in room frame and in meters
                                                                    const gtsam::Vector9 &initial_table,
                                                                    const rc::ActionableRoom &current_room,
                                                                    const Eigen::Vector3d &mass_center,
                                                                    QGraphicsScene *scene)
    {
        //        std::cout << "---------- factor_graph_expr_points --------------------------------" << std::endl;
        //---- Create Variable with noise
        const gtsam::Symbol tableSym('f', 1);
        gtsam::Key tableKey = tableSym.key();

        //Print residual points size
        std::cout << __FUNCTION__ << " Residual points size: " << residual_points.size() << std::endl;

        // Create an expression for the table params
        gtsam::Expression<gtsam::Vector9> table_(tableKey);

        ///// Priors (x, y, alpha, w, d) //////////////////////////////
        auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector9(params.PRIOR_CX_SIGMA,
                                                                              params.PRIOR_CY_SIGMA,
                                                                              params.PRIOR_CZ_SIGMA,
                                                                              params.PRIOR_ALPHA_SIGMA,
                                                                              params.PRIOR_BETA_SIGMA,
                                                                              params.PRIOR_GAMMA_SIGMA,
                                                                              params.PRIOR_WIDTH_SIGMA,
                                                                              params.PRIOR_DEPTH_SIGMA,
                                                                              params.PRIOR_HEIGHT_SIGMA));

        ///////////////////////////////////////////////////////////////////////////////////////////////////
        /// FACTORS
        ///////////////////////////////////////////////////////////////////////////////////////////////////
        auto factor_table_top_height = [table_](const gtsam::Vector3 &p)
        {
            return
                    gtsam::Double_(&factors::dist2table_top, table_, gtsam::Point3_(&factors::table2top, table_, gtsam::Point3_ (&factors::room2table, table_, gtsam::Point3_(p))));
        };

        // Define a noise model for the points loss (e.g., isotropic noise)
        auto noise_model_height = gtsam::noiseModel::Isotropic::Sigma(1, params.HEIGHT_SIGMA);

        auto factor_table_top_fit = [table_](const gtsam::Vector3 &p)-> gtsam::Double_
        {
            return {
                    gtsam::Double_(&factors::min_dist_to_side_top, table_, gtsam::Double_(1.0),
                            gtsam::Point3_(&factors::room2table, table_, gtsam::Point3_(p)))};
        };


        // Define a noise model for the points loss (e.g., isotropic noise)
        auto noise_model_points = gtsam::noiseModel::Isotropic::Sigma(1, params.POINTS_SIGMA);

        ////////////////////////////////////////////////////////7
        /// Define the close_to_wall likelihood function
        ////////////////////////////////////////////////////////
        //  Transform the center of the fridge to each room wall frame, the compute the distance to the wall as the y abs of the y coordinate and get the minimum of the four distances.
        //auto sw = current_room.get_room().get_width_meters()/2.0;
        //auto sd = current_room.get_room().get_depth_meters()/2.0;
        //auto sw = inner_model->room->get_width_meters()/2.0; // Now we get the room from the inner model
        //auto sd = inner_model->room->get_depth_meters()/2.0;

        // Define a noise model for the adj loss (e.g., isotropic noise)
        //auto noise_model_adj = gtsam::noiseModel::Isotropic::Sigma(1, params.ADJ_SIGMA);

        ///////////////////////////////////////////////////////////////////
        /// Define the align_to_wall likelihood function


        ///////////////////////////////////////////////////////////////////
        // Create a Factor Graph
        //////////////////////////////////////////////////////////////////
        gtsam::ExpressionFactorGraph graph;

        // Add the fridge variable to the graph
        graph.addExpressionFactor(table_, initial_table, prior_noise);

        /// Add custom expression factor for each liDAR point. The total weight should be normalized by the size of residual_points
        for (const auto &pp: residual_points)
        {
            graph.addExpressionFactor(factor_table_top_fit(pp), 0.0, noise_model_points);  // measurement is zero since we want to minimize the distance
        }
        // stick_to_floor factor
        //graph.addExpressionFactor(gtsam::Double_(&factors::stick_to_floor, table_), 0.0, noise_model_height);

//        /// Add custom expression factor for closest wall adjacency
//        graph.addExpressionFactor(proj_adj_, 0.0, noise_model_adj);  // measurement is zero since we want to minimize the distance
//
//        /// Add custom expression factor for alignment
//        graph.addExpressionFactor(proj_align_, 0.0, noise_model_align);  // measurement is zero since we want to minimize the distance

        //////////////////////////////////////////////////////////////////
        // Provide initial estimates for the table
        ///////////////////////////////////////////////////////////////////
        gtsam::Values initialEstimate;
        auto initial_table_value = initial_table;   // from previous optimization

        if (reset_optimiser)    // if the UI buttom is clicked, add random noise to the initialFridge vector
        {
//            // add random noise to the initialFridge vector
            auto noise = gtsam::Vector9::Random() * 0.12;
            initial_table_value = gtsam::Vector9(mass_center.x() + noise.x(),
                                                 mass_center.y() + noise.y(),
                                                 mass_center.z() + noise.z(),
                                                 params.INIT_ALPHA_VALUE + noise[3],
                                                 params.INIT_BETA_VALUE + noise[4],
                                                 params.INIT_GAMMA_VALUE + noise[5],
                                                 params.INIT_WIDTH_VALUE + noise[6],
                                                 params.INIT_DEPTH_VALUE + noise[7],
                                                 params.INIT_HEIGHT_VALUE + noise[8]);
            reset_optimiser = false;
        }
        //Print initial table value
        std::cout << __FUNCTION__ << " Initial table value: " << initial_table_value.transpose() << std::endl;

        initialEstimate.insert(tableKey, initial_table_value);

        double initialError = graph.error(initialEstimate);

        /// ---------------- OPTIMIZATION ----------------------
        try
        {
            //gtsam::GaussNewtonParams params;
            gtsam::LevenbergMarquardtParams params;
            //params.maxIterations = 100;  // Número máximo de iteraciones
            params.absoluteErrorTol = 1e-9;  // Tolerancia de error absoluto
            params.relativeErrorTol = 1e-9;  // Tolerancia de error relativo
            //params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::VALUES;  // Nivel de verbosidad

            gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
            //gtsam::GaussNewtonOptimizer optimizer(graph, initialEstimate);

            gtsam::Values result = optimizer.optimize();
            this->error = graph.error(result);
            qInfo() << __FUNCTION__ << "Optimization done";
            std::cout << __FUNCTION__ << " Error initial: " << initialError << " - Final error: " << this->error << std::endl;
            std::cout << __FUNCTION__ << " Optimized table center: " << result.at<gtsam::Vector9>(tableKey).transpose() << std::endl;

            //gtsam::Marginals marginals(graph, result);
            //this->covariance = marginals.marginalCovariance(tableKey);
            //std::cout << "covariance:\n " << covariance << std::endl;
            //Compute and print the covariance matrix determinant
            //            std::cout << __FUNCTION__ << " Covariance determinant: " << covariance.determinant() << std::endl;
            // Compute and print matrix trace
            //            std::cout << __FUNCTION__ << "Covariance trace: " << covariance.trace() << std::endl;
            //plotUncertaintyEllipses(covariance, result.at<gtsam::Vector9>(tableKey), scene);

            return result.at<gtsam::Vector9>(tableKey);
        }
        catch (const gtsam::IndeterminantLinearSystemException &e){ std::cout << e.what() << std::endl; };


//         // Compute the marginal covariance for the 5D variable
//         try
//         {
//             gtsam::Marginals marginals(graph, result);
//             this->covariance = marginals.marginalCovariance(tableKey);
//             //std::cout << "covariance:\n " << covariance << std::endl;
//             //Compute and print the covariance matrix determinant
// //            std::cout << __FUNCTION__ << " Covariance determinant: " << covariance.determinant() << std::endl;
//             // Compute and print matrix trace
// //            std::cout << __FUNCTION__ << "Covariance trace: " << covariance.trace() << std::endl;
//             plotUncertaintyEllipses(covariance, result.at<gtsam::Vector5>(tableKey), scene);
//         }
//         catch (const gtsam::IndeterminantLinearSystemException &e){ std::cout << e.what() << std::endl; };
//
//         //Return empty vector
         return gtsam::Vector9();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
     std::vector<Eigen::Vector3d> ActionableThing::add_new_points(const std::vector<Eigen::Vector3d> &points)
    {
        auto is_item_selected = [](double probability)
        {
            std::random_device rd;  // Obtain a random number from hardware
            std::mt19937 gen(rd()); // Seed the generator
            std::uniform_real_distribution<> distr(0.0, 1.0); // Define the range
            return distr(gen) < probability;
        };

        const Eigen::Vector3d mass_center = std::accumulate(points_memory_buffer.begin(), points_memory_buffer.end(), Eigen::Vector3d(0.0, 0.0, 0.0)) / static_cast<float>(points_memory_buffer.size());
        for (const auto &point: points)
        {
            // Generate a lambda function to calculate the distance between the point and every
            // point in aux vector and stops for the first point that is closer than 0.1 meters
            auto it = std::ranges::find_if(points_memory_buffer, [&point](const Eigen::Vector3d &p)
                                           { return (point - p).norm() < 0.1; }); //TODO: magic number
            // If the point is not found, add it to the aux vector
            // Add it with more probability if it far from the mass center
            if (not points_memory_buffer.empty())
            {
                const double dist_to_center = (point - mass_center).norm();
                // pass this distance through a normal distribution with mean 0 and stddev 1
                // and compute a random number between 0 and 1 using std::random to decide if to add the point
                if (const double prob = std::exp(-0.05 * std::pow(dist_to_center, 2)); it == points_memory_buffer.end() and is_item_selected(1-prob))
                    points_memory_buffer.push_back(point); //TODO: magic number
            }
            else
               if (it == points_memory_buffer.end())
                 points_memory_buffer.push_back(point);
        }

        //auto end_time = std::chrono::high_resolution_clock::now();
        //auto duration = std::chrono::duration_cast<std::chrono::microseconds >(end_time - start_time);
        // qDebug() << __FUNCTION__ << "Time to add new points: " << duration.count() << "micros";
        return points_memory_buffer;
    }


    std::vector<Eigen::Vector3d> ActionableThing::generatePerimeterPoints(double width, double depth, double height, double centerX,
                                                                          double centerY, double distance, double ang)
    {
        std::vector<Eigen::Vector3d> perimeterPoints;

        // Calculate the number of points along each side
        int numPointsWidth = static_cast<int>(width / distance);
        int numPointsDepth = static_cast<int>(depth / distance);

        // Rotation matrix
        Eigen::Matrix2d rotation;
        rotation << std::cos(ang), -std::sin(ang),
                    std::sin(ang),  std::cos(ang);

        // Generate points along the width sides
        for (int i = 0; i <= numPointsWidth; ++i) {
            double x = centerX - width / 2 + i * distance;
            Eigen::Vector2d point1 = rotation * Eigen::Vector2d(x, centerY - depth / 2);
            Eigen::Vector2d point2 = rotation * Eigen::Vector2d(x, centerY + depth / 2);
            perimeterPoints.emplace_back(point1.x(), point1.y(), height);
            perimeterPoints.emplace_back(point2.x(), point2.y(), height);
        }

        // Generate points along the depth sides
        for (int i = 1; i < numPointsDepth; ++i) {
            double y = centerY - depth / 2 + i * distance;
            Eigen::Vector2d point1 = rotation * Eigen::Vector2d(centerX - width / 2, y);
            Eigen::Vector2d point2 = rotation * Eigen::Vector2d(centerX + width / 2, y);
            perimeterPoints.emplace_back(point1.x(), point1.y(), height);
            perimeterPoints.emplace_back(point2.x(), point2.y(), height);
        }

        return perimeterPoints;
    }

    std::tuple<Eigen::Matrix3d, Eigen::Vector3d>
    ActionableThing::compute_covariance_matrix(const std::vector<Eigen::Vector3d>& points)
    {
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();

        for (const auto& point : points)
            mean += point;
        mean /= static_cast<double>(points.size());

        for (const auto& point : points)
        {
            Eigen::Vector3d centered = point - mean;
            covariance += centered * centered.transpose();
        }
        covariance /= static_cast<double>(points.size()) - 1.0;
        return {covariance, mean};
    }

     std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d, Eigen::Matrix3d>
     ActionableThing::compute_OBB(const std::vector<Eigen::Vector3d>& points)
    {
        const auto &[covariance, mean] = compute_covariance_matrix(points);
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        auto rotation = solver.eigenvectors();

        Eigen::Vector3d minCorner = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
        Eigen::Vector3d maxCorner = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());

        for (const auto& point : points)
        {
            Eigen::Vector3d transformed = rotation.transpose() * (point - mean);
            minCorner = minCorner.cwiseMin(transformed);
            maxCorner = maxCorner.cwiseMax(transformed);
        }
        minCorner = rotation * minCorner + mean;
        maxCorner = rotation * maxCorner + mean;

        return {covariance, minCorner, maxCorner, rotation};
    }

    void ActionableThing::plotUncertaintyEllipses(const Eigen::MatrixXd& covariance, const Eigen::Matrix<double, 9, 1> &params, QGraphicsScene* scene)
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
        //const double scale = std::sqrt(5.991); // Chi-squared distribution for 2 degrees of freedom

        //const std::vector<QColor> colors = {QColor("yellow"), QColor("cyan"), QColor("orange"), QColor("blue"), QColor("green")};
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
        const double radius = 1000 * covariance.trace();     //TODO: magic number
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

    void ActionableThing::draw_table(const auto &params, const QColor &color, QGraphicsScene *scene) const
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
        const double width = params(6) * 1000;
        const double depth = params(7) * 1000;
        const double theta = params(5);
        const double x = params(0) * 1000;
        const double y = params(1) * 1000;

        //qDebug() << __FUNCTION__<< " Table params: " << "cx " << x << " cy " << y << " theta " << theta << "width " << width << "depth " << depth;

        // Generate QRect based on the table parameters
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


//     gtsam::Vector9 ActionableThing::factor_graph_expr_points_table( const std::vector<Eigen::Vector3d> &residual_points, // residuals in room frame and in meters
//                                                               const gtsam::Vector9 &initial_table,
//                                                               const rc::ActionableRoom &current_room,
//                                                               bool reset_optimiser,
//                                                               const Eigen::Vector3d &mass_center,
//                                                               QGraphicsScene *scene)
//     {
//         //        std::cout << "---------- factor_graph_expr_points --------------------------------" << std::endl;
//         //---- Create Variable with noise
//         const gtsam::Symbol tableSym('f', 1);
//         gtsam::Key tableKey = tableSym.key();
//
//         //Print residual points size
//         std::cout << "Residual points size: " << residual_points.size() << std::endl;
//
//         // Create an expression for the fridge params
//         gtsam::Expression<gtsam::Vector9> table_(tableKey);
//
//         ///// Priors (x, y, alpha, w, d) //////////////////////////////
//         auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector9(params.PRIOR_CX_SIGMA,
//                                                                               params.PRIOR_CY_SIGMA,
//                                                                               params.PRIOR_CZ_SIGMA,
//                                                                               params.PRIOR_ALPHA_SIGMA,
//                                                                               params.PRIOR_BETA_SIGMA,
//                                                                               params.PRIOR_GAMMA_SIGMA,
//                                                                               params.PRIOR_WIDTH_SIGMA,
//                                                                               params.PRIOR_DEPTH_SIGMA,
//                                                                               params.PRIOR_HEIGHT_SIGMA));
//
//
//
//         //////// Points likelihood function /////////////////////////////
//         auto proj_points_ = [table_](const gtsam::Vector3 &p, const size_t num_points)
//         {
//             return
//             gtsam::Double_(&factors::dist2table, table_,gtsam::Point3_(p));
//
//         };
//         // Define a noise model for the points loss (e.g., isotropic noise)
//         auto noise_model_points = gtsam::noiseModel::Isotropic::Sigma(1, params.POINTS_SIGMA);
//
//         ////////////////////////////////////////////////////////7
//         /// Define the close_to_wall likelihood function
//         ////////////////////////////////////////////////////////
//         //  Transform the center of the fridge to each room wall frame, the compute the distance to the wall as the y abs of the y coordinate and get the minimum of the four distances.
//         //auto sw = current_room.get_room().get_width_meters()/2.0;
//         //auto sd = current_room.get_room().get_depth_meters()/2.0;
//         //auto sw = inner_model->room->get_width_meters()/2.0; // Now we get the room from the inner model
//         //auto sd = inner_model->room->get_depth_meters()/2.0;
//
//
//         // Define a noise model for the adj loss (e.g., isotropic noise)
//         auto noise_model_adj = gtsam::noiseModel::Isotropic::Sigma(1, params.ADJ_SIGMA);
//
//         ///////////////////////////////////////////////////////
//         /// Define the align_to_wall likelihood function
//
//
//
//         //////////////////////////////////////
//         // Create a Factor Graph
//         gtsam::ExpressionFactorGraph graph;
//
//         /// Add custom expression factor for each liDAR point. The total weight should be normalized by the size of residual_points
//         for (const auto &pp: residual_points){
//             graph.addExpressionFactor(gtsam::Double_(&factors::dist2table, table_,gtsam::Point3_(pp)), 0.0, noise_model_points);  // measurement is zero since we want to minimize the distance
//
//         }
//
//
// //        /// Add custom expression factor for closest wall adjacency
// //        graph.addExpressionFactor(proj_adj_, 0.0, noise_model_adj);  // measurement is zero since we want to minimize the distance
// //
// //        /// Add custom expression factor for alignment
// //        graph.addExpressionFactor(proj_align_, 0.0, noise_model_align);  // measurement is zero since we want to minimize the distance
//
//         //////////////////////////////////////////////////////////////////
//         // Provide initial estimates for the fridge center
//         ///////////////////////////////////////////////////////////////////
//         gtsam::Values initialEstimate;
//         auto initial_table_value = initial_table;
//
//         //Print initial table value
//         std::cout << "Initial table value: " << initial_table_value.transpose() << std::endl;
//
//         if (reset_optimiser)    // if the UI buttom is clicked, add random noise to the initialFridge vector
//         {
// //            // add random noise to the initialFridge vector
//             auto noise = gtsam::Vector9::Random() * 0.12;
//             initial_table_value = gtsam::Vector9(mass_center.x() + noise.x(),
//                                                   mass_center.y() + noise.y(),
//                                                   mass_center.z() + noise.z(),
//                                                   params.INIT_ALPHA_VALUE + noise[3],
//                                                   params.INIT_BETA_VALUE + noise[4],
//                                                   params.INIT_GAMMA_VALUE + noise[5],
//                                                   params.INIT_WIDTH_VALUE + noise[6],
//                                                   params.INIT_DEPTH_VALUE + noise[7],
//                                                   params.INIT_HEIGHT_VALUE + noise[8]);
//         }
//         initialEstimate.insert(tableKey, initial_table_value);
//
//         double initialError = graph.error(initialEstimate);
//
//         // graph.print("Factor Graph:\n");
//
//         /// ---------------- OPTIMIZATION ----------------------
//         gtsam::LevenbergMarquardtParams params;
//         params.maxIterations = 2;  // Número máximo de iteraciones
//         params.absoluteErrorTol = 1e-5;  // Tolerancia de error absoluto
//         params.relativeErrorTol = 1e-5;  // Tolerancia de error relativo
//         //params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::VALUES;  // Nivel de verbosidad
//         gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
//         gtsam::Values result = optimizer.optimize();
//
//         /// Show optimization results
//         this->error = graph.error(result);
//         std::cout << __FUNCTION__ << " Error initial: " << initialError << " - Final error: " << this->error << std::endl;
//         std::cout << __FUNCTION__ << "Optimized table center: " << result.at<gtsam::Vector9>(tableKey).transpose() << std::endl;
//
//         // Compute the marginal covariance for the 5D variable
// //        try
// //        {
// //            gtsam::Marginals marginals(graph, result);
// //            this->covariance = marginals.marginalCovariance(tableKey);
// //            //std::cout << "covariance:\n " << covariance << std::endl;
// //            //Compute and print the covariance matrix determinant
// ////            std::cout << __FUNCTION__ << " Covariance determinant: " << covariance.determinant() << std::endl;
// //            // Compute and print matrix trace
// ////            std::cout << __FUNCTION__ << "Covariance trace: " << covariance.trace() << std::endl;
// //            plotUncertaintyEllipses(covariance, result.at<gtsam::Vector5>(tableKey), scene);
// //        }
// //        catch (const gtsam::IndeterminantLinearSystemException &e){ std::cout << e.what() << std::endl; };
// //        return result.at<gtsam::Vector9>(tableKey);
//
//         //Return empty vector
//         return gtsam::Vector9();
//     }
