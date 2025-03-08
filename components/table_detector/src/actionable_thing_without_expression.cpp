//
// Created by pbustos on 9/12/24.
//

#include "actionable_thing.h"
#include <iosfwd>
#include <tuple>
#include <vector>
#include "pch.h"
#include <gtsam/nonlinear/AdaptAutoDiff.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>

namespace rc
{
    ActionableThing::ActionableThing()
    {
//        corner_buffer.set_capacity(56);
//        corner_robot_buffer.set_capacity(56);
        error_buffer.set_capacity(25);
    }



    bool ActionableThing::initialize(const ActionableThing &progenitor,
                                     const std::vector<ActionableThing> &actionables,
                                     const rc::ActionableRoom &current_room,
                                     const std::vector<Eigen::Vector3d> residuals,
                                     QGraphicsScene *scene)
    {

        /// OLD APPROACH USING DBSCAN
        // cluster the residuals
//        auto clusters = rc::dbscan_get_point_clusters(residuals, 300, 30);
//        box = fit_rectangle_to_lidar_points(clusters[0], current_room);
//        draw_box({box}, scene);

        /// NEW APPROACH USING GTSAM

        auto fridge = create_factor_graph(residuals, current_room.get_robot_pose(), scene);

        std::cout << "Optimized Fridge Parameters: " << fridge.transpose() << std::endl;
        draw_fridge(fridge, Qt::red, scene);
        return true;
    }

    gtsam::Vector5 ActionableThing::create_factor_graph(const std::vector<Eigen::Vector3d> &residual_points, const Eigen::Affine2d &robot_pose, QGraphicsScene *scene)
    {
        qDebug() << "######################## CREATING NEW FACTOR GRAPH ########################";
        //------------- INITIALIZE GRAPH -------------------//
        gtsam::NonlinearFactorGraph graph;
	// gtsam::ExpressionFactorGraph graph;
        //------------- FRIDGE VAR -----------------------------//
        gtsam::Vector5 initialFridge;


        std::vector<Eigen::Vector2d > points;
        std::ranges::transform(residual_points, std::back_inserter(points), [&robot_pose](const auto &point)
        {return (robot_pose * Eigen::Vector2d(point.x(), point.y()));});      /// to meters

        for (auto &p: points)
            p = p * 0.001; // to mm

        // Compute the mass center using accumulate
        Eigen::Vector2d mass_center = std::accumulate(points.begin(), points.end(), Eigen::Vector2d(0.0, 0.0)) / static_cast<float>(points.size());

        std::cout << "Mass Center =" << mass_center.x() << " " << mass_center.y() << std::endl;

        // Print mass center
//        qDebug() << "-----------------------------------------------";
        // add a random offset to the mass center using std::random
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-3, 3);
        auto noise_mass = Eigen::Vector2d(dis(gen), dis(gen));

        auto mass_center_noise = mass_center + noise_mass;
//        initialFridge << 0.6, 0.6, 0.0, 0.0, 0.0; // (w, d, alpha, x, y) Room frame
        initialFridge << 0.6, 0.6, 0.0, mass_center_noise.x() , mass_center_noise.y();
        //---- Create Variable with noise
        const gtsam::Symbol fridgeSym('f', 0);
        gtsam::Key fridgeKey = fridgeSym.key();

        //---- Variable Prior   (w, d, alpha, x, y)
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector5(0.1,0.1, 10, 10, 10)); //a > menor importancia
        //---- Add var to graph
        // auto fridgeExpression = gtsam::Expression<gtsam::Vector5>('f', 0);
        // graph.addExpressionFactor(fridgeExpression, initialFridge, priorNoise); // PRIOR
        // gtsam::noiseModel::Diagonal::shared_ptr expressionNoise =
        //         gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(0.1, 0.1));
        // auto h2 = gtsam::Expression<gtsam::Vector2>(&ActionableThing::distance, fridgeExpression);
        // graph.addExpressionFactor(h2, 0.f, expressionNoise);

        graph.add(gtsam::PriorFactor<gtsam::Vector5>(fridgeSym, initialFridge, priorNoise));

        //---- Initialization values of optimization
        gtsam::Values initial;
        initial.insert(fridgeKey, initialFridge);
//        initial.print("Initial estimate:\n");
        std::cout << "Initial Fridge =" << mass_center_noise.x() << " " << mass_center_noise.y() << std::endl;
        // Set identity matrix for now
        Eigen::Affine2d T_R_to_r = Eigen::Affine2d::Identity();

//        // Define a noise model (isotropic, sigma = 0.1).
        auto noiseModel = std::shared_ptr<gtsam::noiseModel::Isotropic>(gtsam::noiseModel::Isotropic::Sigma(1, 0.5));

//        // Create and add the custom factor to the graph.
        graph.add(std::make_shared<FridgePointsFactor>(fridgeKey, points, T_R_to_r, noiseModel));

        /// ---------------- OPTIMIZATION ----------------------
        // Optimize the factor graph using Levenberg-Marquardt.
        gtsam::LevenbergMarquardtParams params;
        params.maxIterations = 250;  // Número máximo de iteraciones
        params.relativeErrorTol = 1e-7;  // Tolerancia de error relativo
        params.absoluteErrorTol = 1e-7;  // Tolerancia de error absoluto
        params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::VALUES;  // Nivel de verbosidad
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
        gtsam::Values result = optimizer.optimize();

        /// Show optimization results
        result.print("Final result:\n");

        std::cout << "-------------- FIN OPT -------------------------" << std::endl;
        // Calcular el error final del gráfico de factores
        this->error = graph.error(result);

        // std::cout << "Error final: " << finalError << std::endl;
        // Compute the marginal covariance for the 5D variable
        gtsam::Marginals marginals(graph, result);
        gtsam::Matrix covariance = marginals.marginalCovariance(fridgeKey);
        auto res = result.at<gtsam::Vector5>(fridgeKey);
        plotUncertaintyEllipses(covariance, res, scene);
        // Retrieve and print the optimized fridge parameters.
        return result.at<gtsam::Vector5>(fridgeKey);
    }

    void ActionableThing::plotUncertaintyEllipses(const Eigen::MatrixXd& covariance, const Eigen::Matrix<double, 5, 1> &params, QGraphicsScene* scene)
    {
        // Compute the eigenvalues and eigenvectors of the covariance matrix
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance);
        Eigen::VectorXd eigenvalues = solver.eigenvalues();
        Eigen::MatrixXd eigenvectors = solver.eigenvectors();

        static std::vector<QGraphicsItem*> items;
        for(const auto i: items)
        {  scene->removeItem(i); delete i; }
        items.clear();

        // Scale factor for the ellipse (e.g., 95% confidence interval)
        double scale = std::sqrt(5.991); // Chi-squared distribution for 2 degrees of freedom

        std::vector<QColor> colors = {QColor("yellow"), QColor("green"), QColor("orange"), QColor("blue"), QColor("red")};
        for (int i = 0; i < eigenvalues.size(); ++i)
        {
            double majorAxis = 2000 * scale * std::sqrt(eigenvalues(i));
            double minorAxis = 2000 * scale * std::sqrt(eigenvalues(i));
            // Print the major and minor axes
            //std::cout << "Major axis: " << majorAxis << ", Minor axis: " << minorAxis << std::endl;
            // Compute the rotation angle (in degrees)
            double angle = std::atan2(eigenvectors(1, i), eigenvectors(0, i)) * 180 / M_PI;

            // Create an ellipse item
            QGraphicsEllipseItem* ellipse = scene->addEllipse(-majorAxis / 2, -minorAxis / 2, majorAxis, minorAxis);
            ellipse->setPos(params(3) * 1000, params(4) * 1000); // Center of the ellipse
            ellipse->setRotation(angle);
            ellipse->setBrush(Qt::transparent);
            ellipse->setPen(QPen(colors[i], 15));
            items.push_back(ellipse);
        }
    }

    Eigen::AlignedBox2d ActionableThing::fit_rectangle_to_lidar_points(const LidarPoints &points,
                                                                       const rc::ActionableRoom &current_room)
    {
        // transform all residual points to the room frame
        std::vector<Eigen::Vector3d> points_room_frame; points_room_frame.reserve(points.size());
        auto rpose = current_room.get_robot_pose().matrix();
        for (const auto &point : points)
            points_room_frame.emplace_back(rpose * point);

        // Compute the centroid of the lidar points
        auto centroid = std::accumulate(points_room_frame.begin(), points_room_frame.end(), Eigen::Vector3d(0.0, 0.0, 1.0)) /
                                         static_cast<float>(points_room_frame.size());

        // Find the closest room model line
        auto lines = current_room.room.get_room_lines_eigen();
        const auto closest_line = std::ranges::min_element(lines, [&centroid](auto &line1, auto &line2)
            { return line1.distance(centroid.head(2)) < line2.distance(centroid.head(2)); });

        // TODO:: get here an enum for left, right, top, bottom walls

        if (closest_line == lines.end())
            return {};

        // Compute the axis-aligned bounding rectangle in the rotated space
        Eigen::AlignedBox2d bounding_box;
        for (const auto &point : points_room_frame)
            bounding_box.extend(point.head(2));

        // // Adjust the rectangle to touch the line
        const Eigen::Vector2d rectangleEdge = closest_line->origin() + closest_line->direction() *
                                 (closest_line->direction().dot(bounding_box.center() - closest_line->origin()));
        Eigen::Vector2d translation = rectangleEdge - bounding_box.center();

        // Adjust the translation to make the bounding box adjacent to the line
        const Eigen::Vector2d normal = Eigen::Vector2d(-closest_line->direction().y(), closest_line->direction().x()).normalized();
        translation += normal * (bounding_box.sizes().x() / 2.0);
        bounding_box.translate(translation);

        // check that all points are inside
        for (const auto &point : points_room_frame)
            bounding_box.extend(point.head(2));

        return bounding_box;
    }

    LidarPoints ActionableThing::project(const LidarPoints &points, const rc::ActionableRoom &current_room)
    {
        // remove from points all the points that are inside the rectangle
        LidarPoints filtered_points;
        // get robot pose
        auto rpose = current_room.get_robot_pose().matrix();
        for (const auto &point: points)
        {
            if (not box.contains((rpose * point).head(2)))
                filtered_points.push_back(point);
        }
        return filtered_points;
    }

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
        const double width = params(0) * 1000;
        const double depth = params(1) * 1000;
        const double theta = params(2);
        const double x = params(3) * 1000;
        const double y = params(4) * 1000;
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
        auto item = scene->addPolygon(rotated_poly, QPen(color, 20));
        items.push_back(item);

        auto point = scene ->addEllipse(x, y ,20 ,20 ,QPen(color, 20));
        items.push_back(point);
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
        {
            auto box_draw = scene->addRect( box.x(), box.y(),box.width(), box.height(), QPen(Qt::red, 20));
        }
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
} // rc
