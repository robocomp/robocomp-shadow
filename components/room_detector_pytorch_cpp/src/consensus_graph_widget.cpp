/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 */

#include "consensus_graph_widget.h"
#include "consensus_graph.h"
#include <QPainter>
#include <QPainterPath>
#include <cmath>
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ConsensusGraphWidget::ConsensusGraphWidget(QWidget* parent)
    : QWidget(parent)
    , view_offset_(0, 0)
    , view_scale_(1.0f)
    , auto_fit_(true)
    , is_dragging_(false)
    , hovered_node_index_(-1)
    , has_optimization_info_(false)
    , initial_error_(0.0)
    , final_error_(0.0)
    , loop_error_(0.0)
    , iterations_(0)
    , is_optimizing_(false)
    , needs_full_layout_(true)
{
    setMinimumSize(400, 300);
    setMouseTracking(true);

    // Default background
    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor(250, 250, 250));
    setPalette(pal);
    setAutoFillBackground(true);
}

void ConsensusGraphWidget::setConfig(const GraphVisConfig& config)
{
    config_ = config;
    update();
}

void ConsensusGraphWidget::onGraphCleared()
{
    nodes_.clear();
    edges_.clear();
    key_to_node_index_.clear();
    has_optimization_info_ = false;
    is_optimizing_ = false;
    hovered_node_index_ = -1;
    needs_full_layout_ = true;
    update();
    emit visualizationUpdated();
}

void ConsensusGraphWidget::onOptimizationCompleted(double initial_error,
                                                    double final_error,
                                                    int iterations,
                                                    const gtsam::NonlinearFactorGraph* graph,
                                                    const gtsam::Values* values,
                                                    const std::map<size_t, Eigen::Matrix3d>& covariances)
{
    has_optimization_info_ = true;
    initial_error_ = initial_error;
    final_error_ = final_error;
    iterations_ = iterations;
    is_optimizing_ = false;

    if (graph && values)
    {
        updateFromGTSAM(*graph, *values, covariances);
    }
    else
    {
        update();
    }

    emit visualizationUpdated();
}

void ConsensusGraphWidget::updateFromGraph(const ConsensusGraph* graph)
{
    if (!graph)
    {
        onGraphCleared();
        return;
    }

    const auto& gtsam_graph = graph->getGraph();
    const auto& values = graph->getValues();

    // We don't have direct access to covariances here, but that's okay
    updateFromGTSAM(gtsam_graph, values);
}

void ConsensusGraphWidget::onGraphUpdated(const GraphUpdateEvent& event)
{
    switch (event.type)
    {
        case GraphEventType::ROOM_INITIALIZED:
            if (event.graph && event.values)
                onRoomInitialized(event.graph, event.values);
            break;

        case GraphEventType::ROBOT_POSE_ADDED:
            if (event.graph && event.values)
                onRobotPoseAdded(event.node_index, event.graph, event.values);
            break;

        case GraphEventType::OBJECT_ADDED:
            if (event.graph && event.values)
                onObjectAdded(event.node_index, event.graph, event.values);
            break;

        case GraphEventType::OBSERVATION_ADDED:
            if (event.graph && event.values)
                onObservationAdded(0, event.node_index, event.graph, event.values);
            break;

        case GraphEventType::CONSTRAINT_ADDED:
            if (event.graph && event.values)
                updateFromGTSAM(*event.graph, *event.values, event.covariances);
            break;

        case GraphEventType::OPTIMIZATION_STARTED:
            onOptimizationStarted();
            break;

        case GraphEventType::OPTIMIZATION_COMPLETED:
            if (event.graph && event.values)
                onOptimizationCompleted(event.initial_error, event.final_error,
                                       event.iterations, event.graph, event.values,
                                       event.covariances);
            break;

        case GraphEventType::GRAPH_CLEARED:
            onGraphCleared();
            break;
    }
}

void ConsensusGraphWidget::onRoomInitialized(const gtsam::NonlinearFactorGraph* graph,
                                             const gtsam::Values* values)
{
    if (!graph || !values)
        return;

    // Full update for room initialization
    needs_full_layout_ = true;
    updateFromGTSAM(*graph, *values);

    std::cout << "ConsensusGraphWidget: Room initialized\n";
}

void ConsensusGraphWidget::onRobotPoseAdded(size_t robot_index,
                                            const gtsam::NonlinearFactorGraph* graph,
                                            const gtsam::Values* values)
{
    if (!graph || !values)
        return;

    // Incremental update - just recompute robot layout
    updateFromGTSAM(*graph, *values);

    std::cout << "ConsensusGraphWidget: Robot pose " << robot_index << " added\n";
}

void ConsensusGraphWidget::onObjectAdded(size_t object_index,
                                         const gtsam::NonlinearFactorGraph* graph,
                                         const gtsam::Values* values)
{
    if (!graph || !values)
        return;

    // Incremental update - just recompute object layout
    updateFromGTSAM(*graph, *values);

    std::cout << "ConsensusGraphWidget: Object " << object_index << " added\n";
}

void ConsensusGraphWidget::onObservationAdded(size_t from_index, size_t to_index,
                                              const gtsam::NonlinearFactorGraph* graph,
                                              const gtsam::Values* values)
{
    if (!graph || !values)
        return;

    // Just need to update edges
    updateFromGTSAM(*graph, *values);

    std::cout << "ConsensusGraphWidget: Observation added: " << from_index
              << " -> " << to_index << "\n";
}

void ConsensusGraphWidget::onOptimizationStarted()
{
    is_optimizing_ = true;
    update();

    std::cout << "ConsensusGraphWidget: Optimization started...\n";
}

void ConsensusGraphWidget::updateFromGTSAM(const gtsam::NonlinearFactorGraph& graph,
                                           const gtsam::Values& values,
                                           const std::map<size_t, Eigen::Matrix3d>& covariances)
{
    nodes_.clear();
    edges_.clear();
    key_to_node_index_.clear();
    loop_error_ = 0.0;   // << reset de energía de loops

    if (values.empty())
    {
        update();
        emit visualizationUpdated();
        return;
    }

    // Extract nodes from values
    extractNodes(values, covariances);

    // Extract edges from factors
    extractEdges(graph, values);

    if (hovered_node_index_ >= 0)
        updateTooltipForNode(hovered_node_index_);

    // Compute layout
    computeLayout();

    // Auto-fit if enabled
    if (auto_fit_)
    {
        fitToView();
    }

    update();
    emit visualizationUpdated();
}

void ConsensusGraphWidget::extractNodes(const gtsam::Values& values,
                                        const std::map<size_t, Eigen::Matrix3d>& covariances)
{
    // First pass: collect all robot pose keys to identify the latest pose
    std::vector<uint64_t> robot_keys;
    uint64_t last_robot_key = 0;
    bool has_robot = false;

    for (const auto& key_value : values)
    {
        uint64_t key = key_value.key;
        gtsam::Symbol sym(key);
        char chr = sym.chr();

        if (chr == 'r')
        {
            robot_keys.push_back(key);
            last_robot_key = key;
            has_robot = true;
        }
    }

    // Second pass: create visual nodes
    for (const auto& key_value : values)
    {
        uint64_t key = key_value.key;

        // Try to get Pose2
        try
        {
            gtsam::Pose2 pose = values.at<gtsam::Pose2>(key);

            gtsam::Symbol sym(key);
            char chr = sym.chr();

            // For robot poses, only create a node for the latest pose.
            // All older robot poses will be mapped to this node as self-edges
            // through key_to_node_index_ in the post-processing step.
            if (chr == 'r' && has_robot && key != last_robot_key)
            {
                continue;
            }

            GraphNode node;
            node.label = symbolToString(key);
            node.pose = pose;
            node.is_fixed = false;
            node.covariance = Eigen::Matrix3d::Identity();

            // If we have a covariance matrix for this variable, use it
            auto cov_it = covariances.find(static_cast<size_t>(key));
            if (cov_it != covariances.end())
            {
                node.covariance = cov_it->second;
            }

            // Determine node type and appearance from symbol
            switch (chr)
            {
                case 'R':  // Room
                    node.color = config_.room_color;
                    node.radius = config_.room_node_radius;
                    node.is_fixed = true;  // Room is fixed at origin
                    break;

                case 'W':  // Wall
                    node.color = config_.wall_color;
                    node.radius = config_.wall_node_radius;
                    break;

                case 'r':  // Robot (only latest pose becomes a node)
                    node.color = config_.robot_color;
                    node.radius = config_.robot_node_radius;
                    break;

                case 'L':  // Landmark/Object (doors)
                    node.color = config_.object_color;
                    node.radius = config_.object_node_radius;
                    break;

                default:
                    node.color = QColor(150, 150, 150);
                    node.radius = 15.0f;
                    break;
            }

            // Fixed nodes get darker color
            if (node.is_fixed)
            {
                node.color = config_.fixed_color;
            }

            size_t index = nodes_.size();
            nodes_.push_back(node);
            key_to_node_index_[key] = index;
        }
        catch (...)
        {
            // Skip if not a Pose2
            continue;
        }
    }

    // If we compressed multiple robot poses into a single visual node,
    // map all robot keys to that node index so that all factors touching
    // any robot pose become self-edges on the current robot node.
    if (has_robot)
    {
        auto it = key_to_node_index_.find(last_robot_key);
        if (it != key_to_node_index_.end())
        {
            size_t robot_node_index = it->second;
            for (uint64_t r_key : robot_keys)
            {
                key_to_node_index_[r_key] = robot_node_index;
            }
        }
    }
}


void ConsensusGraphWidget::extractEdges(const gtsam::NonlinearFactorGraph& graph,
                                        const gtsam::Values& values)
{
    for (size_t i = 0; i < graph.size(); ++i)
    {
        auto factor = graph[i];
        if (!factor) continue;

        auto keys = factor->keys();

        GraphEdge edge;
        edge.thickness = config_.between_edge_thickness;
        edge.is_constraint = false;
        edge.error = 0.0;

        // FACTOR CON UNA VARIABLE: PRIOR
        if (keys.size() == 1)
        {
            edge.factor_type = "prior";
            edge.color = config_.prior_edge_color;
            edge.thickness = config_.prior_edge_thickness;

            // Self-loop sobre el nodo correspondiente
            auto it = key_to_node_index_.find(keys[0]);
            if (it != key_to_node_index_.end())
            {
                edge.from_node = it->second;
                edge.to_node = it->second;

                // Error de este factor
                edge.error = factor->error(values);

                edges_.push_back(edge);
            }
        }
        // FACTOR CON DOS VARIABLES: BETWEEN
        else if (keys.size() == 2)
        {
            auto it_from = key_to_node_index_.find(keys[0]);
            auto it_to   = key_to_node_index_.find(keys[1]);

            if (it_from != key_to_node_index_.end() &&
                it_to   != key_to_node_index_.end())
            {
                edge.from_node = it_from->second;
                edge.to_node   = it_to->second;

                // Clasificar tipo de arista por tipo de nodo
                gtsam::Symbol sym_from(keys[0]);
                gtsam::Symbol sym_to(keys[1]);

                char chr_from = sym_from.chr();
                char chr_to   = sym_to.chr();

                // Room <-> Wall: rigid constraint
                if ((chr_from == 'R' && chr_to == 'W') ||
                    (chr_from == 'W' && chr_to == 'R'))
                {
                    edge.factor_type = "rigid_constraint";
                    edge.color = config_.constraint_edge_color;
                    edge.thickness = config_.constraint_edge_thickness;
                    edge.is_constraint = true;
                }
                // Wall <-> Object: attachment constraint
                else if ((chr_from == 'W' && chr_to == 'L') ||
                         (chr_from == 'L' && chr_to == 'W'))
                {
                    edge.factor_type = "wall_attachment";
                    edge.color = config_.constraint_edge_color;
                    edge.thickness = config_.constraint_edge_thickness;
                    edge.is_constraint = true;
                }
                // Robot <-> Robot: odometry
                else if (chr_from == 'r' && chr_to == 'r')
                {
                    edge.factor_type = "odometry";
                    edge.color = config_.odometry_edge_color;
                }
                // Robot <-> Room: room observation
                else if ((chr_from == 'r' && chr_to == 'R') ||
                         (chr_from == 'R' && chr_to == 'r'))
                {
                    edge.factor_type = "room_obs";
                    edge.color = config_.between_edge_color;
                }
                // Robot <-> Object: object observation
                else if ((chr_from == 'r' && chr_to == 'L') ||
                         (chr_from == 'L' && chr_to == 'r'))
                {
                    edge.factor_type = "object_obs";
                    edge.color = config_.observation_edge_color;
                }
                else
                {
                    edge.factor_type = "between";
                    edge.color = config_.between_edge_color;
                }

                // Error de este factor
                edge.error = factor->error(values);

                // Acumular energía de loops:
                // aquí consideramos como “loops” las observaciones que cierran lazo
                // entre trayectorias y el mapa (puertas, sala, etc).
                if (edge.factor_type == "room_obs" ||
                    edge.factor_type == "object_obs" ||
                    edge.factor_type == "wall_attachment")  // || edge.factor_type == "rigid_constraint") to include rigid constraints
                {
                    loop_error_ += edge.error;
                }

                edges_.push_back(edge);
            }
        }
    }
}

void ConsensusGraphWidget::computeLayout()
{
    if (nodes_.empty()) return;

    layoutRoomAndWalls();
    layoutRobots();
    layoutObjects();

    // Compute bounds
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    for (const auto& node : nodes_)
    {
        min_x = std::min(min_x, static_cast<float>(node.position.x() - node.radius));
        max_x = std::max(max_x, static_cast<float>(node.position.x() + node.radius));
        min_y = std::min(min_y, static_cast<float>(node.position.y() - node.radius));
        max_y = std::max(max_y, static_cast<float>(node.position.y() + node.radius));
    }

    graph_bounds_ = QRectF(QPointF(min_x, min_y), QPointF(max_x, max_y));
}

void ConsensusGraphWidget::layoutRoomAndWalls()
{
    // Place room at center
    for (auto& node : nodes_)
    {
        if (node.label[0] == 'R')
        {
            node.position = QPointF(0, 0);
        }
    }

    // Place walls in a circle around room
    float angle = 0.0f;
    float angle_step = 360.0f / 4.0f;  // 4 walls

    for (auto& node : nodes_)
    {
        if (node.label[0] == 'W')
        {
            float rad = angle * M_PI / 180.0f;
            float x = config_.room_to_wall_distance * std::cos(rad);
            float y = config_.room_to_wall_distance * std::sin(rad);
            node.position = QPointF(x, y);
            angle += angle_step;
        }
    }
}

void ConsensusGraphWidget::layoutRobots()
{
    // Place robots in a vertical column to the right
    float x = config_.horizontal_spacing * 3;
    float y = -config_.vertical_spacing * 1.5;  // Start at top

    for (auto& node : nodes_)
    {
        if (node.label[0] == 'r')
        {
            node.position = QPointF(x, y);
            y += config_.vertical_spacing;
        }
    }
}

void ConsensusGraphWidget::layoutObjects()
{
    // Place objects (doors) to the right of robots, since robot observes doors
    // This creates a visual flow: Room -> Walls -> Robot -> Door
    float x = config_.horizontal_spacing * 1.5;  // closer than robots
    float y = 0;  // Start at same height as robots

    for (auto& node : nodes_)
    {
        if (node.label[0] == 'L')
        {
            node.position = QPointF(x, y);
            y += config_.vertical_spacing;
        }
    }
}

void ConsensusGraphWidget::fitToView()
{
    if (graph_bounds_.isEmpty()) return;

    // Add padding
    float padding = 50.0f;
    QRectF padded_bounds = graph_bounds_.adjusted(-padding, -padding, padding, padding);

    // Calculate scale to fit
    float scale_x = width() / padded_bounds.width();
    float scale_y = height() / padded_bounds.height();
    view_scale_ = std::min(scale_x, scale_y);

    // Center the graph
    QPointF center = padded_bounds.center();
    view_offset_ = QPointF(width() / 2.0f - center.x() * view_scale_,
                          height() / 2.0f - center.y() * view_scale_);
}

void ConsensusGraphWidget::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    drawBackground(painter);

    if (!nodes_.empty())
    {
        drawEdges(painter);

        if (config_.show_covariances)
        {
            drawCovariances(painter);
        }

        drawNodes(painter);

        if (config_.show_labels)
        {
            drawLabels(painter);
        }
    }

    if (config_.show_error_info && has_optimization_info_)
    {
        drawInfoPanel(painter);
    }
}

void ConsensusGraphWidget::drawBackground(QPainter& painter)
{
    // Already filled by autoFillBackground
    // Draw subtle grid
    painter.setPen(QPen(QColor(230, 230, 230), 1));

    int grid_spacing = 50;

    for (int x = 0; x < width(); x += grid_spacing)
    {
        painter.drawLine(x, 0, x, height());
    }

    for (int y = 0; y < height(); y += grid_spacing)
    {
        painter.drawLine(0, y, width(), y);
    }
}

void ConsensusGraphWidget::drawEdges(QPainter& painter)
{
    for (const auto& edge : edges_)
    {
        const auto& from_node = nodes_[edge.from_node];
        const auto& to_node = nodes_[edge.to_node];

        QPointF from_pos = graphToWidget(from_node.position);
        QPointF to_pos = graphToWidget(to_node.position);

        // Don't draw self-loops (prior factors)
        if (from_pos == to_pos) continue;

        // Draw line
        QPen pen(edge.color, edge.thickness);

        if (edge.is_constraint)
        {
            pen.setStyle(Qt::DashLine);
        }

        painter.setPen(pen);

        // Draw arrow for directed edges
        if (edge.factor_type == "odometry" ||
            edge.factor_type == "room_obs" ||
            edge.factor_type == "object_obs")
        {
            drawArrow(painter, from_pos, to_pos, edge.color, edge.thickness);
        }
        else
        {
            painter.drawLine(from_pos, to_pos);
        }
    }
}

void ConsensusGraphWidget::drawNodes(QPainter& painter)
{
    for (size_t i = 0; i < nodes_.size(); ++i)
    {
        const auto& node = nodes_[i];
        QPointF pos = graphToWidget(node.position);
        float radius = graphToWidgetScale(node.radius);

        // Highlight hovered node
        bool is_hovered = (static_cast<int>(i) == hovered_node_index_);

        // Draw node circle
        QPen pen(Qt::black, is_hovered ? 3 : 2);
        painter.setPen(pen);

        QColor fill_color = node.color;
        if (is_hovered)
        {
            fill_color = fill_color.lighter(120);
        }
        painter.setBrush(QBrush(fill_color));
        painter.drawEllipse(pos, radius, radius);

        // Draw additional indicator for fixed nodes
        if (node.is_fixed)
        {
            painter.setPen(QPen(Qt::white, 3));
            float inner_radius = radius * 0.5f;
            painter.drawEllipse(pos, inner_radius, inner_radius);
        }

        // Draw pulsing ring if optimizing
        if (is_optimizing_)
        {
            painter.setPen(QPen(QColor(255, 150, 0), 2, Qt::DashLine));
            painter.setBrush(Qt::NoBrush);
            painter.drawEllipse(pos, radius * 1.3f, radius * 1.3f);
        }
    }
}

void ConsensusGraphWidget::drawCovariances(QPainter& painter)
{
    for (const auto& node : nodes_)
    {
        // Skip if no covariance or zero covariance
        if (node.covariance.isZero()) continue;

        QPointF pos = graphToWidget(node.position);

        // Extract 2D position covariance (top-left 2x2 block)
        Eigen::Matrix2d cov_2d = node.covariance.block<2, 2>(0, 0);

        drawEllipse(painter, pos, cov_2d, config_.covariance_scale);
    }
}

void ConsensusGraphWidget::drawLabels(QPainter& painter)
{
    painter.setFont(config_.label_font);
    painter.setPen(config_.label_color);

    for (const auto& node : nodes_)
    {
        QPointF pos = graphToWidget(node.position);
        float radius = graphToWidgetScale(node.radius);

        // Draw label below node
        QRectF text_rect(pos.x() - radius, pos.y() + radius + 5,
                        radius * 2, 20);
        painter.drawText(text_rect, Qt::AlignCenter, QString::fromStdString(node.label));
    }
}

void ConsensusGraphWidget::drawInfoPanel(QPainter& painter)
{
    painter.setFont(config_.info_font);

    // Position panel in bottom-right corner
    float panel_width = 200;
    float panel_height = 80;
    float margin = 10;

    QRectF panel_rect(

    margin,   // Left side
        height() - panel_height - margin,  // Bottom side
        panel_width,
        panel_height
    );

    // Draw semi-transparent background
    painter.fillRect(panel_rect, QColor(255, 255, 255, 200));
    painter.setPen(QPen(Qt::black, 1));
    painter.drawRect(panel_rect);

    // Draw text
    painter.setPen(Qt::black);
    double improvement = 0.0;
    if (initial_error_ > 0.0)
        improvement = (initial_error_ - final_error_) / initial_error_ * 100.0;

    QString info_text = QString("Initial error: %1\n"
                               "Final error:   %2\n"
                               "Loop energy:   %3\n"
                               "Iterations:    %4\n"
                               "Improvement:   %5%")
                        .arg(initial_error_, 0, 'f', 3)
                        .arg(final_error_, 0, 'f', 3)
                        .arg(loop_error_, 0, 'f', 3)
                        .arg(iterations_)
                        .arg(improvement, 0, 'f', 1);

    painter.drawText(panel_rect.adjusted(10, 10, -10, -10), Qt::AlignLeft | Qt::TextWordWrap, info_text);
}

QPointF ConsensusGraphWidget::graphToWidget(const QPointF& graph_pos) const
{
    return QPointF(graph_pos.x() * view_scale_ + view_offset_.x(),
                   graph_pos.y() * view_scale_ + view_offset_.y());
}

QPointF ConsensusGraphWidget::widgetToGraph(const QPointF& widget_pos) const
{
    return QPointF((widget_pos.x() - view_offset_.x()) / view_scale_,
                   (widget_pos.y() - view_offset_.y()) / view_scale_);
}

float ConsensusGraphWidget::graphToWidgetScale(float graph_size) const
{
    return graph_size * view_scale_;
}

void ConsensusGraphWidget::drawEllipse(QPainter& painter, const QPointF& center,
                                       const Eigen::Matrix2d& covariance, float scale)
{
    // Compute eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(covariance);
    Eigen::Vector2d eigenvalues = solver.eigenvalues();
    Eigen::Matrix2d eigenvectors = solver.eigenvectors();

    // Semi-axes lengths (scale by sigma multiplier)
    float a = std::sqrt(eigenvalues(0)) * scale;
    float b = std::sqrt(eigenvalues(1)) * scale;

    // Rotation angle
    float angle = std::atan2(eigenvectors(1, 0), eigenvectors(0, 0)) * 180.0 / M_PI;

    // Scale to widget coordinates
    a = graphToWidgetScale(a);
    b = graphToWidgetScale(b);

    // Draw ellipse
    painter.save();
    painter.translate(center);
    painter.rotate(angle);

    painter.setPen(QPen(config_.covariance_color, 1));
    painter.setBrush(Qt::NoBrush);
    painter.drawEllipse(QPointF(0, 0), a, b);

    painter.restore();
}

void ConsensusGraphWidget::drawArrow(QPainter& painter, const QPointF& from, const QPointF& to,
                                    const QColor& color, float thickness)
{
    QPen pen(color, thickness);
    painter.setPen(pen);

    // Draw line
    painter.drawLine(from, to);

    // Draw arrowhead
    QLineF line(from, to);
    double angle = std::atan2(-line.dy(), line.dx());

    float arrow_size = 10.0f;
    QPointF arrow_p1 = to - QPointF(std::sin(angle + M_PI / 3) * arrow_size,
                                    std::cos(angle + M_PI / 3) * arrow_size);
    QPointF arrow_p2 = to - QPointF(std::sin(angle + M_PI - M_PI / 3) * arrow_size,
                                    std::cos(angle + M_PI - M_PI / 3) * arrow_size);

    QPolygonF arrow_head;
    arrow_head << to << arrow_p1 << arrow_p2;
    painter.setBrush(QBrush(color));
    painter.drawPolygon(arrow_head);
}

size_t ConsensusGraphWidget::findNodeByKey(uint64_t key) const
{
    auto it = key_to_node_index_.find(key);
    if (it != key_to_node_index_.end())
        return it->second;
    return static_cast<size_t>(-1);
}

std::string ConsensusGraphWidget::symbolToString(uint64_t key) const
{
    gtsam::Symbol sym(key);
    return std::string(1, static_cast<char>(sym.chr())) + std::to_string(sym.index());
}

void ConsensusGraphWidget::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton)
    {
        // Check if clicking on a node
        int node_idx = findNodeAtPosition(event->pos());
        if (node_idx >= 0 && node_idx < static_cast<int>(nodes_.size()))
        {
            const auto& node = nodes_[node_idx];
            emit nodeClicked(node_idx, node.label);
            std::cout << "Node clicked: " << node.label << "\n";
        }

        is_dragging_ = true;
        last_mouse_pos_ = event->pos();
    }
}

void ConsensusGraphWidget::mouseMoveEvent(QMouseEvent* event)
{
    // Check for hover
    int node_idx = findNodeAtPosition(event->pos());
    if (node_idx != hovered_node_index_)
    {
        hovered_node_index_ = node_idx;
        if (node_idx >= 0 && node_idx < static_cast<int>(nodes_.size()))
        {
            const auto& node = nodes_[node_idx];
            emit nodeHovered(node_idx, node.label);
        }

        // Rebuild tooltip for the (possibly new) hovered node
        updateTooltipForNode(hovered_node_index_);

        update();
    }

    if (is_dragging_)
    {
        QPoint delta = event->pos() - last_mouse_pos_;
        view_offset_ += QPointF(delta);
        last_mouse_pos_ = event->pos();
        update();

        // Emit view changed signal
        QRectF visible = QRectF(widgetToGraph(QPointF(0, 0)),
                                widgetToGraph(QPointF(width(), height())));
        emit viewChanged(visible);
    }
}

void ConsensusGraphWidget::updateTooltipForNode(int node_idx)
{
    if (node_idx >= 0 && node_idx < static_cast<int>(nodes_.size()))
    {
        const auto& node = nodes_[node_idx];
        const gtsam::Pose2& p = node.pose;
        const Eigen::Matrix3d& C = node.covariance;

        QString tooltip = QString::fromStdString(node.label);
        tooltip += "\n";
        tooltip += QString("μ = [ %1, %2, %3 ]")
                       .arg(p.x(), 0, 'f', 3)
                       .arg(p.y(), 0, 'f', 3)
                       .arg(p.theta(), 0, 'f', 3);
        tooltip += "\nΣ = ["
                   + QString(" [%1, %2, %3];")
                         .arg(C(0,0), 0, 'f', 3)
                         .arg(C(0,1), 0, 'f', 3)
                         .arg(C(0,2), 0, 'f', 3)
                   + QString(" [%1, %2, %3];")
                         .arg(C(1,0), 0, 'f', 3)
                         .arg(C(1,1), 0, 'f', 3)
                         .arg(C(1,2), 0, 'f', 3)
                   + QString(" [%1, %2, %3] ]")
                         .arg(C(2,0), 0, 'f', 3)
                         .arg(C(2,1), 0, 'f', 3)
                         .arg(C(2,2), 0, 'f', 3);

        setToolTip(tooltip);
    }
    else
    {
        setToolTip("");
    }
}


void ConsensusGraphWidget::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton)
    {
        is_dragging_ = false;
    }
}

void ConsensusGraphWidget::wheelEvent(QWheelEvent* event)
{
    // Zoom in/out
    float zoom_factor = 1.1f;

    if (event->angleDelta().y() > 0)
    {
        view_scale_ *= zoom_factor;
    }
    else
    {
        view_scale_ /= zoom_factor;
    }

    // Clamp scale
    view_scale_ = std::max(0.1f, std::min(view_scale_, 10.0f));

    update();

    // Emit view changed signal
    QRectF visible = QRectF(widgetToGraph(QPointF(0, 0)),
                           widgetToGraph(QPointF(width(), height())));
    emit viewChanged(visible);
}

int ConsensusGraphWidget::findNodeAtPosition(const QPointF& widget_pos) const
{
    QPointF graph_pos = widgetToGraph(widget_pos);

    for (size_t i = 0; i < nodes_.size(); ++i)
    {
        const auto& node = nodes_[i];
        float dx = graph_pos.x() - node.position.x();
        float dy = graph_pos.y() - node.position.y();
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq <= node.radius * node.radius)
        {
            return static_cast<int>(i);
        }
    }

    return -1;
}

void ConsensusGraphWidget::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    if (auto_fit_ && !graph_bounds_.isEmpty())
    {
        fitToView();
    }
}