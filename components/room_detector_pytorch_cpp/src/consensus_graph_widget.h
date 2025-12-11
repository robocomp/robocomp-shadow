/*
 *    Copyright (C) 2025 by Pablo Bustos
 *
 *    This file is part of RoboComp - CORTEX
 *
 *    Qt Widget for visualizing GTSAM factor graphs
 */

#ifndef CONSENSUS_GRAPH_WIDGET_H
#define CONSENSUS_GRAPH_WIDGET_H

#include <QWidget>
#include <QPainter>
#include <QMouseEvent>
#include <QWheelEvent>
#include <vector>
#include <map>
#include <memory>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <Eigen/Dense>

// Forward declarations
class NodeInfoDialog;
class ConsensusGraph;
enum class WallID;

/**
 * @brief Visual representation of a factor graph node
 */
struct GraphNode
{
    std::string label;           // Node label (e.g., "R", "r0", "L0")
    QPointF position;            // Position in graph layout space
    gtsam::Pose2 pose;          // Actual pose value (for pose nodes)
    QColor color;                // Node color
    float radius;                // Node radius
    bool is_fixed;               // Whether this node is fixed (e.g., room at origin)
    Eigen::Matrix3d covariance;  // Covariance matrix (if available)
};

/**
 * @brief Visual representation of a factor graph edge
 */
struct GraphEdge
{
    size_t from_node;           // Index into nodes vector
    size_t to_node;             // Index into nodes vector
    std::string factor_type;    // "prior", "between", "odometry", etc.
    QColor color;               // Edge color
    float thickness;            // Edge thickness
    bool is_constraint;         // Whether this is a constraint (rigid)
    double error;               // Factor error (contribución a la energía)
};

/**
 * @brief Configuration for graph visualization
 */
struct GraphVisConfig
{
    // Node appearance
    float room_node_radius = 30.0f;
    float wall_node_radius = 20.0f;
    float robot_node_radius = 15.0f;
    float object_node_radius = 18.0f;

    QColor room_color = QColor(100, 150, 255);     // Blue
    QColor wall_color = QColor(150, 150, 150);     // Gray
    QColor robot_color = QColor(255, 100, 100);    // Red
    QColor object_color = QColor(100, 255, 100);   // Green
    QColor fixed_color = QColor(50, 50, 50);       // Dark (fixed nodes)

    // Edge appearance
    QColor prior_edge_color = QColor(100, 100, 100, 150);      // Gray
    QColor between_edge_color = QColor(50, 150, 200, 150);     // Blue
    QColor odometry_edge_color = QColor(200, 100, 50, 150);    // Orange
    QColor constraint_edge_color = QColor(50, 50, 50, 200);    // Dark (rigid)
    QColor observation_edge_color = QColor(100, 200, 100, 150); // Green

    float prior_edge_thickness = 2.0f;
    float between_edge_thickness = 1.5f;
    float constraint_edge_thickness = 3.0f;

    // Layout parameters
    float horizontal_spacing = 150.0f;   // Spacing between columns
    float vertical_spacing = 80.0f;      // Spacing between rows
    float room_to_wall_distance = 120.0f;
    float wall_spread_angle = 90.0f;     // Degrees for wall arrangement

    // Covariance visualization
    bool show_covariances = true;
    float covariance_scale = 3.0f;       // Scale factor for 1-sigma ellipses
    QColor covariance_color = QColor(255, 200, 0, 100); // Yellow, semi-transparent

    // Labels
    bool show_labels = true;
    QFont label_font = QFont("Arial", 10);
    QColor label_color = QColor(0, 0, 0);

    // Info display
    bool show_error_info = true;
    QFont info_font = QFont("Courier", 9);
};

/**
 * @brief Event types for graph changes
 */
enum class GraphEventType
{
    ROOM_INITIALIZED,
    ROBOT_POSE_ADDED,
    OBJECT_ADDED,
    OBSERVATION_ADDED,
    CONSTRAINT_ADDED,
    OPTIMIZATION_STARTED,
    OPTIMIZATION_COMPLETED,
    GRAPH_CLEARED
};

/**
 * @brief Structure for passing graph update information via signals
 */
struct GraphUpdateEvent
{
    GraphEventType type;
    const gtsam::NonlinearFactorGraph* graph = nullptr;
    const gtsam::Values* values = nullptr;
    std::map<size_t, Eigen::Matrix3d> covariances;

    // Event-specific data
    double initial_error = 0.0;
    double final_error = 0.0;
    int iterations = 0;

    // Node indices for incremental updates
    size_t node_index = 0;
    std::string node_label;
};

/**
 * @brief Qt Widget for visualizing GTSAM factor graphs with signal/slot architecture
 *
 * Features:
 * - Event-driven updates via Qt signals
 * - Hierarchical layout: Room at center, walls around it, robots and objects in columns
 * - Color-coded nodes by type (room, wall, robot, object)
 * - Different edge styles for different factor types
 * - Covariance ellipse visualization
 * - Pan and zoom controls
 * - Automatic layout update when graph changes
 * - Incremental updates for efficiency
 *
 * Usage:
 *   // Connect signals from your graph manager
 *   connect(manager, &Manager::graphUpdated, widget, &ConsensusGraphWidget::onGraphUpdated);
 *   connect(manager, &Manager::optimizationCompleted, widget, &ConsensusGraphWidget::onOptimizationCompleted);
 */
class ConsensusGraphWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ConsensusGraphWidget(QWidget* parent = nullptr);
    ~ConsensusGraphWidget() override;

    /**
     * @brief Set visualization configuration
     */
    void setConfig(const GraphVisConfig& config);

    /**
     * @brief Get current configuration
     */
    const GraphVisConfig& getConfig() const { return config_; }

    /**
     * @brief Enable/disable auto-fit on update
     */
    void setAutoFit(bool enabled) { auto_fit_ = enabled; }

public Q_SLOTS:
    /**
     * @brief Handle graph update event (main entry point for all updates)
     *
     * This is the primary slot for receiving graph changes. It processes
     * the event and triggers appropriate visualization updates.
     */
    void onGraphUpdated(const GraphUpdateEvent& event);

    /**
     * @brief Update visualization from ConsensusGraph
     *
     * Legacy method - can still be called directly if needed.
     */
    void updateFromGraph(const ConsensusGraph* graph);

    /**
     * @brief Update visualization from GTSAM graph and values directly
     */
    void updateFromGTSAM(const gtsam::NonlinearFactorGraph& graph,
                         const gtsam::Values& values,
                         const std::map<size_t, Eigen::Matrix3d>& covariances = {});

    /**
     * @brief Handle room initialization event
     */
    void onRoomInitialized(const gtsam::NonlinearFactorGraph* graph,
                          const gtsam::Values* values);

    /**
     * @brief Handle robot pose added event
     */
    void onRobotPoseAdded(size_t robot_index,
                         const gtsam::NonlinearFactorGraph* graph,
                         const gtsam::Values* values);

    /**
     * @brief Handle object (door) added event
     */
    void onObjectAdded(size_t object_index,
                      const gtsam::NonlinearFactorGraph* graph,
                      const gtsam::Values* values);

    /**
     * @brief Handle observation added event
     */
    void onObservationAdded(size_t from_index, size_t to_index,
                           const gtsam::NonlinearFactorGraph* graph,
                           const gtsam::Values* values);

    /**
     * @brief Handle optimization started event
     */
    void onOptimizationStarted();

    /**
     * @brief Handle optimization completed event
     */
    void onOptimizationCompleted(double initial_error,
                                 double final_error,
                                 int iterations,
                                 const gtsam::NonlinearFactorGraph* graph,
                                 const gtsam::Values* values,
                                 const std::map<size_t, Eigen::Matrix3d>& covariances = {});

    /**
     * @brief Handle graph cleared event
     */
    void onGraphCleared();

    /**
     * @brief Reset view to show entire graph
     */
    void fitToView();

Q_SIGNALS:
    /**
     * @brief Emitted when user clicks on a node
     */
    void nodeClicked(size_t node_index, const std::string& label);

    /**
     * @brief Emitted when user hovers over a node
     */
    void nodeHovered(size_t node_index, const std::string& label);

    /**
     * @brief Emitted when visualization update is complete
     */
    void visualizationUpdated();

    /**
     * @brief Emitted when view changes (pan/zoom)
     */
    void viewChanged(const QRectF& visible_area);

protected:
    // Qt event handlers
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    // Layout computation
    void computeLayout();
    void layoutRoomAndWalls();
    void layoutRobots();
    void layoutObjects();

    // Node/edge extraction from GTSAM
    void extractNodes(const gtsam::Values& values,
                      const std::map<size_t, Eigen::Matrix3d>& covariances);
    void extractEdges(const gtsam::NonlinearFactorGraph& graph,
                      const gtsam::Values& values);

    // Drawing functions
    void drawBackground(QPainter& painter);
    void drawEdges(QPainter& painter);
    void drawNodes(QPainter& painter);
    void drawCovariances(QPainter& painter);
    void drawLabels(QPainter& painter);
    void drawInfoPanel(QPainter& painter);

    // Coordinate transformations
    QPointF graphToWidget(const QPointF& graph_pos) const;
    QPointF widgetToGraph(const QPointF& widget_pos) const;
    float graphToWidgetScale(float graph_size) const;

    // Helper functions
    size_t findNodeByKey(uint64_t key) const;
    std::string symbolToString(uint64_t key) const;
    void drawEllipse(QPainter& painter, const QPointF& center,
                     const Eigen::Matrix2d& covariance, float scale);
    void drawArrow(QPainter& painter, const QPointF& from, const QPointF& to,
                   const QColor& color, float thickness);

    // Graph data
    std::vector<GraphNode> nodes_;
    std::vector<GraphEdge> edges_;
    std::map<uint64_t, size_t> key_to_node_index_;  // GTSAM key to node index

    // Configuration
    GraphVisConfig config_;

    // View transform
    QPointF view_offset_;        // Pan offset
    float view_scale_;           // Zoom scale
    bool auto_fit_;              // Auto-fit on update

    // Interaction state
    bool is_dragging_;
    QPoint last_mouse_pos_;
    int hovered_node_index_;

    // Node info popup
    std::unique_ptr<NodeInfoDialog> node_info_dialog_;
    std::string active_info_label_;

    // Optimization info
    bool has_optimization_info_;
    double initial_error_;
    double final_error_;
    int iterations_;
    bool is_optimizing_;
    double loop_error_;

    // Layout bounds (in graph coordinates)
    QRectF graph_bounds_;

    // Incremental update tracking
    bool needs_full_layout_;

    // Node hit testing
    int findNodeAtPosition(const QPointF& widget_pos) const;
    void updateTooltipForNode(int node_idx);

};

#endif // CONSENSUS_GRAPH_WIDGET_H