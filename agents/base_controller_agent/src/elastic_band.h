/*
 * Elastic Band Local Path Planner with Obstacle Avoidance
 * Generates a deformable path from robot to target, repels the path away
 * from lidar obstacles via internal/external forces, and produces velocity commands.
 */

#ifndef ELASTIC_BAND_H
#define ELASTIC_BAND_H

#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QColor>
#include <QPen>
#include <QBrush>

class ElasticBand
{
public:
    struct Params
    {
        float k_internal = 0.08f;            // Internal spring gain (contraction toward neighbors)
        float k_external = 3.0f;            // External repulsion gain from obstacles
        float obstacle_influence = 569.f;  // Max distance (mm) at which obstacles exert repulsion
        float bubble_spacing = 150.f;       // Target spacing between bubbles (mm)
        float safety_margin = 150.f;        // Extra margin beyond robot half-width (mm)
        int   iterations = 12;               // Optimization iterations per cycle
        float robot_semi_width = 125.f;     // Robot half-width (mm)
        float robot_semi_length = 240.f;    // Robot half-length (mm)
        float max_adv = 600.f;              // Max advance velocity (mm/s)
        float max_side = 150.f;             // Max side velocity (mm/s)
        float max_rot = 0.6f;              // Max rotation velocity (rad/s)
        float lookahead = 600.f;            // Pure-pursuit lookahead distance (mm)
        float min_clearance_to_stop = 200.f;// If path clearance < this, stop (mm)
    };

    struct Bubble
    {
        Eigen::Vector2f pos;
        float clearance = std::numeric_limits<float>::max(); // Distance to nearest obstacle
    };

    ElasticBand() = default;
    explicit ElasticBand(const Params &params);

    /// Initialize a straight-line path from start to target with evenly spaced bubbles
    void initialize(const Eigen::Vector2f &start, const Eigen::Vector2f &target);

    /// Optimize the band: apply internal contraction + external repulsion forces
    void optimize(const std::vector<Eigen::Vector3f> &lidar_points);

    /// Compute velocity commands using pure-pursuit on the elastic band path
    std::tuple<float, float, float> compute_velocities() const;

    /// Get current path as list of 2D positions
    std::vector<Eigen::Vector2f> get_path() const;

    /// Check if the path is collision-free (all bubble clearances above threshold)
    bool is_valid() const;

    /// Check if band has been initialized and has bubbles
    bool is_active() const { return bubbles_.size() >= 2; }

    /// Draw the elastic band on a QGraphicsScene
    void draw(QGraphicsScene *scene, QColor path_color = QColor("orange"), QColor bubble_color = QColor(255, 165, 0, 60));

    /// Clear the band
    void clear();

    /// Set new target - returns true if target changed significantly
    bool update_target(const Eigen::Vector2f &new_target, float threshold = 300.f);

    /// Get minimum clearance along the path
    float min_clearance() const;

    /// Get minimum clearance along the path, skipping first skip_n bubbles
    float path_min_clearance(size_t skip_n = 0) const;

    Params params_;

private:
    std::vector<Bubble> bubbles_;
    Eigen::Vector2f current_target_ = Eigen::Vector2f::Zero();
    bool initialized_ = false;

    // Drawing items
    std::vector<QGraphicsItem*> draw_items_;

    /// Compute clearance (distance to nearest lidar point) for a position
    float compute_clearance(const Eigen::Vector2f &pos, const std::vector<Eigen::Vector3f> &lidar_points) const;

    /// Insert bubbles where spacing is too large, remove where too small
    void redistribute_bubbles();

    /// Update clearances for all bubbles
    void update_clearances(const std::vector<Eigen::Vector3f> &lidar_points);

    /// Check if a point is inside the robot body exclusion zone
    bool is_inside_robot_body(const Eigen::Vector2f &pos) const;

    /// Push a position out of the robot body to the nearest edge
    void push_out_of_robot_body(Eigen::Vector2f &pos) const;

    /// Remove bubbles (except first and last) that are inside the robot body
    void prune_bubbles_inside_robot();
};

#endif // ELASTIC_BAND_H

