//
// Created by pbustos on 19/11/25.
//

#ifndef MODEL_BASED_FILTER_H
#define MODEL_BASED_FILTER_H


#include <torch/torch.h>
#include <Lidar3D.h>
#include "room_model.h"

/**
 * @brief Filters LiDAR points based on how well they fit the room model. Implements top down attention.
 *
 * In LOCALIZED mode with good fit, we can save computation by:
 * - Filtering out "explained" points (near walls, SDF ≈ 0)
 * - Keeping only "residual" points (far from model, potentially interesting)
 *
 * This implements hierarchical attention - focus on what's surprising.
 */

class ModelBasedFilter
{
public:
    struct Result
    {
        RoboCompLidar3D::TPoints explained_points;   // Fit model well (inliers)
        RoboCompLidar3D::TPoints residual_points;    // Don't fit (outliers/anomalies)
        RoboCompLidar3D::TPoints filtered_set;       // Recommended set for optimization

        float explained_ratio;                       // Fraction of inliers
        float mean_residual;                         // Mean SDF of residuals
        int points_saved;                            // Computation savings

        enum class Strategy {
            USE_ALL,          // Use all points (poor fit or mapping mode)
            USE_RESIDUALS,    // Use only residuals (good fit, many inliers)
            USE_HYBRID        // Downsampled inliers + all residuals
        } strategy;
    };

    struct Params
    {
        // Thresholds
        float inlier_threshold = 0.05f;              // SDF threshold for inliers (meters)
        float min_explained_ratio = 0.7f;             // Minimum % to enable residual-only mode
        int min_residual_points = 100;                // Minimum residuals for robust optimization

        // Downsampling
        int downsample_factor = 5;                    // Keep every Nth inlier in hybrid mode

        // Adaptive thresholding
        bool use_adaptive_threshold = true;             // Scale threshold with uncertainty
        float uncertainty_scale = 3.0f;                  // Multiplier for uncertainty (3-sigma)
        float min_threshold = 0.03f;                     // Minimum threshold (3cm)
        float max_threshold = 0.15f;                     // Maximum threshold (15cm)

        // Constructor con valores por defecto
        Params() = default;
    };

    explicit ModelBasedFilter() : params_(Params{}) {}

    /**
     * @brief Filter points based on model fit
     *
     * @param points Input LiDAR points
     * @param room Current room model
     * @param robot_uncertainty Optional position uncertainty for adaptive threshold
     * @return Filtering result with strategy recommendation
     */
    Result filter(const RoboCompLidar3D::TPoints& points,
                  std::shared_ptr<RoomModel>& room,
                  float robot_uncertainty = 0.0f);

    /**
     * @brief Compute statistics without filtering (for analysis)
     */
    Result analyze_fit(const RoboCompLidar3D::TPoints& points,
                       std::shared_ptr<RoomModel>& room);

    /**
     * @brief Update parameters
     */
    void set_params(const Params& params) { params_ = params; }
    Params get_params() const { return params_; }

    void print_result(int num_points, const Result &result) const;

private:
    Params params_;

    /**
     * @brief Compute adaptive threshold based on uncertainty
     */
    float compute_adaptive_threshold(float robot_uncertainty) const;

    /**
     * @brief Select best filtering strategy
     */
    Result::Strategy select_strategy(float explained_ratio,
                                      size_t num_residuals) const;

    /**
     * @brief Build filtered set according to strategy
     */
    RoboCompLidar3D::TPoints build_filtered_set(
        const RoboCompLidar3D::TPoints& explained,
        const RoboCompLidar3D::TPoints& residuals,
        Result::Strategy strategy) const;


};

#endif //MODEL_BASED_FILTER_H
