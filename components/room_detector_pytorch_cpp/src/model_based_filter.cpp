//
// Created by pbustos on 19/11/25.
//

#include "model_based_filter.h"
#include <QDebug>
#include <numeric>
#include "model_based_filter.h"

ModelBasedFilter::Result ModelBasedFilter::filter(
    const RoboCompLidar3D::TPoints& points,
    RoomModel& room,
    float robot_uncertainty)
{
    Result result;

    if (points.empty()) {
        result.strategy = Result::Strategy::USE_ALL;
        return result;
    }

    // Compute adaptive threshold
    float threshold = params_.inlier_threshold;
    if (params_.use_adaptive_threshold && robot_uncertainty > 0.0f) {
        threshold = compute_adaptive_threshold(robot_uncertainty);
    }

    // Convert points to tensor
    std::vector<float> points_data;
    points_data.reserve(points.size() * 2);
    for (const auto& p : points) {
        points_data.push_back(p.x / 1000.0f);
        points_data.push_back(p.y / 1000.0f);
    }

    torch::Tensor points_tensor = torch::from_blob(
        points_data.data(),
        {static_cast<long>(points.size()), 2},
        torch::kFloat32
    ).clone();

    // Compute SDF for all points
    torch::Tensor sdf_values = room.sdf(points_tensor);
    auto sdf_accessor = sdf_values.accessor<float, 1>();

    // Classify points
    std::vector<float> residual_sdfs;
    for (size_t i = 0; i < points.size(); ++i) {
        float sdf = sdf_accessor[i];
        float abs_sdf = std::abs(sdf);

        if (abs_sdf < threshold) {
            result.explained_points.push_back(points[i]);
        } else {
            result.residual_points.push_back(points[i]);
            residual_sdfs.push_back(abs_sdf);
        }
    }

    // Compute statistics
    result.explained_ratio =
        static_cast<float>(result.explained_points.size()) / points.size();

    result.mean_residual = residual_sdfs.empty() ? 0.0f :
        std::accumulate(residual_sdfs.begin(), residual_sdfs.end(), 0.0f) / residual_sdfs.size();

    // Select strategy
    result.strategy = select_strategy(result.explained_ratio,
                                      result.residual_points.size());

    // Build filtered set
    result.filtered_set = build_filtered_set(
        result.explained_points,
        result.residual_points,
        result.strategy
    );

    result.points_saved = points.size() - result.filtered_set.size();

    return result;
}

ModelBasedFilter::Result ModelBasedFilter::analyze_fit(
    const RoboCompLidar3D::TPoints& points,
    RoomModel& room)
{
    Result result = filter(points, room, 0.0f);
    result.strategy = Result::Strategy::USE_ALL;
    result.filtered_set = points;
    return result;
}

float ModelBasedFilter::compute_adaptive_threshold(float robot_uncertainty) const
{
    // Scale threshold with uncertainty (3-sigma rule)
    float adaptive = params_.uncertainty_scale * robot_uncertainty;

    // Clamp to reasonable bounds
    return std::clamp(adaptive, params_.min_threshold, params_.max_threshold);
}

ModelBasedFilter::Result::Strategy ModelBasedFilter::select_strategy(
    float explained_ratio,
    size_t num_residuals) const
{
    // Strategy 1: Use only residuals (best speedup)
    if (num_residuals >= params_.min_residual_points &&
        explained_ratio >= params_.min_explained_ratio)
    {
        return Result::Strategy::USE_RESIDUALS;
    }

    // Strategy 2: Hybrid (moderate speedup)
    if (explained_ratio >= 0.5f)
    {
        return Result::Strategy::USE_HYBRID;
    }

    // Strategy 3: Use all (poor fit)
    return Result::Strategy::USE_ALL;
}

RoboCompLidar3D::TPoints ModelBasedFilter::build_filtered_set(
    const RoboCompLidar3D::TPoints& explained,
    const RoboCompLidar3D::TPoints& residuals,
    Result::Strategy strategy) const
{
    RoboCompLidar3D::TPoints filtered;

    switch (strategy)
    {
        case Result::Strategy::USE_RESIDUALS:
            // Only residuals (maximum speedup)
            return residuals;

        case Result::Strategy::USE_HYBRID:
            // All residuals + downsampled inliers
            filtered = residuals;
            for (size_t i = 0; i < explained.size(); i += params_.downsample_factor) {
                filtered.push_back(explained[i]);
            }
            return filtered;

        case Result::Strategy::USE_ALL:
        default:
            // All points (no filtering)
            filtered = explained;
            filtered.insert(filtered.end(), residuals.begin(), residuals.end());
            return filtered;
    }
}

void ModelBasedFilter::print_result(int num_points, const Result& result) const
{
    qInfo() << "======= Model-Based Filter Result =======";
    qInfo()  << "   Input points =" << num_points << "\n"
             << "   Explained points =" << result.explained_points.size() << "\n"
             << "   Removed points =" << result.residual_points.size() << "\n"
             << "   Filtered set =" << result.filtered_set.size() << "\n"
             << "   Explained ratio =" << result.explained_ratio * 100.0f << "%" << "\n"
             << "   Mean residual SDF =" << result.mean_residual * 100.0f << "m" << "\n"
             << "   Points saved =" << result.points_saved;
}