/*
 * Standalone test for RoomModel and SDF optimization
 * Demonstrates simultaneous room mapping and robot localization
 */

#include "room_model.h"
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    std::cout << "========================================\n";
    std::cout << "Room Model SDF + Robot Localization Test\n";
    std::cout << "========================================\n\n";

    // Ground truth room (in global frame)
    float true_room_cx = 0.0f, true_room_cy = 0.0f;
    float true_w = 8.0f, true_h = 6.0f;

    // Ground truth robot pose (in global frame)
    float true_robot_x = -1.5f, true_robot_y = 1.0f;
    float true_robot_theta = 0.3f;  // ~17 degrees

    std::cout << "Ground truth:\n";
    std::cout << "  Room center: (" << true_room_cx << ", " << true_room_cy << ")\n";
    std::cout << "  Room size: " << true_w << " x " << true_h << " m\n";
    std::cout << "  Robot pose: (" << true_robot_x << ", " << true_robot_y
              << ", " << true_robot_theta << " rad)\n\n";

    // Generate synthetic LiDAR data FROM THE ROOM WALLS
    // but transform to ROBOT FRAME
    int num_points = 360;
    std::vector<float> points_robot_frame;
    points_robot_frame.reserve(num_points * 2);

    // Generate points along the four walls in GLOBAL frame
    for (int i = 0; i < num_points; i++) {
        float x_global, y_global;

        // Distribute points among four walls
        int wall = i % 4;
        float wall_t = (i / 4) / static_cast<float>(num_points / 4);

        switch(wall) {
            case 0: // Bottom wall
                x_global = true_room_cx - true_w/2 + true_w * wall_t;
                y_global = true_room_cy - true_h/2;
                break;
            case 1: // Right wall
                x_global = true_room_cx + true_w/2;
                y_global = true_room_cy - true_h/2 + true_h * wall_t;
                break;
            case 2: // Top wall
                x_global = true_room_cx - true_w/2 + true_w * wall_t;
                y_global = true_room_cy + true_h/2;
                break;
            case 3: // Left wall
                x_global = true_room_cx - true_w/2;
                y_global = true_room_cy - true_h/2 + true_h * wall_t;
                break;
        }

        // Add noise
        float noise = 0.05f;
        x_global += noise * (static_cast<float>(rand()) / RAND_MAX - 0.5);
        y_global += noise * (static_cast<float>(rand()) / RAND_MAX - 0.5);

        // Transform from GLOBAL to ROBOT frame
        // p_robot = R(-theta) * (p_global - robot_pos)
        float dx = x_global - true_robot_x;
        float dy = y_global - true_robot_y;
        float cos_theta = std::cos(-true_robot_theta);
        float sin_theta = std::sin(-true_robot_theta);

        float x_robot = cos_theta * dx - sin_theta * dy;
        float y_robot = sin_theta * dx + cos_theta * dy;

        points_robot_frame.push_back(x_robot);
        points_robot_frame.push_back(y_robot);
    }

    torch::Tensor points = torch::from_blob(
        points_robot_frame.data(),
        {num_points, 2},
        torch::kFloat32
    ).clone();

    std::cout << "Generated " << num_points << " LiDAR points (in robot frame)\n\n";

    // Initial guess: assume robot at origin, room centered on point cloud
    auto x_coords = points.index({torch::indexing::Slice(), 0});
    auto y_coords = points.index({torch::indexing::Slice(), 1});

    float x_min = x_coords.min().item<float>();
    float x_max = x_coords.max().item<float>();
    float y_min = y_coords.min().item<float>();
    float y_max = y_coords.max().item<float>();

    float init_room_cx = (x_min + x_max) / 2.0f;
    float init_room_cy = (y_min + y_max) / 2.0f;
    float init_hw = (x_max - x_min) / 2.0f;
    float init_hh = (y_max - y_min) / 2.0f;

    float init_robot_x = 0.0f;
    float init_robot_y = 0.0f;
    float init_robot_theta = 0.0f;

    std::cout << "Initial guess:\n";
    std::cout << "  Room center: (" << init_room_cx << ", " << init_room_cy << ")\n";
    std::cout << "  Room size: " << 2*init_hw << " x " << 2*init_hh << " m\n";
    std::cout << "  Robot pose: (" << init_robot_x << ", " << init_robot_y
              << ", " << init_robot_theta << " rad)\n\n";

    // Create room model with robot pose
    RoomModel room(init_hw, init_hh,
                   init_robot_x, init_robot_y, init_robot_theta);

    // Setup optimizer
    auto params = room.parameters();
    torch::optim::Adam optimizer(params, torch::optim::AdamOptions(0.01));

    // Optimization loop
    std::cout << "Optimizing room and robot pose...\n";
    const int num_iterations = 350;

    for (int iter = 0; iter < num_iterations; ++iter) {
        optimizer.zero_grad();

        torch::Tensor loss = RoomLoss::compute(points, room, 0.1f);
        loss.backward();
        optimizer.step();

        if (iter % 30 == 0 || iter == num_iterations - 1) {
            auto pose = room.get_robot_pose();
            std::cout << "  Iter " << std::setw(3) << iter
                      << " | Loss: " << std::fixed << std::setprecision(6)
                      << loss.item<float>()
                      << " | Robot: (" << std::setprecision(2)
                      << pose[0] << ", " << pose[1] << ", " << pose[2] << ")\n";
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "Optimization Result\n";
    std::cout << "========================================\n";
    room.print_info();

    // Compute uncertainty
    std::cout << "\nComputing uncertainty (Laplace approximation)...\n";
    torch::Tensor covariance = UncertaintyEstimator::compute_covariance(points, room, 0.1f);
    UncertaintyEstimator::print_uncertainty(covariance, room);

    // Compute errors
    auto final_room = room.get_room_parameters();
    auto final_robot = room.get_robot_pose();
    auto std_devs = UncertaintyEstimator::get_std_devs(covariance);
    float width_std = 2 * std_devs[0];
    float height_std = 2 * std_devs[1];
    //7float room_cx_err = std::abs(final_room[0] - true_room_cx);
    //float room_cy_err = std::abs(final_room[1] - true_room_cy);
    float room_cx_err = 0.0f;
    float room_cy_err = 0.0f;
    float w_err = std::abs(2 * final_room[0] - true_w);
    float h_err = std::abs(2 * final_room[1] - true_h);

    float robot_x_err = std::abs(final_robot[0] - true_robot_x);
    float robot_y_err = std::abs(final_robot[1] - true_robot_y);
    float robot_theta_err = std::abs(final_robot[2] - true_robot_theta);

    std::cout << "\n=== ERROR vs GROUND TRUTH ===\n";
    std::cout << "Room:\n";
    std::cout << "  Half-width:   " << std::setprecision(4) << w_err / 2.0f << " m";
    std::cout << "  (" << (w_err / (2*width_std)) << " sigma)\n";
    std::cout << "  Half-height:  " << h_err / 2.0f << " m";
    std::cout << "  (" << (h_err / (2*height_std)) << " sigma)\n";
    std::cout << "  Full width:   " << w_err << " m";
    std::cout << "  (" << (w_err / width_std) << " sigma)\n";
    std::cout << "  Full height:  " << h_err << " m";
    std::cout << "  (" << (h_err / height_std) << " sigma)\n";

    std::cout << "\nRobot:\n";
    std::cout << "  Position X:   " << robot_x_err << " m";
    std::cout << "  (" << (robot_x_err / std_devs[2]) << " sigma)\n";
    std::cout << "  Position Y:   " << robot_y_err << " m";
    std::cout << "  (" << (robot_y_err / std_devs[3]) << " sigma)\n";
    std::cout << "  Orientation:  " << robot_theta_err << " rad";
    std::cout << "  (" << (robot_theta_err / std_devs[4]) << " sigma)\n";

    std::cout << "\n=== CALIBRATION CHECK ===\n";
    std::cout << "Good calibration: errors should be < 2-3 sigma\n";
    bool well_calibrated = true;
    if (w_err / width_std > 3) { std::cout << "⚠️  Room width uncertainty too small!\n"; well_calibrated = false; }
    if (h_err / height_std > 3) { std::cout << "⚠️  Room height uncertainty too small!\n"; well_calibrated = false; }
    if (robot_x_err / std_devs[2] > 3) { std::cout << "⚠️  Robot X uncertainty too small!\n"; well_calibrated = false; }
    if (robot_y_err / std_devs[3] > 3) { std::cout << "⚠️  Robot Y uncertainty too small!\n"; well_calibrated = false; }
    if (robot_theta_err / std_devs[4] > 3) { std::cout << "⚠️  Robot theta uncertainty too small!\n"; well_calibrated = false; }
    if (well_calibrated) {
        std::cout << "✅ Uncertainty estimates are well calibrated!\n";
    }
    std::cout << "========================================\n";

    return 0;
}