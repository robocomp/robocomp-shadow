//
// Created by pbustos on 16/11/25.
//

#include "room_loss.h"


torch::Tensor RoomLoss::compute_loss(const torch::Tensor& points,
                                 RoomModel& room,
                                 float wall_thickness)
{
    const torch::Tensor sdf_values = room.sdf(points);
    torch::Tensor loss = torch::mean(torch::square(sdf_values));
    return loss;
}

// torch::Tensor RoomLoss::compute_loss(const torch::Tensor& points,
//                                      RoomModel& room,
//                                      float wall_thickness)
// {
//     const torch::Tensor sdf_values = room.sdf(points);
//     const torch::Tensor sdf_abs = torch::abs(room.sdf(points));
//     const torch::Tensor interior_penalty = torch::clamp(sdf_abs, 0.0f, wall_thickness);
//     const torch::Tensor exterior_penalty = torch::square(sdf_abs);
//     torch::Tensor loss = torch::where(sdf_values < 0, interior_penalty, exterior_penalty).mean();
//     return loss;
// }