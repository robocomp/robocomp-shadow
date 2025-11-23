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

