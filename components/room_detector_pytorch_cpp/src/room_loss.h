//
// Created by pbustos on 16/11/25.
//

#ifndef ROOM_LOSS_H
#define ROOM_LOSS_H

#include <torch/torch.h>
#include "room_model.h"

/**
 * @brief Loss function for room fitting
 *
 * Measures how well the LiDAR points fit the room model.
 * Points should be close to the box surface (SDF ≈ 0)
 */
class RoomLoss
{
    public:
        /**
         * @brief Compute loss between LiDAR points and room model
         *
         * @param points Tensor of shape [N, 2] with (x, y) LiDAR points in robot frame
         * @param room Room model to compare against
         * @param wall_thickness Expected thickness of walls for robust fitting
         * @return Scalar tensor with the loss value
         */
        static torch::Tensor compute_loss(const torch::Tensor& points,
                                      std::shared_ptr<RoomModel>& room,
                                      float wall_thickness = 0.1f);
};

#endif //ROOM_LOSS_H
