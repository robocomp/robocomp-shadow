import sys

import numpy as np
import cv2
import time
class DWA_Optimizer():
    def __init__(self):
        #self.samples = self.sample_points_2()
        self.trajectories = self.sample_points()

        # add to create masks the lane skeleton
        #self.masks = self.create_masks((384, 384), self.samples)
        self.masks = self.create_masks_3d((384, 384), self.trajectories)

    def optimize(self, loss, mask_img, x0=None):
        losses = []
        alternative_masks = []
        curvatures = []
        curvature_threshold = 25
        for mask, sample in zip(self.masks, self.samples):
            losses.append(loss(mask_img, mask, sample))

        sorted_loss_index = np.argsort(losses)
        winner_curvature, winner_arc, _ = self.samples[sorted_loss_index[0]]
        curvatures.append(winner_curvature)
        for i in range(1, len(sorted_loss_index)):
            curvature, arc, _ = self.samples[sorted_loss_index[i]]
            if np.all(abs(curvatures-curvature) > curvature_threshold):
                    #and arc > winner_arc * 0.5:
                alternative_masks.append(self.masks[sorted_loss_index[i]])
                curvatures.append(curvature)

        return losses[sorted_loss_index[0]], self.masks[sorted_loss_index[0]], alternative_masks, curvatures

    def sample_points_2(self):
        curvature = np.arange(-38, 38, 3)
        arc = np.arange(100, 400, 10)
        projection = np.arange(1, 2, 1)
        return np.stack(np.meshgrid(curvature, arc, projection), -1).reshape(-1, 3)

    def sample_points(self):
        adv_max_accel = 100
        rot_max_accel = 1
        time_ahead = 1.5
        step_along_arc = 10
        advance_step = 20
        rotation_step = 0.2
        current_adv_speed = 0
        current_rot_speed = 0
        max_reachable_adv_speed = adv_max_accel * time_ahead
        max_reachable_rot_speed = rot_max_accel * time_ahead

        trajectories = []
        num_advance_points = max_reachable_adv_speed * 2 // advance_step
        num_rotation_points = max_reachable_rot_speed * 2 // rotation_step
        for v in np.linspace(0, max_reachable_adv_speed, int(num_advance_points)):
            for w in np.linspace(-max_reachable_rot_speed, max_reachable_rot_speed, int(num_rotation_points)):
                points = []
                new_advance = current_adv_speed + v
                new_rotation = -current_rot_speed + w
                if abs(w) > 0.001:
                    r = new_advance / new_rotation
                    arc_length = abs(new_rotation * time_ahead * r)
                    for t in np.linspace(step_along_arc, arc_length, int(arc_length // step_along_arc)):
                        x = r - r * np.cos(t / r)
                        y = r * np.sin(t / r)
                        points.append([x, y, v, w])
                else:       # para evitar la división por cero
                    for t in np.linspace(step_along_arc, new_advance*time_ahead, int(new_advance*time_ahead/step_along_arc)):
                        points.append([0, t, t, 0])
                if len(points) > 2:
                    trajectories.append(points)

        #for x, y, z, u in points:
        #    print(f'{x:.2f}, {y:.2f}, {z:.2f}, {u:.2f}')
        print(len(trajectories), "trajectories")
        return trajectories

    def create_masks(self, shape, samples):
        masks = []
        height, width = shape[0:2]
        max_curvature = 150
        halfcurv = max_curvature // 2
        height = height - 10  # margin from bottom margin
        hwidth = width // 2
        lane_width = 150
        number_of_points = 5
        for curvature, arc, projection in samples:
            left_points = []
            right_points = []
            points = []
            #print(arc, curvature, projection)
            if curvature == 0:
                left_points.append((hwidth - lane_width // 2, height))
                left_points.append((hwidth - lane_width // 2, height - arc))
                right_points.append((hwidth + lane_width // 2, height))
                right_points.append((hwidth + lane_width // 2, height - arc))
            else:
                k = np.log(10000) / halfcurv  # mapping halfcurv -> 0 to 10.000 -> 0
                if curvature > 0:
                    left_radius = halfcurv - curvature  # goes from 250 to 0. High radius means low curvature. The exp maps from 10000 to 0
                    left_radius = np.exp(left_radius * k)
                    right_radius = halfcurv - curvature - lane_width / 20  # goes from 250 to 0. High radius means low curvature. The exp maps from 10000 to 0
                    right_radius = np.exp(right_radius * k)
                else:
                    left_radius = -curvature - halfcurv
                    left_radius = -np.exp(-left_radius * k)
                    right_radius = -curvature - halfcurv - lane_width / 20
                    right_radius = -np.exp(-right_radius * k)

                # left
                max_angle_left = arc / left_radius
                for t in np.linspace(np.pi, np.pi + max_angle_left, number_of_points):
                    p = (int(left_radius * np.cos(t) + left_radius + hwidth - lane_width // 2),
                         int(left_radius * np.sin(t)) + height)
                    left_points.append(p)

                # right
                max_angle_right = arc / right_radius
                for t in np.linspace(np.pi, np.pi + max_angle_right, number_of_points):
                    p = (int(right_radius * np.cos(t) + right_radius + hwidth + lane_width // 2),
                         int(right_radius * np.sin(t)) + height)
                    right_points.append(p)

            # fake projection on polygon points into image
            # we want to reduce the current separation between lines at the middle length
            # down to its half, and recompute the points according to a linear law
            # starting from the base (which is not changed) and up to the end.
            max_projection = 10
            #projection = 4
            proj = ((1 - 0.4) / max_projection) * projection + 0.4
            left_proj_points = []
            right_proj_points = []
            proj_points = []
            for i in range(len(left_points)):
                length_reduc = -((1 - proj) / len(
                    left_points)) * i + 1  # this is the reduction in length for each pair of opposite points
                lp = np.array(left_points[i])
                rp = np.array(right_points[i])
                left_proj_points.append((lp + ((rp - lp) * length_reduc)))
                right_proj_points.append((rp + ((lp - rp) * length_reduc)))

            left_proj_points = [(int(x), int(y)) for x, y in left_proj_points]
            right_proj_points = [(int(x), int(y)) for x, y in right_proj_points]
            target = (np.array(left_proj_points)[-1] + np.array(right_proj_points)[-1]) / 2
            points.extend(left_points)
            points.extend(right_points[::-1])
            proj_points.extend(left_proj_points)
            proj_points.extend(right_proj_points[::-1])

            mask_poly = np.zeros(shape, np.uint8)
            if proj_points:
                cv2.fillPoly(mask_poly, pts=np.array([proj_points]), color=255)

            masks.append(mask_poly)

            # cv2.imshow("mask", mask_poly)
            # cv2.waitKey(2)

        return masks

    def create_masks_3d(self, shape, trajectories):
        # samples are tuples [x, y]
        # we need to duplicate and translate the tuple

        masks = []
        height, width = shape[0:2]
        max_curvature = 150
        halfcurv = max_curvature // 2
        height = height - 10  # margin from bottom margin
        hwidth = width // 2
        lane_width = 150
        number_of_points = 5

        for t in trajectories:
            points = []
            nt = np.array(t).copy()
            print(nt.size)
            nt[:, 0] += hwidth
            nt[:, 1] = height - nt[:, 1]
            ls = nt.copy()
            rs = nt.copy()
            ls[:, 0] -= 100
            #ls[ls[:, 0] < 0] = 0
            rs[:, 0] += 100
            #rs[rs[:, 0] > width] = width
            points.extend(ls.astype(int).tolist())
            points.extend(rs[::-1].astype(int).tolist())  #invert order
            mask_poly = np.zeros(shape, np.uint8)
            cv2.fillPoly(mask_poly, pts=[np.array(points)[:, [0, 1]]], color=255)
            masks.append(mask_poly)
            #cv2.polylines(mask_poly, pts=[nt[:, [0, 1]].astype(int)], isClosed=False, color=255)

            cv2.imshow("mask", mask_poly)
            cv2.waitKey(100)

        return masks