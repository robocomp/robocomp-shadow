import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class DWA_Optimizer():
    def __init__(self, camera_matrix, focalx, focaly, frame_shape):
        self.candidates = self.sample_points()
        #self.params = np.array(candidates["params"])
        self.camera_matrix = camera_matrix
        self.focalx = focalx
        self.focaly = focaly
        self.frame_shape = frame_shape
        # project polygons on image
        self.candidates = self.project_polygons(self.candidates)
        self.candidates = self.create_masks_3d(frame_shape, self.candidates)

    def discard(self, mask_img):
        # discard all paths with less than 95% occupancy
        survivors = []
        for c in self.candidates:
            mask = c["mask"]
            result = cv2.bitwise_and(mask_img, mask)
            lane_size = np.count_nonzero(mask)
            inliers = np.count_nonzero(result)
            segmented_size = np.count_nonzero(mask_img)
            if lane_size == 0:
                continue
            #print(inliers * 100 / lane_size)
            if inliers * 100 / lane_size > 90:
                loss = abs(segmented_size - lane_size) + 5 * abs(lane_size - inliers)
                survivor = {"mask": mask, "polygon": c["polygon"],
                            "params": c["params"], "loss": loss,
                            "projected_polygon": c["projected_polygon"],
                            "trajectory": c["trajectory"]}
                survivors.append(survivor)
        return survivors

    def optimize(self, mask_img, target_object, x0=None):
        curvature_threshold = 0.025  # range: -1, 1
        losses = []
        for i, mask in enumerate(self.masks):
            distance_to_target_object = 0
            if target_object is not None:
                target_center = np.array([target_object.right - target_object.left, target_object.bot - target_object.top])
                tip_of_path = self.get_tip_of_path(self.polygons[i])
                distance_to_target_object = np.linalg.norm(target_center - tip_of_path)
            loss = self.loss_function(mask_img, mask, distance_to_target_object)
            losses.append(loss)
        losses = np.array(losses)
        #sorted_loss_index = np.argsort(losses)
        params = self.params
        params = np.c_[params, losses]

        # initialize path_set, curvatures and selected_trajectories
        # with the first mask in the sorted list with curvature = 0 aka the first straight line
        # index = int(np.argwhere(self.params[sorted_loss_index][:, 2] == 0)[0][0])     # take curvature
        # path_set = [self.masks[sorted_loss_index[index]]]
        # curvatures = np.array([self.params[sorted_loss_index[index]][2]])
        # selected_target_trajs = [self.targets[sorted_loss_index[index]]]
        # controls = self.params[sorted_loss_index[index]][0:2].reshape(1, 2)

        # clustering
        # path_set = []
        # curvatures = []
        # selected_target_trajs = []
        # controls = []
        # kmeans = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42)
        # kmeans.fit(self.params[:, 2].reshape(-1, 1))
        # # sort each set wrt loss
        # for i in range(len(kmeans.cluster_centers_)):
        #     print(np.array(losses)[np.argwhere(kmeans.labels_ == i)])
        #     index = 0
        #     #index = int(np.argsort(np.array(losses)[kmeans.labels_ == i].nonzero()])[0])
        #     curvatures.append(self.params[index, 2])
        #     print(i, index)
        #     path_set.append(self.masks[index])
        #     selected_target_trajs.append(self.targets[index])
        #     advance = self.params[index][0]
        #     rotation = self.params[index][0]
        #     controls.append([advance, rotation])

        # assign selected trajs to cluster centers
        # for c in len(kmeans.cluster_centers_):
        #     index = np.argmin(curvatures - c)
        #     path_set.append(self.masks[index])
        #     selected_target_trajs.append(self.targets[index])
        #     advance, rotation, curvature = self.params[index]
        #     controls = np.append(controls, np.array([advance, rotation]).reshape(1, 2), axis=0)

        # for i in sorted_loss_index:
        #     advance, rotation, curvature = self.params[i]
        #     if np.all(np.abs(curvatures-curvature) > curvature_threshold):   # next one is separated by thresh.
        #         path_set.append(self.masks[i])
        #         curvatures = np.append(curvatures, curvature)
        #         selected_target_trajs.append(self.targets[i])
        #         controls = np.append(controls, np.array([advance, rotation]).reshape(1, 2), axis=0)

        #return path_set, curvatures, selected_target_trajs, controls
        return params, self.targets

    def loss_function(self, mask_img, mask_path, distance_to_target_object=0):
        result = cv2.bitwise_and(mask_img, mask_path)
        lane_size = np.count_nonzero(mask_path)
        segmented_size = np.count_nonzero(mask_img)
        inliers = np.count_nonzero(result)
        loss = abs(segmented_size - lane_size) + 5*abs(lane_size - inliers)  + distance_to_target_object
        return float(loss)

    def sample_points(self):
        adv_max_accel = 600
        rot_max_accel = 0.4
        time_ahead = 2.5
        step_along_ang = 0.05
        advance_step = 100
        rotation_step = 0.1
        current_adv_speed = 0
        current_rot_speed = 0
        max_reachable_adv_speed = adv_max_accel * time_ahead
        max_reachable_rot_speed = rot_max_accel * time_ahead
        robot_semi_width = 200

        params = []
        targets = []
        trajectories = []
        for v in np.arange(400, max_reachable_adv_speed, advance_step):
            for w in np.arange(0, max_reachable_rot_speed, rotation_step):
                new_advance = current_adv_speed + v
                new_rotation = current_rot_speed + w
                arc_length = new_advance * time_ahead
                if w > 0:
                    r = new_advance / new_rotation
                    ang_length = arc_length / r

                    # compute LEFT arcs corresponding to r - 100 and r + 100
                    points = []
                    central = []
                    for t in np.arange(0, ang_length, step_along_ang):
                        xl = (r-robot_semi_width)*np.cos(t)-r
                        yl = (r-robot_semi_width)*np.sin(t)
                        points.append([xl, yl])
                    ipoints = []
                    for t in np.arange(0, ang_length, step_along_ang):     # inverse order to build a proper polygon
                        xh = (r+robot_semi_width)*np.cos(t)-r
                        yh = (r+robot_semi_width)*np.sin(t)
                        ipoints.append([xh, yh])
                    ipoints.reverse()
                    # compute central line
                    for t in np.arange(0, ang_length, step_along_ang):
                        xl = r*np.cos(t)-r
                        yl = r*np.sin(t)
                        central.append([xl, yl])
                    # add to trajectories
                    if len(points) > 2 and len(ipoints) > 2:
                        points.extend(ipoints)
                        trajectories.append(points)
                        params.append([new_advance, -new_rotation, -r])
                        targets.append(central)

                    # now compute RIGHT arcs corresponding to r - 100 and r + 100
                    points = []
                    central = []
                    for t in np.arange(0, ang_length, step_along_ang):
                        xl = r - (r-robot_semi_width) * np.cos(t)
                        yl = (r-robot_semi_width) * np.sin(t)
                        points.append([xl, yl])
                    ipoints = []
                    for t in np.arange(0, ang_length, step_along_ang):  # inverse order to build a proper polygon
                        xh = r - (r+robot_semi_width) * np.cos(t)
                        yh = (r+robot_semi_width) * np.sin(t)
                        ipoints.append([xh, yh])
                    ipoints.reverse()
                    # compute central line
                    for t in np.arange(0, ang_length, step_along_ang):
                        xl = r - r*np.cos(t)
                        yl = r*np.sin(t)
                        central.append([xl, yl])
                    # add to trajectories
                    if len(points) > 2 and len(ipoints) > 2:
                        points.extend(ipoints)
                        trajectories.append(points)
                        params.append([new_advance, new_rotation, r])
                        targets.append(central)

            else:       # avoid division by zero
                    points = []
                    central = []
                    points.append([-robot_semi_width, 0])
                    points.append([-robot_semi_width, arc_length])
                    points.append([robot_semi_width, arc_length])
                    points.append([robot_semi_width, 0])
                    trajectories.append(points)
                    params.append([new_advance, 0.0, np.inf])
                    central.append([0, arc_length/5])
                    central.append([0, 2*arc_length/5])
                    central.append([0, 3*arc_length/5])
                    central.append([0, 4*arc_length/5])
                    central.append([0, arc_length])
                    targets.append(central)

        params = np.array(params)
        params[:, 2] = 100*np.reciprocal(params[:, 2])     # compute curvature from radius
        candidates = []
        for pa, tg, tr in zip(params, targets, trajectories):
            candidate = {}
            candidate["polygon"] = tr
            candidate["params"] = pa
            candidate["trajectory"] = tg
            candidates.append(candidate)
        return candidates

    def create_masks_3d(self, shape, candidates):
        #masks = []
        height, width, _ = shape
        for c in candidates:  # [xl,yl]
            t = c["projected_polygon"]
            nt = np.array(t).copy()
            mask_poly = np.zeros((shape[0], shape[1]), np.uint8)
            cv2.fillPoly(mask_poly, pts=[nt.astype(int)], color=255)
            #masks.append(mask_poly)
            c["mask"] = mask_poly
            #cv2.polylines(mask_poly, pts=[nt.astype(int)], isClosed=True, color=255)
            #cv2.imshow("mask", mask_poly)
            #cv2.waitKey(200)
        return candidates

    def project_polygons(self, candidates):

        # get 3D points in robot CS, transform to camera CS and project with projection equations
        # transform to camera CS
        frame_width, frame_height, _ = self.frame_shape
        for c in candidates:
            po = c["polygon"]
            extra_cols = np.array([np.zeros(len(po)), np.ones(len(po))]).transpose()
            npo = np.array(po)
            npo = np.append(npo, extra_cols, axis=1).transpose()
            cam_poly = self.camera_matrix.dot(npo).transpose()    # n x 4
            # project on camera
            xs = np.array([((cam_poly[:, 0] * self.focaly) / cam_poly[:, 1]) + (frame_width/2),
                           ((cam_poly[:, 2] * -self.focalx) / cam_poly[:, 1]) + (frame_height/2)]).transpose()
            c["projected_polygon"] = xs.astype(int)  # now in camera CS

        return candidates

 # def sample_points_2(self):
 #        curvature = np.arange(-38, 38, 3)
 #        arc = np.arange(100, 400, 10)
 #        projection = np.arange(1, 2, 1)
 #        return np.stack(np.meshgrid(curvature, arc, projection), -1).reshape(-1, 3)

# def create_masks(self, shape, samples):
#     masks = []
#     height, width = shape[0:2]
#     max_curvature = 150
#     halfcurv = max_curvature // 2
#     height = height - 10  # margin from bottom margin
#     hwidth = width // 2
#     lane_width = 150
#     number_of_points = 5
#     for curvature, arc, projection in samples:
#         left_points = []
#         right_points = []
#         points = []
#         # print(arc, curvature, projection)
#         if curvature == 0:
#             left_points.append((hwidth - lane_width // 2, height))
#             left_points.append((hwidth - lane_width // 2, height - arc))
#             right_points.append((hwidth + lane_width // 2, height))
#             right_points.append((hwidth + lane_width // 2, height - arc))
#         else:
#             k = np.log(10000) / halfcurv  # mapping halfcurv -> 0 to 10.000 -> 0
#             if curvature > 0:
#                 left_radius = halfcurv - curvature  # goes from 250 to 0. High radius means low curvature. The exp maps from 10000 to 0
#                 left_radius = np.exp(left_radius * k)
#                 right_radius = halfcurv - curvature - lane_width / 20  # goes from 250 to 0. High radius means low curvature. The exp maps from 10000 to 0
#                 right_radius = np.exp(right_radius * k)
#             else:
#                 left_radius = -curvature - halfcurv
#                 left_radius = -np.exp(-left_radius * k)
#                 right_radius = -curvature - halfcurv - lane_width / 20
#                 right_radius = -np.exp(-right_radius * k)
#
#             # left
#             max_angle_left = arc / left_radius
#             for t in np.linspace(np.pi, np.pi + max_angle_left, number_of_points):
#                 p = (int(left_radius * np.cos(t) + left_radius + hwidth - lane_width // 2),
#                      int(left_radius * np.sin(t)) + height)
#                 left_points.append(p)
#
#             # right
#             max_angle_right = arc / right_radius
#             for t in np.linspace(np.pi, np.pi + max_angle_right, number_of_points):
#                 p = (int(right_radius * np.cos(t) + right_radius + hwidth + lane_width // 2),
#                      int(right_radius * np.sin(t)) + height)
#                 right_points.append(p)
#
#         # fake projection on polygon points into image
#         # we want to reduce the current separation between lines at the middle length
#         # down to its half, and recompute the points according to a linear law
#         # starting from the base (which is not changed) and up to the end.
#         max_projection = 10
#         # projection = 4
#         proj = ((1 - 0.4) / max_projection) * projection + 0.4
#         left_proj_points = []
#         right_proj_points = []
#         proj_points = []
#         for i in range(len(left_points)):
#             length_reduc = -((1 - proj) / len(
#                 left_points)) * i + 1  # this is the reduction in length for each pair of opposite points
#             lp = np.array(left_points[i])
#             rp = np.array(right_points[i])
#             left_proj_points.append((lp + ((rp - lp) * length_reduc)))
#             right_proj_points.append((rp + ((lp - rp) * length_reduc)))
#
#         left_proj_points = [(int(x), int(y)) for x, y in left_proj_points]
#         right_proj_points = [(int(x), int(y)) for x, y in right_proj_points]
#         target = (np.array(left_proj_points)[-1] + np.array(right_proj_points)[-1]) / 2
#         points.extend(left_points)
#         points.extend(right_points[::-1])
#         proj_points.extend(left_proj_points)
#         proj_points.extend(right_proj_points[::-1])
#
#         mask_poly = np.zeros(shape, np.uint8)
#         if proj_points:
#             cv2.fillPoly(mask_poly, pts=np.array([proj_points]), color=255)
#
#         masks.append(mask_poly)
#
#         # cv2.imshow("mask", mask_poly)
#         # cv2.waitKey(2)
#
#     return masks

 # initilize path_set with the first mask in the sorted list with curvature = 0 aka the first straight line
        #path_set = [self.masks[np.where(np.array(self.samples)[sorted_loss_index][:, 1] == 0)[0][0]]]

        # radius = self.params[:, 1]
        # radius_histogram, bins = np.histogram(radius, 200, density=True)
        # cdf = radius_histogram.cumsum()  # cumulative distribution function
        # cdf = (200-1)*cdf/cdf[-1]  # normalize
        # use linear interpolation of cdf to find new pixel values
        # self.params[:, 1] = np.interp(radius, bins[:-1], cdf)
        # print(self.params[:, 1])
        # cluster by curvatures
        winner_arc, winner_curvature = self.params[sorted_loss_index[0]]
        # print(self.params[sorted_loss_index])