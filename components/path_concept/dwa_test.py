import numpy as np

# create all samples for a dwa evaluation


curvature = np.arange(-22, 22, 1)
arc = np.arange(1, 400, 5)
projection = np.arange(0, 5, 1)
pts = np.stack(np.meshgrid(curvature, arc, projection), -1).reshape(-1, 3)
print(pts)


adv_max_accel = 300
rot_max_accel = 1
time_ahead = 1.5
step_along_arc = 100
advance_step = 100
rotation_step = 0.3
current_adv_speed = 0
current_rot_speed = 0

max_reachable_adv_speed = adv_max_accel * time_ahead
max_reachable_rot_speed = rot_max_accel * time_ahead

points = []

num_advance_points = max_reachable_adv_speed * 2 // advance_step
num_rotation_points = max_reachable_rot_speed * 2 // rotation_step
for v in np.linspace(-max_reachable_adv_speed, max_reachable_adv_speed, int(num_advance_points)):
	for w in np.linspace(-max_reachable_rot_speed, max_reachable_rot_speed, int(num_rotation_points)):
		new_advance = current_adv_speed + v
		new_rotation = -current_rot_speed + w
		if abs(w) > 0.001:
			r = new_advance / new_rotation
			arc_length = abs(new_rotation * time_ahead * r)
			for t in np.linspace(step_along_arc, arc_length, int(arc_length//step_along_arc)):
				x = r - r*np.cos(t/r)
				y = r * np.sin(t/r)
				points.append([x, y, t, t/r])

# for x, y, z, u in points:
# 	print(f'{x:.2f}, {y:.2f}, {z:.2f}, {u:.2f}')
# print(len(points), "points")

