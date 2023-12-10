import pybullet as pb
import pybullet_data
import numpy as np

class Model_Manager:
    def __init__(self):
        self.physics_client_id = pb.connect(pb.GUI)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        pb.setGravity(0, 0, -9.81)

        # Camera settings for a zenithal view
        cameraDistance = 4  # Adjust as needed
        cameraTargetPosition = [0, 0, 0]  # Adjust based on your scene
        cameraYaw = 180  # Adjust as needed
        cameraPitch = -95  # -90 degrees for straight down

        # Position the camera
        pb.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

        # Load the plane
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.loadURDF("plane.urdf")

        self.pb = pb
        self.current_room = None
        self.robot_id = None
        self.objects_in_the_room = []

    def step_simulation(self):
        pb.stepSimulation(physicsClientId=self.physics_client_id)

    def create_wall_with_door(self, wall_length, wall_height, wall_center, wall_orientation, door_width=None,
                              door_height=None, door_position=None, rotated_wall=False):
        """
        Creates a single wall with at most one door.

        Parameters:
        wall_length (float): The length of the wall.
        wall_height (float): The height of the wall.
        wall_center (tuple): The (x, y, z) coordinates for the center of the wall.
        wall_orientation (tuple): The (roll, pitch, yaw) orientation of the wall in radians.
        door_width (float, optional): The width of the door.
        door_height (float, optional): The height of the door.
        door_position (float, optional): The position of the door along the wall's length, from the left side.

        Returns:
        list: IDs of the created wall body parts.
        """
        wall_thickness = 0.1  # Thickness of the wall
        wall_parts_ids = []

        # Function to create a wall segment
        def add_wall_segment(center, half_extents):
            segment_shape_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half_extents)
            segment_body_id = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=segment_shape_id,
                                                basePosition=center,
                                                baseOrientation=pb.getQuaternionFromEuler(wall_orientation))
            wall_parts_ids.append(segment_body_id)

        # Create segments of the wall if there is a door
        if door_width and door_height and door_position is not None:
            # Calculate the start and end positions of the door
            door_start = door_position - door_width / 2
            door_end = door_position + door_width / 2

            # Left wall segment (if any)
            if door_start > 0:
                if not rotated_wall:
                    left_wall_center = [wall_center[0] - wall_length / 2 + door_start / 2, wall_center[1], wall_center[2]]
                else:
                    left_wall_center = [wall_center[0], wall_center[1] - wall_length / 2 + door_start / 2, wall_center[2]]
                add_wall_segment(left_wall_center, [door_start / 2, wall_thickness / 2, wall_height / 2])

            # Right wall segment (if any)
            if door_end < wall_length:
                if not rotated_wall:
                    right_wall_center = [wall_center[0] - wall_length / 2 + (door_end + (wall_length - door_end) / 2),
                                         wall_center[1], wall_center[2]]
                else:
                    right_wall_center = [wall_center[0], wall_center[1] - wall_length / 2 + (door_end + (wall_length - door_end) / 2),
                                     wall_center[2]]
                add_wall_segment(right_wall_center, [(wall_length - door_end) / 2, wall_thickness / 2, wall_height / 2])

            # Upper wall segment (above the door)
            if not rotated_wall:
                upper_wall_center = [wall_center[0] - wall_length / 2 + door_position, wall_center[1],
                                     wall_height + door_height/2]
            else:
                upper_wall_center = [wall_center[0], wall_center[1] - wall_length / 2 + door_position,
                                 wall_height + door_height / 2]
            add_wall_segment(upper_wall_center, [door_width / 2, wall_thickness / 2, (wall_height - door_height) / 2])

        # If there is no door, create one full wall segment
        else:
            add_wall_segment(wall_center, [wall_length / 2, wall_thickness / 2, wall_height / 2])

        return wall_parts_ids
    def create_room_with_doors(self, room_width, room_depth, room_height, room_center, room_rotation, doors_specs):
        """
        Creates a room with four walls, each with at most one door.

        Parameters:
        room_width (float): The width of the room.
        room_depth (float): The depth of the room.
        room_height (float): The height of the walls.
        room_center (tuple): The (x, y, z) coordinates for the center of the room.
        doors_specs (dict): Specifications for the doors in each wall {wall_index: (door_width, door_height, door_position)}

        Returns:
        list: IDs of all the created wall parts.
        """
        all_wall_parts_ids = []

        # Define the center positions for each wall
        wall_centers = [
            (room_center[0], room_center[1] - room_depth / 2, room_center[2]),  # Front wall
            (room_center[0], room_center[1] + room_depth / 2, room_center[2]),  # Back wall
            (room_center[0] - room_width / 2, room_center[1], room_center[2]),  # Left wall
            (room_center[0] + room_width / 2, room_center[1], room_center[2]),  # Right wall
        ]

        # Define the orientation for each wall
        wall_orientations = [
            (0, 0, 0),  # Front wall
            (0, 0, 0),  # Back wall
            (0, 0, 1.5708),  # Left wall, rotated 90 degrees
            (0, 0, 1.5708),  # Right wall, rotated 90 degrees
        ]

        # Define the lengths for the walls parallel to the x and y axes
        lengths = [room_width, room_width, room_depth, room_depth]

        # Create each wall with the specifications provided
        for i in range(4):
            door_spec = doors_specs.get(i)
            if door_spec:  # If there are door specs for the wall
                door_width, door_height, door_position = door_spec
            else:  # If there is no door in the wall
                door_width, door_height, door_position = None, None, None

            # Create the wall with or without a door
            wall_parts_ids = self.create_wall_with_door(
                wall_length=lengths[i],
                wall_height=room_height,
                wall_center=wall_centers[i],
                wall_orientation=wall_orientations[i],
                door_width=door_width,
                door_height=door_height,
                door_position=door_position,
                rotated_wall=(i==2 or i==3)
            )
            all_wall_parts_ids.extend(wall_parts_ids)

        self.current_room = {"width": room_width, "depth": room_depth, "height": room_height,
                             "center_x": room_center[0], "center_y": room_center[1], "rotation": room_rotation}
        self.objects_in_the_room = all_wall_parts_ids
        return all_wall_parts_ids
    def loadURDF(self, urdf_path):
        m = self.pb.loadURDF(fileName=urdf_path)
        self.pb.resetBasePositionAndOrientation(m, [0, 0, 1], [0, 0, 0, 1])
        return m
    def add_robot(self):
        size = [0.5/2, 0.5/2, 1.5/2]
        robot_id = self.pb.createCollisionShape(self.pb.GEOM_BOX, halfExtents=size,
                                                      physicsClientId=self.physics_client_id)
        robot_visual_shape_id = self.pb.createVisualShape(self.pb.GEOM_BOX, halfExtents=size,
                                                          physicsClientId=self.physics_client_id)
        robot_body_id = self.pb.createMultiBody(baseCollisionShapeIndex=robot_id,
                                                baseVisualShapeIndex=robot_visual_shape_id,
                                                basePosition=[0, 0, 0],
                                                physicsClientId=self.physics_client_id)
        pb.changeVisualShape(robot_body_id, -1, rgbaColor=[0, 1, 1, 1])
        # add nose
        nose_size = [0.1 / 2, 0.1 / 2, 1.8 / 2]
        nose_visual_shape_id = self.pb.createVisualShape(self.pb.GEOM_BOX, halfExtents=nose_size,
                                                               physicsClientId=self.physics_client_id)
        nose_id = self.pb.createMultiBody(baseVisualShapeIndex=nose_visual_shape_id,
                                          basePosition=[0, 0.5/2, 0.1],
                                          baseOrientation=pb.getQuaternionFromEuler([0, 0, np.pi/2]),
                                          physicsClientId=self.physics_client_id)
        pb.changeVisualShape(nose_id, -1, rgbaColor=[1, 0, 0, 1])

        self.robot_id = robot_id
        return robot_id
    def set_robot_velocity(self, side, adv, rot):
        for obj in self.objects_in_the_room:
            pb.resetBaseVelocity(obj, linearVelocity=[side*12, -adv*12, 0])

    def get_room(self):
        return self.current_room

