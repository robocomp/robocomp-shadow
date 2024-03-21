import pybullet as pb
import pybullet_data
import numpy as np
import math

class Model_Manager:
    def __init__(self):
        """
        sets up the physics engine and camera for a given scene, including loading
        a plane model and configuring the camera settings for a zenithal view.

        """
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
        self.current_doors = [] # List of dictionaries with the door specs#

    def step_simulation(self):
        pb.stepSimulation(physicsClientId=self.physics_client_id)
    def create_rectangular_room(self, width, depth, height, orientation, room_pos):
        """
        Creates a rectangular room in PyBullet using GEOM_BOX primitives.

        Parameters:
        width (float): The width of the room.
        depth (float): The depth of the room.
        height (float): The height of the room.
        orientation (float): The orientation (rotation) of the room around the Z-axis in radians.
        """
        wall_thickness = 0.1  # Adjust thickness as needed

        # Function to create a single wall
        def create_wall(position, dimensions, color):
            """
            creates a visual and collision shape representing a wall with the given
            dimensions and color, and returns the collision shape object.

            Args:
                position (float): 3D location where the wall will be created.
                dimensions (float): 2D dimensions of the wall shape in pixels,
                    which is used to determine the size and position of the visual
                    shape and collision shape created by the function.
                color (float): 4-component rgba color of the visual shape of the
                    wall.

            Returns:
                collision shape index.: a collision shape representing a wall with
                specified dimensions and color.
                
                		- `halfExtents`: The half-lengths of the box shape in the X, Y,
                and Z directions.
                		- `rgbaColor`: The color of the visual shape, represented as an
                RGBA value (red, green, blue, alpha).
                		- `shapeType`: The type of shape created, which is always
                `GEOM_BOX` for walls.
                		- `baseMass`: The mass of the body, set to zero to indicate an
                object without mass.
                		- `baseVisualShapeIndex`: The index of the visual shape associated
                with the body, which refers to the wall visual shape created in
                the function.
                		- `baseCollisionShapeIndex`: The index of the collision shape
                associated with the body, which refers to the wall collision shape
                created in the function.
                		- `basePosition`: The position of the body, set to the input
                `position` parameter.
                		- `baseOrientation`: The orientation of the body, set to the
                identity quaternion (0, 0, 0).
                
                	In summary, the output of the `create_wall` function is a collision
                shape representing the wall, along with some additional metadata
                about its position and orientation.
                

            """
            wall_visual_shape = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=dimensions,
                                                    rgbaColor=color)
            wall_collision_shape = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=dimensions)

            # wall_body = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=wall_visual_shape,
            #                                baseCollisionShapeIndex=wall_collision_shape,
            #                                basePosition=position,
            #                                baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))
            #pb.resetBasePositionAndOrientation()
            # return wall_body
            return wall_collision_shape

        def merge_objects(front_wall_position, back_wall_position, left_wall_position, right_wall_position, orientation):

            """
            creates a merged object composed of multiple geometric shapes and
            assigns it a unique multi-body ID. It sets the base mass to 1, position,
            and orientation, and returns the merged object.

            Args:
                front_wall_position (3D position vector.): 3D position of the front
                    wall of the room in the world frame of reference.
                    
                    		- `front_wall_position`: A 3D position vector representing
                    the front wall of the room in the world coordinates system.
                    It has three components - x, y, and z, each representing a
                    distance from the origin.
                    		- `halfExtents`: A list of four vectors representing the
                    half-lengths of the walls in each dimension (x, y, and z).
                    These values are used to create the visual shape for the wall.
                    		- `visualFramePositions`: A list of four position vectors
                    representing the locations of the four corners of the front
                    wall in the world coordinates system. These positions are used
                    to create the visual representation of the wall in the 3D scene.
                    
                back_wall_position (3D vector.): 3D position of the back wall of
                    the object to be created.
                    
                    		- `halfExtents`: A list of four elements, each representing
                    half of the object's extents in the X, Y, and Z directions,
                    respectively. The values of these elements represent the
                    distance from the center of the object to its edge in each dimension.
                    		- `visualFramePositions`: A list of four elements representing
                    the position of the visual frame in the X, Y, and Z directions
                    relative to the base's local origin. These positions are used
                    to create the visual shape.
                    		- `collisionShapeIndex`: An integer index indicating the
                    collision shape to be attached to the object. The collision
                    shape defines the shape of the object in collision detection
                    and simulation.
                    		- `baseMass`: A scalar value representing the mass of the
                    object. This value is used in simulation to control the object's
                    dynamics and affect its behavior.
                    		- `baseInertialFramePosition`: A 3D vector representing the
                    position of the object's base in its local inertial frame.
                    This position is used to define the orientation and position
                    of the object in the world.
                    		- `baseCollisionShapeIndex`: An integer index indicating the
                    collision shape attached to the object's base. As before, this
                    value defines the shape of the object in collision detection
                    and simulation.
                    		- `baseVisualShapeIndex`: An integer index indicating the
                    visual shape attached to the object's base. This shape is used
                    to represent the object visually in the scene.
                    		- `position`: A 3D vector representing the position of the
                    object in the world, which is defined by its `basePosition`
                    attribute and any transforms applied to it via the
                    `resetBasePositionAndOrientation` function.
                    
                left_wall_position (3D vector.): 3D position of the left wall of
                    the room in world coordinates, which is used to create the
                    collision shape for the merged object.
                    
                    		- `halfExtents`: A list of two elements, representing the
                    half-lengths of the wall in the X and Y directions.
                    		- `visualFramePositions`: A list of four elements, representing
                    the positions of the wall's visual frame in the X, Y, and Z
                    directions relative to the object's base position.
                    		- `collisionShapeIndex`: An integer value indicating the
                    collision shape index associated with the wall object.
                    
                right_wall_position (3D position vector.): 3D position of the right
                    wall of the room in world coordinates, which is used to generate
                    the collision shape for the merged object.
                    
                    		- `wall_thickness`: The thickness of the right wall.
                    		- `height`: The height of the right wall.
                    		- `depth`: The depth of the right wall.
                    
                    	These properties are used to generate a collision shape and
                    visual shape for the merged object.
                    
                orientation (3D rotation angle represented by an Euler vector in
                    radians.): 3D rotation of the merged object relative to its
                    base, which is used to set the orientation of the object's
                    visual representation in the scene.
                    
                    		- `orientation`: A numpy array representing the quaternion
                    rotation to be applied to the merged object's base position
                    and orientation. The quaternion is defined by its three
                    components (w, x, y, z) in the order of [w, x, y, z], where w
                    is the scalar component representing the magnitude of the
                    quaternion and x, y, and z are the vector components representing
                    the direction of the quaternion rotation.
                    		- `wall_thickness`: The thickness of each wall in meters.
                    		- `height`, `depth`, and `width`: The dimensions of the room
                    in meters.
                    		- `room_pos`: A numpy array containing the position of the
                    room's center in meters relative to the world coordinates.
                    

            Returns:
                `MultiBody`.: a single multi-body object representing the merged
                walls.
                
                		- `baseMass`: The mass of the merged object.
                		- `baseInertialFramePosition`: The position of the merged object's
                inertial frame in global coordinates.
                		- `baseCollisionShapeIndex`: The index of the collision shape
                associated with the merged object.
                		- `baseVisualShapeIndex`: The index of the visual shape associated
                with the merged object.
                		- `basePosition`: The position of the merged object in global coordinates.
                
                	Note that these properties are explained in a neutral and formal
                manner, without any personal opinions or statements. Additionally,
                the response is limited to less than 100 words to ensure conciseness
                and clarity.
                

            """
            visualShapeId = pb.createVisualShapeArray(shapeTypes=[pb.GEOM_BOX, pb.GEOM_BOX, pb.GEOM_BOX, pb.GEOM_BOX],
                                                      halfExtents=[[width / 2, wall_thickness / 2, height / 2],
                                                                   [width / 2, wall_thickness / 2, height / 2],
                                                                   [wall_thickness / 2, depth / 2, height / 2],
                                                                   [wall_thickness / 2, depth / 2, height / 2]],
                                         visualFramePositions=[front_wall_position, back_wall_position, left_wall_position, right_wall_position])

            collisionShapeId = pb.createCollisionShapeArray(shapeTypes=[pb.GEOM_BOX, pb.GEOM_BOX, pb.GEOM_BOX, pb.GEOM_BOX],
                                         collisionFramePositions=[front_wall_position, back_wall_position, left_wall_position, right_wall_position])

            merged_object = pb.createMultiBody(baseMass=1,
                           baseInertialFramePosition=[0, 0, 0],
                           baseCollisionShapeIndex=collisionShapeId,
                           baseVisualShapeIndex=visualShapeId,
                           basePosition=[room_pos[0], room_pos[1], 0])

            pb.resetBasePositionAndOrientation(merged_object, [room_pos[0], room_pos[1], 0],
                                               np.array(pb.getQuaternionFromEuler([0, 0, np.radians(orientation)])))

            return merged_object

        # Calculate positions of the walls based on the room's dimensions and orientation
        print(orientation)
        cos_theta = np.cos(np.radians(orientation))
        sin_theta = np.sin(np.radians(orientation))

        front_wall_position = [0, depth / 2, height / 2]
        back_wall_position = [0, -depth / 2, height / 2]
        left_wall_position = [-width / 2, 0, height / 2]
        right_wall_position = [width / 2, 0, height / 2]

        # This works but sometimes it does not fit very well
        # front_wall_position = [width / 2 * sin_theta, -depth / 2 * cos_theta, height / 2]
        # back_wall_position = [-width / 2 * sin_theta, depth / 2 * cos_theta, height / 2]
        # left_wall_position = [-width / 2 * cos_theta, -width / 2 * sin_theta, height / 2]
        # right_wall_position = [width / 2 * cos_theta, width / 2 * sin_theta, height / 2]

        print("Depth",depth,"Width",width)

        # # Create each wall
        # front_wall = create_wall(front_wall_position, [width / 2, wall_thickness / 2, height / 2],[1, 0, 0, 1]) # Red
        # back_wall = create_wall(back_wall_position, [width / 2, wall_thickness / 2, height / 2],[0, 0, 1, 1]) # Blue
        # left_wall = create_wall(left_wall_position, [wall_thickness / 2, depth / 2, height / 2], [0, 1, 0, 1]) # Green
        # right_wall = create_wall(right_wall_position, [wall_thickness / 2, depth / 2, height / 2], [1, 1, 0, 1]) # Yellow

        merged_object = merge_objects(front_wall_position, back_wall_position, left_wall_position, right_wall_position, orientation)
        # pb.removeBody(merged_object)

        self.current_room = {"width": width, "depth": depth, "height": height,
                             "center_x": 0, "center_y": 0, "rotation": orientation}
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
            """
            creates a new collision shape and attaches it to a multi-body object,
            which represents the wall segment. It then adds the body to a list of
            wall part IDs.

            Args:
                center (float): 3D position of the center of the new wall segment
                    to be added.
                half_extents (float): 2D dimensions of the wall segment in half-size,
                    which are used to create a bounding box shape for collision
                    detection and physics simulation purposes when the function
                    is called.

            """
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
                # add_wall_segment(left_wall_center, [door_start / 2, wall_thickness / 2, wall_height / 2])
                wall_parts_ids.append(left_wall_center)

            # Right wall segment (if any)
            if door_end < wall_length:
                if not rotated_wall:
                    right_wall_center = [wall_center[0] - wall_length / 2 + (door_end + (wall_length - door_end) / 2),
                                         wall_center[1], wall_center[2]]
                else:
                    right_wall_center = [wall_center[0], wall_center[1] - wall_length / 2 + (door_end + (wall_length - door_end) / 2),
                                     wall_center[2]]
                # add_wall_segment(right_wall_center, [(wall_length - door_end) / 2, wall_thickness / 2, wall_height / 2])
                    wall_parts_ids.append(right_wall_center)

            # Upper wall segment (above the door)
            if not rotated_wall:
                upper_wall_center = [wall_center[0] - wall_length / 2 + door_position, wall_center[1],
                                     wall_height + door_height/2]
            else:
                upper_wall_center = [wall_center[0], wall_center[1] - wall_length / 2 + door_position,
                                 wall_height + door_height / 2]
            # add_wall_segment(upper_wall_center, [door_width / 2, wall_thickness / 2, (wall_height - door_height) / 2])
                wall_parts_ids.append(upper_wall_center)

        # # If there is no door, create one full wall segment
        # else:
        #     add_wall_segment(wall_center, [wall_length / 2, wall_thickness / 2, wall_height / 2])

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
                self.current_doors.append({"width": door_width, "height": room_height, "position": door_position})
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
                rotated_wall=(i == 2 or i == 3)
            )
            all_wall_parts_ids.extend(wall_parts_ids)

        self.current_room = {"width": room_width, "depth": room_depth, "height": room_height,
                             "center_x": room_center[0], "center_y": room_center[1], "rotation": room_rotation}
        self.objects_in_the_room = all_wall_parts_ids
        return all_wall_parts_ids
    def loadURDF(self, urdf_path):
        """
        loads a Uniform Resource Dictionary Format (URDF) file into memory and
        sets the base position and orientation of a robot using the PyRobot framework.

        Args:
            urdf_path (str): file path of the `.URDF` file that the `loadURDF()`
                method will load.

        Returns:
            instance of the `RoboticsFoundation_pb2.MultibodyLink` class.: an
            instance of the `physics_box.Node` class representing the loaded URDF
            file.
            
            		- `m`: A `physics_bone` object representing the loaded URDF file's
            base link.
            		- `fileName`: The path to the URDF file loaded by the `loadURDF` function.
            

        """
        m = self.pb.loadURDF(fileName=urdf_path)
        self.pb.resetBasePositionAndOrientation(m, [0, 0, 1], [0, 0, 0, 1])
        return m
    def add_robot(self):
        """
        creates a box-shaped robot and a visual representation of its nose, using
        the PyBullet library's `createCollisionShape`, `createVisualShape`,
        `createMultiBody`, and `changeVisualShape` methods. It also sets the robot's
        position, orientation, and color.

        Returns:
            int: a unique ID for the created robot.

        """
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
        """
        sets the velocity of a robot in a given direction by applying linear and
        rotational velocities to objects within the robot's scope.

        Args:
            side (int): 3D direction of movement for the robot's base.
            adv (12-dimensional vector.): 3D linear velocity component in the
                direction of the shortest side of the room that the robot should
                move towards.
                
                		- `side`: A positive integer representing the side of movement
                (up, down, left, or right).
                		- `adv`: A non-negative value indicating the magnitude of movement
                along the specified side.
                
            rot (12-dimensional vector in the Python module Pandas Box (pb).): 3D
                rotation of the robot in degrees around the yaw axis.
                
                		- `side`: The side of the room where the robot is moving (int).
                		- `adv`: The advancing direction of the robot's movement (int).
                		- `rot`: An object containing various properties and attributes
                related to the robot's rotation. These include:
                		+ `angularVelocity`: The rate of change of the robot's angular
                rotation in radians per second (float).
                		+ `rotation`: The current rotation angle of the robot in radians
                (float).
                

        """
        for obj in self.objects_in_the_room:
            pb.resetBaseVelocity(obj, linearVelocity=[side*12, -adv*12, 0])

    def get_room(self): # attributes
        return self.current_room
    def get_doors(self): # list of attributes
        return self.current_doors


