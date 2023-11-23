import pybullet
import time

class PersonManager:
    def __init__(self, physics_client_id, _pybullet):
        self.pybullet = _pybullet
        self.physics_client_id = physics_client_id
        self.persons = {}
        self.last_velocities = {}
        self.last_orientation = {}
        self.velocity_history = {}  # Historial de velocidades para cada persona
        self.observation_count = {}
        self.M = 15 #Número de frames pre-inserción persona.
        self.N = 10  # Número de muestras a mantener

        self.last_seen = {}  # Nuevo: Registro del  último momento en que se vio a cada persona
        self.disappearance_time = 4.5  # Tiempo en segundos para considerar desaparecida a una persona

    def add_person(self, person_name, position, orientation, size):

        # Cargar el modelo URDF
        mass = 0.5
        if self.observation_count[person_name] >= self.M:
            person_id = self.pybullet.createCollisionShape(self.pybullet.GEOM_BOX, halfExtents=size, physicsClientId=self.physics_client_id)
            visual_shape_id = self.pybullet.createVisualShape(self.pybullet.GEOM_BOX, halfExtents=size, physicsClientId=self.physics_client_id)

            person_body_id = self.pybullet.createMultiBody(baseMass=mass,
                                               baseCollisionShapeIndex=person_id,
                                               baseVisualShapeIndex=visual_shape_id,
                                               basePosition=position,
                                               physicsClientId=self.physics_client_id)
            # Ruta al archivo URDF
            self.persons[person_name] = person_body_id
            return person_body_id
        else:
            return None

    def remove_person(self, person_name):
        if person_name in self.persons:
            self.pybullet.removeBody(self.persons[person_name], physicsClientId=self.physics_client_id)
            del self.persons[person_name]

    def change_velocity(self, person_name, velocity, pose, orientation):
        if person_name in self.persons:
            self.pybullet.resetBaseVelocity(self.persons[person_name], linearVelocity=velocity, physicsClientId=self.physics_client_id)
            pos, _ = self.pybullet.getBasePositionAndOrientation(self.persons[person_name])
            # self.pybullet.resetBasePositionAndOrientation(self.persons[person_name], pose, orientation, physicsClientId=self.physics_client_id )


    def get_person_position(self, person_name):
        if person_name in self.persons:
            person_body_id = self.persons[person_name]
            position, orientation = self.pybullet.getBasePositionAndOrientation(person_body_id, physicsClientId=self.physics_client_id)
            return position  # Devuelve solo la posición, ignorando la orientación
        return None  # Devuelve None si la persona no existe en la simulación

    def update_person(self, person_name, position, orientation, velocity):
        if person_name in self.persons:
            person_body_id = self.persons[person_name]
            self.pybullet.resetBasePositionAndOrientation(person_body_id, position, orientation,
                                              physicsClientId=self.physics_client_id)
            self.pybullet.resetBaseVelocity(person_body_id, linearVelocity=velocity, physicsClientId=self.physics_client_id)
            # Actualizar historial de velocidades
            if person_name not in self.velocity_history:
                self.velocity_history[person_name] = []
            self.velocity_history[person_name].append(velocity)
            if len(self.velocity_history[person_name]) > self.N:
                self.velocity_history[person_name].pop(0)
            self.last_orientation[person_name] = orientation

            # Actualizar el último momento visto
            self.last_seen[person_name] = time.time()

    def apply_last_velocity(self, person_name):
        if person_name in self.persons:
            person_body_id = self.persons[person_name]

            # Usar el promedio de las últimas N velocidades
            if person_name in self.velocity_history and self.velocity_history[person_name]:
                average_velocity = [sum(vals) / len(vals) for vals in zip(*self.velocity_history[person_name])]
                print("AVG", average_velocity)
                pose, orientation = self.pybullet.getBasePositionAndOrientation(person_body_id)
                self.pybullet.resetBasePositionAndOrientation(person_body_id, pose, self.last_orientation[person_name],
                                                  physicsClientId=self.physics_client_id)
                self.pybullet.resetBaseVelocity(person_body_id, linearVelocity=average_velocity,
                                    physicsClientId=self.physics_client_id)

    def remove_disappeared_persons(self):
        current_time = time.time()
        for person_name in list(self.last_seen.keys()):
            if current_time - self.last_seen[person_name] > self.disappearance_time:
                # La persona ha desaparecido durante demasiado tiempo, eliminarla
                if person_name in self.persons:
                    self.pybullet.removeBody(self.persons[person_name], physicsClientId=self.physics_client_id)
                    del self.persons[person_name]
                del self.last_seen[person_name]
                if person_name in self.velocity_history:
                    del self.velocity_history[person_name]
                if person_name in self.last_orientation:
                    del self.last_orientation[person_name]
                if person_name in self.last_velocities:
                    del self.last_velocities[person_name]

            # # Aplicar la última velocidad conocida
            # if person_name in self.last_velocities:
            #     velocity = self.last_velocities[person_name]
            #     pose , _ = self.pybullet.getBasePositionAndOrientation(person_body_id)
            #     self.pybullet.resetBasePositionAndOrientation(person_body_id, pose, self.last_orientation[person_name],
            #                                       physicsClientId=self.physics_client_id)
            #     self.pybullet.resetBaseVelocity(person_body_id, linearVelocity=velocity, physicsClientId=self.physics_client_id)


    # def add_person(self, person_name, position, qorientation, size, mass=0.5):
    #
    #     # person_id = self.pybullet.createCollisionShape(self.pybullet.GEOM_BOX, halfExtents=size, physicsClientId=self.physics_client_id)
    #     # visual_shape_id = self.pybullet.createVisualShape(self.pybullet.GEOM_BOX, halfExtents=size, physicsClientId=self.physics_client_id)
    #
    #     urdf_path = '/home/robolab/.local/lib/python3.10/site-packages/pybullet_data/humanoid/humanoid.urdf'
    #
    #     body = self.pybullet.loadURDF(urdf_path, basePosition=[0,0,0],
    #                       baseOrientation=[0,0,0,0], globalScaling=1)
    #
    #     person_id = self.pybullet.createCollisionShape(body, halfExtents=[1,1,1], physicsClientId=self.physics_client_id)
    #     visual_shape_id = self.pybullet.createVisualShape(body, halfExtents=[1,1,1], physicsClientId=self.physics_client_id)
    #
    #     person_body_id = self.pybullet.createMultiBody(baseMass=mass,
    #                                        baseCollisionShapeIndex=person_id,
    #                                        baseVisualShapeIndex=visual_shape_id,
    #                                        basePosition=position,
    #                                        physicsClientId=self.physics_client_id)
    #
    #
    #     self.persons[person_name] = person_body_id
    #     # Actualizar el último momento visto
    #     self.last_seen[person_name] = time.time()
    #     return person_body_id