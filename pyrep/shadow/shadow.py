from pyrep.robots.mobiles.holonomic_base import HolonomicBase


class Shadow(HolonomicBase):
    def __init__(self, count: int = 0, distance_from_target: float = 0):
        super().__init__(
            count, 4, distance_from_target, '/Shadow', 4, 6, 0.035)
