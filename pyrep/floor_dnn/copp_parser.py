#!/usr/bin/python3
# -*- coding: utf-8 -*-

from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
import numpy as np
from pprint import pprint
import traceback
import json

# rooms -> 50
# walls -> 30
# objects -> 60
# floors -> 20

_OBJECT_NAMES = ['person', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                 'chair', 'couch',
                 'potted_plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

SCENE_FILE = scene_file = '../../etc/salabeta_door.ttt'
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()

world = {"DSRModel": {"symbols": {}}}
# world
world["DSRModel"]["symbols"]["1"] = {"attribute": {
        "OuterRegionBottom": {
            "type": 1,
            "value": -7500
         },
         "OuterRegionLeft": {
            "type": 1,
            "value": -7500
         },
        "OuterRegionRight": {
            "type": 1,
            "value": 7500
        },
        "OuterRegionTop": {
            "type": 1,
            "value": 7500
        },
        "color": {
            "type": 0,
            "value": "SeaGreen"
        },
        "level": {
            "type": 1,
            "value": 0
        },
        "parent": {
            "type": 7,
            "value": "0"
        },
        "pos_x": {
            "type": 2,
            "value": 220
        },
        "pos_y": {
            "type": 2,
            "value": -400
        }
    },
    "id": "1",
    "links": [
        {
            "dst": "50",    # floor
            "label": "RT",
            "linkAttribute": {
                "rt_rotation_euler_xyz": {
                    "type": 3,
                    "value": [
                        0,
                        0,
                        0
                    ]
                },
                "rt_translation": {
                    "type": 3,
                    "value": [
                        0,
                        0,
                        0
                    ]
                }
            },
            "src": "1"
        }
    ],
    "name": "world",
    "type": "world"}

#robot SHADOW
world["DSRModel"]["symbols"]["200"] = {
    "attribute": {
        "base_target_x": {
            "type": 2,
            "value": 3600
        },
        "base_target_y": {
            "type": 2,
            "value": 1550
        },
        "color": {
            "type": 0,
            "value": "Blue"
        },
        "level": {
            "type": 1,
            "value": 2
        },
        "parent": {
            "type": 7,
            "value": "50"
        },
        "port": {
            "type": 1,
            "value": 12238
        },
        "pos_x": {
            "type": 2,
            "value": -174.637238
        },
        "pos_y": {
            "type": 2,
            "value": -74.730019
        },
        "robot_ref_adv_speed": {
            "type": 2,
            "value": 0
        },
        "robot_ref_rot_speed": {
            "type": 2,
            "value": 0
        },
        "robot_ref_side_speed": {
            "type": 2,
            "value": 0
        }
    },
    "id": "200",
    "links": [
        {
            "dst": "50",
            "label": "in",
            "linkAttribute": {
            },
            "src": "200"
        },
        {
            "dst": "201",
            "label": "RT",
            "linkAttribute": {
                "rt_rotation_euler_xyz": {
                    "type": 3,
                    "value": [
                        0,
                        0,
                        0
                    ]
                },
                "rt_translation": {
                    "type": 3,
                    "value": [
                        0,
                        0,
                        0
                    ]
                }
            },
            "src": "200"
        },
        {
            "dst": "206",
            "label": "RT",
            "linkAttribute": {
                "rt_rotation_euler_xyz": {
                    "type": 3,
                    "value": [
                        -0.40142571926116943,
                        0,
                        0
                    ]
                },
                "rt_translation": {
                    "type": 3,
                    "value": [
                        0,
                        0,
                        1569
                    ]
                }
            },
            "src": "200"
        },
        {
            "dst": "211",
            "label": "RT",
            "linkAttribute": {
                "rt_rotation_euler_xyz": {
                    "type": 3,
                    "value": [
                        0,
                        0,
                        0
                    ]
                },
                "rt_translation": {
                    "type": 3,
                    "value": [
                        0,
                        245,
                        574
                    ]
                }
            },
            "src": "200"
        },
        {
            "dst": "1000",
            "label": "has",
            "linkAttribute": {
                "rt_rotation_euler_xyz": {
                    "type": 3,
                    "value": [
                        0,
                        0,
                        0
                    ]
                },
                "rt_translation": {
                    "type": 3,
                    "value": [
                        0,
                        0,
                        0
                    ]
                }
            },
            "src": "200"
        }
    ],
    "name": "robot",
    "type": "omnirobot"
}       # robot
world["DSRModel"]["symbols"]["201"] = {
    "attribute": {
        "color": {
            "type": 0,
            "value": "LightBlue"
        },
        "depth": {
            "type": 1,
            "value": 1
        },
        "height": {
            "type": 1,
            "value": 1
        },
        "level": {
            "type": 1,
            "value": 3
        },
        "parent": {
            "type": 7,
            "value": "200"
        },
        "path": {
            "type": 0,
            "value": "../../etc/viriato_base_concept_3/viriato_mesh.ive"
        },
        "pos_x": {
            "type": 2,
            "value": -100.720825
        },
        "pos_y": {
            "type": 2,
            "value": -291.153015
        },
        "scalex": {
            "type": 1,
            "value": 1
        },
        "scaley": {
            "type": 1,
            "value": 1
        },
        "scalez": {
            "type": 1,
            "value": 1
        },
        "width": {
            "type": 1,
            "value": 1
        }
    },
    "id": "201",
    "links": [
    ],
    "name": "shadow_mesh",
    "type": "mesh"
}       # mesh
world["DSRModel"]["symbols"]["206"] = {
    "attribute": {
        "color": {
            "type": 0,
            "value": "SteelBlue"
        },
        "level": {
            "type": 1,
            "value": 3
        },
        "mass": {
            "type": 1,
            "value": 0
        },
        "nose_pose_ref": {
            "type": 3,
            "value": [
                1000,
                0,
                0
            ]
        },
        "parent": {
            "type": 7,
            "value": "200"
        },
        "pos_x": {
            "type": 2,
            "value": -253.711151
        },
        "pos_y": {
            "type": 2,
            "value": -185.093811
        }
    },
    "id": "206",
    "links": [
        {
            "dst": "210",
            "label": "RT",
            "linkAttribute": {
                "rt_rotation_euler_xyz": {
                    "type": 3,
                    "value": [
                        0,
                        0,
                        0
                    ]
                },
                "rt_translation": {
                    "type": 3,
                    "value": [
                        0,
                        0,
                        74
                    ]
                }
            },
            "src": "206"
        }
    ],
    "name": "camera_pan_joint",
    "type": "joint"
}       # joint
world["DSRModel"]["symbols"]["210"] = {
    "attribute": {
        "cam_depth": {
            "type": 5,
            "value": [
            ]
        },
        "cam_depthFactor": {
            "type": 2,
            "value": 1
        },
        "cam_depth_alivetime": {
            "type": 7,
            "value": 1620400089
        },
        "cam_depth_cameraID": {
            "type": 1,
            "value": 0
        },
        "cam_depth_focalx": {
            "type": 1,
            "value": 554
        },
        "cam_depth_focaly": {
            "type": 1,
            "value": 554
        },
        "cam_depth_height": {
            "type": 1,
            "value": 480
        },
        "cam_depth_width": {
            "type": 1,
            "value": 640
        },
        "cam_rgb": {
            "type": 5,
            "value": [
            ]
        },
        "cam_rgb_alivetime": {
            "type": 7,
            "value": 1620400089
        },
        "cam_rgb_cameraID": {
            "type": 1,
            "value": 0
        },
        "cam_rgb_depth": {
            "type": 1,
            "value": 3
        },
        "cam_rgb_focalx": {
            "type": 1,
            "value": 554
        },
        "cam_rgb_focaly": {
            "type": 1,
            "value": 554
        },
        "cam_rgb_height": {
            "type": 1,
            "value": 480
        },
        "cam_rgb_width": {
            "type": 1,
            "value": 640
        },
        "color": {
            "type": 0,
            "value": "Blue"
        },
        "height": {
            "type": 1,
            "value": 480
        },
        "ifconfig": {
            "type": 0,
            "value": "40000,50000"
        },
        "level": {
            "type": 1,
            "value": 4
        },
        "parent": {
            "type": 7,
            "value": "206"
        },
        "pos_x": {
            "type": 2,
            "value": -383.363739
        },
        "pos_y": {
            "type": 2,
            "value": -257.117462
        },
        "width": {
            "type": 1,
            "value": 640
        }
    },
    "id": "210",
    "links": [
    ],
    "name": "camera_top",
    "type": "rgbd"
}       # top_camera
world["DSRModel"]["symbols"]["211"] = {
    "attribute": {
        "cam_depth": {
            "type": 5,
            "value": [
            ]
        },
        "cam_depthFactor": {
            "type": 2,
            "value": 1
        },
        "cam_depth_alivetime": {
            "type": 7,
            "value": 1620400089
        },
        "cam_depth_cameraID": {
            "type": 1,
            "value": 0
        },
        "cam_depth_focalx": {
            "type": 1,
            "value": 554
        },
        "cam_depth_focaly": {
            "type": 1,
            "value": 554
        },
        "cam_depth_height": {
            "type": 1,
            "value": 480
        },
        "cam_depth_width": {
            "type": 1,
            "value": 640
        },
        "cam_rgb": {
            "type": 5,
            "value": [
            ]
        },
        "cam_rgb_alivetime": {
            "type": 7,
            "value": 1620400089
        },
        "cam_rgb_cameraID": {
            "type": 1,
            "value": 0
        },
        "cam_rgb_depth": {
            "type": 1,
            "value": 3
        },
        "cam_rgb_focalx": {
            "type": 1,
            "value": 554
        },
        "cam_rgb_focaly": {
            "type": 1,
            "value": 554
        },
        "cam_rgb_height": {
            "type": 1,
            "value": 480
        },
        "cam_rgb_width": {
            "type": 1,
            "value": 640
        },
        "color": {
            "type": 0,
            "value": "Blue"
        },
        "height": {
            "type": 1,
            "value": 480
        },
        "ifconfig": {
            "type": 0,
            "value": "40000,50000"
        },
        "level": {
            "type": 1,
            "value": 3
        },
        "parent": {
            "type": 7,
            "value": "200"
        },
        "pos_x": {
            "type": 2,
            "value": -353.363739
        },
        "pos_y": {
            "type": 2,
            "value": -277.117462
        },
        "width": {
            "type": 1,
            "value": 640
        }
    },
    "id": "211",
    "links": [
    ],
    "name": "omni_camera",
    "type": "rgbd"
}       # omni_camera

# Read rooms
room_id = 50
floor_id = 20
i = 0
while True:
    try:
        room = "/room_" + str(i)
        room_handle = Shape(room)
        print("Read", room)
        world["DSRModel"]["symbols"][str(room_id)] = {"id": str(room_id), "links": [], "name": "room_"+str(i), "type": "room", "attribute": {
            "color": {
                "type": 0,
                "value": "White"
            },
            "level": {
                "type": 1,
                "value": 1  # under world
            },
            "parent": {
                "type": 7,
                "value": "1"   # world
            },
            "pos_x": {
                "type": 2,
                "value": np.random.uniform(0, 200, 1)[0]
            },
            "pos_y": {
                "type": 2,
                "value": np.random.uniform(-100, -500, 1)[0]
            },
            "texture": {
                "type": 0,
                "value": "#dbdbdb"
            },
            "bounding_box": {
                "type": 3,  # vec6
                 "value": [v*1000 for v in room_handle.get_bounding_box()]
            }
        }}

        # read and connect floor
        handle = Shape(room + "/floor_" + str(i))
        world["DSRModel"]["symbols"][str(floor_id)] = {"id": str(floor_id), "links": [], "name": "floor_" + str(i), "type": "plane", "attribute": {
            "color": {
                "type": 0,
                "value": "LightGrey"
            },
            "level": {
                "type": 1,
                "value": 2  # under room
            },
            "parent": {
                "type": 7,
                "value": "50"  # room
            },
            "pos_x": {
                "type": 2,
                "value": np.random.uniform(100, 500, 1)[0]
            },
            "pos_y": {
                "type": 2,
                "value": np.random.uniform(100, 500, 1)[0]
            },
            "texture": {
                "type": 0,
                "value": "#dbdbdb"
            },
            "depth": {      # Z world axis
                "type": 1,
                "value": int((abs(handle.get_bounding_box()[4]) + abs(handle.get_bounding_box()[5]))*1000.0)
            },
            "height": {     # Y world axis
                "type": 1,
                "value": int((abs(handle.get_bounding_box()[2]) + abs(handle.get_bounding_box()[3]))*1000.0)
            },
            "width": {      # X world axis
                "type": 1,
                "value": int((abs(handle.get_bounding_box()[1]) + abs(handle.get_bounding_box()[0]))*1000.0)
            },
            "bounding_box": {
                "type": 3,  # vec6
                "value":  [v*1000 for v in handle.get_bounding_box()]
            }
        }}
        # connect from room
        world["DSRModel"]["symbols"][str(room_id)]["links"].append({"dst": str(floor_id),
                                    "label": "RT",
                                    "linkAttribute": {
                                        "rt_rotation_euler_xyz": {
                                            "type": 3,
                                            "value": [0, 0, 0]
                                        },
                                        "rt_translation": {
                                            "type": 3,
                                            "value": [0, 0, -60]
                                        }
                                    },
                                    "src": str(room_id)})

        # read and connect walls
        wall_id = 30
        w = 0
        while True:
            try:
                handle = Shape(room + "/Walls/wall_" + str(w))
                print("Read", room + "/Walls/wall_" + str(w))
                world["DSRModel"]["symbols"][str(wall_id)] = {"id": str(wall_id), "links": [], "name": "wall_" + str(w), "type": "plane", "attribute": {
                    "color": {
                        "type": 0,
                        "value": "Khaki"
                    },
                    "level": {
                        "type": 1,
                        "value": 2  # under room
                    },
                    "parent": {
                        "type": 7,
                        "value": "50"  # room
                    },
                    "pos_x": {
                        "type": 2,
                        "value": np.random.uniform(400, 500, 1)[0]
                    },
                    "pos_y": {
                        "type": 2,
                        "value": np.random.uniform(-500, 500, 1)[0]
                    },
                    "texture": {
                        "type": 0,
                        "value": "#dbdbdb"
                    },
                    "depth": {
                        "type": 1,
                        "value": int((abs(handle.get_bounding_box()[5]) + abs(handle.get_bounding_box()[4]))*1000.0)
                    },
                    "height": {
                        "type": 1,
                        "value": int((abs(handle.get_bounding_box()[2]) + abs(handle.get_bounding_box()[3]))*1000.0)
                    },
                    "width": {
                        "type": 1,
                        "value": int((abs(handle.get_bounding_box()[1]) + abs(handle.get_bounding_box()[0]))*1000.0)
                    },
                    "bounding_box": {
                        "type": 3,  # vec6
                        "value": [v*1000 for v in handle.get_bounding_box()]
                    }
                }}
                # connect from rooms
                print("---- ", handle.get_orientation(room_handle))
                world["DSRModel"]["symbols"][str(room_id)]["links"].append({"dst": str(wall_id),
                                            "label": "RT",
                                            "linkAttribute": {
                                                "rt_rotation_euler_xyz": {
                                                    "type": 3,
                                                    "value":  handle.get_orientation(room_handle).tolist()
                                                },
                                                "rt_translation": {
                                                    "type": 3,
                                                    "value": (handle.get_position(room_handle)*1000).tolist()
                                                }
                                            },
                                            "src": str(room_id)})
                wall_id += 1
                w += 1
            except:
                break

        # read and connect objects
        object_id = 60
        o = 0
        for name in _OBJECT_NAMES:
            try:
                handle = Shape(room + "/Objects/" + name)
                world["DSRModel"]["symbols"][str(object_id)] = {"id": str(object_id), "links": [], "name": name, "type": name, "attribute": {
                    "color": {
                        "type": 0,
                        "value": "Green"
                    },
                    "level": {
                        "type": 1,
                        "value": 2  # under room
                    },
                    "parent": {
                        "type": 7,
                        "value": "50"  # room
                    },
                    "pos_x": {
                        "type": 2,
                        "value": np.random.uniform(100, 500, 1)[0]
                    },
                    "pos_y": {
                        "type": 2,
                        "value": np.random.uniform(-100, -500, 1)[0]
                    },
                    "texture": {
                        "type": 0,
                        "value": "#dbdbdb"
                    },
                    "depth": {
                        "type": 1,
                        "value": int((abs(handle.get_bounding_box()[5]) + abs(handle.get_bounding_box()[4]))*1000.0)
                    },
                    "height": {
                        "type": 1,
                        "value": int((abs(handle.get_bounding_box()[2]) + abs(handle.get_bounding_box()[3]))*1000.0)
                    },
                    "width": {
                        "type": 1,
                        "value": int((abs(handle.get_bounding_box()[1]) + abs(handle.get_bounding_box()[0]))*1000.0)
                    },
                    "bounding_box": {
                        "type": 3,  # vec6
                        "value": [v*1000 for v in handle.get_bounding_box()]
                    }
                }}
                # connect from rooms
                world["DSRModel"]["symbols"][str(room_id)]["links"].append({"dst": str(object_id),
                                                "label": "RT",
                                                "linkAttribute": {
                                                    "rt_rotation_euler_xyz": {
                                                        "type": 3,
                                                        "value": handle.get_orientation(room_handle).tolist()
                                                    },
                                                    "rt_translation": {
                                                        "type": 3,
                                                        "value": (handle.get_position(room_handle)*1000).tolist()
                                                    }
                                                },
                                                "src": str(room_id)})
                object_id += 1
                o += 1
            except:
                pass

        # connect robot
        robot_handle = Shape("/room_0/Shadow")
        world["DSRModel"]["symbols"][str(room_id)]["links"].append({"dst": "200",
                                                                    "label": "RT",
                                                                    "linkAttribute": {
                                                                        "rt_rotation_euler_xyz": {
                                                                            "type": 3,
                                                                            "value": robot_handle.get_orientation(
                                                                                room_handle).tolist()
                                                                        },
                                                                        "rt_translation": {
                                                                            "type": 3,
                                                                            "value": (robot_handle.get_position(
                                                                                room_handle) * 1000).tolist()
                                                                        }
                                                                    },
                                                                    "src": str(room_id)})

        room_id += 1
        i += 1
    except:
        traceback.print_exc()
        break

pprint(world)
with open('world.json', 'w') as f:
    json.dump(world, f,  indent=4, sort_keys=True)

print("---------------- FILE SAVED ----------------")
    
    

