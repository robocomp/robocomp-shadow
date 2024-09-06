import json
import copy
import os
import math

# Get folder number in "/home/robocomp/robocomp/components/proceduralRoomGeneration/generatedRooms/"
def get_folder_number():
    folder = "/home/robocomp/robocomp/components/proceduralRoomGeneration/generatedRooms/"
    return len([name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])

scenario_number = "5"

folder_number = get_folder_number()
# Iterate over all the folders in the directory
# for i in range(folder_number):
#     # Open the json file
#     with open(f"/home/robocomp/robocomp/components/proceduralRoomGeneration/generatedRooms/{i}/apartmentData.json") as f:
#         data = json.load(f)
#         # Get first room center
#         room_center = data['rooms'][0]['center']
#         print(room_center)
#         # Substract room_center to all the rooms and doors
#         for room in data['rooms']:
#             room['center'][0] -= room_center[0]
#             room['center'][1] -= room_center[1]
#         for door in data['doors']:
#             door['center'][0] -= room_center[0]
#             door['center'][1] -= room_center[1]
#         print("Ground truth")
#         print(data['rooms'])
#         print(data['doors'])
#     # Considering files which name starts with "generated_data", iterate over them
#     for file in os.listdir(f"/home/robocomp/robocomp/components/proceduralRoomGeneration/generatedRooms/{i}"):
#         if file.startswith("generated_data"):
#             # Open the json file
#             with open(f"/home/robocomp/robocomp/components/proceduralRoomGeneration/generatedRooms/{i}/{file}") as f:
#                 data = json.load(f)
#                 print("Generated data")
#                 print(data['rooms'])
#                 print(data['doors'])

# Open the json file
with open(f"/home/robocomp/robocomp/components/proceduralRoomGeneration/generatedRooms/"+scenario_number+"/apartmentData.json") as f:
    gt_data = json.load(f)
    # Get first room center
    room_center = copy.deepcopy(gt_data['rooms'][0]['center'])
    print(gt_data['rooms'])
    # Substract room_center to all the rooms and doors
    for room in gt_data['rooms']:
        room['center'][0] -= room_center[0]
        room['center'][1] -= room_center[1]
        room['center'][0] *= 1000
        room['center'][1] *= 1000
        room["x"] *= 1000
        room["y"] *= 1000
        room["x"] -= 200
        room["y"] -= 200

    for door in gt_data['doors']:
        door['center'][0] -= room_center[0]
        door['center'][1] -= room_center[1]
        door['center'][0] *= 1000
        door['center'][1] *= 1000
        door["width"] *= 1000
    print(gt_data['rooms'])
# Considering files which name starts with "generated_data", iterate over them
for file in os.listdir(f"/home/robocomp/robocomp/components/proceduralRoomGeneration/generatedRooms/"+scenario_number):
    if file.startswith("generated_data"):
        # Open the json file
        with open("/home/robocomp/robocomp/components/proceduralRoomGeneration/generatedRooms/"+scenario_number+f"/{file}") as f:
            data = json.load(f)

            # Match ground truth with generated data using Hungarian algorithm
            # Hungarian algorithm
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
            from scipy.optimize import linear_sum_assignment
            import numpy as np
            cost_matrix = np.zeros((len(gt_data['rooms']), len(data['rooms'])))
            for i in range(len(gt_data['rooms'])):
                for j in range(len(data['rooms'])):
                    cost_matrix[i][j] = math.sqrt(pow(gt_data['rooms'][i]['center'][0] - data['rooms'][j]['global_center'][0], 2) + pow(gt_data['rooms'][i]['center'][1] - data['rooms'][j]['global_center'][1], 2))
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # Print the matched rooms using gt_data and data
            print("Matched rooms")
            for i in range(len(row_ind)):
                print(gt_data['rooms'][row_ind[i]], data['rooms'][col_ind[i]])
                # Calculate the error between the matched rooms in every axis considering sign
                error_x = gt_data['rooms'][row_ind[i]]['center'][0] - data['rooms'][col_ind[i]]['global_center'][0]
                error_y = gt_data['rooms'][row_ind[i]]['center'][1] - data['rooms'][col_ind[i]]['global_center'][1]
                error_width = gt_data['rooms'][row_ind[i]]['x'] - data['rooms'][col_ind[i]]['x']
                error_height = gt_data['rooms'][row_ind[i]]['y'] - data['rooms'][col_ind[i]]['y']
                print("Errors: ", error_x, error_y, error_width, error_height)


            # Match doors
            cost_matrix = np.zeros((len(gt_data['doors']), len(data['doors'])))
            for i in range(len(gt_data['doors'])):
                for j in range(len(data['doors'])):
                    cost_matrix[i][j] = math.sqrt(pow(gt_data['doors'][i]['center'][0] - data['doors'][j]['global_center'][0], 2) + pow(gt_data['doors'][i]['center'][1] - data['doors'][j]['global_center'][1], 2))

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # Print the matched doors using gt_data and data
            print("Matched doors")
            for i in range(len(row_ind)):
                print(gt_data['doors'][row_ind[i]], data['doors'][col_ind[i]])
                # Calculate the error between the matched doors in every axis considering sign
                error_x = gt_data['doors'][row_ind[i]]['center'][0] - data['doors'][col_ind[i]]['global_center'][0]
                error_y = gt_data['doors'][row_ind[i]]['center'][1] - data['doors'][col_ind[i]]['global_center'][1]
                error_width = gt_data['doors'][row_ind[i]]['width'] - data['doors'][col_ind[i]]['width']
                print("Errors: ", error_x, error_y, error_width)

            # Draw space with matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            fig, ax = plt.subplots()
            min_x = min([room['center'][0] - room['x'] / 2 for room in gt_data['rooms']])
            max_x = max([room['center'][0] + room['x'] / 2 for room in gt_data['rooms']])
            min_y = min([room['center'][1] - room['y'] / 2 for room in gt_data['rooms']])
            max_y = max([room['center'][1] + room['y'] / 2 for room in gt_data['rooms']])
            for room in gt_data['rooms']:
                ax.add_patch(
                    patches.Rectangle((room['center'][0] - room['x'] / 2, room['center'][1] - room['y'] / 2), room['x'],
                                      room['y'], fill=False))
                ax.add_patch(patches.Circle((room['center'][0], room['center'][1]), 100, fill=False))
            for door in gt_data['doors']:
                ax.add_patch(patches.Circle((door['center'][0], door['center'][1]), door['width'] / 2, fill=False))
            # Adapt the axis to the rooms and doors
            plt.xlim(min_x - 1000, max_x + 1000)
            plt.ylim(min_y - 1000, max_y + 1000)

            # Add the generated rooms and doors in other subplot
            min_x = min([room['global_center'][0] - room['x'] / 2 for room in data['rooms']])
            max_x = max([room['global_center'][0] + room['x'] / 2 for room in data['rooms']])
            min_y = min([room['global_center'][1] - room['y'] / 2 for room in data['rooms']])
            max_y = max([room['global_center'][1] + room['y'] / 2 for room in data['rooms']])
            for room in data['rooms']:
                ax.add_patch(
                    patches.Rectangle((room['global_center'][0] - room['x'] / 2, room['global_center'][1] - room['y'] / 2), room['x'],
                                      room['y'], fill=False, color="red"))
                ax.add_patch(patches.Circle((room['global_center'][0], room['global_center'][1]), 100, fill=False, color="red"))
            for door in data['doors']:
                ax.add_patch(patches.Circle((door['global_center'][0], door['global_center'][1]), door['width'] / 2, fill=False, color="red"))
            # Adapt the axis to the rooms and doors
            plt.xlim(min_x - 1000, max_x + 1000)
            plt.ylim(min_y - 1000, max_y + 1000)
            plt.show()