import json
import numpy as np

def frPyObjects(i, pyobj):
    # encode rle from a list of python objects
    if type(pyobj) == np.ndarray:
        print("{}, {}, {}".format(i, type(pyobj), len(pyobj[0])))
    elif type(pyobj) == list and len(pyobj[0]) == 4:
        print("{}, {}, {}".format(i, type(pyobj), len(pyobj[0])))
    elif type(pyobj) == list and len(pyobj[0]) > 4:
        print("{}, {}, {}".format(i, type(pyobj), len(pyobj[0])))
    elif type(pyobj) == list and type(pyobj) == dict and 'counts' in pyobj[0] and 'size' in pyobj[0]:
        print("{}, {}, {}".format(i, type(pyobj), len(pyobj[0])))
    # encode rle from single python object
    elif type(pyobj) == list and len(pyobj[0]) == 4:
        print("{}, {}, {}".format(i, type(pyobj), len(pyobj[0])))
    elif type(pyobj) == list and len(pyobj[0]) > 4:
        print("{}, {}, {}".format(i, type(pyobj), len(pyobj[0])))
    elif type(pyobj) == dict and 'counts' in pyobj and 'size' in pyobj:
        print("{}, {}, {}".format(i, type(pyobj), len(pyobj[0])))
    else:
        print("{}, {}, {}".format(i, type(pyobj), len(pyobj[0])))
        raise Exception('input type is not supported.')


JSON_LOC="/ssd3/wyy/projects/mmdetection/data/ESD_COCO_instances/annotations/instances_val2017.json"

#Open JSON
val_json = open(JSON_LOC, "r")
json_object = json.load(val_json)
val_json.close()

for i, instance in enumerate(json_object["annotations"]):
    frPyObjects(i, instance["segmentation"])
    # if(type(instance["segmentation"]) != list):
    #     print("i: ", type(instance["segmentation"]))

    # if len(instance["segmentation"][0]) == 4:
    #     print("instance number", i, "raises arror:", instance["segmentation"][0])

# #Alter object generating the error with something random not causing the error
# json_object["annotations"][1510]["segmentation"] = [[230.83333333333331, 773.8888888888889, 231.83333333333331, 773.8888888888889, 237.22222222222223, 770.5555555555555]]

# #Write back altered JSON
# val_json = open(JSON_LOC, "w")
# json.dump(json_object, val_json)
# val_json.close()

