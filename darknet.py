#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"
- Set environment variable "DARKNET_PATH" to path darknet lib .so (for Linux)

Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)

Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn
@date: 20180503
"""
from ctypes import *
import math
import random
import os
import cv2
from array import *
from shapely.geometry import Point, Polygon
import numpy as np
import time
import requests
import json

id = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
obsazeno = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
now_time = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

obsazeno_frame = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
neobsazeno_frame = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# Délka obsazenosti parkovacího místa
cas = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
cas_last = 0
cas_real = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

url = "http://167.71.38.238:3080/api/data"

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)


def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}


def load_network(config_file, data_file, weights, batch_size=1):
    """
    load model description and weights from config files
    args:
        config_file (str): path to .cfg model file
        data_file (str): path to .data model file
        weights (str): path to weights
    returns:
        network: trained model
        class_names
        class_colors
    """
    network = load_net_custom(
        config_file.encode("ascii"),
        weights.encode("ascii"), 0, batch_size)
    metadata = load_meta(data_file.encode("ascii"))
    class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]
    colors = class_colors(class_names)
    return network, class_names, colors


def print_detections(detections, coordinates=False):
    print("\nObjects:")
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))


def pointInRect(x,y,x1, y1, w, h): # point(x,y)  rectangle(x1,y1, w, h)
    x2, y2 = x1+w, y1+h
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False


def draw_boxes(detections, image, colors):
    import cv2
    parking_spots = [
        [403,1327], [513,1365], [623,1399] # invalidi (3x)
        ,[949,1101], [1029,909], [1085,751], [1139,633], [1169,545]
        ,[1201,465], [1223,405], [1247,355], [1265,305] #podelné parkování (9x)
        ,[1403,759], [1413,677], [1417,605], [1425,547], [1427,489] 
        ,[1435,437], [1437,395], [1441,355], [1445,315], [1447,287] #první řada (10x)
        ,[1565,753], [1561,671], [1557,595], [1557,531], [1559,467] 
        ,[1551,423], [1553,383], [1551,339], [1549,315], [1549,277]] #druhá řada (10x)
    cout_of_parking_places = len(parking_spots)
    parking_spots_available = np.ones((cout_of_parking_places), dtype=bool) # = [True, True, True, True, True]


    for i in range(0,cout_of_parking_places):
        x = parking_spots[i][0]
        y = parking_spots[i][1]
        point = Point((x, y))
        cv2.circle(image, ((x,y)), 10, (0,255,255), 2) # yellow circle for every parking spot
    

    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        if label == ('car' or 'bicycle' or 'motorbike' or 'truck' or 'bus'):
            for i in range(0,cout_of_parking_places):
                x_ = parking_spots[i][0]
                y_ = parking_spots[i][1]
                l, t, w, h = bbox

                if(pointInRect(x_,y_, left, top, w, h) == True):
                    parking_spots_available[i] = False
                    #continue

        #cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        #cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
        #            (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #            colors[label], 2)
    for k in range(0,cout_of_parking_places):
        if parking_spots_available[k] == True:
            cv2.circle(image, ((parking_spots[k][0],parking_spots[k][1])), 10, (0,255,0), -1) # green circle if spot is not availavle
            neobsazeno_frame[k] = neobsazeno_frame[k] + 1
            print(str(k) + " Počet neobsazeno frame: " + str(neobsazeno_frame[k]))
            if neobsazeno_frame[k] >= 8:
                obsazeno_frame[k] = 0
                obsazeno[k] = False
                now_time[k] = 0
                cas[k] = 0
        else:
            cv2.circle(image, ((parking_spots[k][0],parking_spots[k][1])), 10, (0,0,255), -1) # red circle if spot is not availavle
            obsazeno_frame[k] = obsazeno_frame[k] + 1
            if obsazeno_frame[k] == 1:
                # now_time[k] = time.ctime()
                cas[k] = time.time()
            print(str(k) + " Počet obsazeno frame: " + str(obsazeno_frame[k]))
            neobsazeno_frame[k] = 0
            obsazeno[k] = True

    cas_last = time.time()
    x = 0
    for x in range(32):
        cas_real[x] = cas_last - cas[x]
        now_time[x] = int(cas_real[x])
        if obsazeno[x] == False:
            now_time[x] = 0

    JSON = [
    {
        "id": id[0],
        "obsazeno": obsazeno[0],
        "datum": now_time[0]
    },
    {
        "id": id[1],
        "obsazeno": obsazeno[1],
        "datum": now_time[1]
    },
    {
        "id": id[2],
        "obsazeno": obsazeno[2],
        "datum": now_time[2]
    },
    {
        "id": id[3],
        "obsazeno": obsazeno[3],
        "datum": now_time[3]
    },
    {
        "id": id[4],
        "obsazeno": obsazeno[4],
        "datum": now_time[4]
    },
    {
        "id": id[5],
        "obsazeno": obsazeno[5],
        "datum": now_time[5]
    },
    {
        "id": id[6],
        "obsazeno": obsazeno[6],
        "datum": now_time[6]
    },
    {
        "id": id[7],
        "obsazeno": obsazeno[7],
        "datum": now_time[7]
    },
    {
        "id": id[8],
        "obsazeno": obsazeno[8],
        "datum": now_time[8]
    },
    {
        "id": id[9],
        "obsazeno": obsazeno[9],
        "datum": now_time[9]
    },
    {
        "id": id[10],
        "obsazeno": obsazeno[10],
        "datum": now_time[10]
    },
    {
        "id": id[11],
        "obsazeno": obsazeno[11],
        "datum": now_time[11]
    },
    {
        "id": id[12],
        "obsazeno": obsazeno[12],
        "datum": now_time[12]
    },
    {
        "id": id[13],
        "obsazeno": obsazeno[13],
        "datum": now_time[13]
    },
    {
        "id": id[14],
        "obsazeno": obsazeno[14],
        "datum": now_time[14]
    },
    {
        "id": id[15],
        "obsazeno": obsazeno[15],
        "datum": now_time[15]
    },
    {
        "id": id[16],
        "obsazeno": obsazeno[16],
        "datum": now_time[16]
    },
    {
        "id": id[17],
        "obsazeno": obsazeno[17],
        "datum": now_time[17]
    },
    {
        "id": id[18],
        "obsazeno": obsazeno[18],
        "datum": now_time[18]
    },
    {
        "id": id[19],
        "obsazeno": obsazeno[19],
        "datum": now_time[19]
    },
    {
        "id": id[20],
        "obsazeno": obsazeno[20],
        "datum": now_time[20]
    },
    {
        "id": id[21],
        "obsazeno": obsazeno[21],
        "datum": now_time[21]
    },
    {
        "id": id[22],
        "obsazeno": obsazeno[22],
        "datum": now_time[22]
    },
    {
        "id": id[23],
        "obsazeno": obsazeno[23],
        "datum": now_time[23]
    },
    {
        "id": id[24],
        "obsazeno": obsazeno[24],
        "datum": now_time[24]
    },
    {
        "id": id[25],
        "obsazeno": obsazeno[25],
        "datum": now_time[25]
    },
    {
        "id": id[26],
        "obsazeno": obsazeno[26],
        "datum": now_time[26]
    },
    {
        "id": id[27],
        "obsazeno": obsazeno[27],
        "datum": now_time[27]
    },
    {
        "id": id[28],
        "obsazeno": obsazeno[28],
        "datum": now_time[28]
    },
    {
        "id": id[29],
        "obsazeno": obsazeno[29],
        "datum": now_time[29]
    },
    {
        "id": id[30],
        "obsazeno": obsazeno[30],
        "datum": now_time[30]
    },
    {
        "id": id[31],
        "obsazeno": obsazeno[31],
        "datum": now_time[31]
    }
    ]
    print("Data z detekce: " + str(JSON))
    print("________________________________________________________________________________________________________________________")
    data = json.dumps(JSON)
    response = requests.post(url, data=data, headers={"Content-Type": "application/json"})
    print("odpověď serveru/cloudu: " + str(response.json()))

    return image


def decode_detection(detections):
    decoded = []
    for label, confidence, bbox in detections:
        confidence = str(round(confidence * 100, 2))
        decoded.append((str(label), confidence, bbox))
    return decoded


def remove_negatives(detections, class_names, num):
    """
    Remove all classes with 0% confidence within the detection
    """
    predictions = []
    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[j].prob[idx], (bbox)))
    return predictions



def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
    """
        Returns a list with highest confidence class and their bbox
    """
    print("Probiha detekce")
    pnum = pointer(c_int(0))
    predict_image(network, image)
    detections = get_network_boxes(network, image.w, image.h,
                                   thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(detections, num, len(class_names), nms)
    predictions = remove_negatives(detections, class_names, num)
    predictions = decode_detection(predictions)
    free_detections(detections, num)
    return sorted(predictions, key=lambda x: x[1])


#  lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
# lib = CDLL("libdarknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt": # nt
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value {} not forcing CPU mode".format(tmp))
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError as cpu_error:
                print(cpu_error)
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find {}. Trying a GPU run anyway.".format(winNoGPUdll))
else:
    lib = CDLL(os.path.join(os.environ.get('DARKNET_PATH', './'),"libdarknet.so"), RTLD_GLOBAL)

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

free_network_ptr = lib.free_network_ptr
free_network_ptr.argtypes = [c_void_p]
free_network_ptr.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)
