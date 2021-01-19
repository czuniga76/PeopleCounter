# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 21:37:18 2021

@author: Christian Zuniga
    
"""
#chris
import cv2 as cv2
import os
from openvino.inference_engine import IECore, IENetwork
import numpy as np
import matplotlib.pyplot as plt
import time

def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    #print(" input shape ",input_image.shape)
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)
    
    return image

def draw_boxes(frame, result, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    coords = []
    m = 0
    s = 0
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        
        if conf >= 0.5:
            #print(" box detection ", conf)
            #print(" class ", box[1])
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            xc = xmin + (xmax-xmin)/2
            yc = ymin + (ymax-ymin)/2
           # print("center ",xc,yc, " xmin, xmax  ", xmin,xmax)
           # print(xmin,ymin,xmax,ymax)
            coords.extend([xmin,ymin,xmax,ymax])
            rectFrame = frame[ymin:ymax,xmin:xmax,:]
            m = np.mean(rectFrame)/255.0
            s =  np.std(rectFrame)/255.0
            #print("frame dim ", frame.shape , " box dim ",rectFrame.shape)
            #print(" mean and std val ",m,s)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    #print("coords from frames ",coords)
    return frame,m,s,coords

def analyzeFrame(frame,result):
    
    #boxes = result[0][0]
    numPeople = 0
    for box in result[0][0]:
        conf = box[2]
        cat = box[1]
        if cat == 1:
            #print("possible person ", conf)
            if conf >= 0.5:
                numPeople += 1
    #print(" number people ", numPeople)
    return numPeople
    
            

def async_inference(exec_net, input_blob,output_blob, image):
    ### TODO: Add code to perform asynchronous inference
    ### Note: Return the exec_net
    exec_net.start_async(request_id=0, inputs={input_blob: image})
    t1 = time.time()
    while True:
        status = exec_net.requests[0].wait(-1)
        if status == 0:
            res = exec_net.requests[0].outputs[output_blob]
            t2 = time.time() - t1
            break
        else:
            time.sleep(1)
    return res, t2



def sync_inference(exec_net, input_blob, image):
    ### TODO: Add code to perform synchronous inference
    ### Note: Return the result of inference
    res = exec_net.infer({input_blob: image})
   
    return res

def perform_inference(exec_net, request_type, input_image, input_shape):
    '''
    Performs inference on an input image, given an ExecutableNetwork
    '''
    # Get input image
    image = cv2.imread(input_image)
    width,height,_ = image.shape
    print("image shape ",width,height)
    #plt.imshow(image)
    ## Extract the input shape
    n, c, h, w = input_shape
    # Preprocess it (applies for the IRs from the Pre-Trained Models lesson)
    preprocessed_image = preprocessing(image, h,w)
    print("preprocessed shape ", preprocessed_image.shape)
    # Get the input blob for the inference request
    input_blob = next(iter(exec_net.inputs))
    output_blob = next(iter(exec_net.outputs))
    # Perform either synchronous or asynchronous inference
    request_type = request_type.lower()
    if request_type == 'a':
        detectionOutput = async_inference(exec_net, input_blob, output_blob,preprocessed_image)
    elif request_type == 's':
        t0 = time.time()
        output = sync_inference(exec_net, input_blob, preprocessed_image)
        t1 = time.time() - t0
        print("inf time ", t1)
        # Return the output for testing purposes
        detectionOutput = output["DetectionOutput"]
    else:
        print("Unknown inference request type, should be 'A' or 'S'.")
        exit(1)

    
    
    frame = draw_boxes(image, detectionOutput,   height,width)
    #cv2.rectangle(image,,2)
    cv2.imshow("Detected", frame)
    return detectionOutput


def infer_on_video(exec_net, videoPath, input_shape):

    cap = cv2.VideoCapture(videoPath)
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    n, c, h, w = input_shape
    
    input_blob = next(iter(exec_net.inputs))
    output_blob = next(iter(exec_net.outputs))
    totalPeople = 0
    duration = 0
    #numInFrame = 0
    firstDetection = True
    gapDetected = False
    gapTime = 0
    gapCounter = 0
    xmin = -1
    xmax = width+1
    ymin = -1
    ymax = height+1
    dList = []
    numMissedFrames = 0
    while(cap.isOpened()):
    
        # Capture frame-by-frame
        
        ret, frame = cap.read()    
        
        if ret == True:
            # Preprocess it (applies for the IRs from the Pre-Trained Models lesson)
            preprocessed_image = preprocessing(frame, h,w)
            detectionOutput,detectionTime = async_inference(exec_net, input_blob, output_blob,preprocessed_image)
            #frame = draw_boxes(frame, detectionOutput,   height,width)
           # print("detection time ",detectionTime)
            dList.append(detectionTime)
            frame,meanBox,stdBox,coords = draw_boxes(frame, detectionOutput,   width,height)
            p = analyzeFrame(frame,detectionOutput)
            if p > 0:
                if firstDetection == True:
                    totalPeople += 1
                    # TODO. Compare to oldMean, OldStd
                    print("New person detected, total ", totalPeople)
                    firstDetection = False
                    oldMean = meanBox
                    oldStd = stdBox
                    startTime = time.time()
                else:
                    if gapDetected == True:
                        print(" possible gap ", gapCounter)
                        print(" meand, std " ,meanBox, stdBox)
                        v1 = np.array([meanBox,stdBox])
                        v2 = np.array([oldMean,oldStd])
                        diff2 = np.linalg.norm(v1-v2)
                        
                        #print(v1)
                        #print(v2)
                        #print(" diff2 ", diff2)
                        if diff2 > 0.1 and gapCounter > 15:
                            print("new person detected ")
                            totalPeople += 1
                            print()
                            print("Total number ", totalPeople)
                            firstDetection = False
                            oldMean = meanBox
                            oldStd = stdBox
                            duration = time.time() - startTime
                            print("duration ",duration)
                            startTime = time.time()
                        else:
                            print("same person")
                        gapDetected = False
                        gapCounter = 0
                    else:
                        oldMean = meanBox
                        oldStd = stdBox
                        xmin = coords[0]
                        ymin = coords[1]
                        xmax = coords[2]
                        ymax = coords[3]
                                           
            else: 
                #print("no person detected but may be a miss")
                if firstDetection == False:
                    if gapDetected == False: # possible start of gap or person left
                        gapTime = time.time()
                        gapDetected = True
                        gapCounter = 1
                        numMissedFrames +=1
                    else:
                        gapCounter += 1
                        numMissedFrames +=1
                        rectFrame = frame[ymin:ymax,xmin:xmax,:]
                        m = np.mean(rectFrame)/255.0
                        s =  np.std(rectFrame)/255.0
                        v1 = np.array([m,s])
                        v2 = np.array([oldMean,oldStd])
                        diff2 = np.linalg.norm(v1-v2)
                       # print(" person may have left diff2 ",diff2)
                        # compare oldMean, oldStd with values in current frame, same box as old
                        if diff2 > 0.1 and gapCounter > 15:
                            firstDetection = True
                            gapDetected = False
                            duration = time.time() - startTime
                            print(" person left ")
                            print("duration ",duration)
                        # else:
                        #     # same person
                        #     print("Same person or person left")
                # else:
                #     print(" no person in view")
                        
                    
                    
               
                
                           
            
            cv2.imshow('Frame',frame)
 
            # Press Q on keyboard to  exit
    
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()

    print(" Total number of people ", totalPeople)
    avgDet = np.mean(dList)
    print(" Average detection Time ",avgDet)
    print(" missed frames ",numMissedFrames)
    
    
ie = IECore()
direc = "C:\\Scripts\\Python\\TF_Models\\ssdlite_mobilenet_v2_coco_2018_05_09\\"
#direc = "C:\\Scripts\\Python\\TF_Models\\ssd_mobilenet_v2_coco_2018_03_29\\"
# TODONeed to check conversion for faster_rcnn models
#direc = "C:\\Scripts\\Python\\TF_Models\\faster_rcnn_resnet50_coco_2018_01_28\\"
binfile = "frozen_inference_graph.bin"
xmlfile = "frozen_inference_graph.xml"
binPath = direc + binfile
xmlPath = direc + xmlfile

net = IENetwork(model=xmlPath, weights=binPath)
exec_net = ie.load_network(net,"CPU")

supported_layers = ie.query_network(network=net, device_name="CPU")

    # Check for any unsupported layers, and let the user
    # know if anything is missing. Exit the program, if so.
# unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
# if len(unsupported_layers) != 0:
#     print("Unsupported layers found: {}".format(unsupported_layers))
#     print("Check whether extensions are available to add to IECore.")
#     exit(1)
print("IR successfully loaded into Inference Engine.")

 # Get the input layer
input_blob = next(iter(net.inputs))
input_blob = next(iter(net.inputs))
# Get the input shape
input_shape = net.inputs[input_blob].shape
print(" network input shape ",input_shape)
#imageFile = "C:\\Scripts\\Python\\TF_Models\\blue-car.jpg"
#imageFile = "C:\\Scripts\\Python\\TF_Models\\person.jpg"
imageFile = "C:\\Scripts\\Python\\TF_Models\\London.jpg"
#out  = perform_inference(exec_net, "a", imageFile, input_shape)


direc = "C:\\Scripts\\Python\\TF_Models\\"
videoFile = "test_video.mp4"
videoFile2 = "Pedestrian_Detect_2_1_1.mp4"
videoPath = direc + videoFile2

infer_on_video(exec_net, videoPath, input_shape)
print()



