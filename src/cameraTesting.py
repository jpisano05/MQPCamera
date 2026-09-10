import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
import pandas as pd
import time
from IPython import display
import PIL
import glob
import imageio
import os
import matplotlib as mpl
import PIL.Image
import functools
import cv2
import threading

#get model
model = YOLO("wasteDetectorLatest.pt")

#make pipeline
pipeline = rs.pipeline()

#setup stream config
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640, 480,
    rs.format.bgr8,
    30
)

config.enable_stream(
    rs.stream.depth,
    640, 480,
    rs.format.z16,
    30
)

#turn on the camera
pipeline.start(config)

#variables for storing found objects
detectedIds = []
trackedObjects = {}
maxMissingFrames = 15

#threading
trackedObjectsLock = threading.Lock()
stopEvent = threading.Event()

#arm coordinate for coordinate system conversion
#place holder 0, 0, 0 for now
armX, armY, armZ = 0, 0, 0

#Function to run the main camera loop, records found objects into the trackedObjects var
def cameraLoop():
    while not stopEvent.is_set():
        #reset detectedIds for this frames
        detectedIds = []
        
        #wait for the next set of frames
        frames = pipeline.wait_for_frames()

        #extract individual frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        #convert to NumPy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        print("Color:", color_image.shape, color_image.dtype)
        print("Depth:", depth_image.shape, depth_image.dtype)
                
        #run inference
        results = model.track(
            source=color_image,
            persist=True,
            #device = "cpu"
        )
        
        #Process detections
        for result in results:
            for box in result.boxes:
                if box.id is None:
                    continue
                
                trackId = int(box.id)
                cls = int(box.cls)
                confidence = float(box.conf.item())
                
                detectedIds.append(trackId)

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                #get the center of detected object
                centerX = (x1 + x2) // 2
                centerY = (y1 + y2) // 2
                
                #get distance at center
                distance = depth_frame.get_distance(centerX, centerY)
                            
                #get the distance of the object from the center of the camera frame
                intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                cameraCoordinate = rs.rs2_deproject_pixel_to_point(intrinsics, [centerX, centerY], distance)
                            
                #convert to global coordinate (relative to arm)
                globalCoordinate = [cameraCoordinate[2] - armX, cameraCoordinate[1] - armY, cameraCoordinate[0] - armZ]
                
                with trackedObjectsLock:
                    #add to tracking if new, or update location if not new
                    trackedObjects[trackId] = {
                        "box": (x1, y1, x2, y2),
                        "class": cls,
                        "confidence": confidence,
                        "center": (centerX, centerY),
                        "coordinate": (globalCoordinate[0], globalCoordinate[1], globalCoordinate[2]),
                        "missingFrames": 0
                    }

        with trackedObjectsLock:
            for track_id in list(trackedObjects.keys()):

                if track_id not in detectedIds:
                        trackedObjects[track_id]["missingFrames"] += 1

                        #remove object if it has been missing too long
                        if trackedObjects[track_id]["missingFrames"] > maxMissingFrames:
                            del trackedObjects[track_id]
        
        #annotated image
        annotated = color_image.copy()
        
        #draw boxes
        #get a copy first so that the lock isnt held for too long
        with trackedObjectsLock:
            objectsToDraw = trackedObjects.copy()
            
        for track_id, obj in objectsToDraw.items():

            x1, y1, x2, y2 = obj["box"]
            cls = obj["class"]
            confidence = obj["confidence"]
            missing = obj["missingFrames"]
            globalCoordinate = obj["coordinate"]

            #get class name
            class_name = model.names[cls]

            #different color if YOLO did not detect it this frame
            if missing == 0:
                #green if found
                color = (0, 255, 0)
            else:
                #yellow if remembered
                color = (0, 255, 255)

            #draw bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            #label
            label = f"{class_name} ID:{track_id}"

            if missing > 0:
                label += f" ({missing})"

            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
            #draw center
            centerX = (x1 + x2) // 2
            centerY = (y1 + y2) // 2
            
            coordinateLabel = "Coord: (" + str(round(globalCoordinate[0], 3)) + ", " + str(round(globalCoordinate[1], 3)) + ", " + str(round(globalCoordinate[2], 3)) +")"
            
            #label the coordinate of the object
            cv2.putText(
                annotated,
                coordinateLabel,
                (x1, max(y2 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
            cv2.circle(
                annotated,
                (centerX, centerY),
                4,
                color,
                -1
            )

        #display color
        cv2.imshow("Color", color_image)

        #display depth
        depth_display = cv2.convertScaleAbs(
            depth_image,
            alpha=0.03
        )
        cv2.imshow("Depth", depth_display)
        
        #display annotated
        cv2.imshow("annotated: ", annotated)

        #press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stopEvent.set()
            break


#function to retrieve a copy of trackedObjects safely outside of the camera thread
def getTrackedObjects():
    with trackedObjectsLock:
        return trackedObjects.copy()

#main function to begin the camera thread and then to handle later retrieval of tracked objects
def main():
    trackingThread = threading.Thread(
        target = cameraLoop,
        daemon = True
    )
    
    trackingThread.start()
    
    try:
        while not stopEvent.is_set():
            #for getting the current state of tracked objects call it here
            #currently just gets it every 1 second but this block can be changed as needed
            objects = getTrackedObjects()
            
            print(objects)
            
            time.sleep(1)
    
    #when done close thread, stop pipeline, close cv2
    finally:
        stopEvent.set()
        trackingThread.join()
        
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()