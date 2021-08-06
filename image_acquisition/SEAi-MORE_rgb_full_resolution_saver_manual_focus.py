#!/usr/bin/env python3

# GSI Environmental 
#
# Script name:       SEAi-MORE_rgb_full_resolution_saver_manual_focus.py
# 
# Purpose of script: Collect timelapse imagery from OAK-D camera at user-defined interval (MANUAL FOCUS). 
#
# Author:            B. Sackmann  
#
# Date Created:      2021-07-12
#
# Job Name:          SEAi-MORE
# Job Number:        9400-910-102
# 
# Notes: 
#   
# Data Source: 

import argparse
import time
from pathlib import Path
import cv2
import depthai as dai

parser = argparse.ArgumentParser(prog='SEAi-MORE_rgb_full_resolution_saver_manual_focus.py',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description='''\
SEAi-MORE RGB FULL-RESOLUTION SAVER - MANUAL FOCUS
                                 ''',
                                 epilog='''\
ARGUMENT DESCRIPTIONS:
==========
i -or- interval:    Interval between images (seconds; default = 1).

FOCUS CONTROLS:
==========
    * 'T' autofocus -- trigger (static)
    * 'F' autofocus -- continuous
    Manual focus:
        Control:    key[dec/inc]    min..max
        focus:           ,   .        0..255 [far..near]

AUTHOR(S)
=========
     Brandon S. Sackmann, Ph.D.

HISTORY
=======
     Date        Remarks
     ----------  -----------------------------------------------------------
     20210712    Initial script development. (BS)

Copyright (c) 2021 GSI Environmental Inc.
All Rights Reserved
E-mail: bssackmann@gsi-net.com
$Revision: 1.0$ Created on: 2021/07/12
                                 ''')
parser.add_argument('-i', '--interval', type=int, default=1, action='store', help='Interval between images (seconds).')
args = parser.parse_args()

# Manual focus setup
LENS_STEP = 3

def clamp(num, v0, v1):
    return max(v0, min(num, v1))

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(768, 432)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)

# Create RGB output
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

# Create encoder to produce JPEG images
videoEnc = pipeline.createVideoEncoder()
videoEnc.setDefaultProfilePreset(camRgb.getVideoSize(), camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
camRgb.video.link(videoEnc.input)

# Create JPEG output
xoutJpeg = pipeline.createXLinkOut()
xoutJpeg.setStreamName("jpeg")
videoEnc.bitstream.link(xoutJpeg.input)

# Create control input
controlIn = pipeline.createXLinkIn()
controlIn.setStreamName("control")
controlIn.out.link(camRgb.inputControl)

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Input control queue
    qControl = device.getInputQueue("control")

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=30, blocking=False)
    qJpeg = device.getOutputQueue(name="jpeg", maxSize=30, blocking=True)

    # Defaults and limits for manual focus controls
    lensPos = 150
    lensMin = 0
    lensMax = 255

    # Make sure the destination path is present before starting to store the examples
    Path('SEAi-MORE').mkdir(parents=True, exist_ok=True)

    count = 0
    while True:
        inRgb = qRgb.tryGet()  # Non-blocking call, will return a new data that has arrived or None otherwise

        if inRgb is not None:
            cv2.imshow("rgb", inRgb.getCvFrame())

        for encFrame in qJpeg.tryGetAll():
            if count == args.interval*camRgb.getFps():
                with open(f"SEAi-MORE/{int(time.time() * 10000)}.jpeg", "wb") as f:
                    f.write(bytearray(encFrame.getData()))
                    count = 0
            else:
                count = count + 1
        
        key = cv2.waitKey(1)
        if key == ord("q"):       
            break
        elif key == ord("t"):
            print("Autofocus trigger (and disable continuous)")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
            qControl.send(ctrl)
        elif key == ord("f"):
            print("Autofocus enable, continuous")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            qControl.send(ctrl)
        elif key in [ord(","), ord(".")]:
            if key == ord(","): lensPos -= LENS_STEP
            if key == ord("."): lensPos += LENS_STEP
            lensPos = clamp(lensPos, lensMin, lensMax)
            print("Setting manual focus,lens position: ", lensPos)
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(lensPos)
            qControl.send(ctrl)
