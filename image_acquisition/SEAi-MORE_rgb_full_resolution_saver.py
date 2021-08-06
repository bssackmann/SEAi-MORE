#!/usr/bin/env python3

# GSI Environmental 
#
# Script name:       SEAi-MORE_rgb_full_resolution_saver.py
# 
# Purpose of script: Collect timelapse imagery from OAK-D camera at user-defined interval (VARIABLE FOCUS). 
#
# Author:            B. Sackmann  
#
# Date Created:      2021-05-05
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

parser = argparse.ArgumentParser(prog='SEAi-MORE_rgb_full_resolution_saver.py',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description='''\
SEAi-MORE RGB FULL-RESOLUTION SAVER - VARIABLE FOCUS
                                 ''',
                                 epilog='''\
ARGUMENT DESCRIPTIONS:
==========
i -or- interval:    Interval between images (seconds; default = 1).

REFERENCES:
=========

AUTHOR(S)
=========
     Brandon S. Sackmann, Ph.D.

HISTORY
=======
     Date        Remarks
     ----------  -----------------------------------------------------------
     20210505    Initial script development. (BS)

Copyright (c) 2021 GSI Environmental Inc.
All Rights Reserved
E-mail: bssackmann@gsi-net.com
$Revision: 1.0$ Created on: 2021/05/05
                                 ''')
parser.add_argument('-i', '--interval', type=int, default=1, action='store', help='Interval between images (seconds).')
args = parser.parse_args()

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


# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=30, blocking=False)
    qJpeg = device.getOutputQueue(name="jpeg", maxSize=30, blocking=True)

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
                
        if cv2.waitKey(1) == ord('q'):
            break
