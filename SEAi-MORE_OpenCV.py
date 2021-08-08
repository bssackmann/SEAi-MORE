# -*- coding: utf-8 -*-

# GSI ENVIRONMENTAL INC.
#
# SCRIPT NAME
# ==========
#      SEAi-MORE_OpenCV.py (Streamlit web app)
#
#      Job Name:          SEAi-MORE
#      Job Number:        9400-910-102
#
# PURPOSE 
# ==========
#      This Streamlit app lets users process time-lapse imagery of bivalves (e.g., oysters) 
#      to better understand their feeding behavior and how it changes through time. 
#
#      Developed as part of GSI's Phase 2 final submission for the OpenCV AI Competition 2021.
#
# AUTHOR(S)
# ==========          
#      Brandon S. Sackmann, Ph.D.
#      Emerson Sirk  
#
# HISTORY
# =======
#      Date        Remarks
#      ----------  -----------------------------------------------------------
#      20210806    Initial script development. (ES)
#      20210807    Script QA and cleanup. (BS)
# 
# Copyright (c) 2021 GSI Environmental Inc.
# All Rights Reserved
# E-mail: bssackmann@gsi-net.com
# $Revision: 1.0$ Created on: 2021/08/06

# import required packages
import streamlit as st
import requests
import base64
import io
import cv2
import glob
import json
import time
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.express as px
import pandas
import pickle

# add web app title
st.title('SEAi-MORE')
st.subheader('TEAM: The Oly Shuckers')

# add sidebar title
st.sidebar.title('SEAi-MORE')
st.sidebar.subheader('A Shellfish Behavior Monitoring Device')

# add user input directory path
dir = st.sidebar.text_input('Image Directory')

if dir == "": # no input directory
    st.sidebar.text("Specify Image Directory...")

files = glob.glob(dir + '/*.jpeg') # files is list of image (jpeg) files in directory
num_files = len(files) # store the total number of image files for future reference
st.sidebar.text("Image Files: " + str(num_files)) # print number of files

st.sidebar.markdown('---')

# add task selector
st.sidebar.markdown('***What Would You Like to Do?***')
app_mode = st.sidebar.selectbox('',
['Project Overview','Pre-Processing & Inference','Annotation Review','Post-Processing & Time Series Analysis']
)

################################################################################
#PROJECT OVERVIEW INFORMATION
################################################################################

if app_mode =='Project Overview':

    # add horizontal line to sidebar
    st.sidebar.markdown('---')

    #main display
    st.markdown('***Project Overview***')
    st.markdown('''This Python [**Streamlit**] (https://streamlit.io/) app was developed as 
                part of our team's Phase 2 final submission for the [**OpenCV AI Competition 2021**] 
                (https://opencv.org/opencv-ai-competition-2021/). The app lets users process 
                time-lapse imagery of bivalves (e.g., oysters) to better understand their feeding 
                behavior and how it changes through time. To learn more about the project check 
                out the video below and the [**SEAi-MORE GitHub repository**] (https://github.com/bssackmann/SEAi-MORE)!
                ''')

    st.video('https://www.youtube.com/watch?v=dutLudeih64')

    st.markdown('''
                # GSI's AI & CVPR Capabilities \n
            
                Harness the power of AI and Computer Vision and Pattern Recognition (CVPR) technology by working 
                with GSI’s team of data scientists and computer vision specialists. Our team has built custom solutions 
                for environmental applications using cost-effective, state-of-the-art approaches that yield 
                actionable information. Whether you have a specific project or use-case in mind, or want to brainstorm 
                about how these tools can help address your needs, just contact Dr. Brandon Sackmann via 
                <a href="mailto:bssackmann@gsi-net.com">email</a> or connect on 
                [LinkedIn] (https://www.linkedin.com/in/bsackmann/) today! \n
            
                Follow [GSI] (https://www.gsi-net.com/en/) on [YouTube] (https://www.youtube.com/channel/UCHfe41c67XLRS8fUZvmcxLw/featured) or 
                [LinkedIn] (https://www.linkedin.com/company/groundwater-services-inc/). \n
            
                <a href="https://www.youtube.com/channel/UCHfe41c67XLRS8fUZvmcxLw/featured" target="new"><img src="https://github.com/bssackmann/SEAi-MORE/raw/main/html_images/YouTube-Emblem.png" width="50"></a>
                <a href="https://www.linkedin.com/company/groundwater-services-inc/" target="new"><img src="https://github.com/bssackmann/SEAi-MORE/raw/main/html_images/Linkedin-Symbol.png" width="50"></a>
                <a href="https://github.com/GSIEnvironmental" target="new"><img src="https://github.com/bssackmann/SEAi-MORE/raw/main/html_images/GitHub-Mark.png" width="50"></a>
                ''',unsafe_allow_html=True)

################################################################################
#PRE-PROCESSING & INFERENCE INFORMATION
################################################################################

elif app_mode =='Pre-Processing & Inference':

    # add horizontal line to sidebar
    st.sidebar.markdown('---')
    
    # main display
    st.markdown('***Pre-Processing & Inference***')
    st.markdown('''
                To begin, paste the full path to a directory containing JPEG images into the sidebar on the left. \n

                A collection of sample images (<i>n</i> = 100) that can be used to evaluate the app can 
                be downloaded [here] (https://gsienvironmental-my.sharepoint.com/:f:/g/personal/bssackmann_gsi-net_com/EuO6V5Wa2OxNuGPf6-s4FA8B6TgjtU5BVDR6vWNfY3nYGQ?e=yuOLy). \n
                
                User-defined inputs include bounding box prediction **confidence**, **overlap**, and **shellfish**
                type. Once all selections are made, hit the run button in the sidebar on the left. \n

                The app sends each image in the specified directory to the Roboflow server hosted API endpoint.
                The API returns bounding box predictions for all detected shellfish in a standardized JSON-formatted response. \n

                JSON responses are saved as simple text files (.json) into the user-defined input directory using the
                same file name as the associated JPEG image.
                ''',unsafe_allow_html=True)

    # model confidence
    confidence = st.sidebar.slider('Confidence', min_value=0, max_value=100, value=40)
    st.sidebar.markdown('''
                        A threshold for the returned predictions on a scale of 0-100. A lower number will return more predictions. A higher number will return fewer high-certainty predictions (Default = 40).
                        ''')

    st.sidebar.markdown('---')

    # overlap setting of model
    overlap = st.sidebar.slider('Overlap', min_value=0, max_value=100, value=30)
    st.sidebar.markdown('''
                        The maximum percentage (on a scale of 0-100) that bounding box predictions of the same class 
                        are allowed to overlap before being combined into a single box (Default = 30).
                        ''')

    st.sidebar.markdown('---')

    # classes to detect in the model
    classes = st.sidebar.selectbox('Choose Your Shellfish', ['Oyster', 'Mussel'])
    st.sidebar.markdown('Select which shellfish you would like the model to identify.')
    st.sidebar.markdown('[Oysters](https://en.wikipedia.org/wiki/Oyster)')
    st.sidebar.markdown('[Mussels](https://en.wikipedia.org/wiki/Mussel)')

    st.sidebar.markdown('---')

    run = st.sidebar.button("Run") # run button when all user inputs are set

    st.sidebar.markdown('''
                        For more information about Roboflow's server hosted API,
                        click [here] (https://docs.roboflow.com/inference/hosted-api).
                        ''')

    st.markdown('---')

    # initialize processing bar
    infProgTxt = st.text("Processing File 0 of " + str(num_files) + "...")
    infProgBar = st.progress(0)

    if run == True:

        index = 1 # index of files list

        for filename in files:

            # replace the processing bar with an updated version
            infProgTxt.text("Processing File " + str(index) + " of " + str(num_files) + "...")
            infProgBar.progress(index/num_files)

            if filename is not None:
                # load image with PIL
                image = Image.open(filename).convert("RGB")
                current_dims = image.size

                # convert to JPEG buffer
                buffered = io.BytesIO()
                image.save(buffered, quality=90, format="JPEG")

                # base 64 encode
                img_str = base64.b64encode(buffered.getvalue())
                img_str = img_str.decode("ascii")

                # construct the URL
                upload_url = "".join([
                    "https://detect.roboflow.com/seai-more_v2-v3/1",
                    "?api_key=SEbFyx4JkiQNHAwl6LEK",
                    "&confidence=",str(confidence),
                    "&overlap=",str(overlap),
                    "&classes=",str(classes)
                    ])

                # POST to the API
                r = requests.post(upload_url, data=img_str, headers={
                "Content-Type": "application/x-www-form-urlencoded"
                })

                # output result
                preds = r.json()
                detections = preds['predictions']

                # save JSON to the file directory
                with open(filename.replace('jpeg','json'),'w') as f:
                    json.dump(detections,f)

                index = index + 1 # add one to the index variable

        st.text("Images Processed Successfully!")

################################################################################
#Annotation Review
################################################################################

elif app_mode =='Annotation Review':

    # add horizontal line to sidebar
    st.sidebar.markdown('---')
    
    # main display
    st.markdown('***Annotation Review***')
    st.markdown('''
            Once bounding box predictions have been made, results can be reviewed and displayed below. \n

            The **select image** input box below lets users display bounding box predictions for specific 
            JPEG images found in the user-specified input directory. \n

            Users can also change the thickness of the lines used to draw bounding boxes by adjusting the **box stroke width** value. \n
            
            Labels can be turned on/off by toggling the **display labels** checkbox and the JSON-formatted predictions can 
            be viewed by checking the **display JSON** checkbox found below the image.
            ''')

    # image reference user selection
    image_select = int(st.number_input('Select Image', min_value=1, max_value=num_files, value=1))

    # thickness of box labels
    stroke = int(st.number_input('Box Stroke Width', min_value=1, max_value=20, value=10))
    
    # choose if labels should be present in image
    labels = st.checkbox("Display Labels", 1)

    # load user selected image from directory path
    image_file = files[image_select-1]
    st.text("Current Image File Name:")
    st.text(str(files[image_select-1]))

    if image_file is None:
        st.text("No Image Selected")

    else:
        image_disp = Image.open(image_file).convert("RGB")
        current_dims = image_disp.size

        # load relevant JSON file
        json_filename = str(image_file.replace('jpeg','json')) #string replacement
        json_open = open(json_filename)
        json_disp = json.load(json_open)

        # annotate image and display
        draw = ImageDraw.Draw(image_disp)
        font = ImageFont.truetype("arial.ttf", 60)

        for box in json_disp:
            color = "#4892EA"
            x1 = int(box['x'] - box['width'] / 2)
            x2 = int(box['x'] + box['width'] / 2)
            y1 = int(box['y'] - box['height'] / 2)
            y2 = int(box['y'] + box['height'] / 2)

            draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke)

            if labels == True:
                text = box['class']
                text_size = font.getsize(text)
                # set button size + 10px margins
                button_size = (int((text_size[0]+35)), int((text_size[1]+35)))
                button_img = Image.new('RGBA', button_size, color)
                # put text on button with 10px margins
                button_draw = ImageDraw.Draw(button_img)
                button_draw.text((10, 10), text, font=font, fill=(255,255,255,255))

                # put button on source image in position (0, 0)
                image_disp.paste(button_img, (int(x1), int(y1)))

        # display annotated image
        st.image(image_disp)
        current_dims = image_disp.size
        st.text("Image Dimensions:" + str(current_dims))

        # choose if JSON file should appear below image
        json_present = st.checkbox("Display JSON")

        # display JSON file
        if json_present == True:
            st.text("JSON File")
            st.write(json_disp)

################################################################################
#POST-PROCESSING & DATA VISUALIZATION INFORMATION
################################################################################

elif app_mode =='Post-Processing & Time Series Analysis':

    # main display
    st.markdown('***Post-processing & Time Series Analysis***')

    st.markdown('''
                Once bounding box predictions have been made, results can be compiled, and additional post-processing 
                can be done on an animal-specific basis. \n
                
                Currently, the SEAi-MORE app only calculates a select subset of derived **metrics**.  For example, the size of the
                bounding boxes predicted for individual animals can be viewed as interactive time series plots and the variability
                that one sees in these plots relates, in part, to the animals opening and closing their shells at different times. \n    

                As the SEAi-MORE project matures we will be introducing additional functionality to the app and expanding the suite of 
                derived metrics that we calculate. *Stay tuned!*
                ''')

    json_files = glob.glob(dir + '/*.json') # files is list of image (jpeg) files in directory
    num_json_files = len(json_files) # store the total number of image files for future reference
    st.sidebar.text("JSON Annotation Files: " + str(num_json_files)) # print number of files

    # add horizontal line to sidebar
    st.sidebar.markdown('---')
    
    live_updates = st.checkbox('Live Updates', 1)
    
    st.markdown('*Turn off live updates to improve app performance*')

    if live_updates == 1:

        progBarTxt = st.text('Processing Images...')
        counter = 0
        progBar = st.progress(counter)

        # determine centroids
        json_open = open(json_files[0])
        annotation = json.load(json_open)
        num_shellfish = len(annotation)

        centroids_dict = {}
        count = 0

        for i in range(num_shellfish):
            animal = annotation[count]
            center_x = int(animal['x'] + (animal['width'] / 2))
            center_y = int(animal['y'] + (animal['height'] / 2))

            name = 'Animal ' + str(i + 1)

            centroids_dict[name] = {'x' : center_x, 'y' : center_y}

            count = count + 1

        # initialize summary annotation files
        nan_list = [np.nan] * num_json_files
        x_summary = {}
        y_summary = {}
        width_summary = {}
        height_summary = {}
        area_summary = {}
        conf_summary = {}

        for x in range(num_shellfish):
            key = 'Animal ' + str(x + 1)
            x_summary[key] = [np.nan] * num_json_files # nan_list
            y_summary[key] = [np.nan] * num_json_files # nan_list
            width_summary[key] = [np.nan] * num_json_files # nan_list
            height_summary[key] = [np.nan] * num_json_files # nan_list
            area_summary[key] = [np.nan] * num_json_files # nan_list
            conf_summary[key] = [np.nan] * num_json_files # nan_list

        file_index = 0

        for f in json_files: # cycle through all the stored annotations
            json_open = open(json_files[file_index]) # open file
            annot = json.load(json_open) # variable for one file dict
            tot_shells = len(annot)

            animal_index = 0
            for j in range(tot_shells): # cycle through all the animals in a given annotation

                # determine which animal in the image it is
                animal_file_stats = annot[animal_index] # grab current animal in  a given file
                x_value = animal_file_stats['x']
                y_value = animal_file_stats['y']
                width_value = animal_file_stats['width']
                height_value = animal_file_stats['height']

                x_upper = x_value + width_value
                y_upper = y_value + height_value

                successful_match = 0

                for w in range(tot_shells): # cycle though all animals to find a match

                    animal_test = 'Animal ' + str(w + 1)
                    if centroids_dict[animal_test]['x'] > x_value:
                         if centroids_dict[animal_test]['x'] < x_upper:
                             if centroids_dict[animal_test]['y'] > y_value:
                                if centroids_dict[animal_test]['y'] < y_upper:

                                    animal_match = animal_test
                                    successful_match = 1

                if successful_match == 1:
                    animal_file_stats['Name'] = animal_match

                    box_area = int(animal_file_stats['width']) * int(animal_file_stats['height'])

                    # store stats in summary dict
                    if np.isnan(x_summary[animal_match][file_index]) == 1:
                        x_summary[animal_match][file_index] = animal_file_stats['x']

                    if np.isnan(y_summary[animal_match][file_index]) == 1:
                        y_summary[animal_match][file_index] = animal_file_stats['y']

                    if np.isnan(width_summary[animal_match][file_index]) == 1:
                        width_summary[animal_match][file_index] = animal_file_stats['width']

                    if np.isnan(height_summary[animal_match][file_index]) == 1:
                        height_summary[animal_match][file_index] = animal_file_stats['height']

                    if np.isnan(area_summary[animal_match][file_index]) == 1:
                        area_summary[animal_match][file_index] = box_area

                    if np.isnan(conf_summary[animal_match][file_index]) == 1:
                        conf_summary[animal_match][file_index] = animal_file_stats['confidence']

                counter = counter + 1

                iterations = tot_shells * num_files
                progBar.progress(counter/iterations)

                animal_index = animal_index + 1 # move to next animal in the file

            file_index = file_index + 1

        progBar.progress(iterations/iterations)

        # saving the objects:
        with open(dir + '\summary_stats.pkl', 'wb') as f:
            pickle.dump([x_summary,y_summary,width_summary,height_summary,area_summary,conf_summary], f)

    else:
        # load files from disk
        with open(dir + '\summary_stats.pkl', 'rb') as f:
            x_summary, y_summary, width_summary, height_summary, area_summary, conf_summary = pickle.load(f)

    # draw a box around the animals
    image_select = int(st.number_input('Select Image', min_value=1, max_value=num_files, value=1))

    image_select = image_select - 1
    # load user selected image from directory path
    image_file = files[image_select]
    st.text("Current Image File Name:")
    st.text(str(files[image_select]))

    color_style = st.selectbox('Image Style', ['RGB','HSV','Binary (Simple Threshold)'])
    
    st.markdown('*Change image color space (placeholder for future app upgrades) *')

    if color_style == 'HSV':
        image1 = cv2.imread(image_file)
        image_disp = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

        st.image(image_disp)
        current_dims = image_disp.size
        st.text("Image Dimensions:" + str(current_dims))

    if color_style == 'Binary (Simple Threshold)':
        image2 = cv2.imread(image_file,0)
        # image threshold: 1)should be grayscale 2) 127 is value 3) assign 255
        ret, image_disp = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)

        st.image(image_disp)
        current_dims = image_disp.size
        st.text("Image Dimensions:" + str(current_dims))

    if color_style == 'RGB':
        image_disp = Image.open(image_file).convert("RGB")

        annotations_on = st.checkbox('Display Annotations',1)

        if annotations_on == True:
            stroke = int(st.number_input('Box Stroke Width', min_value=1, max_value=20, value=10))

            # annotate image and display
            draw = ImageDraw.Draw(image_disp)
            font = ImageFont.truetype("arial.ttf", 60)

            json_disp = x_summary.keys()
            for box in json_disp:

                if np.isnan(x_summary[box][image_select]) == False:

                    color = "#4892EA"
                    x1 = int(x_summary[box][image_select] - width_summary[box][image_select] / 2)
                    x2 = int(x_summary[box][image_select] + width_summary[box][image_select] / 2)
                    y1 = int(y_summary[box][image_select] - height_summary[box][image_select] / 2)
                    y2 = int(y_summary[box][image_select] + height_summary[box][image_select] / 2)

                    draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke)

                    text = box[7:]
                    text_size = font.getsize(text)
                    # set button size + 10px margins
                    button_size = (int(text_size[0]+35), int(text_size[1])+35)
                    button_img = Image.new('RGBA', button_size, color)
                    # put text on button with 10px margins
                    button_draw = ImageDraw.Draw(button_img)
                    button_draw.text((20, 20), text, font=font, fill=(255,255,255,255))

                    image_disp.paste(button_img, (int((x2 - x1)/3 + x1), int((y2 - y1)/4 + y1)))

        # display annotated image
        st.image(image_disp)
        current_dims = image_disp.size
        st.text("Image Dimensions:" + str(current_dims))

        # plot the summary Statistics
        metric = st.selectbox('Metric', ['Box Area','X','Y','Width','Height','Confidence'])

        st.markdown('*Double click on legend item to isolate a specific animal*')

        if metric == 'X':
            dict_select = x_summary
            unit = '(Pixel Value)'
        if metric == 'Y':
            dict_select = y_summary
            unit = '(Pixel Value)'
        if metric == 'Width':
            dict_select = width_summary
            unit = '(Pixels)'
        if metric == 'Height':
            dict_select = height_summary
            unit = '(Pixels)'
        if metric == 'Box Area':
            dict_select = area_summary
            unit = '(Pixel Count)'
        if metric == 'Confidence':
            dict_select = conf_summary
            unit = '(Probability)'

        df = pandas.DataFrame.from_dict(dict_select)
        df.index = range(1,len(df)+1) # align the index with the file numbers

        title = str(metric) + ' Over Time'
        fig = px.line(data_frame = df,  labels={
                         "index": "Image Index",
                         "value": str(metric) + ' ' + unit,
                         "variable": "Shellfish"})

        fig.update_layout(title={'text': title,'y':0.9, 'x':0.5, 'xanchor': 'center','yanchor': 'top'})

        st.write(fig)

################################################################################

# add GSI logo
st.sidebar.markdown('''<center><a href="https://www.gsi-net.com/en/" target="new">
                    <img src="https://github.com/bssackmann/SEAi-MORE/raw/main/html_images/GSI%20Logo%20-%20Transparent_v2.png" width="200"></a><br>
                    <a href="http://pacshell.org/" target="new">
                    <img src="https://github.com/bssackmann/SEAi-MORE/raw/main/html_images/pacshell-logo.png" width="200"></a></center>
                    ''',unsafe_allow_html=True)

################################################################################
