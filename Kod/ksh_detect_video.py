# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 12:54:40 2021

@author: hamza
"""

#%% 1. Bölüm

import cv2
import numpy as np
import os 

os.environ['KMP_DUPLICATE_LIB_OK']='True'


cap = cv2.VideoCapture("D:/01_Programlama/Yolo-Tez/3.mp4")
#cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    frame = cv2.resize(frame,(640,480))

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    
    #%% 2. Bölüm

    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True, crop=False)

    labels = ["Kalem","Silgi","Hesap Makinesi"]

    colors = ["0,255,255","0,0,255","255,0,0","255,255,0","0,255,0"]
    
    #colors dizisi içindeki değerleri virgüllerden ayırıp int e çeviriyoruz
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    
    #diziyi çoğaltma
    colors = np.tile(colors,(18,1))

    #%% 3. Bölüm

    model = cv2.dnn.readNetFromDarknet("D:/01_Programlama/Yolo-Tez/darknet/yolov4.cfg",
                                   "D:/01_Programlama/Yolo-Tez/weigths/yolov4_last.weights")

    layers = model.getLayerNames()

    # Outputların olduğu katmanlardan detectionları seçmek için
    output_layer = [ layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]

    model.setInput(frame_blob)

    detection_layers = model.forward(output_layer)

    ############ NON-MAXIMUM SUPRESSION - OPERATION 1 ###########

    ids_list = []
    boxes_list = []
    confidence_list= []
    
    ############ END OF OPERATION 1 ###########

    #%% 4.Bölüm
    
    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            
            #gelen ilk 5 değer nesnenin çevrelendiği kutu ile alakalı onları çıkarıyoruz
            scores = object_detection[5:]
            
            #en yüksek skorlu değerin indexini alıyoruz
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]
            
            if confidence > 0.40:
                
                #bulunan nesne hangi isimde
                label = labels[predicted_id]
                
                #gelen değerler anlamlı değildir bunları resimin en ve boyu ile genişletiyoruz
                bounding_box = object_detection[0:4] * np.array([frame_width,frame_height,frame_width,frame_height])
                
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
    
                start_x = int(box_center_x - (box_width/2))
                start_y = int(box_center_y - (box_height/2))
                
                ############ NON-MAXIMUM SUPRESSION - OPERATION 2 ###########
                
                ids_list.append(predicted_id)
                confidence_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
                
                ############ END OF OPERATION 2 ###########
            
            
    ############ NON-MAXIMUM SUPRESSION - OPERATION 3 ###########
              
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidence_list, 0.5, 0.4)
    
    for max_id in max_ids:
        max_class_id = max_id[0]
        box = boxes_list[max_class_id]
        
        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]
        
        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidence_list[max_class_id]
                
    ############ END OF OPERATION 3 ###########
                
        end_x = start_x + box_width
        end_y = start_y + box_height
    
        box_color = colors[predicted_id]
        box_color = [ int(each) for each in box_color]
                
                
        label="{}: {:.2f}%".format(label, confidence*100)
        print("predicted object {}".format(label))
        
        (text_width, text_height) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)[0]
    
        box_coords = ((start_x, start_y), (start_x + text_width, start_y - text_height * 2))
    
        cv2.rectangle(frame, box_coords[0], box_coords[1], box_color, cv2.FILLED) 
        cv2.rectangle(frame, (start_x, start_y), (end_x,end_y), box_color, 1)
        cv2.putText(frame, label, (start_x, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    
    cv2.imshow("Detection Window", frame)
    
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
    
    
cap.release()
cv2.destroyAllWindows()

















