import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
tf.config.set_visible_devices([], 'GPU')

#path of the trained model
model_path = 'model/model.h5'#replace with your path
#Loading the model
model = load_model(model_path)
#Loading haarcascade to detect the faces
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#some initializations
labels_dict = {0:'MASK',1:'NO MASK'}
color_dict = {0:(0,255,0),1:(0,0,255)}

#set the image size
IMG_SIZE = 100

#setting the source as webcam
cap = cv2.VideoCapture(0)
while(True):
    #reading the image frames
    ret,img = cap.read()
    #converting to grayscale images
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #multiple faces
    faces = face_detector.detectMultiScale(gray,1.3,5)  
    
    for (x,y,w,h) in faces:
        #region of interest
        roi = gray[y:y+w,x:x+w]
        #resize the image to specified dimension
        roi_resized = cv2.resize(roi,(IMG_SIZE,IMG_SIZE))
        #normalize
        roi_normalized = roi_resized/255
        #converting into a batch of 1 image
        roi_reshaped = np.reshape(roi_normalized,(1,IMG_SIZE,IMG_SIZE,1))
        #model prediction
        result = model.predict(roi_reshaped)
        #getting the label
        label = int(np.round(result[0][0]))
        #drawing the rectangle around the faces
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        #drawing one more rectangle for giving title
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        #mask or no mask text
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow('Detecting mask',img)
    key = cv2.waitKey(1)
    
    #break if escape key is pressed
    if(key == 27):
        break
#destroy all the windows
cv2.destroyAllWindows()
cap.release()