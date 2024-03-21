import argparse
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# Parse the video file path argument
ap = argparse.ArgumentParser()
ap.add_argument('video', help='path to input video file')
args = vars(ap.parse_args())

# Loading JSON model
json_file = open('top_models\\fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Loading weights
model.load_weights('top_models\\fer.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(args['video'])

while True:
    ret, img = cap.read()
    if not ret:
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.2, 6)


    resized_img = cv2.resize(img, (1024, 768))
    cv2.imshow('Facial Emotion Recognition', resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
