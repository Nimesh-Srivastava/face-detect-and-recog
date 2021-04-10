from imutils import paths
import face_recognition
import pickle
import cv2
import os

img_path = list(paths.list_images('images'))
known_encodings = []
known_names = []

for (i, img_path) in enumerate(img_path):
    name = img_path.split(os.path.sep)[-2]
    image = cv2.imread(img_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        # convert openCV ordering to dlib ordering
    boxes = face_recognition.face_locations(rgb, model = 'hog')
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(name)
    
# Encodings and names are being saved in 'data' directory
data = {'Encodings' : known_encodings, 'Name' : known_names}
fil = open("face_encodings", "wb")
fil.write(pickle.dumps(data))
fil.close()