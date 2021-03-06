'''
LAB COMPONENT 08
Submitted by - Nimesh Srivastava
               19BAI10026

For complete repository with encodings file, please refer - https://github.com/Nimesh-Srivastava/face-detect-and-recog
'''

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

cap = cv2.VideoCapture(0)
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
cascade_classifier = cv2.CascadeClassifier(haar)
data = pickle.loads(open('face_enc', "rb").read())

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(image)
    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1            
            name = max(counts, key=counts.get)

        names.append(name)

        for ((x, y, w, h), name) in zip(faces, names):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(1) &0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()
