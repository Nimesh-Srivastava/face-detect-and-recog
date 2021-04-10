import cv2
import face_recognition
import imutils
import pickle
import time
import os

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
