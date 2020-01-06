import numpy as np  
import cv2  
import dlib  
from scipy.spatial import distance as dist  
from scipy.spatial import ConvexHull  

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

FULL_POINTS = list(range(0, 68))  
FACE_POINTS = list(range(17, 68))  
   
JAWLINE_POINTS = list(range(0, 17))  
RIGHT_EYEBROW_POINTS = list(range(17, 22))  
LEFT_EYEBROW_POINTS = list(range(22, 27))  
NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42))  
LEFT_EYE_POINTS = list(range(42, 48))  
MOUTH_OUTLINE_POINTS = list(range(48, 61))  
MOUTH_INNER_POINTS = list(range(61, 68))

filter_img = cv2.imread('filters/eye.png', -1)
mask = filter_img[:, :, 3]
mask_inv = cv2.bitwise_not(mask)
filter_img = filter_img[:, :, 0:3]

filter_height, filter_width = filter_img.shape[:2]
# cv2.imshow("image", filter_img)
# cv2.waitKey(0)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)
        face_landmark = []
        for p in landmarks.parts():
        	face_landmark.append([p.x, p.y])

        landmark = np.matrix(face_landmark)
        left_eye = landmark[LEFT_EYE_POINTS]
        right_eye = landmark[RIGHT_EYE_POINTS]
        print(left_eye[0])
        print(left_eye[3])


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break