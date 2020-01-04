import cv2
import numpy as np
import dlib


points = []
image = None

def detectFaceLandmarks(frame):
    """Sets 68 Facemark-points if a face is detected in the frame"""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    result=[]
    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x,y), 2, (255, 0, 0), -1) #mark points in the image
            result.append("{} {}".format(x,y))
    h, w, c = frame.shape
    result+= ["0 0",f"{w} 0", f"0 {h}", f"{w} {h}"]
    return result

def click_and_mark(event, x, y, flags, param):
    """Marks a point in the frame per click"""
    global points, image
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append(f"{x} {y}")
        cv2.circle(image, (x,y), 2, (255, 0, 0), -1)
        image = cv2.putText(image, str(len(points)), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                           .3, (255, 0, 0)) # display the point number

        cv2.imshow("Add Points -- (c)ontinue , (s)ave Points", image)

def eval_points(frame, help_frame, name):
    """Interface for adding warp-points"""
    global points, image
    face_points = detectFaceLandmarks(frame)
    image = frame
    points = face_points
    cv2.namedWindow("Add Points -- (c)ontinue , (s)ave Points")
    cv2.setMouseCallback("Add Points -- (c)ontinue , (s)ave Points", click_and_mark)
    while True:
        # display the frame and wait for a keypress
        cv2.imshow("Other Window", help_frame)
        cv2.imshow("Add Points -- (c)ontinue , (s)ave Points", image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == ord("c"):
            cv2.destroyAllWindows()
            break
        if key == ord("s"):
        # if the 's' key is pressed, save points to file
            with open(name, 'w') as file:
                file.write("\n".join(points))
    return points
