import cv2

def livePhoto():
    """Take two live photos through your camera"""
    cap = cv2.VideoCapture(0)
    num = 1
    while True:
        _, frame = cap.read()

        cv2.imshow("Image -- (s)ave", frame)
        key = cv2.waitKey(1)&0xFF
        if key == ord("s"):
            frame = cv2.resize(frame, None, fx=.5, fy=.5) # reduce size by half
            cv2.imwrite(f"Image_{num}.jpg", frame)
            if num >= 2:
                break
            num+=1
