import cv2
import numpy as np
import glob
from PIL import ImageGrab
import lanedetect as ld
from lanedetect import Line
from lanedetect import final_pipeline
import time

src = np.float32([
    [210, 700],
    [570, 460],
    [705, 460],
    [1075, 700]
])

# You can Quit with pushing ESC
def show_webcam(mirror=False,lane=False):
    cam = cv2.VideoCapture(0)
    #3. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
    #4. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
    #5. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
    #4. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
    #5. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
    #cam.set(5,720)
    #width = cam.get(3)  # float
    #height = cam.get(4) # float
    print("Resolution:",width,'x',height)
    # 640 x 480
    width = cam.set(3,1280)  # float
    height = cam.set(4,720) # float

    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        if lane:
            img = final_pipeline(img)
        else:
            img = ld.add_points(img,src)
        cv2.imshow('my webcam', img)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()

def main():
    show_webcam(mirror=True,lane=False)



if __name__ == '__main__':
    main()
