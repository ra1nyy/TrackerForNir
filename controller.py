import numpy as np
import cv2

from time import time

# ====== Глобальные переменные =======
IS_OBJ_SELECTED  = False
IS_TRACKING_INIT = False
IS_ON_TRACKING   = False

WIDTH  = 0 
HEIGHT = 0

INTERVAL = 1
DURATION = 0.01

ix, iy, cx, cy = -1, -1, -1, -1
# =====================================


class Controller:
    def __init__(self, tracker, cap):
        self.tracker = tracker
        self.cap = cap

    def draw_bounding_box(self, event, x, y, flags, param):
        global IS_OBJ_SELECTED, IS_TRACKING_INIT, IS_ON_TRACKING, WIDTH, HEIGHT, ix, iy, cx, cy
        if event == cv2.EVENT_LBUTTONDOWN:
            IS_OBJ_SELECTED = True
            IS_ON_TRACKING = False
            ix, iy = x, y
            cx, cy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            cx, cy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            IS_OBJ_SELECTED = False
            if(abs(x - ix) > 10 and abs(y - iy) > 10):
                WIDTH, HEIGHT = abs(x - ix), abs(y - iy)
                ix, iy = min(x, ix), min(y, iy)
                IS_TRACKING_INIT = True
            else:
                IS_ON_TRACKING = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            IS_ON_TRACKING = False
            if(WIDTH > 0):
                ix, iy = x - WIDTH / 2, y - HEIGHT / 2
                IS_TRACKING_INIT = True
    
    def update(self):
        cv2.namedWindow('tracking')
        cv2.setMouseCallback('tracking', self.draw_bounding_box)
        global IS_OBJ_SELECTED, IS_TRACKING_INIT, IS_ON_TRACKING, ix, iy, cx, cy, WIDTH, HEIGHT, DURATION

        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if not ret:
                break

            if (IS_OBJ_SELECTED):
                cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)
            elif (IS_TRACKING_INIT):
                cv2.rectangle(frame, (ix, iy), (ix + WIDTH, iy + HEIGHT), (0, 255, 255), 2)
                print([ix, iy, WIDTH, HEIGHT])
                self.tracker.init([ix, iy, WIDTH, HEIGHT], frame)

                IS_TRACKING_INIT = False
                IS_ON_TRACKING = True
            elif (IS_ON_TRACKING):
                t0 = time()
                boundingbox = self.tracker.update(frame)
                t1 = time()

                boundingbox = list(map(int, boundingbox))
                print(boundingbox)
                cv2.rectangle(frame, (boundingbox[0], boundingbox[1]), (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 255, 255), 1)

                DURATION = 0.8 * DURATION + 0.2 * (t1 - t0)
                #DURATION = t1-t0
                cv2.putText(frame, 'FPS: ' + str(1 / DURATION)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow('tracking', frame)
            c = cv2.waitKey(INTERVAL) & 0xFF
            if c == 27 or c == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
