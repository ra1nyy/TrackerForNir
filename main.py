from controller import Controller
from tracker import KCFTracker

import cv2
import sys


def main():
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(sys.argv[1])

    #                    hog,   fixed_window, multiscale
    tracker = KCFTracker(False, True,         True)

    controller = Controller(tracker, cap)

    # начинаем работу алгоритма
    controller.update()


if __name__ == '__main__':
    main()
