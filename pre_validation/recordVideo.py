#!/usr/bin/python

import cv2
import datetime
import time

# import sys

import argparse

parser = argparse.ArgumentParser(description='Recording videos and timestamps.')
parser.add_argument('-v', '--videoFilename', default='./video.avi', )
parser.add_argument('-t', '--timeFilename', default='./time.txt', )
parser.add_argument('-d', '--device', default=1, )

args = parser.parse_args()

if __name__ == "__main__":
    """
        采集数据时录制洗手视频和时间相应帧所在的时间戳
    """
    try:

        fps = 30
        frameWidth = 1920
        frameHeight = 1080

        cap = cv2.VideoCapture(args.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
        #	time.sleep()
        cap.set(cv2.CAP_PROP_FPS, fps)

        cameraFPS = cap.get(cv2.CAP_PROP_FPS)

        print("FPS:", cameraFPS)
        print("Frame size:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # fourcc = cv2.VideoWriter_fourcc(*'MJPG') # + .avi works, .mp4 not works
        # fourcc = cv2.cv.CV_FOURCC(*'XVID')MP4V

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        videofile = cv2.VideoWriter(args.videoFilename,
                                    fourcc,
                                    int(cameraFPS),
                                    (frameWidth, frameHeight))

        # file = open('/media/csipose1/XPG SD700X/time', 'w+')

        with open(args.timeFilename, 'w+') as file:
            while (cap.isOpened()):
                ret, frame = cap.read()
                # time.sleep(delay)
                t = datetime.datetime.now()
                # t = time.clock()
                # print(ret)
                if ret:
                    file.write(str(t) + '\n')
                    print(str(t))
                    videofile.write(frame)
                    cv2.imshow('Camera', frame)
                    cv2.waitKey(1)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break


    except KeyboardInterrupt:
        print("Quit")
        cap.release()
        videofile.release()
        # cv2.destroyAllWindows()
        file.close()
