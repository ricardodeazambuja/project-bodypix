"""
$ sudo pip3 install git+git://github.com/ricardodeazambuja/pyfakewebcam --upgrade
$ sudo apt install v4l2loopback-utils

And at least once (until next reboot) before running this script:
$ sudo modprobe v4l2loopback video_nr=2 card_label="fake webcam" exclusive_caps=1

Open another window and check:
$ ffplay /dev/video2

Or test directly on your browser:
https://webrtc.github.io/samples/src/content/devices/input-output/

Problems? Check the instructions: https://github.com/ricardodeazambuja/pyfakewebcam#caveats
"""

import pyfakewebcam

from bodypix import *

width = 640
height = 480

try:
    camera = pyfakewebcam.FakeWebcam('/dev/video2', width, height)
except FileNotFoundError:
    print("""Did you follow the instructions to enable v4l2loopback? 
    https://ricardodeazambuja.com/python/2021/02/13/edgetpu_virtual_webcam/
    """)
    raise

try:
    cap = cv.VideoCapture(0)

    model = f'models/bodypix_mobilenet_v1_075_{width}_{height}_16_quant_edgetpu_decoder.tflite'

    poseseg = PoseSeg(model, anonymize=True, bodyparts=False)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    else:
        print("Ctrl+C to exit...")

    while True:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Exiting...")
            break

        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Display the resulting frame
        img = cv.cvtColor(poseseg.process(img, only_mask=False), cv.COLOR_RGB2BGR)

        camera.schedule_frame(img)
except KeyboardInterrupt:
    pass
finally:
    cap.release()