# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import argparse

import numpy as np
import scipy.ndimage
from PIL import Image, ImageDraw
import cv2 as cv

from pose_engine import PoseEngine, EDGES, BODYPIX_PARTS


# Color mapping for bodyparts
RED_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "right" in v]
GREEN_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "hand" in v or "torso" in v]
BLUE_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "leg" in v or "arm" in v or "face" in v or "hand" in v]

def draw_pose(dwg, pose, color='blue', threshold=0.2, marker_size=10, linewidth=5):
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue
        xys[label] = (int(keypoint.yx[1]), int(keypoint.yx[0]))
        dwg.ellipse((int(keypoint.yx[1]-marker_size/2), int(keypoint.yx[0]-marker_size/2), int(keypoint.yx[1]+marker_size/2), int(keypoint.yx[0]+marker_size/2)), fill=color)
    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        dwg.line((ax, ay, bx, by), fill=color, width=linewidth)

class PoseSeg:
  def __init__(self, model, anonymize=True, bodyparts=True, drawposes=True):
    self.anonymize = anonymize
    self.bodyparts = bodyparts
    self.drawposes = drawposes
    self.background_image = None
    self.last_time = time.monotonic()
    self.frames = 0
    self.sum_fps = 0
    self.sum_process_time = 0
    self.sum_inference_time = 0

    self.engine = PoseEngine(model)
    inference_size = (self.engine.image_width, self.engine.image_height)
    print('Inference size: {}'.format(inference_size))

  def process(self, image, only_mask = False):
    start_time = time.monotonic()
    inference_time, poses, heatmap, bodyparts = self.engine.DetectPosesInImage(image)

    def clip_heatmap(heatmap, v0, v1):
      a = v0 / (v0 - v1)
      b = 1.0 / (v1 - v0)
      return np.clip(a + b * heatmap, 0.0, 1.0)

    # clip heatmap to create a mask
    heatmap = clip_heatmap(heatmap,  -1.0,  1.0)

    rescale_factor = [
      image.shape[0]/heatmap.shape[0],
      image.shape[1]/heatmap.shape[1],
      1]
      
    if only_mask:
      rgb_heatmap = cv.cvtColor(heatmap*255, cv.COLOR_GRAY2RGB)
      return np.uint8(np.clip(scipy.ndimage.zoom(rgb_heatmap, rescale_factor, order=0),0,255))

    if self.bodyparts:
      rgb_heatmap = np.dstack([
            heatmap*(np.sum(bodyparts[:,:,RED_BODYPARTS], axis=2)-0.5)*100,
            heatmap*(np.sum(bodyparts[:,:,GREEN_BODYPARTS], axis=2)-0.5)*100,
            heatmap*(np.sum(bodyparts[:,:,BLUE_BODYPARTS], axis=2)-0.5)*100,
          ])
    else:
      rgb_heatmap = np.dstack([heatmap[:,:]*100]*3)
      rgb_heatmap[:,:,1:] = 0 # make it red

    rgb_heatmap= 155*np.clip(rgb_heatmap, 0, 1)
    rgb_heatmap = scipy.ndimage.zoom(rgb_heatmap, rescale_factor, order=0)

    if self.anonymize:
      if self.background_image is None:
        self.background_image = np.float32(np.zeros_like(image))
      # Estimate instantaneous background
      mask = np.clip(np.sum(rgb_heatmap, axis=2), 0, 1)[:,:,np.newaxis]
      background_estimate = (self.background_image*mask+ image*(1.0-mask))

      # Mix into continuous estimate with decay
      ratio = 1/max(1,self.frames/2.0)
      self.background_image = self.background_image*(1.0-ratio) + ratio*background_estimate
    else:
      self.background_image = image

    output_image = self.background_image + rgb_heatmap
    int_img = np.uint8(np.clip(output_image,0,255))

    end_time = time.monotonic()

    self.frames += 1
    self.sum_fps += 1.0 / (end_time - self.last_time)
    self.sum_process_time += 1000 * (end_time - start_time) - inference_time
    self.sum_inference_time += inference_time
    self.last_time = end_time
    text_line = 'PoseNet: %.1fms Frame IO: %.2fms TrueFPS: %.2f Nposes %d' % (
        self.sum_inference_time / self.frames,
        self.sum_process_time / self.frames,
        self.sum_fps / self.frames,
        len(poses)
    )

    if self.drawposes:
      int_img = Image.fromarray(int_img)
      draw = ImageDraw.Draw(int_img)
      for pose in poses:
          draw_pose(draw, pose)
      print(text_line)
      return np.asanyarray(int_img)
    else:
      return int_img

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--width', help='Source width', default='640')
    parser.add_argument('--height', help='Source height', default='480')
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')

    parser.add_argument('--anonymize', dest='anonymize', action='store_true', help='Use anonymizer mode [--noanonymize]')
    parser.add_argument('--noanonymize', dest='anonymize', action='store_false', help=argparse.SUPPRESS)
    parser.set_defaults(anonymize=False)

    parser.add_argument('--bodyparts', dest='bodyparts', action='store_true', help='Color by bodyparts [--nobodyparts]')
    parser.add_argument('--nobodyparts', dest='bodyparts', action='store_false', help=argparse.SUPPRESS)
    parser.set_defaults(bodyparts=True)

    args = parser.parse_args()



    cap = cv.VideoCapture(int(args.videosrc[-1]))

    codec = 0x47504A4D # MJPG
    if not cap.set(cv.CAP_PROP_FOURCC, codec):
      print("Your webcam doesn't support MJPG :(")
      print("Check the output of: uvcdynctrl -f")
      width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
      height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    else:
      width = args.width
      height = args.height
      cap.set(cv.CAP_PROP_FRAME_WIDTH, int(width))
      cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(height))


    default_model = f'models/bodypix_mobilenet_v1_075_{width}_{height}_16_quant_edgetpu_decoder.tflite'
    model = args.model if args.model else default_model
    print('Model: {}'.format(model))


    src_size = (int(width), int(height))
    if args.videosrc.startswith('/dev/video'):
        print('Source size: {}'.format(src_size))


    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    else:
        print("Press Q to exit...")

    poseseg = PoseSeg(model, anonymize=args.anonymize, bodyparts=args.bodyparts)
    while True:
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            print("Exiting...")
            break
          
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Display the resulting frame
        img = cv.cvtColor(poseseg.process(img), cv.COLOR_RGB2BGR)
        cv.imshow('frame', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
