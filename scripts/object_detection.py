#!/usr/bin/env python3                                                    
"""A ROS wrapper of the example of semantic segmentation from pycoral 
more detail can be found in https://github.com/google-coral/pycoral/blob/
master/examples/small_object_detection.py"""

import roslib
roslib.load_manifest("object_detection_tpu")
import argparse
import collections
import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
from sensor_msgs.msg import Image as ImageMsg
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

class object_detection_tpu:
  def __init__(self, model_path, label_path,score_threshold,tile_sizes_str):
    self.model = model_path
    self.label = label_path
    self.tile_sizes_str = tile_sizes_str
    self.tile_overlap =  20
    self.score_threshold = score_threshold
    self.iou_threshold = 0.1
    self.Object = collections.namedtuple('Object', ['label', 'score', 'bbox'])

    self.interpreter = make_interpreter(self.model)
    self.interpreter.allocate_tensors()
    self.labels = read_label_file(self.label) if self.label else {}
    # self.img_sub = rospy.Subscriber("~input", ImageMsg, self.callback)
    self.img_sub = rospy.Subscriber("/openni_camera/rgb/image_raw", ImageMsg, self.callback)    
    self.bridge = CvBridge()
    self.tile_sizes = []
    for tile_size in self.tile_sizes_str.split(','):
      size = tile_size.split('x')
      tile_size_int = [int(i) for i in size]
      self.tile_sizes.append(tile_size_int)


  def tiles_location_gen(self, img_size, tile_size, overlap):

    tile_width, tile_height = tile_size
    img_width, img_height = img_size
    h_stride = tile_height - overlap
    w_stride = tile_width - overlap
    for h in range(0, img_height, h_stride):
      for w in range(0, img_width, w_stride):
        xmin = w
        ymin = h
        xmax = min(img_width, w + tile_width)
        ymax = min(img_height, h + tile_height)
        yield [xmin, ymin, xmax, ymax]


  def non_max_suppression(self, objects, threshold):
    if len(objects) == 1:
      return [0]

    boxes = np.array([o.bbox for o in objects])
    xmins = boxes[:, 0]
    ymins = boxes[:, 1]
    xmaxs = boxes[:, 2]
    ymaxs = boxes[:, 3]

    areas = (xmaxs - xmins) * (ymaxs - ymins)
    scores = [o.score for o in objects]
    idxs = np.argsort(scores)

    selected_idxs = []
    while idxs.size != 0:

      selected_idx = idxs[-1]
      selected_idxs.append(selected_idx)

      overlapped_xmins = np.maximum(xmins[selected_idx], xmins[idxs[:-1]])
      overlapped_ymins = np.maximum(ymins[selected_idx], ymins[idxs[:-1]])
      overlapped_xmaxs = np.minimum(xmaxs[selected_idx], xmaxs[idxs[:-1]])
      overlapped_ymaxs = np.minimum(ymaxs[selected_idx], ymaxs[idxs[:-1]])

      w = np.maximum(0, overlapped_xmaxs - overlapped_xmins)
      h = np.maximum(0, overlapped_ymaxs - overlapped_ymins)

      intersections = w * h
      unions = areas[idxs[:-1]] + areas[selected_idx] - intersections
      ious = intersections / unions

      idxs = np.delete(
        idxs, np.concatenate(([len(idxs) - 1], np.where(ious > threshold)[0])))

    return selected_idxs


  def draw_object(self, draw, obj):

    draw.rectangle(obj.bbox, outline='red')
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    # draw.text((obj.bbox[0], obj.bbox[3] - 20), obj.label, fill='#0000',font=font)
    draw.text((obj.bbox[0], obj.bbox[1] - 20), obj.label, fill='#0000',font=font)
    draw.text((obj.bbox[0], obj.bbox[3] + 10), str(obj.score), fill='#0000',font=font)


  def reposition_bounding_box(self, bbox, tile_location):
    bbox[0] = bbox[0] + tile_location[0]
    bbox[1] = bbox[1] + tile_location[1]
    bbox[2] = bbox[2] + tile_location[0]
    bbox[3] = bbox[3] + tile_location[1]
    return bbox

  def callback(self,data):
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
    cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img)
    
    objects_by_label = dict()
    img_size = img.size
    for tile_size in self.tile_sizes:
      for tile_location in self.tiles_location_gen(img_size, tile_size,
                                                   self.tile_overlap):
        tile = img.crop(tile_location)
        _, scale = common.set_resized_input(
          self.interpreter, tile.size,
          lambda size, img=tile: img.resize(size, Image.NEAREST))
        self.interpreter.invoke()
        objs = detect.get_objects(self.interpreter, self.score_threshold, scale)

        for obj in objs:
          bbox = [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax]

          bbox = self.reposition_bounding_box(bbox, tile_location)

          label = self.labels.get(obj.id, '')

          ## test: upperside and lowerside of input pic is cropped
          img_width, img_height = img_size
          if img_width == 640:
            if bbox[3] < 100 or bbox[1] > 400 or bbox[3] > 440:
              continue
            if bbox[0] < 180 or bbox[2] > 550:
              continue

          objects_by_label.setdefault(label,
                            []).append(self.Object(label, obj.score, bbox))

    for label, objects in objects_by_label.items():
      idxs = self.non_max_suppression(objects, self.iou_threshold)
      for idx in idxs:
        self.draw_object(draw, objects[idx])
        rospy.logdebug("found id:{}({}) conf:{}".format(label, 0, objects[idx].score))
        rospy.loginfo("found id:{} conf:{}".format(label, objects[idx].score))
    cvimg = np.array(img)
    cv_show = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    cv2.imshow("output",cv_show)
    cv2.waitKey(3)
    
def main():
  rospy.init_node('object_detection_tpu', anonymous=True)
  # model_path = rospy.get_param('~model_path')
  # label_path = rospy.get_param('~label_path')
  # score_threshold = rospy.get_param('~score_threshold')
  # tile_sizes = rospy.get_param('~tile_sizes',default="500x500,300x300,250x250")
  model_path = "../models/my_object_detection.tflite"
  label_path = "../models/my_label.txt"
  # score_threshold = 0.65
  # score_threshold = 0.72
  score_threshold = 0.55
  # tile_sizes = rospy.get_param('~tile_sizes',default="750x500,300x300,250x250")
  tile_sizes = rospy.get_param('~tile_sizes',default="640x480,300x300,250x250")
  # tile_sizes = rospy.get_param('~tile_sizes',default="750x500,640x480,300x300")
  #tile_overlap = rospy.get_param('~tile_overlap')
  #iou_threshold = rospy.get_param('~iou_threshold')
  odt = object_detection_tpu(model_path, label_path,score_threshold,tile_sizes)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
  main()
