import numpy as np
import tensorflow as tf
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util

cap = cv2.VideoCapture('drone.mp4')

PATH_TO_CKPT = "E:/drone_detection/train2/frozen_inference_graph.pb"

PATH_TO_LABELS = "E:/drone_detection/records/output.pbtxt"

NUM_CLASSES = 1

img_width = 1920
img_height = 1080

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter("drone.avi", fourcc, 20.0, (1920,1080))

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()

      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      boxes = np.squeeze(boxes)
      scores = np.squeeze(scores)
      classes = np.squeeze(classes)

      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          boxes,
          classes.astype(np.int32),
          scores,
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4, min_score_thresh=.1)

      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      # out.write(image_np)

      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break