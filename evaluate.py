# Append to $PYTHONPATH path to models/research and cocoapi/PythonAPI
from object_detection.metrics import coco_evaluation
from object_detection.core import standard_fields
from object_detection.utils.label_map_util import create_categories_from_labelmap, get_label_map_dict
import tensorflow as tf
from PIL import Image
import xml.etree.ElementTree as Et
import pandas as pd
import numpy as np
import cv2
import os

cl = ["S maskou", "Bez masky"]


def prepare_input(image_path):
    """ Input image preprocessing for SSD MobileNet format
    args:
        image_path: path to image
    returns:
        input_data: numpy array of shape (1, width, height, channel) after preprocessing
    """
    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(image_path).convert('RGB').resize((width, height))
    # Using OpenCV
    # img = cv2.resize(cv2.imread(image_path), (width,height))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    # check the type of the input tensor
    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    elif input_details[0]['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details[0]["quantization"]
        input_data = input_data / input_scale + input_zero_point
        input_data = input_data.astype(np.uint8)
    return input_data


def voc_parser(path_to_xml_file, label_map_dict):
    """Parser for Pascal VOC format annotation to TF OD API format
    args:
        path_to_xml_file : path to annotation in Pascal VOC format
        label_map_dict : dictionary of class name to index
    returns
        boxes: array of boundary boxes (m, 4) where each row is [ymin, xmin, ymax, xmax]
        classes: list of class index (m, 1)
        where m is the number of objects
    """
    boxes = []
    classes = []

    xml = open(path_to_xml_file, "r")
    tree = Et.parse(xml)
    root = tree.getroot()
    xml_size = root.find("size")

    objects = root.findall("object")
    if len(objects) == 0:
        print("No objects for {}")
        return boxes, classes

    obj_index = 0
    for obj in objects:
        class_id = label_map_dict[obj.find("name").text]
        xml_bndbox = obj.find("bndbox")
        xmin = float(xml_bndbox.find("xmin").text)
        ymin = float(xml_bndbox.find("ymin").text)
        xmax = float(xml_bndbox.find("xmax").text)
        ymax = float(xml_bndbox.find("ymax").text)
        boxes.append([ymin, xmin, ymax, xmax])
        classes.append(class_id)
    return boxes, classes


def draw_boundaryboxes(image_path, annotation_path):
    """ Draw the detection boundary boxes
    args:
        image_path: path to image
        annotation_path: path to groundtruth in Pascal VOC format .xml
    """
    # Draw detection boundary boxes
    dt_boxes, dt_classes, dt_scores = postprocess_output(image_path)
    image = cv2.imread(image_path)

    # Draw groundtruth boundary boxes
    label_map_dict = get_label_map_dict(_label_file)
    # Read groundtruth from XML file in Pascal VOC format
    gt_boxes, gt_classes = voc_parser(annotation_path, label_map_dict)

    for i in range(1):
        [ymin, xmin, ymax, xmax] = list(map(int, dt_boxes[i]))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
        cv2.putText(image, cl[gt_classes[i] - 1], (xmin + 10, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (10, 255, 0), 2)
        # + '({}% score)'.format(int(dt_scores[i] * 100))

    for i in range(len(gt_boxes)):
        [ymin, xmin, ymax, xmax] = list(map(int, gt_boxes[i]))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

    saved_path = "evaluated_new/out_" + os.path.basename(image_path)
    cv2.imwrite(os.path.join(saved_path), image)
    print("Saved at", saved_path)


def postprocess_output(image_path):
    """ Output post processing
    args:
        image_path: path to image
    returns:
        boxes: numpy array (num_det, 4) of boundary boxes at image scale
        classes: numpy array (num_det) of class index
        scores: numpy array (num_det) of scores
        num_det: (int) the number of detections
    """
    # SSD Mobilenet tflite model returns 10 boxes by default.
    # Use the output tensor at 4th index to get the number of valid boxes
    # num_det = int(interpreter.get_tensor(output_details[3]['index']))
    # boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    # classes = interpreter.get_tensor(output_details[1]['index'])[0]
    # scores = interpreter.get_tensor(output_details[2]['index'])[0]
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    # Scale the output to the input image size
    img_width, img_height = Image.open(image_path).size  # PIL
    # img_height, img_width, _ = cv2.imread(image_path).shape # OpenCV

    df = pd.DataFrame(boxes)
    df['ymin'] = df[0].apply(lambda y: max(1, (y * img_height)))
    df['xmin'] = df[1].apply(lambda x: max(1, (x * img_width)))
    df['ymax'] = df[2].apply(lambda y: min(img_height, (y * img_height)))
    df['xmax'] = df[3].apply(lambda x: min(img_width, (x * img_width)))
    boxes_scaled = df[['ymin', 'xmin', 'ymax', 'xmax']].to_numpy()

    return boxes_scaled, classes, scores


def evaluate_single_image(image_path, annotation_path, label_file):
    """ Evaluate mAP on image
    args:
        image_path: path to image
        annotation_path: path to groundtruth in Pascal VOC format .xml
        label_file: path to label_map.pbtxt
    """

    categories = create_categories_from_labelmap(label_file)
    label_map_dict = get_label_map_dict(label_file)
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(categories)
    image_name = os.path.basename(image_path).split('.')[0]

    # Read groundtruth (here, an XML file in Pascal VOC format)
    gt_boxes, gt_classes = voc_parser(annotation_path, label_map_dict)
    # Get the detection after post processing
    dt_boxes, dt_classes, dt_scores = postprocess_output(image_path)

    coco_evaluator.add_single_ground_truth_image_info(
        image_id=image_name,
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array(gt_boxes),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array(gt_classes)
        })
    coco_evaluator.add_single_detected_image_info(
        image_id=image_name,
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
                dt_boxes,
            standard_fields.DetectionResultFields.detection_scores:
                dt_scores,
            standard_fields.DetectionResultFields.detection_classes:
                dt_classes
        })

    coco_evaluator.evaluate()


interpreter = tf.lite.Interpreter(model_path='detect.tflite')

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

_label_file = 'annotations/label_map.pbtxt'


def evaluate(_images):
    for _image in _images:
        _image_path = 'images/' + _image + '.jpg'
        _annotation_path = 'annotations/eval_labels/' + _image + '.xml'

        input_data = prepare_input(_image_path)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        boxes, classes, scores = postprocess_output(_image_path)

        evaluate_single_image(image_path=_image_path, annotation_path=_annotation_path, label_file=_label_file)
        draw_boundaryboxes(image_path=_image_path, annotation_path=_annotation_path)


_images = []
for filename in os.listdir('images'):
    _images.append(filename[0:-4])

evaluate(_images)
