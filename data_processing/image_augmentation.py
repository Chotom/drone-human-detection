import os
import random

import pandas as pd
import numpy as np
import albumentations as A
import cv2


def read_yolo_annotation_to_list(annotations_path: str):
    try:
        bboxes = pd.read_csv(annotations_path,
                             names=['label', 'x', 'y', 'w', 'h'],
                             dtype={'label': str, 'x': float, 'y': float, 'w': float, 'h': float}, sep=' ')
    except FileNotFoundError:
        return []

    return bboxes[['x', 'y', 'w', 'h', 'label']].values.tolist()


def selected_crop(image_path: str,
                  annotation_path: str,
                  new_width=256,
                  new_height=256,
                  new_filename='',
                  new_image_dir='',
                  new_annotation_dir='',
                  annotation_count=5,
                  scale_height=1080):
    """
    For every object in the given image use crop function and save cropped images with annotations.

    :param image_path: Path to image to crop.
    :param annotation_path: Path to annotation in xywhn (YOLO) format.
    :param new_width: Width of crop.
    :param new_height: Height of crop.
    :param new_filename: Filename suffix to save image and annotation (without extension).
    :param new_image_dir: Directory path to save cropped image.
    :param new_annotation_dir: Directory path to save cropped image annotations in xywhn (YOLO) format.
    :param annotation_count: Max number of random annotations to crop.
    :param scale_height: value to scale image = image height / scale_height.
    :return: calculated new image and annotations
    """
    assert new_image_dir != ''
    assert new_annotation_dir != ''
    os.makedirs(new_image_dir, exist_ok=True)
    os.makedirs(new_annotation_dir, exist_ok=True)

    image = cv2.imread(image_path)
    height, width, _ = image.shape
    annotation_list = read_yolo_annotation_to_list(annotation_path)
    annotation_count = min(annotation_count, len(annotation_list))
    scale = height / scale_height
    new_height, new_width = max(int(new_height * scale), 256), max(int(new_width * scale), 256)

    for i, annotation in enumerate(random.sample(annotation_list, annotation_count)):
        sample_w = float(annotation[2]) * width
        sample_h = float(annotation[3]) * height
        x1, y1 = float(annotation[0]) * width - sample_w / 2, float(annotation[1]) * height - sample_h / 2
        # x2, y2 = x1 + sample_w, y1 + sample_h

        dx = random.uniform(max(0, new_width - width + x1), min(new_width - sample_w, x1))
        dy = random.uniform(max(0, new_height - height + y1), min(new_height - sample_h, y1))
        crop_x1 = int(x1 - dx)
        crop_y1 = int(y1 - dy)
        crop_x2 = crop_x1 + new_width
        crop_y2 = crop_y1 + new_height

        # Images with objects visibility between min and max will not be saved.
        transform_min_vis = A.Compose([
            A.Crop(x_min=crop_x1, y_min=crop_y1, x_max=crop_x2, y_max=crop_y2)
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.2))
        transformed = transform_min_vis(image=image, bboxes=annotation_list)
        transformed_min_annotation = transformed['bboxes']

        transform_max_vis = A.Compose([
            A.Crop(x_min=crop_x1, y_min=crop_y1, x_max=crop_x2, y_max=crop_y2)
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.8))
        transformed = transform_max_vis(image=image, bboxes=annotation_list)
        transformed_max_annotation = transformed['bboxes']
        transformed_image = transformed['image']

        # Save image
        if np.shape(transformed_max_annotation) == np.shape(transformed_min_annotation):
            image_filename = f'{new_image_dir}/{new_filename}_{i}.jpg'
            annotation_filename = f'{new_annotation_dir}/{new_filename}_{i}.txt'
            cv2.imwrite(f'{image_filename}', transformed_image)

            df_annotation = pd.DataFrame(transformed_max_annotation, columns=['x', 'y', 'w', 'h', 'label'])
            df_annotation = df_annotation[['label', 'x', 'y', 'w', 'h']]
            df_annotation.to_csv(annotation_filename, sep=' ', header=False, index=False)
