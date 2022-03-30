import os
import shutil
import cv2

from matplotlib import pyplot as plt


def copy_annotated_images(source_dir: str,
                          annotation_dir: str,
                          result_dir: str):
    """
    Copy images only with annotations from source directory.
    Images and annotation are matched by filename.

    :param source_dir: Data directory only with images.
    :param annotation_dir: Data directory only with annotations.
    :param result_dir: Data directory to store chosen images.
    """
    assert source_dir != result_dir
    os.makedirs(result_dir, exist_ok=True)

    for filename in os.listdir(annotation_dir):
        src = f'{source_dir}/{filename.split(".")[0]}.jpg'
        dst = f'{result_dir}/{filename.split(".")[0]}.jpg'
        shutil.copyfile(src, dst)


def plot_yolo_format_annotated_image(img_path: str, annotation_path: str):
    """
    Plot image with bounding box.

    :param img_path: path to image
    :param annotation_path: txt file with yolo format annotation
    """
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    annotation_file = open(annotation_path)

    for annotation in annotation_file.readlines():
        sample = annotation.split(' ')
        sample_w = float(sample[3]) * width
        sample_h = float(sample[4]) * height
        x1, y1 = float(sample[1]) * width - sample_w / 2, float(sample[2]) * height + sample_h / 2
        x2, y2 = x1 + sample_w, y1 + sample_h
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    annotation_file.close()

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
