import os
import shutil
import cv2
import numpy as np
from numpy.linalg import norm
import pandas as pd

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


def plot_xywhn_annotated_image_from_file(img_path: str, annotation_path: str):
    """
    Plot image with bounding box from annotation file.

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
        x1, y1 = float(sample[1]) * width - sample_w / 2, float(sample[2]) * height - sample_h / 2
        x2, y2 = x1 + sample_w, y1 + sample_h
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    annotation_file.close()

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def plot_xywhn_annotated_image_from_df(img_path: str, df_annotation: pd.DataFrame):
    """
    Plot image with bounding box from dataframe.

    :param img_path: path to image
    :param df_annotation: dataframe with xywhn format annotation
    """
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    for index, row in df_annotation.iterrows():
        sample_w = float(row['width']) * width
        sample_h = float(row['height']) * height
        x1, y1 = float(row['xcenter']) * width - sample_w / 2, float(row['ycenter']) * height - sample_h / 2
        x2, y2 = x1 + sample_w, y1 + sample_h
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def plot_xywhn_annotated_array_from_df(img: np.array, df_annotation: pd.DataFrame):
    """
    Plot image with bounding box from dataframe.

    :param img: image array
    :param df_annotation: dataframe with xywhn format annotation
    """
    img_copy = img.copy()
    height, width, _ = img_copy.shape

    for index, row in df_annotation.iterrows():
        sample_w = float(row['width']) * width
        sample_h = float(row['height']) * height
        x1, y1 = float(row['xcenter']) * width - sample_w / 2, float(row['ycenter']) * height - sample_h / 2
        x2, y2 = x1 + sample_w, y1 + sample_h
        cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_copy)


def get_brightness_stats(source_dir: str) -> pd.DataFrame:
    """
    Calculate avg brightness for every image in given source directory.

    :param source_dir: Path to directory with images.
    :return: Dataframe with 'filename', 'avg brightness' columns.
    """
    stats: list[list[str, float]] = []

    for filename in os.listdir(source_dir):
        img = cv2.imread(f'{source_dir}/{filename}')
        # For RGB
        if len(img.shape) == 3:
            stats.append([filename, np.average(norm(img, axis=2)) / np.sqrt(3)])
        # Grayscale
        else:
            stats.append([filename, np.average(img)])
    return pd.DataFrame(stats, columns=['filename', 'avg brightness'])


def get_number_of_objects_stats(source_dir: str) -> pd.DataFrame:
    """
    Calculate number of objects in every image in given source directory.

    :param source_dir: Path to directory with annotations.
    :return: Dataframe with 'filename', 'avg brightness' columns.
    """
    stats: list[list[str, int]] = []

    for filename in os.listdir(source_dir):
        annotations = pd.read_csv(f'{source_dir}/{filename}', dtype=str, sep=' ')
        stats.append([filename, len(annotations.index)])
    return pd.DataFrame(stats, columns=['filename', 'number of objects'])


def remove_xywhn_img_with_small_obj_from_dir(img_source_dir: str, source_dir: str):
    """
    Remove images from dir with too small objects.

    :param img_source_dir: Directory with images.
    :param source_dir: Directory with labels.
    """

    for filename in os.listdir(source_dir):

        annotations = pd.read_csv(f'{source_dir}/{filename}', sep=' ', names=['class', 'x', 'y', 'w', 'h'])

        h_is_small: pd.Series = annotations['h'] < 0.05
        if h_is_small.any():
            print(f'{source_dir}/{filename}')
            os.remove(f'{source_dir}/{filename}')
            os.remove(f'{img_source_dir}/{filename.split(".")[0]}.jpg')


def remove_xywhn_img_with_big_obj_from_dir(img_source_dir: str, source_dir: str):
    """
    Remove images from dir with too small objects.

    :param img_source_dir: Directory with images.
    :param source_dir: Directory with labels.
    """

    for filename in os.listdir(source_dir):

        annotations = pd.read_csv(f'{source_dir}/{filename}', sep=' ', names=['class', 'x', 'y', 'w', 'h'])

        h_is_big: pd.Series = annotations['h'] > 0.40
        if h_is_big.any():
            print(f'{source_dir}/{filename}')
            os.remove(f'{source_dir}/{filename}')
            os.remove(f'{img_source_dir}/{filename.split(".")[0]}.jpg')


def remove_img_below_min_brightness(img_source_dir: str, source_dir: str, min_brightness: int = 60):
    """
    Remove images from dir with too small objects.

    :param img_source_dir: Directory with images.
    :param source_dir: Directory with labels.
    :param min_brightness: Value between 0 and 255.
    """

    for filename in os.listdir(img_source_dir):
        img = cv2.imread(f'{img_source_dir}/{filename}')
        # For RGB
        if len(img.shape) == 3:
            brightness = np.average(norm(img, axis=2)) / np.sqrt(3)
        # Grayscale
        else:
            brightness = np.average(img)

        if brightness < min_brightness:
            print(f'{source_dir}/{filename}')
            os.remove(f'{source_dir}/{filename.split(".")[0]}.txt')
            os.remove(f'{img_source_dir}/{filename}')
