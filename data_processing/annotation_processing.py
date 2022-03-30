import os
import pandas as pd
import cv2


def clean_annotation_files(source_dir: str,
                           result_dir: str,
                           file_cols_names: list[str],
                           labels_to_convert: list[str],
                           label: str):
    """
    Reads data from all files in the specified directory with columns separated with ',' format,
    and then convert chosen classes to new label, to saved them in new directory.

    :param source_dir: Data directory only with annotations.
    :param result_dir: Data directory to store cleaned annotation files.
    :param file_cols_names: Array with columns name, where column with label is named 'class'.
    :param labels_to_convert: Annotations to convert. Data without these annotations will be rejected.
    :param label: New label for annotations.
    """
    assert source_dir != result_dir
    os.makedirs(result_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        annotations = pd.read_csv(f'{source_dir}/{filename}', names=file_cols_names, dtype=str)
        annotations = annotations.loc[annotations['class'].isin(labels_to_convert)]
        annotations['class'] = label
        if not annotations.empty:
            annotations.to_csv(f'{result_dir}/{filename}', header=False, index=False)


def convert_visdrone_to_yolo_format(source_dir: str,
                                    img_source_dir: str,
                                    result_dir: str,
                                    file_cols_names: list[str]):
    """
    Reads data from all files in the specified directory with columns separated with ',' format,
    and convert x_left, y_top, width, height columns to x_center, y_center, scaled_width, scaled_height,
    to saved them in new directory. In a result each annotation file will be stored as
    `class x_center y_center width height` format, separated by space.

    :param source_dir: Directory path with annotations to convert.
    :param img_source_dir: Directory path with images.
    :param result_dir: Directory path to store result annotations.
    :param file_cols_names: Array with source columns name, where column with label is named 'class'.
    """
    assert source_dir != result_dir
    os.makedirs(result_dir, exist_ok=True)
    yolo_cols = ['class', 'x_center', 'y_center', 'scaled_width', 'scaled_height']

    for filename in os.listdir(source_dir):
        height, width, _ = cv2.imread(f'{img_source_dir}/{filename.split(".")[0]}.jpg').shape
        annotations = pd.read_csv(f'{source_dir}/{filename}', names=file_cols_names)
        annotations['x_center'] = (annotations['x_left'] + (annotations['width'] / 2)) / width
        annotations['y_center'] = (annotations['y_top'] - (annotations['height'] / 2)) / height
        annotations['scaled_width'] = annotations['width'] / width
        annotations['scaled_height'] = annotations['height'] / height
        annotations[yolo_cols].to_csv(f'{result_dir}/{filename}', sep=' ', header=False, index=False)
