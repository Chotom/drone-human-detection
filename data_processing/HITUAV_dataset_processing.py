import json
import os
import shutil

def from_HITUAV_json_to_xywhn_yolo_format(json_source: str, img_source_dir: str, anno_result_dir: str):
    os.makedirs(anno_result_dir, exist_ok=True)
    yolo_cols = ['class', 'x_center', 'y_center', 'width', 'height']
    person_class_id = 0
    json_file = json.load(open(json_source))
    for image in json_file['images']:
        img_file_path = image['filename']
        img_height = image['height']
        img_width = image['width']
        img_file_path_parts = img_file_path.split('.')
        img_title = img_file_path_parts[0]
        anno_exists = False
        if os.path.exists(f'{img_source_dir}/{img_file_path}'):
            anno_file = open(f'{anno_result_dir}/{img_title}.txt', 'w')
            for annotation in json_file['annotation']:
                if annotation['image_id'] == image['id'] and annotation['category_id'] == person_class_id:
                    anno_exists = True
                    new_annotation = {}
                    new_annotation[yolo_cols[0]] = annotation['category_id']
                    new_annotation[yolo_cols[1]] = (annotation['bbox'][0] + annotation['bbox'][2]/2) / img_width
                    new_annotation[yolo_cols[2]] = (annotation['bbox'][1] + annotation['bbox'][3]/2) / img_height
                    new_annotation[yolo_cols[3]] = annotation['bbox'][2] / img_width
                    new_annotation[yolo_cols[4]] = annotation['bbox'][3] / img_height
                    anno_file.write(f'{new_annotation[yolo_cols[0]]} {new_annotation[yolo_cols[1]]} {new_annotation[yolo_cols[2]]} '
                                    f'{new_annotation[yolo_cols[3]]} {new_annotation[yolo_cols[4]]}\n')
            anno_file.close()
            if not anno_exists:
                os.remove(f'{anno_result_dir}/{img_title}.txt')

def copy_HITUAV_images_to_given_dir(anno_dir: str,img_source_dir: str, img_result_dir: str):
    os.makedirs(img_result_dir, exist_ok=True)
    for path in os.listdir(anno_dir):
        file_name = path.split('.')[0]
        if os.path.isfile(f'{anno_dir}/{file_name}.txt'):
            if os.path.isfile(f'{img_source_dir}/{file_name}.jpg'):
                shutil.copyfile(f'{img_source_dir}/{file_name}.jpg', f'{img_result_dir}/{file_name}.jpg')