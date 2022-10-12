import json
import os
import random
import shutil
import cv2

def from_tiny_people_json_to_xywhn_yolo_format(json_source: str, img_source_dir: str, anno_result_dir: str):
    os.makedirs(anno_result_dir, exist_ok=True)
    yolo_cols = ['class', 'x_center', 'y_center', 'width', 'height']
    new_class_id = 0
    file = json.load(open(json_source))
    for image in file["images"]:
        img_file_path = image['file_name']
        img_file = img_file_path.split('/')
        img_file_name = img_file[1].split('.')[0]
        verified_file_name = check_if_viable_file_name(img_file[1])
        if img_file[1] != verified_file_name:
            os.rename(f'{img_source_dir}/{img_file[1]}', f'{img_source_dir}/{verified_file_name}.jpg')
            img_file_name = verified_file_name
        if os.path.exists(f'{img_source_dir}/{img_file_name}.jpg'):
            img = cv2.imread(f'{img_source_dir}/{img_file_name}.jpg')
            height, width, _ = img.shape
            if not img_file[0] != 'labeled_images':
                f = open(f'{anno_result_dir}/{img_file_name}.txt', 'w')
                for annotations in file['annotations']:
                    if annotations['image_id'] == image['id']:
                        annotation = {}
                        annotation[yolo_cols[0]] = new_class_id
                        annotation[yolo_cols[1]] = (annotations['bbox'][0] + annotations['bbox'][2] / 2) / width
                        annotation[yolo_cols[2]] = (annotations['bbox'][1] + annotations['bbox'][3] / 2) / height
                        annotation[yolo_cols[3]] = annotations['bbox'][2] / width
                        annotation[yolo_cols[4]] = annotations['bbox'][3] / height
                        area = annotations['area']
                        f.write(f'{annotation[yolo_cols[0]]} {annotation[yolo_cols[1]]} {annotation[yolo_cols[2]]} '
                                f'{annotation[yolo_cols[3]]} {annotation[yolo_cols[4]]} {area} \n')
                f.close()


def check_if_viable_file_name(file_name: str):
    if len(file_name.split('.')) > 2:
        return file_name.split('.')[0]
    if len(file_name.split('©')) > 1:
        return file_name.split('©')[0]
    else:
        return file_name


def clean_up_bboxes_in_blurred_areas(train_path: str, validation_path: str, test_path: str):
    dirs = [train_path, validation_path, test_path]
    for dir in dirs:
        for path in os.listdir(dir):
            file_path = dir + path
            file = open(file_path, 'r')
            total_area = 0
            num_of_annos = 0
            lines = file.readlines()
            for line in lines:
                clipped_line = line.split(" ")
                if len(clipped_line) > 5:
                    total_area += float(clipped_line[5])
                    num_of_annos += 1
            if num_of_annos > 0:
                avg_area = (total_area / num_of_annos)
                file.close()
                file = open(file_path, 'w')
                for line in lines:
                    clipped_line = line.split(" ")
                    if float(clipped_line[5]) > 2.25 * avg_area:
                        continue
                    file.write(line)
                file.close()


def copy_tiny_people_images_to_given_dir(img_source_dir: str, img_result_dir: str):
    os.makedirs(img_result_dir, exist_ok=True)
    for path in os.listdir(img_source_dir):
        if len(path.split('.')) > 2:
            continue
        if os.path.isfile(f'{img_source_dir}/{path}'):
            shutil.copyfile(f'{img_source_dir}/{path}', f'{img_result_dir}/{path}')


def split_test_into_validation_set(source_dir: str, result_dir: str):
    os.makedirs(f'{result_dir}/images', exist_ok=True)
    os.makedirs(f'{result_dir}/labels', exist_ok=True)
    files = os.listdir(f'{source_dir}/images/')
    number_of_photos_total = len(files)
    validate_photos_number = int(0.75 * number_of_photos_total)
    random.shuffle(files)
    val_filenames = files[:validate_photos_number]
    for filename in val_filenames:
        clipped_filename = filename.split('.')[0]
        if os.path.isfile(f'{source_dir}/images/{filename}'):
            shutil.move(f'{source_dir}/images/{filename}', f'{result_dir}/images/{filename}')
            shutil.move(f'{source_dir}/labels/{clipped_filename}.txt', f'{result_dir}/labels/{clipped_filename}.txt')


