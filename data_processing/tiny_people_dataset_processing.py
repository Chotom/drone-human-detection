import json
import os
import random
import shutil
import cv2
from matplotlib import pyplot as plt

ROOT_DIR = 'C:/Users/enhet/Desktop/drone-human-detection-dev'

TRAIN_DIR = f'{ROOT_DIR}/data/source/tiny_people/train'
TEST_DIR = f'{ROOT_DIR}/data/source/tiny_people/test'

TRAIN_PROCESSED_DIR = f'{ROOT_DIR}/data/processed/tiny_people/train'
TEST_PROCESSED_DIR = f'{ROOT_DIR}/data/processed/tiny_people/test'
VALIDATE_PROCESSED_DIR = f'{ROOT_DIR}/data/processed/tiny_people/validate'

TRAIN_ANNO_DIR = f'{TRAIN_DIR}/tiny_set_train.json'
TEST_ANNO_DIR = f'{TEST_DIR}/tiny_set_test.json'

def from_tiny_people_json_to_xywhn_yolo_format(json_source: str, anno_result_dir:str):
    os.makedirs(anno_result_dir, exist_ok=True)
    yolo_cols = ['class', 'x_center', 'y_center', 'width', 'height']
    new_class_id = 0
    file = json.load(open(json_source))
    for image in file["images"]:
        img_file_path = image['file_name']
        img_file = img_file_path.split('/')
        if not img_file[0] != 'labeled_images':
            img_file_name = img_file[1].split('.')[0]
            f = open(f'{anno_result_dir}/{img_file_name}.txt','w')
            # avg_area = get_avg_annotation_area(json_source,image)
            for annotations in file['annotations']:
                if annotations['image_id'] == image['id']:
                    # if annotations['area'] > 1.5*avg_area:
                    #     continue
                    annotation = {}
                    annotation[yolo_cols[0]] = new_class_id
                    annotation[yolo_cols[1]] = annotations['bbox'][0] + annotations['bbox'][2]/2
                    annotation[yolo_cols[2]] = annotations['bbox'][1] + annotations['bbox'][3]/2
                    annotation[yolo_cols[3]] = annotations['bbox'][2]
                    annotation[yolo_cols[4]] = annotations['bbox'][3]
                    area = annotations['area']
                    f.write(f'{annotation[yolo_cols[0]]} {annotation[yolo_cols[1]]} {annotation[yolo_cols[2]]} '
                          f'{annotation[yolo_cols[3]]} {annotation[yolo_cols[4]]} {area} \n')
            f.close()

# cleanup of random bboxes in blurred areas
def clean_up_bboxes_in_blurred_areas(train_path:str, validation_path:str, test_path:str):
    dirs = [train_path,validation_path,test_path]
    for dir in dirs:
        for path in os.listdir(dir):
            file_path = dir+path
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
                avg_area = (total_area/num_of_annos)
                file.close()
                file = open(file_path, 'w')
                for line in lines:
                    clipped_line = line.split(" ")
                    if float(clipped_line[5]) > 2.25 * avg_area:
                        continue
                    file.write(line)
                file.close()



def copy_tiny_people_images_to_given_dir(img_source_dir:str,img_result_dir:str):
    os.makedirs(img_result_dir, exist_ok=True)
    for path in os.listdir(img_source_dir):
        if os.path.isfile(f'{img_source_dir}/{path}'):
            shutil.copyfile(f'{img_source_dir}/{path}',f'{img_result_dir}/{path}')

def split_test_into_validation_set(source_dir: str,result_dir:str):
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

def plot_xywhn_annotated_image_from_file(img_path:str, annotation_path:str):
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    annotation_file = open(annotation_path)
    for annotation in annotation_file.readlines():
        sample = annotation.split(' ')
        sample_w = float(sample[3])
        sample_h = float(sample[4])
        x1, y1 = float(sample[1]) - sample_w / 2, float(sample[2]) - sample_h / 2
        x2, y2 = x1 + sample_w, y1 + sample_h
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    annotation_file.close()
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# from_tiny_people_json_to_xywhn_yolo_format(TEST_ANNO_DIR,f'{TEST_PROCESSED_DIR}/labels')
# copy_tiny_people_images_to_given_dir(f'{TEST_DIR}/labeled_images',f'{TEST_PROCESSED_DIR}/images')

# from_tiny_people_json_to_xywhn_yolo_format(TRAIN_ANNO_DIR,f'{TRAIN_PROCESSED_DIR}/labels')
# copy_tiny_people_images_to_given_dir(f'{TRAIN_DIR}/labeled_images',f'{TRAIN_PROCESSED_DIR}/images')

# split_test_into_validation_set(TEST_PROCESSED_DIR,VALIDATE_PROCESSED_DIR)

# img_path = "C:/Users/enhet/Desktop/drone-human-detection-dev/data/processed/tiny_people/train/images/bb_V0014_I0000120.jpg"
# annotation_path = "C:/Users/enhet/Desktop/drone-human-detection-dev/data/processed/tiny_people/train/labels/bb_V0014_I0000120.txt"

# clean_up_bboxes_in_blurred_areas("C:/Users/enhet/Desktop/drone-human-detection-dev/data/processed/tiny_people/train/labels/","C:/Users/enhet/Desktop/drone-human-detection-dev/data/processed/tiny_people/validate/labels/","C:/Users/enhet/Desktop/drone-human-detection-dev/data/processed/tiny_people/test/labels/")