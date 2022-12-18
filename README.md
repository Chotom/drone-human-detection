# Drone-human-detection

A research project to develop a machine learning-based system using computer vision to support human search operations
using a multimodal drone camera.

## Datasets

Processed data is stored in `./data/processed/{dataset_name}` directory and grouped into train, validate and test
datasets.

List of used datasets in the project. Follow instructions in notebooks to prepare data.

| Dataset name              | Type          | Data preparation notebook                                                              | Source                                                                                                                |
|---------------------------|---------------|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| VisDrone Object Detection | RGB           | [visdrone_object_detection](data_processing/notebooks/visdrone_object_detection.ipynb) | [Link - aiskyeye.com](http://aiskyeye.com/download/object-detection-2/)                                               |
| Heridal IPSAR             | RGB           | [heridal_ipsar](data_processing/notebooks/heridal_ipsar.ipynb)                         | [Link - ipsar.fesb.unist.hr](http://ipsar.fesb.unist.hr/HERIDAL%20database.html)                                      |
| Tiny Person               | RGB           | [tiny person](data_processing/notebooks/tiny_person_dataset.ipynb)                     | [Link - github.com/ucas-vg](https://github.com/ucas-vg/PointTinyBenchmark/tree/TinyBenchmark)                         |
| SARD                      | RGB           | [SARD](data_processing/notebooks/sard.ipynb)                                           | [Link - ieee-dataport.org](https://ieee-dataport.org/documents/search-and-rescue-image-dataset-person-detection-sard) |
| HIT-UAV                   | Thermal       | [HIT-UAV](data_processing/notebooks/HITUAV_dataset.ipynb)                              | [Link - github.com/suojiashun](https://github.com/suojiashun/hit-uav-infrared-thermal-dataset)                        |
| ASL ETH FLIR              | Thermal       | [asl_eth_flir](data_processing/notebooks/asl_eth_flir_dataset.ipynb)                   | [Link - projects.asl.ethz.ch](https://projects.asl.ethz.ch/datasets/doku.php?id=ir:iricra2014)                        |
| NII-CU                    | RGB + Thermal | [NII-CU](data_processing/notebooks/NII_CU.ipynb)                                       | [Link - nii-cu-multispectral.org](https://www.nii-cu-multispectral.org/)                                              |
| Private Dataset           | RGB + Thermal | [private_dataset](data_processing/notebooks/private_dataset.ipynb)                     | Unavailable at this moment                                                                                            |

Follow instructions in notebooks to combined above datasets.

| Dataset name         | Data preparation notebook                                                             |
|----------------------|---------------------------------------------------------------------------------------|
| RGB Combined Dataset | [rgb_combined_dataset](data_processing/notebooks/combined/rgb_combined_dataset.ipynb) |
| Thermal Dataset      |
