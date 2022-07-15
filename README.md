# Anomaly Detection in Lidar Data by Combining Supervised and Self-Supervised Methods

This repo is part of the following thesis **[Anomaly Detection in Lidar Data by Combining Supervised and Self-Supervised Methods](https://publikationen.bibliothek.kit.edu/1000147668)**.


## Method Overview

![Overview](/figures/method_flowchart.png)

## Video demonstrations
Sequence 19

[![IMAGE ALT TEXT](https://img.youtube.com/vi/0JhJzvJtQ9o/0.jpg)](https://www.youtube.com/watch?v=0JhJzvJtQ9o "Anomaly Detection in Lidar Data (KITTI Odometry Seq. 19)")

Sequence 11-21

[![IMAGE ALT TEXT](https://img.youtube.com/vi/1MRW6LRDDJQ/0.jpg)](https://www.youtube.com/watch?v=1MRW6LRDDJQ "Anomaly Detection in Lidar Data (KITTI Odometry Seq. 11-21))")

Sequence 00-21

[![IMAGE ALT TEXT](https://img.youtube.com/vi/Dip3UJC09SE/0.jpg)](https://www.youtube.com/watch?v=Dip3UJC09SE "Anomaly Detection in Lidar Data (KITTI Odometry Seq. 00-21)")

## How to use the code
* Clone this repo including the submodules: 
```bash
git clone --recurse-submodules git@github.com:daniel-bogdoll/supervised_unsupervised_anomaly.git
```
* Download the entire **[KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)** except for the greyscale images
* **When using the individual models, a path must be specified for storing the predictions in each case. Always use the same inference folder so that the predictions from each model end up in the same folder and can be easily combined and compared.**

### Supervised Semantic Segmentation (sup_semantic)
* Download and unzip model weights from training with KITTI-360 dataset: **[SalsaNext semantic](https://drive.google.com/file/d/1F96PqSejX_kXAoTa88gDH6NpM5Ze_Lv0/view)**
* Install dependencies as described in the submodule
* Specify the sequences to be inferred in the kitti_odometry_data_cfg.yaml file that came with the model weights
* Infering semantic labels by running ```./eval.sh ``` and specifying some parameters:
    * ```-d ```: Path to the dataset
    * ```-p ```: Path to the main log folder 
    * ```-m ```: Path to the model weights
    * ```-s ```: Select from `[train, valid, test]`
    * ```-c ```: Number of MC samplings
    * ```-e ```: Set to True to additionally get the model uncertainty

    Example:
    ```bash
    ./eval.sh -d PATH_TO_DATASET -p PATH_TO_SAVE -m PATH_TO_MODEL_WEIGHTS  -c 30 -s valid  -e False
    ```

### Supervised Motion Segmentation (sup_mos)
* Download and unzip model weights from training with KITTI-360 dataset: **[SalsaNext motion](https://drive.google.com/file/d/150z3yCYLpwAD6KpdsUbWyOs0GKpLsJVW/view)**
* Use the conda environment of the supervised semantic segmentation model SalsaNext, as it has the same dependencies
* Generate for each scan 8 residual images as a preprocessing steps
    * Set `sequences` and `scan_folder` in `config/data_preparing.yaml`
    + Run `python utils/gen_residual_images_Kitti_odometry.py`
* Infering motion labels
    * Specify the sequences to be inferred in the ```kitti_odometry_data_cfg_mos.yaml``` file that came with the model weights
    * Navigate to SalsaNext model ```cd  mos_SalsaNext/```
    * Infering by running ```./eval.sh ``` and specifying some parameters:
        * ```-d```: Path to the dataset
        * ```-p```: Path to the main log folder 
        * ```-m```: Path to the model weights
        * ```-s```: Select from `[train, valid, test]`
        * ```-c```: Number of MC samplings

        Example:
        ```bash
        ./eval.sh -d PATH_TO_DATASET -p PATH_TO_SAVE -m PATH_TO_MODEL_WEIGHTS  -c 30 -s valid
        ```
* Combine semantic and motion labels and generate semantic motion labels
    * Choose sequences in `config/combine_mos_semantics.yaml`
    * Set `scan_root`, `inference_root`, and `split` in `config/post-processing.yaml`
    * Run `python utils/combine_semantics.py`

### Supervised Ground Segmentation (sup_ground_seg)
* Install dependencies as described in the submodule
* Infering by running `python evaluate_SemanticKITTI.py` and specifying some parameters:
    * ```--config```: Path to config (config/config_kittiSem.yaml)
    * ```--resume```: Path to the checkpoint (trained_models/checkpoint.pth.tar)
    * ```--data_dir```: Path to the dataset
    * ```--logdir```: Path to the main log folder 

    Example:
    ```bash
    python evaluate_SemanticKITTI.py --config .../GndNet/config/config_kittiSem.yaml --resume .../GndNet/trained_models/checkpoint.pth.tar --data_dir data/dataset/sequences --logdir data/inference
    ```

### Self-Supervised Scene Flow (self_scene_flow)
* Install dependencies as described in the submodule
* Download pretrained models by running `bash scripts/download_models.sh`
* Set `test_data_root`, `save_path`, and `sequence` in `configs/test/flowstep3d_self_KITTI_odometry.yaml`
* Infering by running `python run.py -c configs/test/flowstep3d_self_KITTI_odometry.yaml`

### Self-Supervised Odometry (self_odometry)
* Set up the conda environment as described in the submodule (no need for setting up ROS)
* Preprocessing
    * Set `data_path` and `preprocessed_path` in `configs/config_datasets.yaml`
    * Run `python preprocess_data.py`
* Infering 
    * Run `python run_testing.py --checkpoint .../DeLORA_pretrained/kitti_example.pth` where `--checkpoint` indicates where the pretrained model is located
    * The predictions are automatically saved in a newly created folder under `DeLORA/bin/mlruns`
* Now copy the relative transformations (e.g. for sequence 10 the transformation you are looking for ends with `transformations_kitti.npy` and is located in the folder `DeLORA/bin/mlruns`) into the main folder of the inference where the predictions of the other models are located. Create a new folder called `self_pose_estimation`. This way, the predictions of all models will be in the same folder.

### Inference folder after running all the models
```bash
├── ground_removal                          # ground segmentation labels
├── SalsaNext_combined_semantics_mos        # semantic motion labels
├── SalsaNext_mos                           # motion labels
├── SalsaNext_semantics                     # semantic labels
├── scene_flow                              # scene flow predictions
└── self_pose_estimation                    # relative transformations
```

### Anomaly Detection
#### Self-Supervised Motion Labels
* Set `path_dataset`, `path_inference`, and `sequences` in `anomaly_detection/config/config_paths.yaml`
* Run `python anomaly_detection/self_motion_labels/self_motion_labels_2stage.py`

#### Comparison between Supervised and Self-Supervised Models
* Run `python anomaly_detection/compare_and_cluster/compare_labels_and_cluster.py`
* Run `python anomaly_detection/compare_and_cluster/map_anomalies_on_image.py`
