# MIMAMO-Net API
MIMAMO Net: Integrating Micro- and Macro-motion for Video Emotion Recognition

This repository contain all the scripts needed for running MIMAMO-Net on videos.
MIMAMO-Net is a model designed for temporal emotion recognition in the valence 
and arousal space. Valence roughly describes how positive or negative a person
is, whereas arousal describes how active or calm the person is. This simple API
allows you to get valence and arousal predictions at a frame level, from
an input video where at least a human face can be detected.

![alt text](model.png "Architecture of the model.")

This repository is a fork of Didan Deng's original project, which can be found [here](https://github.com/wtomin/MIMAMO-Net) and is based on the following paper.

>Deng, Didan, et al. "MIMAMO Net: Integrating Micro-and Macro-motion for Video Emotion Recognition." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 03. 2020.

***

## Setup
First of all, you need to install and configure [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), which may be painful, although there is plenty of documentation (see
the official repository of the project, or check the `docs/` folder).

Next assuming that you have a conda system, you can run the `setup.sh` script:
```
bash setup.sh
```
This will create a conda environment based on the `.yml` specifications,
download and setup the missing repositories and download the checkpoints of the 
models.

## Usage
There are 2 ways to use MIMAMO-Net for inference: via python (with more
granular control over the parameters and the execution modes); via the simple
`run.sh` bash script (which only requires 2 parameters, and runs as a daemon).

The former option is provided in `api/main.py`, with the following arguments.

```
Facial Emotion Recognition from videos, using MIMAMO-Net.

positional arguments:
  video_path            Path to a video file to process for FER.

optional arguments:
  -h, --help            show this help message and exit

  --out_dir OUT_DIR     A directory where the results will be saved (will be
                        created in setup_dir if None). (default: None)

  --smoothing_size SMOOTHING_SIZE
                        Size (in seconds) of the moving-average smoothing.
                        (default: None)

  --model_path MODEL_PATH
                        Checkpoint of the trained MIMAMO network to use.
                        (default: None)

  --batch_size BATCH_SIZE
                        The number of face snippets to process at once.
                        (default: 1)

  --n_workers N_WORKERS
                        The number of threads to use for data loading.
                        (default: 1)

  --device DEVICE       The device to use for processing and inference.
                        (default: cpu)

  --openface_path OPENFACE_PATH
                        Path to the OpenFace binaries for feature ext.
                        (default: None)

  --multi_face          Whether to enable multi-face recognition. (default:
                        False)

  --face_size FACE_SIZE
                        Size to consider for each face detected. (default: 112)

  --mask                Whether the output face image will be masked (mask the
                        region except the face). Otherwise, the output face
                        image will not contain any background. (default:
                        False)

  --benchmark_path BENCHMARK_PATH
                        Path to the pytorch-benchmarks folder. (default: None)

  --model_name MODEL_NAME
                        ResNet model. (default: resnet50_ferplus_dag)

  --feature_layer FEATURE_LAYER
                        Feature layer name. (default: pool5_7x7_s1)

  --num_phase NUM_PHASE
                        Number of phase difference images to consider[0/1870]
                        (default: 12)

  --phase_size PHASE_SIZE
                        Phase image size, default is 48x48. (default: 48)

  --length LENGTH       The length of snippets returned. (default: 64)

  --stride STRIDE       The stride taken when sampling sequence of snippets.
                        If stride<length, the adjacent sequence will overlap
                        with each other. (default: 64)

  --height HEIGHT       The coefficients levels inc. low- and high pass
                        (default: 4)

  --nbands NBANDS       The number of orientations of bandpass filters.
                        (default: 2)

  --scale_factor SCALE_FACTOR
                        Spatial resolution reduction scale. (default: 2)

  --log                 Enables logging and printing on the console. (default:
                        False)

  --logdir LOGDIR       Path to a directory where logs will be saved.
                        (default: None)

  --keep_tmp            Whether to keep all temporary files after a new vide$
                        has been successfully processed. (default: False)

  --watchdir            Whether to run the script in watch mode. (default:
                        False)

  --watch_interval WATCH_INTERVAL
                        Time interval (in seconds) betwwen checks. (default: 60)

  --video_exts VIDEO_EXTS [VIDEO_EXTS ...]
                        Extension(s) of the video files to process. (default$
                        ['.mp4'])
```

Alternatively, you can directly use the bash script `run.sh` where some arguments
related to the configuration of your dependencies is already provided. Feel free
to edit this file as a configuration file, where only the required dependecies
(e.g. OpenFace path, model checkpoint path, etc.), as well as the optional
parameters of the FER algorithm are hard-coded, hence fixed for future executions.
In this way, only 2 parameters are required: the file system path of the video
to process (or a directory containing videos), and the output directory where
the output will be stored. For example:

```
bash run.sh examples/ processed-videos/
```