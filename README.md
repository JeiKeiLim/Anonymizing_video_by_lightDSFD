# Anonymizing videos by lightDSFD
By [Jian Li](https://lijiannuist.github.io/)

Modified by [Jongkuk Lim](http://limjk.com?refer=github_lightDSFD)

## History

This repository was forked from [lightDSFD](https://github.com/JeiKeiLim/lightDSFD)

## Introduction
Simple implementation of video anonymization.
If you are looking for a more accurate version, check [Anonymizing videos by DSFD](https://github.com/JeiKeiLim/FaceDetection-DSFD). 
And, if you are looking for a simpler example, [noone video](https://github.com/JeiKeiLim/noone_video) is implemented by only OpenCV examples.

## Comparisons

<img src="https://github.com/JeiKeiLim/mygifcontainer/raw/master/deep_face_detector/compare_01.gif" />

<img src="https://github.com/JeiKeiLim/mygifcontainer/raw/master/deep_face_detector/compare_02.gif" />

<img src="https://github.com/JeiKeiLim/mygifcontainer/raw/master/deep_face_detector/compare_03.gif" />

<img src="https://github.com/JeiKeiLim/mygifcontainer/raw/master/deep_face_detector/compare_04.gif" />

## Requirements
CUDA supported enviornment

- Torch >= 0.3.1
- Torchvision >= 0.2.1
- (Tested on torch 1.3.1 and Torchvision 0.4.2)
- Python 3.6

## Usage
    usage: blur_video.py [-h] [--vertical VERTICAL] [--verbose VERBOSE]
                         [--reduce_scale REDUCE_SCALE] [--rotate ROTATE]
                         [--trained_model TRAINED_MODEL] [--threshold THRESHOLD]
                         [--cuda CUDA] [--widerface_root WIDERFACE_ROOT]
                         file out

## Detailed arguments   

    positional arguments:
      file                  Video file path
      out                   Output video path
      
     optional arguments:
      -h, --help            show this help message and exit
      --vertical VERTICAL   0 : horizontal video(default), 1 : vertical video
      --verbose VERBOSE     Show current progress and remaining time
      --reduce_scale REDUCE_SCALE
                            Reduce scale ratio. ex) 2 = half size of the input.
                            Default : 2
      --rotate ROTATE       Detect faces with rotation. 0 : No rotation, 1 : 90°,
                            2: 90°, 270°, 3 : 90°, 180°, 270°. Default : 0
      --trained_model TRAINED_MODEL
                            Trained state_dict file path to open
      --threshold THRESHOLD
                            Final confidence threshold
      --cuda CUDA           Use cuda to train model
      --widerface_root WIDERFACE_ROOT
                            Location of VOC root directory
