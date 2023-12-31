# Noiseprint: a CNN-based camera model fingerprint
[Noiseprint](https://ieeexplore.ieee.org/document/8713484) is a CNN-based camera model fingerprint
extracted by a fully Convolutional Neural Network (CNN).

## License :page_with_curl:
Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
Modified by Alberto Casagrande (University of Trento).

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) 

## Installation :wrench:
The code requires Python 3.5 and Tensorflow 1.2.1 .
In order to install everything that is needed, create and activate a conda environment through the following command:
```
conda create -n noiseprint python=3.5 pip
conda activate noiseprint
```

### Installation with GPU
Install Cuda8 and Cudnn5, more informetion on sites:
- https://developer.nvidia.com/cuda-downloads
- https://developer.nvidia.com/cudnn

Then install the requested libraries using:
```
cat noiseprint/requirements-gpu.txt | xargs -n 1 -L 1 pip install
```

### Installation without GPU
Install the requested libraries using:
```
cat noiseprint/requirements-cpu.txt | xargs -n 1 -L 1 pip install
```

## Usage :key:
### Dataset 📁
The *DivNoise* dataset can be downloaded at [https://divnoise.fotoverifier.eu/](https://divnoise.fotoverifier.eu/). Part 1 refers to smartphones, tablets and webcams, while the remaining parts contain the Canon cameras. 

When using the *DivNoise* dataset, in order to comply with the code, the downloaded dataset must be slightly modified by splitting the natural images of each camera into separate `Train` and `Test` sets (within the last `JPG` subfolder). Afterwards, simply include the dataset in the `data` folder (`data/Dataset/`) and everything should work.

In case you use your own dataset:
- Firstly, the dataset has to be included in the `data` folder (`data/Dataset/`).
- Within the `Dataset`, each camera's images must be contained in its respective folder, named as `Brand_Model_CameraLocation_ID`, where camera location is either *Frontal* or *Rear*. In particular, they should be divided into separate `Train` and `Test` splits, as explained above.
- The images belonging to a specific camera should have the name in the form *Brand_Model_CamLocation_ID_Content_X.jpg*, where 'Content' identifies the image type (*flat* or *natural*) and 'X' is an incremental number (example: *Apple_iPadmini5_Frontal_0_Nat_0.jpg*)

### Compute the fingerprints of the cameras:
```
python3 compute_fingerprints.py -c dimension_squared_crop_size
```
The paramter `-c` allows to specify the crop size. The noiseprints of the cameras are saved in the `noiseprints` directory (you may need to first create that folder).

### Test the performance of the method:
```
python3 test.py -c dimension_squared_crop_size
```
The charts showing the performance of the method are saved in the `plots` folder (you may need to first create that folder).

## How it works :gear:
### Training
The noiseprint characterizing each device is computed by making the average over the noiseprints of 100 training images belonging to the specific device.

### Testing
A pairwise comparison is performed between the noiseprints of (100) test images per camera and the reference patterns of the cameras using the Euclidean Distance as a similarity metric.
Each test image is assigned to the camera showing the lowest euclidean distance with respect to the noiseprint residual of the test image.

## Author :man_technologist:

**Alberto Casagrande -- Univeristy of Trento**

## Reference

```js
@article{Cozzolino2019_Noiseprint,
  title={Noiseprint: A CNN-Based Camera Model Fingerprint},
  author={D. Cozzolino and L. Verdoliva},
  journal={IEEE Transactions on Information Forensics and Security},
  doi={10.1109/TIFS.2019.2916364},
  pages={144-159},
  year={2020},
  volume={15}
} 
```
The reference code can be found [here](https://github.com/grip-unina/noiseprint)
