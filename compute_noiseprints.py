# This is the code to extract Noiseprint
#    python main_extraction.py input.png noiseprint.mat
#    python main_showout.py input.png noiseprint.mat
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

import argparse
import tqdm
from noiseprint.noiseprint import genNoiseprint
from noiseprint.utility.utilityRead import imread2f
from noiseprint.utility.utilityRead import jpeg_qtableinv
from noiseprint.utility.functions import cut_ctr
import numpy as np
import glob
from random import randrange
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#Our adaptation
fingerprint_devices = os.listdir("data/Dataset/")
fingerprint_devices = sorted(np.unique(fingerprint_devices))
fingerprint_devices.remove('.DS_Store')

def compute_noiseprints(crop_size):
    print('Computing fingerprints...')
    # for each device, we extract the images belonging to that device and we compute the corresponding noiseprint, which is saved in the array k
    for device in fingerprint_devices:
        print('Computing fingerprint of device: ' + device + '...')
        noises = []
        #ff_dirlist = np.array(sorted(glob.glob('data/Dataset/' + device + '/Images/Flat/JPG/*.jpg')))
        #ff_dirlist = np.array(sorted(glob.glob('data/Dataset/' + device + '/Images/Natural/JPG/Train/*.jpg')))
        ff_dirlist = np.array(sorted(glob.glob('data/Videos/' + device + '/Videos/VideoLevel/Train/*.jpg')))
        for img_path in tqdm.tqdm(ff_dirlist):
            img, mode = imread2f(img_path, channel=1)
            img = cut_ctr(img, crop_size)
            try:
                QF = jpeg_qtableinv(strimgfilenameeam)
            except:
                QF = 200
            res = genNoiseprint(img,QF)
            noises.append(res)
        fingerprint = np.average(noises, axis=0)
        np.save("noiseprints/fingerprint_"+device+".npy", fingerprint)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Noiseprint extraction", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--crop_size", type=int, action="store", help="Specifies the crop size", required=True)
    args = parser.parse_args()

    crop_size = (args.crop_size, args.crop_size)
    compute_noiseprints(crop_size)