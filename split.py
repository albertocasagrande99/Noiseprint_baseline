import os
import numpy as np
import random
from glob import glob
import shutil

fingerprint_devices = os.listdir("data/Dataset/")
fingerprint_devices = sorted(np.unique(fingerprint_devices))

fingerprint_devices.remove('.DS_Store')
for device in fingerprint_devices:
    if(device == "Logitech_C920_Frontal_0"):
        isExist = os.path.exists("data/Dataset/" + device + "/Images/Natural/JPG/Train")
        if not isExist:
            imgs = os.listdir("data/Dataset/" + device + "/Images/Natural/JPG/")
            dsExist = os.path.exists("data/Dataset/" + device + "/Images/Natural/JPG/.DS_Store")
            if dsExist:
                imgs.remove('.DS_Store')
            random.shuffle(imgs)

            os.makedirs("data/Dataset/" + device + "/Images/Natural/JPG/Train")
            os.makedirs("data/Dataset/" + device + "/Images/Natural/JPG/Test")
            train_imgs = imgs[:100]
            test_imgs = imgs[100:]

            for img in train_imgs:
                shutil.move(os.path.join("data/Dataset/" + device + "/Images/Natural/JPG/", img), "data/Dataset/" + device + "/Images/Natural/JPG/Train/")

            for img in test_imgs:
                shutil.move(os.path.join("data/Dataset/" + device + "/Images/Natural/JPG/", img), "data/Dataset/" + device + "/Images/Natural/JPG/Test/")