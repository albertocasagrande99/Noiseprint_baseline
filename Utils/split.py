import os
import numpy as np
import random
from glob import glob
import shutil

fingerprint_devices = os.listdir("Dataset_NN")
fingerprint_devices = sorted(np.unique(fingerprint_devices))
fingerprint_devices.remove('.DS_Store')

for device in fingerprint_devices:
    temp_imgs_dir = "Dataset_NN/" + device + "/Images/Natural/JPG/Temp"
    train_imgs_dir = "Dataset_NN/" + device + "/Images/Natural/JPG/Train"
    val_imgs_dir = "Dataset_NN/" + device + "/Images/Natural/JPG/Val"
    test_imgs_dir = "Dataset_NN/" + device + "/Images/Natural/JPG/Test"

    temp_imgs = os.listdir(temp_imgs_dir)
    if '.DS_Store' in temp_imgs:
        temp_imgs.remove('.DS_Store')

    # Shuffle the list of images
    random.shuffle(temp_imgs)

    total_imgs = len(temp_imgs)
    train_count = int(0.7 * total_imgs)
    val_count = int(0.15 * total_imgs)
    test_count = total_imgs - train_count - val_count

    # Create the 'Train', 'Val', and 'Test' directories if they don't exist
    os.makedirs(train_imgs_dir, exist_ok=True)
    os.makedirs(val_imgs_dir, exist_ok=True)
    os.makedirs(test_imgs_dir, exist_ok=True)

    # Move images from Temp folder to Train folder
    for img in temp_imgs[:train_count]:
        src = os.path.join(temp_imgs_dir, img)
        dest = os.path.join(train_imgs_dir, img)
        shutil.move(src, dest)

    # Move images from Temp folder to Val folder
    for img in temp_imgs[train_count:train_count + val_count]:
        src = os.path.join(temp_imgs_dir, img)
        dest = os.path.join(val_imgs_dir, img)
        shutil.move(src, dest)

    # Move remaining images from Temp folder to Test folder
    for img in temp_imgs[train_count + val_count:]:
        src = os.path.join(temp_imgs_dir, img)
        dest = os.path.join(test_imgs_dir, img)
        shutil.move(src, dest)

    # Remove the 'Temp' folder after moving the images
    shutil.rmtree(temp_imgs_dir)