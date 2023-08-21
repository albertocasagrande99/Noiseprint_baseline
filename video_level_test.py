import argparse
from noiseprint.noiseprint import genNoiseprint
from noiseprint.utility.utilityRead import imread2f
from noiseprint.utility.utilityRead import jpeg_qtableinv
from noiseprint.utility.functions import cut_ctr, crosscorr_2d, pce, stats, gt, aligned_cc
import numpy as np
from glob import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import tqdm
from scipy import spatial

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from sklearn.metrics import roc_curve, auc
from scipy.interpolate import PchipInterpolator
from collections import Counter

fingerprint_devices = os.listdir("data/Dataset/")
fingerprint_devices = sorted(np.unique(fingerprint_devices))
fingerprint_devices.remove('.DS_Store')
device_to_index = {device: index for index, device in enumerate(fingerprint_devices)}
cm = np.zeros((len(fingerprint_devices), len(fingerprint_devices)))

# Dictionary to map devices to colors
device_colors = {
    'Apple_iPhone13_Frontal': 'red',
    'Apple_iPhone13_Rear': 'orange',
    'Apple_iPadmini5_Frontal': 'yellow',
    'Apple_iPadmini5_Rear': 'gold',
    'Huawei_P20Lite_Frontal': 'orchid',
    'Huawei_P20Lite_Rear': 'plum',
    'Motorola_MotoG6Play_Frontal': 'darkmagenta',
    'Motorola_MotoG6Play_Rear': 'darkviolet',
    'Samsung_GalaxyA71_Frontal': 'deepskyblue',
    'Samsung_GalaxyA71_Rear': 'aqua',
    'Samsung_GalaxyTabA_Frontal': 'turquoise',
    'Samsung_GalaxyTabA_Rear': 'teal',
    'Samsung_GalaxyTabS5e_Frontal': 'green',
    'Samsung_GalaxyTabS5e_Rear': 'lime',
    'Sony_XperiaZ5_Frontal': 'gray',
    'Sony_XperiaZ5_Rear': 'black'
}

def load_noiseprints():
    print("Loading noiseprints...")
    k = []
    noiseprints_dirlist = np.array(sorted(glob('noiseprints/*')))
    for noise in noiseprints_dirlist:
        k+=[np.load(noise)]
    return k

def compute_residuals(crop_size):
    for device in fingerprint_devices:
        videos = os.listdir("data/Videos/"+device+"/Videos/VideoLevel/Test/")
        if('.DS_Store' in videos):
            videos.remove('.DS_Store')
        for video in videos:
            nat_device = []
            nat_dirlist = []
            imgs_list = np.array(sorted(glob('data/Videos/' + device + '/Videos/VideoLevel/Test/' + video +'/*.jpg')))
            nat_list = imgs_list
            nat_device_sofar = np.array([os.path.split(i)[1].rsplit('_', 2)[0] for i in nat_list])
            nat_device = np.concatenate((nat_device, nat_device_sofar))
            nat_dirlist = np.concatenate((nat_dirlist,imgs_list))
            
            print('Computing residuals...')
            w=[]
            for img_path in tqdm.tqdm(nat_dirlist):
                img, mode = imread2f(img_path, channel=1)
                img = cut_ctr(img, crop_size)
                try:
                    QF = jpeg_qtableinv(strimgfilenameeam)
                except:
                    QF = 200
                w+=[genNoiseprint(img,QF)]

            gt_ = gt(fingerprint_devices, nat_device)

            euclidean_rot = np.zeros((len(fingerprint_devices), len(nat_device)))
            for fingerprint_idx, fingerprint_k in enumerate(k):
                for natural_idx, natural_w in enumerate(w):
                    dist = np.linalg.norm(fingerprint_k-natural_w)
                    euclidean_rot[fingerprint_idx, natural_idx] = dist
            
            # Count the occurrences of each class
            class_counts = Counter(euclidean_rot.argmin(0))

            # Find the class with the highest count (mode)
            major_class = class_counts.most_common(1)[0][0]

            print("Major class:", fingerprint_devices[major_class])
            true_class_index = device_to_index[device]
            cm[true_class_index,major_class]+=1
            print(cm)

            accuracy_dist = accuracy_score(gt_.argmax(0), euclidean_rot.argmin(0))
            print('Accuracy with Euclidean Distance {:.2f}'.format(accuracy_dist))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Noiseprint extraction", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--crop_size", type=int, action="store", help="Specifies the crop size", required=True)
    args = parser.parse_args()

    crop_size = (args.crop_size, args.crop_size)
    k = load_noiseprints()
    compute_residuals(crop_size)