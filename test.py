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

fingerprint_devices = os.listdir("data/Dataset/")
fingerprint_devices = sorted(np.unique(fingerprint_devices))
fingerprint_devices.remove('.DS_Store')

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

nat_device = []
for device in fingerprint_devices:
    #nat_dirlist = np.array(sorted(glob('data/Dataset/' + device + '/Images/Natural/JPG/Test/*.jpg')))[:100]
    nat_dirlist = np.array(sorted(glob('data/Videos/' + device + '/Videos/FrameLevel+/Test/*.jpg')))
    nat_device_sofar = np.array([os.path.split(i)[1].rsplit('_', 2)[0] for i in nat_dirlist])
    nat_device = np.concatenate((nat_device, nat_device_sofar))

nat_dirlist = []
for device in fingerprint_devices:
    #nat_dirlist = np.concatenate((nat_dirlist,np.array(sorted(glob('data/Dataset/' + device + '/Images/Natural/JPG/Test/*.jpg')))[:100]))
    nat_dirlist = np.concatenate((nat_dirlist,np.array(sorted(glob('data/Videos/' + device + '/Videos/FrameLevel+/Test/*.jpg')))))

def load_noiseprints():
    print("Loading noiseprints...")
    k = []
    noiseprints_dirlist = np.array(sorted(glob('noiseprints/*')))
    for noise in noiseprints_dirlist:
        k+=[np.load(noise)]
    return k

def compute_residuals(crop_size):
    print('Computing residuals...')
    # for each device, we extract the images belonging to that device and we compute the corresponding noiseprint, which is saved in the array w
    w=[]
    for img_path in tqdm.tqdm(nat_dirlist):
        img, mode = imread2f(img_path, channel=1)
        img = cut_ctr(img, crop_size)
        try:
            QF = jpeg_qtableinv(strimgfilenameeam)
        except:
            QF = 200
        w+=[genNoiseprint(img,QF)]
    return w

# Horizontal Violin plot with increased font size and coloured violins
def plot_device(fingerprint_device, natural_indices, values, label):
    plt.style.use('default')
    plt.rc('xtick', labelsize=14) 
    plt.rc('ytick', labelsize=12)
    plt.figure(figsize=(8, 8))  # Adjust the values (width, height) as needed
    plt.title(str(fingerprint_device) + "'s fingerprint")
    plt.xlabel('Euclidean distance for query images')

    # Create a dictionary with the natural indices as keys and corresponding values as lists
    data = {}
    for idx, value in zip(natural_indices, values):
        if idx in data:
            data[idx].append(value)
        else:
            data[idx] = [value]

    # Convert the data dictionary to a list of lists
    data_list = [data[idx] for idx in np.unique(natural_indices)]

    # Create the violin plot
    parts = plt.violinplot(data_list, showmeans=True, showmedians=False, vert=False)

    # Set facecolor for violin plots
    for pc, lab in zip(parts['bodies'], np.unique(natural_indices)):
        other = fingerprint_device
        if "Frontal" in fingerprint_device:
            other = other.replace("Frontal", "Rear")
        elif "Rear" in fingerprint_device:
            other = other.replace("Rear", "Frontal")
        if (lab == fingerprint_device):
            pc.set_facecolor('red')
        elif (lab==other):
            pc.set_facecolor("orange")
        else:
            pc.set_facecolor('skyblue')

    # Set x-axis ticks and labels
    unique_indices = np.unique(natural_indices)
    if unique_indices is not None and len(unique_indices) > 0:
        ticks = range(1, len(unique_indices) + 1)
        labels = unique_indices
        plt.yticks(ticks, labels)

        # Set the tick label corresponding to the fingerprint_device to red text color
        for tick, lab in zip(ticks, labels):
            if lab == fingerprint_device:
                plt.gca().get_yticklabels()[tick - 1].set_color('red')

    plt.tight_layout()
    plt.savefig('plots/' + label + '/' + str(fingerprint_device) + '.pdf', format="pdf")
    plt.clf()
    plt.close()


def plot_roc_curve(stats_cc):
    roc_curve_cc = metrics.RocCurveDisplay(fpr=stats_cc['fpr'], tpr=stats_cc['tpr'], roc_auc=stats_cc['auc'], estimator_name='ROC curve')
    plt.style.use('seaborn')
    roc_curve_cc.plot(linestyle='--', color='blue')
    plt.savefig('plots/' +'roc_curve_cc.pdf', format="pdf")
    plt.clf()
    plt.close()

def plot_confusion_matrix(cm, name):
    labels = []
    for elem in fingerprint_devices:
        labels.append(elem[:-2])
    labels_cm = [d.replace('Frontal', 'F').replace('Rear', 'R') for d in labels]

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.style.use('seaborn')
    plt.rcParams.update({'font.size': 20})
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_cm)
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(20,20))
    cax = disp.plot(cmap=plt.cm.Blues, xticks_rotation=90, ax=ax, values_format='.2f')
    plt.grid(False)

    # Access the colorbar from the Axes object
    cax = ax.images[-1].colorbar
    cax.set_clim(0, 1)

    # Increase the font size of x and y ticks
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.tight_layout()

    plt.savefig('plots/'+name, format="pdf", pad_inches=5)
    plt.clf()
    plt.close()

def plot_residuals_2D(w):
    # Flatten noise residuals
    flattened_residuals = [residual.flatten() for residual in w]

    # Apply PCA for dimensionality reduction to 2 components
    #pca = PCA(n_components=2)
    #reduced_residuals = pca.fit_transform(flattened_residuals)
    # Apply t-SNE for dimensionality reduction to 2 components
    tsne = TSNE(n_components=2, random_state=42)
    reduced_residuals = tsne.fit_transform(flattened_residuals)

    # Extract x and y coordinates from reduced residuals
    x = reduced_residuals[:, 0]
    y = reduced_residuals[:, 1]

    # Find and remove the outlier point (in case there is a point far away from the others)
    '''
    outlier_index = find_outlier_index(x, y)
    if outlier_index is not None:
        x = np.delete(x, outlier_index)
        y = np.delete(y, outlier_index)'''

    # Extract the device associated with each noise residual
    devices = [(nat_device[i])[:-2] for i in range(len(nat_device))]
    devices = sorted(devices)
    # Assign colors to the devices
    colors = [device_colors.get(device, 'gray') for device in devices]

    # Get unique devices and their corresponding colors
    unique_devices = list(set(devices))
    unique_devices = sorted(unique_devices)
    unique_colors = [device_colors.get(device, 'gray') for device in unique_devices]

    plt.style.use('seaborn')
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 0.7
    # Create a larger figure
    plt.figure(figsize=(9, 5))  # Adjust the values (width, height) as desired

    # Create scatter plot with different colors for each device
    plt.scatter(x, y, c=colors, s=10)

    labels_legend = [d.replace('Frontal', 'F').replace('Rear', 'R') for d in unique_devices]
    # Create custom legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) for color in unique_colors]
    # Create the legend based on the device colors
    legend = plt.legend(legend_handles, labels_legend, bbox_to_anchor=(1.04, 0.5), loc='center left')

    plt.title('Noise Residuals Scatter Plot')
    # Adjust the layout to make room for the legend
    plt.subplots_adjust(right=0.75)

    # Set the legend to have a tight layout
    plt.tight_layout()

    plt.savefig('plots/TSNE_2D_without_outliers.pdf', format="pdf")
    plt.clf()
    plt.close()

def plot_residuals_2D_convexHull(w):
    # Flatten noise residuals
    flattened_residuals = [residual.flatten() for residual in w]

    # Apply t-SNE for dimensionality reduction to 2 components
    tsne = TSNE(n_components=2, random_state=42)
    reduced_residuals = tsne.fit_transform(flattened_residuals)

    # Extract x and y coordinates from reduced residuals
    x = reduced_residuals[:, 0]
    y = reduced_residuals[:, 1]

    # Extract the device associated with each noise residual
    devices = [(nat_device[i])[:-2] for i in range(len(nat_device))]
    devices = sorted(devices)

    # Assign colors to the devices
    colors = [device_colors.get(device, 'gray') for device in devices]

    # Get unique devices and their corresponding colors
    unique_devices = list(set(devices))
    unique_devices = sorted(unique_devices)
    unique_colors = [device_colors.get(device, 'gray') for device in unique_devices]

    plt.style.use('seaborn')
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 0.7
    # Create a larger figure
    plt.figure(figsize=(9, 5))

    # Create polygons for each device using convex hull excluding outliers
    for device, color in zip(unique_devices, unique_colors):
        device_points = np.array([(xi, yi) for xi, yi, d in zip(x, y, devices) if d == device])
        if len(device_points) > 2:
            # Calculate Z-score for each coordinate
            z_scores = np.abs((device_points - np.mean(device_points, axis=0)) / np.std(device_points, axis=0))
            # Set threshold for outliers (e.g., Z-score > 3)
            threshold = 1.5
            # Exclude outliers based on Z-score threshold
            device_points = device_points[np.all(z_scores < threshold, axis=1)]
            if len(device_points) > 2:
                hull = ConvexHull(device_points)
                polygon = Polygon(device_points[hull.vertices], closed=True, fill=False, edgecolor=color, linewidth=1.8, joinstyle='round')
                plt.gca().add_patch(polygon)

    # Create scatter plot with different colors for each device (excluding outliers)
    plt.scatter(x, y, c=colors, alpha=0.5, s=10)

    labels_legend = [d.replace('Frontal', 'F').replace('Rear', 'R') for d in unique_devices]

    # Create custom legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) for color in unique_colors]
    # Create the legend based on the device colors
    legend = plt.legend(legend_handles, labels_legend, bbox_to_anchor=(1.04, 0.5), loc='center left')

    plt.title('Noise Residuals Scatter Plot')
    plt.tight_layout()

    plt.savefig('plots/TSNE_2D_CH.pdf', format="pdf")
    plt.clf()
    plt.close()

def plot_residuals_3D(w):
    # Flatten noise residuals
    flattened_residuals = [residual.flatten() for residual in w]

    # Apply t-SNE for dimensionality reduction to 3 components
    tsne = TSNE(n_components=3, random_state=42)
    reduced_residuals = tsne.fit_transform(flattened_residuals)

    # Extract x, y, and z coordinates from reduced residuals
    x = reduced_residuals[:, 0]
    y = reduced_residuals[:, 1]
    z = reduced_residuals[:, 2]

    # Extract the device associated with each noise residual
    devices = [(nat_device[i])[:-2] for i in range(len(nat_device))]
    devices = sorted(devices)

    # Assign colors to the devices
    colors = [device_colors.get(device, 'gray') for device in devices]

    # Get unique devices and their corresponding colors
    unique_devices = list(set(devices))
    unique_devices = sorted(unique_devices)
    unique_colors = [device_colors.get(device, 'gray') for device in unique_devices]

    plt.style.use('seaborn')
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 0.7
    # Create a larger figure
    fig = plt.figure(figsize=(8, 5))  # Adjust the values (width, height) as desired

    # Create 3D scatter plot with different colors for each device
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=colors, s=10)

    labels_legend = [d.replace('Frontal', 'F').replace('Rear', 'R') for d in unique_devices]

    # Create custom legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) for color in unique_colors]

    # Create the legend based on the device colors
    legend = plt.legend(legend_handles, labels_legend, bbox_to_anchor=(1.04, 0.5), loc='center left')
    ax.set_title('Noise Residuals Scatter Plot (t-SNE, 3D)')

    # Adjust the layout to make room for the legend
    plt.subplots_adjust(right=0.75)

    # Set the legend to have a tight layout
    plt.tight_layout()

    plt.savefig('plots/TSNE_3D.pdf', format="pdf")
    plt.clf()
    plt.close()

def plot_residuals_2D_without_outliers(w):
    # Flatten noise residuals
    flattened_residuals = [residual.flatten() for residual in w]

    # Apply t-SNE for dimensionality reduction to 2 components
    tsne = TSNE(n_components=2, random_state=42)
    reduced_residuals = tsne.fit_transform(flattened_residuals)

    # Extract x and y coordinates from reduced residuals
    x = reduced_residuals[:, 0]
    y = reduced_residuals[:, 1]

    # Find and remove the outlier point
    for i in range(40):
        outlier_index = find_outlier_index(x, y)
        if outlier_index is not None:
            x = np.delete(x, outlier_index)
            y = np.delete(y, outlier_index)

    # Extract the device associated with each noise residual
    devices = [(nat_device[i])[:-2] for i in range(len(nat_device))]
    devices = sorted(devices)

    # Assign colors to the devices
    colors = [device_colors.get(device, 'gray') for device in devices]

    # Get unique devices and their corresponding colors
    unique_devices = list(set(devices))
    unique_devices = sorted(unique_devices)
    unique_colors = [device_colors.get(device, 'gray') for device in unique_devices]

    plt.style.use('seaborn')
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 0.7
    # Create a larger figure
    plt.figure(figsize=(9, 5))

    # Create polygons for each device using convex hull excluding outliers
    for device, color in zip(unique_devices, unique_colors):
        device_points = np.array([(xi, yi) for xi, yi, d in zip(x, y, devices) if d == device])
        if len(device_points) > 2:
            # Calculate Z-score for each coordinate
            z_scores = np.abs((device_points - np.mean(device_points, axis=0)) / np.std(device_points, axis=0))
            # Set threshold for outliers (e.g., Z-score > 3)
            threshold = 1.4
            # Exclude outliers based on Z-score threshold
            device_points = device_points[np.all(z_scores < threshold, axis=1)]
            if len(device_points) > 2:
                hull = ConvexHull(device_points)
                polygon = Polygon(device_points[hull.vertices], closed=True, fill=False, edgecolor=color, linewidth=1.8, joinstyle='round')
                plt.gca().add_patch(polygon)

    # Create scatter plot with different colors for each device (excluding outliers)
    plt.scatter(x, y, c=colors, alpha=0.5, s=10)

    labels_legend = [d.replace('Frontal', 'F').replace('Rear', 'R') for d in unique_devices]
    # Create custom legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) for color in unique_colors]
    # Create the legend based on the device colors
    legend = plt.legend(legend_handles, labels_legend, bbox_to_anchor=(1.04, 0.5), loc='center left')

    plt.title('Noise Residuals Scatter Plot')
    plt.tight_layout()

    plt.savefig('plots/TSNE_2D_CH_without_outliers.pdf', format="pdf")
    plt.clf()
    plt.close()

def find_outlier_index(x, y):
    # Calculate distance from each point to the mean
    distances = np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2)

    # Calculate Z-score for each distance
    z_scores = np.abs((distances - np.mean(distances)) / np.std(distances))

    # Find the index of the outlier point
    outlier_index = np.argmax(z_scores) if np.max(z_scores) > 3 else None

    return outlier_index

def plot_residuals_3D_without_outliers(w):
    # Flatten noise residuals
    flattened_residuals = [residual.flatten() for residual in w]

    # Apply t-SNE for dimensionality reduction to 3 components
    tsne = TSNE(n_components=3, random_state=42)
    reduced_residuals = tsne.fit_transform(flattened_residuals)

    # Extract x, y, and z coordinates from reduced residuals
    x = reduced_residuals[:, 0]
    y = reduced_residuals[:, 1]
    z = reduced_residuals[:, 2]

    # Extract the device associated with each noise residual
    devices = [(nat_device[i])[:-2] for i in range(len(nat_device))]
    devices = sorted(devices)

    # Assign colors to the devices
    colors = [device_colors.get(device, 'gray') for device in devices]

    # Get unique devices and their corresponding colors
    unique_devices = list(set(devices))
    unique_devices = sorted(unique_devices)
    unique_colors = [device_colors.get(device, 'gray') for device in unique_devices]

    # Find and remove the outlier point
    for i in range(5):
        outlier_index = find_outlier_index_3D(x, y, z)
        if outlier_index is not None:
            mask = np.ones(len(x), dtype=bool)
            mask[outlier_index] = False
            x = x[mask]
            y = y[mask]
            z = z[mask]
            colors = np.array(colors)[mask]

    plt.style.use('seaborn')
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 0.7
    # Create a larger figure
    fig = plt.figure(figsize=(8, 5))  # Adjust the values (width, height) as desired

    # Create 3D scatter plot with different colors for each device
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=colors, s=10)

    labels_legend = [d.replace('Frontal', 'F').replace('Rear', 'R') for d in unique_devices]
    # Create custom legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) for color in unique_colors]

    # Create the legend based on the device colors
    legend = plt.legend(legend_handles, labels_legend, bbox_to_anchor=(1.04, 0.5), loc='center left')
    ax.set_title('Noise Residuals Scatter Plot (t-SNE, 3D)')

    # Adjust the layout to make room for the legend
    plt.subplots_adjust(right=0.75)

    # Set the legend to have a tight layout
    plt.tight_layout()

    plt.savefig('plots/TSNE_3D_without_outliers.pdf', format="pdf")
    plt.clf()
    plt.close()

def find_outlier_index_3D(x, y, z):
    # Calculate distance from each point to the mean
    distances = np.sqrt((x - np.mean(x))**2 + (y - np.mean(y))**2 + (z - np.mean(z))**2)

    # Calculate Z-score for each distance
    z_scores = np.abs((distances - np.mean(distances)) / np.std(distances))

    # Find the index of the outlier point
    outlier_index = np.argmax(z_scores) if np.max(z_scores) > 3 else None

    return outlier_index

def plot_device_circles(w):
    # Flatten noise residuals
    flattened_residuals = [residual.flatten() for residual in w]

    # Apply t-SNE for dimensionality reduction to 2 components
    tsne = TSNE(n_components=2, random_state=42)
    reduced_residuals = tsne.fit_transform(flattened_residuals)

    # Extract x and y coordinates from reduced residuals
    x = reduced_residuals[:, 0]
    y = reduced_residuals[:, 1]

    # Extract the device associated with each noise residual
    devices = [(nat_device[i])[:-2] for i in range(len(nat_device))]
    devices = sorted(devices)

    # Get unique devices
    unique_devices = list(set(devices))
    unique_devices = sorted(unique_devices)

    plt.style.use('seaborn')
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 0.7
    plt.figure(figsize=(9, 5))

    # Iterate over unique devices
    for device in unique_devices:
        # Find the indices of points belonging to the current device
        indices = [i for i in range(len(devices)) if devices[i] == device]
        if len(indices) > 0:
            # Compute the mean x and y values for the current device
            mean_x = np.mean(x[indices])
            mean_y = np.mean(y[indices])

            # Set the range within which to count the points
            range_threshold = 3  # Adjust the range threshold as needed

            # Count the number of points within the range threshold
            num_points = sum((x[i] - mean_x) ** 2 + (y[i] - mean_y) ** 2 <= range_threshold ** 2 for i in indices)        

            # Determine the size of the circle based on the number of points
            circle_size = 80 + (num_points * 10)  # Adjust the scaling factor for size

            # Plot a circle centered at the mean values
            plt.scatter(mean_x, mean_y, marker='o', color=device_colors.get(device, 'gray'), s=circle_size, edgecolor='black')

    # Get unique devices and their corresponding colors
    unique_devices = sorted(unique_devices)
    unique_colors = [device_colors.get(device, 'gray') for device in unique_devices]
    labels_legend = [d.replace('Frontal', 'F').replace('Rear', 'R') for d in unique_devices]

    # Create custom legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) for color in unique_colors]
    # Create the legend based on the device colors
    legend = plt.legend(legend_handles, labels_legend, bbox_to_anchor=(1.04, 0.5), loc='center left')

    plt.title('Mean Noise Residuals')
    plt.tight_layout()
    plt.savefig('plots/Mean_Residuals.pdf', format="pdf")
    plt.clf()
    plt.close()

def plot_roc_curves(confusion_matrix, device_names):
    plt.figure(figsize=(7, 7))
    fprs = []
    tprs = []
    for i, device_name in enumerate(device_names):
        device_confusion_matrix = confusion_matrix[i]
        TP = device_confusion_matrix[i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        TN = np.sum(confusion_matrix) - np.sum(confusion_matrix[i, :]) - np.sum(confusion_matrix[:, i]) + TP
        FN = np.sum(confusion_matrix[i, :]) - TP

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        fpr, tpr, _ = roc_curve([0, 1], [0, 1], pos_label=1)  # Dummy values for initialization
        fpr[1] = FPR
        tpr[1] = TPR

        # Use PchipInterpolator to create smooth curves
        fpr_smooth = np.linspace(0, 1, num=1000)
        tpr_smooth = PchipInterpolator(fpr, tpr)(fpr_smooth)

        #roc_auc = auc(fpr_smooth, tpr_smooth)
        roc_auc = auc(fpr_smooth, tpr_smooth)
        color = device_colors[device_name[:-2]]
        plt.plot(fpr_smooth, tpr_smooth, label="{} (AUC = {:.2f})".format(device_name[:-2].replace('Frontal', 'F').replace('Rear', 'R'), roc_auc), color=color, linewidth=1)

    plt.style.use('seaborn')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Devices')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('plots/ROC_curves.pdf', format="pdf")
    plt.clf()
    plt.close()

def heatmap_residuals(w):
    # Get list of all device names
    devices = sorted(set(nat_device))

    # Extract residuals for each device
    residuals = {d:[] for d in devices}
    for i, res in enumerate(w):
        device = nat_device[i]
        residuals[device].append(res.flatten())

    # Flatten each residual 
    for device in devices:
        residuals[device] = [r.flatten() for r in residuals[device]]

    # Compute distance matrix  
    dist_matrix = np.zeros((len(devices), len(devices)))
    for i in range(len(devices)):
        for j in range(i+1, len(devices)):
            dist_matrix[i,j] = spatial.distance.euclidean(residuals[devices[i]][0], residuals[devices[j]][0])
            dist_matrix[j,i] = dist_matrix[i, j]

    # Generate heatmap 
    plt.figure(figsize=(13, 12))
    heatmap = plt.pcolor(dist_matrix, cmap='viridis')
    plt.rcParams.update({'font.size': 12})
    for y in range(dist_matrix.shape[0]):
        for x in range(dist_matrix.shape[1]):
            value = dist_matrix[y, x]  
            if value == 0:
                color = 'white' 
            else:
                color = 'black' 
            plt.text(x + 0.5, y + 0.5, '%.1f' % value, horizontalalignment='center', verticalalignment='center', color=color)

    # Increase the font size of x and y ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Get list of all device names
    labels_heatmap = []
    for device in devices:
        labels_heatmap.append(device[:-2])
    labels_heatmap = [d.replace('Frontal', 'F').replace('Rear', 'R') for d in labels_heatmap]

    plt.xticks(np.arange(0.5, len(devices) + 0.5), labels_heatmap, rotation=90)
    plt.yticks(np.arange(0.5, len(devices) + 0.5), labels_heatmap)
    plt.colorbar(heatmap)
    plt.tight_layout()
    plt.savefig('plots/heatmap.pdf')

def test(k, w):
    # Computing Ground Truth
    # gt function return a matrix where the number of rows is equal to the number of cameras used for computing the fingerprints, and number of columns equal to the number of natural images
    # True means that the image is taken with the camera of the specific row
    gt_ = gt(fingerprint_devices, nat_device)
    '''
    print('Computing cross correlation')
    cc_aligned_rot = aligned_cc(k, w)['cc']

    print('Computing statistics cross correlation')
    stats_cc = stats(cc_aligned_rot, gt_)
    print('AUC on CC {:.2f}, expected {:.2f}'.format(stats_cc['auc'], 0.98))
    accuracy_cc = accuracy_score(gt_.argmax(0), cc_aligned_rot.argmax(0))
    print('Accuracy CC {:.2f}'.format(accuracy_cc))
    cm_cc = confusion_matrix(gt_.argmax(0), cc_aligned_rot.argmax(0))
    plot_confusion_matrix(cm_cc, "Confusion_matrix_CC.pdf")

    plot_roc_curve(stats_cc)
    '''
    
    print("Computing Euclidean Distance/Cosine similarity...")
    euclidean_rot = np.zeros((len(fingerprint_devices), len(nat_device)))
    cosine_rot = np.zeros((len(fingerprint_devices), len(nat_device)))

    for fingerprint_idx, fingerprint_k in enumerate(k):
        dist_values = []
        natural_indices = []
        for natural_idx, natural_w in enumerate(w):
            dist = np.linalg.norm(fingerprint_k-natural_w)
            cosine_sim = 1 - spatial.distance.cosine(fingerprint_k.flatten(), natural_w.flatten())
            cosine_rot[fingerprint_idx, natural_idx] = cosine_sim
            euclidean_rot[fingerprint_idx, natural_idx] = dist
            dist_values.append(dist)
            natural_indices.append(nat_device[natural_idx][:-2])

        plot_device(fingerprint_devices[fingerprint_idx][:-2], natural_indices, dist_values, "EuclDist")
    
    #plot_residuals_2D(w)
    #plot_residuals_2D_convexHull(w)
    #plot_residuals_3D(w)
    #plot_residuals_2D_without_outliers(w)
    #plot_residuals_3D_without_outliers(w)
    #plot_device_circles(w)
    #heatmap_residuals(w)
    
    accuracy_dist = accuracy_score(gt_.argmax(0), euclidean_rot.argmin(0))
    cm_dist = confusion_matrix(gt_.argmax(0), euclidean_rot.argmin(0))
    print('Accuracy with Euclidean Distance {:.2f}'.format(accuracy_dist))
    plot_confusion_matrix(cm_dist, "Confusion_matrix_Euclidean_Distance.pdf")

    accuracy_cos = accuracy_score(gt_.argmax(0), cosine_rot.argmax(0))
    cm_cosine = confusion_matrix(gt_.argmax(0), cosine_rot.argmax(0))
    print('Accuracy with Cosine similarity {:.2f}'.format(accuracy_cos))
    plot_confusion_matrix(cm_cosine, "Confusion_matrix_Cosine_Similarity.pdf")

    #plot_roc_curves(cm_dist, fingerprint_devices)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Noiseprint extraction", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--crop_size", type=int, action="store", help="Specifies the crop size", required=True)
    args = parser.parse_args()

    crop_size = (args.crop_size, args.crop_size)
    k = load_noiseprints()
    w = compute_residuals(crop_size)
    np.save('256x256_FrameLevel+.npy', w)
    #w = np.load('Noise residuals/Videos/256x256_FrameLevel+.npy')
    test(k, w)