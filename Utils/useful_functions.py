#This file contains useful functions that can be used sometimes

#convex hulls considering all the points (2D scatter plot)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE

def plot_residuals_2D(w):
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
    # Create a larger figure
    plt.figure(figsize=(9, 5))

    # Create scatter plot with different colors for each device
    plt.scatter(x, y, c=colors)

    # Create custom legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) for color in unique_colors]
    # Create the legend based on the device colors
    legend = plt.legend(legend_handles, unique_devices, bbox_to_anchor=(1.04, 0.5), loc='center left')

    # Create polygons for each device using convex hull
    for device, color in zip(unique_devices, unique_colors):
        device_points = np.array([(xi, yi) for xi, yi, d in zip(x, y, devices) if d == device])
        if len(device_points) > 2:
            hull = ConvexHull(device_points)
            polygon = Polygon(device_points[hull.vertices], closed=True, fill=False, edgecolor=color, linewidth=2)
            plt.gca().add_patch(polygon)

    plt.title('Noise Residuals Scatter Plot')
    plt.tight_layout()

    plt.savefig('plots/TSNE_2D.pdf', format="pdf")
    plt.clf()
    plt.close()


#3D scatter plot with convex hulls
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE

def plot_residuals_3D_new(w):
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
    # Create a larger figure
    fig = plt.figure(figsize=(8, 5))  # Adjust the values (width, height) as desired

    # Create 3D scatter plot with different colors for each device
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=colors, alpha=0.5, s=10)

    # Create convex hulls for each device
    for device, color in zip(unique_devices, unique_colors):
        device_points = np.array([(xi, yi, zi) for xi, yi, zi, d in zip(x, y, z, devices) if d == device])
        if len(device_points) > 2:
            hull = ConvexHull(device_points)
            hull_vertices = device_points[hull.vertices]
            polygon = Poly3DCollection([hull_vertices], alpha=0.2, linewidths=1, edgecolors=color)
            ax.add_collection3d(polygon)

    # Create custom legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) for color in unique_colors]

    # Create the legend based on the device colors
    legend = plt.legend(legend_handles, unique_devices, bbox_to_anchor=(1.04, 0.5), loc='center left')

    ax.set_title('Noise Residuals Scatter Plot (t-SNE, 3D)')

    # Adjust the layout to make room for the legend
    plt.subplots_adjust(right=0.75)

    # Set the legend to have a tight layout
    plt.tight_layout()

    plt.savefig('plots/TSNE_3D.pdf', format="pdf")
    plt.clf()
    plt.close()

# Bar chart
def plot_device(fingerprint_device, natural_indices, values, label):
    avgResult = []
    start_index = 0

    for size in n_values:
        end_index = start_index + size
        chunk = values[start_index:end_index]
        avg_result = np.average(chunk)
        avgResult.append(avg_result)
        start_index = end_index

    plt.figure(figsize=(15, 10))  # Adjust the values (width, height) as needed

    plt.title('Noiseprint for ' + str(fingerprint_device))
    plt.xlabel('query images')
    plt.ylabel(label)
    bars = plt.bar(np.unique(natural_indices), avgResult)

    # Adding value labels to the bars
    for bar in bars:
        height = bar.get_height()
        value = "{:.2f}".format(height)
        plt.text(bar.get_x() + bar.get_width() / 2, height, value, ha='center', va='bottom')

    plt.xticks(np.unique(natural_indices), rotation=90)
    plt.tight_layout()
    plt.savefig('plots/'+ label + '/' +str(fingerprint_device)+'.png')

    plt.clf()
    plt.close()

# Vertical violin plot
def plot_device(fingerprint_device, natural_indices, values, label):
    plt.style.use('default')
    plt.figure(figsize=(13, 8))  # Adjust the values (width, height) as needed
    plt.title(str(fingerprint_device) + "'s fingerprint")
    plt.xlabel('query images')
    plt.ylabel(label)

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
    plt.violinplot(data_list, showmeans=True, showmedians=False)

    # Set x-axis ticks and labels
    unique_indices = np.unique(natural_indices)
    if unique_indices is not None and len(unique_indices) > 0:
        ticks = range(1, len(unique_indices) + 1)
        labels = unique_indices
        plt.xticks(ticks, labels, rotation=90)
        # Set the tick label corresponding to the fingerprint_device to red text color
        for tick, lab in zip(ticks, labels):
            if lab == fingerprint_device:
                plt.gca().get_xticklabels()[tick - 1].set_color('red')

    plt.tight_layout()
    plt.savefig('plots/' + label + '/' + str(fingerprint_device) + '.pdf', format="pdf")

    plt.clf()
    plt.close()