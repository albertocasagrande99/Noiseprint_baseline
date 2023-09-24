import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Path to the main dataset folder
dataset_folder = 'Dataset_NN'

# Set a custom color palette
sns.set_palette("Blues")

# Get a list of class folders
class_folders = [folder for folder in sorted(os.listdir(dataset_folder)) if os.path.isdir(os.path.join(dataset_folder, folder))]

# Initialize lists to store class names and image counts
class_names = []
train_counts = []
val_counts = []
test_counts = []

# Iterate through class folders
for class_folder in class_folders:
    class_path = os.path.join(dataset_folder, class_folder)
    
    # Count images in 'Train', 'Val', and 'Test' folders
    train_images = len(os.listdir(os.path.join(class_path, 'Images/Natural/JPG/Train')))
    val_images = len(os.listdir(os.path.join(class_path, 'Images/Natural/JPG/Val')))
    test_images = len(os.listdir(os.path.join(class_path, 'Images/Natural/JPG/Test')))
    
    class_names.append(class_folder)
    train_counts.append(train_images)
    val_counts.append(val_images)
    test_counts.append(test_images)

labels = []
for elem in class_names:
    labels.append(elem[:-2])
labels = [d.replace('Frontal', 'F').replace('Rear', 'R') for d in labels]
labels = [d.replace('GalaxyTab', 'Tab') for d in labels]
plt.rcParams.update({'font.size': 14})
# Create a bar plot
plt.figure(figsize=(12, 8.5))
plt.bar(labels, train_counts, label='Train', capstyle='round')
plt.bar(labels, val_counts, bottom=train_counts, label='Val', capstyle='round')
plt.bar(labels, test_counts, bottom=[train + val for train, val in zip(train_counts, val_counts)], label='Test', capstyle='round')
plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.title('Dataset Distribution')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3)  # Adjust the position as needed
plt.xticks(rotation=45, ha="right")

# Display the plot
plt.tight_layout()
plt.savefig('data_distribution.pdf', format="pdf")