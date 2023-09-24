import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt

#crop_sizes = ['8x8', '16x16', '32x32', '64x64', '128x128', '256x256', '512x512', '720x720']
#accuracy = [0.20, 0.58, 0.75, 0.82, 0.85, 0.87, 0.87, 0.88]
'''
crop_sizes = ['16×16', '32×32', '64×64', '128×128', '256×256', '512×512', '720×720', '1024×1024']
accuracy = [0.07, 0.14, 0.30, 0.54, 0.71, 0.79, 0.79, 0.78]

plt.style.use('bmh')
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=12) 
fig, ax1 = plt.subplots()

# Plotting accuracy with a line plot
ax1.plot(crop_sizes, accuracy, marker='o', linestyle='-', label='Accuracy')
ax1.set_xticklabels(crop_sizes, rotation=90)
# Adding labels and title for the first y-axis

ax1.set_xlabel('Crop Size')
#ax1.set_title('Accuracy and AUC at Different Crop Sizes')

for x, y in zip(crop_sizes, accuracy):
    ax1.text(x, y, f'{y}', ha='center', va='bottom', fontsize=14)

plt.ylim(0,1)

# Handling legends for both lines
lines_1, labels_1 = ax1.get_legend_handles_labels()
ax1.legend(lines_1, labels_1, loc='lower right')
fig.set_size_inches(7, 6)
fig.tight_layout()
# Displaying the plot
plt.savefig("acc_PRNU.pdf", format="pdf")
'''

#Noiseprint - natural vs flat
'''
import matplotlib.pyplot as plt

crop_sizes = ['8×8', '16×16', '32×32', '64×64', '128×128', '256×256', '512×512']
accuracy = [0.22, 0.57, 0.75, 0.82, 0.84, 0.86, 0.87]
new_data = [0.17, 0.44, 0.58, 0.64, 0.66, 0.68, 0.69]

plt.style.use('bmh')
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=12) 
fig, ax1 = plt.subplots()

# Plotting accuracy with a line plot
line1 = ax1.plot(crop_sizes, accuracy, marker='o', linestyle='-', label='Natural')
ax1.set_xticklabels(crop_sizes, rotation=90)

# Adding labels and title for the first y-axis
ax1.set_xlabel('Crop Size')
ax1.set_ylabel('Accuracy')
#ax1.set_title('Accuracy and AUC at Different Crop Sizes')

for x, y in zip(crop_sizes, accuracy):
    ax1.text(x, y, f'{y}', ha='center', va='bottom', fontsize=14)

# Plotting the second line on the twin y-axis
line2 = ax1.plot(crop_sizes, new_data, marker='s', linestyle='-', color='orange', label='Flat')
#ax2.set_ylabel('New Data')

for x, y in zip(crop_sizes, new_data):
    ax1.text(x, y, f'{y}', ha='center', va='bottom', fontsize=14)

# Combine the legends from both axes
lines = line1 + line2
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='lower right')

plt.ylim(0, 1)

fig.set_size_inches(8, 6)
fig.tight_layout()

# Displaying the plot or saving it to a file
plt.savefig("acc_PRNU.pdf", format="pdf")
plt.show()
'''

#Accuracy and loss as a function of the number of epochs - ResNet50
plt.style.use('bmh')
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
# Plotting the accuracy and loss
epochs = range(1, 53)
#accuracy_values = [0.216, 0.541, 0.698, 0.740, 0.814, 0.847, 0.846, 0.872, 0.832, 0.869, 0.874, 0.863, 0.852, 0.903, 0.884, 0.853, 0.882, 0.846, 0.876, 0.906, 0.905, 0.935, 0.893, 0.860, 0.832, 0.869, 0.912, 0.915, 0.940, 0.873, 0.927, 0.924, 0.910, 0.899, 0.891, 0.941, 0.933, 0.948, 0.955, 0.915, 0.950, 0.956, 0.970, 0.959, 0.963, 0.942, 0.970, 0.962, 0.962, 0.937, 0.963]
#loss_values = [2.390, 1.512, 0.907, 0.887, 0.549, 0.515, 0.513, 0.432, 0.524, 0.420, 0.377, 0.449, 0.493, 0.318, 0.412, 0.462, 0.396, 0.624, 0.450, 0.280, 0.318, 0.216, 0.362, 0.468, 0.615, 0.421, 0.275, 0.239, 0.198, 0.459, 0.229, 0.272, 0.350, 0.375, 0.355, 0.193, 0.220, 0.180, 0.130, 0.329, 0.184, 0.115, 0.095, 0.125, 0.101, 0.152, 0.092, 0.102, 0.117, 0.163, 0.114]
accuracy_values = [0.303, 0.495, 0.663, 0.728, 0.767, 0.821, 0.744, 0.850, 0.900, 0.842, 0.860, 0.863, 0.907, 0.860, 0.886, 0.894, 0.873, 0.883, 0.907, 0.900, 0.868, 0.904, 0.912, 0.953, 0.935, 0.938, 0.959, 0.928, 0.969, 0.930, 0.935, 0.966, 0.961, 0.948, 0.951, 0.943, 0.948, 0.953, 0.977, 0.969, 0.977, 0.961, 0.951, 0.969, 0.964, 0.979, 0.979, 0.974, 0.982, 0.964, 0.966, 0.969]
loss_values = [2.000, 1.494, 1.020, 0.766, 0.802, 0.600, 0.747, 0.466, 0.366, 0.475, 0.402, 0.360, 0.298, 0.484, 0.294, 0.340, 0.396, 0.342, 0.272, 0.351, 0.362, 0.309, 0.283, 0.147, 0.202, 0.219, 0.112, 0.234, 0.133, 0.207, 0.193, 0.083, 0.128, 0.147, 0.137, 0.135, 0.175, 0.146, 0.092, 0.076, 0.076, 0.138, 0.155, 0.099, 0.096, 0.053, 0.070, 0.078, 0.050, 0.102, 0.103, 0.127]

plt.figure(figsize=(4.5, 4))
plt.plot(epochs, loss_values, 'ro-', label='Validation Loss', linewidth=1.0)
plt.xlabel('Epoch')
#plt.ylabel('Accuracy')
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('loss_plot.pdf', format="pdf")


#Accuracy and loss together
'''
plt.style.use('bmh')
plt.rc('xtick', labelsize=13) 
plt.rc('ytick', labelsize=13)
epochs = range(1, 52)
accuracy_values = [0.216, 0.541, 0.698, 0.740, 0.814, 0.847, 0.846, 0.872, 0.832, 0.869, 0.874, 0.863, 0.852, 0.903, 0.884, 0.853, 0.882, 0.846, 0.876, 0.906, 0.905, 0.935, 0.893, 0.860, 0.832, 0.869, 0.912, 0.915, 0.940, 0.873, 0.927, 0.924, 0.910, 0.899, 0.891, 0.941, 0.933, 0.948, 0.955, 0.915, 0.950, 0.956, 0.970, 0.959, 0.963, 0.942, 0.970, 0.962, 0.962, 0.937, 0.963]
loss_values = [2.390, 1.512, 0.907, 0.887, 0.549, 0.515, 0.513, 0.432, 0.524, 0.420, 0.377, 0.449, 0.493, 0.318, 0.412, 0.462, 0.396, 0.624, 0.450, 0.280, 0.318, 0.216, 0.362, 0.468, 0.615, 0.421, 0.275, 0.239, 0.198, 0.459, 0.229, 0.272, 0.350, 0.375, 0.355, 0.193, 0.220, 0.180, 0.130, 0.329, 0.184, 0.115, 0.095, 0.125, 0.101, 0.152, 0.092, 0.102, 0.117, 0.163, 0.114]

plt.grid(False)
# Create a figure and axis
fig, ax1 = plt.subplots()

# Plot validation accuracy (in blue)
ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Validation Accuracy', color='b', fontsize=13)
ax1.plot(epochs, accuracy_values, color='b', marker='o', label='Validation Accuracy', linewidth=1.3)
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylim([0, 1])
plt.grid(False)
# Create a second y-axis for validation loss (in red)
ax2 = ax1.twinx()
ax2.set_ylabel('Validation Loss', color='r', fontsize=13)
ax2.plot(epochs, loss_values, color='r', marker='s', label='Validation Loss', linewidth=1.3)
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim([0, max(loss_values) * 1.1])
plt.grid(False)
# Add legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
# Place the combined legend horizontally
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=11)

fig.set_size_inches(8, 5.5)
fig.tight_layout()
plt.savefig("accuracy_loss_plot.pdf", format="pdf")'''