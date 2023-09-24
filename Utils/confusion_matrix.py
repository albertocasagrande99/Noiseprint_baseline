import numpy as np
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Assuming you have a confusion matrix as a NumPy array
'''confusion_matrix = np.array([[100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 99, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                             [1, 2, 91, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                             [4, 2, 0, 87, 3, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                             [4, 7, 14, 2, 51, 1, 3, 1, 2, 2, 1, 7, 0, 2, 1, 2],
                             [8, 5, 1, 4, 6, 57, 1, 2, 3, 2, 1, 1, 0, 4, 1, 4],
                             [4, 5, 3, 1, 1, 3, 63, 4, 0, 2, 3, 2, 3, 2, 1, 3],
                             [1, 2, 3, 1, 3, 1, 2, 77, 2, 5, 0, 1, 1, 0, 0, 1],
                             [2, 4, 3, 1, 6, 1, 4, 3, 55, 6, 4, 2, 4, 2, 0, 3],
                             [6, 5, 4, 0, 6, 1, 2, 1, 2, 59, 0, 10, 1, 1, 2, 0],
                             [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 94, 0, 0, 0, 0, 0],
                             [1, 1, 4, 1, 2, 0, 1, 4, 0, 2, 2, 75, 2, 3, 1, 1],
                             [1, 3, 2, 1, 4, 0, 2, 0, 1, 2, 3, 1, 77, 1, 1, 1],
                             [2, 2, 2, 0, 2, 1, 0, 0, 0, 2, 0, 2, 0, 86, 1, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 98, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100]])'''
'''
confusion_matrix = np.array([[96, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                             [28, 69, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [1, 0, 69, 27, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                             [0, 6, 0, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 2, 0, 0, 3, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 80, 20, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 23, 77, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 97, 0, 0, 2, 0, 0, 0, 1],
                             [0, 0, 1, 0, 0, 0, 0, 0, 0, 89, 10, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 96, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 12, 1, 18, 0, 0, 67, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 1, 1, 18, 0, 0, 0, 77, 0, 0, 3],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 84, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 97]])'''

'''confusion_matrix = np.array([[395, 0, 1, 3, 1],
                             [2, 198, 0, 0, 0],
                             [0, 0, 200, 0, 0],
                             [2, 0, 15, 578, 5],
                             [0, 0, 0, 19, 181]])'''

'''confusion_matrix = np.array([[0.59, 0.06, 0.03, 0.02, 0.02, 0.03, 0.02, 0.09, 0.03, 0.03, 0.00, 0.01, 0.01, 0.01, 0.02, 0.01],
                             [0.05, 0.57, 0.02, 0.00, 0.01, 0.05, 0.05, 0.01, 0.01, 0.02, 0.07, 0.02, 0.05, 0.01, 0.04, 0.01],
                             [0.00, 0.00, 0.74, 0.07, 0.01, 0.02, 0.01, 0.01, 0.03, 0.02, 0.02, 0.01, 0.01, 0.00, 0.02, 0.02],
                             [0.04, 0.01, 0.01, 0.70, 0.00, 0.02, 0.04, 0.03, 0.04, 0.00, 0.01, 0.04, 0.00, 0.01, 0.02, 0.00],
                             [0.01, 0.01, 0.02, 0.01, 0.79, 0.02, 0.00, 0.01, 0.01, 0.02, 0.01, 0.01, 0.00, 0.01, 0.05, 0.00],
                             [0.04, 0.00, 0.01, 0.01, 0.00, 0.68, 0.03, 0.03, 0.03, 0.05, 0.00, 0.03, 0.00, 0.03, 0.01, 0.02],
                             [0.04, 0.00, 0.02, 0.06, 0.00, 0.02, 0.69, 0.01, 0.02, 0.02, 0.00, 0.00, 0.02, 0.04, 0.04, 0.01],
                             [0.02, 0.01, 0.04, 0.01, 0.01, 0.04, 0.01, 0.68, 0.07, 0.03, 0.01, 0.00, 0.03, 0.01, 0.01, 0.00],
                             [0.05, 0.04, 0.01, 0.01, 0.01, 0.06, 0.07, 0.01, 0.58, 0.00, 0.01, 0.02, 0.06, 0.02, 0.02, 0.01],
                             [0.02, 0.04, 0.02, 0.07, 0.00, 0.01, 0.03, 0.02, 0.01, 0.66, 0.00, 0.01, 0.02, 0.04, 0.00, 0.02],
                             [0.04, 0.03, 0.02, 0.04, 0.04, 0.00, 0.02, 0.03, 0.01, 0.01, 0.64, 0.00, 0.03, 0.01, 0.03, 0.01],
                             [0.06, 0.05, 0.05, 0.00, 0.01, 0.03, 0.01, 0.02, 0.01, 0.00, 0.03, 0.67, 0.00, 0.01, 0.03, 0.01],
                             [0.02, 0.04, 0.04, 0.02, 0.00, 0.04, 0.02, 0.03, 0.01, 0.02, 0.02, 0.01, 0.67, 0.03, 0.00, 0.01],
                             [0.01, 0.01, 0.01, 0.03, 0.06, 0.01, 0.06, 0.07, 0.04, 0.01, 0.02, 0.02, 0.03, 0.56, 0.01, 0.03],
                             [0.04, 0.04, 0.04, 0.02, 0.05, 0.01, 0.03, 0.05, 0.06, 0.04, 0.04, 0.02, 0.02, 0.03, 0.51, 0.01],
                             [0.04, 0.02, 0.02, 0.02, 0.02, 0.04, 0.02, 0.08, 0.02, 0.04, 0.02, 0.02, 0.01, 0.03, 0.04, 0.54]])'''

'''confusion_matrix = np.array([[1, 0, 1, 1, 0, 2, 0, 0,],
 [0, 0, 0, 2, 1, 1, 0, 0],
 [0, 0, 4, 0, 1, 0, 0, 0],
 [0, 0, 0, 3, 0, 2, 0, 0],
 [0, 0, 1, 1, 2, 0, 0, 1],
 [0, 0, 0, 0, 0, 4, 1, 0],
 [0, 1, 0, 4, 0, 0, 0, 0],
 [1, 1, 1, 0, 1, 1, 0, 0]])'''

'''confusion_matrix = np.array([[1, 0, 1, 0, 3, 0, 0, 0],
 [0, 2, 0, 1, 2, 0, 0, 0],
 [1, 0, 1, 2, 1, 0, 0, 0],
 [1, 4, 0, 0, 0, 0, 0, 0],
 [0, 1, 0, 2, 1, 0, 1, 0],
 [0, 1, 0, 0, 3, 0, 1, 0],
 [1, 1, 0, 1, 0, 0, 2, 0],
 [1, 0, 0, 1, 1, 0, 0, 2]])'''

#CNN
'''
confusion_matrix = np.array([[96, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [2, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                             [0, 0, 98, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 4, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 2, 98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 96, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 2, 98, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 2, 0, 0, 98, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 96, 0, 0, 0, 2, 0, 2],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 98, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 98, 0, 2, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 98, 0, 0, 2],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 2, 92, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0],
                             [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 98]])'''

#Rear-to-Frontal camera generalization
#PRNU
'''confusion_matrix = np.array([[100, 0, 0, 0, 0, 0, 0, 0],
                             [4, 93, 0, 0, 0, 4, 0, 0],
                             [8, 6, 58, 6, 2, 4, 6, 10],
                             [0, 2, 0, 87, 2, 4, 0, 4],
                             [7, 7, 4, 2, 64, 2, 4, 9],
                             [2, 0, 2, 0, 6, 85, 6, 0],
                             [4, 0, 6, 2, 2, 4, 81, 2],
                             [0, 0, 0, 0, 0, 0, 0, 100]])'''

'''confusion_matrix = np.array([[26, 9, 9, 19, 11, 6, 9, 13],
                             [14, 14, 10, 10, 14, 12, 14, 14],
                             [12, 8, 44, 8, 4, 8, 6, 10],
                             [17, 15, 9, 13, 11, 11, 11, 13],
                             [7, 4, 9, 22, 18, 11, 13, 16],
                             [11, 12, 14, 11, 12, 12, 11, 16],
                             [9, 18, 11, 11, 18, 16, 9, 9],
                             [13, 13, 11, 18, 9, 7, 16, 13]])'''
#Noiseprint
'''confusion_matrix = np.array([[96, 2, 0, 2, 0, 0, 0, 0],
                             [2, 96, 0, 2, 0, 0, 0, 0],
                             [0, 0, 100, 0, 0, 0, 0, 0],
                             [0, 0, 0, 98, 0, 2, 0, 0],
                             [0, 0, 0, 0, 100, 0, 0, 0],
                             [0, 0, 0, 8, 0, 87, 0, 6],
                             [0, 0, 0, 0, 0, 0, 100, 0],
                             [0, 0, 0, 0, 0, 2, 0, 98]])'''

'''confusion_matrix = np.array([[96, 2, 0, 0, 0, 2, 0, 0],
                             [2, 25, 0, 0, 69, 2, 2, 0],
                             [0, 0, 100, 0, 0, 0, 0, 0],
                             [0, 0, 0, 100, 0, 0, 0, 0],
                             [0, 0, 0, 29, 0, 69, 0, 2],
                             [0, 0, 0, 0, 100, 0, 0, 0],
                             [0, 0, 0, 29, 0, 9, 0, 62],
                             [0, 0, 0, 0, 0, 0, 100, 0]])''' 

#CNN                  
'''confusion_matrix = np.array([[100, 0, 0, 0, 0, 0, 0, 0],
                             [0, 98, 2, 0, 0, 0, 0, 0],
                             [0, 0, 100, 0, 0, 0, 0, 0],
                             [0, 0, 0, 98, 0, 0, 0, 2],
                             [0, 0, 0, 0, 98, 2, 0, 0],
                             [0, 0, 0, 0, 2, 96, 2, 0],
                             [0, 0, 0, 0, 6, 2, 92, 0],
                             [0, 0, 0, 0, 0, 0, 0, 100]])'''

confusion_matrix = np.array([[81, 15, 0, 0, 0, 2, 2, 0],
                             [4, 96, 0, 0, 0, 0, 0, 0],
                             [0, 2, 86, 0, 10, 2, 0, 0],
                             [0, 0, 0, 100, 0, 0, 0, 0],
                             [0, 4, 13, 0, 31, 38, 13, 0],
                             [0, 0, 0, 0, 46, 41, 12, 0],
                             [7, 13, 0, 7, 33, 4, 27, 9],
                             [16, 0, 4, 11, 0, 2, 4, 62]])
#class_labels =  ['Apple_iPadmini5', 
# 'Apple_iPhone13', 
# 'Huawei_P20Lite', 
# 'Motorola_MotoG6Play', 
# 'Samsung_GalaxyA71', 
# 'Samsung_TabA', 
# 'Samsung_TabS5e', 
# 'Sony_XperiaZ5']

#class_labels = ['Apple', 'Huawei', 'Motorola', 'Samsung', 'Sony']
class_labels = ['Apple_iPadmini5', 'Apple_iPhone13', 'Huawei_P20Lite', 'Moto_MotoG6Play', 'Samsung_A71', 'Samsung_TabA', 'Samsung_TabS5e', 'Sony_XperiaZ5']
plt.style.use('seaborn')
confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
plt.rcParams.update({'font.size': 11})
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=class_labels)
fig, ax = plt.subplots(figsize=(7, 7))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=90, ax=ax, values_format=None)
# Access the colorbar from the Axes object
cax = ax.images[-1].colorbar
cax.set_clim(0, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.savefig('PRNU_Frontal.pdf', format="pdf", pad_inches=5)
plt.clf()
plt.close()

'''
# Define the mapping of rows to be collapsed
row_mapping = {
    'Apple': [0, 1, 2, 3],
    'Huawei': [4, 5],
    'Motorola': [6, 7],
    'Samsung': [8, 9, 10, 11, 12, 13],
    'Sony': [14, 15]
}

class_labels = ['Apple', 'Huawei', 'Motorola', 'Samsung', 'Sony']

column_mapping = {
    'Apple': [0, 1, 2, 3],
    'Huawei': [4, 5],
    'Motorola': [6, 7],
    'Samsung': [8, 9, 10, 11, 12, 13],
    'Sony': [14, 15]
}

# Create a new confusion matrix with collapsed rows and columns
collapsed_matrix = np.zeros((len(row_mapping), len(column_mapping)))

for new_row_index, old_row_indices in enumerate(row_mapping.values()):
    for new_column_index, old_column_indices in enumerate(column_mapping.values()):
        for old_row_index in old_row_indices:
            for old_column_index in old_column_indices:
                collapsed_matrix[new_row_index, new_column_index] += confusion_matrix[old_row_index, old_column_index]

accuracies = collapsed_matrix/collapsed_matrix.sum(1)
plt.style.use('seaborn')
plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(figsize=(10,8))
cb = ax.imshow(accuracies, cmap=plt.cm.Blues)
plt.xticks(range(len(row_mapping.keys())), row_mapping.keys(),rotation=45)
plt.yticks(range(len(row_mapping.keys())), row_mapping.keys())

for i in range(len(row_mapping.keys())):
    for j in range(len(row_mapping.keys())):
        color='blue' if accuracies[i,j] < 0.5 else 'white'
        ax.annotate(f'{collapsed_matrix[i,j]:.0f}', (j,i), 
                    color=color, va='center', ha='center')
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(False)
plt.colorbar(cb, ax=ax)
plt.tight_layout()
plt.savefig('Collapsed_cm.pdf', format="pdf", pad_inches=5)
'''