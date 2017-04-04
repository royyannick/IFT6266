import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

folderO = "/home/nick/Documents/data/x_vs_o/circles"
folderX = "/home/nick/Documents/data/x_vs_o/crosses"

N = len(os.listdir(folderO))
dataO = np.empty((N, 32, 32), dtype=np.uint8)
N = len(os.listdir(folderX))
dataX = np.empty((N, 32, 32), dtype=np.uint8)

for i, fpath in enumerate(os.listdir(folderO)):
    print(i, ":", fpath)
    dataO[i] = cv2.resize(cv2.imread(os.path.join(folderO, fpath), cv2.IMREAD_GRAYSCALE), (32,32), interpolation=cv2.INTER_AREA)

for i, fpath in enumerate(os.listdir(folderX)):
    print(i, ":", fpath)
    dataX[i] = cv2.resize(cv2.imread(os.path.join(folderX, fpath), cv2.IMREAD_GRAYSCALE), (32,32), interpolation=cv2.INTER_AREA)

data_x = np.append(dataO, dataX, axis=0)
data_y = np.append(np.zeros(len(dataO)), np.ones(len(dataX)))

f, axes = plt.subplots(2, 2, sharey=True)
axes[0][0].set_axis_off()
axes[0][0].imshow(dataO[100])
axes[1][0].set_axis_off()
axes[1][0].imshow(dataX[100])
axes[0][1].set_axis_off()
if data_y[100] == 0:
    axes[0][1].set_title("Circle")
else:
    axes[0][1].set_title("Cross")
axes[0][1].imshow(data_x[100])
axes[1][1].set_axis_off()
if data_y[1100] == 0:
    axes[1][1].set_title("Circle")
else:
    axes[1][1].set_title("Cross")
axes[1][1].imshow(data_x[1100])