import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

folderO = "/Users/yannick/Documents/Playground/Python/data/x_vs_o/circles"
folderX = "/Users/yannick/Documents/Playground/Python/data/x_vs_o/crosses"

N = len(os.listdir(folderO))
dataO = np.empty((N, 116, 116), dtype=np.uint8)
N = len(os.listdir(folderX))
dataX = np.empty((N, 116, 116), dtype=np.uint8)

for i, fpath in enumerate(os.listdir(folderO)):
    print(i, ":", fpath)
    dataO[i] = cv2.imread(os.path.join(folderO, fpath), cv2.IMREAD_GRAYSCALE)

for i, fpath in enumerate(os.listdir(folderX)):
    print(i, ":", fpath)
    dataX[i] = cv2.imread(os.path.join(folderX, fpath), cv2.IMREAD_GRAYSCALE)

x_train = np.append(dataO, dataX, axis=0)
y_train = np.append(np.zeros(len(dataO)), np.ones(len(dataX)))

f, axes = plt.subplots(2, 2, sharey=True)
axes[0][0].set_axis_off()
axes[0][0].imshow(dataO[100])
axes[1][0].set_axis_off()
axes[1][0].imshow(dataX[100])
axes[0][1].set_axis_off()
if y_train[100] == 0:
    axes[0][1].set_title("Circle")
else:
    axes[0][1].set_title("Cross")
axes[0][1].imshow(x_train[100])
axes[1][1].set_axis_off()
if y_train[1100] == 0:
    axes[1][1].set_title("Circle")
else:
    axes[1][1].set_title("Cross")
axes[1][1].imshow(x_train[1100])
