import cv2
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


"""## Code (10 pts)"""

img1 = cv2.imread('../input/stop1.jpg')
img2 = cv2.imread('../input/stop2.jpg')

data = loadmat('../input/SIFT_features.mat')
Frame1 = data['Frame1']
Descriptor1 = data['Descriptor1']
Frame2 = data['Frame2']
Descriptor2 = data['Descriptor2']

list_of_distances = []
match_index = []
for i in range(Descriptor1.T.shape[0]):
  for j in range(Descriptor2.T.shape[0]):
    dist = np.sqrt(np.sum((Descriptor1.T[i,:] - Descriptor2.T[j,:])**2))
    list_of_distances.append(dist)
  min_val1 = np.min(list_of_distances)
  ind1 = list_of_distances.index(min_val1, 0, -1)
  list_of_distances.pop(ind1)
  min_val2 = np.min(list_of_distances)
  list_of_distances = []
  match_index.append([ind1, min_val1 ,min_val1/min_val2])
  
X1_d = []
Y1_d = []
X2_d = []
Y2_d = []
X1_r = []
Y1_r = []
X2_r = []
Y2_r = []
thresh_dist = 56
thresh_ratio = 0.84
for i in range(len(match_index)):
  ind1 = i
  ind2 = match_index[i][0]
  dist = match_index[i][1]
  ratio = match_index[i][2]
  x1 = Frame1[0][ind1]
  y1 = Frame1[1][ind1]
  x2 = Frame2[0][ind2]
  y2 = Frame2[1][ind2]
  if dist <= thresh_dist:
    X1_d.append(x1)
    Y1_d.append(y1)
    X2_d.append(x2)
    Y2_d.append(y2)
  if ratio <= thresh_ratio:
    X1_r.append(x1)
    Y1_r.append(y1)
    X2_r.append(x2)
    Y2_r.append(y2)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
vert = img2.shape[0]-img1.shape[0]
horz = img1.shape[1]
error = np.zeros((vert, horz, img1.shape[2])).astype(np.uint8)
img_1 = np.vstack((img1, error))
new_img = np.hstack((img_1, img2))
X2_D = [i + horz for i in X2_d]
X2_R = [j + horz for j in X2_r]

save_dir = '../output/'
try:
    choice = int(input("Please choose the recognition method:\n"
                   "1. Distance Ratio\n"
                   "2. Nearest Neighbors\n"
                   "Your Selection: "))
except ValueError:
    print("Incorrect selection. Please select between 1 and 2")

if choice == 1:
    plt.imshow(new_img)
    for i in range(len(X2_R)):
      X = [X1_r[i], X2_R[i]]
      Y = [Y1_r[i], Y2_r[i]]
      plt.plot(X, Y, color='lime')

    plt.title("Distance Ratio")
    plt.savefig(save_dir + 'distance_ratio.png')
    plt.show()

elif choice == 2:
    plt.imshow(new_img)
    for i in range(len(X2_D)):
      X = [X1_d[i], X2_D[i]]
      Y = [Y1_d[i], Y2_d[i]]
      plt.plot(X, Y, color='lime')

    plt.title("Nearest Neighbor Distance")
    plt.savefig(save_dir + 'nearest_neighbor_distance.png')
    plt.show()

else:
    print("Incorrect selection. Please select between 1 and 2")