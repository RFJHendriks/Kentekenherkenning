import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import cv2
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

#karakters zijn 30 bij 60 groot
path_karakters = os.getcwd() + '\karakters\**\*.jpg'
afbeeldingen = []
tags = []

for path_karakter in glob.glob(path_karakters, recursive = True):
    karakter = path_karakter.split("\\")[-1] #alle mappen weghalen
    karakter, _ = os.path.splitext(karakter)
    tag = karakter[0]    #eerste letter selecteren

    tags.append(tag)

    afbeelding = load_img(path_karakter, target_size=(60,30), color_mode="grayscale")
    afbeelding = img_to_array(afbeelding)
    afbeeldingen.append(afbeelding)

X = np.array(afbeeldingen, dtype="float16")
tags = np.array(tags)
X = X.reshape(len(X),-1)

from sklearn.preprocessing import LabelEncoder
print("[INFO] Find {:d} images with {:d} classes".format(len(X),len(set(tags))))

lb = LabelEncoder()
lb.fit(tags)
labels = lb.transform(tags)
y = to_categorical(labels)
np.save('license_character_classes.npy', lb.classes_)

# split 10% of data as validation set
(trainX, testX, trainY, testY) = train_test_split(X, labels, test_size=0.10, stratify=y, random_state=42)

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters = 26)
kmeans.fit(trainX)



def retrieve_info(cluster_labels, y_train):
  reference_labels = {}
  # For loop to run through each label of cluster label
  for i in range(len(np.unique(kmeans.labels_))):
    index = np.where(cluster_labels == i,1,0)
    num = np.bincount(y_train[index==1]).argmax()
    reference_labels[i] = num
  return reference_labels
reference_labels = retrieve_info(kmeans.labels_,trainY)
print(reference_labels)



def calculate_metrics(model,output):
  print('Number of clusters is {}'.format(model.n_clusters))
  print('Inertia : {}'.format(model.inertia_))
  print('Homogeneity : {}'.format(metrics.homogeneity_score(output,model.labels_)))

calculate_metrics(kmeans, trainY)

# total_clusters = len(np.unique(tags))
# print(str(len(data)))
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data, tags, test_size = 0.2, random_state = 42 )
# cv2.imshow("hoi",X_train[0])
# from sklearn.cluster import MiniBatchKMeans
# kmeans = MiniBatchKMeans(n_clusters= total_clusters).fit(X_train)
# kmeans.fit(afbeeldingen)

prediction = kmeans.predict(testX)
number_labels = np.random.rand(len(kmeans.labels_))

for i in range(len(kmeans.labels_)):

  number_labels[i] = reference_labels[kmeans.labels_[i]]

print('Accuracy score : {}'.format(accuracy_score(number_labels,trainY)))




