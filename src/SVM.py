import pandas as pd
from sklearn import svm
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import joblib
import matplotlib.pyplot as plt
import cv2

Categories = ['0', '1', '2', '3', '4']
data = []
labels = []
datadir = "H:\\Documents\\306\\myData"

# Retrieving the images and their labels
for i in Categories:
    print(f'loading... category : {i}')
    current_path = os.path.join(datadir, i)
    for file in os.listdir(current_path):
        if file[-3:] in {'jpg', 'png'}:
            im = imread(os.path.join(current_path, file))
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            im = resize(im, (100, 100, 3))  # resize
            data.append(im.flatten())
            labels.append(Categories.index(i))
    print(f'loaded category:{i} successfully')

data = np.array(data)
labels = np.array(labels)
df = pd.DataFrame(data)
df['Target'] = labels

print(df)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Splitting training and testing dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.0, random_state=42)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# # functons for RGB to gray and HOG
# from sklearn.base import BaseEstimator, TransformerMixin

# class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
#     """
#     Convert an array of RGB images to grayscale
#     """

#     def __init__(self):
#         pass

#     def fit(self, X, y=None):
#         """returns itself"""
#         return self

#     def transform(self, X, y=None):
#         """perform the transformation and return an array"""
#         return np.array([skimage.color.rgb2gray(img) for img in X])


# class HogTransformer(BaseEstimator, TransformerMixin):
#     """
#     Expects an array of 2d arrays (1 channel images)
#     Calculates hog features for each img
#     """

#     def __init__(self, y=None, orientations=9,
#                  pixels_per_cell=(8, 8),
#                  cells_per_block=(3, 3), block_norm='L2-Hys'):
#         self.y = y
#         self.orientations = orientations
#         self.pixels_per_cell = pixels_per_cell
#         self.cells_per_block = cells_per_block
#         self.block_norm = block_norm

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):

#         def local_hog(X):
#             return hog(X,
#                        orientations=self.orientations,
#                        pixels_per_cell=self.pixels_per_cell,
#                        cells_per_block=self.cells_per_block,
#                        block_norm=self.block_norm)

#         try: # parallel
#             return np.array([local_hog(img) for img in X])
#         except:
#             return np.array([local_hog(img) for img in X])
#             from sklearn.linear_model import SGDClassifier


# # create an instance of each transformer
# grayify = RGB2GrayTransformer()
# hogify = HogTransformer(
#     pixels_per_cell=(14, 14), 
#     cells_per_block=(2,2), 
#     orientations=9, 
#     block_norm='L2-Hys'
# )

# #Use standard scaler to normalise the data
# scalify = MinMaxScaler()

# # call fit_transform on each transform converting X_train step by step
# X_train_gray = grayify.fit_transform(X_train)
# X_train = hogify.fit_transform(X_train)
# X_train = scalify.fit_transform(X_train)

# X_test_gray = grayify.transform(X_test)
# X_test_hog = hogify.transform(X_test)
# X_test = scalify.transform(X_test)

# print(X_train_prepared.shape)

# paramgrid search to find best parameters (commented out)
# param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['linear','rbf','poly']}
# svc=svm.SVC(probability=True)
# print("The training of the model is started, please wait for while as it may take few minutes to complete")
# model=GridSearchCV(svc,param_grid)
# model.fit(X_train_prepared,y_train)
# print('The Model is trained well with the given images')
# print(model.best_params_)

# Train the model with the parameters we found above by gridsearch
svc = svm.SVC(kernel='linear', C=0.1, gamma=0.0001, max_iter=100000, probability=True)
model = svc.fit(X, y)

# Save the model as joblib file.
filename = f"model.joblib"
joblib.dump(model, filename)

# Load the model from saved joblib file.
model_joblib = joblib.load('model.joblib')

# Read one image to predict and test the model.
import glob
import imageio

for file in glob.glob("H:\\P2\\test\\test1_stop.jpg"):
    img = imageio.imread(file)

plt.imshow(img)
plt.show()

print(img.shape)

img = resize(img, (100, 100, 3))
l = [img.flatten()]
probability = model_joblib.predict_proba(l)
