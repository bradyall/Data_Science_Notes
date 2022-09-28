import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# data_dir = '/Users/bropo/Downloads/kagglecatsanddogs_3367a/PetImages'
# categories = ['Dog','Cat']
#
# # for category in categories:
# #     path = os.path.join(data_dir, category)
# #     for img in os.listdir(path):
# #         img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
# #         plt.imshow(img_array, cmap='gray')
# #         plt.show()
# #         break
# #     break
#
# # img_size = 50
# # new_array = cv2.resize(img_array, (img_size,img_size))
# # plt.imshow(new_array, cmap = 'gray')
# # plt.show()
# img_size = 50
# training_data = []
# def create_training_data():
#     for category in categories:
#         path = os.path.join(data_dir, category)
#         class_num = categories.index(category)
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#                 new_array = cv2.resize(img_array, (img_size, img_size))
#                 training_data.append([new_array,class_num])
#             except Exception as e:
#                 pass
#
# create_training_data()
#
# import random
# random.shuffle(training_data)
#
# X = []
# y = []
#
# for features, label in training_data:
#     X.append(features)
#     y.append(label)
#
# X = np.array(X).reshape(-1, img_size, img_size, 1)
#
import pickle
#
# pickle_out = open('X.pickle', 'wb')
# pickle.dump(X, pickle_out)
# pickle_out.close()
#
# pickle_out = open('y.pickle', 'wb')
# pickle.dump(y, pickle_out)
# pickle_out.close()

pickle_in = open('X.pickle','rb')
X = pickle.load(pickle_in)
print(X)


