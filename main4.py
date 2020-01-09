#THIS IS THE TESTING FILE. IT WILL GO ON THE SERVER.


classes = ['Board', 'Dust Pan', 'Fire_Extnguisher','Rubber Glove', 'Apple', 'BackPack', 'Bowl', 'Cap', 'Cell Phone', 'Chair', 'Fan', 'Flashlight', 'Football', 'Glasses', 'Guitar','Laptop', 'Lightbulb', 'Mug', 'Pen', 'Plier', 'Shoe', 'Stapler', 'Wall Clock', 'Water Bottle']



import os
import pickle
from sklearn import cross_validation

import tensorflow as tf
import numpy as np
import re

from tensorflow.python.platform import gfile

#Rescale the testing image to 100x100
# image_data = gfile.FastGFile('football.jpeg', 'rb').read()
# print(image_data)

# image_name = ['football.jpeg']




model_dir = 'imagenet'
images_dir = 'Testing_images/'
list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpeg|JPEG', f)]
# print(list_images)


def create_graph():
    with gfile.FastGFile(os.path.join(
    model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(list_images):
    nb_features = 2048
    features = np.empty((len(list_images), nb_features))
    labels = []

    create_graph()

    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

    for ind, image in enumerate(list_images):
        # print(ind)

        # if (ind % 100 == 0):
        #     print('Processing %s...' % (image))


        image_data = gfile.FastGFile(image, 'rb').read()
        predictions = sess.run(next_to_last_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        features[ind, :] = np.squeeze(predictions)
        labels.append(re.split('_\d+', image.split('/')[1])[0])

    return features, labels



import time
start_time = time.time()


features,labels = extract_features(list_images)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.99)

# print(X_test, y_test)



loaded_model2 =  pickle.load(open('SVC_Classifier_Probabilities.sav', 'rb'))
loaded_model1 =  pickle.load(open('SVC_Classifier.sav', 'rb'))

# image_data = gfile.FastGFile('football.jpeg', 'rb').read()
# print(image_data.dtype)
y_pred = loaded_model1.predict(X_test)
y_pred2 = loaded_model2.predict_proba(X_test)
# print(y_pred)

print("Highest probability is for :", y_pred[0])
# print(y_pred2)
print()

# print(y_pred2[0][0])

print('Top 3 Probabilities are')

print(y_pred2)


# print(y_pred2[0][23])

# y_pred2 = y_pred2.ravel()
# y_pred2 = y_pred2.flatten()

# print(y_pred2)

# print(y_pred2.shape)

# print y_pred2


final_y_pred = []

for i in range (24):
    final_y_pred.append(y_pred2[0][i])

print(final_y_pred)
print(classes)


# print(classes[12])

for i in range(3):
    index = final_y_pred.index(max(final_y_pred))
    # print(index)
    print(classes[index], final_y_pred[index])
    final_y_pred.pop(index)


print("--- %s seconds ---" % (time.time() - start_time))