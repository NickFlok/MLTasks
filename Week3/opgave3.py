import numpy as np
import tensorflow as tf
from tensorflow import keras


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(75, 75)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(39, activation='softmax'))

dataset_training = keras.preprocessing.image_dataset_from_directory(directory="Fundus-data",
                                                                    labels="inferred",
                                                                    label_mode="int",
                                                                    color_mode='grayscale',
                                                                    batch_size=1,
                                                                    image_size=(75, 75),
                                                                    shuffle=True,
                                                                    seed=123,
                                                                    validation_split=0.1,
                                                                    subset="training")

dataset_validation = keras.preprocessing.image_dataset_from_directory(directory="Fundus-data",
                                                                      labels="inferred",
                                                                      label_mode="int",
                                                                      color_mode='grayscale',
                                                                      batch_size=2,
                                                                      image_size=(75, 75),
                                                                      shuffle=True,
                                                                      seed=123,
                                                                      validation_split=0.1,
                                                                      subset="validation")

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(dataset_training, epochs=10, verbose=2)

test = np.concatenate([x for x, y in dataset_validation], axis=0)
pred = np.argmax(model.predict(test), axis=1)
labels = dataset_training.class_names

int_labels = np.concatenate([y for x, y in dataset_validation], axis=0)


cm = tf.math.confusion_matrix(int_labels, pred)

sess = tf.compat.v1.Session()
with sess.as_default():
    data = cm.numpy()

n = len(labels)
tp = np.diagonal(data)

fp = np.empty(n)
for i in range(n):
    sum = 0
    for l in range(n):
        sum += data[l][i]
    fp[i] = sum - tp[i]

fn = np.empty(n)
for i in range(n):
    sum = 0
    for l in range(n):
        sum += data[i][l]
    fn[i] = sum - tp[i]

tn = np.empty(n)
for i in range(n):
    sum = 0
    for l in range(n):
        for k in range(n):
            sum += data[l][k]
    tn[i] = sum - tp[i] - fp[i] - fn[i]

metrics = list(zip(labels, tp, fp, fn, tn))

tp = 0
fp = 0
fn = 0
tn = 0

for i in range(39):
    tp += metrics[i][1]
    fp += metrics[i][2]
    fn += metrics[i][3]
    tn += metrics[i][4]


tpr = tp / (tp + fn)
ppv = tp / (tp + fp)
tnr = tn / (tn + fp)
fpr = fp / (fp + tn)

rv = {'tpr': tpr, 'ppv': ppv, 'tnr': tnr, 'fpr': fpr}

print(rv)
