import numpy as np
import tensorflow as tf


'''
Feed forward (passing data straight through NN):
input -> weight -> hidden layer 1 (activation function) -> weights -> hidden l 2
(activation function) -> weights -> output layer ->

Backpropogation (It goes backward and manipuilates weights):
compare output to intended output: cost/loss function (cross entropy) ->
optimization function (optimizer): minimize cost (Adam Optimizer....SGD, AdaGrad). 

feed forward + backprop = epoch


softmax = converts the model's linear outputs—logits—to probabilities, which should be easier to interpret 
'''
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 5)

val_loss, val_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', val_acc)
model.save('num_reader.model')
new_model = tf.keras.models.load_model('num_reader.model')
predictions = new_model.predict(x_test)
print(predictions[0])
print(np.argmax(predictions[0]))
print(y_test[0])


# import matplotlib.pyplot as plt
#
# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()

'''
one hot encoding: literally one element is on and the other is off
instead of 1=1 and 2=2, if there are 10 elements 0-9 then 0=[1,0,0,0,0,0,0,0,0,0] and 2=[0,0,2,0,0,0,0,0,0,0]
'''
