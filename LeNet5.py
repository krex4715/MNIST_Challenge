from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
# metrics
from keras.metrics import categorical_crossentropy
# optimization method


def LeNet():
  model = Sequential()

  # Convolutional layer
  model.add(Conv2D(filters = 6, kernel_size = (5,5), padding = 'same',
                   activation = 'relu', input_shape = (28,28,3)))

  # Max-pooing layer with pooling window size is 2x2
  model.add(MaxPooling2D(pool_size = (2,2)))

  # Convolutional layer
  model.add(Conv2D(filters = 16, kernel_size = (5,5), activation = 'relu'))

  # Max-pooling layer
  model.add(MaxPooling2D(pool_size = (2,2)))

  # Flatten layer
  model.add(Flatten())

  # The first fully connected layer
  model.add(Dense(120, activation = 'relu'))

  # The output layer
  model.add(Dense(10, activation = 'softmax'))

  # compile the model with a loss function, a metric and an optimizer function
  # In this case, the loss function is categorical crossentropy,
  # we use Stochastic Gradient Descent (SGD) method with learning rate lr = 0.01
    # to optimize the loss function
  # metric: accuracy

  opt = SGD(lr = 0.01)
  model.compile(loss = categorical_crossentropy,
                optimizer = opt,
                metrics = ['accuracy'])

  return model
