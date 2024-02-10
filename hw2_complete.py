### Add lines to import modules as needed
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, DepthwiseConv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
## 

def build_model1():
    model = models.Sequential()
    # Conv2D: 32 filters, 3x3 kernel, stride=2, "same" padding
    model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    # Conv2D: 64 filters, 3x3 kernel, stride=2, "same" padding
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    # Conv2D: 128 filters, 3x3 kernel, stride=2, "same" padding
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())

    # Four more pairs of Conv2D+Batchnorm, with no striding (defaults to 1)
    for _ in range(4):
        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())

    # MaxPooling: 4x4 pooling size, 4x4 stride
    model.add(layers.MaxPooling2D((4, 4), strides=(4, 4)))
    model.add(layers.Flatten())
    # Dense (Fully Connected): 128 units
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    # Dense (Fully Connected): 10 units
    model.add(layers.Dense(10, activation='softmax'))

    return model

def build_model2():
    model = models.Sequential()
    # First Conv2D layer as specified without depthwise separable convolutions
    model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())

    # Replace the following Conv2D layers with DepthwiseConv2D + Conv2D with 1x1 kernel
    # DepthwiseConv2D: 64 filters, 3x3 kernel, stride=2, "same" padding, no bias
    model.add(layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False))
    # Pointwise convolution (1x1 Conv2D without activation, 64 filters)
    model.add(layers.Conv2D(64, (1, 1), strides=(1, 1), use_bias=True))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # Repeat this pattern for the remaining layers
    for _ in range(4):
        model.add(layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False))
        # Pointwise convolution (1x1 Conv2D with activation, 128 filters)
        model.add(layers.Conv2D(128, (1, 1), use_bias=True))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

    # MaxPooling, Flatten, and Dense layers remain the same
    model.add(layers.MaxPooling2D((4, 4), strides=(4, 4)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))

    return model
def build_model3():
    inputs = Input(shape=(32, 32, 3))
    
    # First Conv-BN block
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Initial block output for residual connection
    residual = x

    # Define the number of blocks and filters
    num_blocks = 4
    filters = [64, 128, 128, 128]

    # Convolutional blocks with dropout and residual connections
    for i in range(num_blocks):
        # Convolutional block
        x = Conv2D(filters[i], (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)  # Dropout rate of 0.3
        x = Activation('relu')(x)

        # If needed, adjust channels of the residual to match x using 1x1 convolution
        if x.shape[-1] != residual.shape[-1]:
            residual = Conv2D(filters[i], (1, 1), strides=(1, 1), padding='same')(residual)

        # Add the residual connection to x every two blocks
        if i % 2 == 1: 
            x = Add()([x, residual])
            # Update residual to the current block's output for the next connection
            residual = x

    # Final layers
    x = MaxPooling2D((4, 4), strides=(4, 4))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(10, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=x, name='model3')

    return model
def build_model50k():
    inputs = Input(shape=(32, 32, 3))
    
    # First Conv2D layer with fewer filters to reduce parameters
    x = Conv2D(16, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Use depthwise separable convolutions
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Repeat the depthwise separable convolution block
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Global average pooling to reduce parameters instead of Flatten()
    x = GlobalAveragePooling2D()(x)
    
    # A single dense layer with fewer units
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=x)
  
    return model
# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
  # Normalize pixel values to be between 0 and 1
  train_images, test_images = train_images / 255.0, test_images / 255.0
  # Split the training set into a training and validation subset
  train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
  )
  ########################################
  ## Build and train model 1
  model1 = build_model1()
  # compile and train model 1.
  model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
  # Train the model
  history1 = model1.fit(train_images, train_labels, epochs=50, 
                     validation_data=(val_images, val_labels))
  
  # Optionally, we can evaluate the model on the test set
  test_loss1, test_acc1 = model1.evaluate(test_images, test_labels, verbose=2)
  #print(f'\nTest accuracy: {test_acc1}')

  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()
  model2.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
  history2 = model2.fit(train_images, train_labels, epochs=50, 
                     validation_data=(val_images, val_labels))
  # Optionally, we can evaluate the model on the test set
  test_loss2, test_acc2 = model3.evaluate(test_images, test_labels, verbose=2)
  #print(f'\nTest accuracy: {test_acc2}')
  ### Repeat for model 3 and your best sub-50k params model
  model3 = build_model3()
  model3.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
  history3 = model3.fit(train_images, train_labels, epochs=50, 
                     validation_data=(val_images, val_labels))
  
  # Optionally, we can evaluate the model on the test set
  #test_loss3, test_acc3 = model3.evaluate(test_images, test_labels, verbose=2)
  #print(f'\nTest accuracy: {test_acc3}')

  model50k = build_model50k()
  model50k.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
  history50k = model50k.fit(train_images, train_labels, epochs=50, 
                     validation_data=(val_images, val_labels))
  
  # Optionally, we can evaluate the model on the test set
  #test_loss50k, test_acc50k = model50k.evaluate(test_images, test_labels, verbose=2)
  model50k.save("/content/best_model.h5")
  #print(f'\nTest accuracy: {test_acc50k}')
  
