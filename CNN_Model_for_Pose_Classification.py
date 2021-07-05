import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers.normalization import BatchNormalization
import csv
from tensorflow.keras import optimizers

src_dir = '/content/drive/MyDrive/MultiView_Project/CASIA_B036degree_Centered_Alinged_Pose_Directory_with_length_3/'
save_path = '/content/drive/MyDrive/MultiView_Project'
train_imgs = []
train_labels = []
test_imgs = []
test_labels = []
poses = os.listdir(src_dir)
numberOfPoses = len(subjects)
print('Number of Subjects: ', numberOfPoses)
division = 5 # that define the number of poses, you may change as per requirement
for i in range(1, numberOfPoses + 1):  # numberOfPoses
    if i<10:
        path2 = (src_dir + 'pose0'+str(i) + '/')
    else:
        path2 = (src_dir + 'pose'+str(i) + '/')
    print('path2: ', path2)
    frames = os.listdir(path2)
    numberOfFrames = len(frames)
    print('numberOfFrames: ', numberOfFrames)
    for j in range(0, numberOfFrames-10):
        path3 = path2 + frames[j]
        print(path3 + ' training data')
        img = Image.open(path3)
        #img = img.resize((200, 200))
        #img = img.crop((45, 0, 145, 199))
        img = img.resize((140, 140))
        #img = cv2.imread(path3 , 0)
        x3d = img_to_array(img)
        x = np.expand_dims(x3d[:, :, 0], axis=2)
        train_imgs.append(x)
        label = [0] * numberOfPoses
        label[i-1] = 1
        train_labels.append(label)

    for j in range(numberOfFrames-10, numberOfFrames):
        print('j: ', j)
        path3 = path2 + frames[j]
        print(path3 + ' testing data')
        img = Image.open(path3)
        #img = img.resize((200, 200))
        #img = img.crop((45, 0, 145, 199))
        img = img.resize((140, 140))
        #img = cv2.imread(path3 , 0)
        x3d = img_to_array(img)
        x = np.expand_dims(x3d[:, :, 0], axis=2)
        test_imgs.append(x)
        label = [0] * numberOfPoses
        label[i-1] = 1
        test_labels.append(label)

x_train = np.array(train_imgs)
y_train = np.array(train_labels)

x_test = np.array(test_imgs)
y_test = np.array(test_labels)

save_dir = os.path.join(os.getcwd(), save_path)
model_name = 'GEINet_Model_with_Pose_length_'+str(division)+'.h5'
print(src_dir, 'src_dir')
batch_size = 4
num_classes = numberOfPoses
epochs = 80
#178, 256, 1
model = Sequential()
model.add(Conv2D(8, (5, 5), padding='valid', activation='tanh', input_shape=(140, 140, 1)))
#model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Conv2D(8, (5, 5), activation='tanh', padding='valid'))
#model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Conv2D(8, (5, 5), activation='tanh', padding='valid'))
#model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Conv2D(8, (5, 5), activation='tanh', padding='valid'))
#model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
model.summary()

opt = keras.optimizers.RMSprop(learning_rate=0.001)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
scores = model.evaluate(x_test, y_test, verbose=1)
predict1 = model.predict(x_test, batch_size=batch_size, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('nm')
xx, yy = predict1.shape
print('xx :', xx)
print('yy :', yy)