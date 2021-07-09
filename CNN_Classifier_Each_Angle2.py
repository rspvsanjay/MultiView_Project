from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import cv2
import os
from PIL import Image
from keras.layers.normalization import BatchNormalization
import csv
from tensorflow.keras import optimizers

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

result = np.zeros((11, 11))
# angles_gallery = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
angles_gallery = ['090']
# angles_probe = angles_gallery
# angles_probe = ['090']
path1 = '/content/drive/MyDrive/Ganesh/VTGAN/generated/200000/imgs/'
ix = 0
iy = 0
pid = 0

for g_ang in angles_gallery:
    path2 = path1 + g_ang + '/'
    subjects = os.listdir(path2)
    subjectsNumber = len(subjects)

    train_imgs = []
    train_labels = []

    for number1 in range(pid,subjectsNumber):#subjectsNumber
        path3 = path2 + subjects[number1] + '/'        
        print("subject path to train: ", path3)
        for sequence in ['nm-01', 'nm-02', 'nm-03', 'nm-04']:
            path4 = path3 + sequence + '/' + g_ang + '/'            
            poses = os.listdir(path4)
            posesNumber = len(poses)                        
            for number2 in range(0,posesNumber):
                path5 = path4 + poses[number2]                
                img = Image.open(path5)
                img = img.resize((200, 200))
                img = img.crop((45, 0, 145, 199))
                img = img.resize((80, 128))
                x3d = img_to_array(img)
                x = np.expand_dims(x3d[:, :, 0], axis=2)
                train_imgs.append(x)
                label = [0] * subjectsNumber
                label[number1] = 1
                train_labels.append(label)
    
    x_train = np.array(train_imgs)
    y_train = np.array(train_labels)
    batch_size = 4
    num_classes = subjectsNumber
    epochs = 80

    model = Sequential()
    model.add(Conv2D(filters=18, input_shape=(128, 80, 1), kernel_size=(7, 7), strides=1, activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=45, kernel_size=(5, 5), strides=1, activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(BatchNormalization())
    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    x_train = x_train.astype('float32')
    x_train /= 255
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)

    # angles_probe = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
    angles_probe = ['090']
               
    for p_ang in angles_probe:
        path2 = path1 + g_ang + '/'   
        subjects = os.listdir(path2)
        subjectsNumber = len(subjects)
        testy = []
        predy = []
        for number1 in range(pid,subjectsNumber):#subjectsNumber
            path3 = path2 + subjects[number1] + '/'            
            print("subject path to test: ", path3)
            for sequence in ['nm-05', 'nm-06']:
                path4 = path3 + sequence + '/' + p_ang + '/'
                poses = os.listdir(path4)
                posesNumber = len(poses)
                test_imgs = []
                testy.append(number1)
                for number2 in range(0,posesNumber):
                    path5 = path4 + poses[number2]
                    img = Image.open(path5)
                    img = img.resize((200, 200))
                    img = img.crop((45, 0, 145, 199))
                    img = img.resize((80, 128))
                    x3d = img_to_array(img)
                    x = np.expand_dims(x3d[:, :, 0], axis=2)
                    test_imgs.append(x)

                x_test = np.array(test_imgs)
                x_test = x_test.astype('float32')
                x_test /= 255 

                pred = model.predict(x_test, batch_size=batch_size, verbose=0)
                pb = []          
                cl = []
                for number2 in range(0,posesNumber):
                    pb.append(np.amax(pred[number2]))
                    index = np.where(pred[number2] == np.amax(pred[number2]))
                    cl.append(index[0]+pid+1)
                print("cl: ",cl)
                print("pb: ",pb)
                sum_prob = [];
                unique_cl = unique(cl)
                print("unique_cl: ",unique_cl)
                for number2 in range(0,len(unique_cl)):
                    prob = 0
                    for number3 in range(0,posesNumber):
                        if unique_cl[number2]==cl[number3]:
                            prob = prob + pb[number3]
                    sum_prob.append(prob)
                
                maxval = np.amax(sum_prob)
                index = -1
                for num2 in range(0,len(sum_prob)):
                    if sum_prob[num2] == maxval:
                        index = num2
                
                unique_cl = np.asarray(unique_cl)
                predy.append(int(unique_cl[index])) 

        count1 = 0
        for num in range(0,len(testy)):
            if testy[num] == predy[num]:
                count1 = count1+1
        result[ix][iy] = (float(count1)/float(len(predy)))*100
        iy += 1
        if iy==11:
            iy = 0
        print('Accuracy: ', (float(count1)/float(len(predy)))*100, '%')
        print('Prob angle: ', p_ang)
        print('Gallary angle: ', g_ang)
        print('for nm to nm condition or case')
    ix += 1
# print(result)
# print(np.mean(result))
print(np.mean(result, axis=0))
np.savetxt("/content/drive/MyDrive/Ganesh/VTGAN/generated/view_analysis_for_nm2nm.csv", result)