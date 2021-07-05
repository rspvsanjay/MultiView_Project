import numpy as np
import cv2
import os
from keras.models import load_model
from PIL import Image
from keras.preprocessing.image import img_to_array

mpath = '/content/drive/MyDrive/GAIT_IIT_BHU_Analysis/GEINet_Model_with_Pose_length_3.h5'
model = load_model(mpath)
path1 = '/content/drive/MyDrive/CASIA_B036degree_Centered_Alinged/'
save_path = '/content/drive/MyDrive/MultiView_Project/CASIA_B036degree_Centered_Alinged_PEI_3/'
train_imgs = []
train_labels = []
test_imgs = []
test_labels = []
subjects = os.listdir(path1)
numberOfSubject = len(subjects)
print('Number of Subjects: ', numberOfSubject)
poses = 13
for i in range(1, numberOfSubject + 1):  #
    path2 = (path1 + subjects[i - 1] + '/')
    sequences = os.listdir(path2)
    numberOfsequences = len(sequences)
    for j in range(0, numberOfsequences):  # numberOfsequences
        path3 = path2 + sequences[j] + '/'
        frames = os.listdir(path3)
        numberOfFrames = len(frames)
        print(path3)
        pose = []
        for numpose in range(0, poses):
            pose.append(np.zeros((256, 256), dtype=float))

        for k in range(0, numberOfFrames):
            path4 = path3 + frames[k]
            print(path4)
            test_imgs = []
            img = Image.open(path4)
            im_cv = cv2.imread(path4, 0)
            print('im_cv: ', im_cv.shape)
            img = img.resize((140, 140))
            x3d = img_to_array(img)
            x = np.expand_dims(x3d[:, :, 0], axis=2)
            test_imgs.append(x)
            x_test = np.array(test_imgs)
            x_test = x_test.astype('float32')
            x_test /= 255
            predict1 = model.predict(x_test)
            poseNum = np.argmax(predict1) + 1
            print('max index:', poseNum)
            pose[poseNum-1] = pose[poseNum-1] + im_cv

        path11 = save_path + subjects[i - 1] + '/' + sequences[j] + '/'
        try:
            os.makedirs(path11)
        except OSError:
            print("Creation of the directory %s failed" % path11)
        else:
            print("Successfully created the directory %s " % path11)
        for number1 in range(0, poses):
            if number1 < 10:
                path2save = save_path + subjects[i - 1] + '/' + sequences[j] + '/pose0' + str(number1 + 1) + '.png'
            else:
                path2save = save_path + subjects[i - 1] + '/' + sequences[j] + '/pose' + str(number1 + 1) + '.png'
            print(path2save)
            max1 = np.amax(pose[number1], axis=(0, 1))
            print(pose[number1 - 1].max(), ' :max1')
            #pose[number1-1] = ((pose[number1-1] / max1)*255)
            pose[number1 - 1] = (pose[number1 - 1] / pose[number1 - 1].max())*255
            cv2.imwrite(path2save, pose[number1-1])