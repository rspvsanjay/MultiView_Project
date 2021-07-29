from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cv2
import os
import numpy as np

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def finalClass(cl2,pb2,posesNumber):
    sum_prob = [];
    unique_cl = unique(cl2)
    for number2 in range(0,len(unique_cl)):
        prob = 0
        for number3 in range(0,posesNumber):
            if unique_cl[number2]==cl2[number3]:
                prob = prob + pb2[number3]
        sum_prob.append(prob)
    maxval = np.amax(sum_prob)
    index = -1
    for num2 in range(0,len(sum_prob)):
        if sum_prob[num2] == maxval:
            index = num2                  
    unique_cl = np.asarray(unique_cl)
    return int(unique_cl[index]) 
    


result = np.zeros((11, 11))
rank = 20
angles_gallery = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
#angles_gallery = ['072']
angles_probe = angles_gallery
path1 = '/content/drive/MyDrive/Ganesh/VTGAN/generated/200000/imgs/'
ix = 0
iy = 0
for g_ang in angles_gallery:
    path2 = path1 + g_ang + '/'
    print(path2)
    subjects = os.listdir(path2)
    subjectsNumber = len(subjects)
    print(subjectsNumber)
    X = []
    y = []
    pid = 0
    for number1 in range(pid,subjectsNumber):
        path3 = path2 + subjects[number1] + '/'
        for sequence in ['nm-01', 'nm-02', 'nm-03', 'nm-04']:
            path4 = path3 + sequence + '/' + g_ang + '/'
            poses = os.listdir(path4)
            posesNumber = len(poses)            
            for number2 in range(0,posesNumber):
                path5 = path4 + poses[number2]
                # print("image path: ",path5)
                img = cv2.imread(path5, 0)
                img = img.flatten().astype(np.float32)
                X.append(img) 
                y.append(number1+1)

    X = np.asarray(X)
    y = np.asarray(y).astype(np.int32)
    print(X.shape)
    print(y.shape)
    # print(y)
    pca_model = pca.PCA(n_components=int(min(X.shape)*0.2), whiten=False)
    print(int(min(X.shape)*0.20))
    pca_model.fit(X)
    X = pca_model.transform(X)
    # print(X.shape)
    lda_model = LinearDiscriminantAnalysis(n_components=45)
    lda_model.fit(X, y)
    X = lda_model.transform(X)
    
    nbrs = KNeighborsClassifier(n_neighbors=rank*2, weights='distance', metric='euclidean')
    nbrs.fit(X, y)
    print('train : ', X.shape)

    
    angles_probe = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
    #angles_probe = ['090']
    for p_ang in angles_probe:
        path2 = path1 + g_ang + '/'
        # print(path2)
        subjects = os.listdir(path2)
        subjectsNumber = len(subjects)
        print(subjectsNumber)
        
        testy = []
        predy = []
        for number1 in range(pid,subjectsNumber):#
            path3 = path2 + subjects[number1] + '/'
            # print(path3)
            for sequence in ['nm-05', 'nm-06']:
                path4 = path3 + sequence + '/' + p_ang + '/'
                poses = os.listdir(path4)
                posesNumber = len(poses) 
                testX = []
                pb = []          
                cl = []
                testy.append(number1+1)
                for number2 in range(0,posesNumber):
                    path5 = path4 + poses[number2]
                    # print(path5)
                    img = cv2.imread(path5, 0)
                    img = img.flatten().astype(np.float32)
                    testX.append(img) 
                    # testy.append(number1+1)
                testX = np.asarray(testX).astype(np.float32)
                tX = pca_model.transform(testX)
                tX = lda_model.transform(tX)
                pred = nbrs.predict_proba(tX)
                
                for number2 in range(0,posesNumber):
                    temp = sorted(pred[number2],reverse=True)
                    # print("temp: ", temp)
                    cl1 = []
                    pb1 = []
                    for number21 in range(0,rank):
                        index = np.where(pred[number2] == temp[number21])                        
                        index = np.asarray(index)                        
                        cl1.append(index[0][0]+pid+1)
                        pb1.append(temp[number21])                        
                    cl.append(cl1)
                    # print("cl1: ",cl1)
                    pb.append(pb1)
                    # print("pb1: ",pb1)

                predy1 = []
                for number21 in range(0,rank):
                    cl1 = []
                    pb1 = []
                    for number2 in range(0,posesNumber):
                        cl1.append(cl[number2][number21])
                        # print("cl[number21][number2]: ",cl[number21][number2])
                        pb1.append(pb[number2][number21])
                        
                    tcl = finalClass(cl1,pb1,posesNumber)
                    # print("tcl:",tcl)
                    predy1.append(tcl)
                    # print("predy1:",predy1)
                predy.append(predy1)  

        for number21 in range(0,rank):
            count1 = 0
            for num1 in range(0,len(testy)):
                k=0
                for num2 in range(0,number21+1):
                    if k==0:
                        if testy[num1] == predy[num1][num2]:
                            count1 = count1+1
                            k=1
                            break
            print("Accuracy of Rank",str(number21+1),": ",(float(count1)/float(len(predy)))*100) 
            print('Prob angle: ', p_ang)
            print('Gallary angle: ', g_ang)
            print('for nm to nm condition or case')  
#         result[ix][iy] = 
#         iy += 1
#         if iy==11:
#             iy = 0
#         print('Accuracy: ', (float(count1)/float(len(predy)))*100, '%')
        
#     ix += 1
# print(result)
# print(np.mean(result))
# print(np.mean(result, axis=0))
# np.savetxt("/content/drive/MyDrive/Ganesh/VTGAN/generated/view_analysis_for_nm2nm.csv", result)
