from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import os


DataSet = ""
normal_bias = np.ones((8,2))
filepath = f"{os.getcwd()}/data_set/"


def readfile(fp,uses,mode):
    with open(file = fp+"/"+uses+".csv", mode = 'r',newline='') as f:
        raw_data = csv.reader(f)
        next(raw_data)
        if(mode == "x"):    
            if uses == "train":
                X_train = []
                for row in raw_data:
                    X_train.append(list(map(float,row[:-1])))
                return X_train
            
            elif uses == 'valid':
                X_test = []
                for row in raw_data:
                    X_test.append(list(map(float,row[:-1])))
                return X_test
            
        elif(mode == "y"):
            if uses == 'train':
                y_train = []
                for row in raw_data:
                    y_train.append(int(row[-1]))
                return y_train
            
            elif uses == 'valid':
                y_test = []
                for row in raw_data:
                    y_test.append(int(row[-1]))
                return y_test

def data_cleaning(data):
    
    for describe in range(1,len(data[0])-2):
        total = 0
        quantity = 0
        a=[]
        for i in data:
            total += i[describe] if i[describe] != 0 else 0
            quantity += 1 if i[describe] != 0 else 0
        avg = round(total/quantity,2)
        for i in data:
            i[describe]=avg if i[describe] == 0 else i[describe]
        for i in data:
            a.append(i[describe])

        for i in range(len(data)):
            data[i][describe] = a[i]
        temp = []
        for i in data:
            temp.append(round(i[describe],2))
        data = quantilize(np.array(data),avg,describe,temp)
    for j in [0,len(data[0])-2,len(data[0])-1]:
        temp = []
        for i in data:
            temp.append(round(i[j],2))
        data = normalize(data,j,temp)
    return data

def quantilize(data,avg,row,temp)->np.array:
    
    temp = pd.DataFrame(temp)
    Q1 = temp.quantile(0.25).values.tolist()[0]
    Q3 = temp.quantile(0.75).values.tolist()[0]
    IQR = Q3 - Q1
    temp = list(map(lambda x : avg if (x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR)) else x, temp.values.flatten().tolist()))
    return normalize(data,row,temp)

def normalize(data,row,temp)->np.array:
    temp_max = max(temp)
    temp_min = min(temp)
    NORMAL = temp_max - temp_min
    normal_bias[row] = [NORMAL,temp_min]
    for j in range(len(temp)):
        data[j][row] = round((temp[j]-temp_min)/NORMAL,4)
    return np.array(data)

if __name__ == '__main__':

    AX_train, AX_test = readfile(filepath+'A',"train","x"), readfile(filepath+'A',"valid","x")
    Ay_train, Ay_test = readfile(filepath+'A',"train","y"), readfile(filepath+'A',"valid","y")

    BX_train, BX_test = readfile(filepath+'B',"train","x"), readfile(filepath+'B',"valid","x")
    By_train, By_test = readfile(filepath+'B',"train","y"), readfile(filepath+'B',"valid","y")

    Train = data_cleaning(np.concatenate((AX_train,BX_train),axis=0))
    AX_test = data_cleaning(AX_test)
    BX_test = data_cleaning(BX_test)

    test_range = (np.arange(int(input("input the max range")))+1)*10
    x_tag = [str(n) for n in test_range]
    accuracy = []
    Xaccuracy = []

    #print(X_train)
    #print(y_train)
    #print(XX_train)
    #print(Train)
    #print(Train[-1][6])
    #print(yy_train)

    for t in test_range:
        ada_classifier = AdaBoostClassifier(n_estimators=t, algorithm='SAMME', learning_rate=1, random_state=0)
        ada_classifier.fit(Train, Ay_train+By_train)
        Ay_pred = ada_classifier.predict(AX_test)
        By_pred = ada_classifier.predict(BX_test)

        accuracy.append(accuracy_score(Ay_test, Ay_pred))
        Xaccuracy.append(accuracy_score(By_test, By_pred))

    plt.title("Ada boost")
    plt.plot(x_tag,accuracy,label='A score',color='red')
    plt.plot(x_tag,Xaccuracy,label='B score',color='blue')
    plt.xlabel("Ada boost n_estimators")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()