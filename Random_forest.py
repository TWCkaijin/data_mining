from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import csv
import pandas as pd
import os

DataSet = ""
normal_bias = np.ones((8,2))
filepath = f"{os.getcwd()}/data_set/"

def readfile(fp,uses,mode):
    with open(file = fp+"/"+uses+".csv", mode = 'r',newline='') as f:
        print(fp+"/"+uses+".csv")
        raw_data = csv.reader(f)
        next(raw_data)
        print(uses,mode)
        if(mode == "x"):    
            if uses == "train":
                X_train = []
                print("11111")
                for row in raw_data:
                    X_train.append(list(map(float,row[:-1])))
                return data_cleaning(X_train)
            
            elif uses == 'valid':
                X_test = []
                for row in raw_data:
                    X_test.append(list(map(float,row[:-1])))
                return data_cleaning(X_test)
            
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


def data_cleaning(data)->np.array:
    
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
    for j in [0,len(data[0])-2,len(data[0])-1] if len(data[0])==9 else [0,len(data[0])-1]:
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
    normal_bias[row-1] = [NORMAL,temp_min]
    #print(f"row:{row} max:{temp_max} min:{temp_min} NORMAL:{NORMAL}")
    for j in range(len(temp)):
        data[j][row] = round((temp[j]-temp_min)/NORMAL,4)
    return np.array(data)

if __name__ == '__main__':
    DataSet = str(input("Enter the dataset you want to use(A/B):")).upper()
    filepath+= DataSet   

    X_train, X_test = readfile(filepath,"train","x"), readfile(filepath,"valid","x")
    y_train, y_test = readfile(filepath,"train","y"), readfile(filepath,"valid","y")

    '''
    print(X_train)

    print(y_train)

    print(X_test)

    print(y_test)
    '''

    rf_classifier = RandomForestClassifier()


    rf_classifier.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)