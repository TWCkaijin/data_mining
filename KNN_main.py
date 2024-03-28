import math 
import os
import numpy as np
import csv 
##We're not considering "Overfitting" cases in this code


DataSet = ""
filepath = f"{os.getcwd()}/data_set/"

weight = [1,1,1,1,1,1,1,1]


def readfile(fp,mode)->list:
    with open(file = fp+"/"+mode+".csv", mode = 'r',newline='') as f:
        raw_data = csv.reader(f)
        data = []
        next(raw_data)
        for row in raw_data:
            data.append(list(map(float,row)))
        return data



def neighbor(train,test,k)->list:
    DistanceSet = []
    for i in range(len(train)):
        DistanceSet.append((distance(train[i],test),train[i]))
    DistanceSet.sort(key=lambda x:x[0])
    return DistanceSet[:k]

def distance(point1, point2)->float:
    lenth = 0
    for i in range(len(point1[:-1])):
        lenth += (float(point1[i]) - float(point2[i])) ** 2*weight[i]
    return math.sqrt(lenth)
    
def predict(result)->int:
    cond = {}
    for i in range(len(result)):
        if result[i][1][-1] in cond:
            cond[result[i][1][-1]] += 1
        else:
            cond[result[i][1][-1]] = 1
    return max(cond)

def validation(test_set,k):
    correction=0
    quantity=0
    for data in test_set:
        results = predict(neighbor(readfile(filepath,"train"),data,k))
        print(f"Predicted class: {"無糖尿病"if results == 0 else "有糖尿病"}\tActual class: {"無糖尿病"if data[-1] == 0 else "有糖尿病"}")
        correction += 1 if results == data[-1] else 0
        quantity += 1
    print(f"Accuracy: {correction/quantity*100.0}%")


    
if __name__ == '__main__':
    MODE = str(input("Enter the mode you want to use(1.test\t2.valid):"))
    DataSet = str(input("Enter the dataset you want to use(A/B):")).upper()
    filepath+= DataSet
    k_times = int(input("Enter the number of k:"))    
    if MODE == "1":
        test_data = input("Enter the test data:").split(",")
        results = predict(neighbor(readfile(filepath,"train"),test_data,k_times))
        print(f"Predicted class: {"無糖尿病"if results == 0 else "有糖尿病"}")
    elif MODE == "2":
        validation(readfile(filepath,"valid"),k_times)


