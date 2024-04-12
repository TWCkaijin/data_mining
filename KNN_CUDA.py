import math 
import os
import csv 
import pandas as pd
import numpy as np
import numba as nb
from numba.typed import List
##We're not considering "Overfitting" cases in this code

class ColorFill:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    END = "\033[0m"





DataSet = ""
filepath = f"{os.getcwd()}/data_set/"
y_true = np.array([])
weight = np.zeros(8)  #8
normal_bias = np.ones((8,2))  #6
acc_list = List()  #Accuracy list
weight_mem = List()  #Weight memory


def readfile(fp,mode):
    global y_true
    with open(file = fp+"/"+mode+".csv", mode = 'r',newline='') as f:
        raw_data = csv.reader(f)
        data = []
        next(raw_data)
        if(mode == "train"):    
            for row in raw_data:
                data.append(list(map(float,row)))
        elif(mode == "valid"):
            for row in raw_data:
                data.append(list(map(float,row[:-1])))
                y_true=np.append(y_true,int(row[-1]))

        return data

def data_cleaning(data):
    avg = float()
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

    return data,avg

def deliver(avg,data):
    print(data)
    for j in [0,len(data[0])-2,len(data[0])-1] if len(data[0])==9 else [0,len(data[0])-1]:
        temp = List()
        for i in data:
            temp.append(round(i[j],2))
        data = normalize(data,j,temp)

    for describe in range(1,len(data[0])-2): 
        data = quantilize(np.array(data),avg,describe,temp)

    return data


def quantilize(data,avg,row,temp):
    
    temp = pd.DataFrame(temp)
    Q1 = temp.quantile(0.25).values.tolist()[0]
    Q3 = temp.quantile(0.75).values.tolist()[0]
    IQR = Q3 - Q1
    temp = list(map(lambda x : avg if (x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR)) else x, temp.values.flatten().tolist()))
    return normalize(data,row,temp)

def normalize(data,row,temp):
    temp_max = max(temp)
    temp_min = min(temp)
    NORMAL = temp_max - temp_min
    normal_bias[row-1] = [NORMAL,temp_min]
    #print(f"row:{row} max:{temp_max} min:{temp_min} NORMAL:{NORMAL}")
    for j in range(len(temp)):
        data[j][row] = round((temp[j]-temp_min)/NORMAL,4)
    return np.array(data)

@nb.jit()
def neighbor(train,test,k,weight):
    DistanceSet = List([[]])
    for i in range(len(train)):
        length = 0
        for j in range(len(train[i][:-1])):
            length += (float(train[i][j]) - float(test[j])) ** 2 * weight[i]
        length = math.sqrt(length)
        DistanceSet.append([length,train[i]])
    DistanceSet.sort(key=lambda x:x[0])
    a = DistanceSet[:k]
    return a

    
def outcome(result)->int:
    cond = {}
    for i in range(len(result)):
        if result[i][1][-1] in cond:
            cond[result[i][1][-1]] += 1
        else:
            cond[result[i][1][-1]] = 1


    #print(f"cond:{cond}")
    return max(cond)

def validation(k,epochs):
    global acc_list
    global weight
    global y_true
    global weight_mem
    print("IN")
    train_data= readfile(filepath,"train")
    print(train_data)
    train_data,avg = data_cleaning(train_data)
    deliver(avg,train_data)
    ValidData= readfile(filepath,"valid")
    ValidData,avg = data_cleaning(ValidData)
    deliver(avg,ValidData)
    print("IN")
    for basic_bias in range(5):
        print(f"{ColorFill.BLUE}Basic_bias = {basic_bias}{ColorFill.END}")
        weight = [basic_bias,basic_bias,basic_bias,basic_bias,basic_bias,basic_bias,basic_bias,basic_bias]
        for argument in range(len(weight)):
            print(f"{ColorFill.GREEN}arg={argument}{ColorFill.END}")
            try:
                #print(acc_list)
                weight[argument-1] = weight_mem[basic_bias][argument-1][acc_list[basic_bias][argument-1].index(max(acc_list[basic_bias][argument-1]))]
            except Exception as e:
                print(e)
            acc = train_weights(k,epochs,train_data,ValidData)
            weight[argument] += 0.05
            try:
                acc_list[basic_bias][argument].append(acc)
                weight_mem[basic_bias][argument].append(weight[argument])
            except:
                try:
                    acc_list[basic_bias].append([acc])
                    weight_mem[basic_bias].append([weight[argument]])
                except:
                    acc_list.append([[acc]])
                    weight_mem.append([[weight[argument]]])
            

def train_weights(k,epochs,train_data,ValidData)->float:  #->double:
    global weight
    for epoch in range(1,epochs+1):
        correction=0
        quantity=0
        prediction = List()
        for n in range(len(ValidData)):
            results = outcome(neighbor(train_data,ValidData[n],k,weight))
            prediction.append(results)
            #print(f"No.{quantity} Predicted class: {"無糖尿病"if results == 0 else "有糖尿病"}\tActual class: {"無糖尿病"if y_true[n] == 0 else "有糖尿病"}")
            correction += 1 if results == y_true[n] else 0
            quantity += 1
        
        print(f'{ColorFill.RED}Accuracy: {correction/quantity*100.0}% // epoch:{epoch} {ColorFill.END}')
    return correction/quantity*100.0

def test(k):
    test_data = input("Enter the test data:").split(",")
    results = outcome(neighbor(test_data,test_data,k_times))
    #print(f"Predicted class: {"無糖尿病"if results == 0 else "有糖尿病"}")


if __name__ == '__main__':
    MODE = str(input("Enter the mode you want to use(1.test  2.valid):"))
    DataSet = str(input("Enter the dataset you want to use(A/B):")).upper()
    filepath+= DataSet
    k_times = int(input("Enter the number of k:"))    
    epochs = int(input("Enter the number of train epochs:"))
    if MODE == "1":
        test(k_times)
    elif MODE == "2":
        validation(k_times,epochs)
    best_weight = []
    for j in range(len(acc_list)):
        for i in range(len(acc_list[j])):
            best_weight[i] = weight_mem[i][acc_list[i].index(max(acc_list[i]))]


    train_data = readfile(filepath,"train")
    ValidData = readfile(filepath,"valid")
    train_weights(0,k_times,epochs,train_data,ValidData)
    
    
    print(f'{ColorFill.GREEN}Best Weight: {best_weight}{ColorFill.END}')
    #weight = train_weights(data, labels, weight, learning_rate, epochs)

   