import math 
import os
import csv 
import pandas as pd
import numpy as np
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
acc_list = []  #Accuracy list
weight_mem = []  #Weight memory
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

        return data_cleaning(data)

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

def neighbor(train,test,k)->list:
    DistanceSet = []
    for i in range(len(train)):
        DistanceSet.append((distance(train[i],test),train[i]))
    DistanceSet.sort(key=lambda x:x[0])
    return DistanceSet[:k]

def distance(point1, point2)->float:
    length = 0
    for i in range(len(point1[:-1])):
       length += (float(point1[i]) - float(point2[i])) ** 2 * weight[i]
    return math.sqrt(length)
    
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
    train_data= readfile(filepath,"train")
    
    ValidData= readfile(filepath,"valid")
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
            train_weights(k,epochs,train_data,ValidData,argument,basic_bias)
            
            
def train_weights(k,epochs,train_data,ValidData,argument,basic_bias):  #->double:
    
    for epoch in range(1,epochs+1):
        weight[argument]+=0.05
        correction=0
        quantity=0
        prediction = []
        for n in range(len(ValidData)):
            results = outcome(neighbor(train_data,ValidData[n],k))
            prediction.append(results)
            #print(f"No.{quantity} Predicted class: {"無糖尿病"if results == 0 else "有糖尿病"}\tActual class: {"無糖尿病"if y_true[n] == 0 else "有糖尿病"}")
            correction += 1 if results == y_true[n] else 0
            quantity += 1

        print(f'{ColorFill.RED}Accuracy: {correction/quantity*100.0}% // epoch:{epoch} {ColorFill.END}')
        acc = correction/quantity*100.0
        try:
            #print(acc_list)
            acc_list[basic_bias][argument].append(acc)
            weight_mem[basic_bias][argument].append(weight[argument])
        except Exception as e:
            #print(f'2:{e}')
            try:
                acc_list[basic_bias].append([acc])
                weight_mem[basic_bias].append([weight[argument]])
            except Exception as ex:
                #print(f'3:{ex}')
                acc_list.append([[acc]])
                weight_mem.append([[weight[argument]]])
        #print(weight_mem)
    #return correction/quantity*100.0

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

    print(f'{ColorFill.GREEN}Accuracy: {acc_list}{ColorFill.END}')
    print(f'{ColorFill.GREEN}Weight Memory: {weight_mem}{ColorFill.END}')
    
    best_weight = []
    bp = [0,0,0,0,0,0,0,0]
    for i in range(len(acc_list[0])):
        for j in range(len(acc_list)):
            try:
                best_weight[i] = best_weight[i]
            except :
                best_weight.append(0)
                #print("append")

            if max(acc_list[j][i]) > bp[i] :
                
                bp[i] = max(acc_list[j][i])
                best_weight[i] = round(weight_mem[j][i][acc_list[j][i].index(bp[i])],2)
                #print(best_weight)


    train_data = readfile(filepath,"train")
    ValidData = readfile(filepath,"valid")
    train_weights(k_times,1,train_data,ValidData,0,0)
    #print(best_weight)
    
    print(f'{ColorFill.GREEN}Best Weight: {best_weight}{ColorFill.END}')
    #weight = train_weights(data, labels, weight, learning_rate, epochs)