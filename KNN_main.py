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

weight = np.ones(8)  #8
normal_bias = np.ones((8,2))  #6
weight_bias = 1
formal_weight_bias = 1
acc_list = np.array([0,0])   #Accuracy list
y_true = np.array([])  #True value list
y_pred = np.array([])  #Predicted value list
def readfile(fp,mode)->np.array:
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

    
    for j in [0,len(data[0])-2,len(data[0])-1] if len(data[0]==9) else [0,len(data[0])-1]:
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


def validation(k,epochs,learning_rate):
    global weight_bias
    global formal_weight_bias
    global acc_list
    global weight
    global y_true
    train_data = readfile(filepath,"train")
    
    ValidData = readfile(filepath,"valid")
    #print(y_true)
    for argument in range(len(weight)):
        print(f"{ColorFill.GREEN}arg={argument}{ColorFill.END}")
        weight_bias = 1
        formal_weight_bias= 1
        pos = 0
        neg = 0
        should_break = False   
        for epoch in range(1,epochs+1):
            correction=0
            quantity=0
            prediction = []
            pos_rev = True
            neg_rev = True
            for n in range(len(ValidData)):
                results = outcome(neighbor(train_data,ValidData[n],k))
                prediction.append(results)
                #print(f"No.{quantity} Predicted class: {"無糖尿病"if results == 0 else "有糖尿病"}\tActual class: {"無糖尿病"if y_true[n] == 0 else "有糖尿病"}")
                correction += 1 if results == y_true[n] else 0
                quantity += 1   
            final = np.append(acc_list,[round(correction/quantity*100.0,2)],axis=0)
            acc_list = final.copy()
            #weight = gradient_descent(ValidData,np.array(y_true),weight,learning_rate,epochs)
            pos ,neg = train_weights(weight,learning_rate,argument,acc_list[-1],pos,neg,pos_rev,neg_rev)
            print(f'{ColorFill.RED}Accuracy: {acc_list[-1]}% // epoch:{epoch} // pos|neg:{pos}|{neg}{ColorFill.END}')
            if pos>=10 and neg>=10:
                break


    print(f'{ColorFill.BLUE}Weight: {weight}{ColorFill.END}')


def train_weights(weight, learning_rate,argw,acc,pos,neg,pos_rev,neg_rev)->set:
    global weight_bias
    global acc_list
    global formal_weight_bias
    global epochs
    
    print(weight_bias,acc_list[-2],acc)
    formal_weight_bias = weight[argw] if(acc>acc_list[-2]) else formal_weight_bias
    weight[argw] = weight[argw] * (1+learning_rate * (100-acc) / 20 * weight_bias)
    
    if acc<=acc_list[-2]:
        #print(acc,acc_list[-2])
        weight[argw] = formal_weight_bias
        if(weight_bias < 0):
            neg +=1
        elif(weight_bias > 0):
            pos +=1
    elif acc_list[-2]>acc_list[-3]:
        if(weight_bias > 0):
            pos =0
        elif(weight_bias < 0):
            neg =0

    if(pos>10 and pos_rev==True):
        pos_rev = False
        acc_list = np.append(acc_list,[acc],axis=0) 
    if(neg>10 and neg_rev==True):
        neg_rev = False
        acc_list = np.append(acc_list,[acc],axis=0)
        
    if (pos>=10 and pos_rev==True):
        weight_bias = -1
    elif(acc_list[-1]==acc_list[-2]):
        weight_bias *= 2

    weight_bias = (-1 if (neg>=10 and neg_rev==True) or (pos>=10 and pos_rev==True) else weight_bias*2 if acc_list[-1]==acc_list[-2] else weight_bias )
    

    return pos,neg


def test(k):
    test_data = input("Enter the test data:").split(",")
    results = outcome(neighbor(test_data,test_data,k_times))
    print(f"Predicted class: {"無糖尿病"if results == 0 else "有糖尿病"}")


if __name__ == '__main__':
    MODE = str(input("Enter the mode you want to use(1.test  2.valid):"))
    DataSet = str(input("Enter the dataset you want to use(A/B):")).upper()
    filepath+= DataSet
    learning_rate = 0.05
    k_times = int(input("Enter the number of k:"))    
    epochs = int(input("Enter the number of train epochs:"))
    if MODE == "1":
        test(k_times)
    elif MODE == "2":
        validation(k_times,epochs,learning_rate)

    f'{ColorFill.GREEN}Accuracy: {acc_list[-1]}%{ColorFill.END}'
    #weight = train_weights(data, labels, weight, learning_rate, epochs)