import math 
import os
import csv 
import pandas as pd

##We're not considering "Overfitting" cases in this code

DataSet = ""
filepath = f"{os.getcwd()}/data_set/"

weight = [1,1,1,1,1,1,1,1]  #8
normal_bias = [1,1,1,1,1,1]  #6
weight_bias = 1
acc_list = [0]

def readfile(fp,mode)->list:
    with open(file = fp+"/"+mode+".csv", mode = 'r',newline='') as f:
        raw_data = csv.reader(f)
        data = []
        next(raw_data)
        for row in raw_data:
            data.append(list(map(float,row)))
        return data_cleaning(data)

def data_cleaning(data)->list:

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

        data = normalized(data,avg)
    return data

def normalized(data,avg)->list:
    for row in range(1,len(data[0])-2):
        
        temp = []
        for i in data:
            temp.append(round(i[row],0))
        temp = pd.DataFrame(temp)
        Q1 = temp.quantile(0.25).values.tolist()[0]
        Q3 = temp.quantile(0.75).values.tolist()[0]
        IQR = Q3 - Q1
        temp = list(map(lambda x : avg if (x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR)) else x, temp.values.flatten().tolist()))
        '''
        temp_max = max(temp)
        temp_min = min(temp)
        print(f"Max:{temp_max}\tMin:{temp_min}")
        print(data)
        NORMAL = temp_max - temp_min
        normal_bias[row-1] = [NORMAL,temp_min]
        for j in range(len(temp)):
            print(data[j][row])
            data[j][row] = round((temp[j]-temp_min)/NORMAL,4)
        input("")
        '''
        
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
        lenth += (float(point1[i]) - float(point2[i])) ** 2 * weight[i]
    return math.sqrt(lenth)
    
def outcome(result)->int:
    cond = {}
    for i in range(len(result)):
        if result[i][1][-1] in cond:
            cond[result[i][1][-1]] += 1
        else:
            cond[result[i][1][-1]] = 1
    return max(cond)

def validation(k,epochs,learning_rate):
    for argument in range(len(weight)):
        print(f"arg={argument}")
        for epoch in range(epochs):
            correction=0
            quantity=0
            ValidData = readfile(filepath,"valid")
            for data in ValidData:
                results = outcome(neighbor(readfile(filepath,"train"),data,k))
                #print(f"Predicted class: {"無糖尿病"if results == 0 else "有糖尿病"}\tActual class: {"無糖尿病"if data[-1] == 0 else "有糖尿病"}")
                correction += 1 if results == data[-1] else 0
                quantity += 1
            acc_list.append(round(correction/quantity*100.0,2))
            print(f"Accuracy: {acc_list[-1]}%")
            try:
                if acc_list[-3] == acc_list[-1] and weight_bias == 1:
                    break
            except :
                None
            train_weights(weight,learning_rate,argument,acc_list[-1])


def train_weights(weight, learning_rate,argw,acc):
    global weight_bias
    global acc_list
    print(weight_bias,acc_list[-2],acc)
    weight[argw] = weight[argw] + learning_rate * (100-acc) / 20 * weight_bias
    weight_bias = 1 if acc>acc_list[-2] else (weight_bias*2 if acc==acc_list[-2] else -1)






def test(k):
    test_data = input("Enter the test data:").split(",")
    results = outcome(neighbor(test_data,test_data,k_times))
    print(f"Predicted class: {"無糖尿病"if results == 0 else "有糖尿病"}")


if __name__ == '__main__':
    MODE = str(input("Enter the mode you want to use(1.test  2.valid):"))
    DataSet = str(input("Enter the dataset you want to use(A/B):")).upper()
    filepath+= DataSet
    learning_rate = 0.01
    k_times = int(input("Enter the number of k:"))    
    epochs = int(input("Enter the number of train epochs:"))
    if MODE == "1":
        test(k_times)
    elif MODE == "2":
        validation(k_times,epochs,learning_rate)

    
    #weight = train_weights(data, labels, weight, learning_rate, epochs)