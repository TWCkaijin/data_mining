import math 
import os
import numpy as np
import csv 


DataSet = ""
filepath = f"{os.getcwd()}/data_set/"





def readfile(fp,mode)->list:
    with open(file = fp+"/"+mode+".csv", mode = 'r',newline='') as f:
        raw_data = csv.reader(f)
        data = []
        next(raw_data)
        for row in raw_data:
            data.append(list(map(float,row)))
        return data
    

import pandas as pd

def data_cleaning(a):
    total = 0
    quantity = 0
    a=[]
    for i in data:
        total += i[3] if i[3] != 0 else 0
        quantity += 1 if i[3] != 0 else 0
    avg = total/quantity
    for i in data:
        i[3]=avg if i[3] == 0 else i[3]
    for i in data:
        a.append(round(i[3],0))
    a = pd.DataFrame(a)
    Q1 = a.quantile(0.25)
    Q3 = a.quantile(0.75)
    IQR = Q3 - Q1
    a = a[~((a < (Q1 - 1.5 * IQR)) | (a > (Q3 + 1.5 * IQR))).any(axis=1)]

    return a.values.tofile("output.csv", sep=",", format="%s")

if __name__ == '__main__':
    DataSet = str(input("Enter the dataset you want to use(A/B):")).upper()
    filepath+= DataSet
    data = readfile(filepath,"train")
    
    data_cleaning(data)
    