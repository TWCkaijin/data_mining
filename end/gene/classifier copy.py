import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# gloabl
model_name = "gene"

class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def read_test()->tuple:
    label_dict = dict()
    FT = open(f'{os.getcwd()}/{model_name}/train_data.csv','r')
    data = np.genfromtxt(FT,delimiter=',',dtype='float32',filling_values=0.0)[1:]

    FL = open(f'{os.getcwd()}/{model_name}/train_label.csv','r')
    label = np.genfromtxt(FL,delimiter=',',dtype='str')
    label = pd.DataFrame(label)
    label = label.drop(0,axis=1).to_numpy()[1:]
    for i in range(len(label)):
        if label[i][0] not in label_dict:
            label_dict[label[i][0]] = len(label_dict)
        label[i][0] = label_dict[label[i][0]]



    return Data_preprocessing(data,label)

def read_train()->tuple:
    label_dict = dict()
    FT = open(f'{os.getcwd()}/{model_name}/train_data.csv','r')
    train_data = np.genfromtxt(FT,delimiter=',',dtype='float32',filling_values=0.0)[1:]

    FL = open(f'{os.getcwd()}/{model_name}/train_label.csv','r')
    train_label = np.genfromtxt(FL,delimiter=',',dtype='str')
    train_label = pd.DataFrame(train_label)
    train_label = train_label.drop(0,axis=1).to_numpy()[1:]
    for i in range(len(train_label)):
        if train_label[i][0] not in label_dict:
            label_dict[train_label[i][0]] = len(label_dict)
        train_label[i][0] = label_dict[train_label[i][0]]



    return Data_preprocessing(train_data,train_label)

def Data_cleaning():
    pass

def Data_preprocessing(data,label)->tuple:
    temp = pd.DataFrame(data)
    # Missing value
    for i in range(len(temp.columns)):
        if (temp.loc[:,i] == 0).all():
            temp = temp.drop(i,axis=1)
    # Normalization
    data = temp.to_numpy()
    data = (data - np.min(data,axis=0))/((np.max(data,axis=0)-np.min(data,axis=0)) )
    return data , label

def sampling(train_data,train_labels,rate,used):

    total_range = np.setdiff1d(np.arange(len(train_data)),used)
    SI = np.random.choice(total_range,int(len(train_data)*rate),replace=False)
    valid_data , valid_labels = train_data[SI] , train_labels[SI]
    #train_data , train_labels = np.delete(train_data,SI,axis=0) , np.delete(train_labels,SI,axis=0)

    #valid_data=train_data           # same data test 
    #valid_labels=train_labels

    return train_data , train_labels , valid_data , valid_labels , np.concatenate((SI,used),axis=0) , SI


def Data_aug(data,label):

    ans_quan = dict()
    ans_sheet = [[-1],[-1],[-1]] #存放資料的index

    for i in range(len(label)):
        if(label[i][0] not in ans_quan):
            ans_quan.update({label[i][0]:1})
            ans_sheet[label[i][0]][0] = i
        else:
            ans_quan[label[i][0]]+=1
            #print(ans_sheet[label[i][0]])
            ans_sheet[label[i][0]].append(i)
    

    ans_avg = [np.mean(data[ans_sheet[x]],axis=0) for x in range(3)] # ans_avg 第一維度是類別(3) 第二維度是特徵(20242)

    for i in range(len(ans_sheet)):
        add_quan = max(ans_quan.values())-ans_quan[i]
        if add_quan <= 0:
            continue
        for j in np.random.choice(a=ans_sheet[i],size = add_quan):
            temp = np.array([(ans_avg[i] + data[j])*np.random.random(1)])
            
            data = np.concatenate((data,temp),axis=0)
            label = np.concatenate((label,np.array([[i]])),axis=0)
    return data,label


if __name__ == '__main__':
    
    used = np.array([])
    # Read data
    rate = 0.2 
    Clock_start = time.time()  
    train_data , train_labels = read_train()

    #增強
    train_data,train_labels = Data_aug(train_data,train_labels)
    print(train_data.shape,train_labels.shape)
    
    #隨機抽樣
    train_data, train_labels, valid_data, valid_labels, SI, used= sampling(train_data,train_labels,rate,used)

    #raise Exception("Stop")
    train_labels = tf.keras.utils.to_categorical(train_labels,num_classes=3)
    valid_labels = tf.keras.utils.to_categorical(valid_labels,num_classes=3)

    test_data,test_label = read_test()
    test_label = tf.keras.utils.to_categorical(test_label,num_classes=5)

    print(f'Read time :{time.time()-Clock_start}')

    
    from keras import layers as kl
    model = tf.keras.Sequential([
        kl.InputLayer(input_shape=(train_data.shape[1],)),
        kl.Dense(4096, activation='softmax'),
        kl.Dense(512, activation='sigmoid'),
        kl.Dropout(0.2),
        kl.Dense(512, activation='softmax'),
        kl.Dense(512, activation='sigmoid'),
        kl.Dropout(0.2),
        kl.Dense(3, activation='softmax'),
    ])

    while(True):

        model.summary()
        input("Start training")

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, train_labels,epochs=40, verbose=1, shuffle=True)
        #loaded_model = tf.keras.models.load_model('model.h5')

        valid_predict = np.array(model.predict(test_data, verbose=2))
        f=  open (f'{os.getcwd()}/{model_name}/valid_pred.txt','w')
        f.write(str(list(valid_predict)))
        f.close()

        print("Validation Accuracy: ",sum(valid_predict.argmax(axis=1)==test_label.argmax(axis=1))/len(test_label))

        if(input("keep training? (y/n): ") != 'y'):
            break
        train_data, train_labels, valid_data, valid_labels, SI, used= sampling(train_data,train_labels,rate,used)


    cond = input("save model? (y/n): ")
    if(cond == 'y' or cond == 'Y' ):
        model.save(f'{os.getcwd()}/{model_name}/{input("Enter the model name: ")}')
