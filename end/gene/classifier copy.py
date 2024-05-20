import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #TF 通知設定
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #TF GPU 參數設定

# gloabl
model_name = "gene"


def read_test()->tuple:
    label_dict = dict()
    FT = open(f'{os.getcwd()}/{model_name}/test_data.csv','r')
    data = np.genfromtxt(FT,delimiter=',',dtype='float32',filling_values=0.0)[1:]

    FL = open(f'{os.getcwd()}/{model_name}/test_label.csv','r')
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

    # Missing value delete
    '''
    for i in range(len(temp.columns)):
        if (temp.loc[:,i] == 0).all():
            temp = temp.drop(i,axis=1)
    '''
    # Normalization
    data = temp.to_numpy()
    data = (data - np.min(data,axis=0))/((np.max(data,axis=0)-np.min(data,axis=0)) ) 
    # Missing value handling
    data[np.isnan(data)] = 0
    return data , label

def sampling(train_data,train_labels,rate,used):

    total_range = np.setdiff1d(np.arange(len(train_data)),used)
    SI = np.random.choice(total_range,int(len(train_data)*rate),replace=False)
    valid_data , valid_labels = train_data[SI] , train_labels[SI]
    return train_data , train_labels , valid_data , valid_labels , np.concatenate((SI,used),axis=0) , SI


def Data_aug(data,label):

    ans_quan = dict()
    ans_sheet = [[-1] for x in range(3)] #存放資料的index

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
            temp = np.array([(ans_avg[i] + data[j])*0.5])
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
    
    #隨機抽樣
    train_data, train_labels, valid_data, valid_labels, SI, used= sampling(train_data,train_labels,rate,used)

    #raise Exception("Stop")
    train_labels = tf.keras.utils.to_categorical(train_labels,num_classes=3)
    valid_labels = tf.keras.utils.to_categorical(valid_labels,num_classes=3)

    test_data,test_label = read_test()
    test_label = tf.keras.utils.to_categorical(test_label,num_classes=5)

    print(f'Read time :{time.time()-Clock_start}')


    from tensorflow.keras.layers import Dense, Dropout,BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping

    model = tf.keras.Sequential([
        Dense(1024, input_dim=train_data.shape[1],activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(3, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),


        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(3, activation='softmax'),
    ])

    while(True):

        model.summary()
        input("Start training")

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, train_labels,epochs=40, verbose=1, shuffle=True)
        #loaded_model = tf.keras.models.load_model('model.h5')

        valid_predict = np.array(model.predict(test_data, verbose=2))
        f=  open (f'{os.getcwd()}/{model_name}/valid_pred.txt','w')
        f.write(str(list(valid_predict)))
        f.close()


        loss, acc = model.evaluate(valid_data, valid_labels, verbose=2)
        print("Validation loss: ",loss,"   Accuracy",acc)
        print("Test Accuracy: ",sum(valid_predict.argmax(axis=1)==test_label.argmax(axis=1))/len(test_label))

        test_compare = valid_predict.argmax(axis=1)!=test_label.argmax(axis=1)

        print("Test:" , valid_predict[test_compare],test_label[test_compare])

        if(input("keep training? (y/n): ") != 'y'):
            break
        train_data, train_labels, valid_data, valid_labels, SI, used= sampling(train_data,train_labels,rate,used)


    cond = input("save model? (y/n): ")
    if(cond == 'y' or cond == 'Y' ):
        model.save(f'{os.getcwd()}/{model_name}/model/{input("Enter the model name: ")}')
