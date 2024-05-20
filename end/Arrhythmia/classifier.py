import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# gloabl
model_name = "Arrhythmia"


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

def read_train()->tuple:
    FT = open(f'{os.getcwd()}/end/{model_name}/train_data.csv','r')
    train_data = np.genfromtxt(FT,delimiter=',',dtype='float32',filling_values=0.0)

    FL = open(f'{os.getcwd()}/end/{model_name}/train_label.csv','r')
    train_label = np.genfromtxt(FL,delimiter=',',dtype='int8')

    return train_data,train_label

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

    total_range = np.setdiff1d(len(train_data),used)
    SI = np.random.choice(len(train_data),int(len(train_data)*rate),replace=False)
    valid_data , valid_labels = train_data[SI] , train_labels[SI]
    train_data , train_labels = np.delete(train_data,SI,axis=0) , np.delete(train_labels,SI,axis=0)

    #valid_data=train_data           # same data test 
    #valid_labels=train_labels

    return train_data , train_labels , valid_data , valid_labels , np.concatenate((SI,used),axis=0) , SI

if __name__ == '__main__':
    Clock_start = time.time()  
    used = np.array([])
    # Read data

    train_data , train_labels = read_train()

    #print(f'Read time :{time.time()-Clock_start}')
    rate = float(input("Enter the rate of validation data: "))

    from keras import layers as kl
    model = tf.keras.Sequential([
        kl.InputLayer(input_shape=(train_data.shape[1],)),
        kl.Dense(256, activation='relu',kernel_regularizer='l1'),
        kl.Dense(512, activation='softmax',kernel_regularizer='l1'),
        kl.Dropout(0.2),
        kl.Dense(512, activation='sigmoid',kernel_regularizer='l1'),
        kl.Dense(1024, activation='softmax',kernel_regularizer='l2'),
        kl.Dropout(0.2),
        kl.Dense(512, activation='relu',kernel_regularizer='l1'),
        kl.Dropout(0.2),
        kl.Dense(8, activation='softmax'),
    ])


    while(input("keep training? (y/n): ") == 'y'):
        
        # 抽樣
        train_data, train_labels, valid_data, valid_labels, SI, used= sampling(train_data,train_labels,rate,used)


        #轉換標籤為"標籤位置"的形式
        train_labels = tf.keras.utils.to_categorical(train_labels-1,num_classes=8)
        valid_labels = tf.keras.utils.to_categorical(valid_labels-1,num_classes=8)
        
        

        
        model.summary()
        input("Start training")

        model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, train_labels,validation_split=0.2, epochs=40, verbose=1, shuffle=True)
        #loaded_model = tf.keras.models.load_model('model.h5')

        valid_predict = np.array(model.predict(valid_data, verbose=2))
        f=  open ('valid_pred.txt','w')
        f.write(str(list(valid_predict)))
        f.close()
        print("Validation Accuracy: ",sum(valid_predict.argmax(axis=1)==valid_labels.argmax(axis=1))/len(valid_labels))
        #model.layers[0].trainable = False
        

    print(f'Time taken: {time.time()-Clock_start}')
    cond = input("save model? (y/n): ")
    if(cond == 'y' or cond == 'Y' ):
        model.save(input("Enter the model name: "))