import numpy as np
import tensorflow as tf
import time
import os

# gloabl
model_name = "Arrhythmia"


def read_train()->tuple:
    FT = open(f'{os.getcwd()}/end/{model_name}/train_data.csv','r')
    train_data = np.genfromtxt(FT,delimiter=',',dtype='float32',filling_values=0.0)
    a = []
    for row in train_data:
        a.append(row[-2])
    print(min(a),max(a))


    FL = open(f'{os.getcwd()}/end/{model_name}/train_label.csv','r')
    train_label = np.genfromtxt(FL,delimiter=',',dtype='int8')

    return Data_preprocessing(train_data,train_label)

def Data_cleaning():
    pass

def Data_preprocessing(data,label)->tuple:
    # Normalization
    data = (data - np.min(data,axis=0))/((np.max(data,axis=0)-np.min(data,axis=0)) if np.max(data,axis=0)-np.min(data,axis=0)!=0 else 1)
    # One-hot encoding
    print(list(data)[-1])
    raise ValueError
    return data , label
    


if __name__ == '__main__':
    Clock_start = time.time()  

    # Read data
    train_data , train_labels = read_train()
    
    # Sampling
    sample_index = np.random.choice(len(train_data),int(len(train_data)*0.2),replace=False)
    valid_data , valid_labels = train_data[sample_index] , train_labels[sample_index] 
    #train_data , train_labels = np.delete(train_data,sample_index,axis=0) , np.delete(train_labels,sample_index,axis=0)
    print(f'Read time :{time.time()-Clock_start}')




    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='softmax',kernel_regularizer='l2'),
        tf.keras.layers.Dense(1, activation='sigmoid'),

    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(train_data, train_labels, validation_split=0.2, epochs=40, verbose=1, shuffle=True)

    print(f'Time taken: {time.time()-Clock_start}')

    # Save model
    #model.save('model.h5')
