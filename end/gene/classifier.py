import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #TF 通知設定
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #TF GPU 參數設定

# gloabl
model_name = "gene"

cluster_args = np.array([])


class train:

    def __init__(self):
        self.rate = 0.2
        self.train_data, self.train_labels = self.read_train()
        self.train_data, self.train_labels = self.Data_aug(self.train_data,self.train_labels)
        self.train_data, self.train_labels, self.valid_data, self.valid_labels = self.sampling(self.train_data,self.train_labels)
        self.train_labels = tf.keras.utils.to_categorical(self.train_labels,num_classes=3)
        self.valid_labels = tf.keras.utils.to_categorical(self.valid_labels,num_classes=3)
        self.model = self.model_layer()
        self.train_model()

        cond = input("save model? (y/n): ")
        if(cond == 'y' or cond == 'Y' ):
            self.model.save(f'{os.getcwd()}/{model_name}/model/{input("Enter the model name: ")}')


    def read_train(self)->tuple:
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

        return self.Data_preprocessing(train_data,train_label)


    def Data_preprocessing(self,data,label)->tuple:
        temp = pd.DataFrame(data)

        # Normalization
        data = temp.to_numpy()
        data = (data - np.min(data,axis=0))/((np.max(data,axis=0)-np.min(data,axis=0)) ) 
        # Missing value handling
        data[np.isnan(data)] = 0
        return data , label


    def sampling(self,train_data,train_labels):

        SI = np.random.choice(np.arange(len(train_data)),int(len(train_data)*self.rate),replace=False)
        valid_data , valid_labels = train_data[SI] , train_labels[SI]
        return train_data , train_labels , valid_data , valid_labels


    def Data_aug(self,data,label):
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
        farest_data = []
        for i in range(len(ans_avg)):
            distances = [self.distance_count(data[ans_sheet[i][j]],ans_avg[i]) for j in range(len(ans_sheet[i]))]
            farest_data.append(max(distances))

        global cluster_args
        cluster_args = np.array(farest_data)

        for i in range(len(ans_sheet)):
            add_quan = max(ans_quan.values())-ans_quan[i]
            if add_quan <= 0:
                continue
            for j in np.random.choice(a=ans_sheet[i],size = add_quan):
                temp = np.array([(ans_avg[i] + data[j])*0.5])
                data = np.concatenate((data,temp),axis=0)
                label = np.concatenate((label,np.array([[i]])),axis=0)

        return data,label


    def model_layer(self):
        from tensorflow.keras.layers import Dense, Dropout,BatchNormalization
        model = tf.keras.Sequential([
            Dense(1024, input_dim=self.train_data.shape[1],activation='relu'),
            BatchNormalization(),
            Dropout(0.5),

            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),

            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),

            Dense(3, activation='softmax'),
        ])
        model.summary()
        return model
    

    def train_model(self):
        input("Start training")
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.train_data, self.train_labels,epochs=40, verbose=1, shuffle=True)
        loss, acc = self.model.evaluate(self.valid_data, self.valid_labels, verbose=2)
        print("Validation loss: ",loss,"   Accuracy",acc)
    

    def distance_count(self,x, y):
        return np.sqrt(np.sum((x - y) ** 2))
           

class test(train):

    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.load_model(f'{os.getcwd()}/{model_name}/model/model5')
        self.test_data,self.test_labels = self.read_test()
        self.test_labels = tf.keras.utils.to_categorical(self.test_labels,num_classes=3)
        self.test_model()


    def read_test(self)->tuple:
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

        return self.Data_preprocessing(data,label)
        

    def hierarchical_clustering(self,data,cl_num):
        num_samples = len(data)
        distances = np.zeros((num_samples, num_samples))
        np.fill_diagonal(distances, np.inf) 

        clusters = [[i] for i in range(num_samples)]

        while len(clusters) > cl_num:
            min_distance = np.inf
            merge_indices = (0, 0)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    cluster_i = clusters[i]
                    cluster_j = clusters[j]

                    for index_i in cluster_i:
                        for index_j in cluster_j:
                            distance = self.distance_count(data[index_i], data[index_j])
                            if distance < min_distance:
                                min_distance = distance
                                merge_indices = (i, j)

            i, j = merge_indices
            clusters[i].extend(clusters[j])
            del clusters[j]

            for k in range(len(clusters)):
                if k != i:
                    cluster_k = clusters[k]
                    for index_i in clusters[i]:
                        for index_k in cluster_k:
                            distances[index_i, index_k] = self.distance_count(data[index_i], data[index_k])
                            distances[index_k, index_i] = distances[index_i, index_k]

        return clusters


    def test_model(self):
        pass

if __name__ == '__main__':
    
    used = np.array([])
    # Read data
    rate = 0.2 
    Clock_start = time.time()  
    train()
    print(f'train time :{time.time()-Clock_start}')
    Clock_start = time.time()
    test()
    print(f'test time :{time.time()-Clock_start}')
    


   
    