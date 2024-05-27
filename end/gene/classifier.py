import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #TF 通知設定
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #TF GPU 參數設定


# gloabl
model_name = "gene"

distance_limit = np.array([])
ans_avg = np.array([])


class Color:
    def __init__(self):
        self.RED = '\033[91m'
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.BLUE = '\033[94m'
        self.END = '\033[0m'

class template:
    def Data_preprocessing(self,data,label)->tuple:
        temp = pd.DataFrame(data)

        # Normalization
        data = temp.to_numpy()
        data = (data - np.min(data,axis=0))/((np.max(data,axis=0)-np.min(data,axis=0)) ) 
        # Missing value handling
        data[np.isnan(data)] = 0
        print("Ignore warning")
        return data , label
    

    def distance_count(self,x, y):
        try:
            return np.sqrt(np.sum((x - y) ** 2,axis=1))
        except:
            return np.sqrt(np.sum((x - y) ** 2))

class train(template):

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
        global ans_avg
        ans_avg = [np.mean(data[ans_sheet[x]],axis=0) for x in range(3)] # ans_avg 第一維度是類別(3) 第二維度是特徵(20242)
        global distance_limit

        for i in range(len(ans_avg)):
            distances = self.distance_count(data[ans_sheet[i]],ans_avg[i])
            distance_limit = np.concatenate((distance_limit,[max(distances)]),axis=0)
        np.savetxt(f'{os.getcwd()}/{model_name}/distance_limit.csv',distance_limit,delimiter=',')

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
        self.model.fit(self.train_data, self.train_labels,epochs=12, verbose=1, shuffle=True)
        loss, acc = self.model.evaluate(self.valid_data, self.valid_labels, verbose=2)
        print("Validation loss: ",loss,"   Accuracy",acc)
    


class test(template):

    def __init__(self):
        self.model = tf.keras.models.load_model(f'{os.getcwd()}/{model_name}/model/model5')
        self.test_data,self.test_labels = self.read_test()
        self.test_labels = self.test_labels.flatten()
        self.poss = np.max(self.model.predict(self.test_data),axis=1)
        self.predict_label = np.argmax(self.model.predict(self.test_data),axis=1)
        self.data_out_range,self.index_list = self.drop_far()
        #print(self.data_out_range.shape)
        #print(self.index_list)

        lb,lc = self.kmeans(self.data_out_range,2)
        self.cluster_insert_label(self.index_list,lb)
        

        '''
        self.cluster_result = self.hierarchical_clustering(self.data_out_range,5)
        print(self.cluster_result)
        print(self.index_list[self.cluster_result[0]])
        '''


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


    def kmeans(self,data, k):
        centers = data[np.random.choice(range(data.shape[0]), size=k, replace=False)]

        while True:
            distances = np.sqrt(((data[:, np.newaxis] - centers) ** 2).sum(axis=2))
            labels = distances.argmin(axis=1)
            new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])
            if np.all(centers == new_centers):
                break

            centers = new_centers

        return labels, centers


    def drop_far(self)->np.array:
        global distance_limit
        global ans_avg
        index_bool = np.array([],dtype=bool)
        index = np.array([],dtype=int)
        for i in range(len(self.test_data)):
            index_bool = np.concatenate((index_bool,[self.poss[i]<0.99999]),axis=0)
            index = np.concatenate((index,[i]),axis=0) if self.poss[i]<0.99999 else index
        return self.test_data[index_bool],index


    def cluster_insert_label(self,index,label):
        label = np.array(label,dtype=bool)  
        print(label)


        self.predict_label[index[label==True]]=4

        print(self.predict_label)
        self.predict_label[index[label==False]]=3
        print(self.predict_label)

        print(self.predict_label)
        print(self.test_labels)
        print(np.sum(self.test_labels == self.predict_label))
        print(len(self.test_labels))
        acc = np.sum(self.test_labels == self.predict_label)/len(self.test_labels)
        print(f'Accuracy: {acc}')


        



if __name__ == '__main__':
    
    used = np.array([])
    # Read data
    rate = 0.2 
    Clock_start = time.time()  
    #train()
    print(f'train time :{time.time()-Clock_start}')
    Clock_start = time.time()
    test()
    print(f'test time :{time.time()-Clock_start}')
    


   
    