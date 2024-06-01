import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os


MAX_EPOCH = 40

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #TF 通知設定
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #TF GPU 參數設定
np.seterr(all='ignore')

# gloabl
project_name = "gene"
distance_limit = np.array([])
ans_avg = np.array([])
min_conf = []
predict_matrix = None


class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

class template:
    def Data_preprocessing(self,data,label)->tuple:
        temp = pd.DataFrame(data)
        #print(f'{Color.RED}Ignore warning below{Color.END}{Color.YELLOW}')
        # Normalization
        data = temp.to_numpy()
        data = (data - np.min(data,axis=0))/((np.max(data,axis=0)-np.min(data,axis=0)) ) 
        # Missing value handling
        data[np.isnan(data)] = 0
        #print(f"{Color.RED}Ignore warning above{Color.END}")
        return data , label
    

    def distance_count(self,x, y,a=-1):
        try:
            return np.sqrt(np.sum((x - y) ** 2,axis=a))
        except:
            try:
                return np.sqrt(np.sum((x - y) ** 2,axis=1))
            except:
                return np.sqrt(np.sum((x - y) ** 2))

class train(template):

    def __init__(self,model_name=None ):
        self.model_name = model_name
        self.rate = 0.2
        self.train_data, self.train_labels = self.read_train()
        self.train_data, self.train_labels = self.Data_aug(self.train_data,self.train_labels)
        self.train_data, self.train_labels, self.valid_data, self.valid_labels = self.sampling(self.train_data,self.train_labels)
        self.train_labels = tf.keras.utils.to_categorical(self.train_labels,num_classes=3)
        self.valid_labels = tf.keras.utils.to_categorical(self.valid_labels,num_classes=3)
        self.model = self.model_layer()
        self.train_model()
        

    def save_model(self):
        cond = input("save model? (y/n): ")
        if(cond == 'y' or cond == 'Y' ):
            self.model_name = input("model name: ")
            self.model.save(f'{os.getcwd()}/{project_name}/model/{self.model_name}')

    def read_train(self)->tuple:
        label_dict = dict()
        FT = open(f'{os.getcwd()}/{project_name}/train_data.csv','r')
        train_data = np.genfromtxt(FT,delimiter=',',dtype='float32',filling_values=0.0)[1:]

        FL = open(f'{os.getcwd()}/{project_name}/train_label.csv','r')
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
        #train_data , train_labels = np.delete(train_data,SI,axis=0) , np.delete(train_labels,SI,axis=0)
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
        #np.savetxt(f'{os.getcwd()}/{project_name}/distance_limit.csv',distance_limit,delimiter=',')

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
        self.model.fit(self.train_data, self.train_labels,epochs=MAX_EPOCH, verbose=1, shuffle=True)
        loss, acc = self.model.evaluate(self.train_data, self.train_labels, verbose=2)
        print("Validation loss: ",loss,"   Accuracy",acc)

        valid_conf = self.model.predict(self.train_data)
        temp = np.array([np.max(valid_conf,axis=1),np.argmax(self.train_labels,axis=1)])
        min_temp = [[],[],[]]
        for i in range(3):
            min_temp[i] = temp[0][temp[1]==i]
            min_temp[i] = np.sort(min_temp[i])
        global min_conf
        for i in range(3):
            min_conf.append(min_temp[i][int(len(min_temp[i])*0.1)])

        print(f'min_conf = {min_conf}')

    


class test(template):

    def __init__(self,trained_model):
        try:
            if(trained_model.model_name == None):
                self.model = trained_model.model
            else:
                global min_conf
                self.model = tf.keras.models.load_model(f'{os.getcwd()}/{project_name}/model/{trained_model.model_name}')
                min_conf = np.genfromtxt(f'{os.getcwd()}/{project_name}/model/{trained_model.model_name}/min_conf.csv',delimiter=',')
        except:
            if(trained_model==None):
                print(f'{Color.RED}No model found{Color.END}')
                return
            else:
                self.model = tf.keras.models.load_model(f'{os.getcwd()}/{project_name}/model/{trained_model}')
        
        self.test_data,self.test_labels = self.read_test()
        self.test_labels = self.test_labels.flatten()
        self.poss = np.max(self.model.predict(self.test_data),axis=1)
        self.predict_label = np.argmax(self.model.predict(self.test_data),axis=1)

        '''
        self.data_out_range,self.index_list = self.drop_far(0.9999)
        lb = self.kmeans(self.data_out_range,2)
        self.cluster_insert_label(self.index_list,lb)
        '''

        max_acc = self.conf_iteration()
        print(f'{Color.RED}Accuracy: {max_acc}{Color.END}')

    def read_test(self)->tuple:
        label_dict = dict()
        FT = open(f'{os.getcwd()}/{project_name}/test_data.csv','r')
        data = np.genfromtxt(FT,delimiter=',',dtype='float32',filling_values=0.0)[1:]

        FL = open(f'{os.getcwd()}/{project_name}/test_label.csv','r')
        label = np.genfromtxt(FL,delimiter=',',dtype='str')
        label = pd.DataFrame(label)
        label = label.drop(0,axis=1).to_numpy()[1:]
        for i in range(len(label)):
            if label[i][0] not in label_dict:
                label_dict[label[i][0]] = len(label_dict)
            label[i][0] = label_dict[label[i][0]]

        return self.Data_preprocessing(data,label)

    def kmeans(self,data, k):
        centers = data[np.random.choice(range(data.shape[0]), size=k, replace=False)]

        while True:

            distances = self.distance_count(data[:,np.newaxis], centers,2)
            labels = distances.argmin(axis=1)
            new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])
            
            if np.all(centers==new_centers):
                break

            centers = new_centers

        return labels


    def drop_far(self)->np.array:
        global distance_limit
        global ans_avg
        global min_conf

        index_bool = np.array([],dtype=bool)
        index = np.array([],dtype=int)
        for i in range(len(self.test_data)):
            index_bool = np.concatenate((index_bool,[self.poss[i]<min_conf[self.predict_label[i]]]),axis=0)
            index = np.concatenate((index,[i]),axis=0) if self.poss[i]<min_conf[self.predict_label[i]] else index
        return self.test_data[index_bool],index


    def cluster_insert_label(self,index,label):
        label = np.array(label,dtype=bool)  
        label_set = [3,4]
        acc=[]
        a,b = None , None
        for i in range(len(label_set)):
            self.predict_label[index[label==True]]=label_set[i]
            self.predict_label[index[label==False]]=label_set[i-1]
            acc.append(np.sum(self.test_labels == self.predict_label)/len(self.test_labels))
            
            
            if(i==0):
                a = np.array(self.predict_label==self.test_labels)
            elif(i==1):
                b = np.array(self.predict_label==self.test_labels)

        global predict_matrix
        predict_matrix = a if acc[0]>acc[1] else b
        print(f'{Color.GREEN}{acc}{Color.END}')
        return max(acc)

    def conf_iteration(self):
        self.data_out_range,self.index_list = self.drop_far()
        lb = self.kmeans(self.data_out_range,2)

        return self.cluster_insert_label(self.index_list,lb)

        

if __name__ == '__main__':
    a = None

    cond = input("Do you want to train model? (y/n):")

    if(cond == 'y' or cond == 'Y' ):
        mode = "both"
        Clock_start = time.time()  
        a = train()
        print(f'{Color.BLUE}train time :{time.time()-Clock_start}{Color.END}')
    else:
        mode = "test"
        a = input("model name:")

    
    Clock_start = time.time()
    test(a)
    print(f'{Color.BLUE}test time :{time.time()-Clock_start}{Color.END}')
    if(mode == "both"):
        a.save_model()
        np.savetxt(f'{os.getcwd()}/{project_name}/model/{a.model_name}/min_conf.csv',min_conf,delimiter=',')
        np.savetxt(f'{os.getcwd()}/{project_name}/model/{a.model_name}/predict_label_correct.txt',predict_matrix,delimiter='\n',fmt='%d')
    elif(mode == "test"):
        pass
    


   
    