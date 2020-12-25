
import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split


import pickle


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
#Load Train Dataset
train = pd.read_csv("Train_data.csv")
#Load Test Dataset
test=pd.read_csv("Test_data.csv")

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
categorical_columns=['protocol_type', 'service', 'flag']

train_categorical_values = train[categorical_columns]
test_categorical_values = test[categorical_columns]

train_categorical_values.head()
encoder = LabelEncoder() #create a instance of label encoder
protocol_type_t=train["protocol_type"].values
protocol_type_en=encoder.fit_transform(protocol_type_t)
train["protocol_type"]=train["protocol_type"].replace(protocol_type_t,protocol_type_en)

service_t=train["service"].values
service_en=encoder.fit_transform(service_t)
train["service"]=train["service"].replace(service_t,service_en)

flag_t=train["flag"].values
flag_en=encoder.fit_transform(flag_t)
train["flag"]=train["flag"].replace(flag_t,flag_en)

class_d=train["class"].values
class_en=encoder.fit_transform(class_d)
train["class"]=train["class"].replace(class_d,class_en)


attack_d=train["attack"].values
attack_en=encoder.fit_transform(attack_d)
train["attack"]=train["attack"].replace(attack_d,attack_en)


y = train[['class']]
x = train.drop(['class', ], axis = 1)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn import svm
clf=svm.SVC(gamma=0.001,C=100)
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred,normalize=True)
print("Accuracy Score: {}".format(score))
train=train.drop(columns=['duration','src_bytes','dst_bytes', 'land', 'wrong_fragment','urgent','hot','num_compromised','root_shell','su_attempted','num_root',
                       'num_file_creations','num_shells','num_outbound_cmds','is_host_login','is_guest_login','srv_serror_rate','srv_rerror_rate',
                        'same_srv_rate','diff_srv_rate','srv_diff_host_rate', 'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
                       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate','dst_host_srv_serror_rate',
                        'dst_host_rerror_rate','dst_host_srv_rerror_rate'])

clf.fit(x_train,y_train)
fit = clf.score(x_test, y_test)
print("R-squared:", fit) 
y_pred = clf.predict(x_test)
y_pred

def get_final_ouput(fit, protocol_type, service, flag,num_failed_logins,
                    logged_in,num_access_files,count,srv_count,serror_rate,rerror_rate,):
    
    x_test = pd.DataFrame(columns= ["protocol_type", "service", "flag", "num_failed_logins",
                                    "logged_in","num_access_files","count","srv_count","serror_rate","rerror_rate",])
    x_test["protocol_type"] = [protocol_type]
    x_test["service"]= [service]
    x_test["flag"]=[flag] 
    x_test["num_failed_logins"]=[num_failed_logins]
    x_test["logged_in"]=[logged_in]
    x_test["num_access_files"]=[num_access_files]
    x_test["count"]=[count]
    x_test["srv_count"]=[srv_count]
    x_test["serror_rate"]=[serror_rate]
    x_test["rerror_rate"]=[rerror_rate]                  
  
                                 
    y_pred = clf.predict(x_test)
    
    clf = "normal" if y_pred ==1 else "anormaly"
    
    return clf

  get_final_ouput(fit,1,22,9,0,1,0,30,32,0.0,0.0)
  import pickle
  pickle.dump(clf, open('model.pkl','wb'))
  model=pickle.load(open('model.pkl','rb'))
