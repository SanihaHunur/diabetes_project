import pandas as pd 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pickle

#load the dataset
df=pd.read_csv('diabetes.csv')
print(df.head())

#split dataset into x-axis and y-axis
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

#split dataset into training and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=60)

#create model
algorithm = GaussianNB()

#train the mdoel using training data
algorithm.fit(x_train,y_train)

# #predicting the outcome
# pred=algorithm.predict([[1,81,74,41,57,46.3,1.096,32]])
# print("The predicted outcome is :",pred)

pickle.dump(algorithm,open('model.pkl','wb'))

# mp = pickle.load(open('model.pkl','rb'))
# pred=mp.predict([[1,81,74,41,57,46.3,1.096,32]])
# print(pred)