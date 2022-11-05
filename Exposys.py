import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

print("numpy version: "+np.__version__)
print("pandas version: "+pd.__version__)
print("seaborn version: "+sns.__version__)
print("sklearn version: "+sklearn.__version__)

dataset = pd.read_csv('C:/Users/user/Downloads/50_Startups.csv')
print(dataset)

dataset.describe()

dataset.duplicated().sum() #checking for dupicate values

dataset.isnull().sum() #checking for null values

dataset.info()

# correlation matrix
c = dataset.corr()

sns.heatmap(c,annot=True,cmap='Blues')
plt.show()

#outliers detection in the targeted variable
outliers = ['Profit']
plt.rcParams['figure.figsize'] = [8,8]
sns.boxplot(data=dataset[outliers], orient="v", palette="Set2" , width=0.7)
plt.title("Outliers Variable Distribution")
plt.ylabel("Profit Range")
plt.xlabel("Continuous Variable")
plt.show()

dataset.plot(kind='box',subplots=True,layout=(2,2),figsize=(12,7))

# Histogram on Profit
sns.distplot(dataset['Profit'],bins=5,kde=True)
plt.show()

sns.pairplot(dataset)
plt.show()

# spliting Dataset in Dependent & Independent Variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=0)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train,y_train)
print('Model has been trained successfully')

y_pred = model.predict(x_test)
y_pred


testing_data_model_score = model.score(x_test, y_test)
print("Model Score/Performance on Testing data",testing_data_model_score)

training_data_model_score = model.score(x_train, y_train)
print("Model Score/Performance on Training data",training_data_model_score)

df = pd.DataFrame(data={'Predicted value':y_pred.flatten(),'Actual Value':y_test.flatten()})
print(df)

from sklearn.metrics import r2_score

r2Score = r2_score(y_pred, y_test)
print("R2 score of model is :" ,r2Score*100)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_pred, y_test)
print("Mean Squarred Error is :" ,mse*100)

rmse = np.sqrt(mean_squared_error(y_pred, y_test))
print("Root Mean Squarred Error is : ",rmse*100)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_pred,y_test)
print("Mean Absolute Error is :" ,mae)



















