"""ïmport the important librarires"""
import pandas as pd  
import numpy as np    
import matplotlib.pyplot as plt  

"""load the data"""
url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv" 
data_load = pd.read_csv(url)  
print("Successfully imported data into console" )

"""next phase is to enter distribution scores and plot them """
data_load.plot(x='Hours', y='Scores', style='o')    
plt.title('Hours vs Percentage')    
plt.xlabel('The Hours Studied')    
plt.ylabel('The Percentage Score')    
plt.show() 
 
""" dividing the data into attributes and labels"""
X = data_load.iloc[:, :-1].values    
y = data_load.iloc[:, 1].values  

"""split of data into the training and test sets"""
from sklearn.model_selection import train_test_split    
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)   

"""train the algorithm"""
from sklearn.linear_model import LinearRegression    
regressor = LinearRegression()    
regressor.fit(X_train, y_train)   
print("Training ... Completed !.") 

"""next phase is to implement the plotting test data using the previously trained test data"""
line = regressor.coef_*X+regressor.intercept_  
plt.scatter(X, y)  
plt.plot(X, line);  
plt.show()  

"""Predicting the scores for the model"""
print(X_test)   
y_pred = regressor.predict(X_test)

"""Comparing the actual versus predicted model to understand our model fitting"""
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})    
df  

"""What wiil be the predicted score if the student studies for 9.25hr/day?"""
hours = [[9.25]]  
own_pred = regressor.predict(hours)  
print("Number of hours = {}".format(hours))  
print("Prediction Score = {}".format(own_pred[0])) 
