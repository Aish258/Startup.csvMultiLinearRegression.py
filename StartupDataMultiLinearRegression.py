#Multi Linear Regression-One predictor and four features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
link="C:\\Users\\Aishwarya\\Downloads\\Various Excel file data for practice\\3_Startups.csv"
df=pd.read_csv(link)
print(df)
print("Shape:",df.shape)
print("columns:",df.columns)

#Performing EDA(Explotatory data analysis) to understand the data
#Scatter plot
plt.scatter(df['R&D Spend'],df['Profit'])
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.title('Scatter plot of R&D Spend vs Profit')
plt.show()
#There is Positive corelation between X(R&D Spend) and y(Profit) so this is an example of Linear Regression Model

plt.scatter(df['Administration'],df['Profit'])
plt.xlabel('Administration')
plt.ylabel('Profit')
plt.title('Scatter plot of Administration vs Profit')
plt.show()
#There is No corelation between X(Administration) and y(Profit)

plt.scatter(df['Marketing Spend'],df['Profit'])
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.title('Scatter plot of Marketing Spend vs Profit')
plt.show()
#There is Positive corelation between X(Marketing Spend) and y(Profit)


plt.scatter(df['State'],df['Profit'])
plt.xlabel('State')
plt.ylabel('Profit')
plt.title('Scatter plot of State vs Profit')
plt.show()
#There is No corelation between X(State) and y(Profit)


x=df.iloc[:,:4].values
y=df.iloc[:,4].values

#Handel state column as it is categorical data
#Handeling categorical values using Label Encoder
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lc_x=LabelEncoder()
x[:,3]=lc_x.fit_transform(x[:,3])
print("1.Values of x:\n",x)

#Transformed using One Hot Encoder
from sklearn.compose import ColumnTransformer
transform= ColumnTransformer([('One_hot_encoder',OneHotEncoder(),[3])],remainder='passthrough')
x=transform.fit_transform(x)
print("2.Values of x:\n",x)

#Redoce any one column, we are dropping 1 column
x=x[:,1:]
print("3.Values of x after handeling categorical data:\n",x)


#Splitting data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)

#Model building
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

#Train the data
regressor.fit(x_train,y_train)

#regressor model will leran fro data and will generate m and c
c=regressor.intercept_
m=regressor.coef_
print("intercept/c:",c)
print("coefficient/m/slop of line:",m)


#Multiple regression Equation
#y=m1*x1+m2*x2+m3*x3+--------+C
#intercept/c: 48805.84254904062
#coefficient/m/slop of line: [ 2.65505554e+02  8.49007841e+02  7.61639095e-01 -1.13592334e-03 3.35038917e-02]
#Y=2.655*x1+8.490*x2+7.616*x3+(-1.135)*x4+3.350*x5+c


#Evaluate the model
#Predict x_test and then compair it with y_test
outcome=regressor.predict(x_test)
out_df=pd.DataFrame({'Actual':y_test,'Predicted':outcome})
print("Actual vs Predicted:\n",out_df)

#Model Evaluation
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,outcome)
rmse=mse**0.5
mae=metrics.mean_absolute_error(y_test,outcome)
print("Mean Squared Error=",mse)
print("Root Mean Squared Error=",rmse)
print("Mean Absolute error=",mae)

#Calculating R-Squared value
# This value can be maximum 1(Good fit) and minimum 0 (Bad fit) [R-squared=1-(MSE REGRESSION-MSE AVERAGE)]-Formula
R_Squared=metrics.r2_score(y_test,outcome)
print("R-squared value=",R_Squared)

#Creating OLS(Ordinary Least Square) summary
#P-value should be less than 0.05
import statsmodels.api as sm
from statsmodels.api import  OLS
x=sm.add_constant(x)
x=np.array(x,dtype=float)
summ=OLS(y,x).fit().summary()
print("OLS Summary:\n",summ)

#Running OLS sumaary again with specific columns of x
x_sel=x[:,[0,3,5]]
x_sel=sm.add_constant(x_sel)
x_sel=np.array(x_sel,dtype=float)
summ=OLS(y,x_sel).fit().summary()
print("OLS Summary with selective columns of x:\n",summ)



















