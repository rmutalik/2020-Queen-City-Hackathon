import pandas as pd  
import numpy as np  
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

df = pd.read_csv('C:\\Users\\Cameron.Shadmehry\\Misc\\Hackathon\\Info\\training.csv') #Read in CSV file
df = df.fillna(df.median()) #Change null values to median column value
z = np.abs(stats.zscore(df)) #Get Zscores of data points
threshold = 3
arr = (np.where(z > 3)) #Get list of data points that have a Zscore over the threshold
arrname = df.columns.values #Array of column values
count = 0
while count < len(arr[0]): #Loop to change outlier values to median
    df.iloc[arr[0][count],arr[1][count]] = df[arrname[arr[1][count]]].median()
    count = count+1
df = df.abs()
#Random Forest Model
X = df.iloc[:, 0:494].values
y = df.iloc[:, 495].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)
regressor.fit(X_train, y_train)
X_test = np.nan_to_num(X_test)
y_pred = regressor.predict(X_test)
print(y_pred)
np.where(np.isnan(X))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
