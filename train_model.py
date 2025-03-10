import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error


df=pd.read_csv('Housing.csv')

print(df.head())

print(df.info())

print(df.columns)

## Convert categorical 'yes'/'no' values into 1/0
binary_columns=["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
df[binary_columns]=df[binary_columns].apply(lambda x:x.map({'yes':1,'no':0}))

## Convert 'furnishingstatus' to numeric (one-hot encoding)
df=pd.get_dummies(df,columns=["furnishingstatus"],drop_first=True)

                                            

df[binary_columns]=df[binary_columns].fillna(0)
df[binary_columns]=df[binary_columns].astype(int)
print(df.dtypes)

#feature selection using corelation(Correlation is a statistical measure that shows how strongly two variables are related to each other. It helps in understanding whether increasing or decreasing one variable affects another.)
#price is the target variable.
#You checked how strongly each feature correlates with price.
#Features like area, bathrooms, stories, parking, and bedrooms had meaningful correlations.
#Features like mainroad, guestroom, basement, etc., had NaN correlation values, meaning they had only one unique value (no variance), so they are useless for prediction.


corelation_matrix=df.corr()
print(corelation_matrix["price"].sort_values(ascending=False))

print(df[binary_columns].nunique())
#thse columns like mainroad,guestroom,basemnet,hotwaterheating,airconditioning,prefarea do not contribute much to house price prediction so they can be dropped


columns_to_drop=["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
df=df.drop(columns=columns_to_drop)
print(df.head())

#check for misising values
print(df.isnull().sum())
#since no null values in or dataset

#When working with machine learning models, feature scaling helps to ensure that all features contribute equally 
#and improves model performance. 
#Some algorithms (like Gradient Descent-based models) perform better when numerical values are on the same scale.

#Regression models assume that features are normally distributed.
#Standardization centers the data around mean = 0 and scales it to standard deviation = 1.
#It helps models converge faster and prevents features with large ranges (e.g., area vs. bedrooms) from dominating the model.


from sklearn.preprocessing import StandardScaler
features_to_scale=["area","bedrooms","bathrooms","stories","parking"]

scaler=StandardScaler()
df[features_to_scale]=scaler.fit_transform(df[features_to_scale])
print(df.head())


#We need to separate the data into training (for learning) and testing (for evaluation).

X=df.drop(columns=['price']) #it will drop price columns and will take all other fetaure
y=df['price'] #target variable
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# Train the Regression Model
#Now, let’s fit a simple Linear Regression model.

model=LinearRegression()
model.fit(X_train,y_train)

#make prediction
y_pred_lr=model.predict(X_test)
print(y_pred_lr[0:6])

#Evaluate the Model
#We need to check how well our model performs using key metrics.

mae=mean_absolute_error(y_test,y_pred_lr)
mse=mean_squared_error(y_test,y_pred_lr)
rmse=root_mean_squared_error(y_test,y_pred_lr)
r2=r2_score(y_test,y_pred_lr)

print(f"MAE: {mae}")
print(f"mse:{mse}")
print(f"rmse:{rmse}")
print(f"r2:{r2}")

#now we will use random forest n will check mae,mse,rmse,r2
from sklearn.ensemble import RandomForestRegressor
rf_model=RandomForestRegressor(n_estimators=100,random_state=42)
rf_model.fit(X_train,y_train)

y_pred_rf=rf_model.predict(X_test)
print(y_pred_rf[0:6])  

#now print metrics for random forest
print("MAE:",mean_absolute_error(y_test,y_pred_rf))
print("MSE:",mean_squared_error(y_test,y_pred_rf))
print("R^2 score:",r2_score(y_test,y_pred_rf))

#as the mae,mse,r^2 score did not imporoved in random forst also now we will do hyperparamaeter tunning using GridSearchcv

#get important features
importances=rf_model.feature_importances_
feature_names=X.columns

#sort features by importances
sorted_idx=np.argsort(importances)[::-1]

#plot feature importance
plt.figure(figsize=(10,5))
sns.barplot(x=importances[sorted_idx],y=feature_names[sorted_idx])
plt.xlabel("Feature importance")
plt.ylabel("Feature Name")
plt.title("Feature Importance in Random Forest")
plt.show()



#Hypertune Parametetrs with GridSearchCV
#insted of using default settings,let's find the best n_estimators,max_depth,min_sample_split,etc.
from sklearn.model_selection import GridSearchCV

#Define Hyperparameter grid
param_grid={
    'n_estimators':[50,100,200],
    'max_depth':[None,10,20,30],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4]
}

#Run GridSearchCv
grid_search=GridSearchCV(RandomForestRegressor(random_state=42),param_grid,cv=5,n_jobs=-1)
grid_search.fit(X_train,y_train)

#Best parameters
print("Best Parameters:",grid_search.best_params_)

#train model with best parametrs
best_rf=RandomForestRegressor(**grid_search.best_params_,random_state=42)
best_rf.fit(X_train,y_train)

#preditions
y_pred_xgb=best_rf.predict(X_test)
print(y_pred_xgb[0:6])

#performace evaulation

print("Optimized MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("Optimized MSE:", mean_squared_error(y_test, y_pred_xgb))
print("Optimized R² Score:", r2_score(y_test, y_pred_xgb))



import pandas as pd

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "MAE": [mean_absolute_error(y_test, y_pred_lr), 
            mean_absolute_error(y_test, y_pred_rf), 
            mean_absolute_error(y_test, y_pred_xgb)],
    "MSE": [mean_squared_error(y_test, y_pred_lr), 
            mean_squared_error(y_test, y_pred_rf), 
            mean_squared_error(y_test, y_pred_xgb)],
    "R² Score": [r2_score(y_test, y_pred_lr), 
                 r2_score(y_test, y_pred_rf), 
                 r2_score(y_test, y_pred_xgb)]
})

# Print comparison
print(results.sort_values(by="R² Score", ascending=False))

import joblib

#save the model
joblib.dump(model,"Linear_regression_model.pkl")
print("Model Saved Succcessfully")
