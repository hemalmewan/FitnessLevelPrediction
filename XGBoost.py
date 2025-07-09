# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from xgboost import XGBRegressor
import pickle

# %% [markdown]
# Import DataSet

# %%
fitness=pd.read_csv("C:\\UOC pdf\\3rd Year\\MachineLearning-01\\final_project\\healthFitnessDataset.csv")

# %% [markdown]
# seperate independent variables and dependent variable

# %%
X=fitness.drop(columns=['fitness_level'])
Y=fitness['fitness_level']

# %% [markdown]
# Split Data Training and Testing

# %%
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.7,random_state=42)
X_train = X_train.reset_index(drop = True)
X_test = X_test.reset_index(drop = True)
Y_train = Y_train.reset_index(drop = True)
Y_test = Y_test.reset_index(drop = True)

# %% [markdown]
# Scale and encode training data

# %%
scaler=StandardScaler()

num_cols=X_train.select_dtypes(include='number').columns
X_train[num_cols]=scaler.fit_transform(X_train[num_cols])

# %%
##categorical Data Encoding using Ordinal and One-Hot Encoding

##Ordinal Endcoding
# Define the order manually
intensity_order = ['Low','Medium','High']
# Create encoder
ordinal_encoder = OrdinalEncoder(categories=[intensity_order])
# Fit and transform
X_train['intensity']= ordinal_encoder.fit_transform(X_train[['intensity']])

# %%
##Nominal Endcoding
# Create encoder
onehot_encoder = OneHotEncoder(sparse_output=False)

# Fit and transform
encoded_data = onehot_encoder.fit_transform(X_train[['gender','activity_type','smoking_status']])

# Convert the encoded array to a DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=onehot_encoder.get_feature_names_out(['gender', 'activity_type', 'smoking_status']))

# Combine with the original dataset
X_train = X_train.drop(['gender','activity_type','smoking_status'], axis=1)
X_train = pd.concat([X_train.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# %% [markdown]
# Scale and encoding testing data

# %%
##Scaling,Endcoding Testing set

##Scaling
X_test[num_cols]=scaler.fit_transform(X_test[num_cols])

# %%
##Endcoding categorical variables

###Nominal Endcoding
# Create encoder
onehot_encoder = OneHotEncoder(sparse_output=False)

# Fit and transform
encoded_data = onehot_encoder.fit_transform(X_test[['gender','activity_type','smoking_status']])

# Convert the encoded array to a DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=onehot_encoder.get_feature_names_out(['gender', 'activity_type', 'smoking_status']))

# Combine with the original dataset
X_test = X_test.drop(['gender','activity_type','smoking_status'], axis=1)
X_test = pd.concat([X_test.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# %%
###Ordinal Endcoding
# Define the order manually
intensity_order = ['Low','Medium','High']
# Create encoder
ordinal_encoder = OrdinalEncoder(categories=[intensity_order])
# Fit and transform
X_test['intensity']= ordinal_encoder.fit_transform(X_test[['intensity']])

# %% [markdown]
# Fit XGBoost Model

# %%
xgb=XGBRegressor()

# %%
#xgb.fit(X_train,Y_train)

# %%
#Y_pred_xgb=xgb.predict(X_test)
#Y_pred_train_xgb=xgb.predict(X_train)

# %%
#mae_test_xgb=mean_absolute_error(Y_test,Y_pred_xgb)
#mae_train_xgb=mean_absolute_error(Y_train,Y_pred_train_xgb)

#print("Testing MAE:",mae_test_xgb)
#print("Training MAE:",mae_train_xgb)

# %%
#mse_test_xgb=mean_squared_error(Y_test,Y_pred_xgb)
#mse_train_xgb=mean_squared_error(Y_train,Y_pred_train_xgb)

#print("Testing MSE:",mse_test_xgb)
#print("Training MSE:",mse_train_xgb)

# %%
#print("R2 score:",r2_score(Y_pred_xgb,Y_test))

# %% [markdown]
# Fit XGBoost Model with Hyperparameter Tuninng(GridSerach Method)

# %% [markdown]
# 

# %%
param_grid_xgb = {
    'n_estimators': [100, 200,300],
    'max_depth': [3, 5, 7,10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1],
}
xgb_cv=GridSearchCV(estimator=xgb,param_grid=param_grid_xgb,cv=5,n_jobs=-1)

# %%
xgb_cv.fit(X_train,Y_train)

# %%
#xgb_cv.best_params_

# %%
#Y_pred_xgb_cv=xgb_cv.predict(X_test)
#Y_pred_train_xgb_cv=xgb_cv.predict(X_train)

# %%
#test_mae_xgb_cv=mean_absolute_error(Y_pred_xgb_cv,Y_test)
#train_mae_xgb_cv=mean_absolute_error(Y_pred_train_xgb_cv,Y_train)

#print("Testing MAE(HyperParameter Tuninng):",test_mae_xgb_cv)
#print("Training MAE(Hyperparameter Tuninng):",train_mae_xgb_cv)

# %%
#mse_test_xgb_cv=mean_squared_error(Y_pred_xgb_cv,Y_test)
#mse_train_xgb_cv=mean_squared_error(Y_pred_train_xgb_cv,Y_train)

#print("Mean Square Error(Hyper Parameter Tuninng):",mse_test_xgb_cv)
#print("Mean Sqaure Error(Hyper Parameter Tuninng):",mse_train_xgb_cv)

# %%
#r_2Score_cv=r2_score(Y_test,Y_pred_xgb_cv)
#print("R2 score(HyperParameter Tuninng):",r_2Score_cv)

# %%



##save the model as pickle file
with open('best_model.pkl','wb') as f:
    pickle.dump(xgb_cv,f)

