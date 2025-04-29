import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.model_selection import GridSearchCV, PredefinedSplit, RandomizedSearchCV
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import quantile_transform, power_transform, RobustScaler
from scipy.stats import boxcox
import wbgapi as wb
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import wasserstein_distance
from sklearn import linear_model
from sklearn.inspection import permutation_importance
import shap
from sklearn.inspection import permutation_importance
from PyALE import ale
import seaborn as sns
from itertools import combinations
from sklearn.utils import resample
from sklearn.svm import SVR
from joblib import parallel_backend

def preprocess_min_max_group(df, x, group):
    out = pd.DataFrame()
    for i in df[group].unique():
        df_s = df.loc[df[group] == i]
        scaler = preprocessing.MinMaxScaler().fit(
            df_s[x].values.reshape(-1, 1))
        norm = scaler.transform(df_s[x].values.reshape(-1, 1))
        out = pd.concat([out, pd.DataFrame(norm)], ignore_index=True)
    df[f"{x}_norm"] = out
    df[f"{x}_norm"] = df[f"{x}_norm"].fillna(0)


def dichotomize(df,var,var_out,thresh):
    df[var_out]=0
    df.loc[df[var]>thresh,var_out]=1
    
def consec_zeros_grouped(df,group,var):
    zeros=[]
    for c in df.country.unique():
        counts=0
        df_s=df[var].loc[df[group]==c]
        for i in range(len(df_s)): 
            if df_s.iloc[i] == 0:
                zeros.append(counts)
                counts+=1
            elif df_s.iloc[i]==1: 
                counts=0
                zeros.append(0)
    return zeros

# Define the decay function
def exponential_decay(time):
    return 2**(time/12)

# Apply the nonlinear decay function to each group
def apply_decay(group,var):
    out = exponential_decay(group[var])
    return out

################
### Get data ###
################


def get_wb(years: list,
           countries: list,
           var: list,
           sub: str):
    """ years: specify list of years for which to extract data
        countries: specify list of countries for which to extract data
        var: specify list of variables to extract
        sub: add suffix to name of data file"""

    # Load data from world bank --------
    # https://pypi.org/project/wbgapi/

    wdi = pd.DataFrame()

    # loop through each year and get data
    for i in years:
        print(i)
        wdi_s = wb.data.DataFrame(var, countries, [i])
        wdi_s.reset_index(inplace=True)
        wdi_s["year"] = i
        wdi = pd.concat([wdi, wdi_s], ignore_index=True)  # merge for each year

    # Import country codes  -----
    df_ccodes = pd.read_csv("data/df_ccodes.csv")

    # Add ucdp codes -----
    df_ccodes_s = df_ccodes[["country",
                             'gw_codes', "acled_codes", "iso_alpha3"]]

    # Merge country codes -----
    wdi_final = pd.merge(wdi, df_ccodes_s, how='left', left_on=[
                         'economy'], right_on=['iso_alpha3'])
    wdi_final = wdi_final.drop(columns=['economy'])
    wdi_final = wdi_final[['gw_codes'] +
                          [col for col in wdi_final.columns if col != 'gw_codes']]
    wdi_final = wdi_final[['iso_alpha3'] +
                          [col for col in wdi_final.columns if col != 'iso_alpha3']]
    wdi_final = wdi_final[['acled_codes'] +
                          [col for col in wdi_final.columns if col != 'acled_codes']]
    wdi_final = wdi_final[['year'] +
                          [col for col in wdi_final.columns if col != 'year']]
    wdi_final = wdi_final[['country'] +
                          [col for col in wdi_final.columns if col != 'country']]

    print("Obtained data")
    print(wdi_final.head())

    # Sort and reset index -------
    wdi_final = wdi_final.sort_values(by=["iso_alpha3", 'year'])
    wdi_final = wdi_final.reset_index(drop=True)

    return wdi_final

def lag_groupped(df, group_var, var, lag):
    return df.groupby(group_var)[var].shift(lag).fillna(0)

def emd_vars(var1, var2):
    x1 = np.sort(var1)
    x2 = np.sort(var2)
    emd = wasserstein_distance(x1, x2)
    return emd

def percentage_decrease(old_value, new_value):
    return ((old_value - new_value) / old_value) * 100

def multivariate_imp_bayes(df, country, vars_input, vars_add=None,max_iter=10,min_val=0,time="year",last_train=2021):
    
    ### Split data ###
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]

        # Train, test
        train_s = df_s.loc[df_s[time]<=last_train]
        test_s = df_s.loc[df_s[time]>last_train]

        # Merge
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])
    
    ### Training ###

    # Step 1 Make subset of data for imputation - only explanatory variables
    feat_imp = train[vars_input]
    df_MICE = feat_imp.copy(deep=True)
    df_MICE['lag1'] = lag_groupped(train, country, vars_input, 1)
    df_MICE['lag2'] = lag_groupped(train, country, vars_input, 2)
    df_MICE['lag3'] = lag_groupped(train, country, vars_input, 3)
    df_MICE['lag4'] = lag_groupped(train, country, vars_input, 4)
    df_MICE['lag5'] = lag_groupped(train, country, vars_input, 5)

    if vars_add is not None:
        df_filled = train.fillna(0)
        df_MICE = pd.concat([df_MICE, df_filled[vars_add]], axis=1)

    # Step 2 Make subset of data which are complete and are not involved in the imputation
    feat_complete = train.drop(columns=vars_input)
    feat_complete = train.drop(columns=vars_add)
    feat_complete=feat_complete.reset_index(drop=True)
    
    # Tune model
    train_val = pd.DataFrame()
    test_val = pd.DataFrame()
    for c in df.country.unique():
        df_s = train.loc[train["country"] == c]

        # Train, test
        train_s = df_s[:int(0.8*len(df_s))]
        test_s = df_s[int(0.8*len(df_s)):]
        
        # Merge
        train_val = pd.concat([train_val, train_s])
        test_val = pd.concat([test_val, test_s])

    splits = np.array([-1] * len(train_val.index) + [0] * len(test_val.index))
    ps = PredefinedSplit(test_fold=splits)

    random_grid = {'alpha': [0,0.1,0.5,1,2,4,10,50,100]}
    
    df_input = df_MICE.fillna(0)
    df_input_x = df_input.drop(columns=vars_input)

    grid_search = GridSearchCV(estimator=linear_model.Ridge(), param_grid=random_grid, cv=ps)
    grid_search.fit(df_input_x, df_input[vars_input].values.ravel())
    best_params = grid_search.best_params_
    
    # Step 3 Specify imputer
    imputer = IterativeImputer(
        estimator=linear_model.Ridge(**best_params), random_state=1, max_iter=max_iter, min_value=min_val)

    # Step 4 Run imputation and obtain imputed dataset
    imputer.fit(df_MICE)
    df_MICE_trans = imputer.transform(df_MICE)
    df_MICE_trans_df = pd.DataFrame(df_MICE_trans)
    df_MICE_trans_df = df_MICE_trans_df.iloc[:, :len(vars_input)]
    df_MICE_trans_df = df_MICE_trans_df.rename(columns={0: 'imp'})

    # Step 5 Merge imputed dataset based on index
    train_imp = pd.concat([feat_complete, df_MICE_trans_df],axis=1)
    train_imp['missing_id'] = train_imp[vars_input].isnull().astype(int)
    
    ### Test ###
    
    # Step 1 Make subset of data for imputation - only explanatory variables
    feat_imp = test[vars_input]
    df_MICE = feat_imp.copy(deep=True)
    df_MICE['lag1'] = lag_groupped(test, country, vars_input, 1)
    df_MICE['lag2'] = lag_groupped(test, country, vars_input, 2)
    df_MICE['lag3'] = lag_groupped(test, country, vars_input, 3)
    df_MICE['lag4'] = lag_groupped(test, country, vars_input, 4)
    df_MICE['lag5'] = lag_groupped(test, country, vars_input, 5)

    if vars_add is not None:
        df_filled = test.fillna(0)
        df_MICE = pd.concat([df_MICE, df_filled[vars_add]], axis=1)

    # Step 2 Make subset of data which are complete and are not involved in the imputation
    feat_complete = test.drop(columns=vars_input)
    feat_complete = test.drop(columns=vars_add)
    feat_complete=feat_complete.reset_index(drop=True)
 
    # Step 4 Run imputation and obtain imputed dataset
    df_MICE_trans = imputer.transform(df_MICE)
    df_MICE_trans_df = pd.DataFrame(df_MICE_trans)
    df_MICE_trans_df = df_MICE_trans_df.iloc[:, :len(vars_input)]
    df_MICE_trans_df = df_MICE_trans_df.rename(columns={0: 'imp'})

    # Step 5 Merge imputed dataset based on index
    test_imp = pd.concat([feat_complete, df_MICE_trans_df], axis=1)
    test_imp['missing_id'] = test_imp[vars_input].isnull().astype(int)
    
    # Merge train and test
    _ = pd.concat([train_imp, test_imp])
    _=_.sort_values(by=["gw_codes",time])
    _=_.reset_index(drop=True)
    
    # Similarity between distributions
    base_imp_s = _.loc[_["missing_id"]==1]
    base_s = df.dropna(subset=vars_input)
    emd = emd_vars(base_s[vars_input].values.flatten(),
                           base_imp_s["imp"].values.flatten())

    return _,emd


def multivariate_imp_neigh(df, country, vars_input, vars_add=None, max_iter=10,min_val=0,time="year",last_train=2021):
    
    ### Split data ###
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]

        # Train, test
        train_s = df_s.loc[df_s[time]<=last_train]
        test_s = df_s.loc[df_s[time]>last_train]

        # Merge
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])
        
    ### Training ###

    # Step 1 Make subset of data for imputation - only explanatory variables
    feat_imp = train[vars_input]
    df_MICE = feat_imp.copy(deep=True)
    df_MICE['lag1'] = lag_groupped(train, country, vars_input, 1)
    df_MICE['lag2'] = lag_groupped(train, country, vars_input, 2)
    df_MICE['lag3'] = lag_groupped(train, country, vars_input, 3)
    df_MICE['lag4'] = lag_groupped(train, country, vars_input, 4)
    df_MICE['lag5'] = lag_groupped(train, country, vars_input, 5)
    dummy_df = pd.get_dummies(train[country], columns=[country])
    dummy_df = dummy_df.astype(int)
    df_MICE = pd.concat([df_MICE, dummy_df], axis=1)
    grouped = train.groupby(country)
    df_MICE['t'] = grouped.cumcount()

    if vars_add is not None:
        df_filled = train.fillna(0)
        df_MICE = pd.concat([df_MICE, df_filled[vars_add]], axis=1)

    # Step 2 Make subset of data which are complete and are not involved in the imputation
    feat_complete = train.drop(columns=vars_input)
    feat_complete = train.drop(columns=vars_add)
    feat_complete=feat_complete.reset_index(drop=True)

    # Step 3 Specify imputer

    # Tune model
    train_val = pd.DataFrame()
    test_val = pd.DataFrame()
    for c in df.country.unique():
        df_s = train.loc[train["country"] == c]

        # Train, test
        train_s = df_s[:int(0.8*len(df_s))]
        test_s = df_s[int(0.8*len(df_s)):]

        # Merge
        train_val = pd.concat([train_val, train_s])
        test_val = pd.concat([test_val, test_s])
        
    # Min-max normalization
    df_fill = df_MICE.fillna(0)
    min_vals = df_fill.min()
    max_vals = df_fill.max()
    df_norm = (df_fill - min_vals) / (max_vals - min_vals)

    train_val_df = df_norm.loc[train_val.index]
    test_val_df = df_norm.loc[test_val.index]
    train_x = train_val_df.drop(columns=vars_input)
    test_x = test_val_df.drop(columns=vars_input)

    mse_test = 1
    for k in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]:        
        grid_search = KNeighborsRegressor(n_neighbors=k)
        grid_search.fit(train_x, train_val_df[vars_input])
        preds = grid_search.predict(test_x)
        mse = mean_squared_error(test_val_df[vars_input], preds)
        change = percentage_decrease(mse_test, mse)
        if change > 1:
            mse_test = mse
            k_final = k
            #print(f"Cluster {k} with MSE {mse}")
        else:
            break
            
    imputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=k_final), random_state=1, max_iter=max_iter,min_value=min_val)

    # Step 4 Run imputation and obtain imputed dataset

    # Min-max normalization
    min_vals = df_MICE.min()
    max_vals = df_MICE.max()
    df_MICE_norm = (df_MICE - min_vals) / (max_vals - min_vals)

    imputer.fit(df_MICE_norm)
    df_MICE_trans = imputer.transform(df_MICE_norm)
    df_MICE_trans_df = pd.DataFrame(df_MICE_trans)
    df_MICE_trans_df = df_MICE_trans_df.iloc[:, :len(vars_input)]
    df_MICE_trans_df = df_MICE_trans_df.rename(columns={0: 'imp'})
        
    # Reverse normalization
    df_MICE_trans_df["imp"] = df_MICE_trans_df["imp"] * (max_vals[vars_input][0] - min_vals[vars_input][0]) + min_vals[vars_input][0]

    # Step 5 Merge imputed dataset based on index
    train_imp=pd.concat([feat_complete, df_MICE_trans_df], axis=1)
    train_imp['missing_id'] = train_imp[vars_input].isnull().astype(int)

    ### Test ###

    # Step 1 Make subset of data for imputation - only explanatory variables
    feat_imp = test[vars_input]
    df_MICE = feat_imp.copy(deep=True)
    df_MICE['lag1'] = lag_groupped(test, country, vars_input, 1)
    df_MICE['lag2'] = lag_groupped(test, country, vars_input, 2)
    df_MICE['lag3'] = lag_groupped(test, country, vars_input, 3)
    df_MICE['lag4'] = lag_groupped(test, country, vars_input, 4)
    df_MICE['lag5'] = lag_groupped(test, country, vars_input, 5)
    dummy_df = pd.get_dummies(test[country], columns=[country])
    dummy_df = dummy_df.astype(int)
    df_MICE = pd.concat([df_MICE, dummy_df], axis=1)
    grouped = test.groupby(country)
    df_MICE['t'] = grouped.cumcount()

    if vars_add is not None:
        df_filled = test.fillna(0)
        df_MICE = pd.concat([df_MICE, df_filled[vars_add]], axis=1)

    # Step 2 Make subset of data which are complete and are not involved in the imputation
    feat_complete = test.drop(columns=vars_input)
    feat_complete = test.drop(columns=vars_add)
    feat_complete=feat_complete.reset_index(drop=True)
    
    # Step 4 Run imputation and obtain imputed dataset

    # Min-max normalization
    #min_vals = df_MICE.min()
    #max_vals = df_MICE.max()
    df_MICE_norm = (df_MICE - min_vals) / (max_vals - min_vals)

    df_MICE_trans = imputer.transform(df_MICE_norm)
    df_MICE_trans_df = pd.DataFrame(df_MICE_trans)
    df_MICE_trans_df = df_MICE_trans_df.iloc[:, :len(vars_input)]
    df_MICE_trans_df = df_MICE_trans_df.rename(columns={0: 'imp'})

    df_MICE_trans_df["imp"] = df_MICE_trans_df["imp"] * (max_vals[vars_input][0] - min_vals[vars_input][0]) + min_vals[vars_input][0]

    # Step 5 Merge imputed dataset based on index
    test_imp=pd.concat([feat_complete, df_MICE_trans_df], axis=1)
    test_imp['missing_id'] = test_imp[vars_input].isnull().astype(int)

    # Merge train and test
    _ = pd.concat([train_imp, test_imp])
    _=_.sort_values(by=["gw_codes",time])
    _=_.reset_index(drop=True)
    
    # Similarity between distributions
    base_imp_s = _.loc[_["missing_id"]==1]
    base_s = df.dropna(subset=vars_input)
    emd = emd_vars(base_s[vars_input].values.flatten(),
                           base_imp_s["imp"].values.flatten())

    return _,emd


def multivariate_imp_tree(df, country, vars_input,vars_add=None, max_iter=10,min_val=0,time="year",last_train=2021):
    
    ### Split data ###
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]

        # Train, test
        train_s = df_s.loc[df_s[time]<=last_train]
        test_s = df_s.loc[df_s[time]>last_train]

        # Merge
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])
              
    ### Training ###

    # Step 1 Make subset of data for imputation - only explanatory variables
    feat_imp = train[vars_input]
    df_MICE = feat_imp.copy(deep=True)
    df_MICE['lag1'] = lag_groupped(train, country, vars_input, 1)
    df_MICE['lag2'] = lag_groupped(train, country, vars_input, 2)
    df_MICE['lag3'] = lag_groupped(train, country, vars_input, 3)
    df_MICE['lag4'] = lag_groupped(train, country, vars_input, 4)
    df_MICE['lag5'] = lag_groupped(train, country, vars_input, 5)
    dummy_df = pd.get_dummies(train[country], columns=[country])
    dummy_df = dummy_df.astype(int)
    df_MICE = pd.concat([df_MICE, dummy_df], axis=1)
    grouped = train.groupby(country)
    df_MICE['t'] = grouped.cumcount()

    if vars_add is not None:
        df_filled = train.fillna(0)
        df_MICE = pd.concat([df_MICE, df_filled[vars_add]], axis=1)

    # Step 2 Make subset of data which are complete and are not involved in the imputation
    feat_complete = train.drop(columns=vars_input)
    feat_complete = train.drop(columns=vars_add)
    feat_complete=feat_complete.reset_index(drop=True)

    # Step 3 Specify imputer

    # Tune model
    train_val = pd.DataFrame()
    test_val = pd.DataFrame()
    for c in df.country.unique():
        df_s = train.loc[train["country"] == c]

        # Train, test
        train_s = df_s[:int(0.8*len(df_s))]
        test_s = df_s[int(0.8*len(df_s)):]
        
        # Merge
        train_val = pd.concat([train_val, train_s])
        test_val = pd.concat([test_val, test_s])

    splits = np.array([-1] * len(train_val.index) + [0] * len(test_val.index))
    ps = PredefinedSplit(test_fold=splits)

    random_grid = {'n_estimators': [50,100,200,500],
                    'max_depth': [3,5,10,None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                    }
        
    df_input = df_MICE.fillna(0)
    df_input_x = df_input.drop(columns=vars_input)
    with parallel_backend('threading'):
        grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=random_grid, cv=ps, verbose=0, n_jobs=-1)
        #grid_search = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=random_grid, cv=ps, verbose=0, n_jobs=-1,n_iter=50)
        grid_search.fit(df_input_x, df_input[vars_input].values.ravel())
    best_params = grid_search.best_params_

    imputer = IterativeImputer(estimator=RandomForestRegressor(**best_params, random_state=0),random_state=1, max_iter=max_iter, min_value=min_val)

    # Step 4 Run imputation and obtain imputed dataset
    imputer.fit(df_MICE)
    df_MICE_trans = imputer.transform(df_MICE)
    df_MICE_trans_df = pd.DataFrame(df_MICE_trans)
    df_MICE_trans_df = df_MICE_trans_df.iloc[:, :len(vars_input)]
    df_MICE_trans_df = df_MICE_trans_df.rename(columns={0: 'imp'})

    # Step 5 Merge imputed dataset based on index
    train_imp=pd.concat([feat_complete, df_MICE_trans_df], axis=1)
    train_imp['missing_id'] = train_imp[vars_input].isnull().astype(int)
    
    ### Test ###

    # Step 1 Make subset of data for imputation - only explanatory variables
    feat_imp = test[vars_input]
    df_MICE = feat_imp.copy(deep=True)
    df_MICE['lag1'] = lag_groupped(test, country, vars_input, 1)
    df_MICE['lag2'] = lag_groupped(test, country, vars_input, 2)
    df_MICE['lag3'] = lag_groupped(test, country, vars_input, 3)
    df_MICE['lag4'] = lag_groupped(test, country, vars_input, 4)
    df_MICE['lag5'] = lag_groupped(test, country, vars_input, 5)
    dummy_df = pd.get_dummies(test[country], columns=[country])
    dummy_df = dummy_df.astype(int)
    df_MICE = pd.concat([df_MICE, dummy_df], axis=1)
    grouped = test.groupby(country)
    df_MICE['t'] = grouped.cumcount()

    if vars_add is not None:
        df_filled = test.fillna(0)
        df_MICE = pd.concat([df_MICE, df_filled[vars_add]], axis=1)

    # Step 2 Make subset of data which are complete and are not involved in the imputation
    feat_complete = test.drop(columns=vars_input)
    feat_complete = test.drop(columns=vars_add)
    feat_complete=feat_complete.reset_index(drop=True)
    
    # Step 4 Run imputation and obtain imputed dataset
    df_MICE_trans = imputer.transform(df_MICE)
    df_MICE_trans_df = pd.DataFrame(df_MICE_trans)
    df_MICE_trans_df = df_MICE_trans_df.iloc[:, :len(vars_input)]
    df_MICE_trans_df = df_MICE_trans_df.rename(columns={0: 'imp'})

    # Step 5 Merge imputed dataset based on index
    test_imp=pd.concat([feat_complete, df_MICE_trans_df], axis=1)
    test_imp['missing_id'] = test_imp[vars_input].isnull().astype(int)

    # Merge train and test
    _ = pd.concat([train_imp, test_imp])
    _=_.sort_values(by=["gw_codes",time])
    _=_.reset_index(drop=True)
    
    # Similarity between distributions
    base_imp_s = _.loc[_["missing_id"]==1]
    base_s = df.dropna(subset=vars_input)
    emd = emd_vars(base_s[vars_input].values.flatten(),
                           base_imp_s["imp"].values.flatten())

    return _,emd

def imp_opti(df, country, vars_input, vars_add=None, max_iter=10,min_val=0,time="year",last_train=2021):
        
    _,emd_neigh=multivariate_imp_neigh(df,country,vars_input,vars_add,max_iter,min_val,time,last_train)
    _,emd_tree=multivariate_imp_tree(df,country,vars_input,vars_add,max_iter,min_val,time,last_train)
    _,emd_linear=multivariate_imp_bayes(df,country,vars_input,vars_add,max_iter,min_val,time,last_train)
    
    select=min(emd_neigh,emd_tree,emd_linear)
    
    if select==emd_neigh:
        print("Selected method: Neighbor")
        return _        
    
    elif select==emd_tree:
        print("Selected method: Tree")
        return _        
        
    elif select==emd_linear:
        print("Selected method: Linear")
        return _        

def simple_imp_grouped(df, group, vars_input,time="year",last_train=2021):
    
    ### Split data ###
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]

        # Train, test
        train_s = df_s.loc[df_s[time]<=last_train]
        test_s = df_s.loc[df_s[time]>last_train]
        
        # Merge
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])
        
    ### Training ###

    df_filled = pd.DataFrame()

    for c in df[group].unique():
                
        # Train
        df_s = train.loc[train[group] == c]
        feat_imp = df_s[vars_input]
        df_MICE = feat_imp.copy(deep=True)
        
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(df_MICE)
        df_MICE_train = imputer.transform(df_MICE)
        df_MICE_train_df = pd.DataFrame(df_MICE_train)
        
        # Test
        df_s = test.loc[test[group] == c]
        feat_imp = df_s[vars_input]
        df_MICE = feat_imp.copy(deep=True)        

        df_MICE_test = imputer.transform(df_MICE)
        df_MICE_test_df = pd.DataFrame(df_MICE_test)        

        # Merge
        df_MICE_trans_df = pd.concat([df_MICE_train_df, df_MICE_test_df])
        df_filled = pd.concat([df_filled, df_MICE_trans_df])

    # Merge
    df_filled.columns = vars_input
    df_filled = df_filled.reset_index(drop=True)
    feat_complete = df.drop(columns=vars_input)
    _ = pd.concat([feat_complete, df_filled], axis=1)
    _=_.reset_index(drop=True)
    
    return _

def linear_imp_grouped(df, group, vars_input,time="year",last_train=2021):
    
    ### Split data ###
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]

        # Train, test
        train_s = df_s.loc[df_s[time]<=last_train]
        test_s = df_s.loc[df_s[time]>last_train]

        # Merge
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])
    
    df_filled = pd.DataFrame()

    for c in df[group].unique():
        
        # Train
        df_s = train.loc[train[group] == c]
        feat_imp = df_s[vars_input]
        df_MICE = feat_imp.copy(deep=True)
        df_MICE_train_df = df_MICE.interpolate(limit_direction="both")
        
        # Test
        df_s = test.loc[test[group] == c]
        feat_imp = df_s[vars_input]
        df_MICE = feat_imp.copy(deep=True)        
        df_MICE_test_df = df_MICE.interpolate(limit_direction="forward")
        
        # Merge
        df_MICE_trans_df = pd.concat([df_MICE_train_df, df_MICE_test_df])
        df_filled = pd.concat([df_filled, df_MICE_trans_df])
        
    # Merge
    df_filled.columns = vars_input
    df_filled = df_filled.reset_index(drop=True)
    feat_complete = df.drop(columns=vars_input)
    _ = pd.concat([feat_complete, df_filled], axis=1)
    _=_.reset_index(drop=True)
    
    return _

def calibrate_imp(df_imp, group, var):
    
    df_imp["calib"]=df_imp[var]
    
    # Group by 'Group' column
    grouped = df_imp.groupby(group)
    
    # Compute mean and variance of observed values within each group
    observed_mean = grouped[var].mean()
    observed_variance = grouped[var].var()
    
    # Compute mean and variance of imputed values within each group
    imputed_mean = grouped["imp"].mean()
    imputed_variance = grouped["imp"].var()
    
    observed_variance[observed_variance.isnull()] = imputed_variance[imputed_variance.isnull()]
    observed_mean[observed_mean.isnull()] = imputed_mean[imputed_mean.isnull()]
    
    # Rescale imputed values within each group
    for group_name, group_data in grouped:
        # Calculate scaling factors
        mean_ratio = observed_mean[group_name] / (imputed_mean[group_name]+0.0000000001)
        variance_ratio = observed_variance[group_name] / (imputed_variance[group_name]+0.0000000001)
    
        # Rescale imputed values
        df_imp.loc[group_data.index, "calib"] *= mean_ratio
        df_imp.loc[group_data.index, "calib"] = df_imp.loc[group_data.index, 'imp'] * variance_ratio
    
    df_imp[var] = df_imp[var].fillna(df_imp["calib"])
    df_imp[var] = df_imp[var].fillna(df_imp['imp'])
    
    df_imp = df_imp.drop(columns=['imp',"calib"])

    return df_imp

###################
### Evaluations ###
###################

def earth_mover_distance(actuals, preds):
    a_hist, a_edges = np.histogram(actuals, bins=10, density=True)
    p_hist, p_edges = np.histogram(preds, bins=10, density=True)
    emd = wasserstein_distance(
        p_edges[:-1], a_edges[:-1], u_weights=p_hist, v_weights=a_hist)
    return (emd)

##############################
### General model function ###
##############################

def gen_model(y, x, target, inputs, name, names=None, model_fit=LinearRegression(), opti_grid=None,outcome="categorical",downsampling=False,time="year",last_train=2020):

    if downsampling==False: 
        # Data split
        train_y = pd.DataFrame()
        test_y = pd.DataFrame()
        train_x = pd.DataFrame()
        test_x = pd.DataFrame()
        
        val_train_index = []
        val_test_index = []
        
        for c in y.country.unique():
            y_s = y.loc[y["country"] == c]
            x_s = x.loc[x["country"] == c]
        
            # Train, test
            y_train = y_s[["country",time]+[target]].loc[y_s[time]<f"{last_train}-01"]
            x_train = x_s[["country",time]+inputs].loc[x_s[time]<f"{last_train}-01"]
            y_test = y_s[["country",time]+[target]].loc[(y_s["dd"]>=f"{last_train}-01")]
            x_test = x_s[["country",time]+inputs].loc[(x_s["dd"]>=f"{last_train}-01")]
            # Merge
            train_y = pd.concat([train_y, y_train])
            test_y = pd.concat([test_y, y_test])
            train_x = pd.concat([train_x, x_train])
            test_x = pd.concat([test_x, x_test])
        
            # Validation
            val_train_index += list(y_train[:int(0.8*len(y_train))].index)
            val_test_index += list(y_train[int(0.8*len(y_train)):].index)
        
        splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
        
        # Train model 
        train_y_d=train_y.drop(columns=["country",time])
        train_x_d=train_x.drop(columns=["country",time])
        test_y_d=test_y.drop(columns=["country",time])
        test_x_d=test_x.drop(columns=["country",time])
         
    else: 
        # Data split
        train_y = pd.DataFrame()
        test_y = pd.DataFrame()
        train_x = pd.DataFrame()
        test_x = pd.DataFrame()
        
        val_train_index = []
        val_test_index = []
        
        for c in y.country.unique():
            y_s = y.loc[y["country"] == c]
            x_s = x.loc[x["country"] == c]
        
            # Train, test
            y_train = y_s[["country",time]+[target]].loc[y_s[time]<f"{last_train}-01"]
            x_train = x_s[["country",time]+inputs].loc[y_s[time]<f"{last_train}-01"]
            y_test = y_s[["country",time]+[target]].loc[(y_s["dd"]>=f"{last_train}-01")]
            x_test = x_s[["country",time]+inputs].loc[(y_s["dd"]>=f"{last_train}-01")]
            # Merge
            train_y = pd.concat([train_y, y_train])
            test_y = pd.concat([test_y, y_test])
            train_x = pd.concat([train_x, x_train])
            test_x = pd.concat([test_x, x_test])
        
            # Validation
            val_train_index += list(y_train[:int(0.8*len(y_train))].index)
            val_test_index += list(y_train[int(0.8*len(y_train)):].index)
        
        splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
        train_y["splits"]=splits
        train_x["splits"]=splits
        
        # Upsample the minority class
        X_upsampled, y_upsampled = resample(train_x[train_y[target] == 1], train_y[train_y[target] == 1], 
                                            replace=True,  # Sample with replacement
                                            n_samples=len(train_x[train_y[target] == 0]),  # Match number of samples in majority class
                                            random_state=42)    
        
        # Combine the upsampled minority class with the majority class
        X_balanced = pd.concat([train_x[train_y[target] == 0], X_upsampled])
        y_balanced = pd.concat([train_y[train_y[target] == 0], y_upsampled])
        
        # Train model 
        train_y_d=y_balanced[["country",time]+[target]].drop(columns=["country",time])
        train_x_d=X_balanced[["country",time]+inputs].drop(columns=["country",time])
        splits=np.array(X_balanced.splits)    
        
        test_y_d=test_y.drop(columns=["country",time])
        test_x_d=test_x.drop(columns=["country",time])
        train_y=train_y.drop(columns=["splits"])
        train_x=train_x.drop(columns=["splits"])
                
    if opti_grid is not None:
        ps = PredefinedSplit(test_fold=splits)
        with parallel_backend('threading'):
            grid_search = GridSearchCV(estimator=model_fit, param_grid=opti_grid, cv=ps, verbose=0, n_jobs=-1)
            #grid_search = RandomizedSearchCV(estimator=model_fit, param_distributions=opti_grid, cv=ps, verbose=0, n_jobs=-1,n_iter=50)
            grid_search.fit(train_x_d, train_y_d.values.ravel())
        best_params = grid_search.best_params_
        model_fit.set_params(**best_params)
        model = model_fit
        model.fit(train_x_d, train_y_d.values.ravel())
        pred = pd.DataFrame(model.predict(test_x_d))
        if downsampling==False:
            pred_in_sample = pd.DataFrame(model.predict(train_x_d))
        else:
            # Data split
            train_y = pd.DataFrame()
            train_x = pd.DataFrame()

            for c in y.country.unique():
                y_s = y.loc[y["country"] == c]
                x_s = x.loc[x["country"] == c]
                y_train = y_s[["country",time]+[target]].loc[y_s[time]<f"{last_train}-01"]
                x_train = x_s[["country",time]+inputs].loc[y_s[time]<f"{last_train}-01"]
                train_y = pd.concat([train_y, y_train])
                train_x = pd.concat([train_x, x_train])
            train_y_d=train_y.drop(columns=["country",time])
            train_x_d=train_x.drop(columns=["country",time])
            pred_in_sample = pd.DataFrame(model.predict(train_x_d))
       
        # Get df, out of sample
        pred["test"]=1
        pred_df = pd.concat([test_y.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)

        # Get df, in sample
        pred_in_sample["test"]=0
        pred_in_sample_df = pd.concat([train_y.reset_index(drop=True), pred_in_sample.reset_index(drop=True)], axis=1)

        # Merge
        out_df=pd.concat([pred_in_sample_df,pred_df], axis=0)
        out_df.columns = ["country",time,target,"preds","test"]
        
        if outcome=="categorical":
            pred_proba=pd.DataFrame(model.predict_proba(test_x_d)[:, 1])
            if downsampling==False:
                pred_proba_in_sample=pd.DataFrame(model.predict_proba(train_x_d)[:, 1])
            else:
                # Data split
                train_y = pd.DataFrame()
                train_x = pd.DataFrame()

                for c in y.country.unique():
                    y_s = y.loc[y["country"] == c]
                    x_s = x.loc[x["country"] == c]
                    y_train = y_s[["country",time]+[target]].loc[y_s[time]<f"{last_train}-01"]
                    x_train = x_s[["country",time]+inputs].loc[y_s[time]<f"{last_train}-01"]
                    train_y = pd.concat([train_y, y_train])
                    train_x = pd.concat([train_x, x_train])
                train_y_d=train_y.drop(columns=["country",time])
                train_x_d=train_x.drop(columns=["country",time])
                pred_proba_in_sample=pd.DataFrame(model.predict_proba(train_x_d)[:, 1])
                
            pred_proba_in_sample=pd.DataFrame(model.predict_proba(train_x_d)[:, 1])
            out_df_proba = pd.concat([pred_proba_in_sample,pred_proba], axis=0)

            # Merge
            out_df=pd.concat([out_df.reset_index(drop=True),out_df_proba.reset_index(drop=True)], axis=1)
            out_df.columns = ["country",time,target,"preds","test","preds_proba"]  
            out_df=out_df.sort_values(by=["country",time])
   
    else:
        model = model_fit
        model.fit(train_x_d, train_y_d)
        pred = pd.DataFrame(model.predict(test_x_d))
        pred_in_sample = pd.DataFrame(model.predict(train_x_d))
        if downsampling==False:
            pred_in_sample = pd.DataFrame(model.predict(train_x_d))
        else:
            # Data split
            train_y = pd.DataFrame()
            train_x = pd.DataFrame()

            for c in y.country.unique():
                y_s = y.loc[y["country"] == c]
                x_s = x.loc[x["country"] == c]
                y_train = y_s[["country",time]+[target]].loc[y_s[time]<f"{last_train}-01"]
                x_train = x_s[["country",time]+inputs].loc[y_s[time]<f"{last_train}-01"]
                train_y = pd.concat([train_y, y_train])
                train_x = pd.concat([train_x, x_train])
            train_y_d=train_y.drop(columns=["country",time])
            train_x_d=train_x.drop(columns=["country",time])
            pred_in_sample = pd.DataFrame(model.predict(train_x_d))

        # Get df, out of sample
        pred["test"]=1
        pred_df = pd.concat([test_y.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)

        # Get df, in sample
        pred_in_sample["test"]=0
        pred_in_sample_df = pd.concat([train_y.reset_index(drop=True), pred_in_sample.reset_index(drop=True)], axis=1)

        # Merge
        out_df=pd.concat([pred_in_sample_df,pred_df], axis=0)
        out_df.columns = ["country",time,target,"preds","test"]
        
        if outcome=="categorical":
            pred_proba=pd.DataFrame(model.predict_proba(test_x_d)[:, 1])
            if downsampling==False:
                pred_proba_in_sample=pd.DataFrame(model.predict_proba(train_x_d)[:, 1])
            else:
                # Data split
                train_y = pd.DataFrame()
                train_x = pd.DataFrame()

                for c in y.country.unique():
                    y_s = y.loc[y["country"] == c]
                    x_s = x.loc[x["country"] == c]
                    y_train = y_s[["country",time]+[target]].loc[y_s[time]<f"{last_train}-01"]
                    x_train = x_s[["country",time]+inputs].loc[y_s[time]<f"{last_train}-01"]
                    train_y = pd.concat([train_y, y_train])
                    train_x = pd.concat([train_x, x_train])
                train_y_d=train_y.drop(columns=["country",time])
                train_x_d=train_x.drop(columns=["country",time])
                pred_proba_in_sample=pd.DataFrame(model.predict_proba(train_x_d)[:, 1])
                
            pred_proba_in_sample=pd.DataFrame(model.predict_proba(train_x_d)[:, 1])
            out_df_proba = pd.concat([pred_proba_in_sample,pred_proba], axis=0)

            # Merge
            out_df=pd.concat([out_df.reset_index(drop=True),out_df_proba.reset_index(drop=True)], axis=1)
            out_df.columns = ["country",time,target,"preds","test","preds_proba"]  
            out_df=out_df.sort_values(by=["country",time])
            
    ##################
    ### Evaluation ###
    ##################

    if outcome == "continous":
        val=out_df.loc[(out_df[time]>=f"{last_train}-01")&(out_df[time]<=f"{last_train+1}-12")]
        # MSE
        mse = mean_squared_error(val[target], val["preds"])
        # WMSE
        wmse = mean_squared_error(val[target], val["preds"], sample_weight=val[target])
        # EMD
        emd = earth_mover_distance(val[target], val["preds"])
        # Evals dictionary
        evals = {"mse": mse, "wmse": wmse, "emd": emd}

    elif outcome == "categorical":
        val=out_df.loc[(out_df[time]>=f"{last_train}-01")&(out_df[time]<=f"{last_train+1}-12")]

        # Brier
        brier = brier_score_loss(val[target], val["preds_proba"])
        # AUPR
        precision, recall, _ = precision_recall_curve(val[target], val["preds_proba"])
        aupr = auc(recall, precision)
        # AUROC
        auroc = roc_auc_score(val[target], val["preds_proba"])
        # Evals dictionary
        evals = {"brier": brier, "aupr": aupr, "auroc": auroc}
            
    return out_df, evals






