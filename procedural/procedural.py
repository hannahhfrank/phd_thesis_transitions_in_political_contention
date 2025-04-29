import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import math
import matplotlib.pyplot as plt
from functions import lag_groupped
import random
random.seed(42)
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import matplotlib as mpl
import os
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'


plt.rcParams['xtick.labelsize'] = 20  # X-axis tick label size
plt.rcParams['ytick.labelsize'] = 20  # Y-axis tick label size

# Simple random forest
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)],
               #'max_features': ["sqrt", "log2", None],
               #'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],
               #'min_samples_split': [2,5,10],
               #'min_samples_leaf': [1,2,4],
               }
 
def evals(y_true, y_pred, countries, onset_tolerance=3, alpha=0.7, beta=0.3, horizon=24):

    unique_countries = countries.unique()
    mse_values = []
    final_scores_by_country = []
    d_nn=[]

    for country in unique_countries:
     # Filter data for the current country
        country_mask = countries == country
        y_true_country = y_true[country_mask]
        y_pred_country = y_pred[country_mask]

        # Detect Onsets in Ground Truth and Predictions for the Country
        true_onsets = y_true_country[(y_true_country.shift(1) == 0) & (y_true_country > 0)].index
        pred_onsets = y_pred_country[(y_pred_country.shift(1) == 0) & (y_pred_country > 0)].index
        
        # Calculate Onset Score
        correct_onsets = []
        for true_onset in list(true_onsets):
            # Find the closest predicted onset within the tolerance
            closest_pred = pred_onsets[np.abs(pred_onsets - true_onset) <= onset_tolerance]
            if not closest_pred.empty:
                delta_t = np.abs(closest_pred[0] - true_onset)
                correct_onsets.append(np.exp(-0.1*delta_t))  # Exponential penalty for timing error

        onset_score = np.sum(correct_onsets) / len(true_onsets) if len(true_onsets) > 0 else 0

        # Compute MSE for the current country
        mse = np.mean((y_true_country.values-y_pred_country.values) ** 2)
        mse_values.append(mse)
            
        # Normalize MSE for the Current Country
        # Min and max are calculated for the country's data
        if y_true_country.max() == y_true_country.min() and y_pred_country.max() == y_pred_country.min():
            normalized_true = pd.Series([0] * len(y_true_country))  # or assign 1 if you prefer
            normalized_pred = pd.Series([0] * len(y_pred_country))  # or assign 1 if you prefer
    
        elif y_true_country.max() == y_true_country.min() and y_pred_country.max() != y_pred_country.min():
            normalized_true = pd.Series([0] * len(y_true_country))  # or assign 1 if you prefer
            normalized_pred = (y_pred_country - y_pred_country.min()) / (y_pred_country.max() - y_pred_country.min()) 
        
        elif y_true_country.max() != y_true_country.min() and y_pred_country.max() == y_pred_country.min():
            normalized_true = (y_true_country - y_true_country.min()) / (y_true_country.max() - y_true_country.min()) 
            normalized_pred = pd.Series([0] * len(y_pred_country))  # or assign 1 if you prefer
          
        elif y_true_country.max() != y_true_country.min() and y_pred_country.max() != y_pred_country.min():
            normalized_true = (y_true_country - y_true_country.min()) / (y_true_country.max() - y_true_country.min()) 
            normalized_pred = (y_pred_country - y_pred_country.min()) / (y_pred_country.max() - y_pred_country.min()) 
        normalized_mse = np.mean((normalized_true.values - normalized_pred.values) ** 2)
        
        # DE
        real = y_true_country
        real=real.reset_index(drop=True)
        sf = y_pred_country
        sf=sf.reset_index(drop=True)

        max_s=0
        if (real==0).all()==False:
            for value in real[1:].index:
                if (real[value]==real[value-1]):
                    1
                else:
                    max_exp=0
                    if (real[value]-real[value-1])/(sf[value]-sf[value-1])>0 and sf[value]-sf[value-1] != 0:
                        t=abs(((real[value]-real[value-1])-(sf[value]-sf[value-1]))/(real[value]-real[value-1]))
                        max_exp = np.exp(-(5*t))
                    else:
                        if value==horizon-1:
                            if (real[value]-real[value-1])/(sf[value-1]-sf[value-2])>0 and sf[value-1]-sf[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value-1]-sf[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(5*t)):
                                    max_exp = np.exp(-(5*t))
                        elif value==1:
                            if (real[value]-real[value-1])/(sf[value+1]-sf[value])>0 and sf[value+1]-sf[value] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value+1]-sf[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(5*t)):
                                    max_exp = np.exp(-(5*t))
                        else : 
                            if (real[value]-real[value-1])/(sf[value-1]-sf[value-2])>0 and sf[value-1]-sf[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value-1]-sf[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(5*t)):
                                    max_exp = np.exp(-(5*t))
                            if (real[value]-real[value-1])/(sf[value+1]-sf[value])>0 and sf[value+1]-sf[value] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value+1]-sf[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(5*t)):
                                    max_exp = np.exp(-(5*t))
                    max_s=max_s+max_exp 
            d_nn.append(max_s)
        else:
            d_nn.append(0)           
        
        # Calculate Final Score for the Current Country
        final_score = alpha * onset_score + beta * (1 - normalized_mse)
        final_scores_by_country.append(final_score)
         
    # Calculate the mean MSE across all countries
    final_score=np.mean(final_scores_by_country)
    de_mean = np.mean(d_nn)


    return final_score, de_mean

# Get data
ensemble_ens=pd.read_csv("out/ensemble_ens_df_cm.csv",index_col=0)
df=pd.read_csv("data/data_out/ucdp_cm_sb.csv",index_col=0)
df=pd.merge(ensemble_ens, df[["country","dd","best"]],on=["country","dd"],how="left")
df.columns=["country","dd","preds_proba","dummy","year","sb_fatalities"]
df=df[["country","dd","year","preds_proba","sb_fatalities"]]

# Plot
selects=random.choices(ensemble_ens.country.unique(), k=10)

ensemble_ens['year'] = ensemble_ens['dd'].str[:4]
for c in ensemble_ens.country.unique():
    years=ensemble_ens["year"].loc[ensemble_ens["country"]==c].unique()
    n_plots = len(years)
    n_cols = 5
    n_rows = math.ceil(n_plots / n_cols) 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
    axes = axes.flatten()
    for i,y in zip(range(n_plots),years):
        df_s=ensemble_ens.loc[(ensemble_ens["country"]==c)&(ensemble_ens["year"]==y)]
        axes[i].plot(df_s.dd,df_s.preds_proba,color="black")
        axes[i].axis('off')
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])
    #plt.suptitle(c, fontsize=30)
    if c in selects:
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/procedural/out/proc_latent_{c}.eps",dpi=300,bbox_inches='tight')        
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/proc_latent_{c}.eps",dpi=300,bbox_inches='tight')
        plt.savefig(f"out/proc_latent_{c}.eps",dpi=300,bbox_inches='tight')    
    plt.show()   


# Get Onset catcher predictions

#################
### Test 2022 ###
#################

test_score=0
w1, w2 = 0.8, 0.2

#w1, w2 = 0.8, 0.2
for ar in [2,3,4,5,6,7,8,9,10]:
    lags=[]

    for i in range(1,ar):
        df[f"preds_proba_lag{i}"]=lag_groupped(df,"country","preds_proba",i)
        lags.append(f"preds_proba_lag{i}")
        
    for cut in [0.01,0.05,0.1,0.15,0.2]:
        df_zero=df.loc[df["preds_proba"]<=cut]
        df_nonzero=df.loc[df["preds_proba"]>cut]
    
        # Data split
        train_y = pd.DataFrame()
        test_y = pd.DataFrame()
        train_x = pd.DataFrame()
        test_x = pd.DataFrame()
        train_y_val=pd.DataFrame()
        train_x_val=pd.DataFrame()
            
        val_train_index = []
        val_test_index = []
            
        for c in df.country.unique():
            df_s = df.loc[df["country"] == c]
            df_nonzero_s = df_nonzero.loc[df_nonzero["country"] == c]
            
            # Train, test
            y_train = df_nonzero_s[["country","dd","sb_fatalities"]].loc[df_nonzero_s["dd"]<"2020-01"]
            x_train = df_nonzero_s[["country","dd"]+lags].loc[df_nonzero_s["dd"]<"2020-01"]
            y_test = df_nonzero_s[["country","dd","sb_fatalities"]].loc[(df_nonzero_s["dd"]>="2020-01")]
            x_test = df_nonzero_s[["country","dd"]+lags].loc[(df_nonzero_s["dd"]>="2020-01")]
            # Merge
            train_y = pd.concat([train_y, y_train])
            test_y = pd.concat([test_y, y_test])
            train_x = pd.concat([train_x, x_train])
            test_x = pd.concat([test_x, x_test])
            
            # Validation
            y_train_val = df_nonzero_s[["country","dd","sb_fatalities"]].loc[df_nonzero_s["dd"]<"2022-01"]
            x_train_val = df_nonzero_s[["country","dd"]+lags].loc[df_nonzero_s["dd"]<"2022-01"]        
            val_train_index += list(y_train.loc[y_train["dd"]<"2020-01"].index)
            val_test_index += list(y_test.loc[(y_test["dd"]>="2020-01")&(y_test["dd"]<="2021-12")].index)
            train_y_val = pd.concat([train_y_val, y_train_val])
            train_x_val = pd.concat([train_x_val, x_train_val])
    
            
        splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
        
        # Train model 
        train_y_d=train_y.drop(columns=["country","dd"])
        train_x_d=train_x.drop(columns=["country","dd"])
        train_y_val_d=train_y_val.drop(columns=["country","dd"])
        train_x_val_d=train_x_val.drop(columns=["country","dd"])
        
        # Get test for last 12 months
        test_y_s=test_y.loc[(test_y["dd"]>="2021-01")&(test_y["dd"]<="2021-12")]
        test_y_d=test_y_s.drop(columns=["country","dd"])
        test_x_s=test_x.loc[(test_x["dd"]>="2021-01")&(test_x["dd"]<="2021-12")]
        test_x_d=test_x_s.drop(columns=["country","dd"])
        
        # Train model  
        ps = PredefinedSplit(test_fold=splits)
        grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=1), param_grid=random_grid, cv=ps, verbose=0, n_jobs=-1)
        grid_search.fit(train_x_val_d, train_y_val_d.values.ravel())
        best_params = grid_search.best_params_
        model=RandomForestRegressor(random_state=1,**best_params)
        model.fit(train_x_d, train_y_d.values.ravel())
        
        # Predictions
        pred = pd.DataFrame(model.predict(test_x_d))
        pred["country"]=test_y_s.country.values
        pred["dd"]=test_y_s.dd.values
        pred['dd'] = pred['dd'].str.replace('2021', '2022')
        base=df[["country","dd","sb_fatalities"]].loc[(df["dd"]>="2022-01")&(df["dd"]<="2022-12")]
        catcher=pd.merge(base,pred,on=["country","dd"],how="left")
        catcher.columns=["country","dd","sb_fatalities","preds"]
        catcher=catcher.fillna(0)
        catcher=catcher.sort_values(by=["country","dd"])
        
        # Validation
        test_y_s=test_y.loc[(test_y["dd"]>="2020-01")&(test_y["dd"]<="2021-12")]
        test_y_d=test_y_s.drop(columns=["country","dd"])
        test_x_s=test_x.loc[(test_x["dd"]>="2020-01")&(test_x["dd"]<="2021-12")]
        test_x_d=test_x_s.drop(columns=["country","dd"])
        pred = pd.DataFrame(model.predict(test_x_d))
        pred["country"]=test_y_s.country.values
        pred["dd"]=test_y_s.dd.values
        base=df[["country","dd","sb_fatalities"]].loc[(df["dd"]>="2020-01")&(df["dd"]<="2021-12")]
        val=pd.merge(base,pred,on=["country","dd"],how="left")
        val.columns=["country","dd","sb_fatalities","preds"]
        val=val.fillna(0)
        val=val.sort_values(by=["country","dd"])
        
        onset, de = evals(val.sb_fatalities, val.preds, val.country)
        print(onset,de)
        score = w1 * onset + w2 * de
        print(score)
        
        if score>test_score:
            test_score=score
            print(f"Best: ar={ar}, cut={cut}")
            catcher.to_csv("out/catcher_2022.csv") 
            
    
#################
### Test 2023 ###
#################

test_score=0
w1, w2 = 0.8, 0.2

#w1, w2 = 0.8, 0.2
for ar in [2,3,4,5,6,7,8,9,10]:
    lags=[]

    for i in range(1,ar):
        df[f"preds_proba_lag{i}"]=lag_groupped(df,"country","preds_proba",i)
        lags.append(f"preds_proba_lag{i}")
        
    for cut in [0.01,0.05,0.1,0.15,0.2]:
        df_zero=df.loc[df["preds_proba"]<=cut]
        df_nonzero=df.loc[df["preds_proba"]>cut]
    
        # Data split
        train_y = pd.DataFrame()
        test_y = pd.DataFrame()
        train_x = pd.DataFrame()
        test_x = pd.DataFrame()
        train_y_val=pd.DataFrame()
        train_x_val=pd.DataFrame()
            
        val_train_index = []
        val_test_index = []
            
        for c in df.country.unique():
            df_s = df.loc[df["country"] == c]
            df_nonzero_s = df_nonzero.loc[df_nonzero["country"] == c]
            
            # Train, test
            y_train = df_nonzero_s[["country","dd","sb_fatalities"]].loc[df_nonzero_s["dd"]<"2021-01"]
            x_train = df_nonzero_s[["country","dd"]+lags].loc[df_nonzero_s["dd"]<"2021-01"]
            y_test = df_nonzero_s[["country","dd","sb_fatalities"]].loc[(df_nonzero_s["dd"]>="2021-01")]
            x_test = df_nonzero_s[["country","dd"]+lags].loc[(df_nonzero_s["dd"]>="2021-01")]
            # Merge
            train_y = pd.concat([train_y, y_train])
            test_y = pd.concat([test_y, y_test])
            train_x = pd.concat([train_x, x_train])
            test_x = pd.concat([test_x, x_test])
            
            # Validation
            y_train_val = df_nonzero_s[["country","dd","sb_fatalities"]].loc[df_nonzero_s["dd"]<"2023-01"]
            x_train_val = df_nonzero_s[["country","dd"]+lags].loc[df_nonzero_s["dd"]<"2023-01"]        
            val_train_index += list(y_train.loc[y_train["dd"]<"2021-01"].index)
            val_test_index += list(y_test.loc[(y_test["dd"]>="2021-01")&(y_test["dd"]<="2022-12")].index)
            train_y_val = pd.concat([train_y_val, y_train_val])
            train_x_val = pd.concat([train_x_val, x_train_val])
    
            
        splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
        
        # Train model 
        train_y_d=train_y.drop(columns=["country","dd"])
        train_x_d=train_x.drop(columns=["country","dd"])
        train_y_val_d=train_y_val.drop(columns=["country","dd"])
        train_x_val_d=train_x_val.drop(columns=["country","dd"])
        
        # Get test for last 12 months
        test_y_s=test_y.loc[(test_y["dd"]>="2022-01")&(test_y["dd"]<="2022-12")]
        test_y_d=test_y_s.drop(columns=["country","dd"])
        test_x_s=test_x.loc[(test_x["dd"]>="2022-01")&(test_x["dd"]<="2022-12")]
        test_x_d=test_x_s.drop(columns=["country","dd"])
        
        # Train model  
        ps = PredefinedSplit(test_fold=splits)
        grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=1), param_grid=random_grid, cv=ps, verbose=0, n_jobs=-1)
        grid_search.fit(train_x_val_d, train_y_val_d.values.ravel())
        best_params = grid_search.best_params_
        model=RandomForestRegressor(random_state=1,**best_params)
        model.fit(train_x_d, train_y_d.values.ravel())
        
        # Predictions
        pred = pd.DataFrame(model.predict(test_x_d))
        pred["country"]=test_y_s.country.values
        pred["dd"]=test_y_s.dd.values
        pred['dd'] = pred['dd'].str.replace('2022', '2023')
        base=df[["country","dd","sb_fatalities"]].loc[(df["dd"]>="2023-01")&(df["dd"]<="2023-12")]
        catcher=pd.merge(base,pred,on=["country","dd"],how="left")
        catcher.columns=["country","dd","sb_fatalities","preds"]
        catcher=catcher.fillna(0)
        catcher=catcher.sort_values(by=["country","dd"])
        
        # Validation
        test_y_s=test_y.loc[(test_y["dd"]>="2021-01")&(test_y["dd"]<="2022-12")]
        test_y_d=test_y_s.drop(columns=["country","dd"])
        test_x_s=test_x.loc[(test_x["dd"]>="2021-01")&(test_x["dd"]<="2022-12")]
        test_x_d=test_x_s.drop(columns=["country","dd"])
        pred = pd.DataFrame(model.predict(test_x_d))
        pred["country"]=test_y_s.country.values
        pred["dd"]=test_y_s.dd.values
        base=df[["country","dd","sb_fatalities"]].loc[(df["dd"]>="2021-01")&(df["dd"]<="2022-12")]
        val=pd.merge(base,pred,on=["country","dd"],how="left")
        val.columns=["country","dd","sb_fatalities","preds"]
        val=val.fillna(0)
        val=val.sort_values(by=["country","dd"])
        
        onset, de = evals(val.sb_fatalities, val.preds, val.country)
        print(onset,de)
        score = w1 * onset + w2 * de
        print(score)
        
        if score>test_score:
            test_score=score
            print(f"Best: ar={ar}, cut={cut}")
            catcher.to_csv("out/catcher_2023.csv") 
                
    


