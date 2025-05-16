import pandas as pd
import wbgapi as wb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def lag_groupped(df, group_var, var, lag):
    return df.groupby(group_var)[var].shift(lag).fillna(0)

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

def exponential_decay(time):
    return 2**(time/12)

def apply_decay(group,var):
    out = exponential_decay(group[var])
    return out

def preprocess_min_max_group(df, x, group):
    out = pd.DataFrame()
    for i in df[group].unique():
        df_s = df.loc[df[group] == i]
        scaler = MinMaxScaler().fit(
            df_s[x].values.reshape(-1, 1))
        norm = scaler.transform(df_s[x].values.reshape(-1, 1))
        out = pd.concat([out, pd.DataFrame(norm)], ignore_index=True)
    df[f"{x}_norm"] = out

def get_wb(years: list,
           countries: list,
           var: list):

    # Get data for each year and merge
    wdi = pd.DataFrame()
    for i in years:
        print(i)
        wdi_s = wb.data.DataFrame(var, countries, [i])
        wdi_s.reset_index(inplace=True)
        wdi_s["year"] = i
        wdi = pd.concat([wdi, wdi_s], ignore_index=True)  

    # Get country codes
    df_ccodes = pd.read_csv("data/df_ccodes.csv")
    df_ccodes_s = df_ccodes[["country",'gw_codes', "acled_codes", "iso_alpha3"]]
    wdi_final = pd.merge(wdi, df_ccodes_s, how='left', left_on=['economy'], right_on=['iso_alpha3'])
    wdi_final = wdi_final.drop(columns=['economy'])
    wdi_final = wdi_final[['gw_codes'] + [col for col in wdi_final.columns if col != 'gw_codes']]
    wdi_final = wdi_final[['iso_alpha3'] + [col for col in wdi_final.columns if col != 'iso_alpha3']]
    wdi_final = wdi_final[['acled_codes'] + [col for col in wdi_final.columns if col != 'acled_codes']]
    wdi_final = wdi_final[['year'] + [col for col in wdi_final.columns if col != 'year']]
    wdi_final = wdi_final[['country'] +[col for col in wdi_final.columns if col != 'country']]
    wdi_final = wdi_final.sort_values(by=["iso_alpha3", 'year'])
    wdi_final = wdi_final.reset_index(drop=True)
    
    print("Obtained data")
    print(wdi_final.head())

    return wdi_final 

def simple_imp_grouped(df, group, vars_input,time="year",last_train=2021):
    
    # Split data 
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]
        train_s = df_s.loc[df_s[time]<=last_train]
        test_s = df_s.loc[df_s[time]>last_train]
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])
        
    # Fill
    df_filled = pd.DataFrame()
    for c in df[group].unique():
        
        # Training
        df_s = train.loc[train[group] == c]
        imp = df_s[vars_input]
        df_imp = imp.copy(deep=True)
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(df_imp)
        df_imp_train = imputer.transform(df_imp)
        df_imp_train_df = pd.DataFrame(df_imp_train)
        
        # Test
        df_s = test.loc[test[group] == c]
        imp = df_s[vars_input]
        df_imp = imp.copy(deep=True)        
        df_imp_test = imputer.transform(df_imp)
        df_imp_test_df = pd.DataFrame(df_imp_test)        

        # Merge
        df_imp_final = pd.concat([df_imp_train_df, df_imp_test_df])
        df_filled = pd.concat([df_filled, df_imp_final])

    # Merge
    df_filled.columns = vars_input
    df_filled = df_filled.reset_index(drop=True)
    feat_complete = df.drop(columns=vars_input)
    out = pd.concat([feat_complete, df_filled], axis=1)
    out=out.reset_index(drop=True)
    
    return out


def linear_imp_grouped(df, group, vars_input,time="year",last_train=2021):
    
    # Split
    train = pd.DataFrame()
    test = pd.DataFrame()
    for c in df.country.unique():
        df_s = df.loc[df["country"] == c]
        train_s = df_s.loc[df_s[time]<=last_train]
        test_s = df_s.loc[df_s[time]>last_train]
        train = pd.concat([train, train_s])
        test = pd.concat([test, test_s])
    
    # Fill
    df_filled = pd.DataFrame()
    for c in df[group].unique():
        
        # Training
        df_s = train.loc[train[group] == c]
        imp = df_s[vars_input]
        df_imp = imp.copy(deep=True)
        df_imp_train_df = df_imp.interpolate(limit_direction="both")
        
        # Test
        df_s = test.loc[test[group] == c]
        imp = df_s[vars_input]
        df_imp = imp.copy(deep=True)        
        df_imp_test_df = df_imp.interpolate(limit_direction="forward")
        
        # Merge
        df_imp_final = pd.concat([df_imp_train_df, df_imp_test_df])
        df_filled = pd.concat([df_filled, df_imp_final])
        
    # Merge
    df_filled.columns = vars_input
    df_filled = df_filled.reset_index(drop=True)
    feat_complete = df.drop(columns=vars_input)
    out = pd.concat([feat_complete, df_filled], axis=1)
    out=out.reset_index(drop=True)
    
    return out


