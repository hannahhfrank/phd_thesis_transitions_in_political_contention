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
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from dtaidistance import dtw,ed
import bisect
import pickle
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt 
from matplotlib import gridspec
from functions import lag_groupped
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import os
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'


plt.rcParams['xtick.labelsize'] = 20  # X-axis tick label size
plt.rcParams['ytick.labelsize'] = 20  # Y-axis tick label size


# Country definitions: http://ksgleditsch.com/statelist.html

# List of microstates: 
micro_states={"Dominica":54,
              "Grenada":55,
              "Saint Lucia":56,
              "Saint Vincent and the Grenadines":57,
              "Antigua & Barbuda":58,
              "Saint Kitts and Nevis":60,
              "Monaco":221,
              "Liechtenstein":223,
              "San Marino":331,
              "Andorra":232,
              "Abkhazia":396,
              "South Ossetia":397,
              "São Tomé and Principe":403,
              "Seychelles":591,
              "Vanuatu":935,
              "Kiribati":970,
              "Nauru":971,
              "Tonga":972,
              "Tuvalu":973,
              "Marshall Islands":983,
              "Palau":986,
              "Micronesia":987,
              "Samoa":990}

# Additional countries not included in ACLED: 
# 265 German Democratic Republic	
# 315 Czechoslovakia
# 345 Yugoslavia
# 396 Abkhazia
# 397 South Ossetia
# 680 Yemen, People's Republic of ---> EXCLUDE

# Temporal coverage: 1989--2022

exclude={"German Democratic Republic":265,
         "Czechoslovakia":315,
         "Yugoslavia":345,
         "Abkhazia":396,
         "South Ossetia":397,
         "Yemen, People's Republic of":680}

# Countries not included in World Bank: 
# Or have mostly missing values    
exclude2 ={"Taiwan":713, # Not included in WDI
           "Bahamas":31, # Not included in vdem
           "Belize":80, # Not included in vdem
           "Brunei Darussalam":835, # Not included in vdem
           "Kosovo":347, # Mostly missing in WDI
           "Democratic Peoples Republic of Korea":731} # Mostly missing in WDI

####################
### Shape finder ###
####################

df=pd.read_csv("data/data_out/ucdp_cm_sb.csv",index_col=0)
df = df[~df['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
df=df.sort_values(by=["country","year","dd"])

### Shape finder simplified ###

def shape_finder_simple(df,min_d=0.1,dtw_sel=2,win=10,horizon=12,h_train=10,outcome="best",input_year=2021):
    dict_m={i :[] for i in df.country.unique()} 
    pred_tot_pr=[]
        
    for c in df.country.unique():
        # Get input shape
        df_s=df[["year","dd","best"]].loc[df["country"]==c]
        input_shape = df_s.loc[df_s["year"]==input_year][2:]
        input_shape = input_shape.set_index('dd').drop('year', axis=1)
        input_shape = input_shape["best"]
        input_shape.name = c
        input_shape_copy=input_shape
        
        # If input shape is not flat, otherwise predict 0
        if not (input_shape==0).all():
        
            # Get subsequences and normalize
            df_input_sub=df.loc[df["year"]<=input_year]
            df_input_sub = df_input_sub.pivot(index='dd', columns='country', values='best')
            df_input_sub = df_input_sub.fillna(0)
            seq = []
            for i in range(len(df_input_sub.columns)): 
                seq.append(df_input_sub.iloc[:, i]) 
            seq_n = []
            for i in seq:
                seq_n.append((i - i.mean()) / i.std())
                    
            # Min-max normalize input shape
            if input_shape.var() != 0.0:
                input_shape = (input_shape - input_shape.min()) / (input_shape.max() - input_shape.min())
            else : # if input is plat, set to 0.5
                input_shape= [0.5]*len(input_shape)
                input_shape = np.array(input_shape)
                
            # Get a df with all distances between input and all references
            tot = []
            for lop in range(int(-dtw_sel), int(dtw_sel) + 1): 
                n_test = [] 
                to = 0  
                exclude = []  
                interv = [0]  
                for i in seq_n:
                    n_test = np.concatenate([n_test, i])  
                    to = to + len(i) 
                    exclude = exclude + [*range(to - (win+lop), to)]  
                    interv.append(to)  
                for i in range(len(n_test)):
                    if i not in exclude:
                        seq2 = n_test[i:i + int(10 + lop)]
                        if seq2.var() != 0.0:
                            seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                        else:
                            seq2 = np.array([0.5]*len(seq2))
                        try:
                            dist = dtw.distance(input_shape.values, seq2, use_c=True)
                            tot.append([i, dist, 10 + lop])
                        except:
                            pass
            # Stores indes, distance, and window length
            tot = pd.DataFrame(tot)
                
            # Get filtered repository         
            min_d_d=min_d
            sequences=[]
            while len(sequences)<5:
                matches=[]    
                tot_more = tot.sort_values([1])
                tot_more = tot_more[tot_more[1] < min_d_d]
                toti = tot_more[0]
                n = len(toti)
                diff_data = {f'diff{i}': toti.diff(i) for i in range(1, n + 1)}
                diff_df = pd.DataFrame(diff_data).fillna(10)
                diff_df = abs(diff_df)
                tot_more = tot_more[diff_df.min(axis=1) >= (10/ 2)]
                
                if len(tot_more) > 0:
                    for c_lo in range(len(tot_more)):
                        i = tot_more.iloc[c_lo, 0]
                        win_l = int(tot_more.iloc[c_lo, 2])
                        
                        n_test = [] 
                        to = 0  
                        exclude = []  
                        interv = [0]  
                        for x in seq_n:
                            n_test = np.concatenate([n_test, x])  
                            to = to + len(x)  
                            exclude = exclude + [*range(to - (win+lop), to)]  
                            interv.append(to)  
                            
                        col = seq[bisect.bisect_right(interv, i) - 1].name
                        index_obs = seq[bisect.bisect_right(interv, i) - 1].index[i - interv[bisect.bisect_right(interv, i) - 1]]
                        obs = df_input_sub.loc[index_obs:, col].iloc[:win_l]
                        matches.append([obs, tot_more.iloc[c_lo, 1]])
                        sequences=matches
                min_d_d += 0.05
    
            dict_m[c]=sequences
                           
            ########################
            ### Make predictions ###
            ########################
            
            df_input = df.pivot(index='dd', columns='country', values=outcome)
            df_input = df_input.fillna(0)
            
            # Extract references
            l_find=dict_m[c]
            tot_seq = [[series.name, series.index[-1], series.min(),series.max()] for series, weight in l_find]
            
            pred_seq=[]
            co=[]
            deca=[]
            scale=[]
            # For each reference, get index of last observed value
            for col,last_date,mi,ma in tot_seq:
                date=df_input.loc[:f"{input_year}-12"].index.get_loc(last_date)
                # If future of reference does not lie in testing window, add min-max normalized future
                if date+horizon<len(df_input.loc[:f"{input_year}-12"]):
                    seq=df_input.loc[:f"{input_year}-12"].iloc[date+1:date+1+horizon,df_input.loc[:f"{input_year}-12"].columns.get_loc(col)].reset_index(drop=True)
                    seq = (seq - mi) / (ma - mi)
                    pred_seq.append(seq.tolist())
                
            # Apply clustering allgorithm to futures
            tot_seq=pd.DataFrame(pred_seq)
            linkage_matrix = linkage(tot_seq, method='ward')
            clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
            tot_seq['Cluster'] = clusters
            
            # Get centroid for each cluster
            val_sce = tot_seq.groupby('Cluster').mean()
            # Calculate how many observations are in cluster and select majority cluster
            pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
            pred_ori=val_sce.loc[pr==pr.max(),:]
            pred_ori=pred_ori.mean(axis=0)
            # Reverse min-max normalization and append to predictions
            preds=pred_ori*(input_shape_copy.max()-input_shape_copy.min())+input_shape_copy.min()
            pred_tot_pr.append(preds)
              
        else:
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
                        
    # Save        
    preds = pd.concat(pred_tot_pr,axis=1)
    preds.columns=df.country.unique()
    preds.index=[f"{input_year+1}-01",f"{input_year+1}-02",f"{input_year+1}-03",f"{input_year+1}-04",f"{input_year+1}-05",f"{input_year+1}-06",f"{input_year+1}-07",f"{input_year+1}-08",f"{input_year+1}-09",f"{input_year+1}-10",f"{input_year+1}-11",f"{input_year+1}-12"]
    preds.index.name = 'dd'
    preds_out = preds.reset_index().melt(id_vars='dd', var_name='country', value_name='best')
    preds_out=preds_out.rename(columns={'best': 'preds'})
    #preds_out.to_csv(f'out/preds{input_year+1}.csv')  

    return preds_out     

preds_2022=shape_finder_simple(df)
preds_2023=shape_finder_simple(df,input_year=2022)

shape_finder = pd.concat([preds_2022, preds_2023], axis=0, ignore_index=True)
shape_finder=shape_finder.sort_values(by=["country","dd"])
shape_finder=shape_finder.reset_index(drop=True)
shape_finder["year"] = shape_finder["dd"].str[:4] 
shape_finder["year"]=shape_finder["year"].astype(int)

# Merge
shape_finder=pd.merge(shape_finder,df[["country","dd","gw_codes","best"]],on=["dd","country"],how="left")
shape_finder=shape_finder[["year","dd","country","gw_codes","preds","best"]]
shape_finder=shape_finder.rename(columns={'best': 'sb_fatalities'})
shape_finder.to_csv('out/shape_finder.csv') 

#############
### ViEWS ###
#############

ucdp_sb=pd.read_csv("data/data_out/ucdp_cm_sb.csv",index_col=0)
ucdp_sb = ucdp_sb[["year","dd","gw_codes","country","best","onset"]][~ucdp_sb['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
ucdp_sb.columns=["year","dd","gw_codes","country","sb_fatalities","onset"]
ucdp_sb_s=ucdp_sb.loc[(ucdp_sb["year"]==2022)|(ucdp_sb["year"]==2023)|(ucdp_sb["year"]==2024)]
ucdp_sb_s=ucdp_sb_s.sort_values(by=["country","dd"])

# Countries in Views, but not in my data: Antigua & Barbuda, Bahamas, Belize
# Brunei, Dominica, Grendada, Kiribati, Kosovo, Marshall Is., Micronesia
# Nauru, North Korea, Palau, Samoa, Sao Tome and Principe, Seychelles
# St. Kitts and Nevis, St. Lucia, St. Vincent and the Grenadines, Taiwan, 
# Tonga, Tuvalu, Vanuatu 

# 2022
views2022=pd.DataFrame()
for y in [1,2,3,4,5,6,7]:
    response = requests.get(f'https://api.viewsforecasting.org/fatalities001_2021_12_t01/cm/sb?page={y}')
    json_data = response.json()
    df=pd.DataFrame(json_data["data"])
    views2022 = pd.concat([views2022, df], axis=0, ignore_index=True)
views2022=views2022[["name","isoab","gwcode","year","month","sc_cm_sb_main"]].loc[views2022["year"]==2022]
views2022['dd'] = views2022['month'].astype(str) + '-' + views2022['year'].astype(str).str[-2:]
views2022=views2022[["name","isoab","gwcode","year","dd","sc_cm_sb_main"]]
views2022.columns=["country","isoab","gw_codes","year","dd","preds_log"]
views2022["preds"] = np.exp(views2022["preds_log"]) - 1
views2022['dd'] = views2022['dd'].apply(lambda x: x.zfill(5) if len(x) == 4 else x)

# 2023
views2023=pd.DataFrame()
for y in [1,2,3,4,5,6,7]:
    response = requests.get(f'https://api.viewsforecasting.org/fatalities001_2022_12_t01/cm/sb?page={y}')
    json_data = response.json()
    df=pd.DataFrame(json_data["data"])
    views2023 = pd.concat([views2023, df], axis=0, ignore_index=True)
views2023=views2023[["name","isoab","gwcode","year","month","sc_cm_sb_main"]].loc[views2023["year"]==2023]
views2023['dd'] = views2023['month'].astype(str) + '-' + views2023['year'].astype(str).str[-2:]
views2023=views2023[["name","isoab","gwcode","year","dd","sc_cm_sb_main"]]
views2023.columns=["country","isoab","gw_codes","year","dd","preds_log"]
views2023["preds"] = np.exp(views2023["preds_log"]) - 1
views2023['dd'] = views2023['dd'].apply(lambda x: x.zfill(5) if len(x) == 4 else x)

# 2024
views2024=pd.DataFrame()
for y in [1,2,3,4,5,6,7]:
    response = requests.get(f'https://api.viewsforecasting.org/fatalities002_2023_12_t01/cm/sb?page={y}')
    json_data = response.json()
    df=pd.DataFrame(json_data["data"])
    views2024 = pd.concat([views2024, df], axis=0, ignore_index=True)
views2024=views2024[["name","isoab","gwcode","year","month","main_mean_ln"]].loc[views2024["year"]==2024]
views2024['dd'] = views2024['month'].astype(str) + '-' + views2024['year'].astype(str).str[-2:]
views2024=views2024[["name","isoab","gwcode","year","dd","main_mean_ln"]]
views2024.columns=["country","isoab","gw_codes","year","dd","preds_log"]
views2024["preds"] = np.exp(views2024["preds_log"]) - 1
views2024['dd'] = views2024['dd'].apply(lambda x: x.zfill(5) if len(x) == 4 else x)

views = pd.concat([views2022, views2023, views2024], axis=0, ignore_index=True)
views=views.sort_values(by=["country","year","dd"])
views=views.reset_index(drop=True)
views["dd"]=views["dd"].astype(str)
views["dd"] = "20" + views["dd"].str[-2:] + "-" + views["dd"].str[:2]
views["gw_codes"]=views["gw_codes"].astype(int)

# Merge
views=pd.merge(ucdp_sb_s[["dd","gw_codes","country","sb_fatalities"]],views[['dd','gw_codes',"year",'preds']],on=["dd","gw_codes"],how="left")
views_final=views[["year","dd","country","gw_codes","sb_fatalities","preds"]]
views_final=views_final.sort_values(by=["country","dd"])
views_final.to_csv("out/views.csv") 

views_final=pd.read_csv("out/views.csv",index_col=0) 
shape_finder=pd.read_csv('out/shape_finder.csv',index_col=0) 

# Plots 
for c in views_final.gw_codes.unique():
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    df_s=views_final.loc[views_final["gw_codes"]==c]
    df_ss=shape_finder.loc[shape_finder["gw_codes"]==c]
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df_s["dd"].loc[df_s["year"]==2022],df_s["sb_fatalities"].loc[df_s["year"]==2022],color="black")
    ax1.plot(df_s["dd"].loc[df_s["year"]==2022],df_s["preds"].loc[df_s["year"]==2022],color="black",linestyle="dotted")
    ax1.plot(df_ss["dd"].loc[df_ss["year"]==2022],df_ss["preds"].loc[df_ss["year"]==2022],color="black",linestyle="dashed")   
    ax1.set_xticks(["2022-01","2022-03","2022-05","2022-07","2022-09","2022-11"],["01-22","03-22","05-22","07-22","09-22","11-22"])
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = fig.add_subplot(gs[1])    
    ax2.plot(df_s["dd"].loc[df_s["year"]==2023],df_s["sb_fatalities"].loc[df_s["year"]==2023],color="black")
    ax2.plot(df_s["dd"].loc[df_s["year"]==2023],df_s["preds"].loc[df_s["year"]==2023],color="black",linestyle="dotted")
    ax2.plot(df_ss["dd"].loc[df_ss["year"]==2023],df_ss["preds"].loc[df_ss["year"]==2023],color="black",linestyle="dashed")
    ax2.set_xticks(["2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],["01-23","03-23","05-23","07-23","09-23","11-23"])
    ax2.yaxis.set_ticks_position('right')
    ax2.yaxis.set_label_position('right')
    gs.update(wspace=0)    
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
   
    #fig.suptitle(views_final["country"].loc[views_final["gw_codes"]==c].iloc[0],size=30)
    if df_ss.preds.max()!=0:
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/procedural/out/proc_examples_{views_final['country'].loc[views_final['gw_codes']==c].iloc[0]}.eps",dpi=300,bbox_inches='tight')        
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/proc_examples_{views_final['country'].loc[views_final['gw_codes']==c].iloc[0]}.eps",dpi=300,bbox_inches='tight')
        plt.savefig(f"out/proc_examples_{views_final['country'].loc[views_final['gw_codes']==c].iloc[0]}.eps",dpi=300,bbox_inches='tight')    
    plt.show()   

#########################
### Conflict Forecast ###
#########################

# Countries in my data, but no in cf: Cape Verde, Lesotho, Solomon Is. (only 2022)
# Suriname, Trinidad and Tobago (only 2022)

# Countries in cf, but not in my data: Antigua and Barbuda, Bahamas, Belize, 
# Brunei, Dominica, Grendada, North Korea, Seychelles, Taiwan

#cf2022=pd.read_csv("https://backendlessappcontent.com/C177D0DC-B3D5-818C-FF1E-1CC11BC69600/E39F9861-2A3B-449A-BFC9-776835054E4D/files/conflictForecast/01-2022/conflictforecast_armedconf_12.csv")
#cf2022=cf2022[["isocode","year","best_model"]].loc[(cf2022["year"]==2022)&(cf2022["month"]==1)]

#cf2023=pd.read_csv("https://backendlessappcontent.com/C177D0DC-B3D5-818C-FF1E-1CC11BC69600/E39F9861-2A3B-449A-BFC9-776835054E4D/files/conflictForecast/01-2023/conflictforecast_armedconf_12.csv")
#cf2023=cf2023[["isocode","year","best_model"]].loc[(cf2023["year"]==2023)&(cf2023["month"]==1)]

#cf2024=pd.read_csv("https://backendlessappcontent.com/C177D0DC-B3D5-818C-FF1E-1CC11BC69600/E39F9861-2A3B-449A-BFC9-776835054E4D/files/conflictForecast/01-2024/conflictforecast_armedconf_12.csv")
#cf2024=cf2024[["isocode","year","best_model"]].loc[(cf2024["year"]==2024)&(cf2024["month"]==1)]

#cf = pd.concat([cf2022, cf2023, cf2024], axis=0, ignore_index=True)
#codes=views[["country","isoab","gw_codes"]].drop_duplicates()
#cf=pd.merge(left=cf,right=codes,left_on=["isocode"], right_on=["isoab"],how="left")
#cf=cf.dropna()
#cf=cf.sort_values(by=["country","year"])
#cf=cf[["country","gw_codes","year","best_model"]]
#cf.columns=["country","gw_codes","year","preds"]
#cf=cf.reset_index(drop=True)
#cf["gw_codes"]=cf["gw_codes"].astype(int)
#cf.to_csv("out/cf.csv") 

#########################
### Negative-binomial ###
#########################

### 2022 ###
df=pd.read_csv("out/df_complete_cm.csv",index_col=0)
df['sb_fatalities_lag1']=np.log(df['sb_fatalities_lag1']+1)
df["sb_fatalities_lag2"]=lag_groupped(df,"country","sb_fatalities",2)
df['sb_fatalities_lag2']=np.log(df['sb_fatalities_lag2']+1)
df["sb_fatalities_lag3"]=lag_groupped(df,"country","sb_fatalities",3)
df['sb_fatalities_lag3']=np.log(df['sb_fatalities_lag2']+1)
df['pop']=np.log(df['pop'])
df['gdp']=np.log(df['gdp'])

#df['d_neighbors_sb_fatalities_lag1']=np.log(df['d_neighbors_sb_fatalities_lag1']+1)
#df['pop']=np.log(df['pop'])
#df['gdp']=np.log(df['gdp'])

target='sb_fatalities'
inputs=["sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]
y=df[["dd",'country','sb_fatalities']]
x=df[["dd",'country',"sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]]


#df=pd.read_csv("out/df_complete_cm.csv",index_col=0)
#df = df[["country","dd",'sb_fatalities',"sb_fatalities_lag1","d_civil_conflict_zeros_decay","d_neighbors_sb_fatalities_lag1","pop","ethnic_frac","gdp","mys_male","libdem","civlib","rugged"]]
#df['sb_fatalities']=df['sb_fatalities']+1
#df['sb_fatalities_lag1']=np.log(df['sb_fatalities_lag1']+1)
#df['d_neighbors_sb_fatalities_lag1']=np.log(df['d_neighbors_sb_fatalities_lag1']+1)
#df['pop']=np.log(df['pop'])
#df['gdp']=np.log(df['gdp'])

#target='sb_fatalities'
#inputs=["sb_fatalities_lag1","d_civil_conflict_zeros_decay","d_neighbors_sb_fatalities_lag1","pop","ethnic_frac","gdp","mys_male","libdem","civlib","rugged"]
#y=df[["dd",'country','sb_fatalities']]
#x=df[["dd",'country',"sb_fatalities_lag1","d_civil_conflict_zeros_decay","d_neighbors_sb_fatalities_lag1","pop","ethnic_frac","gdp","mys_male","libdem","civlib","rugged"]]

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
    y_train = y_s[["country","dd"]+[target]].loc[y_s["dd"]<="2021-12"]
    x_train = x_s[["country","dd"]+inputs].loc[y_s["dd"]<="2021-12"]
    y_test = y_s[["country","dd"]+[target]].loc[(y_s["dd"]>="2022-01")&(y_s["dd"]<="2022-12")]
    x_test = x_s[["country","dd"]+inputs].loc[(y_s["dd"]>="2022-01")&(y_s["dd"]<="2022-12")]
    # Merge
    train_y = pd.concat([train_y, y_train])
    test_y = pd.concat([test_y, y_test])
    train_x = pd.concat([train_x, x_train])
    test_x = pd.concat([test_x, x_test])
    
    # Validation
    val_train_index += list(y_train[:int(0.8*len(y_train))].index)
    val_test_index += list(y_train[int(0.8*len(y_train)):].index)
    
splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))

# Train and test model 
train_y_d=train_y.drop(columns=["country","dd"])
train_x_d=train_x.drop(columns=["country","dd"])
test_y_d=test_y.drop(columns=["country","dd"])
test_x_d=test_x.drop(columns=["country","dd"])
train_x_d = sm.add_constant(train_x_d)  
zinb_model = sm.ZeroInflatedNegativeBinomialP(train_y_d, train_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]], exog_infl=train_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","gdp"]])
zinb_model=zinb_model.fit(maxiter=2000)
zinb_model.summary()
test_x_d = sm.add_constant(test_x_d)  
pred=zinb_model.predict(test_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]],exog_infl=test_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","gdp"]])

# Get test df
test_y_d_d=test_y.reset_index(drop=True)
test_y_d_d['sb_fatalities']=test_y_d_d['sb_fatalities']
pred_d=pred.reset_index(drop=True)
zinb2022=pd.concat([test_y_d_d,pred_d],axis=1)
zinb2022.columns=["country","dd","sb_fatalities","preds"]

### 2023 ###
df=pd.read_csv("out/df_complete_cm.csv",index_col=0)
df['sb_fatalities_lag1']=np.log(df['sb_fatalities_lag1']+1)
df["sb_fatalities_lag2"]=lag_groupped(df,"country","sb_fatalities",2)
df['sb_fatalities_lag2']=np.log(df['sb_fatalities_lag2']+1)
df["sb_fatalities_lag3"]=lag_groupped(df,"country","sb_fatalities",3)
df['sb_fatalities_lag3']=np.log(df['sb_fatalities_lag2']+1)
df['pop']=np.log(df['pop'])
df['gdp']=np.log(df['gdp'])

target='sb_fatalities'
inputs=["sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]
y=df[["dd",'country','sb_fatalities']]
x=df[["dd",'country',"sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]]



#df=pd.read_csv("out/df_complete_cm.csv",index_col=0)
#df = df[["country","dd",'sb_fatalities',"sb_fatalities_lag1","d_civil_conflict_zeros_decay","d_neighbors_sb_fatalities_lag1","pop","ethnic_frac","gdp","mys_male","libdem","civlib","rugged"]]
#df['sb_fatalities']=df['sb_fatalities']+1
#df['sb_fatalities_lag1']=np.log(df['sb_fatalities_lag1']+1)
#df['d_neighbors_sb_fatalities_lag1']=np.log(df['d_neighbors_sb_fatalities_lag1']+1)
#df['pop']=np.log(df['pop'])
#df['gdp']=np.log(df['gdp'])

#target='sb_fatalities'
#inputs=["sb_fatalities_lag1","d_civil_conflict_zeros_decay","d_neighbors_sb_fatalities_lag1","pop","ethnic_frac","gdp","mys_male","libdem","civlib","rugged"]
#y=df[["dd",'country','sb_fatalities']]
#x=df[["dd",'country',"sb_fatalities_lag1","d_civil_conflict_zeros_decay","d_neighbors_sb_fatalities_lag1","pop","ethnic_frac","gdp","mys_male","libdem","civlib","rugged"]]

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
    y_train = y_s[["country","dd"]+[target]].loc[y_s["dd"]<="2022-12"]
    x_train = x_s[["country","dd"]+inputs].loc[y_s["dd"]<="2022-12"]
    y_test = y_s[["country","dd"]+[target]].loc[(y_s["dd"]>="2023-01")]
    x_test = x_s[["country","dd"]+inputs].loc[(y_s["dd"]>="2023-01")]
    # Merge
    train_y = pd.concat([train_y, y_train])
    test_y = pd.concat([test_y, y_test])
    train_x = pd.concat([train_x, x_train])
    test_x = pd.concat([test_x, x_test])
    
    # Validation
    val_train_index += list(y_train[:int(0.8*len(y_train))].index)
    val_test_index += list(y_train[int(0.8*len(y_train)):].index)
    
splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))

# Train and test model 
train_y_d=train_y.drop(columns=["country","dd"])
train_x_d=train_x.drop(columns=["country","dd"])
test_y_d=test_y.drop(columns=["country","dd"])
test_x_d=test_x.drop(columns=["country","dd"])
train_x_d = sm.add_constant(train_x_d)  
zinb_model = sm.ZeroInflatedNegativeBinomialP(train_y_d, train_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]], exog_infl=train_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","gdp"]])
zinb_model=zinb_model.fit(maxiter=2000)
zinb_model.summary()
test_x_d = sm.add_constant(test_x_d)  
pred=zinb_model.predict(test_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","pop","gdp_growth","gdp"]],exog_infl=test_x_d[["const","sb_fatalities_lag1","sb_fatalities_lag2","sb_fatalities_lag3","gdp"]])

# Get test df
test_y_d_d=test_y.reset_index(drop=True)
test_y_d_d['sb_fatalities']=test_y_d_d['sb_fatalities']-1
pred_d=pred.reset_index(drop=True)
zinb2023=pd.concat([test_y_d_d,pred_d],axis=1)
zinb2023.columns=["country","dd","sb_fatalities","preds"]

# Merge
zinb = pd.concat([zinb2022, zinb2023], axis=0, ignore_index=True)
zinb=zinb.sort_values(by=["country","dd"])
zinb=zinb.reset_index(drop=True)
zinb['year'] = zinb['dd'].str[:4]
df=pd.read_csv("out/df_complete_cm.csv",index_col=0)
codes=df[["country","gw_codes"]].drop_duplicates()
zinb=pd.merge(zinb, codes,on="country",how="left")
zinb=zinb[["country","gw_codes","year","sb_fatalities","preds"]]
zinb.to_csv("out/zinb.csv") 

