import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import preprocess_min_max_group
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import partial_dependence
import shap
from PyALE import ale
from sklearn.dummy import DummyRegressor
import matplotlib as mpl
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import random
random.seed(42)
import seaborn as sns
import os 
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'

random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)],
               'max_features': ["sqrt", "log2", None],
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],
               'min_samples_split': [2,5,10],
               'min_samples_leaf': [1,2,4],
               }

# Load data
df=pd.read_csv("out/data_examples.csv",index_col=0)
df["sb_fatalities_log"]=np.log(df["sb_fatalities"]+1)
df["sb_fatalities_lag1"]=np.log(df["sb_fatalities_lag1"]+1)
df["gdp"]=np.log(df["gdp"]+1)
preprocess_min_max_group(df,"growth","country")
df["oil_share"]=np.log(df["oil_share"]+1)
df["pop"]=np.log(df["pop"]+1)
preprocess_min_max_group(df,"temp","country")
preprocess_min_max_group(df,"withdrawl","country")

for var in ['sb_fatalities_lag1','d_civil_conflict_zeros_decay','d_civil_war_zeros_decay','d_neighbors_sb_fatalities_lag1','gdp','growth','oil_share','pop','inf_mort','male_youth_share','temp','withdrawl','ethnic_frac','eys_male','polyarchy','libdem','egaldem','civlib','exlsocgr']:
    fig,ax = plt.subplots()
    df[var].hist()
    
# Train model    
target='sb_fatalities_log'
inputs=['sb_fatalities_lag1','d_civil_conflict_zeros_decay','d_civil_war_zeros_decay','d_neighbors_sb_fatalities_lag1','gdp','growth_norm','oil_share','pop','inf_mort','male_youth_share','temp_norm','withdrawl_norm','ethnic_frac','eys_male','polyarchy','libdem','egaldem','civlib','exlsocgr']
y=df[["year",'country','sb_fatalities_log']]
x=df[["year",'country','sb_fatalities_lag1','d_civil_conflict_zeros_decay','d_civil_war_zeros_decay','d_neighbors_sb_fatalities_lag1','gdp','growth_norm','oil_share','pop','inf_mort','male_youth_share','temp_norm','withdrawl_norm','ethnic_frac','eys_male','polyarchy','libdem','egaldem','civlib','exlsocgr']]

train_y = pd.DataFrame()
test_y = pd.DataFrame()
train_x = pd.DataFrame()
test_x = pd.DataFrame()
    
val_train_index = []
val_test_index = []
    
for c in y.country.unique():
    y_s = y.loc[y["country"] == c]
    x_s = x.loc[x["country"] == c]
    y_train = y_s[["country","year"]+[target]][:int(0.8*len(y_s))]
    x_train = x_s[["country","year"]+inputs][:int(0.8*len(x_s))]
    y_test = y_s[["country","year"]+[target]][int(0.8*len(y_s)):]
    x_test = x_s[["country","year"]+inputs][int(0.8*len(x_s)):]
    train_y = pd.concat([train_y, y_train])
    test_y = pd.concat([test_y, y_test])
    train_x = pd.concat([train_x, x_train])
    test_x = pd.concat([test_x, x_test])
    
    val_train_index += list(y_train[:int(0.8*len(y_train))].index)
    val_test_index += list(y_train[int(0.8*len(y_train)):].index)
    
splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
    
train_y_d=train_y.drop(columns=["country","year"])
train_x_d=train_x.drop(columns=["country","year"])
test_y_d=test_y.drop(columns=["country","year"])
test_x_d=test_x.drop(columns=["country","year"])

ps = PredefinedSplit(test_fold=splits)
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=1), param_grid=random_grid, cv=ps, verbose=0, n_jobs=-1)
grid_search.fit(train_x_d, train_y_d.values.ravel())
best_params = grid_search.best_params_
model=RandomForestRegressor(random_state=1,**best_params)
model.fit(train_x_d, train_y_d.values.ravel())
pred = pd.DataFrame(model.predict(test_x_d))
mse = mean_squared_error(test_y_d.values, pred)
wmse = mean_squared_error(test_y_d.values, pred, sample_weight=test_y_d.values)
evals = {"mse": mse, "wmse": wmse}
print(round(evals["mse"],6))
print(round(evals["wmse"],6))

# Baseline
base=DummyRegressor(strategy="mean")
base.fit(train_x_d, train_y_d.values.ravel())
pred_base = pd.DataFrame(base.predict(test_x_d))
mse = mean_squared_error(test_y_d.values, pred_base)
wmse = mean_squared_error(test_y_d.values, pred_base, sample_weight=test_y_d.values)
evals_base = {"mse": mse, "wmse": wmse}
print(round(evals_base["mse"],6))
print(round(evals_base["wmse"],6))

################################
### Interpretability methods ###
################################

names={"sb_fatalities_lag1":"t-1 lag of the number of fatalities (log + 1)",
       "d_civil_conflict_zeros_decay":"Time since last civil conflict",
       "d_civil_war_zeros_decay":"Time since last civil war",
       "d_neighbors_sb_fatalities_lag1":"At least one neighbor had at least one fatality",
       "inf_mort":"Mortality rate, infant",       
       "gdp":"GDP per capita (log + 1)",
       "growth_norm":"GDP growth (norm)",
       "oil_share":"Oil rents (log + 1)",
       "pop":"Population size (log + 1)",
       "male_youth_share":"Male total population 15-19",
       "ethnic_frac":"Ethnolinguistic fractionalization (ELF) index",
       "eys_male":"Expected years of schooling, male",
       "temp_norm":"Average mean surface air temperature (norm)",
       "withdrawl_norm":"Annual freshwater withdrawals, total (norm)",
       "polyarchy":"Electoral democracy index",
       "libdem":"Liberal democracy index",
       "egaldem":"Egalitarian democracy index",
       "civlib":"Civil liberties index",
       "exlsocgr":"Exclusion by Social Group index"}

x_d=x.drop(columns=["country","year"])
y_d=y.drop(columns=["country","year"])

###################
### SHAP values ###
###################

X_train_summary = shap.kmeans(x_d, 10)
explainer = shap.KernelExplainer(model.predict, X_train_summary)
shap_values = explainer.shap_values(x_d)

# Shap importance

# Table
vals=np.abs(shap_values).mean(0)
shap_importance = pd.DataFrame(list(zip(inputs, vals)),columns=['Feature','Feature Shap Value'])
shap_importance.sort_values(by=['Feature Shap Value'],ascending=False, inplace=True) 
for c in inputs:
    shap_importance.loc[shap_importance["Feature"]==c,"Feature"]=names[c]
latex_table = shap_importance.to_latex(index=False)
with open('out/intro_feat_shap.tex', 'w') as f:
    f.write(latex_table)
with open('/Users/hannahfrank/Dropbox/Apps/Overleaf/introduction/out/intro_feat_shap.tex', 'w') as f:
    f.write(latex_table)
with open('/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/intro_feat_shap.tex', 'w') as f:
    f.write(latex_table)
            
# Plot, importances
shap_importance = shap_importance.sort_values(by='Feature Shap Value', ascending=True)
fig,ax = plt.subplots(figsize=(5,len(inputs)/3))
plt.barh(shap_importance['Feature'],shap_importance['Feature Shap Value'],color="black")
ax.set_yticks(list(range(0,len(inputs))),list(shap_importance.Feature))
rects = ax.patches
for rect, label in zip(rects, list(shap_importance['Feature Shap Value'])):
        ax.text(rect.get_width()+(shap_importance['Feature Shap Value'].max()*0.1), rect.get_y()+0.5*rect.get_height(), round(label,5), ha="center", va="center" )
plt.savefig("out/intro_feat_shap.eps",dpi=300,bbox_inches='tight')
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/introduction/out/intro_feat_shap.eps",dpi=300,bbox_inches='tight')
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/intro_feat_shap.eps",dpi=300,bbox_inches='tight')

################################
### Partial dependency plots ###
################################

# Shap
for count in range(0,len(inputs)):
    if len(x_d[inputs[count]].unique())==2:
            
        fig,ax = plt.subplots(figsize=(12, 8))
        x_vals=x_d.iloc[:, [count]]
        y_vals=pd.DataFrame(shap_values)[count]
        violin_parts=ax.violinplot(y_vals.loc[x_vals[inputs[count]] == 0],positions=[0], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
        for pc in violin_parts['bodies']:
            pc.set_facecolor('gray')
        violin_parts=ax.violinplot(y_vals.loc[x_vals[inputs[count]] == 1],positions=[1], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
        for pc in violin_parts['bodies']:
            pc.set_facecolor('gray')
        
        plt.xlabel(names[inputs[count]],size=20)
        plt.ylabel("SHAP value",size=20)
        plt.yticks(size=20)
        plt.xticks([0,1],["0","1"],size=20)
        plt.xlim(-0.5, 1.5)  
        sns.rugplot(x_d[inputs[count]], height=0.05, color='black')
 
        plt.savefig(f"out/intro_shap_depend_{inputs[count]}.eps",dpi=300,bbox_inches='tight')
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/introduction/out/intro_shap_depend_{inputs[count]}.eps",dpi=300,bbox_inches='tight')
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/intro_shap_depend_{inputs[count]}.eps",dpi=300,bbox_inches='tight')

    else:
        fig,ax = plt.subplots(figsize=(12, 8))
        plt.scatter(x_d.iloc[:, [count]],pd.DataFrame(shap_values)[count],color="black",s=60)
        plt.xlabel(names[inputs[count]],size=20)
        plt.ylabel("SHAP value",size=20)
        plt.yticks(size=20)
        plt.xticks(size=20)
        sns.rugplot(x_d[inputs[count]], height=0.05, color='black')

        plt.savefig(f"out/intro_shap_depend_{inputs[count]}.eps",dpi=300,bbox_inches='tight')
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/introduction/out/intro_shap_depend_{inputs[count]}.eps",dpi=300,bbox_inches='tight')
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/intro_shap_depend_{inputs[count]}.eps",dpi=300,bbox_inches='tight')

# Partial dependency, scikit learn 
for count in range(0,len(inputs)):
    if len(x_d[inputs[count]].unique())==2:
        pd_results = partial_dependence(model, x_d, features=count, kind="both", grid_resolution=10,percentiles=(0,1))
        fig,ax = plt.subplots(figsize=(12, 8))
        violin_parts=ax.violinplot(pd_results["individual"][0][:, 0],positions=[0], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
        for pc in violin_parts['bodies']:
            pc.set_facecolor('gray')
        violin_parts=ax.violinplot(pd_results["individual"][0][:, 1],positions=[1], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
        for pc in violin_parts['bodies']:
            pc.set_facecolor('gray')
        ax2 = ax.twinx()
        ax2.plot(pd_results["grid_values"][0],pd_results["average"][0],marker='o',linestyle='None',color="black",markersize=10)
        ax2.set_xticks([0,1],["0","1"],size=20)
        ax2.set_xlim(-0.5, 1.5)        
        ax2.tick_params(axis='y', which='major', labelsize=20)
        ax.set_xlabel(names[inputs[count]],size=20)
        ax.set_ylabel("Partial dependence, ICE",size=20)
        ax2.set_ylabel("Partial dependence, Average",size=20)
        sns.rugplot(x_d[inputs[count]], height=0.05, color='black')
        plt.savefig(f"out/intro_partial_depend_{inputs[count]}.png",dpi=300,bbox_inches='tight')
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/introduction/out/intro_partial_depend_{inputs[count]}.png",dpi=300,bbox_inches='tight')
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/intro_partial_depend_{inputs[count]}.png",dpi=300,bbox_inches='tight')
   
    else: 
        pd_results = partial_dependence(model, x_d, features=count, kind="both", grid_resolution=10,percentiles=(0,1))
        fig,ax = plt.subplots(figsize=(12, 8))
        for i in range(len(pd_results["individual"][0])):
            ax.plot(pd_results["grid_values"][0],pd_results["individual"][0][i],color="gray",linewidth=1,alpha=0.3) 
        ax2 = ax.twinx()
        ax2.plot(pd_results["grid_values"][0],pd_results["average"][0],color="black",linewidth=3)
        ax.set_xlabel(names[inputs[count]],size=20)
        ax.set_ylabel("Partial dependence, ICE",size=20)
        ax2.set_ylabel("Partial dependence, Average",size=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(axis='y', which='major', labelsize=20)
        sns.rugplot(x_d[inputs[count]], height=0.05, color='black')
        plt.savefig(f"out/intro_partial_depend_{inputs[count]}.png",dpi=300,bbox_inches='tight')
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/introduction/out/intro_partial_depend_{inputs[count]}.png",dpi=300,bbox_inches='tight')
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/intro_partial_depend_{inputs[count]}.png",dpi=300,bbox_inches='tight')
      
# ALE
for count in range(0,len(inputs)):
    if len(x_d[inputs[count]].unique())==2:
        ale_eff = ale(X=x_d, model=model, feature=[inputs[count]],grid_size=40,include_CI=True,plot=False)
        ale_eff=ale_eff.fillna(0)
        fig,ax = plt.subplots(figsize=(12, 8))        
        plt.plot(ale_eff.iloc[:, 0].index,ale_eff.iloc[:, 0].values, marker='o', linestyle='None',color="black",markersize=10)
        for i in range(len(ale_eff.iloc[:, 0])):
            plt.plot([ale_eff.iloc[:, 0].index[i],ale_eff.iloc[:, 0].index[i]], [ale_eff.iloc[:, 2].values[i], ale_eff.iloc[:, 3].values[i]], color='gray',linewidth=2)
        plt.xlabel(names[inputs[count]],size=20)
        plt.ylabel("Partial dependence",size=20)
        plt.yticks(size=20)
        plt.xticks([0,1],size=20)
        plt.xlim(-0.5, 1.5)
        sns.rugplot(x_d[inputs[count]], height=0.05, color='black')
        plt.savefig(f"out/intro_ale_{inputs[count]}.eps",dpi=300,bbox_inches='tight')
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/introduction/out/intro_ale_{inputs[count]}.eps",dpi=300,bbox_inches='tight')
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/intro_ale_{inputs[count]}.eps",dpi=300,bbox_inches='tight')
        
    else: 
        ale_eff = ale(X=x_d, model=model, feature=[inputs[count]],grid_size=40,include_CI=True,plot=False)
        ale_eff=ale_eff.fillna(0)
        fig,ax = plt.subplots(figsize=(12, 8))
    
        plt.plot(ale_eff.iloc[:, 0].index,ale_eff.iloc[:, 0].values, marker='o', linestyle='None',color="black",markersize=10)
        for i in range(len(ale_eff.iloc[:, 0])):
            plt.plot([ale_eff.iloc[:, 0].index[i],ale_eff.iloc[:, 0].index[i]], [ale_eff.iloc[:, 2].values[i], ale_eff.iloc[:, 3].values[i]], color='gray',linewidth=2)
    
        plt.xlabel(names[inputs[count]],size=20)
        plt.ylabel("Partial dependence",size=20)
        plt.yticks(size=20)
        plt.xticks(size=20)
        sns.rugplot(x_d[inputs[count]], height=0.05, color='black')
        plt.savefig(f"out/intro_ale_{inputs[count]}.eps",dpi=300,bbox_inches='tight')
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/introduction/out/intro_ale_{inputs[count]}.eps",dpi=300,bbox_inches='tight')
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/intro_ale_{inputs[count]}.eps",dpi=300,bbox_inches='tight')

# Interaction ALE
for i in inputs:
    vars_list=[s for s in inputs if s != i]
    for x in vars_list:
        print(i,x)
        fig,ax = plt.subplots(figsize=(12, 8))        
        ale_eff = ale(X=x_d, model=model, feature=[i,x],grid_size=40,include_CI=True,plot=False)
        heatmap = sns.heatmap(ale_eff,cmap='binary')
        plt.xlabel(names[x],size=20)
        plt.ylabel(names[i],size=20)
        xticklabels = [round(float(label.get_text()), 2) for label in heatmap.get_xticklabels()]
        yticklabels = [round(float(label.get_text()), 2) for label in heatmap.get_yticklabels()]
        heatmap.set_xticklabels(xticklabels, rotation=45, ha='center')
        heatmap.set_yticklabels(yticklabels, rotation=45, ha='right')
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Partial dependence', size=20)
        plt.savefig(f"out/intro_ale_{i}_{x}.png",dpi=300,bbox_inches='tight')
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/introduction/out/intro_ale_{i}_{x}.png",dpi=300,bbox_inches='tight')
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/intro_ale_{i}_{x}.png",dpi=300,bbox_inches='tight')

#################
### OLS model ###
#################

dummies_c = pd.get_dummies(df['country'], prefix='c').astype(int)
df = pd.concat([df, dummies_c], axis=1)
country_dummies=[
"c_Afghanistan",
"c_Albania",
"c_Algeria",
"c_Angola",
"c_Argentina",
"c_Armenia",
"c_Australia",
"c_Austria",
"c_Azerbaijan",
"c_Bahrain",
"c_Bangladesh",
"c_Barbados",
"c_Belarus",
"c_Belgium",
"c_Benin",
"c_Bhutan",
"c_Bolivia",
"c_Bosnia-Herzegovina",
"c_Botswana",
"c_Brazil",
"c_Bulgaria",
"c_Burkina Faso",
"c_Burundi",
"c_Cabo Verde",
"c_Cambodia (Kampuchea)",
"c_Cameroon",
"c_Canada",
"c_Central African Republic",
"c_Chad",
"c_Chile",
"c_China",
"c_Colombia",
"c_Comoros",
"c_Congo",
"c_Costa Rica",
"c_Croatia",
"c_Cuba",
"c_Cyprus",
"c_Czechia",
"c_DR Congo (Zaire)",
"c_Denmark",
"c_Djibouti",
"c_Dominican Republic",
"c_East Timor",
"c_Ecuador",
"c_Egypt",
"c_El Salvador",
"c_Equatorial Guinea",
"c_Eritrea",
"c_Estonia",
"c_Ethiopia",
"c_Fiji",
"c_Finland",
"c_France",
"c_Gabon",
"c_Gambia",
"c_Georgia",
"c_Germany",
"c_Ghana",
"c_Greece",
"c_Guatemala",
"c_Guinea",
"c_Guinea-Bissau",
"c_Guyana",
"c_Haiti",
"c_Honduras",
"c_Hungary",
"c_Iceland",
"c_India",
"c_Indonesia",
"c_Iran",
"c_Iraq",
"c_Ireland",
"c_Israel",
"c_Italy",
"c_Ivory Coast",
"c_Jamaica",
"c_Japan",
"c_Jordan",
"c_Kazakhstan",
"c_Kenya",
"c_Kuwait",
"c_Kyrgyzstan",
"c_Laos",
"c_Latvia",
"c_Lebanon",
"c_Lesotho",
"c_Liberia",
"c_Libya",
"c_Lithuania",
"c_Luxembourg",
"c_Madagascar",
"c_Malawi",
"c_Malaysia",
"c_Maldives",
"c_Mali",
"c_Malta",
"c_Mauritania",
"c_Mauritius",
"c_Mexico",
"c_Moldova",
"c_Mongolia",
"c_Montenegro",
"c_Morocco",
"c_Mozambique",
"c_Myanmar (Burma)",
"c_Namibia",
"c_Nepal",
"c_Netherlands",
"c_New Zealand",
"c_Nicaragua",
"c_Niger",
"c_Nigeria",
"c_North Macedonia",
"c_Norway",
"c_Oman",
"c_Pakistan",
"c_Panama",
"c_Papua New Guinea",
"c_Paraguay",
"c_Peru",
"c_Philippines",
"c_Poland",
"c_Portugal",
"c_Qatar",
"c_Romania",
"c_Russia (Soviet Union)",
"c_Rwanda",
"c_Saudi Arabia",
"c_Senegal",
"c_Serbia",
"c_Sierra Leone",
"c_Singapore",
"c_Slovakia",
"c_Slovenia",
"c_Solomon Islands",
"c_Somalia",
"c_South Korea",
"c_South Sudan",
"c_Spain",
"c_Sri Lanka",
"c_Sudan",
"c_Suriname",
"c_Sweden",
"c_Switzerland",
"c_Syria",
"c_Tajikistan",
"c_Tanzania",
"c_Thailand",
"c_Togo",
"c_Trinidad and Tobago",
"c_Tunisia",
"c_Turkey",
"c_Turkmenistan",
"c_Uganda",
"c_Ukraine",
"c_United Arab Emirates",
"c_United Kingdom",
"c_United States of America",
"c_Uruguay",
"c_Uzbekistan",
"c_Venezuela",
"c_Vietnam",
"c_Yemen (North Yemen)",
"c_Zambia",
"c_Zimbabwe",
"c_eSwatini",
]

# Model I
X=df[['sb_fatalities_lag1','d_civil_conflict_zeros_decay','d_civil_war_zeros_decay','d_neighbors_sb_fatalities_lag1']+country_dummies]
X = sm.add_constant(X)
y=df[['sb_fatalities_log']]
est1 = sm.OLS(y, X)
est1 = est1.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
est1.summary()

# Model II
X=df[['sb_fatalities_lag1','d_civil_conflict_zeros_decay','d_civil_war_zeros_decay','d_neighbors_sb_fatalities_lag1','inf_mort','gdp','growth_norm','oil_share']+country_dummies]
X = sm.add_constant(X)
y=df[['sb_fatalities_log']]
est2 = sm.OLS(y, X)
est2 = est2.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
est2.summary()

# Model III
X=df[['sb_fatalities_lag1','d_civil_conflict_zeros_decay','d_civil_war_zeros_decay','d_neighbors_sb_fatalities_lag1','inf_mort','gdp','growth_norm','oil_share','pop','male_youth_share','ethnic_frac','eys_male']+country_dummies]
X = sm.add_constant(X)
y=df[['sb_fatalities_log']]
est3 = sm.OLS(y, X)
est3 = est3.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
est3.summary()

# Model IV
X=df[['sb_fatalities_lag1','d_civil_conflict_zeros_decay','d_civil_war_zeros_decay','d_neighbors_sb_fatalities_lag1','inf_mort','gdp','growth_norm','oil_share','pop','male_youth_share','ethnic_frac','eys_male','temp_norm','withdrawl_norm']+country_dummies]
X = sm.add_constant(X)
y=df[['sb_fatalities_log']]
est4 = sm.OLS(y, X)
est4 = est4.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
est4.summary()

# Model V
X=df[['sb_fatalities_lag1','d_civil_conflict_zeros_decay','d_civil_war_zeros_decay','d_neighbors_sb_fatalities_lag1','inf_mort','gdp','growth_norm','oil_share','pop','male_youth_share','ethnic_frac','eys_male','temp_norm','withdrawl_norm','polyarchy','libdem','egaldem','civlib','exlsocgr']+country_dummies]
X = sm.add_constant(X)
y=df[['sb_fatalities_log']]
est5 = sm.OLS(y, X)
est5 = est5.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
est5.summary()
 
summary = summary_col([est1,est2,est3,est4,est5], float_format='%0.5f', stars=True,regressor_order=['sb_fatalities_lag1','d_civil_conflict_zeros_decay','d_civil_war_zeros_decay','d_neighbors_sb_fatalities_lag1','inf_mort','gdp','growth_norm','oil_share','pop','male_youth_share','ethnic_frac','eys_male','temp_norm','withdrawl_norm','polyarchy','libdem','egaldem','civlib','exlsocgr']+country_dummies)
print(summary.as_latex())





