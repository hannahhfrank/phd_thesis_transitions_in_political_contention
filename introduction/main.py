import pandas as pd
import numpy as np
from functions import data_split,earth_mover_distance,preprocess_min_max_group
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
import matplotlib.pyplot as plt
from pdpbox import pdp
import shap
from sklearn.inspection import permutation_importance
from PyALE import ale
import lime
import seaborn as sns
from sklearn.dummy import DummyRegressor
import geopandas as gpd
import matplotlib as mpl
from matplotlib import cm 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import os 
from sklearn.svm import SVR
import random
random.seed(42)
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

# Plot parameters 
plot_params = {"text.usetex":True,"font.family":"serif","font.size":15,"xtick.labelsize":15,"ytick.labelsize":15,"axes.labelsize":15,"figure.titlesize":20,}
plt.rcParams.update(plot_params)

df=pd.read_csv("out/df_consensus.csv",index_col=0)
duplicate_rows = df[df.duplicated(subset=['year', 'country'])]

# Simple random forest
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)],
               'max_features': ["sqrt", "log2", None],
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],
               'min_samples_split': [2,5,10],
               'min_samples_leaf': [1,2,4],
               }

for var in ['sb_fatalities','time_since_civil_conflict','sb_fatalities_lag1','total_fatalities_lag1','time_since_independ','growth','gdp','pop_density','pop','nat_res','hydro_carb','percip','agri','ethnic_frac','dominant_share','rel_frac','discriminated_share','eys_male','polity2','polity2_sqr','polity2_change','polity2_anoc','rugged']:
    fig,ax = plt.subplots()
    df[var].hist()
    plt.title(var)

# Transforms
df["sb_fatalities_log"]=np.log(df["sb_fatalities"]+1)
df["sb_fatalities_lag1_log"]=np.log(df["sb_fatalities_lag1"]+1)
df["total_fatalities_lag1_log"]=np.log(df["total_fatalities_lag1"]+1)
df["gdp_log"]=np.log(df["gdp"])
preprocess_min_max_group(df,"growth","country")
df["pop_density_log"]=np.log(df["pop_density"])
df["pop_log"]=np.log(df["pop"])
df["nat_res_log"]=np.log(df["nat_res"])
df["hydro_carb_log"]=np.log(df["hydro_carb"]+1)

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
"c_Mali",
"c_Mauritania",
"c_Mauritius",
"c_Mexico",
"c_Moldova",
"c_Mongolia",
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
"c_Singapore",
"c_Slovakia",
"c_Slovenia",
"c_Solomon Islands",
"c_South Africa",
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
"c_Yemen (North Yemen)",
"c_Zambia",
"c_Zimbabwe",
"c_eSwatini",
]


# Model I
X=df[['time_since_civil_conflict','sb_fatalities_lag1_log','total_fatalities_lag1_log','time_since_independ']+country_dummies]
X = sm.add_constant(X)
y=df[['sb_fatalities_log']]
est1 = sm.OLS(y, X)
est1 = est1.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
est1.summary()

# Model II
X=df[['time_since_civil_conflict','sb_fatalities_lag1_log','total_fatalities_lag1_log','time_since_independ','growth_norm','gdp_log','nat_res_log','hydro_carb_log']+country_dummies]
X = sm.add_constant(X)
y=df[['sb_fatalities_log']]
est2 = sm.OLS(y, X)
est2 = est2.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
est2.summary()

# Model III
X=df[['time_since_civil_conflict','sb_fatalities_lag1_log','total_fatalities_lag1_log','time_since_independ','growth_norm','gdp_log','nat_res_log','hydro_carb_log','pop_density_log','pop_log','ethnic_frac','dominant_share','rel_frac','discriminated_share']+country_dummies]
X = sm.add_constant(X)
y=df[['sb_fatalities_log']]
est3 = sm.OLS(y, X)
est3 = est3.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
est3.summary()

# Model IV
X=df[['time_since_civil_conflict','sb_fatalities_lag1_log','total_fatalities_lag1_log','time_since_independ','growth_norm','gdp_log','nat_res_log','hydro_carb_log','pop_density_log','pop_log','ethnic_frac','dominant_share','rel_frac','discriminated_share','percip','agri','rugged']+country_dummies]
X = sm.add_constant(X)
y=df[['sb_fatalities_log']]
est4 = sm.OLS(y, X)
est4 = est4.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
est4.summary()

# Model V
X=df[['time_since_civil_conflict','sb_fatalities_lag1_log','total_fatalities_lag1_log','time_since_independ','growth_norm','gdp_log','nat_res_log','hydro_carb_log','pop_density_log','pop_log','ethnic_frac','dominant_share','rel_frac','discriminated_share','percip','agri','rugged','eys_male','polity2','polity2_sqr','polity2_change','polity2_anoc']+country_dummies]
X = sm.add_constant(X)
y=df[['sb_fatalities_log']]
est5 = sm.OLS(y, X)
est5 = est5.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
est5.summary()
 
summary = summary_col([est1,est2,est3,est4,est5], float_format='%0.5f', stars=True,regressor_order=['time_since_civil_conflict','sb_fatalities_lag1_log','total_fatalities_lag1_log','time_since_independ','growth_norm','gdp_log','nat_res_log','hydro_carb_log','pop_density_log','pop_log','ethnic_frac','dominant_share','rel_frac','discriminated_share','percip','agri','rugged','eys_male','polity2','polity2_sqr','polity2_change','polity2_anoc']+country_dummies)
print(summary.as_latex())


#############################
### Test all coallitions ####
#############################

# Function to generate all coalitions
def generate_coalitions(variables):
    coalitions = []
    # Iterate over all possible subset sizes
    for r in range(len(variables) + 1):
        # Generate all combinations of the current size
        combinations = list(itertools.combinations(variables, r))
        coalitions.extend([list(comb) for comb in combinations])
    return coalitions

sig_coallitions={"var":[],"coallition":[],"par":[],"p":[]}

for var in ['time_since_civil_conflict','sb_fatalities_lag1_log','total_fatalities_lag1_log','time_since_independ','growth_norm','gdp_log','nat_res_log','hydro_carb_log','pop_density_log','pop_log','ethnic_frac','dominant_share','rel_frac','discriminated_share','percip','agri','rugged','eys_male','polity2','polity2_sqr','polity2_change','polity2_anoc']:
#for var in ['d_civil_war_zeros_decay']:

    print(var)
    vars_s = [s for s in ['time_since_civil_conflict','sb_fatalities_lag1_log','total_fatalities_lag1_log','time_since_independ','growth_norm','gdp_log','nat_res_log','hydro_carb_log','pop_density_log','pop_log','ethnic_frac','dominant_share','rel_frac','discriminated_share','percip','agri','rugged','eys_male','polity2','polity2_sqr','polity2_change','polity2_anoc'] if s != var]
    all_coalitions = generate_coalitions(vars_s)
    coallitions_random = random.sample(all_coalitions, 5000)

    for x in coallitions_random:
        #bootstrap_sample = df.sample(n=len(df), replace=True)

        X=df[[var]+x+country_dummies]
        X = sm.add_constant(X)
        y=df[['sb_fatalities_log']]
        est1 = sm.OLS(y, X)
        est1 = est1.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
        sig_coallitions["var"].append(var)
        sig_coallitions["coallition"].append(x)
        sig_coallitions["par"].append(est1.params[1])
        sig_coallitions["p"].append(est1.pvalues[1])

    coallitions_df=pd.DataFrame(sig_coallitions)
    coallitions_df.to_csv("coallitions_df_consesus.csv")
    
tab={"var":[],"pos_sig":[],"pos_nonsig":[],"neg_nonsig":[],"neg_sig":[]}

for var in ['time_since_civil_conflict','sb_fatalities_lag1_log','total_fatalities_lag1_log','time_since_independ','growth_norm','gdp_log','nat_res_log','hydro_carb_log','pop_density_log','pop_log','ethnic_frac','dominant_share','rel_frac','discriminated_share','percip','agri','rugged','eys_male','polity2','polity2_sqr','polity2_change','polity2_anoc']:
    tab["var"].append(var)
    tab["pos_sig"].append(len(coallitions_df.loc[(coallitions_df["var"]==var)&(coallitions_df["par"]>0)&(coallitions_df["p"]<0.05)])/len(coallitions_df.loc[(coallitions_df["var"]==var)]))
    tab["pos_nonsig"].append(len(coallitions_df.loc[(coallitions_df["var"]==var)&(coallitions_df["par"]>0)&(coallitions_df["p"]>=0.05)])/len(coallitions_df.loc[(coallitions_df["var"]==var)]))
    tab["neg_nonsig"].append(len(coallitions_df.loc[(coallitions_df["var"]==var)&(coallitions_df["par"]<=0)&(coallitions_df["p"]>=0.05)])/len(coallitions_df.loc[(coallitions_df["var"]==var)]))
    tab["neg_sig"].append(len(coallitions_df.loc[(coallitions_df["var"]==var)&(coallitions_df["par"]<=0)&(coallitions_df["p"]<0.05)])/len(coallitions_df.loc[(coallitions_df["var"]==var)]))
tab_df=pd.DataFrame(tab)

##########################
### Critical variables ###
##########################
#names={'time_since_civil_conflict':"Time since civil conflict",
#       'sb_fatalities_lag1_log':"t-1 lag number of fatalities (log)",
#       'total_fatalities_lag1_log':"t-1 lag number of fatalities others (log)",
#       'time_since_independ':"Time since independence",
#       'growth_norm':"GDP growth (norm)",
#       'gdp_log':"GDP per capita (log)",
#       'pop_density_log': "Population density (log)",
#       'pop_log': "Population size (log)",
#       'nat_res_log': "Natural resource rents (log)",
#       'hydro_carb_log': "Hydrocarbonat rents (log)",
#       'percip':"Precipitation",
#       'agri':"Share agricultural land",
#       'ethnic_frac':"Ethnolinguistic fractionalization",
#       'dominant_share': "Dominant share",
#       'rel_frac':"Religous fractionalization",
#       'discriminated_share':"Discriminated share",
#       'eys_male':"Expected years of schooling, male",
#       'polity2':"PolityIV",
#       'polity2_sqr':"PolityIV (sqr)",
#       'polity2_change':"PolityIV (change)",
#       'polity2_anoc':"Anocracy",
#       'rugged':"Ruggedness"
#       }

cmap = plt.cm.get_cmap('gist_earth')
colors = cmap(np.linspace(0, 1, 11))
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "time_since_civil_conflict"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "sb_fatalities_lag1_log"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "total_fatalities_lag1_log"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "growth_norm"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "hydro_carb_log"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "pop_log"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "ethnic_frac"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "dominant_share"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "rel_frac"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "discriminated_share"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "percip"])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(22, 8),gridspec_kw={'height_ratios': [3, 1]})
palette = {'time_since_civil_conflict': colors[0], 
           'sb_fatalities_lag1_log':colors[1], 
           'total_fatalities_lag1_log':colors[2], 
           'growth_norm':colors[3],
           'hydro_carb_log':colors[4], 
           'pop_log':colors[5],
           'ethnic_frac':colors[6], 
           'dominant_share':colors[7],
           "rel_frac":colors[8],
           "discriminated_share":colors[9],           
           "percip":colors[10],           
           }
# Iterate over unique categories in 'Category' column
for category in ["time_since_civil_conflict","sb_fatalities_lag1_log","total_fatalities_lag1_log","growth_norm","hydro_carb_log"]:
    # Filter data for the current category
    subset = coallitions_df[coallitions_df['var'] == category]
    # Plot density plot for 'Value' within each category
    sns.kdeplot(subset['par'], label=category, shade=True, color=palette[category],ax=ax1)

# Iterate over unique categories in 'Category' column
for category in ["pop_log","ethnic_frac","dominant_share","rel_frac","discriminated_share"]:
    # Filter data for the current category
    subset = coallitions_df[coallitions_df['var'] == category]
    # Plot density plot for 'Value' within each category
    sns.kdeplot(subset['par'], label=category, shade=True, color=palette[category],ax=ax2)


# Add labels and title
ax1.set_xlabel('Slope coefficient')
ax1.set_ylabel('Density')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, ["Time since civil conflict","t-1 lag number of fatalities (log)","t-1 lag number of fatalities others (log)","GDP growth (norm)","Hydrocarbonat rents (log)"], title='Variables', bbox_to_anchor=(0.48, 1), loc='upper left',fontsize=12)

# Add a table to the plot
data = [
        ["","Positive (sig)","Positive (non sig)","Negattive (non sig)","Negative (sig)"],
        ["Time since civil conflict",     0.0000 ,     0.0000  ,    0.0000  , 1.0000],
        ["t-1 lag number of fatalities (log)" , 1.0000,      0.0000,      0.0000,   0.0000],
        ["t-1 lag number of fatalities others (log)", 1.0000  ,    0.0000  ,    0.0000 ,  0.0000],
        ["GDP growth (norm)", 0.0000 ,     0.0000   ,   0.0000 ,  1.0000],
        ["Hydrocarbonat rents (log)",  0.0000  ,    0.0000   ,   0.9698 ,  0.0302],
        ["Precipitation", 0.0000   ,   0.0000   ,   0.0000 ,  1.0000],
        ]
table = ax3.table(cellText=data, loc='upper center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(14)

# Hide axes for ax2 (table plot)
ax3.axis('off')

# Remove horizontal lines from the table
for key, cell in table.get_celld().items():
    cell.set_linewidth(0)
    
# Increase column width in the table
cell_dict = table.get_celld()
for key in cell_dict.keys():
    cell_dict[key]._text.set_fontsize(12)  # Set font size for text in cells

# Adjust column widths (adjust as needed)
table.auto_set_column_width(col=list(range(len(data[0]))))


# Add labels and title
ax2.set_xlabel('Slope coefficient')
ax2.set_ylabel('Density')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, ["Population size (log)","Ethnolinguistic fractionalization","Dominant share","Religous fractionalization","Discriminated share"], title='Variables', bbox_to_anchor=(0, 1), loc='upper left',fontsize=12)

# Add a table to the plot
data = [
        ["","Positive (sig)","Positive (non sig)","Negattive (non sig)","Negative (sig)"],
        ["Population size (log)",    0.2346 ,     0.7318   ,   0.0336 ,  0.0000],
        ["Ethnolinguistic fractionalization", 0.5366    ,  0.4634  ,    0.0000 ,  0.0000],
        ["Dominant share", 0.0000   ,   0.0102  ,    0.9690  , 0.0208],
        ["Religous fractionalization", 0.0000   ,   0.0018   ,   0.9874  , 0.0108],
        ["Discriminated share", 0.0000   ,   0.9994   ,   0.0006  , 0.0000],
        ]
table = ax4.table(cellText=data, loc='upper center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(14)

# Hide axes for ax2 (table plot)
ax4.axis('off')

# Remove horizontal lines from the table
for key, cell in table.get_celld().items():
    cell.set_linewidth(0)
    
# Increase column width in the table
cell_dict = table.get_celld()
for key in cell_dict.keys():
    cell_dict[key]._text.set_fontsize(12)  # Set font size for text in cells

# Adjust column widths (adjust as needed)
table.auto_set_column_width(col=list(range(len(data[0]))))

plt.savefig(f"out/critical_vars.png",dpi=300,bbox_inches='tight')
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/interpretability/out/critical_vars.png",dpi=300,bbox_inches='tight')
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/critical_vars.png",dpi=300,bbox_inches='tight')



sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "time_since_independ"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "gdp_log"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "nat_res_log"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "pop_density_log"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "agri"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "rugged"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "eys_male"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "polity2"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "polity2_sqr"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "polity2_change"])
sns.kdeplot(coallitions_df["par"][coallitions_df['var'] == "polity2_anoc"])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(22, 8),gridspec_kw={'height_ratios': [3, 1]})
palette = {'time_since_independ': colors[0], 
           'gdp_log':colors[1], 
           'nat_res_log':colors[2], 
           'pop_density_log':colors[3],
           'agri':colors[4], 
           'rugged':colors[5],
           'eys_male':colors[6], 
           'polity2':colors[7],
           "polity2_sqr":colors[8],
           "polity2_change":colors[9],           
           "polity2_anoc":colors[10],           
           }
# Iterate over unique categories in 'Category' column
for category in ["time_since_independ","nat_res_log","agri","polity2","polity2_anoc","polity2_change"]:
    # Filter data for the current category
    subset = coallitions_df[coallitions_df['var'] == category]
    # Plot density plot for 'Value' within each category
    sns.kdeplot(subset['par'], label=category, shade=True, color=palette[category],ax=ax1)

# Iterate over unique categories in 'Category' column
for category in ["gdp_log","pop_density_log","rugged",]:
    # Filter data for the current category
    subset = coallitions_df[coallitions_df['var'] == category]
    # Plot density plot for 'Value' within each category
    sns.kdeplot(subset['par'], label=category, shade=True, color=palette[category],ax=ax2)

# Add labels and title
ax1.set_xlabel('Slope coefficient')
ax1.set_ylabel('Density')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, ["Time since independence","Natural resource rents (log)","Share agricultural land","PolityIV","Anocracy","PolityIV (change)"], title='Variables', bbox_to_anchor=(0.48, 1), loc='upper left',fontsize=12)

# Add a table to the plot
data = [
        ["","Positive (sig)","Positive (non sig)","Negattive (non sig)","Negative (sig)"],
        ["Time since independence",     0.5056   ,   0.2264   ,   0.2166 ,  0.0514],
        ["Natural resource rents (log)",  0.0000   ,   0.6802   ,   0.3198 ,  0.0000],
        ["Share agricultural land", 0.0000   ,   0.3716   ,   0.5764  , 0.0520],
        ["PolityIV",  0.0000  ,    0.0532   ,   0.9468  , 0.0000],
        ["Anocracy", 0.0000  ,    0.2134   ,   0.7866 ,  0.0000 ],
        ["PolityIV (change)", 0.0010  ,    0.3652   ,   0.6338  , 0.0000],
        ["PolityIV (sqr)", 0.0000   ,   0.1590   ,   0.8410 ,  0.0000],
        ]
table = ax3.table(cellText=data, loc='upper center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(14)

# Hide axes for ax2 (table plot)
ax3.axis('off')

# Remove horizontal lines from the table
for key, cell in table.get_celld().items():
    cell.set_linewidth(0)
    
# Increase column width in the table
cell_dict = table.get_celld()
for key in cell_dict.keys():
    cell_dict[key]._text.set_fontsize(12)  # Set font size for text in cells

# Adjust column widths (adjust as needed)
table.auto_set_column_width(col=list(range(len(data[0]))))


# Add labels and title
ax2.set_xlabel('Slope coefficient')
ax2.set_ylabel('Density')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, ["GDP per capita (log)","Population density (log)","Ruggedness"], title='Variables', bbox_to_anchor=(0, 1), loc='upper left',fontsize=12)

# Add a table to the plot
data = [
        ["","Positive (sig)","Positive (non sig)","Negattive (non sig)","Negative (sig)"],
        ["GDP per capita (log)", 0.0140   ,   0.2226    ,  0.4614  , 0.3020],
        ["Population density (log)",  0.1830   ,   0.2716   ,   0.5428 ,  0.0026],
        ["Ruggedness",0.1222 ,     0.4908    ,  0.2906  , 0.0964],
        ]
table = ax4.table(cellText=data, loc='upper center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(14)

# Hide axes for ax2 (table plot)
ax4.axis('off')

# Remove horizontal lines from the table
for key, cell in table.get_celld().items():
    cell.set_linewidth(0)
    
# Increase column width in the table
cell_dict = table.get_celld()
for key in cell_dict.keys():
    cell_dict[key]._text.set_fontsize(12)  # Set font size for text in cells

# Adjust column widths (adjust as needed)
table.auto_set_column_width(col=list(range(len(data[0]))))

plt.savefig(f"out/critical_vars2.png",dpi=300,bbox_inches='tight')
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/interpretability/out/critical_vars2.png",dpi=300,bbox_inches='tight')
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/critical_vars2.png",dpi=300,bbox_inches='tight')


