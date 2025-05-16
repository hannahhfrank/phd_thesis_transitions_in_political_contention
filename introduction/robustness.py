import pandas as pd
import numpy as np
from functions import preprocess_min_max_group
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import statsmodels.api as sm
import os 
import random
random.seed(42)
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'

# Prepare data
df=pd.read_csv("out/df_consensus.csv",index_col=0)
df.isnull().any()
df["sb_fatalities_log"]=np.log(df["sb_fatalities"]+1)
df["sb_fatalities_lag1_log"]=np.log(df["sb_fatalities_lag1"]+1)
df["total_fatalities_lag1_log"]=np.log(df["total_fatalities_lag1"]+1)
df["gdp_log"]=np.log(df["gdp"])
preprocess_min_max_group(df,"growth","country")
df["pop_density_log"]=np.log(df["pop_density"])
df["pop_log"]=np.log(df["pop"])
df["nat_res_log"]=np.log(df["nat_res"]+1)
df["hydro_carb_log"]=np.log(df["hydro_carb"]+1)

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

for var in ['sb_fatalities_log','total_fatalities_lag1_log','sb_fatalities_lag1_log','time_since_civil_conflict','time_since_independ','eys_male','growth_norm','gdp_log','hydro_carb_log','nat_res_log','pop_log','pop_density_log','ethnic_frac','rel_frac','dominant_share','discriminated_share','polity2_anoc','polity2_change','polity2','polity2_sqr','rugged','agri','percip']:
    fig,ax = plt.subplots()
    df[var].hist()
    plt.title(var)

# Function to get coalitions
def generate_coalitions(variables):
    coalitions = []
    for r in range(len(variables) + 1):
        combinations = list(itertools.combinations(variables, r))
        coalitions.extend([list(comb) for comb in combinations])
    return coalitions

# Get 5000 estimates for each variable
sig_coallitions={"var":[],"coallition":[],"par":[],"p":[]}
for var in ['total_fatalities_lag1_log','sb_fatalities_lag1_log','time_since_civil_conflict','time_since_independ','eys_male','growth_norm','gdp_log','hydro_carb_log','nat_res_log','pop_log','pop_density_log','ethnic_frac','rel_frac','dominant_share','discriminated_share','polity2_anoc','polity2_change','polity2','polity2_sqr','rugged','agri','percip']:
    print(var)
    vars_s = [s for s in ['total_fatalities_lag1_log','sb_fatalities_lag1_log','time_since_civil_conflict','time_since_independ','eys_male','growth_norm','gdp_log','hydro_carb_log','nat_res_log','pop_log','pop_density_log','ethnic_frac','rel_frac','dominant_share','discriminated_share','polity2_anoc','polity2_change','polity2','polity2_sqr','rugged','agri','percip'] if s != var]
    all_coalitions = generate_coalitions(vars_s)
    coallitions_random = random.sample(all_coalitions, 5000)
    for x in coallitions_random:
        if "polity2_sqr" in x:
            if "polity2" not in x:
                x=x+["polity2"]
                
        if var=="polity2_sqr":
            if "polity2" not in x:
                x=x+["polity2"]
                
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
    coallitions_df.to_csv("out/coallitions_df_consesus.csv") 
coallitions_df=pd.read_csv("coallitions_df_consesus.csv",index_col=0)

# Calculate proportions and save
tab={"var":[],"pos_sig":[],"pos_nonsig":[],"neg_nonsig":[],"neg_sig":[]}
for var in ['total_fatalities_lag1_log','sb_fatalities_lag1_log','time_since_civil_conflict','time_since_independ','eys_male','growth_norm','gdp_log','hydro_carb_log','nat_res_log','pop_log','pop_density_log','ethnic_frac','rel_frac','dominant_share','discriminated_share','polity2_anoc','polity2_change','polity2','polity2_sqr','rugged','agri','percip']:
    tab["var"].append(var)
    tab["pos_sig"].append(len(coallitions_df.loc[(coallitions_df["var"]==var)&(coallitions_df["par"]>0)&(coallitions_df["p"]<0.05)])/len(coallitions_df.loc[(coallitions_df["var"]==var)]))
    tab["pos_nonsig"].append(len(coallitions_df.loc[(coallitions_df["var"]==var)&(coallitions_df["par"]>0)&(coallitions_df["p"]>=0.05)])/len(coallitions_df.loc[(coallitions_df["var"]==var)]))
    tab["neg_nonsig"].append(len(coallitions_df.loc[(coallitions_df["var"]==var)&(coallitions_df["par"]<=0)&(coallitions_df["p"]>=0.05)])/len(coallitions_df.loc[(coallitions_df["var"]==var)]))
    tab["neg_sig"].append(len(coallitions_df.loc[(coallitions_df["var"]==var)&(coallitions_df["par"]<=0)&(coallitions_df["p"]<0.05)])/len(coallitions_df.loc[(coallitions_df["var"]==var)]))
tab_df=pd.DataFrame(tab)
tab_df.to_latex("out/tab_df.tex",index=False)





