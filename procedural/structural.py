import pandas as pd
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.dummy import DummyClassifier,DummyRegressor
from sklearn.linear_model import LinearRegression
import numpy as np 
from functions import preprocess_min_max_group,gen_model
from sklearn.model_selection import GridSearchCV,PredefinedSplit
from sklearn.inspection import PartialDependenceDisplay,partial_dependence
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc,brier_score_loss,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, mean_squared_error
import geopandas as gpd
import matplotlib as mpl
from matplotlib import cm 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import math
import random
random.seed(42)

# Plot parameters 
plot_params = {"text.usetex":True,"font.family":"serif","font.size":15,"xtick.labelsize":15,"ytick.labelsize":15,"axes.labelsize":15,"figure.titlesize":20,}
plt.rcParams.update(plot_params)

# Simple random forest
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)],
               #'max_features': ["sqrt", "log2", None],
               #'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],
               #'min_samples_split': [2,5,10],
               #'min_samples_leaf': [1,2,4],
               }


# Inputs
demog_theme=['pop','pop_density','pop_density_id','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','group_counts_id','monopoly_share','monopoly_share_id','discriminated_share','discriminated_share_id','powerless_share','powerless_share_id','dominant_share','dominant_share_id','ethnic_frac','ethnic_frac_id','rel_frac','rel_frac_id','lang_frac','lang_frac_id','race_frac','race_frac_id']
geog_theme=['land','forest','forest_id','temp_norm','temp_id','co2','co2_id','percip','percip_id','waterstress','waterstress_id','agri_land','agri_land_id','arable_land','arable_land_id','rugged','soil','desert','tropical','cont_africa','cont_asia','d_neighbors_con','no_neigh','d_neighbors_non_dem','libdem_id_neigh']
econ_theme=['oil_deposits','oil_deposits_id','oil_production','oil_production_id','oil_exports','oil_exports_id','natres_share','natres_share_id','oil_share','oil_share_id','gas_share','gas_share_id','coal_share','coal_share_id','forest_share','forest_share_id','minerals_share','minerals_share_id','gdp','gdp_id','gni','gni_id','gdp_growth_norm','gdp_growth_id','unemploy','unemploy_id','unemploy_male','unemploy_male_id','inflat','inflat_id','conprice','conprice_id','undernour','undernour_id','foodprod_norm','foodprod_id','water_rural','water_rural_id','water_urb','water_urb_id','agri_share','agri_share_id','trade_share','trade_share_id','fert','fert_id','lifeexp_female','lifeexp_female_id','lifeexp_male','lifeexp_male_id','pop_growth_norm','pop_growth_id','inf_mort','inf_mort_id','mig_norm','exports','exports_id','imports','imports_id','primary_female','primary_female_id','primary_male','primary_male_id','second_female','second_female_id','second_male','second_male_id','tert_female','tert_female_id','tert_male','tert_male_id','eys','eys_id','eys_male','eys_male_id','eys_female','eys_female_id','mys','mys_id','mys_male','mys_male_id','mys_female','mys_female_id']
pol_theme=['armedforces_share','armedforces_share_id','milex_share','milex_share_id','corruption','corruption_id', 'effectiveness', 'effectiveness_id', 'polvio','polvio_id','regu','regu_id','law','law_id','account','account_id','tax','tax_id','broadband','broadband_id','telephone','telephone_id','internet_use','internet_use_id','mobile','mobile_id','polyarchy','libdem','libdem_id','partipdem','delibdem','egaldem','civlib','phyvio','pollib','privlib','execon','execon_id','exgender','exgender_id','exgeo','exgeo_id','expol','expol_id','exsoc','exsoc_id','shutdown','shutdown_id','filter','filter_id','tenure_months','tenure_months_id','dem_duration','dem_duration_id','election_recent','election_recent_id','lastelection','lastelection_id']


                                ###############
                                ### Protest ###
                                ###############
                                
# List of microstates: 
exclude={"Dominica":54,
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
         "Samoa":990,
         "German Democratic Republic":265,
         "Czechoslovakia":315,
         "Yugoslavia":345,
         "Abkhazia":396,
         "South Ossetia":397,
         "Yemen, People's Republic of":680,
         "Taiwan":713, 
         "Bahamas":31,
         "Belize":80,
         "Brunei Darussalam":835, 
         "Kosovo":347, 
         "Democratic Peoples Republic of Korea":731}                       
                                
base=pd.read_csv("data/data_out/acled_cm_protest.csv",index_col=0)
base = base[["dd","gw_codes","n_protest_events"]][~base['gw_codes'].isin(list(exclude.values()))]
ucdp=pd.read_csv("data/data_out/ucdp_cm_sb.csv",index_col=0)
ucdp = ucdp[["dd","gw_codes"]][~ucdp['gw_codes'].isin(list(exclude.values()))]
base = ucdp.merge(base, on=["dd","gw_codes"])
base=base.dropna()
                            
#####################
### History theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Transforms
x["pop_refugee"]=np.log(x["pop_refugee"]+1)

# Categorical
target="d_protest"
inputs=['d_protest_lag1',"d_protest_zeros_decay","d_neighbors_proteset_event_counts_lag1","regime_duration",'pop_refugee','pop_refugee_id']
history_protest_df,history_protest_evals=gen_model(y,x,target,inputs,name="history_protest",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
history_protest_df.to_csv("out/history_protest_df_cm.csv")
history_protest_evals_df = pd.DataFrame.from_dict(history_protest_evals, orient='index').reset_index()

########################
### Demography theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Categorical
target="d_protest"
inputs=demog_theme
demog_protest_df,demog_protest_evals=gen_model(y,x,target,inputs,name="demog_protest",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
demog_protest_df.to_csv("out/demog_protest_df_cm.csv")
demog_protest_evals_df = pd.DataFrame.from_dict(demog_protest_evals, orient='index').reset_index()
     
#####################################
### Geography & environment theme ###
#####################################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_geog_full_cm.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

x.country.unique()
x.gw_codes.unique()

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
preprocess_min_max_group(x,"temp","country")

# Categorical
target="d_protest"
inputs=geog_theme
geog_protest_df,geog_protest_evals=gen_model(y,x,target,inputs,name="geog_protest",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
geog_protest_df.to_csv("out/geog_protest_df_cm.csv")
geog_protest_evals_df = pd.DataFrame.from_dict(geog_protest_evals, orient='index').reset_index()

#####################
### Economy theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Transforms
x["oil_deposits"]=np.log(x["oil_deposits"]+1)
x["oil_production"]=np.log(x["oil_production"]+1)
x["oil_exports"]=np.log(x["oil_exports"]+1)
x["gdp"]=np.log(x["gdp"]+1)
preprocess_min_max_group(x,"gdp_growth","country")
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)
preprocess_min_max_group(x,"foodprod","country")
preprocess_min_max_group(x,"pop_growth","country")
preprocess_min_max_group(x,"mig","country")

# Categorical
target="d_protest"
inputs=econ_theme
econ_protest_df,econ_protest_evals=gen_model(y,x,target,inputs,name="econ_protest",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
econ_protest_df.to_csv("out/econ_protest_df_cm.csv")
econ_protest_evals_df = pd.DataFrame.from_dict(econ_protest_evals, orient='index').reset_index()

###############################
### Regime and policy theme ###
###############################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_pol_full_cm.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)

# Categorical
target="d_protest"
inputs=pol_theme
pol_protest_df,pol_protest_evals=gen_model(y,x,target,inputs,name="pol_protest",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
pol_protest_df.to_csv("out/pol_protest_df_cm.csv")
pol_protest_evals_df = pd.DataFrame.from_dict(pol_protest_evals, orient='index').reset_index()

################
### Ensemble ###
################

predictions=pd.concat([history_protest_df.preds_proba,demog_protest_df.preds_proba,geog_protest_df.preds_proba,econ_protest_df.preds_proba,pol_protest_df.preds_proba], axis=1)
weights=[1-history_protest_evals["brier"],1-demog_protest_evals["brier"],1-geog_protest_evals["brier"],1-econ_protest_evals["brier"],1-pol_protest_evals["brier"]]
weights_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_protest_n = [x / sum(weights_n) for x in weights_n]
ensemble = (history_protest_df.preds_proba*weights_protest_n[0])+(demog_protest_df.preds_proba*weights_protest_n[1])+(geog_protest_df.preds_proba*weights_protest_n[2])+(econ_protest_df.preds_proba*weights_protest_n[3])+(pol_protest_df.preds_proba*weights_protest_n[4])
ensemble_protest=pd.concat([history_protest_df[["country","dd","d_protest","test"]],pd.DataFrame(ensemble)],axis=1)
ensemble_protest.columns=["country","dd","d_protest","test","preds_proba"]
ensemble_protest=ensemble_protest.reset_index(drop=True)

                                    #############
                                    ### Riots ###
                                    #############

#####################
### History theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Transforms
x["pop_refugee"]=np.log(x["pop_refugee"]+1)

# Categorical
target="d_riot"
inputs=['d_riot_lag1',"d_riot_zeros_decay","d_neighbors_riot_event_counts_lag1",'regime_duration', 'pop_refugee', 'pop_refugee_id']
history_riot_df,history_riot_evals=gen_model(y,x,target,inputs,name="history_riot",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
history_riot_df.to_csv("out/history_riot_df_cm.csv")
history_riot_evals_df = pd.DataFrame.from_dict(history_riot_evals, orient='index').reset_index()

########################
### Demography theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Categorical
target="d_riot"
inputs=demog_theme
demog_riot_df,demog_riot_evals=gen_model(y,x,target,inputs,name="demog_riot",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
demog_riot_df.to_csv("out/demog_riot_df_cm.csv")
demog_riot_evals_df = pd.DataFrame.from_dict(demog_riot_evals, orient='index').reset_index()

#####################################
### Geography & environment theme ###
#####################################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_geog_full_cm.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
preprocess_min_max_group(x,"temp","country")

# Categorical
target="d_riot"
inputs=geog_theme
geog_riot_df,geog_riot_evals=gen_model(y,x,target,inputs,name="geog_riot",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
geog_riot_df.to_csv("out/geog_riot_df_cm.csv")
geog_riot_evals_df = pd.DataFrame.from_dict(geog_riot_evals, orient='index').reset_index()

#####################
### Economy theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Transforms
x["oil_deposits"]=np.log(x["oil_deposits"]+1)
x["oil_production"]=np.log(x["oil_production"]+1)
x["oil_exports"]=np.log(x["oil_exports"]+1)
x["gdp"]=np.log(x["gdp"]+1)
preprocess_min_max_group(x,"gdp_growth","country")
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)
preprocess_min_max_group(x,"foodprod","country")
preprocess_min_max_group(x,"pop_growth","country")
preprocess_min_max_group(x,"mig","country")

# Categorical
target="d_riot"
inputs=econ_theme
econ_riot_df,econ_riot_evals=gen_model(y,x,target,inputs,name="econ_riot",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
econ_riot_df.to_csv("out/econ_riot_df_cm.csv")
econ_riot_evals_df = pd.DataFrame.from_dict(econ_riot_evals, orient='index').reset_index()

###############################
### Regime and policy theme ###
###############################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_pol_full_cm.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["dd","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["dd","gw_codes"],how="left")

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)

# Categorical
target="d_riot"
inputs=pol_theme
pol_riot_df,pol_riot_evals=gen_model(y,x,target,inputs,name="pol_riot",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
pol_riot_df.to_csv("out/pol_riot_df_cm.csv")
pol_riot_evals_df = pd.DataFrame.from_dict(pol_riot_evals, orient='index').reset_index()

################
### Ensemble ###
################

predictions=pd.concat([history_riot_df.preds_proba,demog_riot_df.preds_proba,geog_riot_df.preds_proba,econ_riot_df.preds_proba,pol_riot_df.preds_proba], axis=1)
weights=[1-history_riot_evals["brier"],1-demog_riot_evals["brier"],1-geog_riot_evals["brier"],1-econ_riot_evals["brier"],1-pol_riot_evals["brier"]]
weights_riot_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_riot_n = [x / sum(weights_riot_n) for x in weights_riot_n]
ensemble = (history_riot_df.preds_proba*weights_riot_n[0])+(demog_riot_df.preds_proba*weights_riot_n[1])+(geog_riot_df.preds_proba*weights_riot_n[2])+(econ_riot_df.preds_proba*weights_riot_n[3])+(pol_riot_df.preds_proba*weights_riot_n[4])
ensemble_riot=pd.concat([history_riot_df[["country","dd","d_riot","test"]],pd.DataFrame(ensemble)],axis=1)
ensemble_riot.columns=["country","dd","d_riot","test","preds_proba"]
ensemble_riot=ensemble_riot.reset_index(drop=True)

                                #################
                                ### Terrorism ###
                                #################

#####################
### History theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)
y=y.loc[y["year"]<=2020]
x=x.loc[x["year"]<=2020]

# Categorical
target="d_terror"
inputs=['d_terror_lag1',"d_terror_zeros_decay","d_neighbors_terror_fatalities_lag1",'regime_duration','pop_refugee','pop_refugee_id']
history_terror_df,history_terror_evals=gen_model(y,x,target,inputs,name="history_terror",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd",last_train=2017)
history_terror_df.to_csv("out/history_terror_df_cm.csv")
history_terror_evals_df = pd.DataFrame.from_dict(history_terror_evals, orient='index').reset_index()

########################
### Demography theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)
y=y.loc[y["year"]<=2020]
x=x.loc[x["year"]<=2020]

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Categorical
target="d_terror"
inputs=demog_theme
demog_terror_df,demog_terror_evals=gen_model(y,x,target,inputs,name="demog_terror",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd",last_train=2017)
demog_terror_df.to_csv("out/demog_terror_df_cm.csv")
demog_terror_evals_df = pd.DataFrame.from_dict(demog_terror_evals, orient='index').reset_index()

#####################################
### Geography & environment theme ###
#####################################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_geog_full_cm.csv",index_col=0)
y=y.loc[y["year"]<=2020]
x=x.loc[x["year"]<=2020]

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
preprocess_min_max_group(x,"temp","country")

# Categorical
target="d_terror"
inputs=geog_theme
geog_terror_df,geog_terror_evals=gen_model(y,x,target,inputs,name="geog_terror",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd",last_train=2017)
geog_terror_df.to_csv("out/geog_terror_df_cm.csv")
geog_terror_evals_df = pd.DataFrame.from_dict(geog_terror_evals, orient='index').reset_index()

#####################
### Economy theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)
y=y.loc[y["year"]<=2020]
x=x.loc[x["year"]<=2020]

# Transforms
x["oil_deposits"]=np.log(x["oil_deposits"]+1)
x["oil_production"]=np.log(x["oil_production"]+1)
x["oil_exports"]=np.log(x["oil_exports"]+1)
x["gdp"]=np.log(x["gdp"]+1)
preprocess_min_max_group(x,"gdp_growth","country")
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)
preprocess_min_max_group(x,"foodprod","country")
preprocess_min_max_group(x,"pop_growth","country")
preprocess_min_max_group(x,"mig","country")

# Categorical
target="d_terror"
inputs=econ_theme
econ_terror_df,econ_terror_evals=gen_model(y,x,target,inputs,name="econ_terror",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd",last_train=2017)
econ_terror_df.to_csv("out/econ_terror_df_cm.csv")
econ_terror_evals_df = pd.DataFrame.from_dict(econ_terror_evals, orient='index').reset_index()

###############################
### Regime and policy theme ###
###############################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_pol_full_cm.csv",index_col=0)
y=y.loc[y["year"]<=2020]
x=x.loc[x["year"]<=2020]

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)

# Categorical
target="d_terror"
inputs=pol_theme
pol_terror_df,pol_terror_evals=gen_model(y,x,target,inputs,name="pol_terror",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd",last_train=2017)
pol_terror_df.to_csv("out/pol_terror_df_cm.csv")
pol_terror_evals_df = pd.DataFrame.from_dict(pol_terror_evals, orient='index').reset_index()

################
### Ensemble ###
################

predictions=pd.concat([history_terror_df.preds_proba,demog_terror_df.preds_proba,geog_terror_df.preds_proba,econ_terror_df.preds_proba,pol_terror_df.preds_proba], axis=1)
weights=[1-history_terror_evals["brier"],1-demog_terror_evals["brier"],1-geog_terror_evals["brier"],1-econ_terror_evals["brier"],1-pol_terror_evals["brier"]]
weights_terror_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_terror_n = [x / sum(weights_terror_n) for x in weights_terror_n]
ensemble = (history_terror_df.preds_proba*weights_terror_n[0])+(demog_terror_df.preds_proba*weights_terror_n[1])+(geog_terror_df.preds_proba*weights_terror_n[2])+(econ_terror_df.preds_proba*weights_terror_n[3])+(pol_terror_df.preds_proba*weights_terror_n[4])
ensemble_terror=pd.concat([history_terror_df[["country","dd","d_terror","test"]],pd.DataFrame(ensemble)],axis=1)
ensemble_terror.columns=["country","dd","d_terror","test","preds_proba"]
ensemble_terror=ensemble_terror.reset_index(drop=True)

                            ###################
                            ### State-based ###
                            ###################


#####################
### History theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)

# Transforms
x["pop_refugee"]=np.log(x["pop_refugee"]+1)

# Categorical
target="d_sb"
inputs=['d_sb_lag1',"d_sb_zeros_decay","d_neighbors_sb_fatalities_lag1",'regime_duration', 'pop_refugee', 'pop_refugee_id']
history_sb_df,history_sb_evals=gen_model(y,x,target,inputs,name="history_sb",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
history_sb_df.to_csv("out/history_sb_df_cm.csv")
history_sb_evals_df = pd.DataFrame.from_dict(history_sb_evals, orient='index').reset_index()

########################
### Demography theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Categorical
target="d_sb"
inputs=demog_theme
demog_sb_df,demog_sb_evals=gen_model(y,x,target,inputs,name="demog_sb",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
demog_sb_df.to_csv("out/demog_sb_df_cm.csv")
demog_sb_evals_df = pd.DataFrame.from_dict(demog_sb_evals, orient='index').reset_index()

#####################################
### Geography & environment theme ###
#####################################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_geog_full_cm.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
preprocess_min_max_group(x,"temp","country")

# Categorical
target="d_sb"
inputs=geog_theme
geog_sb_df,geog_sb_evals=gen_model(y,x,target,inputs,name="geog_sb",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
geog_sb_df.to_csv("out/geog_sb_df_cm.csv")
geog_sb_evals_df = pd.DataFrame.from_dict(geog_sb_evals, orient='index').reset_index()

#####################
### Economy theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)

# Transforms
x["oil_deposits"]=np.log(x["oil_deposits"]+1)
x["oil_production"]=np.log(x["oil_production"]+1)
x["oil_exports"]=np.log(x["oil_exports"]+1)
x["gdp"]=np.log(x["gdp"]+1)
preprocess_min_max_group(x,"gdp_growth","country")
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)
preprocess_min_max_group(x,"foodprod","country")
preprocess_min_max_group(x,"pop_growth","country")
preprocess_min_max_group(x,"mig","country")

# Categorical
target="d_sb"
inputs=econ_theme
econ_sb_df,econ_sb_evals=gen_model(y,x,target,inputs,name="econ_sb",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
econ_sb_df.to_csv("out/econ_sb_df_cm.csv")
econ_sb_evals_df = pd.DataFrame.from_dict(econ_sb_evals, orient='index').reset_index()

###############################
### Regime and policy theme ###
###############################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_pol_full_cm.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)

# Categorical
target="d_sb"
inputs=pol_theme
pol_sb_df,pol_sb_evals=gen_model(y,x,target,inputs,name="pol_sb",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
pol_sb_df.to_csv("out/pol_sb_df_cm.csv")
pol_sb_evals_df = pd.DataFrame.from_dict(pol_sb_evals, orient='index').reset_index()

################
### Ensemble ###
################

predictions=pd.concat([history_sb_df.preds_proba,demog_sb_df.preds_proba,geog_sb_df.preds_proba,econ_sb_df.preds_proba,pol_sb_df.preds_proba], axis=1)
weights=[1-history_sb_evals["brier"],1-demog_sb_evals["brier"],1-geog_sb_evals["brier"],1-econ_sb_evals["brier"],1-pol_sb_evals["brier"]]
weights_sb_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_sb_n = [x / sum(weights_sb_n) for x in weights_sb_n]
ensemble = (history_sb_df.preds_proba*weights_sb_n[0])+(demog_sb_df.preds_proba*weights_sb_n[1])+(geog_sb_df.preds_proba*weights_sb_n[2])+(econ_sb_df.preds_proba*weights_sb_n[3])+(pol_sb_df.preds_proba*weights_sb_n[4])
ensemble_sb=pd.concat([history_sb_df[["country","dd","d_sb","test"]],pd.DataFrame(ensemble)],axis=1)
ensemble_sb.columns=["country","dd","d_sb","test","preds_proba"]
ensemble_sb=ensemble_sb.reset_index(drop=True)

                            ##########################
                            ### One-sided violence ###
                            ##########################

#####################
### History theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)

# Transforms
x["pop_refugee"]=np.log(x["pop_refugee"]+1)

# Categorical
target="d_osv"
inputs=['d_osv_lag1',"d_osv_zeros_decay","d_neighbors_osv_fatalities_lag1",'regime_duration','pop_refugee','pop_refugee_id']
history_osv_df,history_osv_evals=gen_model(y,x,target,inputs,name="history_osv",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
history_osv_df.to_csv("out/history_osv_df_cm.csv")
history_osv_evals_df = pd.DataFrame.from_dict(history_osv_evals, orient='index').reset_index()

########################
### Demography theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Categorical
target="d_osv"
inputs=demog_theme
demog_osv_df,demog_osv_evals=gen_model(y,x,target,inputs,name="demog_osv",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
demog_osv_df.to_csv("out/demog_osv_df_cm.csv")
demog_osv_evals_df = pd.DataFrame.from_dict(demog_osv_evals, orient='index').reset_index()

#####################################
### Geography & environment theme ###
#####################################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_geog_full_cm.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
preprocess_min_max_group(x,"temp","country")

# Categorical
target="d_osv"
inputs=geog_theme
geog_osv_df,geog_osv_evals=gen_model(y,x,target,inputs,name="geog_osv",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
geog_osv_df.to_csv("out/geog_osv_df_cm.csv")
geog_osv_evals_df = pd.DataFrame.from_dict(geog_osv_evals, orient='index').reset_index()

#####################
### Economy theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)

# Transforms
x["oil_deposits"]=np.log(x["oil_deposits"]+1)
x["oil_production"]=np.log(x["oil_production"]+1)
x["oil_exports"]=np.log(x["oil_exports"]+1)
x["gdp"]=np.log(x["gdp"]+1)
preprocess_min_max_group(x,"gdp_growth","country")
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)
preprocess_min_max_group(x,"foodprod","country")
preprocess_min_max_group(x,"pop_growth","country")
preprocess_min_max_group(x,"mig","country")

# Categorical
target="d_osv"
inputs=econ_theme
econ_osv_df,econ_osv_evals=gen_model(y,x,target,inputs,name="econ_osv",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
econ_osv_df.to_csv("out/econ_osv_df_cm.csv")
econ_osv_evals_df = pd.DataFrame.from_dict(econ_osv_evals, orient='index').reset_index()

###############################
### Regime and policy theme ###
###############################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_pol_full_cm.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)

# Categorical
target="d_osv"
inputs=pol_theme
pol_osv_df,pol_osv_evals=gen_model(y,x,target,inputs,name="pol_osv",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
pol_osv_df.to_csv("out/pol_osv_df_cm.csv")
pol_osv_evals_df = pd.DataFrame.from_dict(pol_osv_evals, orient='index').reset_index()

################
### Ensemble ###
################

predictions=pd.concat([history_osv_df.preds_proba,demog_osv_df.preds_proba,geog_osv_df.preds_proba,econ_osv_df.preds_proba,pol_osv_df.preds_proba], axis=1)
weights=[1-history_osv_evals["brier"],1-demog_osv_evals["brier"],1-geog_osv_evals["brier"],1-econ_osv_evals["brier"],1-pol_osv_evals["brier"]]
weights_osv_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_osv_n = [x / sum(weights_osv_n) for x in weights_osv_n]
ensemble = (history_osv_df.preds_proba*weights_osv_n[0])+(demog_osv_df.preds_proba*weights_osv_n[1])+(geog_osv_df.preds_proba*weights_osv_n[2])+(econ_osv_df.preds_proba*weights_osv_n[3])+(pol_osv_df.preds_proba*weights_osv_n[4])
ensemble_osv=pd.concat([history_osv_df[["country","dd","d_osv","test"]],pd.DataFrame(ensemble)],axis=1)
ensemble_osv.columns=["country","dd","d_osv","test","preds_proba"]
ensemble_osv=ensemble_osv.reset_index(drop=True)

                            #######################
                            ### Non-state based ###
                            #######################

#####################
### History theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full_cm.csv",index_col=0)

# Transforms
x["pop_refugee"]=np.log(x["pop_refugee"]+1)

# Categorical
target="d_ns"
inputs=['d_ns_lag1',"d_ns_zeros_decay","d_neighbors_ns_fatalities_lag1",'regime_duration', 'pop_refugee', 'pop_refugee_id']
history_ns_df,history_ns_evals=gen_model(y,x,target,inputs,name="history_ns",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
history_ns_df.to_csv("out/history_ns_df_cm.csv")
history_ns_evals_df = pd.DataFrame.from_dict(history_ns_evals, orient='index').reset_index()

########################
### Demography theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_demog_full_cm.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Categorical
target="d_ns"
inputs=demog_theme
demog_ns_df,demog_ns_evals=gen_model(y,x,target,inputs,name="demog_ns",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
demog_ns_df.to_csv("out/demog_ns_df_cm.csv")
demog_ns_evals_df = pd.DataFrame.from_dict(demog_ns_evals, orient='index').reset_index()

#####################################
### Geography & environment theme ###
#####################################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_geog_full_cm.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
preprocess_min_max_group(x,"temp","country")

# Categorical
target="d_ns"
inputs=geog_theme
geog_ns_df,geog_ns_evals=gen_model(y,x,target,inputs,name="geog_ns",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
geog_ns_df.to_csv("out/geog_ns_df_cm.csv")
geog_ns_evals_df = pd.DataFrame.from_dict(geog_ns_evals, orient='index').reset_index()

#####################
### Economy theme ###
#####################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_econ_full_cm.csv",index_col=0)

# Transforms
x["oil_deposits"]=np.log(x["oil_deposits"]+1)
x["oil_production"]=np.log(x["oil_production"]+1)
x["oil_exports"]=np.log(x["oil_exports"]+1)
x["gdp"]=np.log(x["gdp"]+1)
preprocess_min_max_group(x,"gdp_growth","country")
x["inflat"]=np.sign(x["inflat"]) * np.log(np.abs(x["inflat"]) + 1)
x["conprice"]=np.log(x["conprice"]+1)
preprocess_min_max_group(x,"foodprod","country")
preprocess_min_max_group(x,"pop_growth","country")
preprocess_min_max_group(x,"mig","country")

# Categorical
target="d_ns"
inputs=econ_theme
econ_ns_df,econ_ns_evals=gen_model(y,x,target,inputs,name="econ_ns",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
econ_ns_df.to_csv("out/econ_ns_df_cm.csv")
econ_ns_evals_df = pd.DataFrame.from_dict(econ_ns_evals, orient='index').reset_index()

###############################
### Regime and policy theme ###
###############################

# Load dataset
y=pd.read_csv("out/df_out_full_cm.csv",index_col=0)
x=pd.read_csv("out/df_pol_full_cm.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)

# Categorical
target="d_ns"
inputs=pol_theme
pol_ns_df,pol_ns_evals=gen_model(y,x,target,inputs,name="pol_ns",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",time="dd")
pol_ns_df.to_csv("out/pol_ns_df_cm.csv")
pol_ns_evals_df = pd.DataFrame.from_dict(pol_ns_evals, orient='index').reset_index()

################
### Ensemble ###
################

predictions=pd.concat([history_ns_df.preds_proba,demog_ns_df.preds_proba,geog_ns_df.preds_proba,econ_ns_df.preds_proba,pol_ns_df.preds_proba], axis=1)
weights=[1-history_ns_evals["brier"],1-demog_ns_evals["brier"],1-geog_ns_evals["brier"],1-econ_ns_evals["brier"],1-pol_ns_evals["brier"]]
weights_ns_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_ns_n = [x / sum(weights_ns_n) for x in weights_ns_n]
ensemble = (history_ns_df.preds_proba*weights_ns_n[0])+(demog_ns_df.preds_proba*weights_ns_n[1])+(geog_ns_df.preds_proba*weights_ns_n[2])+(econ_ns_df.preds_proba*weights_ns_n[3])+(pol_ns_df.preds_proba*weights_ns_n[4])
ensemble_ns=pd.concat([history_ns_df[["country","dd","d_ns","test"]],pd.DataFrame(ensemble)],axis=1)
ensemble_ns.columns=["country","dd","d_ns","test","preds_proba"]
ensemble_ns=ensemble_ns.reset_index(drop=True)

                ################################
                ################################
                ################################
                ### Ensemble of the Ensemble ###
                ################################
                ################################
                ################################
                

#############################
### Ensemble of ensembles ###
#############################

### Get weights ###
weights=[]
weights.append(1-brier_score_loss(ensemble_protest["d_protest"].loc[(ensemble_protest["dd"]>="2020-01")&(ensemble_protest["dd"]<="2021-12")], ensemble_protest["preds_proba"].loc[(ensemble_protest["dd"]>="2020-01")&(ensemble_protest["dd"]<="2021-12")]))
weights.append(1-brier_score_loss(ensemble_riot["d_riot"].loc[(ensemble_riot["dd"]>="2020-01")&(ensemble_riot["dd"]<="2021-12")],ensemble_riot["preds_proba"].loc[(ensemble_riot["dd"]>="2020-01")&(ensemble_riot["dd"]<="2021-12")]))
weights.append(1-brier_score_loss(ensemble_terror["d_terror"].loc[(ensemble_terror["dd"]>="2017-01")&(ensemble_terror["dd"]<="2018-12")],ensemble_terror["preds_proba"].loc[(ensemble_terror["dd"]>="2017-01")&(ensemble_terror["dd"]<="2018-12")]))
weights.append(1-brier_score_loss(ensemble_sb["d_sb"].loc[(ensemble_sb["dd"]>="2020-01")&(ensemble_sb["dd"]<="2021-12")], ensemble_sb["preds_proba"].loc[(ensemble_sb["dd"]>="2020-01")&(ensemble_sb["dd"]<="2021-12")]))
weights.append(1-brier_score_loss(ensemble_ns["d_ns"].loc[(ensemble_ns["dd"]>="2020-01")&(ensemble_ns["dd"]<="2021-12")], ensemble_ns["preds_proba"].loc[(ensemble_ns["dd"]>="2020-01")&(ensemble_ns["dd"]<="2021-12")]))
weights.append(1-brier_score_loss(ensemble_osv["d_osv"].loc[(ensemble_osv["dd"]>="2020-01")&(ensemble_osv["dd"]<="2021-12")],ensemble_osv["preds_proba"].loc[(ensemble_osv["dd"]>="2020-01")&(ensemble_osv["dd"]<="2021-12")]))
weights_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_n = [x / sum(weights_n) for x in weights_n]

### Protest, riotes, terrorism, sb, ns, osv ###

base=ensemble_protest[["dd","country"]]
ensemble_sb_short=pd.merge(base, ensemble_sb,on=["dd","country"],how="left")

base=ensemble_protest[["dd","country"]]
ensemble_ns_short=pd.merge(base, ensemble_ns,on=["dd","country"],how="left")

base=ensemble_protest[["dd","country"]]
ensemble_osv_short=pd.merge(base, ensemble_osv,on=["dd","country"],how="left")

base=ensemble_protest[["dd","country"]]
ensemble_terror_short=pd.merge(base, ensemble_terror,on=["dd","country"],how="left")
ensemble_terror_short=ensemble_terror_short.fillna(0)

ensemble = (ensemble_protest.preds_proba*weights_n[0])+(ensemble_riot.preds_proba*weights_n[1])+(ensemble_terror_short.preds_proba*weights_n[2])+(ensemble_sb_short.preds_proba*weights_n[3])+(ensemble_ns_short.preds_proba*weights_n[4])+(ensemble_osv_short.preds_proba*weights_n[5])
ensemble_ens=pd.concat([ensemble_protest[["country","dd"]],pd.DataFrame(ensemble)],axis=1)
ensemble_ens.columns=["country","dd","preds_proba"]

### terrorism, sb, ns, osv  ###
base=ensemble_protest[["dd","country"]]
base["drop"] = base['dd'].astype(str) + '-' + base['country']
drop=list(base["drop"])

ensemble_sb["id"] = ensemble_sb['dd'].astype(str) + '-' + ensemble_sb['country']
ensemble_sb_s = ensemble_sb[~ensemble_sb['id'].isin(drop)]
ensemble_sb_s=ensemble_sb_s.reset_index(drop=True)

ensemble_ns["id"] = ensemble_ns['dd'].astype(str) + '-' + ensemble_ns['country']
ensemble_ns_s = ensemble_ns[~ensemble_ns['id'].isin(drop)]
ensemble_ns_s=ensemble_ns_s.reset_index(drop=True)

ensemble_osv["id"] = ensemble_osv['dd'].astype(str) + '-' + ensemble_osv['country']
ensemble_osv_s = ensemble_osv[~ensemble_osv['id'].isin(drop)]
ensemble_osv_s=ensemble_osv_s.reset_index(drop=True)

ensemble_terror["id"] = ensemble_terror['dd'].astype(str) + '-' + ensemble_terror['country']
ensemble_terror_s = ensemble_terror[~ensemble_terror['id'].isin(drop)]
ensemble_terror_s=ensemble_terror_s.reset_index(drop=True)

ensemble = (ensemble_sb_s.preds_proba*weights_n[3])+(ensemble_ns_s.preds_proba*weights_n[4])+(ensemble_osv_s.preds_proba*weights_n[5])+(ensemble_terror_s.preds_proba*weights_n[2])
ensemble_ens_vio=pd.concat([ensemble_sb_s[["country","dd"]],pd.DataFrame(ensemble)],axis=1)
ensemble_ens_vio.columns=["country","dd","preds_proba"]

ensemble_ens=pd.concat([ensemble_ens,ensemble_ens_vio],axis=0)
ensemble_ens=ensemble_ens.sort_values(by=["country","dd"])


#######################
### Prediction maps ###
#######################
    
ensemble_ens["dummy"]=0
ensemble_ens.loc[ensemble_ens["preds_proba"]>=0.5,"dummy"]=1
#ensemble_ens.to_csv("out/ensemble_ens_df_cm.csv")

def get_centroid(geom):
    if geom.geom_type == 'MultiPolygon':
        # Choose the largest polygon by area
        largest_polygon = max(geom, key=lambda p: p.area)
        return largest_polygon.centroid
    else:
        return geom.centroid
    
for year in list(ensemble_ens.dd.unique()):
    worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    worldmap['centroid'] =  worldmap.geometry.apply(get_centroid)
    worldmap=worldmap.loc[(worldmap["continent"]!="Antarctica")]
    worldmap.loc[worldmap["name"]=="Bosnia and Herz.","name"]='Bosnia-Herzegovina'
    worldmap.loc[worldmap["name"]=="Cambodia","name"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["name"]=="Central African Rep.","name"]='Central African Republic'
    worldmap.loc[worldmap["name"]=="Dem. Rep. Congo","name"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["name"]=="Côte d'Ivoire","name"]='Ivory Coast'
    worldmap.loc[worldmap["name"]=="Dominican Rep.","name"]='Dominican Republic'
    worldmap.loc[worldmap["name"]=='Timor-Leste',"name"]='East Timor'
    worldmap.loc[worldmap["name"]=='Eq. Guinea',"name"]='Equatorial Guinea'
    worldmap.loc[worldmap["name"]=='Macedonia',"name"]='Macedonia, FYR'
    worldmap.loc[worldmap["name"]=='Myanmar',"name"]='Myanmar (Burma)'
    worldmap.loc[worldmap["name"]=='Myanmar',"name"]='Myanmar (Burma)'
    worldmap.loc[worldmap["name"]=='Russia',"name"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["name"]=='S. Sudan',"name"]='South Sudan'
    worldmap.loc[worldmap["name"]=='Solomon Is.',"name"]='Solomon Islands'
    worldmap.loc[worldmap["name"]=='Yemen',"name"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["name"]=='Somaliland',"name"]='Somalia'
    worldmap.loc[worldmap["name"]=='Palestine',"name"]='Israel'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["name","geometry",'centroid']].merge(ensemble_ens[["country","preds_proba","dummy"]].loc[ensemble_ens["dd"]==year], right_on=["country"],left_on=["name"],how='left')
    worldmap_m.plot(column="preds_proba",ax=ax,cmap="Greens",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    worldmap_m_s=worldmap_m.loc[worldmap_m["dummy"]==1]
    ax.scatter(worldmap_m_s['centroid'].x, worldmap_m_s['centroid'].y, color='black', marker='o', s=20)
    plt.title(f"Predicted probability of collective action in {year}", size=25)
    cmap = plt.cm.get_cmap('Greens')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.01)   
    fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/procedural/out/struc_map_preds_ens_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_map_preds_ens_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"out/struc_map_preds_ens_{year}.jpeg",dpi=400,bbox_inches="tight")



ensemble_ens.to_csv("out/ensemble_ens_df_cm.csv")





       