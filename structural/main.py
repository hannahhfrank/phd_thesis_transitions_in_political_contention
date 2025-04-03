import pandas as pd
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.dummy import DummyClassifier,DummyRegressor
from sklearn.linear_model import LinearRegression
import numpy as np 
from functions import get_wb,multivariate_imp_bayes,preprocess_min_max_group,earth_mover_distance,gen_model
from sklearn.model_selection import GridSearchCV,PredefinedSplit
from sklearn.inspection import PartialDependenceDisplay,partial_dependence
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc,brier_score_loss,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, mean_squared_error
import geopandas as gpd
import matplotlib as mpl
from matplotlib import cm 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
import seaborn as sns

import os
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'

# Simple random forest
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)],
               #'max_features': ["sqrt", "log2", None],
               #'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],
               #'min_samples_split': [2,5,10],
               #'min_samples_leaf': [1,2,4],
               }

names={'d_protest_lag1':"t-1 lag of at least one protest event",
       "d_protest_zeros_decay":"Time since last protest event",
       "d_neighbors_proteset_event_counts_lag1":"One neighbour had one protest event in previous year (t-1 lag)",
       'd_riot_lag1':"t-1 lag of at least one riot event",
       "d_riot_zeros_decay":"Time since last riot event",
       "d_neighbors_riot_event_counts_lag1":"One neighbour had one riot event in previous year (t-1 lag)",
       'd_remote_lag1':"t-1 lag of at least one fatality in remote event",
       "d_remote_zeros_decay":"Time since last remote fatalities",
       "d_neighbors_remote_fatalities_lag1":"One neighbour had one fatality from remote in previous year (t-1 lag)",
       'd_sb_lag1':"t-1 lag of at least one fatality in state-based violence",
       "d_sb_zeros_decay":"Time since last state-based violence fatalities",
       "d_neighbors_sb_fatalities_lag1":"One neighbour had one fatality from state-based violence in previous year (t-1 lag)",
       'd_ns_lag1':"t-1 lag of at least one fatality in non-state violence",
       "d_ns_zeros_decay":"Time since last non-state violence fatalities in years",
       "d_neighbors_ns_fatalities_lag1":"One neighbour had one fatality from non-state violence in previous year (t-1 lag)",
       'd_osv_lag1':"t-1 lag of at least one fatality in one-sided violence",
       "d_osv_zeros_decay":"Time since last one-sided violence fatalities",
       "d_neighbors_osv_fatalities_lag1":"One neighbour had one fatality from one-sided violence in previous year (t-1 lag)", 
       "regime_duration":"Years since the country became independent",
       'pop_refugee':"Refugee population",
       'pop_refugee_id':"Refugee population (nan imp)",
       "pop":"Population size",
       "pop_density":"Population desnity",
       "pop_density_id":"Population desnity (nan imp)",
       "urb_share":"Urbanization, share",
       "rural_share":"Urbanization, share (nan imp)",
       "pop_male_share":"Male population, proportion",
       "pop_male_share_0_14":"Male population 0-14 years, proportion",
       "pop_male_share_15_19":"Male population 15-19 years, proportion",
       "pop_male_share_20_24":"Male population 20-24 years, proportion",
       "pop_male_share_25_29":"Male population 25-29 years, proportion",
       "pop_male_share_30_34":"Male population 30-34 years, proportion",
       "group_counts":"Number of politically relevant ethnic group",
       "group_counts_id":"Number of ethnic group (nan imp)",
       "monopoly_share":"Population with monopoly status",
       "monopoly_share_id":"Population with monopoly status (nan imp)",
       "discriminated_share":"Population with discriminated status",
       "discriminated_share_id":"Population with discriminated status (nan imp)",
       "powerless_share":"Population with powerless status",
       "powerless_share_id":"Population with powerless status (nan imp)",
       "dominant_share":"Population with dominant status",
       "dominant_share_id":"Population with dominant status (nan imp)",
       "ethnic_frac":"ELF, considering ethnic group",
       "ethnic_frac_id":"ELF, considering ethnic group (nan imp)",
       "rel_frac":"ELF, considering religious group",
       "rel_frac_id":"ELF, considering religious group (nan imp)",
       "lang_frac":"ELF, considering language group",
       "lang_frac_id":"ELF, considering language group (nan imp)",
       "race_frac":"ELF, considering race group",
       "race_frac_id":"ELF, considering race group (nan imp)",
       'land':"Land area (sq. km)",
       'temp_norm':"Average mean surface air temperature",
       'temp_id':"Average mean surface air temperature (nan imp)",
       'forest':"Forest area",
       'forest_id':"Forest area (nan imp)",
       'co2':"CO2 emissions (kt)",
       'co2_id':"CO2 emissions (kt) (nan imp)",
       'percip':"Average precipitation in depth (mm per year)",
       'percip_id':"Average precipitation in depth (mm per year) (nan imp)",
       'waterstress':"Level of water stress",
       'waterstress_id':"Level of water stress (nan imp)",
       'agri_land':"Agricultural land",
       'agri_land_id':"Agricultural land (nan imp)",
       'arable_land':"Arable land",
       'arable_land_id':"Arable land (nan imp)",
       'rugged':"Terrain ruggedness index",
       'soil':"Fertile soil",
       'desert':"Desert",
       'tropical':"Tropical climate",
       'cont_africa':"Country in Africa",
       'cont_asia':"Country in Asia",
       'd_neighbors_con':"One neighbour has one fatality in state-based violence",
       'no_neigh':"Country has no direct neighbours",
       'd_neighbors_non_dem':"At least one neighbour is not democratic",
       'libdem_id_neigh':"At least one neighbour is not democratic (nan imp)",
       'oil_deposits':"Crude oil reserves",
       'oil_deposits_id':"Crude oil reserves (nan imp)",
       'oil_production':"Total petroleum and other liquids production",
       'oil_production_id':"Total petroleum and other liquids production (nan imp)",
       'oil_exports':"Total petroleum and other liquids exports",
       'oil_exports_id':"Total petroleum and other liquids exports (nan imp)",
       'natres_share':"Total natural resources rents",
       'natres_share_id':"Total natural resources rents (nan imp)",
       'oil_share':"Oil rents", 
       'oil_share_id':"Oil rents (nan imp)", 
       'gas_share':"Natural gas rents ", 
       'gas_share_id':"Natural gas rents (nan imp)", 
       'coal_share':"Coal rents",
       'coal_share_id':"Coal rents (nan imp)",
       'forest_share':"Forest rent", 
       'forest_share_id':"Forest rent (nan imp)", 
       'minerals_share':"Mineral rents",
       'minerals_share_id':"Mineral rents (nan imp)",
       'gdp':"GDP per capita (current US)",
       'gdp_id':"GDP per capita (current US) (nan imp)",
       'gni':"GNI per capita (current US)",
       'gni_id':"GNI per capita (current US) (nan imp)",
       'gdp_growth_norm':"GDP growth",
       'gdp_growth_id':"GDP growth (i.d.)",
       'unemploy':"Unemployment, total",
       'unemploy_id':"Unemployment, total (nan imp)",
       'unemploy_male':"Unemployment, male",
       'unemploy_male_id':"Unemployment, male (nan imp)",
       'inflat':"Inflation, consumer prices",
       'inflat_id':"Inflation, consumer prices (nan imp)",
       'conprice':"Consumer price index",
       'conprice_id':"Consumer price index (nan imp)",
       'undernour':"Prevalence of undernourishment",
       'undernour_id':"Prevalence of undernourishment (nan imp)",
       'foodprod_norm':"Food production index",
       'foodprod_id':"Food production index (nan imp)",
       'water_rural':"People using at least basic drinking water services, rural",
       'water_rural_id':"People using at least basic drinking water services, rural (nan imp)",
       'water_urb':"People using at least basic drinking water services, urban",
       'water_urb_id':"People using at least basic drinking water services, urban (nan imp)",
       'agri_share':"Agriculture percentage of GDP",
       'agri_share_id':"Agriculture percentage of GDP (nan imp)",
       'trade_share':"Trade percentage of GDP",
       'trade_share_id':"Trade percentage of GDP (nan imp)",
       'fert':"Fertility rate, total (births per woman)",
       'fert_id':"Fertility rate, total (births per woman) (nan imp)",
       'lifeexp_female':"Life expectancy at birth, female (years)",
       'lifeexp_female_id':"Life expectancy at birth, female (years) (nan imp)",
       'lifeexp_male':"Life expectancy at birth, male (years)",
       'lifeexp_male_id':"Life expectancy at birth, male (years) (nan imp)",
       'pop_growth_norm':"Population growth",
       'pop_growth_id':"Population growth (nan imp)",
       'inf_mort':"Mortality rate, infant (per 1,000 live births)",
       'inf_mort_id':"Mortality rate, infant (per 1,000 live births) (nan imp)",
       'mig_norm':"Net migration",
       'exports':"Exports of goods and services",
       'exports_id':"Exports of goods and services (nan imp)",
       'imports':"Imports of goods and services",
       'imports_id':"Imports of goods and services (nan imp)",
       'primary_female':"School enrollment, primary, female",
       'primary_female_id':"School enrollment, primary, female (nan imp)",
       'primary_male':"School enrollment, primary, male",
       'primary_male_id':"School enrollment, primary, male (nan imp)",
       'second_female':"School enrollment, secondary, female",
       'second_female_id':"School enrollment, secondary, female (nan imp)",
       'second_male':"School enrollment, secondary, male",
       'second_male_id':"School enrollment, secondary, male (nan imp)",
       'tert_female':"School enrollment, tertiary, female",
       'tert_female_id':"School enrollment, tertiary, female (nan imp)",
       'tert_male':"School enrollment, tertiary, male",
       'tert_male_id':"School enrollment, tertiary, male (nan imp)",
       'broadband':"Fixed broadband subscriptions (per 100 people)",
       'broadband_id':"Fixed broadband subscriptions (per 100 people) (nan imp)",
       'telephone':"Fixed telephone subscriptions (per 100 people)",
       'telephone_id':"Fixed telephone subscriptions (per 100 people) (nan imp)",
       'internet_use':"Individuals using the Internet",
       'internet_use_id':"Individuals using the Internet (nan imp)",
       'mobile':"Mobile cellular subscriptions (per 100 people)",
       'mobile_id':"Mobile cellular subscriptions (per 100 people) (nan imp)",
       'eys':"Expected years of schooling",
       'eys_id':"Expected years of schooling (nan imp)",
       'eys_male':"Expected years of schooling, male",
       'eys_male_id':"Expected years of schooling, male (nan imp)",
       'eys_female':"Expected years of schooling, female",
       'eys_female_id':"Expected years of schooling, female (nan imp)",
       'mys':"Mean years of schooling",
       'mys_id':"Mean years of schooling (nan imp)",
       'mys_female':"Mean years of schooling, female",
       'mys_female_id':"Mean years of schooling, female (nan imp)",
       'mys_male':"Mean years of schooling, male",
       'mys_male_id':"Mean years of schooling, male (nan imp)",
       'armedforces_share':"Armed forces personnel",
       'armedforces_share_id':"Armed forces personnel (nan imp)",
       'milex_share':"Military expenditure",
       'milex_share_id':"Military expenditure (nan imp)",
       'corruption':"Control of corruption index",
       'corruption_id':"Control of corruption index (nan imp)",
       'effectiveness':"Government effectiveness index",
       'effectiveness_id':"Government effectiveness index (nan imp)",
       'polvio':"Political stability and absence of violence/terrorism index",
       'polvio_id':"Political stability and absence of violence/terrorism index (nan imp)",
       'regu':"Regulatory quality index",
       'regu_id':"Regulatory quality index (nan imp)",
       'law':"Rule of law index",
       'law_id':"Rule of law index (nan imp)",
       'account':"Voice and accountability index",
       'account_id':"Voice and accountability index (nan imp)",
       'tax':"Tax revenue",
       'tax_id':"Tax revenue (nan imp)",
       'polyarchy':"Electoral democracy index",
       'libdem':"Liberal democracy index",
       'libdem_id':"Liberal democracy index (nan imp)",
       'partipdem':"Participatory democracy index",
       'delibdem':"Deliberative democracy index",
       'egaldem':"Egalitarian democracy index",
       'civlib':"Civil liberties index",
       'phyvio':"Physical violence index",
       'pollib':"Political civil liberties index",
       'privlib':"Private civil liberties index",
       'execon':"Exclusion by socio-economic group index",
       'execon_id':"Exclusion by socio-economic group index (nan imp)",
       'exgender':"Exclusion by gender index",
       'exgender_id':"Exclusion by gender index (nan imp)",
       'exgeo':"Exclusion by urban-rural location index",
       'exgeo_id':"Exclusion by urban-rural location index (nan imp)",
       'expol':"Exclusion by political group index",
       'expol_id':"Exclusion by political group index (nan imp)",
       'exsoc':"Exclusion by social group index",
       'exsoc_id':"Exclusion by social group index (nan imp)",
       'shutdown':"Government Internet shut down in practice",
       'shutdown_id':"Government Internet shut down in practice (nan imp)",
       'filter':"Government Internet filtering in practice",
       'filter_id':"Government Internet filtering in practice (nan imp)",
       'tenure_months':"Number of months that leader has been in power",
       'tenure_months_id':"Number of months that leader has been in power (nan imp)",
       'dem_duration':"Logged number of months that a country is democratic",
       'dem_duration_id':"Logged number of months that a country is democratic (nan imp)",
       'elections':"Election for leadership taking place in that year",
       'elections_id':"Election for leadership taking place in that year (nan imp)",
       'lastelection':"Time since the last election for leadership",
       'lastelection_id':"Time since the last election for leadership (nan imp)"
       }


# Inputs
demog_theme=['pop','pop_density','pop_density_id','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','group_counts_id','monopoly_share','monopoly_share_id','discriminated_share','discriminated_share_id','powerless_share','powerless_share_id','dominant_share','dominant_share_id','ethnic_frac','ethnic_frac_id','rel_frac','rel_frac_id','lang_frac','lang_frac_id','race_frac','race_frac_id']
geog_theme=['land','forest','forest_id','temp_norm','temp_id','co2','co2_id','percip','percip_id','waterstress','waterstress_id','agri_land','agri_land_id','arable_land','arable_land_id','rugged','soil','desert','tropical','cont_africa','cont_asia','d_neighbors_con','no_neigh','d_neighbors_non_dem','libdem_id_neigh']
econ_theme=['oil_deposits','oil_deposits_id','oil_production','oil_production_id','oil_exports','oil_exports_id','natres_share','natres_share_id','oil_share','oil_share_id','gas_share','gas_share_id','coal_share','coal_share_id','forest_share','forest_share_id','minerals_share','minerals_share_id','gdp','gdp_id','gni','gni_id','gdp_growth_norm','gdp_growth_id','unemploy','unemploy_id','unemploy_male','unemploy_male_id','inflat','inflat_id','conprice','conprice_id','undernour','undernour_id','foodprod_norm','foodprod_id','water_rural','water_rural_id','water_urb','water_urb_id','agri_share','agri_share_id','trade_share','trade_share_id','fert','fert_id','lifeexp_female','lifeexp_female_id','lifeexp_male','lifeexp_male_id','pop_growth_norm','pop_growth_id','inf_mort','inf_mort_id','mig_norm','exports','exports_id','imports','imports_id','primary_female','primary_female_id','primary_male','primary_male_id','second_female','second_female_id','second_male','second_male_id','tert_female','tert_female_id','tert_male','tert_male_id','eys','eys_id','eys_male','eys_male_id','eys_female','eys_female_id','mys','mys_id','mys_male','mys_male_id','mys_female','mys_female_id']
pol_theme=['armedforces_share','armedforces_share_id','milex_share','milex_share_id','corruption','corruption_id', 'effectiveness', 'effectiveness_id', 'polvio','polvio_id','regu','regu_id','law','law_id','account','account_id','tax','tax_id','broadband','broadband_id','telephone','telephone_id','internet_use','internet_use_id','mobile','mobile_id','polyarchy','libdem','libdem_id','partipdem','delibdem','egaldem','civlib','phyvio','pollib','privlib','execon','execon_id','exgender','exgender_id','exgeo','exgeo_id','expol','expol_id','exsoc','exsoc_id','shutdown','shutdown_id','filter','filter_id','tenure_months','tenure_months_id','dem_duration','dem_duration_id','elections','elections_id','lastelection','lastelection_id']

# Check downsampling
y=pd.read_csv("out/df_out_full.csv",index_col=0)
fig,ax = plt.subplots()
y["d_civil_war"].hist()
fig,ax = plt.subplots()
y["d_civil_conflict"].hist()
fig,ax = plt.subplots()
y["d_remote"].hist()
fig,ax = plt.subplots()
y["d_sb"].hist()
fig,ax = plt.subplots()
y["d_ns"].hist()
fig,ax = plt.subplots()
y["d_osv"].hist()
fig,ax = plt.subplots()
y["d_protest"].hist()
fig,ax = plt.subplots()
y["d_riot"].hist()

# Check distributions
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
for var in ['d_civil_war_lag1','d_civil_war_lag1',"d_civil_war_zeros_decay","d_neighbors_civil_war_lag1",'d_civil_conflict_lag1',"d_civil_conflict_zeros_decay","d_neighbors_civil_conflict_lag1",'d_protest_lag1',"d_protest_zeros_decay","d_neighbors_proteset_event_counts_lag1",'d_riot_lag1',"d_riot_zeros_decay","d_neighbors_riot_event_counts_lag1",'d_remote_lag1',"d_remote_zeros_decay","d_neighbors_remote_fatalities_lag1",'d_sb_lag1',"d_sb_zeros_decay","d_neighbors_sb_fatalities_lag1",'d_ns_lag1',"d_ns_zeros_decay","d_neighbors_ns_fatalities_lag1",'d_osv_lag1',"d_osv_zeros_decay","d_neighbors_osv_fatalities_lag1","regime_duration",'pop_refugee','pop_refugee_id']:
    fig,ax = plt.subplots()
    x[var].hist()
    plt.title(var)
    plt.show()

x=pd.read_csv("out/df_demog_full.csv",index_col=0)
for var in ['pop','pop_density','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','monopoly_share','discriminated_share','powerless_share','dominant_share','ethnic_frac','rel_frac','lang_frac','race_frac']:
    fig,ax = plt.subplots()
    x[var].hist()
    plt.title(var)
    plt.show()
    
x=pd.read_csv("out/df_geog_full.csv",index_col=0)
for var in ['land','forest','forest_id','temp','temp_id','co2','co2_id','percip','percip_id','waterstress','waterstress_id','agri_land','agri_land_id','arable_land','arable_land_id','rugged','soil','desert','tropical','cont_africa','cont_asia','d_neighbors_con','no_neigh','d_neighbors_non_dem','libdem_id_neigh']:
    fig,ax = plt.subplots()
    x[var].hist()
    plt.title(var)
    plt.show()
    
x=pd.read_csv("out/df_econ_full.csv",index_col=0)
for var in ["oil_deposits","oil_production","oil_exports","natres_share","gdp","gni","gdp_growth","unemploy","unemploy_male","inflat","conprice","undernour","foodprod","water_rural","water_urb","agri_share","trade_share","fert","lifeexp_female","lifeexp_male","pop_growth","inf_mort","mig",'exports','exports_id','imports','imports_id','primary_female','primary_female_id','primary_male','primary_male_id','second_female','second_female_id','second_male','second_male_id','tert_female','tert_female_id','tert_male','tert_male_id','eys','eys_id','eys_male','eys_male_id','eys_female','eys_female_id','mys','mys_id','mys_male','mys_male_id','mys_female','mys_female_id']:
    fig,ax = plt.subplots()
    x[var].hist()
    plt.title(var)
    plt.show()
    
x=pd.read_csv("out/df_pol_full.csv",index_col=0)
for var in ['armedforces_share','milex_share','corruption','effectiveness','polvio','regu','law','account','tax','broadband','telephone','internet_use','mobile','polyarchy','libdem','partipdem','delibdem','egaldem','civlib','phyvio','pollib','privlib','execon','exgender','exgeo','expol','exsoc','shutdown','filter','tenure_months','dem_duration','elections','lastelection']:
    fig,ax = plt.subplots()
    x[var].hist()
    plt.title(var)
    plt.show()
    
                                #################
                                ### Civil war ###
                                #################
###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Categorical
target="d_civil_war"
inputs=['d_civil_war_lag1']
base_war_df,base_war_evals,base_war_val=gen_model(y,x,target,inputs,names=names,name="base_war",model_fit=DummyClassifier(strategy="most_frequent"),outcome="categorical",int_methods=False)
base_war_df.to_csv("out/base_war_df.csv")
base_war_evals_df = pd.DataFrame.from_dict(base_war_evals, orient='index').reset_index()
base_war_evals_df.to_csv("out/base_war_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Transforms
x["pop_refugee"]=np.log(x["pop_refugee"]+1)

# Categorical
target="d_civil_war"
inputs=['d_civil_war_lag1',"d_civil_war_zeros_decay","d_neighbors_civil_war_lag1","regime_duration",'pop_refugee','pop_refugee_id']
history_war_df,history_war_evals,history_war_val=gen_model(y,x,target,inputs,names=names,name="history_war",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=False,downsampling=False)
history_war_df.to_csv("out/history_war_df.csv")
history_war_evals_df = pd.DataFrame.from_dict(history_war_evals, orient='index').reset_index()
history_war_evals_df.to_csv("out/history_war_evals_df.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Categorical
target="d_civil_war"
inputs=demog_theme
demog_war_df,demog_war_evals,demog_war_val=gen_model(y,x,target,inputs,names=names,name="demog_war",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=False,downsampling=False)
demog_war_df.to_csv("out/demog_war_df.csv")
demog_war_evals_df = pd.DataFrame.from_dict(demog_war_evals, orient='index').reset_index()
demog_war_evals_df.to_csv("out/demog_war_evals_df.csv")

#########################################
### E.  Geography & environment theme ###
#########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
preprocess_min_max_group(x,"temp","country")


# Categorical
target="d_civil_war"
inputs=geog_theme
geog_war_df,geog_war_evals,geog_war_val=gen_model(y,x,target,inputs,names=names,name="geog_war",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=False,downsampling=False)
geog_war_df.to_csv("out/geog_war_df.csv")
geog_war_evals_df = pd.DataFrame.from_dict(geog_war_evals, orient='index').reset_index()
geog_war_evals_df.to_csv("out/geog_war_evals_df.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)

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
target="d_civil_war"
inputs=econ_theme
econ_war_df,econ_war_evals,econ_war_val=gen_model(y,x,target,inputs,names=names,name="econ_war",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=False,downsampling=False)
econ_war_df.to_csv("out/econ_war_df.csv")
econ_war_evals_df = pd.DataFrame.from_dict(econ_war_evals, orient='index').reset_index()
econ_war_evals_df.to_csv("out/econ_war_evals_df.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)

# Categorical
target="d_civil_war"
inputs=pol_theme
pol_war_df,pol_war_evals,pol_war_val=gen_model(y,x,target,inputs,names=names,name="pol_war",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=False,downsampling=False)
pol_war_df.to_csv("out/pol_war_df.csv")
pol_war_evals_df = pd.DataFrame.from_dict(pol_war_evals, orient='index').reset_index()
pol_war_evals_df.to_csv("out/pol_war_evals_df.csv")

################
### Ensemble ###
################

# Calculate weighted average
predictions=pd.concat([history_war_df.preds_proba,demog_war_df.preds_proba,geog_war_df.preds_proba,econ_war_df.preds_proba,pol_war_df.preds_proba], axis=1)
weights=[1-history_war_val["brier"],1-demog_war_val["brier"],1-geog_war_val["brier"],1-econ_war_val["brier"],1-pol_war_val["brier"]]
weights_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_war_n = [x / sum(weights_n) for x in weights_n]
ensemble = (history_war_df.preds_proba*weights_war_n[0])+(demog_war_df.preds_proba*weights_war_n[1])+(geog_war_df.preds_proba*weights_war_n[2])+(econ_war_df.preds_proba*weights_war_n[3])+(pol_war_df.preds_proba*weights_war_n[4])
ensemble_war=pd.concat([history_war_df[["country","year","d_civil_war","test"]],pd.DataFrame(ensemble)],axis=1)
ensemble_war.columns=["country","year","d_civil_war","test","preds_proba"]
ensemble_war=ensemble_war.reset_index(drop=True)

# Evals
brier = brier_score_loss(ensemble_war.loc[ensemble_war["test"]==1].d_civil_war, ensemble_war.loc[ensemble_war["test"]==1].preds_proba)
precision, recall, _ = precision_recall_curve(ensemble_war.loc[ensemble_war["test"]==1].d_civil_war, ensemble_war.loc[ensemble_war["test"]==1].preds_proba)
aupr = auc(recall, precision)
auroc = roc_auc_score(ensemble_war.loc[ensemble_war["test"]==1].d_civil_war, ensemble_war.loc[ensemble_war["test"]==1].preds_proba)
evals_war_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_war_ensemble_df = pd.DataFrame.from_dict(evals_war_ensemble, orient='index').reset_index()
evals_war_ensemble_df.to_csv("out/evals_war_ensemble_df.csv")

###################
### Evaluations ###
###################

print(f"{round(base_war_evals['aupr'],5)} &  \\\
      {round(base_war_evals['auroc'],5)} &  \\\
      {round(base_war_evals['brier'],5)}")
      
print(f"{round(history_war_evals['aupr'],5)} &  \\\
      {round(history_war_evals['auroc'],5)} &  \\\
      {round(history_war_evals['brier'],5)}")
      
print(f"{round(demog_war_evals['aupr'],5)} &  \\\
      {round(demog_war_evals['auroc'],5)} &  \\\
      {round(demog_war_evals['brier'],5)}")

print(f"{round(geog_war_evals['aupr'],5)} &  \\\
      {round(geog_war_evals['auroc'],5)} &  \\\
      {round(geog_war_evals['brier'],5)}")
      
print(f"{round(econ_war_evals['aupr'],5)} &  \\\
      {round(econ_war_evals['auroc'],5)} &  \\\
      {round(econ_war_evals['brier'],5)}")
           
print(f"{round(pol_war_evals['aupr'],5)} &  \\\
       {round(pol_war_evals['auroc'],5)} &  \\\
       {round(pol_war_evals['brier'],5)}")          
      
print(f"{round(evals_war_ensemble['aupr'],5)} &  \\\
       {round(evals_war_ensemble['auroc'],5)} &  \\\
       {round(evals_war_ensemble['brier'],5)}")   
 
                                ######################
                                ### Civil conflict ###
                                ######################
###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Categorical
target="d_civil_conflict"
inputs=['d_civil_conflict_lag1']
base_conflict_df,base_conflict_evals,base_conflict_val=gen_model(y,x,target,inputs,names=names,name="base_conflict",model_fit=DummyClassifier(strategy="most_frequent"),outcome="categorical",int_methods=False)
base_conflict_df.to_csv("out/base_conflict_df.csv")
base_conflict_evals_df = pd.DataFrame.from_dict(base_conflict_evals, orient='index').reset_index()
base_conflict_evals_df.to_csv("out/base_conflict_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Transforms
x["pop_refugee"]=np.log(x["pop_refugee"]+1)

# Categorical
target="d_civil_conflict"
inputs=['d_civil_conflict_lag1',"d_civil_conflict_zeros_decay","d_neighbors_civil_conflict_lag1","regime_duration",'pop_refugee','pop_refugee_id']
history_conflict_df,history_conflict_evals,history_conflict_val=gen_model(y,x,target,inputs,names=names,name="history_conflict",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=False)
history_conflict_df.to_csv("out/history_conflict_df.csv")
history_conflict_evals_df = pd.DataFrame.from_dict(history_conflict_evals, orient='index').reset_index()
history_conflict_evals_df.to_csv("out/history_conflict_evals_df.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Categorical
target="d_civil_conflict"
inputs=demog_theme
demog_conflict_df,demog_conflict_evals,demog_conflict_val=gen_model(y,x,target,inputs,names=names,name="demog_conflict",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=False)
demog_conflict_df.to_csv("out/demog_conflict_df.csv")
demog_conflict_evals_df = pd.DataFrame.from_dict(demog_conflict_evals, orient='index').reset_index()
demog_conflict_evals_df.to_csv("out/demog_conflict_evals_df.csv")

#########################################
### E.  Geography & environment theme ###
#########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
preprocess_min_max_group(x,"temp","country")

# Categorical
target="d_civil_conflict"
inputs=geog_theme
geog_conflict_df,geog_conflict_evals,geog_conflict_val=gen_model(y,x,target,inputs,names=names,name="geog_conflict",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=False)
geog_conflict_df.to_csv("out/geog_conflict_df.csv")
geog_conflict_evals_df = pd.DataFrame.from_dict(geog_conflict_evals, orient='index').reset_index()
geog_conflict_evals_df.to_csv("out/geog_conflict_evals_df.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)

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
target="d_civil_conflict"
inputs=econ_theme
econ_conflict_df,econ_conflict_evals,econ_conflict_val=gen_model(y,x,target,inputs,names=names,name="econ_conflict",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=False)
econ_conflict_df.to_csv("out/econ_conflict_df.csv")
econ_conflict_evals_df = pd.DataFrame.from_dict(econ_conflict_evals, orient='index').reset_index()
econ_conflict_evals_df.to_csv("out/econ_conflict_evals_df.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)


# Categorical
target="d_civil_conflict"
inputs=pol_theme
pol_conflict_df,pol_conflict_evals,pol_conflict_val=gen_model(y,x,target,inputs,names=names,name="pol_conflict",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=False)
pol_conflict_df.to_csv("out/pol_conflict_df.csv")
pol_conflict_evals_df = pd.DataFrame.from_dict(pol_conflict_evals, orient='index').reset_index()
pol_conflict_evals_df.to_csv("out/pol_conflict_evals_df.csv")

################
### Ensemble ###
################

# Calculate weighted average
predictions=pd.concat([history_conflict_df.preds_proba,demog_conflict_df.preds_proba,geog_conflict_df.preds_proba,econ_conflict_df.preds_proba,pol_conflict_df.preds_proba], axis=1)
weights=[1-history_conflict_val["brier"],1-demog_conflict_val["brier"],1-geog_conflict_val["brier"],1-econ_conflict_val["brier"],1-pol_conflict_val["brier"]]
weights_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_conflict_n = [x / sum(weights_n) for x in weights_n]
ensemble = (history_conflict_df.preds_proba*weights_conflict_n[0])+(demog_conflict_df.preds_proba*weights_conflict_n[1])+(geog_conflict_df.preds_proba*weights_conflict_n[2])+(econ_conflict_df.preds_proba*weights_conflict_n[3])+(pol_conflict_df.preds_proba*weights_conflict_n[4])
ensemble_conflict=pd.concat([history_conflict_df[["country","year","d_civil_conflict","test"]],pd.DataFrame(ensemble)],axis=1)
ensemble_conflict.columns=["country","year","d_civil_conflict","test","preds_proba"]
ensemble_conflict=ensemble_conflict.reset_index(drop=True)

# Evals
brier = brier_score_loss(ensemble_conflict.loc[ensemble_conflict["test"]==1].d_civil_conflict, ensemble_conflict.loc[ensemble_conflict["test"]==1].preds_proba)
precision, recall, _ = precision_recall_curve(ensemble_conflict.loc[ensemble_conflict["test"]==1].d_civil_conflict, ensemble_conflict.loc[ensemble_conflict["test"]==1].preds_proba)
aupr = auc(recall, precision)
auroc = roc_auc_score(ensemble_conflict.loc[ensemble_conflict["test"]==1].d_civil_conflict, ensemble_conflict.loc[ensemble_conflict["test"]==1].preds_proba)
evals_conflict_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_conflict_ensemble_df = pd.DataFrame.from_dict(evals_conflict_ensemble, orient='index').reset_index()
evals_conflict_ensemble_df.to_csv("out/evals_conflict_ensemble_df.csv")

###################
### Evaluations ###
###################

print(f"{round(base_conflict_evals['aupr'],5)} &  \\\
      {round(base_conflict_evals['auroc'],5)} &  \\\
      {round(base_conflict_evals['brier'],5)}")
      
print(f"{round(history_conflict_evals['aupr'],5)} &  \\\
      {round(history_conflict_evals['auroc'],5)} &  \\\
      {round(history_conflict_evals['brier'],5)}")
      
print(f"{round(demog_conflict_evals['aupr'],5)} &  \\\
      {round(demog_conflict_evals['auroc'],5)} &  \\\
      {round(demog_conflict_evals['brier'],5)}")

print(f"{round(geog_conflict_evals['aupr'],5)} &  \\\
      {round(geog_conflict_evals['auroc'],5)} &  \\\
      {round(geog_conflict_evals['brier'],5)}")

print(f"{round(econ_conflict_evals['aupr'],5)} &  \\\
      {round(econ_conflict_evals['auroc'],5)} &  \\\
      {round(econ_conflict_evals['brier'],5)}")
           
print(f"{round(pol_conflict_evals['aupr'],5)} &  \\\
       {round(pol_conflict_evals['auroc'],5)} &  \\\
       {round(pol_conflict_evals['brier'],5)}")          
      
print(f"{round(evals_conflict_ensemble['aupr'],5)} &  \\\
       {round(evals_conflict_ensemble['auroc'],5)} &  \\\
       {round(evals_conflict_ensemble['brier'],5)}")   
             
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
                                
base=pd.read_csv("data/data_out/acled_cy_protest.csv",index_col=0)
base = base[["year","gw_codes"]][~base['gw_codes'].isin(list(exclude.values()))]
                              
###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Categorical
target="d_protest"
inputs=['d_protest_lag1']
base_protest_df,base_protest_evals,base_protest_val=gen_model(y,x,target,inputs,names,"base_protest",model_fit=DummyClassifier(strategy="most_frequent"),outcome="categorical",int_methods=False)
base_protest_df.to_csv("out/base_protest_df.csv")
base_protest_evals_df = pd.DataFrame.from_dict(base_protest_evals, orient='index').reset_index()
base_protest_evals_df.to_csv("out/base_protest_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Transforms
x["pop_refugee"]=np.log(x["pop_refugee"]+1)

# Categorical
target="d_protest"
inputs=['d_protest_lag1',"d_protest_zeros_decay","d_neighbors_proteset_event_counts_lag1","regime_duration",'pop_refugee','pop_refugee_id']
history_protest_df,history_protest_evals,history_protest_val,history_protest_shap=gen_model(y,x,target,inputs,names=names,name="history_protest",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=['d_protest_lag1',"d_protest_zeros_decay","d_neighbors_proteset_event_counts_lag1","regime_duration",'pop_refugee'])
history_protest_df.to_csv("out/history_protest_df.csv")
history_protest_evals_df = pd.DataFrame.from_dict(history_protest_evals, orient='index').reset_index()
history_protest_evals_df.to_csv("out/history_protest_evals_df.csv")
pd.DataFrame(history_protest_shap[:,:,1]).to_csv("out/history_protest_shap.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Categorical
target="d_protest"
inputs=demog_theme
demog_protest_df,demog_protest_evals,demog_protest_val,demog_protest_shap=gen_model(y,x,target,inputs,names=names,name="demog_protest",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=['pop','pop_density','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','monopoly_share','discriminated_share','powerless_share','dominant_share','ethnic_frac','rel_frac','lang_frac','race_frac'])
demog_protest_df.to_csv("out/demog_protest_df.csv")
demog_protest_evals_df = pd.DataFrame.from_dict(demog_protest_evals, orient='index').reset_index()
demog_protest_evals_df.to_csv("out/demog_protest_evals_df.csv")
pd.DataFrame(demog_protest_shap[:,:,1]).to_csv("out/demog_protest_shap.csv")

#########################################
### E.  Geography & environment theme ###
#########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
preprocess_min_max_group(x,"temp","country")

# Categorical
target="d_protest"
inputs=geog_theme
geog_protest_df,geog_protest_evals,geog_protest_val,geog_protest_shap=gen_model(y,x,target,inputs,names=names,name="geog_protest",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["land","temp_norm","forest","co2","percip","waterstress","agri_land","arable_land","rugged","soil","desert","tropical","cont_africa","cont_asia","d_neighbors_con","no_neigh","d_neighbors_non_dem"])
geog_protest_df.to_csv("out/geog_protest_df.csv")
geog_protest_evals_df = pd.DataFrame.from_dict(geog_protest_evals, orient='index').reset_index()
geog_protest_evals_df.to_csv("out/geog_protest_evals_df.csv")
pd.DataFrame(geog_protest_shap[:,:,1]).to_csv("out/geog_protest_shap.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

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
econ_protest_df,econ_protest_evals,econ_protest_val,econ_protest_shap=gen_model(y,x,target,inputs,names=names,name="econ_protest",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["oil_deposits","oil_production","oil_exports","natres_share","gdp","gni","gdp_growth_norm","unemploy","unemploy_male","inflat","conprice","undernour","foodprod_norm","water_rural","water_urb","agri_share","trade_share","fert","lifeexp_female","lifeexp_male","pop_growth_norm","inf_mort","mig_norm","broadband","telephone","internet_use","mobile"])
econ_protest_df.to_csv("out/econ_protest_df.csv")
econ_protest_evals_df = pd.DataFrame.from_dict(econ_protest_evals, orient='index').reset_index()
econ_protest_evals_df.to_csv("out/econ_protest_evals_df.csv")
pd.DataFrame(econ_protest_shap[:,:,1]).to_csv("out/econ_protest_shap.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)

# Categorical
target="d_protest"
inputs=pol_theme
pol_protest_df,pol_protest_evals,pol_protest_val,pol_protest_shap=gen_model(y,x,target,inputs,names=names,name="pol_protest",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["eys","eys_male","eys_female","mys","mys_female","mys_male","armedforces_share","milex_share","corruption","effectiveness","polvio","regu","law","account","tax","polyarchy","libdem","partipdem","delibdem","egaldem","civlib","phyvio","pollib","privlib","execon","exgender","exgeo","expol","exsoc","shutdown","filter","tenure_months","dem_duration","elections","lastelection"])
pol_protest_df.to_csv("out/pol_protest_df.csv")
pol_protest_evals_df = pd.DataFrame.from_dict(pol_protest_evals, orient='index').reset_index()
pol_protest_evals_df.to_csv("out/pol_protest_evals_df.csv")
pd.DataFrame(pol_protest_shap[:,:,1]).to_csv("out/pol_protest_shap.csv")

################
### Ensemble ###
################

# Calculate weighted average
predictions=pd.concat([history_protest_df.preds_proba,demog_protest_df.preds_proba,geog_protest_df.preds_proba,econ_protest_df.preds_proba,pol_protest_df.preds_proba], axis=1)
weights=[1-history_protest_val["brier"],1-demog_protest_val["brier"],1-geog_protest_val["brier"],1-econ_protest_val["brier"],1-pol_protest_val["brier"]]
weights_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_protest_n = [x / sum(weights_n) for x in weights_n]
ensemble = (history_protest_df.preds_proba*weights_protest_n[0])+(demog_protest_df.preds_proba*weights_protest_n[1])+(geog_protest_df.preds_proba*weights_protest_n[2])+(econ_protest_df.preds_proba*weights_protest_n[3])+(pol_protest_df.preds_proba*weights_protest_n[4])
ensemble_protest=pd.concat([history_protest_df[["country","year","d_protest","test"]],pd.DataFrame(ensemble)],axis=1)
ensemble_protest.columns=["country","year","d_protest","test","preds_proba"]
ensemble_protest=ensemble_protest.reset_index(drop=True)

# Evals
brier = brier_score_loss(ensemble_protest.loc[ensemble_protest["test"]==1].d_protest, ensemble_protest.loc[ensemble_protest["test"]==1].preds_proba)
precision, recall, _ = precision_recall_curve(ensemble_protest.loc[ensemble_protest["test"]==1].d_protest, ensemble_protest.loc[ensemble_protest["test"]==1].preds_proba)
aupr = auc(recall, precision)
auroc = roc_auc_score(ensemble_protest.loc[ensemble_protest["test"]==1].d_protest, ensemble_protest.loc[ensemble_protest["test"]==1].preds_proba)
evals_protest_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_protest_ensemble_df = pd.DataFrame.from_dict(evals_protest_ensemble, orient='index').reset_index()
evals_protest_ensemble_df.to_csv("out/evals_protest_ensemble_df.csv")

###################
### Evaluations ###
###################

print(f"{round(base_protest_evals['aupr'],5)} &  \\\
      {round(base_protest_evals['auroc'],5)} &  \\\
      {round(base_protest_evals['brier'],5)}")
      
print(f"{round(history_protest_evals['aupr'],5)} &  \\\
      {round(history_protest_evals['auroc'],5)} &  \\\
      {round(history_protest_evals['brier'],5)}")
      
print(f"{round(demog_protest_evals['aupr'],5)} &  \\\
      {round(demog_protest_evals['auroc'],5)} &  \\\
      {round(demog_protest_evals['brier'],5)}")

print(f"{round(geog_protest_evals['aupr'],5)} &  \\\
      {round(geog_protest_evals['auroc'],5)} &  \\\
      {round(geog_protest_evals['brier'],5)}")

print(f"{round(econ_protest_evals['aupr'],5)} &  \\\
      {round(econ_protest_evals['auroc'],5)} &  \\\
      {round(econ_protest_evals['brier'],5)}")
           
print(f"{round(pol_protest_evals['aupr'],5)} &  \\\
       {round(pol_protest_evals['auroc'],5)} &  \\\
       {round(pol_protest_evals['brier'],5)}")          
      
print(f"{round(evals_protest_ensemble['aupr'],5)} &  \\\
       {round(evals_protest_ensemble['auroc'],5)} &  \\\
       {round(evals_protest_ensemble['brier'],5)}")          
      
        
                                    #############
                                    ### Riots ###
                                    #############

###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Categorical
target="d_riot"
inputs=['d_riot_lag1']
base_riot_df,base_riot_evals,base_riot_val=gen_model(y,x,target,inputs,names=names,name="base_riot",model_fit=DummyClassifier(strategy="most_frequent"),outcome="categorical",int_methods=False,downsampling=False)
base_riot_df.to_csv("out/base_riot_df.csv")
base_riot_evals_df = pd.DataFrame.from_dict(base_riot_evals, orient='index').reset_index()
base_riot_evals_df.to_csv("out/base_riot_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Transforms
x["pop_refugee"]=np.log(x["pop_refugee"]+1)

# Categorical
target="d_riot"
inputs=['d_riot_lag1',"d_riot_zeros_decay","d_neighbors_riot_event_counts_lag1",'regime_duration', 'pop_refugee', 'pop_refugee_id']
history_riot_df,history_riot_evals,history_riot_val,history_riot_shap=gen_model(y,x,target,inputs,names=names,name="history_riot",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=['d_riot_lag1',"d_riot_zeros_decay","d_neighbors_riot_event_counts_lag1","regime_duration",'pop_refugee'],downsampling=False)
history_riot_df.to_csv("out/history_riot_df.csv")
history_riot_evals_df = pd.DataFrame.from_dict(history_riot_evals, orient='index').reset_index()
history_riot_evals_df.to_csv("out/history_riot_evals_df.csv")
pd.DataFrame(history_riot_shap[:,:,1]).to_csv("out/history_riot_shap.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Categorical
target="d_riot"
inputs=demog_theme
demog_riot_df,demog_riot_evals,demog_riot_val,demog_riot_shap=gen_model(y,x,target,inputs,names=names,name="demog_riot",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=['pop','pop_density','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','monopoly_share','discriminated_share','powerless_share','dominant_share','ethnic_frac','rel_frac','lang_frac','race_frac'],downsampling=False)
demog_riot_df.to_csv("out/demog_riot_df.csv")
demog_riot_evals_df = pd.DataFrame.from_dict(demog_riot_evals, orient='index').reset_index()
demog_riot_evals_df.to_csv("out/demog_riot_evals_df.csv")
pd.DataFrame(demog_riot_shap[:,:,1]).to_csv("out/demog_riot_shap.csv")

#########################################
### E.  Geography & environment theme ###
#########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
preprocess_min_max_group(x,"temp","country")

# Categorical
target="d_riot"
inputs=geog_theme
geog_riot_df,geog_riot_evals,geog_riot_val,geog_riot_shap=gen_model(y,x,target,inputs,names=names,name="geog_riot",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["land","temp","forest","co2","percip","waterstress","agri_land","arable_land","rugged","soil","desert","tropical","cont_africa","cont_asia","d_neighbors_con","no_neigh","d_neighbors_non_dem"],downsampling=False)
geog_riot_df.to_csv("out/geog_riot_df.csv")
geog_riot_evals_df = pd.DataFrame.from_dict(geog_riot_evals, orient='index').reset_index()
geog_riot_evals_df.to_csv("out/geog_riot_evals_df.csv")
pd.DataFrame(geog_riot_shap[:,:,1]).to_csv("out/geog_riot_shap.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

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
econ_riot_df,econ_riot_evals,econ_riot_val,econ_riot_shap=gen_model(y,x,target,inputs,names=names,name="econ_riot",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["oil_deposits","oil_production","oil_exports","natres_share","gdp","gni","gdp_growth_norm","unemploy","unemploy_male","inflat","conprice","undernour","foodprod_norm","water_rural","water_urb","agri_share","trade_share","fert","lifeexp_female","lifeexp_male","pop_growth_norm","inf_mort","mig_norm","broadband","telephone","internet_use","mobile"],downsampling=False)
econ_riot_df.to_csv("out/econ_riot_df.csv")
econ_riot_evals_df = pd.DataFrame.from_dict(econ_riot_evals, orient='index').reset_index()
econ_riot_evals_df.to_csv("out/econ_riot_evals_df.csv")
pd.DataFrame(econ_riot_shap[:,:,1]).to_csv("out/econ_riot_shap.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)

# Categorical
target="d_riot"
inputs=pol_theme
pol_riot_df,pol_riot_evals,pol_riot_val,pol_riot_shap=gen_model(y,x,target,inputs,names=names,name="pol_riot",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["eys","eys_male","eys_female","mys","mys_female","mys_male","armedforces_share","milex_share","corruption","effectiveness","polvio","regu","law","account","tax","polyarchy","libdem","partipdem","delibdem","egaldem","civlib","phyvio","pollib","privlib","execon","exgender","exgeo","expol","exsoc","shutdown","filter","tenure_months","dem_duration","elections","lastelection"],downsampling=False)
pol_riot_df.to_csv("out/pol_riot_df.csv")
pol_riot_evals_df = pd.DataFrame.from_dict(pol_riot_evals, orient='index').reset_index()
pol_riot_evals_df.to_csv("out/pol_riot_evals_df.csv")
pd.DataFrame(pol_riot_shap[:,:,1]).to_csv("out/pol_riot_shap.csv")

################
### Ensemble ###
################

# Calculate weighted average
predictions=pd.concat([history_riot_df.preds_proba,demog_riot_df.preds_proba,geog_riot_df.preds_proba,econ_riot_df.preds_proba,pol_riot_df.preds_proba], axis=1)
weights=[1-history_riot_val["brier"],1-demog_riot_val["brier"],1-geog_riot_val["brier"],1-econ_riot_val["brier"],1-pol_riot_val["brier"]]
weights_riot_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_riot_n = [x / sum(weights_riot_n) for x in weights_riot_n]
ensemble = (history_riot_df.preds_proba*weights_riot_n[0])+(demog_riot_df.preds_proba*weights_riot_n[1])+(geog_riot_df.preds_proba*weights_riot_n[2])+(econ_riot_df.preds_proba*weights_riot_n[3])+(pol_riot_df.preds_proba*weights_riot_n[4])
ensemble_riot=pd.concat([history_riot_df[["country","year","d_riot","test"]],pd.DataFrame(ensemble)],axis=1)
ensemble_riot.columns=["country","year","d_riot","test","preds_proba"]
ensemble_riot=ensemble_riot.reset_index(drop=True)

# Evals
brier = brier_score_loss(ensemble_riot.loc[ensemble_riot["test"]==1].d_riot, ensemble_riot.loc[ensemble_riot["test"]==1].preds_proba)
precision, recall, _ = precision_recall_curve(ensemble_riot.loc[ensemble_riot["test"]==1].d_riot, ensemble_riot.loc[ensemble_riot["test"]==1].preds_proba)
aupr = auc(recall, precision)
auroc = roc_auc_score(ensemble_riot.loc[ensemble_riot["test"]==1].d_riot, ensemble_riot.loc[ensemble_riot["test"]==1].preds_proba)
evals_riot_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_riot_ensemble_df = pd.DataFrame.from_dict(evals_riot_ensemble, orient='index').reset_index()
evals_riot_ensemble_df.to_csv("out/evals_riot_ensemble_df.csv")

###################
### Evaluations ###
###################

print(f"{round(base_riot_evals['aupr'],5)} &  \\\
      {round(base_riot_evals['auroc'],5)} &  \\\
      {round(base_riot_evals['brier'],5)}")
      
print(f"{round(history_riot_evals['aupr'],5)} &  \\\
      {round(history_riot_evals['auroc'],5)} &  \\\
      {round(history_riot_evals['brier'],5)}")
      
print(f"{round(demog_riot_evals['aupr'],5)} &  \\\
      {round(demog_riot_evals['auroc'],5)} &  \\\
      {round(demog_riot_evals['brier'],5)}")

print(f"{round(geog_riot_evals['aupr'],5)} &  \\\
      {round(geog_riot_evals['auroc'],5)} &  \\\
      {round(geog_riot_evals['brier'],5)}")

print(f"{round(econ_riot_evals['aupr'],5)} &  \\\
      {round(econ_riot_evals['auroc'],5)} &  \\\
      {round(econ_riot_evals['brier'],5)}")
           
print(f"{round(pol_riot_evals['aupr'],5)} &  \\\
       {round(pol_riot_evals['auroc'],5)} &  \\\
       {round(pol_riot_evals['brier'],5)}")          
      
print(f"{round(evals_riot_ensemble['aupr'],5)} &  \\\
       {round(evals_riot_ensemble['auroc'],5)} &  \\\
       {round(evals_riot_ensemble['brier'],5)}")  
      
                                ##############
                                ### Remote ###
                                ##############

###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Categorical
target="d_remote"
inputs=['d_remote_lag1']
base_terror_df,base_terror_evals,base_terror_val=gen_model(y,x,target,inputs,names=names,name="base_terror",model_fit=DummyClassifier(strategy="most_frequent"),outcome="categorical",int_methods=False)
base_terror_df.to_csv("out/base_terror_df.csv")
base_terror_evals_df = pd.DataFrame.from_dict(base_terror_evals, orient='index').reset_index()
base_terror_evals_df.to_csv("out/base_terror_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Transforms
x["pop_refugee"]=np.log(x["pop_refugee"]+1)

# Categorical
target="d_remote"
inputs=['d_remote_lag1',"d_remote_zeros_decay","d_neighbors_remote_fatalities_lag1",'regime_duration','pop_refugee','pop_refugee_id']
history_terror_df,history_terror_evals,history_terror_val,history_terror_shap=gen_model(y,x,target,inputs,names=names,name="history_terror",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=['d_terror_lag1',"d_terror_zeros_decay","d_neighbors_terror_event_counts_lag1","regime_duration",'pop_refugee'])
history_terror_df.to_csv("out/history_terror_df.csv")
history_terror_evals_df = pd.DataFrame.from_dict(history_terror_evals, orient='index').reset_index()
history_terror_evals_df.to_csv("out/history_terror_evals_df.csv")
pd.DataFrame(history_terror_shap[:,:,1]).to_csv("out/history_terror_shap.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Categorical
target="d_remote"
inputs=demog_theme
demog_terror_df,demog_terror_evals,demog_terror_val,demog_terror_shap=gen_model(y,x,target,inputs,names=names,name="demog_terror",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=['pop','pop_density','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','monopoly_share','discriminated_share','powerless_share','dominant_share','ethnic_frac','rel_frac','lang_frac','race_frac'])
demog_terror_df.to_csv("out/demog_terror_df.csv")
demog_terror_evals_df = pd.DataFrame.from_dict(demog_terror_evals, orient='index').reset_index()
demog_terror_evals_df.to_csv("out/demog_terror_evals_df.csv")
pd.DataFrame(demog_terror_shap[:,:,1]).to_csv("out/demog_terror_shap.csv")

########################################
### E. Geography & environment theme ###
########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
preprocess_min_max_group(x,"temp","country")

# Categorical
target="d_remote"
inputs=geog_theme
geog_terror_df,geog_terror_evals,geog_terror_val,geog_terror_shap=gen_model(y,x,target,inputs,names=names,name="geog_terror",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["land","temp","forest","co2","percip","waterstress","agri_land","arable_land","rugged","soil","desert","tropical","cont_africa","cont_asia","d_neighbors_con","no_neigh","d_neighbors_non_dem"])
geog_terror_df.to_csv("out/geog_terror_df.csv")
geog_terror_evals_df = pd.DataFrame.from_dict(geog_terror_evals, orient='index').reset_index()
geog_terror_evals_df.to_csv("out/geog_terror_evals_df.csv")
pd.DataFrame(geog_terror_shap[:,:,1]).to_csv("out/geog_terror_shap.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

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
target="d_remote"
inputs=econ_theme
econ_terror_df,econ_terror_evals,econ_terror_val,econ_terror_shap=gen_model(y,x,target,inputs,names=names,name="econ_terror",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["oil_deposits","oil_production","oil_exports","natres_share","gdp","gni","gdp_growth_norm","unemploy","unemploy_male","inflat","conprice","undernour","foodprod_norm","water_rural","water_urb","agri_share","trade_share","fert","lifeexp_female","lifeexp_male","pop_growth_norm","inf_mort","mig_norm","broadband","telephone","internet_use","mobile"])
econ_terror_df.to_csv("out/econ_terror_df.csv")
econ_terror_evals_df = pd.DataFrame.from_dict(econ_terror_evals, orient='index').reset_index()
econ_terror_evals_df.to_csv("out/econ_terror_evals_df.csv")
pd.DataFrame(econ_terror_shap[:,:,1]).to_csv("out/econ_terror_shap.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)
y=pd.merge(left=base,right=y,on=["year","gw_codes"],how="left")
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)

# Categorical
target="d_remote"
inputs=pol_theme
pol_terror_df,pol_terror_evals,pol_terror_val,pol_terror_shap=gen_model(y,x,target,inputs,names=names,name="pol_terror",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["eys","eys_male","eys_female","mys","mys_female","mys_male","armedforces_share","milex_share","corruption","effectiveness","polvio","regu","law","account","tax","polyarchy","libdem","partipdem","delibdem","egaldem","civlib","phyvio","pollib","privlib","execon","exgender","exgeo","expol","exsoc","shutdown","filter","tenure_months","dem_duration","elections","lastelection"])
pol_terror_df.to_csv("out/pol_terror_df.csv")
pol_terror_evals_df = pd.DataFrame.from_dict(pol_terror_evals, orient='index').reset_index()
pol_terror_evals_df.to_csv("out/pol_terror_evals_df.csv")
pd.DataFrame(pol_terror_shap[:,:,1]).to_csv("out/pol_terror_shap.csv")

################
### Ensemble ###
################

# Calculate weighted average
predictions=pd.concat([history_terror_df.preds_proba,demog_terror_df.preds_proba,geog_terror_df.preds_proba,econ_terror_df.preds_proba,pol_terror_df.preds_proba], axis=1)
weights=[1-history_terror_val["brier"],1-demog_terror_val["brier"],1-geog_terror_val["brier"],1-econ_terror_val["brier"],1-pol_terror_val["brier"]]
weights_terror_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_terror_n = [x / sum(weights_terror_n) for x in weights_terror_n]
ensemble = (history_terror_df.preds_proba*weights_terror_n[0])+(demog_terror_df.preds_proba*weights_terror_n[1])+(geog_terror_df.preds_proba*weights_terror_n[2])+(econ_terror_df.preds_proba*weights_terror_n[3])+(pol_terror_df.preds_proba*weights_terror_n[4])
ensemble_terror=pd.concat([history_terror_df[["country","year","d_remote","test"]],pd.DataFrame(ensemble)],axis=1)
ensemble_terror.columns=["country","year","d_terror","test","preds_proba"]
ensemble_terror=ensemble_terror.reset_index(drop=True)

# Evals
brier = brier_score_loss(ensemble_terror.loc[ensemble_terror["test"]==1].d_terror, ensemble_terror.loc[ensemble_terror["test"]==1].preds_proba)
precision, recall, _ = precision_recall_curve(ensemble_terror.loc[ensemble_terror["test"]==1].d_terror, ensemble_terror.loc[ensemble_terror["test"]==1].preds_proba)
aupr = auc(recall, precision)
auroc = roc_auc_score(ensemble_terror.loc[ensemble_terror["test"]==1].d_terror, ensemble_terror.loc[ensemble_terror["test"]==1].preds_proba)
evals_terror_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_terror_ensemble_df = pd.DataFrame.from_dict(evals_terror_ensemble, orient='index').reset_index()
evals_terror_ensemble_df.to_csv("out/evals_terror_ensemble_df.csv")

###################
### Evaluations ###
###################

print(f"{round(base_terror_evals['aupr'],5)} &  \\\
      {round(base_terror_evals['auroc'],5)} &  \\\
      {round(base_terror_evals['brier'],5)}")
      
print(f"{round(history_terror_evals['aupr'],5)} &  \\\
      {round(history_terror_evals['auroc'],5)} &  \\\
      {round(history_terror_evals['brier'],5)}")
      
print(f"{round(demog_terror_evals['aupr'],5)} &  \\\
      {round(demog_terror_evals['auroc'],5)} &  \\\
      {round(demog_terror_evals['brier'],5)}")

print(f"{round(geog_terror_evals['aupr'],5)} &  \\\
      {round(geog_terror_evals['auroc'],5)} &  \\\
      {round(geog_terror_evals['brier'],5)}")

print(f"{round(econ_terror_evals['aupr'],5)} &  \\\
      {round(econ_terror_evals['auroc'],5)} &  \\\
      {round(econ_terror_evals['brier'],5)}")
           
print(f"{round(pol_terror_evals['aupr'],5)} &  \\\
       {round(pol_terror_evals['auroc'],5)} &  \\\
       {round(pol_terror_evals['brier'],5)}")          
      
print(f"{round(evals_terror_ensemble['aupr'],5)} &  \\\
       {round(evals_terror_ensemble['auroc'],5)} &  \\\
       {round(evals_terror_ensemble['brier'],5)}")  
      
                            ###################
                            ### State-based ###
                            ###################

###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Categorical
target="d_sb"
inputs=['d_sb_lag1']
base_sb_df,base_sb_evals,base_sb_val=gen_model(y,x,target,inputs,names,"base_sb",model_fit=DummyClassifier(strategy="most_frequent"),outcome="categorical",int_methods=False)
base_sb_df.to_csv("out/base_sb_df.csv")
base_sb_evals_df = pd.DataFrame.from_dict(base_sb_evals, orient='index').reset_index()
base_sb_evals_df.to_csv("out/base_sb_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Transforms
x["pop_refugee"]=np.log(x["pop_refugee"]+1)

# Categorical
target="d_sb"
inputs=['d_sb_lag1',"d_sb_zeros_decay","d_neighbors_sb_fatalities_lag1",'regime_duration', 'pop_refugee', 'pop_refugee_id']
history_sb_df,history_sb_evals,history_sb_val,history_sb_shap=gen_model(y,x,target,inputs,names=names,name="history_sb",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=['d_sb_lag1',"d_sb_zeros_decay","d_neighbors_sb_event_counts_lag1","regime_duration",'pop_refugee'])
history_sb_df.to_csv("out/history_sb_df.csv")
history_sb_evals_df = pd.DataFrame.from_dict(history_sb_evals, orient='index').reset_index()
history_sb_evals_df.to_csv("out/history_sb_evals_df.csv")
pd.DataFrame(history_sb_shap[:,:,1]).to_csv("out/history_sb_shap.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Categorical
target="d_sb"
inputs=demog_theme
demog_sb_df,demog_sb_evals,demog_sb_val,demog_sb_shap=gen_model(y,x,target,inputs,names=names,name="demog_sb",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=['pop','pop_density','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','monopoly_share','discriminated_share','powerless_share','dominant_share','ethnic_frac','rel_frac','lang_frac','race_frac'])
demog_sb_df.to_csv("out/demog_sb_df.csv")
demog_sb_evals_df = pd.DataFrame.from_dict(demog_sb_evals, orient='index').reset_index()
demog_sb_evals_df.to_csv("out/demog_sb_evals_df.csv")
pd.DataFrame(demog_sb_shap[:,:,1]).to_csv("out/demog_sb_shap.csv")

########################################
### E. Geography & environment theme ###
########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
preprocess_min_max_group(x,"temp","country")

# Categorical
target="d_sb"
inputs=geog_theme
geog_sb_df,geog_sb_evals,geog_sb_val,geog_sb_shap=gen_model(y,x,target,inputs,names=names,name="geog_sb",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["land","temp","forest","co2","percip","waterstress","agri_land","arable_land","rugged","soil","desert","tropical","cont_africa","cont_asia","d_neighbors_con","no_neigh","d_neighbors_non_dem"])
geog_sb_df.to_csv("out/geog_sb_df.csv")
geog_sb_evals_df = pd.DataFrame.from_dict(geog_sb_evals, orient='index').reset_index()
geog_sb_evals_df.to_csv("out/geog_sb_evals_df.csv")
pd.DataFrame(geog_sb_shap[:,:,1]).to_csv("out/geog_sb_shap.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)

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
econ_sb_df,econ_sb_evals,econ_sb_val,econ_sb_shap=gen_model(y,x,target,inputs,names=names,name="econ_sb",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["oil_deposits","oil_production","oil_exports","natres_share","gdp","gni","gdp_growth_norm","unemploy","unemploy_male","inflat","conprice","undernour","foodprod_norm","water_rural","water_urb","agri_share","trade_share","fert","lifeexp_female","lifeexp_male","pop_growth_norm","inf_mort","mig_norm","broadband","telephone","internet_use","mobile"])
econ_sb_df.to_csv("out/econ_sb_df.csv")
econ_sb_evals_df = pd.DataFrame.from_dict(econ_sb_evals, orient='index').reset_index()
econ_sb_evals_df.to_csv("out/econ_sb_evals_df.csv")
pd.DataFrame(econ_sb_shap[:,:,1]).to_csv("out/econ_sb_shap.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)

# Categorical
target="d_sb"
inputs=pol_theme
pol_sb_df,pol_sb_evals,pol_sb_val,pol_sb_shap=gen_model(y,x,target,inputs,names=names,name="pol_sb",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["eys","eys_male","eys_female","mys","mys_female","mys_male","armedforces_share","milex_share","corruption","effectiveness","polvio","regu","law","account","tax","polyarchy","libdem","partipdem","delibdem","egaldem","civlib","phyvio","pollib","privlib","execon","exgender","exgeo","expol","exsoc","shutdown","filter","tenure_months","dem_duration","elections","lastelection"])
pol_sb_df.to_csv("out/pol_sb_df.csv")
pol_sb_evals_df = pd.DataFrame.from_dict(pol_sb_evals, orient='index').reset_index()
pol_sb_evals_df.to_csv("out/pol_sb_evals_df.csv")
pd.DataFrame(pol_sb_shap[:,:,1]).to_csv("out/pol_sb_shap.csv")

################
### Ensemble ###
################

# Calculate weighted average
predictions=pd.concat([history_sb_df.preds_proba,demog_sb_df.preds_proba,geog_sb_df.preds_proba,econ_sb_df.preds_proba,pol_sb_df.preds_proba], axis=1)
weights=[1-history_sb_val["brier"],1-demog_sb_val["brier"],1-geog_sb_val["brier"],1-econ_sb_val["brier"],1-pol_sb_val["brier"]]
weights_sb_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_sb_n = [x / sum(weights_sb_n) for x in weights_sb_n]
ensemble = (history_sb_df.preds_proba*weights_sb_n[0])+(demog_sb_df.preds_proba*weights_sb_n[1])+(geog_sb_df.preds_proba*weights_sb_n[2])+(econ_sb_df.preds_proba*weights_sb_n[3])+(pol_sb_df.preds_proba*weights_sb_n[4])
ensemble_sb=pd.concat([history_sb_df[["country","year","d_sb","test"]],pd.DataFrame(ensemble)],axis=1)
ensemble_sb.columns=["country","year","d_sb","test","preds_proba"]
ensemble_sb=ensemble_sb.reset_index(drop=True)

# Evals
brier = brier_score_loss(ensemble_sb.loc[ensemble_sb["test"]==1].d_sb, ensemble_sb.loc[ensemble_sb["test"]==1].preds_proba)
precision, recall, _ = precision_recall_curve(ensemble_sb.loc[ensemble_sb["test"]==1].d_sb, ensemble_sb.loc[ensemble_sb["test"]==1].preds_proba)
aupr = auc(recall, precision)
auroc = roc_auc_score(ensemble_sb.loc[ensemble_sb["test"]==1].d_sb, ensemble_sb.loc[ensemble_sb["test"]==1].preds_proba)
evals_sb_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_sb_ensemble_df = pd.DataFrame.from_dict(evals_sb_ensemble, orient='index').reset_index()
evals_sb_ensemble_df.to_csv("out/evals_sb_ensemble_df.csv")

###################
### Evaluations ###
###################

print(f"{round(base_sb_evals['aupr'],5)} &  \\\
      {round(base_sb_evals['auroc'],5)} &  \\\
      {round(base_sb_evals['brier'],5)}")
      
print(f"{round(history_sb_evals['aupr'],5)} &  \\\
      {round(history_sb_evals['auroc'],5)} &  \\\
      {round(history_sb_evals['brier'],5)}")
      
print(f"{round(demog_sb_evals['aupr'],5)} &  \\\
      {round(demog_sb_evals['auroc'],5)} &  \\\
      {round(demog_sb_evals['brier'],5)}")

print(f"{round(geog_sb_evals['aupr'],5)} &  \\\
      {round(geog_sb_evals['auroc'],5)} &  \\\
      {round(geog_sb_evals['brier'],5)}")

print(f"{round(econ_sb_evals['aupr'],5)} &  \\\
      {round(econ_sb_evals['auroc'],5)} &  \\\
      {round(econ_sb_evals['brier'],5)}")
           
print(f"{round(pol_sb_evals['aupr'],5)} &  \\\
       {round(pol_sb_evals['auroc'],5)} &  \\\
       {round(pol_sb_evals['brier'],5)}")          
      
print(f"{round(evals_sb_ensemble['aupr'],5)} &  \\\
       {round(evals_sb_ensemble['auroc'],5)} &  \\\
       {round(evals_sb_ensemble['brier'],5)}")  

                            ##########################
                            ### One-sided violence ###
                            ##########################

###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Categorical
target="d_osv"
inputs=['d_osv_lag1']
base_osv_df,base_osv_evals,base_osv_val=gen_model(y,x,target,inputs,names=names,name="base_osv",model_fit=DummyClassifier(strategy="most_frequent"),outcome="categorical",int_methods=False)
base_osv_df.to_csv("out/base_osv_df.csv")
base_osv_evals_df = pd.DataFrame.from_dict(base_osv_evals, orient='index').reset_index()
base_osv_evals_df.to_csv("out/base_osv_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Transforms
x["pop_refugee"]=np.log(x["pop_refugee"]+1)

# Categorical
target="d_osv"
inputs=['d_osv_lag1',"d_osv_zeros_decay","d_neighbors_osv_fatalities_lag1",'regime_duration','pop_refugee','pop_refugee_id']
history_osv_df,history_osv_evals,history_osv_val,history_osv_shap=gen_model(y,x,target,inputs,names=names,name="history_osv",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=['d_osv_lag1',"d_osv_zeros_decay","d_neighbors_osv_event_counts_lag1","regime_duration",'pop_refugee'])
history_osv_df.to_csv("out/history_osv_df.csv")
history_osv_evals_df = pd.DataFrame.from_dict(history_osv_evals, orient='index').reset_index()
history_osv_evals_df.to_csv("out/history_osv_evals_df.csv")
pd.DataFrame(history_osv_shap[:,:,1]).to_csv("out/history_osv_shap.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Categorical
target="d_osv"
inputs=demog_theme
demog_osv_df,demog_osv_evals,demog_osv_val,demog_osv_shap=gen_model(y,x,target,inputs,names=names,name="demog_osv",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=['pop','pop_density','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','monopoly_share','discriminated_share','powerless_share','dominant_share','ethnic_frac','rel_frac','lang_frac','race_frac'])
demog_osv_df.to_csv("out/demog_osv_df.csv")
demog_osv_evals_df = pd.DataFrame.from_dict(demog_osv_evals, orient='index').reset_index()
demog_osv_evals_df.to_csv("out/demog_osv_evals_df.csv")
pd.DataFrame(demog_osv_shap[:,:,1]).to_csv("out/demog_osv_shap.csv")

########################################
### E. Geography & environment theme ###
########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
preprocess_min_max_group(x,"temp","country")

# Categorical
target="d_osv"
inputs=geog_theme
geog_osv_df,geog_osv_evals,geog_osv_val,geog_osv_shap=gen_model(y,x,target,inputs,names=names,name="geog_osv",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["land","temp","forest","co2","percip","waterstress","agri_land","arable_land","rugged","soil","desert","tropical","cont_africa","cont_asia","d_neighbors_con","no_neigh","d_neighbors_non_dem"])
geog_osv_df.to_csv("out/geog_osv_df.csv")
geog_osv_evals_df = pd.DataFrame.from_dict(geog_osv_evals, orient='index').reset_index()
geog_osv_evals_df.to_csv("out/geog_osv_evals_df.csv")
pd.DataFrame(geog_osv_shap[:,:,1]).to_csv("out/geog_osv_shap.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)

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
econ_osv_df,econ_osv_evals,econ_osv_val,econ_osv_shap=gen_model(y,x,target,inputs,names=names,name="econ_osv",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["oil_deposits","oil_production","oil_exports","natres_share","gdp","gni","gdp_growth_norm","unemploy","unemploy_male","inflat","conprice","undernour","foodprod_norm","water_rural","water_urb","agri_share","trade_share","fert","lifeexp_female","lifeexp_male","pop_growth_norm","inf_mort","mig_norm","broadband","telephone","internet_use","mobile"])
econ_osv_df.to_csv("out/econ_osv_df.csv")
econ_osv_evals_df = pd.DataFrame.from_dict(econ_osv_evals, orient='index').reset_index()
econ_osv_evals_df.to_csv("out/econ_osv_evals_df.csv")
pd.DataFrame(econ_osv_shap[:,:,1]).to_csv("out/econ_osv_shap.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)

# Categorical
target="d_osv"
inputs=pol_theme
pol_osv_df,pol_osv_evals,pol_osv_val,pol_osv_shap=gen_model(y,x,target,inputs,names=names,name="pol_osv",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["eys","eys_male","eys_female","mys","mys_female","mys_male","armedforces_share","milex_share","corruption","effectiveness","polvio","regu","law","account","tax","polyarchy","libdem","partipdem","delibdem","egaldem","civlib","phyvio","pollib","privlib","execon","exgender","exgeo","expol","exsoc","shutdown","filter","tenure_months","dem_duration","elections","lastelection"])
pol_osv_df.to_csv("out/pol_osv_df.csv")
pol_osv_evals_df = pd.DataFrame.from_dict(pol_osv_evals, orient='index').reset_index()
pol_osv_evals_df.to_csv("out/pol_osv_evals_df.csv")
pd.DataFrame(pol_osv_shap[:,:,1]).to_csv("out/pol_osv_shap.csv")

################
### Ensemble ###
################

# Calculate weighted average
predictions=pd.concat([history_osv_df.preds_proba,demog_osv_df.preds_proba,geog_osv_df.preds_proba,econ_osv_df.preds_proba,pol_osv_df.preds_proba], axis=1)
weights=[1-history_osv_val["brier"],1-demog_osv_val["brier"],1-geog_osv_val["brier"],1-econ_osv_val["brier"],1-pol_osv_val["brier"]]
weights_osv_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_osv_n = [x / sum(weights_osv_n) for x in weights_osv_n]
ensemble = (history_osv_df.preds_proba*weights_osv_n[0])+(demog_osv_df.preds_proba*weights_osv_n[1])+(geog_osv_df.preds_proba*weights_osv_n[2])+(econ_osv_df.preds_proba*weights_osv_n[3])+(pol_osv_df.preds_proba*weights_osv_n[4])
ensemble_osv=pd.concat([history_osv_df[["country","year","d_osv","test"]],pd.DataFrame(ensemble)],axis=1)
ensemble_osv.columns=["country","year","d_osv","test","preds_proba"]
ensemble_osv=ensemble_osv.reset_index(drop=True)

# Evals
brier = brier_score_loss(ensemble_osv.loc[ensemble_osv["test"]==1].d_osv, ensemble_osv.loc[ensemble_osv["test"]==1].preds_proba)
precision, recall, _ = precision_recall_curve(ensemble_osv.loc[ensemble_osv["test"]==1].d_osv, ensemble_osv.loc[ensemble_osv["test"]==1].preds_proba)
aupr = auc(recall, precision)
auroc = roc_auc_score(ensemble_osv.loc[ensemble_osv["test"]==1].d_osv, ensemble_osv.loc[ensemble_osv["test"]==1].preds_proba)
evals_osv_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_osv_ensemble_df = pd.DataFrame.from_dict(evals_osv_ensemble, orient='index').reset_index()
evals_osv_ensemble_df.to_csv("out/evals_osv_ensemble_df.csv")

###################
### Evaluations ###
###################

print(f"{round(base_osv_evals['aupr'],5)} &  \\\
      {round(base_osv_evals['auroc'],5)} &  \\\
      {round(base_osv_evals['brier'],5)}")
      
print(f"{round(history_osv_evals['aupr'],5)} &  \\\
      {round(history_osv_evals['auroc'],5)} &  \\\
      {round(history_osv_evals['brier'],5)}")
      
print(f"{round(demog_osv_evals['aupr'],5)} &  \\\
      {round(demog_osv_evals['auroc'],5)} &  \\\
      {round(demog_osv_evals['brier'],5)}")

print(f"{round(geog_osv_evals['aupr'],5)} &  \\\
      {round(geog_osv_evals['auroc'],5)} &  \\\
      {round(geog_osv_evals['brier'],5)}")

print(f"{round(econ_osv_evals['aupr'],5)} &  \\\
      {round(econ_osv_evals['auroc'],5)} &  \\\
      {round(econ_osv_evals['brier'],5)}")
           
print(f"{round(pol_osv_evals['aupr'],5)} &  \\\
       {round(pol_osv_evals['auroc'],5)} &  \\\
       {round(pol_osv_evals['brier'],5)}")          
      
print(f"{round(evals_osv_ensemble['aupr'],5)} &  \\\
       {round(evals_osv_ensemble['auroc'],5)} &  \\\
       {round(evals_osv_ensemble['brier'],5)}")  
            
                            #######################
                            ### Non-state based ###
                            #######################

###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Categorical
target="d_ns"
inputs=['d_ns_lag1']
base_ns_df,base_ns_evals,base_ns_val=gen_model(y,x,target,inputs,names=names,name="base_ns",model_fit=DummyClassifier(strategy="most_frequent"),outcome="categorical",int_methods=False,downsampling=False)
base_ns_df.to_csv("out/base_ns_df.csv")
base_ns_evals_df = pd.DataFrame.from_dict(base_ns_evals, orient='index').reset_index()
base_ns_evals_df.to_csv("out/base_ns_evals_df.csv")

########################
### B. History theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)

# Transforms
x["pop_refugee"]=np.log(x["pop_refugee"]+1)

# Categorical
target="d_ns"
inputs=['d_ns_lag1',"d_ns_zeros_decay","d_neighbors_ns_fatalities_lag1",'regime_duration', 'pop_refugee', 'pop_refugee_id']
history_ns_df,history_ns_evals,history_ns_val,history_ns_shap=gen_model(y,x,target,inputs,names=names,name="history_ns",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=['d_ns_lag1',"d_ns_zeros_decay","d_neighbors_ns_fatalities_lag1","regime_duration",'pop_refugee'],downsampling=False)
history_ns_df.to_csv("out/history_ns_df.csv")
history_ns_evals_df = pd.DataFrame.from_dict(history_ns_evals, orient='index').reset_index()
history_ns_evals_df.to_csv("out/history_ns_evals_df.csv")
pd.DataFrame(history_ns_shap[:,:,1]).to_csv("out/history_ns_shap.csv")

###########################
### C. Demography theme ###
###########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_demog_full.csv",index_col=0)

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Categorical
target="d_ns"
inputs=demog_theme
demog_ns_df,demog_ns_evals,demog_ns_val,demog_ns_shap=gen_model(y,x,target,inputs,names=names,name="demog_ns",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=['pop','pop_density','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','monopoly_share','discriminated_share','powerless_share','dominant_share','ethnic_frac','rel_frac','lang_frac','race_frac'],downsampling=False)
demog_ns_df.to_csv("out/demog_ns_df.csv")
demog_ns_evals_df = pd.DataFrame.from_dict(demog_ns_evals, orient='index').reset_index()
demog_ns_evals_df.to_csv("out/demog_ns_evals_df.csv")
pd.DataFrame(demog_ns_shap[:,:,1]).to_csv("out/demog_ns_shap.csv")

########################################
### E. Geography & environment theme ###
########################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_geog_full.csv",index_col=0)

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
preprocess_min_max_group(x,"temp","country")

# Categorical
target="d_ns"
inputs=geog_theme
geog_ns_df,geog_ns_evals,geog_ns_val,geog_ns_shap=gen_model(y,x,target,inputs,names=names,name="geog_ns",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["land","temp","forest","co2","percip","waterstress","agri_land","arable_land","rugged","soil","desert","tropical","cont_africa","cont_asia","d_neighbors_con","no_neigh","d_neighbors_non_dem"],downsampling=False)
geog_ns_df.to_csv("out/geog_ns_df.csv")
geog_ns_evals_df = pd.DataFrame.from_dict(geog_ns_evals, orient='index').reset_index()
geog_ns_evals_df.to_csv("out/geog_ns_evals_df.csv")
pd.DataFrame(geog_ns_shap[:,:,1]).to_csv("out/geog_ns_shap.csv")

########################
### G. Economy theme ###
########################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_econ_full.csv",index_col=0)

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
econ_ns_df,econ_ns_evals,econ_ns_val,econ_ns_shap=gen_model(y,x,target,inputs,names=names,name="econ_ns",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["oil_deposits","oil_production","oil_exports","natres_share","gdp","gni","gdp_growth_norm","unemploy","unemploy_male","inflat","conprice","undernour","foodprod_norm","water_rural","water_urb","agri_share","trade_share","fert","lifeexp_female","lifeexp_male","pop_growth_norm","inf_mort","mig_norm","broadband","telephone","internet_use","mobile"],downsampling=False)
econ_ns_df.to_csv("out/econ_ns_df.csv")
econ_ns_evals_df = pd.DataFrame.from_dict(econ_ns_evals, orient='index').reset_index()
econ_ns_evals_df.to_csv("out/econ_ns_evals_df.csv")
pd.DataFrame(econ_ns_shap[:,:,1]).to_csv("out/econ_ns_shap.csv")

##################################
### H. Regime and policy theme ###
##################################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_pol_full.csv",index_col=0)

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)

# Categorical
target="d_ns"
inputs=pol_theme
pol_ns_df,pol_ns_evals,pol_ns_val,pol_ns_shap=gen_model(y,x,target,inputs,names=names,name="pol_ns",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=["eys","eys_male","eys_female","mys","mys_female","mys_male","armedforces_share","milex_share","corruption","effectiveness","polvio","regu","law","account","tax","polyarchy","libdem","partipdem","delibdem","egaldem","civlib","phyvio","pollib","privlib","execon","exgender","exgeo","expol","exsoc","shutdown","filter","tenure_months","dem_duration","elections","lastelection"],downsampling=False)
pol_ns_df.to_csv("out/pol_ns_df.csv")
pol_ns_evals_df = pd.DataFrame.from_dict(pol_ns_evals, orient='index').reset_index()
pol_ns_evals_df.to_csv("out/pol_ns_evals_df.csv")
pd.DataFrame(pol_ns_shap[:,:,1]).to_csv("out/pol_ns_shap.csv")

################
### Ensemble ###
################

# Calculate weighted average
predictions=pd.concat([history_ns_df.preds_proba,demog_ns_df.preds_proba,geog_ns_df.preds_proba,econ_ns_df.preds_proba,pol_ns_df.preds_proba], axis=1)
weights=[1-history_ns_val["brier"],1-demog_ns_val["brier"],1-geog_ns_val["brier"],1-econ_ns_val["brier"],1-pol_ns_val["brier"]]
weights_ns_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_ns_n = [x / sum(weights_ns_n) for x in weights_ns_n]
ensemble = (history_ns_df.preds_proba*weights_ns_n[0])+(demog_ns_df.preds_proba*weights_ns_n[1])+(geog_ns_df.preds_proba*weights_ns_n[2])+(econ_ns_df.preds_proba*weights_ns_n[3])+(pol_ns_df.preds_proba*weights_ns_n[4])
ensemble_ns=pd.concat([history_ns_df[["country","year","d_ns","test"]],pd.DataFrame(ensemble)],axis=1)
ensemble_ns.columns=["country","year","d_ns","test","preds_proba"]
ensemble_ns=ensemble_ns.reset_index(drop=True)

# Evals
brier = brier_score_loss(ensemble_ns.loc[ensemble_ns["test"]==1].d_ns, ensemble_ns.loc[ensemble_ns["test"]==1].preds_proba)
precision, recall, _ = precision_recall_curve(ensemble_ns.loc[ensemble_ns["test"]==1].d_ns, ensemble_ns.loc[ensemble_ns["test"]==1].preds_proba)
aupr = auc(recall, precision)
auroc = roc_auc_score(ensemble_ns.loc[ensemble_ns["test"]==1].d_ns, ensemble_ns.loc[ensemble_ns["test"]==1].preds_proba)
evals_ns_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_ns_ensemble_df = pd.DataFrame.from_dict(evals_ns_ensemble, orient='index').reset_index()
evals_ns_ensemble_df.to_csv("out/evals_ns_ensemble_df.csv")

###################
### Evaluations ###
###################

print(f"{round(base_ns_evals['aupr'],5)} &  \\\
      {round(base_ns_evals['auroc'],5)} &  \\\
      {round(base_ns_evals['brier'],5)}")
      
print(f"{round(history_ns_evals['aupr'],5)} &  \\\
      {round(history_ns_evals['auroc'],5)} &  \\\
      {round(history_ns_evals['brier'],5)}")
      
print(f"{round(demog_ns_evals['aupr'],5)} &  \\\
      {round(demog_ns_evals['auroc'],5)} &  \\\
      {round(demog_ns_evals['brier'],5)}")

print(f"{round(geog_ns_evals['aupr'],5)} &  \\\
      {round(geog_ns_evals['auroc'],5)} &  \\\
      {round(geog_ns_evals['brier'],5)}")

print(f"{round(econ_ns_evals['aupr'],5)} &  \\\
      {round(econ_ns_evals['auroc'],5)} &  \\\
      {round(econ_ns_evals['brier'],5)}")
           
print(f"{round(pol_ns_evals['aupr'],5)} &  \\\
       {round(pol_ns_evals['auroc'],5)} &  \\\
       {round(pol_ns_evals['brier'],5)}")          
      
print(f"{round(evals_ns_ensemble['aupr'],5)} &  \\\
       {round(evals_ns_ensemble['auroc'],5)} &  \\\
       {round(evals_ns_ensemble['brier'],5)}")  

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
weights.append(1-brier_score_loss(ensemble_protest["d_protest"].loc[(ensemble_protest["year"]>=2013)&(ensemble_protest["year"]<=2016)], ensemble_protest["preds_proba"].loc[(ensemble_protest["year"]>=2013)&(ensemble_protest["year"]<=2016)]))
weights.append(1-brier_score_loss(ensemble_riot["d_riot"].loc[(ensemble_riot["year"]>=2013)&(ensemble_riot["year"]<=2016)],ensemble_riot["preds_proba"].loc[(ensemble_riot["year"]>=2013)&(ensemble_riot["year"]<=2016)]))
weights.append(1-brier_score_loss(ensemble_terror["d_terror"].loc[(ensemble_terror["year"]>=2013)&(ensemble_terror["year"]<=2016)],ensemble_terror["preds_proba"].loc[(ensemble_terror["year"]>=2013)&(ensemble_terror["year"]<=2016)]))
weights.append(1-brier_score_loss(ensemble_sb["d_sb"].loc[(ensemble_sb["year"]>=2013)&(ensemble_sb["year"]<=2016)], ensemble_sb["preds_proba"].loc[(ensemble_sb["year"]>=2013)&(ensemble_sb["year"]<=2016)]))
weights.append(1-brier_score_loss(ensemble_ns["d_ns"].loc[(ensemble_ns["year"]>=2013)&(ensemble_ns["year"]<=2016)], ensemble_ns["preds_proba"].loc[(ensemble_ns["year"]>=2013)&(ensemble_ns["year"]<=2016)]))
weights.append(1-brier_score_loss(ensemble_osv["d_osv"].loc[(ensemble_osv["year"]>=2013)&(ensemble_osv["year"]<=2016)],ensemble_osv["preds_proba"].loc[(ensemble_osv["year"]>=2013)&(ensemble_osv["year"]<=2016)]))
weights_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_n = [x / sum(weights_n) for x in weights_n]

### Protest, riotes, terrorism, sb, ns, osv ###

base=ensemble_protest[["year","country"]]
ensemble_sb_short=pd.merge(base, ensemble_sb,on=["year","country"],how="left")

base=ensemble_protest[["year","country"]]
ensemble_ns_short=pd.merge(base, ensemble_ns,on=["year","country"],how="left")

base=ensemble_protest[["year","country"]]
ensemble_osv_short=pd.merge(base, ensemble_osv,on=["year","country"],how="left")

ensemble = (ensemble_protest.preds_proba*weights_n[0])+(ensemble_riot.preds_proba*weights_n[1])+(ensemble_terror.preds_proba*weights_n[2])+(ensemble_sb_short.preds_proba*weights_n[3])+(ensemble_ns_short.preds_proba*weights_n[4])+(ensemble_osv_short.preds_proba*weights_n[5])
ensemble_ens=pd.concat([ensemble_protest[["country","year"]],pd.DataFrame(ensemble)],axis=1)
ensemble_ens.columns=["country","year","preds_proba"]

### sb, ns, osv  ###
base=ensemble_protest[["year","country"]]
base["drop"] = base['year'].astype(str) + '-' + base['country']
drop=list(base["drop"])

ensemble_sb["id"] = ensemble_sb['year'].astype(str) + '-' + ensemble_sb['country']
ensemble_sb_s = ensemble_sb[~ensemble_sb['id'].isin(drop)]
ensemble_sb_s=ensemble_sb_s.reset_index(drop=True)

ensemble_ns["id"] = ensemble_ns['year'].astype(str) + '-' + ensemble_ns['country']
ensemble_ns_s = ensemble_ns[~ensemble_ns['id'].isin(drop)]
ensemble_ns_s=ensemble_ns_s.reset_index(drop=True)

ensemble_osv["id"] = ensemble_osv['year'].astype(str) + '-' + ensemble_osv['country']
ensemble_osv_s = ensemble_osv[~ensemble_osv['id'].isin(drop)]
ensemble_osv_s=ensemble_osv_s.reset_index(drop=True)

ensemble = (ensemble_sb_s.preds_proba*weights_n[3])+(ensemble_ns_s.preds_proba*weights_n[4])+(ensemble_osv_s.preds_proba*weights_n[5])
ensemble_ens_vio=pd.concat([ensemble_sb_s[["country","year"]],pd.DataFrame(ensemble)],axis=1)
ensemble_ens_vio.columns=["country","year","preds_proba"]

ensemble_ens=pd.concat([ensemble_ens,ensemble_ens_vio],axis=0)
ensemble_ens=ensemble_ens.sort_values(by=["country","year"])

#######################
### Prediction maps ###
#######################
y=pd.read_csv("out/df_out_full.csv",index_col=0)

for year in list(y.year.unique()):
    worldmap = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    worldmap=worldmap.loc[(worldmap["CONTINENT"]!="Antarctica")]    
    worldmap.loc[worldmap["NAME"]=="Bosnia and Herz.","NAME"]='Bosnia-Herzegovina' 
    worldmap.loc[worldmap["NAME"]=="Cambodia","NAME"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["NAME"]=="Central African Rep.","NAME"]='Central African Republic'
    worldmap.loc[worldmap["NAME"]=="Dem. Rep. Congo","NAME"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["NAME"]=="Côte d'Ivoire","NAME"]='Ivory Coast'
    worldmap.loc[worldmap["NAME"]=="Dominican Rep.","NAME"]='Dominican Republic'
    worldmap.loc[worldmap["NAME"]=='Timor-Leste',"NAME"]='East Timor'
    worldmap.loc[worldmap["NAME"]=='Eq. Guinea',"NAME"]='Equatorial Guinea'
    worldmap.loc[worldmap["NAME"]=='Macedonia',"NAME"]='Macedonia, FYR'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Russia',"NAME"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["NAME"]=='S. Sudan',"NAME"]='South Sudan'
    worldmap.loc[worldmap["NAME"]=='Solomon Is.',"NAME"]='Solomon Islands'
    worldmap.loc[worldmap["NAME"]=='Yemen',"NAME"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["NAME"]=='Somaliland',"NAME"]='Somalia'
    worldmap.loc[worldmap["NAME"]=='Palestine',"NAME"]='Israel'
    worldmap.loc[worldmap["NAME"]=='Greenland',"NAME"]='Denmark'
    worldmap.loc[worldmap["NAME"]=='W.Sahara',"NAME"]='Morocco'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["NAME","geometry"]].merge(y[["country","d_sb"]].loc[y["year"]==year], right_on=["country"],left_on=["NAME"],how='left')
    worldmap_m.plot(column="d_sb",ax=ax,cmap="Greys",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    plt.title(f"At least one fatality from civil conflict, {year}", size=25)
    #cmap = plt.cm.get_cmap('Greys')
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="2%", pad=0.01)   
    #fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_actuals_sb_{year}.jpeg",dpi=400,bbox_inches="tight")
    plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_actuals_sb_{year}.jpeg",dpi=400,bbox_inches="tight")
    plt.savefig(f"out/struc_actuals_sb_{year}.jpeg",dpi=400,bbox_inches="tight")

for year in list(y.year.unique()):
    worldmap = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    worldmap=worldmap.loc[(worldmap["CONTINENT"]!="Antarctica")]    
    worldmap.loc[worldmap["NAME"]=="Bosnia and Herz.","NAME"]='Bosnia-Herzegovina' 
    worldmap.loc[worldmap["NAME"]=="Cambodia","NAME"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["NAME"]=="Central African Rep.","NAME"]='Central African Republic'
    worldmap.loc[worldmap["NAME"]=="Dem. Rep. Congo","NAME"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["NAME"]=="Côte d'Ivoire","NAME"]='Ivory Coast'
    worldmap.loc[worldmap["NAME"]=="Dominican Rep.","NAME"]='Dominican Republic'
    worldmap.loc[worldmap["NAME"]=='Timor-Leste',"NAME"]='East Timor'
    worldmap.loc[worldmap["NAME"]=='Eq. Guinea',"NAME"]='Equatorial Guinea'
    worldmap.loc[worldmap["NAME"]=='Macedonia',"NAME"]='Macedonia, FYR'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Russia',"NAME"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["NAME"]=='S. Sudan',"NAME"]='South Sudan'
    worldmap.loc[worldmap["NAME"]=='Solomon Is.',"NAME"]='Solomon Islands'
    worldmap.loc[worldmap["NAME"]=='Yemen',"NAME"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["NAME"]=='Somaliland',"NAME"]='Somalia'
    worldmap.loc[worldmap["NAME"]=='Palestine',"NAME"]='Israel'
    worldmap.loc[worldmap["NAME"]=='Greenland',"NAME"]='Denmark'
    worldmap.loc[worldmap["NAME"]=='W.Sahara',"NAME"]='Morocco'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["NAME","geometry"]].merge(y[["country","d_ns"]].loc[y["year"]==year], right_on=["country"],left_on=["NAME"],how='left')
    worldmap_m.plot(column="d_ns",ax=ax,cmap="Greys",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    plt.title(f"At least one fatality from non-state conflict, {year}", size=25)
    #cmap = plt.cm.get_cmap('Greys')
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="2%", pad=0.01)   
    #fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_actuals_ns_{year}.jpeg",dpi=400,bbox_inches="tight")
    plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_actuals_ns_{year}.jpeg",dpi=400,bbox_inches="tight")
    plt.savefig(f"out/struc_actuals_ns_{year}.jpeg",dpi=400,bbox_inches="tight")

for year in list(y.year.unique()):
    worldmap = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    worldmap=worldmap.loc[(worldmap["CONTINENT"]!="Antarctica")]    
    worldmap.loc[worldmap["NAME"]=="Bosnia and Herz.","NAME"]='Bosnia-Herzegovina' 
    worldmap.loc[worldmap["NAME"]=="Cambodia","NAME"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["NAME"]=="Central African Rep.","NAME"]='Central African Republic'
    worldmap.loc[worldmap["NAME"]=="Dem. Rep. Congo","NAME"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["NAME"]=="Côte d'Ivoire","NAME"]='Ivory Coast'
    worldmap.loc[worldmap["NAME"]=="Dominican Rep.","NAME"]='Dominican Republic'
    worldmap.loc[worldmap["NAME"]=='Timor-Leste',"NAME"]='East Timor'
    worldmap.loc[worldmap["NAME"]=='Eq. Guinea',"NAME"]='Equatorial Guinea'
    worldmap.loc[worldmap["NAME"]=='Macedonia',"NAME"]='Macedonia, FYR'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Russia',"NAME"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["NAME"]=='S. Sudan',"NAME"]='South Sudan'
    worldmap.loc[worldmap["NAME"]=='Solomon Is.',"NAME"]='Solomon Islands'
    worldmap.loc[worldmap["NAME"]=='Yemen',"NAME"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["NAME"]=='Somaliland',"NAME"]='Somalia'
    worldmap.loc[worldmap["NAME"]=='Palestine',"NAME"]='Israel'
    worldmap.loc[worldmap["NAME"]=='Greenland',"NAME"]='Denmark'
    worldmap.loc[worldmap["NAME"]=='W.Sahara',"NAME"]='Morocco'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["NAME","geometry"]].merge(y[["country","d_osv"]].loc[y["year"]==year], right_on=["country"],left_on=["NAME"],how='left')
    worldmap_m.plot(column="d_osv",ax=ax,cmap="Greys",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    plt.title(f"At least one fatality from one-sided violence, {year}", size=25)
    #cmap = plt.cm.get_cmap('Greys')
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="2%", pad=0.01)   
    #fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_actuals_osv_{year}.jpeg",dpi=400,bbox_inches="tight")
    plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_actuals_osv_{year}.jpeg",dpi=400,bbox_inches="tight")
    plt.savefig(f"out/struc_actuals_osv_{year}.jpeg",dpi=400,bbox_inches="tight")
                
for year in list(y.year.unique()):
    worldmap = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    worldmap=worldmap.loc[(worldmap["CONTINENT"]!="Antarctica")]    
    worldmap.loc[worldmap["NAME"]=="Bosnia and Herz.","NAME"]='Bosnia-Herzegovina' 
    worldmap.loc[worldmap["NAME"]=="Cambodia","NAME"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["NAME"]=="Central African Rep.","NAME"]='Central African Republic'
    worldmap.loc[worldmap["NAME"]=="Dem. Rep. Congo","NAME"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["NAME"]=="Côte d'Ivoire","NAME"]='Ivory Coast'
    worldmap.loc[worldmap["NAME"]=="Dominican Rep.","NAME"]='Dominican Republic'
    worldmap.loc[worldmap["NAME"]=='Timor-Leste',"NAME"]='East Timor'
    worldmap.loc[worldmap["NAME"]=='Eq. Guinea',"NAME"]='Equatorial Guinea'
    worldmap.loc[worldmap["NAME"]=='Macedonia',"NAME"]='Macedonia, FYR'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Russia',"NAME"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["NAME"]=='S. Sudan',"NAME"]='South Sudan'
    worldmap.loc[worldmap["NAME"]=='Solomon Is.',"NAME"]='Solomon Islands'
    worldmap.loc[worldmap["NAME"]=='Yemen',"NAME"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["NAME"]=='Somaliland',"NAME"]='Somalia'
    worldmap.loc[worldmap["NAME"]=='Palestine',"NAME"]='Israel'
    worldmap.loc[worldmap["NAME"]=='Greenland',"NAME"]='Denmark'
    worldmap.loc[worldmap["NAME"]=='W.Sahara',"NAME"]='Morocco'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["NAME","geometry"]].merge(y[["country","d_protest"]].loc[y["year"]==year], right_on=["country"],left_on=["NAME"],how='left')
    worldmap_m.plot(column="d_protest",ax=ax,cmap="Greys",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    plt.title(f"At least 25 protest events, {year}", size=25)
    #cmap = plt.cm.get_cmap('Blues')
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="2%", pad=0.01)   
    #fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_actuals_protest_{year}.jpeg",dpi=400,bbox_inches="tight")
    plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_actuals_protest_{year}.jpeg",dpi=400,bbox_inches="tight")
    plt.savefig(f"out/struc_actuals_protest_{year}.jpeg",dpi=400,bbox_inches="tight")
     
for year in list(y.year.unique()):
    worldmap = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    worldmap=worldmap.loc[(worldmap["CONTINENT"]!="Antarctica")]    
    worldmap.loc[worldmap["NAME"]=="Bosnia and Herz.","NAME"]='Bosnia-Herzegovina' 
    worldmap.loc[worldmap["NAME"]=="Cambodia","NAME"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["NAME"]=="Central African Rep.","NAME"]='Central African Republic'
    worldmap.loc[worldmap["NAME"]=="Dem. Rep. Congo","NAME"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["NAME"]=="Côte d'Ivoire","NAME"]='Ivory Coast'
    worldmap.loc[worldmap["NAME"]=="Dominican Rep.","NAME"]='Dominican Republic'
    worldmap.loc[worldmap["NAME"]=='Timor-Leste',"NAME"]='East Timor'
    worldmap.loc[worldmap["NAME"]=='Eq. Guinea',"NAME"]='Equatorial Guinea'
    worldmap.loc[worldmap["NAME"]=='Macedonia',"NAME"]='Macedonia, FYR'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Russia',"NAME"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["NAME"]=='S. Sudan',"NAME"]='South Sudan'
    worldmap.loc[worldmap["NAME"]=='Solomon Is.',"NAME"]='Solomon Islands'
    worldmap.loc[worldmap["NAME"]=='Yemen',"NAME"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["NAME"]=='Somaliland',"NAME"]='Somalia'
    worldmap.loc[worldmap["NAME"]=='Palestine',"NAME"]='Israel'
    worldmap.loc[worldmap["NAME"]=='Greenland',"NAME"]='Denmark'
    worldmap.loc[worldmap["NAME"]=='W.Sahara',"NAME"]='Morocco'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["NAME","geometry"]].merge(y[["country","d_riot"]].loc[y["year"]==year], right_on=["country"],left_on=["NAME"],how='left')
    worldmap_m.plot(column="d_riot",ax=ax,cmap="Greys",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    plt.title(f"At least 25 riot events, {year}", size=25)
    #cmap = plt.cm.get_cmap('Blues')
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="2%", pad=0.01)   
    #fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_actuals_riot_{year}.jpeg",dpi=400,bbox_inches="tight")
    plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_actuals_riot_{year}.jpeg",dpi=400,bbox_inches="tight")
    plt.savefig(f"out/struc_actuals_riot_{year}.jpeg",dpi=400,bbox_inches="tight")
     
for year in list(y.year.unique()):
    worldmap = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    worldmap=worldmap.loc[(worldmap["CONTINENT"]!="Antarctica")]    
    worldmap.loc[worldmap["NAME"]=="Bosnia and Herz.","NAME"]='Bosnia-Herzegovina' 
    worldmap.loc[worldmap["NAME"]=="Cambodia","NAME"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["NAME"]=="Central African Rep.","NAME"]='Central African Republic'
    worldmap.loc[worldmap["NAME"]=="Dem. Rep. Congo","NAME"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["NAME"]=="Côte d'Ivoire","NAME"]='Ivory Coast'
    worldmap.loc[worldmap["NAME"]=="Dominican Rep.","NAME"]='Dominican Republic'
    worldmap.loc[worldmap["NAME"]=='Timor-Leste',"NAME"]='East Timor'
    worldmap.loc[worldmap["NAME"]=='Eq. Guinea',"NAME"]='Equatorial Guinea'
    worldmap.loc[worldmap["NAME"]=='Macedonia',"NAME"]='Macedonia, FYR'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Russia',"NAME"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["NAME"]=='S. Sudan',"NAME"]='South Sudan'
    worldmap.loc[worldmap["NAME"]=='Solomon Is.',"NAME"]='Solomon Islands'
    worldmap.loc[worldmap["NAME"]=='Yemen',"NAME"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["NAME"]=='Somaliland',"NAME"]='Somalia'
    worldmap.loc[worldmap["NAME"]=='Palestine',"NAME"]='Israel'
    worldmap.loc[worldmap["NAME"]=='Greenland',"NAME"]='Denmark'
    worldmap.loc[worldmap["NAME"]=='W.Sahara',"NAME"]='Morocco'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["NAME","geometry"]].merge(y[["country","d_remote"]].loc[y["year"]==year], right_on=["country"],left_on=["NAME"],how='left')
    worldmap_m.plot(column="d_remote",ax=ax,cmap="Greys",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    plt.title(f"At least one fatality from remote violence, {year}", size=25)
    #cmap = plt.cm.get_cmap('Blues')
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="2%", pad=0.01)   
    #fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_actuals_terror_{year}.jpeg",dpi=400,bbox_inches="tight")
    plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_actuals_terror_{year}.jpeg",dpi=400,bbox_inches="tight")
    plt.savefig(f"out/struc_actuals_terror_{year}.jpeg",dpi=400,bbox_inches="tight")
     
### Predictions ###
for year in list(ensemble_war.year.unique()):
    worldmap = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    worldmap=worldmap.loc[(worldmap["CONTINENT"]!="Antarctica")]    
    worldmap.loc[worldmap["NAME"]=="Bosnia and Herz.","NAME"]='Bosnia-Herzegovina' 
    worldmap.loc[worldmap["NAME"]=="Cambodia","NAME"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["NAME"]=="Central African Rep.","NAME"]='Central African Republic'
    worldmap.loc[worldmap["NAME"]=="Dem. Rep. Congo","NAME"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["NAME"]=="Côte d'Ivoire","NAME"]='Ivory Coast'
    worldmap.loc[worldmap["NAME"]=="Dominican Rep.","NAME"]='Dominican Republic'
    worldmap.loc[worldmap["NAME"]=='Timor-Leste',"NAME"]='East Timor'
    worldmap.loc[worldmap["NAME"]=='Eq. Guinea',"NAME"]='Equatorial Guinea'
    worldmap.loc[worldmap["NAME"]=='Macedonia',"NAME"]='Macedonia, FYR'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Russia',"NAME"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["NAME"]=='S. Sudan',"NAME"]='South Sudan'
    worldmap.loc[worldmap["NAME"]=='Solomon Is.',"NAME"]='Solomon Islands'
    worldmap.loc[worldmap["NAME"]=='Yemen',"NAME"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["NAME"]=='Somaliland',"NAME"]='Somalia'
    worldmap.loc[worldmap["NAME"]=='Palestine',"NAME"]='Israel'
    worldmap.loc[worldmap["NAME"]=='Greenland',"NAME"]='Denmark'
    worldmap.loc[worldmap["NAME"]=='W.Sahara',"NAME"]='Morocco'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["NAME","geometry"]].merge(ensemble_war[["country","preds_proba"]].loc[ensemble_war["year"]==year], right_on=["country"],left_on=["NAME"],how='left')
    worldmap_m.plot(column="preds_proba",ax=ax,cmap="Blues",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    plt.title(f"Prediction of 1,000 fatalities from civil conflict in {year}", size=25)
    cmap = plt.cm.get_cmap('Blues')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.01)   
    fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_map_preds_war_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_map_preds_war_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"out/struc_map_preds_war_{year}.jpeg",dpi=400,bbox_inches="tight")
     
for year in list(ensemble_conflict.year.unique()):
    worldmap = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    worldmap=worldmap.loc[(worldmap["CONTINENT"]!="Antarctica")]    
    worldmap.loc[worldmap["NAME"]=="Bosnia and Herz.","NAME"]='Bosnia-Herzegovina' 
    worldmap.loc[worldmap["NAME"]=="Cambodia","NAME"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["NAME"]=="Central African Rep.","NAME"]='Central African Republic'
    worldmap.loc[worldmap["NAME"]=="Dem. Rep. Congo","NAME"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["NAME"]=="Côte d'Ivoire","NAME"]='Ivory Coast'
    worldmap.loc[worldmap["NAME"]=="Dominican Rep.","NAME"]='Dominican Republic'
    worldmap.loc[worldmap["NAME"]=='Timor-Leste',"NAME"]='East Timor'
    worldmap.loc[worldmap["NAME"]=='Eq. Guinea',"NAME"]='Equatorial Guinea'
    worldmap.loc[worldmap["NAME"]=='Macedonia',"NAME"]='Macedonia, FYR'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Russia',"NAME"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["NAME"]=='S. Sudan',"NAME"]='South Sudan'
    worldmap.loc[worldmap["NAME"]=='Solomon Is.',"NAME"]='Solomon Islands'
    worldmap.loc[worldmap["NAME"]=='Yemen',"NAME"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["NAME"]=='Somaliland',"NAME"]='Somalia'
    worldmap.loc[worldmap["NAME"]=='Palestine',"NAME"]='Israel'
    worldmap.loc[worldmap["NAME"]=='Greenland',"NAME"]='Denmark'
    worldmap.loc[worldmap["NAME"]=='W.Sahara',"NAME"]='Morocco'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["NAME","geometry"]].merge(ensemble_conflict[["country","preds_proba"]].loc[ensemble_war["year"]==year], right_on=["country"],left_on=["NAME"],how='left')
    worldmap_m.plot(column="preds_proba",ax=ax,cmap="Blues",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    plt.title(f"Prediction of 25 fatalities from civil conflict in {year}", size=25)
    cmap = plt.cm.get_cmap('Blues')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.01)   
    fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_map_preds_conflict_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_map_preds_conflict_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"out/struc_map_preds_conflict_{year}.jpeg",dpi=400,bbox_inches="tight")
     
for year in list(ensemble_protest.year.unique()):
    worldmap = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    worldmap=worldmap.loc[(worldmap["CONTINENT"]!="Antarctica")]    
    worldmap.loc[worldmap["NAME"]=="Bosnia and Herz.","NAME"]='Bosnia-Herzegovina' 
    worldmap.loc[worldmap["NAME"]=="Cambodia","NAME"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["NAME"]=="Central African Rep.","NAME"]='Central African Republic'
    worldmap.loc[worldmap["NAME"]=="Dem. Rep. Congo","NAME"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["NAME"]=="Côte d'Ivoire","NAME"]='Ivory Coast'
    worldmap.loc[worldmap["NAME"]=="Dominican Rep.","NAME"]='Dominican Republic'
    worldmap.loc[worldmap["NAME"]=='Timor-Leste',"NAME"]='East Timor'
    worldmap.loc[worldmap["NAME"]=='Eq. Guinea',"NAME"]='Equatorial Guinea'
    worldmap.loc[worldmap["NAME"]=='Macedonia',"NAME"]='Macedonia, FYR'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Russia',"NAME"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["NAME"]=='S. Sudan',"NAME"]='South Sudan'
    worldmap.loc[worldmap["NAME"]=='Solomon Is.',"NAME"]='Solomon Islands'
    worldmap.loc[worldmap["NAME"]=='Yemen',"NAME"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["NAME"]=='Somaliland',"NAME"]='Somalia'
    worldmap.loc[worldmap["NAME"]=='Palestine',"NAME"]='Israel'
    worldmap.loc[worldmap["NAME"]=='Greenland',"NAME"]='Denmark'
    worldmap.loc[worldmap["NAME"]=='W.Sahara',"NAME"]='Morocco'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["NAME","geometry"]].merge(ensemble_protest[["country","preds_proba"]].loc[ensemble_war["year"]==year], right_on=["country"],left_on=["NAME"],how='left')
    worldmap_m.plot(column="preds_proba",ax=ax,cmap="Blues",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    plt.title(f"Prediction of 25 protest events in {year}", size=25)
    cmap = plt.cm.get_cmap('Blues')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.01)   
    fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_map_preds_protest_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_map_preds_protest_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"out/struc_map_preds_protest_{year}.jpeg",dpi=400,bbox_inches="tight")
     
for year in list(ensemble_riot.year.unique()):
    worldmap = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    worldmap=worldmap.loc[(worldmap["CONTINENT"]!="Antarctica")]    
    worldmap.loc[worldmap["NAME"]=="Bosnia and Herz.","NAME"]='Bosnia-Herzegovina' 
    worldmap.loc[worldmap["NAME"]=="Cambodia","NAME"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["NAME"]=="Central African Rep.","NAME"]='Central African Republic'
    worldmap.loc[worldmap["NAME"]=="Dem. Rep. Congo","NAME"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["NAME"]=="Côte d'Ivoire","NAME"]='Ivory Coast'
    worldmap.loc[worldmap["NAME"]=="Dominican Rep.","NAME"]='Dominican Republic'
    worldmap.loc[worldmap["NAME"]=='Timor-Leste',"NAME"]='East Timor'
    worldmap.loc[worldmap["NAME"]=='Eq. Guinea',"NAME"]='Equatorial Guinea'
    worldmap.loc[worldmap["NAME"]=='Macedonia',"NAME"]='Macedonia, FYR'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Russia',"NAME"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["NAME"]=='S. Sudan',"NAME"]='South Sudan'
    worldmap.loc[worldmap["NAME"]=='Solomon Is.',"NAME"]='Solomon Islands'
    worldmap.loc[worldmap["NAME"]=='Yemen',"NAME"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["NAME"]=='Somaliland',"NAME"]='Somalia'
    worldmap.loc[worldmap["NAME"]=='Palestine',"NAME"]='Israel'
    worldmap.loc[worldmap["NAME"]=='Greenland',"NAME"]='Denmark'
    worldmap.loc[worldmap["NAME"]=='W.Sahara',"NAME"]='Morocco'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["NAME","geometry"]].merge(ensemble_riot[["country","preds_proba"]].loc[ensemble_war["year"]==year], right_on=["country"],left_on=["NAME"],how='left')
    worldmap_m.plot(column="preds_proba",ax=ax,cmap="Blues",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    plt.title(f"Prediction of 25 riot events in {year}", size=25)
    cmap = plt.cm.get_cmap('Blues')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.01)   
    fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_map_preds_riot_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_map_preds_riot_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"out/struc_map_preds_riot_{year}.jpeg",dpi=400,bbox_inches="tight")
     
for year in list(ensemble_terror.year.unique()):
    worldmap = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    worldmap=worldmap.loc[(worldmap["CONTINENT"]!="Antarctica")]    
    worldmap.loc[worldmap["NAME"]=="Bosnia and Herz.","NAME"]='Bosnia-Herzegovina' 
    worldmap.loc[worldmap["NAME"]=="Cambodia","NAME"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["NAME"]=="Central African Rep.","NAME"]='Central African Republic'
    worldmap.loc[worldmap["NAME"]=="Dem. Rep. Congo","NAME"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["NAME"]=="Côte d'Ivoire","NAME"]='Ivory Coast'
    worldmap.loc[worldmap["NAME"]=="Dominican Rep.","NAME"]='Dominican Republic'
    worldmap.loc[worldmap["NAME"]=='Timor-Leste',"NAME"]='East Timor'
    worldmap.loc[worldmap["NAME"]=='Eq. Guinea',"NAME"]='Equatorial Guinea'
    worldmap.loc[worldmap["NAME"]=='Macedonia',"NAME"]='Macedonia, FYR'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Russia',"NAME"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["NAME"]=='S. Sudan',"NAME"]='South Sudan'
    worldmap.loc[worldmap["NAME"]=='Solomon Is.',"NAME"]='Solomon Islands'
    worldmap.loc[worldmap["NAME"]=='Yemen',"NAME"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["NAME"]=='Somaliland',"NAME"]='Somalia'
    worldmap.loc[worldmap["NAME"]=='Palestine',"NAME"]='Israel'
    worldmap.loc[worldmap["NAME"]=='Greenland',"NAME"]='Denmark'
    worldmap.loc[worldmap["NAME"]=='W.Sahara',"NAME"]='Morocco'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["NAME","geometry"]].merge(ensemble_terror[["country","preds_proba"]].loc[ensemble_war["year"]==year], right_on=["country"],left_on=["NAME"],how='left')
    worldmap_m.plot(column="preds_proba",ax=ax,cmap="Blues",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    plt.title(f"Prediction of at least one fatality from terrorism in {year}", size=25)
    cmap = plt.cm.get_cmap('Blues')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.01)   
    fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_map_preds_terror_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_map_preds_terror_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"out/struc_map_preds_terror_{year}.jpeg",dpi=400,bbox_inches="tight")
     
for year in list(ensemble_sb.year.unique()):
    worldmap = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    worldmap=worldmap.loc[(worldmap["CONTINENT"]!="Antarctica")]    
    worldmap.loc[worldmap["NAME"]=="Bosnia and Herz.","NAME"]='Bosnia-Herzegovina' 
    worldmap.loc[worldmap["NAME"]=="Cambodia","NAME"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["NAME"]=="Central African Rep.","NAME"]='Central African Republic'
    worldmap.loc[worldmap["NAME"]=="Dem. Rep. Congo","NAME"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["NAME"]=="Côte d'Ivoire","NAME"]='Ivory Coast'
    worldmap.loc[worldmap["NAME"]=="Dominican Rep.","NAME"]='Dominican Republic'
    worldmap.loc[worldmap["NAME"]=='Timor-Leste',"NAME"]='East Timor'
    worldmap.loc[worldmap["NAME"]=='Eq. Guinea',"NAME"]='Equatorial Guinea'
    worldmap.loc[worldmap["NAME"]=='Macedonia',"NAME"]='Macedonia, FYR'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Russia',"NAME"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["NAME"]=='S. Sudan',"NAME"]='South Sudan'
    worldmap.loc[worldmap["NAME"]=='Solomon Is.',"NAME"]='Solomon Islands'
    worldmap.loc[worldmap["NAME"]=='Yemen',"NAME"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["NAME"]=='Somaliland',"NAME"]='Somalia'
    worldmap.loc[worldmap["NAME"]=='Palestine',"NAME"]='Israel'
    worldmap.loc[worldmap["NAME"]=='Greenland',"NAME"]='Denmark'
    worldmap.loc[worldmap["NAME"]=='W.Sahara',"NAME"]='Morocco'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["NAME","geometry"]].merge(ensemble_sb[["country","preds_proba"]].loc[ensemble_war["year"]==year], right_on=["country"],left_on=["NAME"],how='left')
    worldmap_m.plot(column="preds_proba",ax=ax,cmap="Blues",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    plt.title(f"Prediction of at least one fatality from civil conflict in {year}", size=25)
    cmap = plt.cm.get_cmap('Blues')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.01)   
    fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_map_preds_sb_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_map_preds_sb_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"out/struc_map_preds_sb_{year}.jpeg",dpi=400,bbox_inches="tight")
     
for year in list(ensemble_ns.year.unique()):
    worldmap = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    worldmap=worldmap.loc[(worldmap["CONTINENT"]!="Antarctica")]    
    worldmap.loc[worldmap["NAME"]=="Bosnia and Herz.","NAME"]='Bosnia-Herzegovina' 
    worldmap.loc[worldmap["NAME"]=="Cambodia","NAME"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["NAME"]=="Central African Rep.","NAME"]='Central African Republic'
    worldmap.loc[worldmap["NAME"]=="Dem. Rep. Congo","NAME"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["NAME"]=="Côte d'Ivoire","NAME"]='Ivory Coast'
    worldmap.loc[worldmap["NAME"]=="Dominican Rep.","NAME"]='Dominican Republic'
    worldmap.loc[worldmap["NAME"]=='Timor-Leste',"NAME"]='East Timor'
    worldmap.loc[worldmap["NAME"]=='Eq. Guinea',"NAME"]='Equatorial Guinea'
    worldmap.loc[worldmap["NAME"]=='Macedonia',"NAME"]='Macedonia, FYR'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Russia',"NAME"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["NAME"]=='S. Sudan',"NAME"]='South Sudan'
    worldmap.loc[worldmap["NAME"]=='Solomon Is.',"NAME"]='Solomon Islands'
    worldmap.loc[worldmap["NAME"]=='Yemen',"NAME"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["NAME"]=='Somaliland',"NAME"]='Somalia'
    worldmap.loc[worldmap["NAME"]=='Palestine',"NAME"]='Israel'
    worldmap.loc[worldmap["NAME"]=='Greenland',"NAME"]='Denmark'
    worldmap.loc[worldmap["NAME"]=='W.Sahara',"NAME"]='Morocco'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["NAME","geometry"]].merge(ensemble_ns[["country","preds_proba"]].loc[ensemble_war["year"]==year], right_on=["country"],left_on=["NAME"],how='left')
    worldmap_m.plot(column="preds_proba",ax=ax,cmap="Blues",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    plt.title(f"Prediction of at least one fatality from non-state conflict in {year}", size=25)
    cmap = plt.cm.get_cmap('Blues')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.01)   
    fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_map_preds_ns_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_map_preds_ns_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"out/struc_map_preds_ns_{year}.jpeg",dpi=400,bbox_inches="tight")
     
for year in list(ensemble_osv.year.unique()):
    worldmap = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    worldmap=worldmap.loc[(worldmap["CONTINENT"]!="Antarctica")]    
    worldmap.loc[worldmap["NAME"]=="Bosnia and Herz.","NAME"]='Bosnia-Herzegovina' 
    worldmap.loc[worldmap["NAME"]=="Cambodia","NAME"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["NAME"]=="Central African Rep.","NAME"]='Central African Republic'
    worldmap.loc[worldmap["NAME"]=="Dem. Rep. Congo","NAME"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["NAME"]=="Côte d'Ivoire","NAME"]='Ivory Coast'
    worldmap.loc[worldmap["NAME"]=="Dominican Rep.","NAME"]='Dominican Republic'
    worldmap.loc[worldmap["NAME"]=='Timor-Leste',"NAME"]='East Timor'
    worldmap.loc[worldmap["NAME"]=='Eq. Guinea',"NAME"]='Equatorial Guinea'
    worldmap.loc[worldmap["NAME"]=='Macedonia',"NAME"]='Macedonia, FYR'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Russia',"NAME"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["NAME"]=='S. Sudan',"NAME"]='South Sudan'
    worldmap.loc[worldmap["NAME"]=='Solomon Is.',"NAME"]='Solomon Islands'
    worldmap.loc[worldmap["NAME"]=='Yemen',"NAME"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["NAME"]=='Somaliland',"NAME"]='Somalia'
    worldmap.loc[worldmap["NAME"]=='Palestine',"NAME"]='Israel'
    worldmap.loc[worldmap["NAME"]=='Greenland',"NAME"]='Denmark'
    worldmap.loc[worldmap["NAME"]=='W.Sahara',"NAME"]='Morocco'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["NAME","geometry"]].merge(ensemble_osv[["country","preds_proba"]].loc[ensemble_war["year"]==year], right_on=["country"],left_on=["NAME"],how='left')
    worldmap_m.plot(column="preds_proba",ax=ax,cmap="Blues",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    plt.title(f"Prediction of at least one fatality from one-sided violence in {year}", size=25)
    cmap = plt.cm.get_cmap('Blues')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.01)   
    fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_map_preds_osv_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_map_preds_osv_{year}.jpeg",dpi=400,bbox_inches="tight")
    #plt.savefig(f"out/struc_map_preds_osv_{year}.jpeg",dpi=400,bbox_inches="tight")
 
    
ensemble_ens["dummy"]=0
ensemble_ens.loc[ensemble_ens["preds_proba"]>=0.5,"dummy"]=1
ensemble_ens.to_csv("out/ensemble_ens_df.csv")
    
for year in list(ensemble_ens.year.unique()):
    worldmap = gpd.read_file('data/ne_110m_admin_0_countries.shp')
    worldmap['centroid'] =  worldmap.geometry.centroid
    worldmap=worldmap.loc[(worldmap["CONTINENT"]!="Antarctica")]    
    worldmap.loc[worldmap["NAME"]=="Bosnia and Herz.","NAME"]='Bosnia-Herzegovina' 
    worldmap.loc[worldmap["NAME"]=="Cambodia","NAME"]='Cambodia (Kampuchea)'
    worldmap.loc[worldmap["NAME"]=="Central African Rep.","NAME"]='Central African Republic'
    worldmap.loc[worldmap["NAME"]=="Dem. Rep. Congo","NAME"]='DR Congo (Zaire)'
    worldmap.loc[worldmap["NAME"]=="Côte d'Ivoire","NAME"]='Ivory Coast'
    worldmap.loc[worldmap["NAME"]=="Dominican Rep.","NAME"]='Dominican Republic'
    worldmap.loc[worldmap["NAME"]=='Timor-Leste',"NAME"]='East Timor'
    worldmap.loc[worldmap["NAME"]=='Eq. Guinea',"NAME"]='Equatorial Guinea'
    worldmap.loc[worldmap["NAME"]=='Macedonia',"NAME"]='Macedonia, FYR'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Myanmar',"NAME"]='Myanmar (Burma)'
    worldmap.loc[worldmap["NAME"]=='Russia',"NAME"]='Russia (Soviet Union)'
    worldmap.loc[worldmap["NAME"]=='S. Sudan',"NAME"]='South Sudan'
    worldmap.loc[worldmap["NAME"]=='Solomon Is.',"NAME"]='Solomon Islands'
    worldmap.loc[worldmap["NAME"]=='Yemen',"NAME"]='Yemen (North Yemen)'
    worldmap.loc[worldmap["NAME"]=='Somaliland',"NAME"]='Somalia'
    worldmap.loc[worldmap["NAME"]=='Palestine',"NAME"]='Israel'
    worldmap.loc[worldmap["NAME"]=='Greenland',"NAME"]='Denmark'
    worldmap.loc[worldmap["NAME"]=='W.Sahara',"NAME"]='Morocco'
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.axis('off')
    worldmap.boundary.plot(color="black",ax=ax,linewidth=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    worldmap_m = worldmap[["NAME","geometry",'centroid']].merge(ensemble_ens[["country","preds_proba","dummy"]].loc[ensemble_ens["year"]==year], right_on=["country"],left_on=["NAME"],how='left')
    worldmap_m.plot(column="preds_proba",ax=ax,cmap="Greens",norm=norm,missing_kwds={"color": "white","edgecolor": "gray","hatch": "///","label": "Missing values"})
    worldmap_m_s=worldmap_m.loc[worldmap_m["dummy"]==1]
    ax.scatter(worldmap_m_s['centroid'].x, worldmap_m_s['centroid'].y, color='black', marker='o', s=20)
    plt.title(f"Predicted probability of collective action in {year}", size=25)
    cmap = plt.cm.get_cmap('Greens')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.01)   
    fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_map_preds_ens_{year}.jpeg",dpi=400,bbox_inches="tight")
    plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_map_preds_ens_{year}.jpeg",dpi=400,bbox_inches="tight")
    plt.savefig(f"out/struc_map_preds_ens_{year}.jpeg",dpi=400,bbox_inches="tight")
         
############
### Plot ###
############

base_war_evals=pd.read_csv("out/base_war_evals_df.csv",index_col=0)
history_war_evals=pd.read_csv("out/history_war_evals_df.csv",index_col=0)
demog_war_evals=pd.read_csv("out/demog_war_evals_df.csv",index_col=0)
geog_war_evals=pd.read_csv("out/geog_war_evals_df.csv",index_col=0)
econ_war_evals=pd.read_csv("out/econ_war_evals_df.csv",index_col=0)
pol_war_evals=pd.read_csv("out/pol_war_evals_df.csv",index_col=0)
evals_war_ensemble_df=pd.read_csv("out/evals_war_ensemble_df.csv",index_col=0)
      
base_conflict_evals=pd.read_csv("out/base_conflict_evals_df.csv",index_col=0)
history_conflict_evals=pd.read_csv("out/history_conflict_evals_df.csv",index_col=0)
demog_conflict_evals=pd.read_csv("out/demog_conflict_evals_df.csv",index_col=0)
geog_conflict_evals=pd.read_csv("out/geog_conflict_evals_df.csv",index_col=0)
econ_conflict_evals=pd.read_csv("out/econ_conflict_evals_df.csv",index_col=0)
pol_conflict_evals=pd.read_csv("out/pol_conflict_evals_df.csv",index_col=0)
evals_conflict_ensemble_df=pd.read_csv("out/evals_conflict_ensemble_df.csv",index_col=0)

base_protest_evals=pd.read_csv("out/base_protest_evals_df.csv",index_col=0)
history_protest_evals=pd.read_csv("out/history_protest_evals_df.csv",index_col=0)
demog_protest_evals=pd.read_csv("out/demog_protest_evals_df.csv",index_col=0)
geog_protest_evals=pd.read_csv("out/geog_protest_evals_df.csv",index_col=0)
econ_protest_evals=pd.read_csv("out/econ_protest_evals_df.csv",index_col=0)
pol_protest_evals=pd.read_csv("out/pol_protest_evals_df.csv",index_col=0)
evals_protest_ensemble_df=pd.read_csv("out/evals_protest_ensemble_df.csv",index_col=0)

base_riot_evals=pd.read_csv("out/base_riot_evals_df.csv",index_col=0)
history_riot_evals=pd.read_csv("out/history_riot_evals_df.csv",index_col=0)
demog_riot_evals=pd.read_csv("out/demog_riot_evals_df.csv",index_col=0)
geog_riot_evals=pd.read_csv("out/geog_riot_evals_df.csv",index_col=0)
econ_riot_evals=pd.read_csv("out/econ_riot_evals_df.csv",index_col=0)
pol_riot_evals=pd.read_csv("out/pol_riot_evals_df.csv",index_col=0)
evals_riot_ensemble_df=pd.read_csv("out/evals_riot_ensemble_df.csv",index_col=0)

base_terror_evals=pd.read_csv("out/base_terror_evals_df.csv",index_col=0)
history_terror_evals=pd.read_csv("out/history_terror_evals_df.csv",index_col=0)
demog_terror_evals=pd.read_csv("out/demog_terror_evals_df.csv",index_col=0)
geog_terror_evals=pd.read_csv("out/geog_terror_evals_df.csv",index_col=0)
econ_terror_evals=pd.read_csv("out/econ_terror_evals_df.csv",index_col=0)
pol_terror_evals=pd.read_csv("out/pol_terror_evals_df.csv",index_col=0)
evals_terror_ensemble_df=pd.read_csv("out/evals_terror_ensemble_df.csv",index_col=0)

base_sb_evals=pd.read_csv("out/base_sb_evals_df.csv",index_col=0)
history_sb_evals=pd.read_csv("out/history_sb_evals_df.csv",index_col=0)
demog_sb_evals=pd.read_csv("out/demog_sb_evals_df.csv",index_col=0)
geog_sb_evals=pd.read_csv("out/geog_sb_evals_df.csv",index_col=0)
econ_sb_evals=pd.read_csv("out/econ_sb_evals_df.csv",index_col=0)
pol_sb_evals=pd.read_csv("out/pol_sb_evals_df.csv",index_col=0)
evals_sb_ensemble_df=pd.read_csv("out/evals_sb_ensemble_df.csv",index_col=0)

base_ns_evals=pd.read_csv("out/base_ns_evals_df.csv",index_col=0)
history_ns_evals=pd.read_csv("out/history_ns_evals_df.csv",index_col=0)
demog_ns_evals=pd.read_csv("out/demog_ns_evals_df.csv",index_col=0)
geog_ns_evals=pd.read_csv("out/geog_ns_evals_df.csv",index_col=0)
econ_ns_evals=pd.read_csv("out/econ_ns_evals_df.csv",index_col=0)
pol_ns_evals=pd.read_csv("out/pol_ns_evals_df.csv",index_col=0)
evals_ns_ensemble_df=pd.read_csv("out/evals_ns_ensemble_df.csv",index_col=0)

base_osv_evals=pd.read_csv("out/base_osv_evals_df.csv",index_col=0)
history_osv_evals=pd.read_csv("out/history_osv_evals_df.csv",index_col=0)
demog_osv_evals=pd.read_csv("out/demog_osv_evals_df.csv",index_col=0)
geog_osv_evals=pd.read_csv("out/geog_osv_evals_df.csv",index_col=0)
econ_osv_evals=pd.read_csv("out/econ_osv_evals_df.csv",index_col=0)
pol_osv_evals=pd.read_csv("out/pol_osv_evals_df.csv",index_col=0)
evals_osv_ensemble_df=pd.read_csv("out/evals_osv_ensemble_df.csv",index_col=0)

#############
### AUROC ###
#############

#colormap = plt.get_cmap("terrain")
#values = np.linspace(0, 1, 9)
#colors = [colormap(value) for value in values]
colors=["black","gray","forestgreen","lightgreen","steelblue","lightblue","purple","violet"]

fig, ax1 = plt.subplots(figsize=(12,8))

ax1.plot([0,2,4,6,8,10,12],[base_war_evals["0"].iloc[2],history_war_evals["0"].iloc[2],demog_war_evals["0"].iloc[2],geog_war_evals["0"].iloc[2],econ_war_evals["0"].iloc[2],pol_war_evals["0"].iloc[2],evals_war_ensemble_df["0"].iloc[2]], marker='o',color=colors[0],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10,12],[base_conflict_evals["0"].iloc[2],history_conflict_evals["0"].iloc[2],demog_conflict_evals["0"].iloc[2],geog_conflict_evals["0"].iloc[2],econ_conflict_evals["0"].iloc[2],pol_conflict_evals["0"].iloc[2],evals_conflict_ensemble_df["0"].iloc[2]], marker='o',color=colors[1],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10,12],[base_protest_evals["0"].iloc[2],history_protest_evals["0"].iloc[2],demog_protest_evals["0"].iloc[2],geog_protest_evals["0"].iloc[2],econ_protest_evals["0"].iloc[2],pol_protest_evals["0"].iloc[2],evals_protest_ensemble_df["0"].iloc[2]], marker="o",color=colors[2],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10,12],[base_riot_evals["0"].iloc[2],history_riot_evals["0"].iloc[2],demog_riot_evals["0"].iloc[2],geog_riot_evals["0"].iloc[2],econ_riot_evals["0"].iloc[2],pol_riot_evals["0"].iloc[2],evals_riot_ensemble_df["0"].iloc[2]], marker='o',color=colors[3],markersize=10,linewidth=3) # Riots
ax1.plot([0,2,4,6,8,10,12],[base_terror_evals["0"].iloc[2],history_terror_evals["0"].iloc[2],demog_terror_evals["0"].iloc[2],geog_terror_evals["0"].iloc[2],econ_terror_evals["0"].iloc[2],pol_terror_evals["0"].iloc[1],evals_terror_ensemble_df["0"].iloc[2]], marker='o',color=colors[4],markersize=10,linewidth=3) # Terror
ax1.plot([0,2,4,6,8,10,12],[base_sb_evals["0"].iloc[2],history_sb_evals["0"].iloc[2],demog_sb_evals["0"].iloc[2],geog_sb_evals["0"].iloc[2],econ_sb_evals["0"].iloc[2],pol_sb_evals["0"].iloc[2],evals_sb_ensemble_df["0"].iloc[2]], marker='o',color=colors[5],markersize=10,linewidth=3) # Civil conflict
ax1.plot([0,2,4,6,8,10,12],[base_ns_evals["0"].iloc[2],history_ns_evals["0"].iloc[2],demog_ns_evals["0"].iloc[2],geog_ns_evals["0"].iloc[2],econ_ns_evals["0"].iloc[2],pol_ns_evals["0"].iloc[2],evals_ns_ensemble_df["0"].iloc[2]], marker='o',color=colors[6],markersize=10,linewidth=3) # Non-state
ax1.plot([0,2,4,6,8,10,12],[base_osv_evals["0"].iloc[2],history_osv_evals["0"].iloc[2],demog_osv_evals["0"].iloc[2],geog_osv_evals["0"].iloc[2],econ_osv_evals["0"].iloc[2],pol_osv_evals["0"].iloc[2],evals_osv_ensemble_df["0"].iloc[2]], marker='o',color=colors[7],markersize=10,linewidth=3) # One-sided

ax1.set_xlim(-0.5, 12.5)
ax1.set_xticks([0,2,4,6,8,10,12], ["Baseline","History","Demography","Geography","Economy","Regime","Ensemble"],size=20)
ax1.set_yticks([0.5,0.6,0.7,0.8,0.9,1],[0.5,0.6,0.7,0.8,0.9,1],size=20)
ax1.set_ylabel("Area Under Receiver Operating Characteristic Curve",size=19)

# Manually create a custom legend with different marker styles
custom_legend = [
    plt.Line2D([], [], color=colors[0], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[1], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[2], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[3], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[4], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[5], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[6], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[7], marker='o',linestyle='', markersize=8),
    ]
legend_labels = ['1,000 fatalities','25 fatalities','Protest','Riots','Terrorism','Civil conflict',"Non-state","One-sided"]

# Add legend with custom markers
plt.legend(custom_legend, legend_labels,loc='center left', bbox_to_anchor=(0, -0.14),ncol=4,prop={'size': 18})

plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_evals_auroc_full.jpeg",dpi=400,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_evals_auroc_full.jpeg",dpi=400,bbox_inches="tight")
plt.savefig("out/struc_evals_auroc_full.jpeg",dpi=400,bbox_inches="tight")


fig, ax1 = plt.subplots(figsize=(12,8))

ax1.plot([0,2,4,6,8,10],[history_war_evals["0"].iloc[2],demog_war_evals["0"].iloc[2],geog_war_evals["0"].iloc[2],econ_war_evals["0"].iloc[2],pol_war_evals["0"].iloc[2],evals_war_ensemble_df["0"].iloc[2]], marker='o',color=colors[0],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10],[history_conflict_evals["0"].iloc[2],demog_conflict_evals["0"].iloc[2],geog_conflict_evals["0"].iloc[2],econ_conflict_evals["0"].iloc[2],pol_conflict_evals["0"].iloc[2],evals_conflict_ensemble_df["0"].iloc[2]], marker='o',color=colors[1],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10],[history_protest_evals["0"].iloc[2],demog_protest_evals["0"].iloc[2],geog_protest_evals["0"].iloc[2],econ_protest_evals["0"].iloc[2],pol_protest_evals["0"].iloc[2],evals_protest_ensemble_df["0"].iloc[2]], marker="o",color=colors[2],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10],[history_riot_evals["0"].iloc[2],demog_riot_evals["0"].iloc[2],geog_riot_evals["0"].iloc[2],econ_riot_evals["0"].iloc[2],pol_riot_evals["0"].iloc[2],evals_riot_ensemble_df["0"].iloc[2]], marker='o',color=colors[3],markersize=10,linewidth=3) # Riots
ax1.plot([0,2,4,6,8,10],[history_terror_evals["0"].iloc[2],demog_terror_evals["0"].iloc[2],geog_terror_evals["0"].iloc[2],econ_terror_evals["0"].iloc[2],pol_terror_evals["0"].iloc[1],evals_terror_ensemble_df["0"].iloc[2]], marker='o',color=colors[4],markersize=10,linewidth=3) # Terror
ax1.plot([0,2,4,6,8,10],[history_sb_evals["0"].iloc[2],demog_sb_evals["0"].iloc[2],geog_sb_evals["0"].iloc[2],econ_sb_evals["0"].iloc[2],pol_sb_evals["0"].iloc[2],evals_sb_ensemble_df["0"].iloc[2]], marker='o',color=colors[5],markersize=10,linewidth=3) # Civil conflict
ax1.plot([0,2,4,6,8,10],[history_ns_evals["0"].iloc[2],demog_ns_evals["0"].iloc[2],geog_ns_evals["0"].iloc[2],econ_ns_evals["0"].iloc[2],pol_ns_evals["0"].iloc[2],evals_ns_ensemble_df["0"].iloc[2]], marker='o',color=colors[6],markersize=10,linewidth=3) # Non-state
ax1.plot([0,2,4,6,8,10],[history_osv_evals["0"].iloc[2],demog_osv_evals["0"].iloc[2],geog_osv_evals["0"].iloc[2],econ_osv_evals["0"].iloc[2],pol_osv_evals["0"].iloc[2],evals_osv_ensemble_df["0"].iloc[2]], marker='o',color=colors[7],markersize=10,linewidth=3) # One-sided

ax1.set_xlim(-0.5, 10.5)
ax1.set_xticks([0,2,4,6,8,10], ["History","Demography","Geography","Economy","Regime","Ensemble"],size=20)
ax1.set_yticks([0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1],[0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1],size=20)
ax1.set_ylabel("Area Under Receiver Operating Characteristic Curve",size=19)

# Manually create a custom legend with different marker styles
custom_legend = [
    plt.Line2D([], [], color=colors[0], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[1], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[2], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[3], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[4], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[5], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[6], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[7], marker='o',linestyle='', markersize=8),
    ]
legend_labels = ['1,000 fatalities','25 fatalities','Protest','Riots','Terrorism','Civil conflict',"Non-state","One-sided"]

# Add legend with custom markers
plt.legend(custom_legend, legend_labels,loc='center left', bbox_to_anchor=(0, -0.14),ncol=4,prop={'size': 18})

plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_evals_auroc.jpeg",dpi=400,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_evals_auroc.jpeg",dpi=400,bbox_inches="tight")
plt.savefig("out/struc_evals_auroc.jpeg",dpi=400,bbox_inches="tight")

############
### AUPR ###
############

fig, ax1 = plt.subplots(figsize=(12,8))

ax1.plot([0,2,4,6,8,10,12],[base_war_evals["0"].iloc[1],history_war_evals["0"].iloc[1],demog_war_evals["0"].iloc[1],geog_war_evals["0"].iloc[1],econ_war_evals["0"].iloc[1],pol_war_evals["0"].iloc[1],evals_war_ensemble_df["0"].iloc[1]], marker='o',color=colors[0],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10,12],[base_conflict_evals["0"].iloc[1],history_conflict_evals["0"].iloc[1],demog_conflict_evals["0"].iloc[1],geog_conflict_evals["0"].iloc[1],econ_conflict_evals["0"].iloc[1],pol_conflict_evals["0"].iloc[1],evals_conflict_ensemble_df["0"].iloc[1]], marker='o',color=colors[1],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10,12],[base_protest_evals["0"].iloc[1],history_protest_evals["0"].iloc[1],demog_protest_evals["0"].iloc[1],geog_protest_evals["0"].iloc[1],econ_protest_evals["0"].iloc[1],pol_protest_evals["0"].iloc[1],evals_protest_ensemble_df["0"].iloc[1]], marker='o',color=colors[2],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10,12],[base_riot_evals["0"].iloc[1],history_riot_evals["0"].iloc[1],demog_riot_evals["0"].iloc[1],geog_riot_evals["0"].iloc[1],econ_riot_evals["0"].iloc[1],pol_riot_evals["0"].iloc[1],evals_riot_ensemble_df["0"].iloc[1]], marker='o',color=colors[3],markersize=10,linewidth=3) # Riots
ax1.plot([0,2,4,6,8,10,12],[base_terror_evals["0"].iloc[1],history_terror_evals["0"].iloc[1],demog_terror_evals["0"].iloc[1],geog_terror_evals["0"].iloc[1],econ_terror_evals["0"].iloc[1],pol_terror_evals["0"].iloc[1],evals_terror_ensemble_df["0"].iloc[1]], marker='o',color=colors[4],markersize=10,linewidth=3) # Terror
ax1.plot([0,2,4,6,8,10,12],[base_sb_evals["0"].iloc[1],history_sb_evals["0"].iloc[1],demog_sb_evals["0"].iloc[1],geog_sb_evals["0"].iloc[1],econ_sb_evals["0"].iloc[1],pol_sb_evals["0"].iloc[1],evals_sb_ensemble_df["0"].iloc[1]], marker='o',color=colors[5],markersize=10,linewidth=3) # Civil conflict
ax1.plot([0,2,4,6,8,10,12],[base_ns_evals["0"].iloc[1],history_ns_evals["0"].iloc[1],demog_ns_evals["0"].iloc[1],geog_ns_evals["0"].iloc[1],econ_ns_evals["0"].iloc[1],pol_ns_evals["0"].iloc[1],evals_ns_ensemble_df["0"].iloc[1]], marker='o',color=colors[6],markersize=10,linewidth=3) # Non-state
ax1.plot([0,2,4,6,8,10,12],[base_osv_evals["0"].iloc[1],history_osv_evals["0"].iloc[1],demog_osv_evals["0"].iloc[1],geog_osv_evals["0"].iloc[1],econ_osv_evals["0"].iloc[1],pol_osv_evals["0"].iloc[1],evals_osv_ensemble_df["0"].iloc[1]], marker='o',color=colors[7],markersize=10,linewidth=3) # One-sided


ax1.set_xlim(-0.5, 12.5)
ax1.set_xticks([0,2,4,6,8,10,12], ["Baseline","History","Demography","Geography","Economy","Regime","Ensemble"],size=20)
ax1.set_yticks([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],size=20)
ax1.set_ylabel("Area Under Precicion-Recall Curve",size=20)

# Manually create a custom legend with different marker styles
custom_legend = [
    plt.Line2D([], [], color=colors[0], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[1], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[2], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[3], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[4], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[5], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[6], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[7], marker='o',linestyle='', markersize=8),
    ]
legend_labels = ['1,000 fatalities','25 fatalities','Protest','Riots','Terrorism','Civil conflict',"Non-state","One-sided"]

# Add legend with custom markers
plt.legend(custom_legend, legend_labels,loc='center left', bbox_to_anchor=(0, -0.14),ncol=4,prop={'size': 18})

plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_evals_aupr_full.jpeg",dpi=400,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_evals_aupr_full.jpeg",dpi=400,bbox_inches="tight")
plt.savefig("out/struc_evals_aupr_full",dpi=400,bbox_inches="tight")


fig, ax1 = plt.subplots(figsize=(12,8))

ax1.plot([0,2,4,6,8,10],[history_war_evals["0"].iloc[1],demog_war_evals["0"].iloc[1],geog_war_evals["0"].iloc[1],econ_war_evals["0"].iloc[1],pol_war_evals["0"].iloc[1],evals_war_ensemble_df["0"].iloc[1]], marker='o',color=colors[0],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10],[history_conflict_evals["0"].iloc[1],demog_conflict_evals["0"].iloc[1],geog_conflict_evals["0"].iloc[1],econ_conflict_evals["0"].iloc[1],pol_conflict_evals["0"].iloc[1],evals_conflict_ensemble_df["0"].iloc[1]], marker='o',color=colors[1],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10],[history_protest_evals["0"].iloc[1],demog_protest_evals["0"].iloc[1],geog_protest_evals["0"].iloc[1],econ_protest_evals["0"].iloc[1],pol_protest_evals["0"].iloc[1],evals_protest_ensemble_df["0"].iloc[1]], marker='o',color=colors[2],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10],[history_riot_evals["0"].iloc[1],demog_riot_evals["0"].iloc[1],geog_riot_evals["0"].iloc[1],econ_riot_evals["0"].iloc[1],pol_riot_evals["0"].iloc[1],evals_riot_ensemble_df["0"].iloc[1]], marker='o',color=colors[3],markersize=10,linewidth=3) # Riots
ax1.plot([0,2,4,6,8,10],[history_terror_evals["0"].iloc[1],demog_terror_evals["0"].iloc[1],geog_terror_evals["0"].iloc[1],econ_terror_evals["0"].iloc[1],pol_terror_evals["0"].iloc[1],evals_terror_ensemble_df["0"].iloc[1]], marker='o',color=colors[4],markersize=10,linewidth=3) # Terror
ax1.plot([0,2,4,6,8,10],[history_sb_evals["0"].iloc[1],demog_sb_evals["0"].iloc[1],geog_sb_evals["0"].iloc[1],econ_sb_evals["0"].iloc[1],pol_sb_evals["0"].iloc[1],evals_sb_ensemble_df["0"].iloc[1]], marker='o',color=colors[5],markersize=10,linewidth=3) # Civil conflict
ax1.plot([0,2,4,6,8,10],[history_ns_evals["0"].iloc[1],demog_ns_evals["0"].iloc[1],geog_ns_evals["0"].iloc[1],econ_ns_evals["0"].iloc[1],pol_ns_evals["0"].iloc[1],evals_ns_ensemble_df["0"].iloc[1]], marker='o',color=colors[6],markersize=10,linewidth=3) # Non-state
ax1.plot([0,2,4,6,8,10],[history_osv_evals["0"].iloc[1],demog_osv_evals["0"].iloc[1],geog_osv_evals["0"].iloc[1],econ_osv_evals["0"].iloc[1],pol_osv_evals["0"].iloc[1],evals_osv_ensemble_df["0"].iloc[1]], marker='o',color=colors[7],markersize=10,linewidth=3) # One-sided


ax1.set_xlim(-0.5, 10.5)
ax1.set_xticks([0,2,4,6,8,10], ["History","Demography","Geography","Economy","Regime","Ensemble"],size=20)
ax1.set_yticks([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],size=20)
ax1.set_ylabel("Area Under Precicion-Recall Curve",size=20)

# Manually create a custom legend with different marker styles
custom_legend = [
    plt.Line2D([], [], color=colors[0], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[1], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[2], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[3], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[4], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[5], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[6], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[7], marker='o',linestyle='', markersize=8),
    ]
legend_labels = ['1,000 fatalities','25 fatalities','Protest','Riots','Terrorism','Civil conflict',"Non-state","One-sided"]

# Add legend with custom markers
plt.legend(custom_legend, legend_labels,loc='center left', bbox_to_anchor=(0, -0.14),ncol=4,prop={'size': 18})

plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_evals_aupr.jpeg",dpi=400,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_evals_aupr.jpeg",dpi=400,bbox_inches="tight")
plt.savefig("out/struc_evals_aupr",dpi=400,bbox_inches="tight")

#############
### Brier ###
#############

fig, ax1 = plt.subplots(figsize=(12,8))

ax1.plot([0,2,4,6,8,10,12],[base_war_evals["0"].iloc[0],history_war_evals["0"].iloc[0],demog_war_evals["0"].iloc[0],geog_war_evals["0"].iloc[0],econ_war_evals["0"].iloc[0],pol_war_evals["0"].iloc[0],evals_war_ensemble_df["0"].iloc[0]], marker='o',color=colors[0],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10,12],[base_conflict_evals["0"].iloc[0],history_conflict_evals["0"].iloc[0],demog_conflict_evals["0"].iloc[0],geog_conflict_evals["0"].iloc[0],econ_conflict_evals["0"].iloc[0],pol_conflict_evals["0"].iloc[0],evals_conflict_ensemble_df["0"].iloc[0]], marker='o',color=colors[1],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10,12],[base_protest_evals["0"].iloc[0],history_protest_evals["0"].iloc[0],demog_protest_evals["0"].iloc[0],geog_protest_evals["0"].iloc[0],econ_protest_evals["0"].iloc[0],pol_protest_evals["0"].iloc[0],evals_protest_ensemble_df["0"].iloc[0]], marker='o',color=colors[2],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10,12],[base_riot_evals["0"].iloc[0],history_riot_evals["0"].iloc[0],demog_riot_evals["0"].iloc[0],geog_riot_evals["0"].iloc[0],econ_riot_evals["0"].iloc[0],pol_riot_evals["0"].iloc[0],evals_riot_ensemble_df["0"].iloc[0]], marker='o',color=colors[3],markersize=10,linewidth=3) # Riots
ax1.plot([0,2,4,6,8,10,12],[base_terror_evals["0"].iloc[0],history_terror_evals["0"].iloc[0],demog_terror_evals["0"].iloc[0],geog_terror_evals["0"].iloc[0],econ_terror_evals["0"].iloc[0],pol_terror_evals["0"].iloc[0],evals_terror_ensemble_df["0"].iloc[0]], marker='o',color=colors[4],markersize=10,linewidth=3) # Terror
ax1.plot([0,2,4,6,8,10,12],[base_sb_evals["0"].iloc[0],history_sb_evals["0"].iloc[0],demog_sb_evals["0"].iloc[0],geog_sb_evals["0"].iloc[0],econ_sb_evals["0"].iloc[0],pol_sb_evals["0"].iloc[0],evals_sb_ensemble_df["0"].iloc[0]], marker='o',color=colors[5],markersize=10,linewidth=3) # Civil conflict
ax1.plot([0,2,4,6,8,10,12],[base_ns_evals["0"].iloc[0],history_ns_evals["0"].iloc[0],demog_ns_evals["0"].iloc[0],geog_ns_evals["0"].iloc[0],econ_ns_evals["0"].iloc[0],pol_ns_evals["0"].iloc[0],evals_ns_ensemble_df["0"].iloc[0]], marker='o',color=colors[6],markersize=10,linewidth=3) # Non-state
ax1.plot([0,2,4,6,8,10,12],[base_osv_evals["0"].iloc[0],history_osv_evals["0"].iloc[0],demog_osv_evals["0"].iloc[0],geog_osv_evals["0"].iloc[0],econ_osv_evals["0"].iloc[0],pol_osv_evals["0"].iloc[0],evals_osv_ensemble_df["0"].iloc[0]], marker='o',color=colors[7],markersize=10,linewidth=3) # One-sided

ax1.set_xticks([0,2,4,6,8,10,12], ["Baseline","History","Demography","Geography","Economy","Regime","Ensemble"],size=20)
ax1.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],size=20)

ax1.set_ylabel("Brier score",size=20)

# Manually create a custom legend with different marker styles
custom_legend = [
    plt.Line2D([], [], color=colors[0], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[1], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[2], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[3], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[4], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[5], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[6], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[7], marker='o',linestyle='', markersize=8),
    ]
legend_labels = ['1,000 fatalities','25 fatalities','Protest','Riots','Terrorism','Civil conflict',"Non-state","One-sided"]

# Add legend with custom markers
plt.legend(custom_legend, legend_labels,loc='center left', bbox_to_anchor=(0, -0.14),ncol=4,prop={'size': 18})

plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_evals_brier_full.jpeg",dpi=400,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_evals_brier_full.jpeg",dpi=400,bbox_inches="tight")
plt.savefig("out/struc_evals_brier_full",dpi=400,bbox_inches="tight")


fig, ax1 = plt.subplots(figsize=(12,8))

ax1.plot([0,2,4,6,8,10],[history_war_evals["0"].iloc[0],demog_war_evals["0"].iloc[0],geog_war_evals["0"].iloc[0],econ_war_evals["0"].iloc[0],pol_war_evals["0"].iloc[0],evals_war_ensemble_df["0"].iloc[0]], marker='o',color=colors[0],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10],[history_conflict_evals["0"].iloc[0],demog_conflict_evals["0"].iloc[0],geog_conflict_evals["0"].iloc[0],econ_conflict_evals["0"].iloc[0],pol_conflict_evals["0"].iloc[0],evals_conflict_ensemble_df["0"].iloc[0]], marker='o',color=colors[1],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10],[history_protest_evals["0"].iloc[0],demog_protest_evals["0"].iloc[0],geog_protest_evals["0"].iloc[0],econ_protest_evals["0"].iloc[0],pol_protest_evals["0"].iloc[0],evals_protest_ensemble_df["0"].iloc[0]], marker='o',color=colors[2],markersize=10,linewidth=3) # Civil War
ax1.plot([0,2,4,6,8,10],[history_riot_evals["0"].iloc[0],demog_riot_evals["0"].iloc[0],geog_riot_evals["0"].iloc[0],econ_riot_evals["0"].iloc[0],pol_riot_evals["0"].iloc[0],evals_riot_ensemble_df["0"].iloc[0]], marker='o',color=colors[3],markersize=10,linewidth=3) # Riots
ax1.plot([0,2,4,6,8,10],[history_terror_evals["0"].iloc[0],demog_terror_evals["0"].iloc[0],geog_terror_evals["0"].iloc[0],econ_terror_evals["0"].iloc[0],pol_terror_evals["0"].iloc[0],evals_terror_ensemble_df["0"].iloc[0]], marker='o',color=colors[4],markersize=10,linewidth=3) # Terror
ax1.plot([0,2,4,6,8,10],[history_sb_evals["0"].iloc[0],demog_sb_evals["0"].iloc[0],geog_sb_evals["0"].iloc[0],econ_sb_evals["0"].iloc[0],pol_sb_evals["0"].iloc[0],evals_sb_ensemble_df["0"].iloc[0]], marker='o',color=colors[5],markersize=10,linewidth=3) # Civil conflict
ax1.plot([0,2,4,6,8,10],[history_ns_evals["0"].iloc[0],demog_ns_evals["0"].iloc[0],geog_ns_evals["0"].iloc[0],econ_ns_evals["0"].iloc[0],pol_ns_evals["0"].iloc[0],evals_ns_ensemble_df["0"].iloc[0]], marker='o',color=colors[6],markersize=10,linewidth=3) # Non-state
ax1.plot([0,2,4,6,8,10],[history_osv_evals["0"].iloc[0],demog_osv_evals["0"].iloc[0],geog_osv_evals["0"].iloc[0],econ_osv_evals["0"].iloc[0],pol_osv_evals["0"].iloc[0],evals_osv_ensemble_df["0"].iloc[0]], marker='o',color=colors[7],markersize=10,linewidth=3) # One-sided

ax1.set_xticks([0,2,4,6,8,10], ["History","Demography","Geography","Economy","Regime","Ensemble"],size=20)
ax1.set_yticks([0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2],[0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2],size=20)

ax1.set_ylabel("Brier score",size=20)

# Manually create a custom legend with different marker styles
custom_legend = [
    plt.Line2D([], [], color=colors[0], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[1], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[2], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[3], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[4], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[5], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[6], marker='o',linestyle='', markersize=8),
    plt.Line2D([], [], color=colors[7], marker='o',linestyle='', markersize=8),
    ]
legend_labels = ['1,000 fatalities','25 fatalities','Protest','Riots','Terrorism','Civil conflict',"Non-state","One-sided"]

# Add legend with custom markers
plt.legend(custom_legend, legend_labels,loc='center left', bbox_to_anchor=(0, -0.14),ncol=4,prop={'size': 18})

plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/struc_evals_brier.jpeg",dpi=400,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/struc_evals_brier.jpeg",dpi=400,bbox_inches="tight")
plt.savefig("out/struc_evals_brier",dpi=400,bbox_inches="tight")


### Brier ###
print(f"{round(base_conflict_evals['0'].iloc[0],5)} &  \\\
      {round(history_conflict_evals['0'].iloc[0],5)} &  \\\
      {round(demog_conflict_evals['0'].iloc[0],5)} &  \\\
      {round(geog_conflict_evals['0'].iloc[0],5)} &  \\\
      {round(econ_conflict_evals['0'].iloc[0],5)} &  \\\
      {round(pol_conflict_evals['0'].iloc[0],5)} &  \\\
      {round(evals_conflict_ensemble_df['0'].iloc[0],5)} ")
           
print(f"{round(base_protest_evals['0'].iloc[0],5)} &  \\\
      {round(history_protest_evals['0'].iloc[0],5)} &  \\\
      {round(demog_protest_evals['0'].iloc[0],5)} &  \\\
      {round(geog_protest_evals['0'].iloc[0],5)} &  \\\
      {round(econ_protest_evals['0'].iloc[0],5)} &  \\\
      {round(pol_protest_evals['0'].iloc[0],5)}&  \\\
      {round(evals_protest_ensemble_df['0'].iloc[0],5)}")
      
print(f"{round(base_riot_evals['0'].iloc[0],5)} &  \\\
      {round(history_riot_evals['0'].iloc[0],5)} &  \\\
      {round(demog_riot_evals['0'].iloc[0],5)} &  \\\
      {round(geog_riot_evals['0'].iloc[0],5)} &  \\\
      {round(econ_riot_evals['0'].iloc[0],5)} &  \\\
      {round(pol_riot_evals['0'].iloc[0],5)} &  \\\
      {round(evals_riot_ensemble_df['0'].iloc[0],5)}")

print(f"{round(base_terror_evals['0'].iloc[0],5)} &  \\\
      {round(history_terror_evals['0'].iloc[0],5)} &  \\\
      {round(demog_terror_evals['0'].iloc[0],5)} &  \\\
      {round(geog_terror_evals['0'].iloc[0],5)} &  \\\
      {round(econ_terror_evals['0'].iloc[0],5)} &  \\\
      {round(pol_terror_evals['0'].iloc[0],5)}&  \\\
      {round(evals_terror_ensemble_df['0'].iloc[0],5)}")

print(f"{round(base_sb_evals['0'].iloc[0],5)} &  \\\
      {round(history_sb_evals['0'].iloc[0],5)} &  \\\
      {round(demog_sb_evals['0'].iloc[0],5)} &  \\\
      {round(geog_sb_evals['0'].iloc[0],5)} &  \\\
      {round(econ_sb_evals['0'].iloc[0],5)} &  \\\
      {round(pol_sb_evals['0'].iloc[0],5)}&  \\\
      {round(evals_sb_ensemble_df['0'].iloc[0],5)}")

print(f"{round(base_ns_evals['0'].iloc[0],5)} &  \\\
      {round(history_ns_evals['0'].iloc[0],5)} &  \\\
      {round(demog_ns_evals['0'].iloc[0],5)} &  \\\
      {round(geog_ns_evals['0'].iloc[0],5)} &  \\\
      {round(econ_ns_evals['0'].iloc[0],5)} &  \\\
      {round(pol_ns_evals['0'].iloc[0],5)} &  \\\
      {round(evals_ns_ensemble_df['0'].iloc[0],5)}")      
 
print(f"{round(base_osv_evals['0'].iloc[0],5)} &  \\\
      {round(history_osv_evals['0'].iloc[0],5)} &  \\\
      {round(demog_osv_evals['0'].iloc[0],5)} &  \\\
      {round(geog_osv_evals['0'].iloc[0],5)} &  \\\
      {round(econ_osv_evals['0'].iloc[0],5)} &  \\\
      {round(pol_osv_evals['0'].iloc[0],5)}&  \\\
      {round(evals_osv_ensemble_df['0'].iloc[0],5)}")            
  
### AUPR ###
print(f"{round(base_conflict_evals['0'].iloc[1],5)} &  \\\
      {round(history_conflict_evals['0'].iloc[1],5)} &  \\\
      {round(demog_conflict_evals['0'].iloc[1],5)} &  \\\
      {round(geog_conflict_evals['0'].iloc[1],5)} &  \\\
      {round(econ_conflict_evals['0'].iloc[1],5)} &  \\\
      {round(pol_conflict_evals['0'].iloc[1],5)}&  \\\
      {round(evals_conflict_ensemble_df['0'].iloc[1],5)}")
           
print(f"{round(base_protest_evals['0'].iloc[1],5)} &  \\\
      {round(history_protest_evals['0'].iloc[1],5)} &  \\\
      {round(demog_protest_evals['0'].iloc[1],5)} &  \\\
      {round(geog_protest_evals['0'].iloc[1],5)} &  \\\
      {round(econ_protest_evals['0'].iloc[1],5)} &  \\\
      {round(pol_protest_evals['0'].iloc[1],5)}&  \\\
      {round(evals_protest_ensemble_df['0'].iloc[1],5)}")
      
print(f"{round(base_riot_evals['0'].iloc[1],5)} &  \\\
      {round(history_riot_evals['0'].iloc[1],5)} &  \\\
      {round(demog_riot_evals['0'].iloc[1],5)} &  \\\
      {round(geog_riot_evals['0'].iloc[1],5)} &  \\\
      {round(econ_riot_evals['0'].iloc[1],5)} &  \\\
      {round(pol_riot_evals['0'].iloc[1],5)}&  \\\
      {round(evals_riot_ensemble_df['0'].iloc[1],5)}")

print(f"{round(base_terror_evals['0'].iloc[1],5)} &  \\\
      {round(history_terror_evals['0'].iloc[1],5)} &  \\\
      {round(demog_terror_evals['0'].iloc[1],5)} &  \\\
      {round(geog_terror_evals['0'].iloc[1],5)} &  \\\
      {round(econ_terror_evals['0'].iloc[1],5)} &  \\\
      {round(pol_terror_evals['0'].iloc[1],5)}&  \\\
      {round(evals_terror_ensemble_df['0'].iloc[1],5)}")

print(f"{round(base_sb_evals['0'].iloc[1],5)} &  \\\
      {round(history_sb_evals['0'].iloc[1],5)} &  \\\
      {round(demog_sb_evals['0'].iloc[1],5)} &  \\\
      {round(geog_sb_evals['0'].iloc[1],5)} &  \\\
      {round(econ_sb_evals['0'].iloc[1],5)} &  \\\
      {round(pol_sb_evals['0'].iloc[1],5)}&  \\\
      {round(evals_sb_ensemble_df['0'].iloc[1],5)}")

print(f"{round(base_ns_evals['0'].iloc[1],5)} &  \\\
      {round(history_ns_evals['0'].iloc[1],5)} &  \\\
      {round(demog_ns_evals['0'].iloc[1],5)} &  \\\
      {round(geog_ns_evals['0'].iloc[1],5)} &  \\\
      {round(econ_ns_evals['0'].iloc[1],5)} &  \\\
      {round(pol_ns_evals['0'].iloc[1],5)}&  \\\
      {round(evals_ns_ensemble_df['0'].iloc[1],5)}")      
 
print(f"{round(base_osv_evals['0'].iloc[1],5)} &  \\\
      {round(history_osv_evals['0'].iloc[1],5)} &  \\\
      {round(demog_osv_evals['0'].iloc[1],5)} &  \\\
      {round(geog_osv_evals['0'].iloc[1],5)} &  \\\
      {round(econ_osv_evals['0'].iloc[1],5)} &  \\\
      {round(pol_osv_evals['0'].iloc[1],5)}&  \\\
      {round(evals_osv_ensemble_df['0'].iloc[1],5)}")        
 
### AUROC ###
print(f"{round(base_conflict_evals['0'].iloc[2],5)} &  \\\
      {round(history_conflict_evals['0'].iloc[2],5)} &  \\\
      {round(demog_conflict_evals['0'].iloc[2],5)} &  \\\
      {round(geog_conflict_evals['0'].iloc[2],5)} &  \\\
      {round(econ_conflict_evals['0'].iloc[2],5)} &  \\\
      {round(pol_conflict_evals['0'].iloc[2],5)}&  \\\
      {round(evals_conflict_ensemble_df['0'].iloc[2],5)}")
           
print(f"{round(base_protest_evals['0'].iloc[2],5)} &  \\\
      {round(history_protest_evals['0'].iloc[2],5)} &  \\\
      {round(demog_protest_evals['0'].iloc[2],5)} &  \\\
      {round(geog_protest_evals['0'].iloc[2],5)} &  \\\
      {round(econ_protest_evals['0'].iloc[2],5)} &  \\\
      {round(pol_protest_evals['0'].iloc[2],5)}&  \\\
      {round(evals_protest_ensemble_df['0'].iloc[2],5)}")
      
print(f"{round(base_riot_evals['0'].iloc[2],5)} &  \\\
      {round(history_riot_evals['0'].iloc[2],5)} &  \\\
      {round(demog_riot_evals['0'].iloc[2],5)} &  \\\
      {round(geog_riot_evals['0'].iloc[2],5)} &  \\\
      {round(econ_riot_evals['0'].iloc[2],5)} &  \\\
      {round(pol_riot_evals['0'].iloc[2],5)}&  \\\
      {round(evals_riot_ensemble_df['0'].iloc[2],5)}")

print(f"{round(base_terror_evals['0'].iloc[2],5)} &  \\\
      {round(history_terror_evals['0'].iloc[2],5)} &  \\\
      {round(demog_terror_evals['0'].iloc[2],5)} &  \\\
      {round(geog_terror_evals['0'].iloc[2],5)} &  \\\
      {round(econ_terror_evals['0'].iloc[2],5)} &  \\\
      {round(pol_terror_evals['0'].iloc[2],5)}&  \\\
      {round(evals_terror_ensemble_df['0'].iloc[2],5)}")

print(f"{round(base_sb_evals['0'].iloc[2],5)} &  \\\
      {round(history_sb_evals['0'].iloc[2],5)} &  \\\
      {round(demog_sb_evals['0'].iloc[2],5)} &  \\\
      {round(geog_sb_evals['0'].iloc[2],5)} &  \\\
      {round(econ_sb_evals['0'].iloc[2],5)} &  \\\
      {round(pol_sb_evals['0'].iloc[2],5)}&  \\\
      {round(evals_sb_ensemble_df['0'].iloc[2],5)}")

print(f"{round(base_ns_evals['0'].iloc[2],5)} &  \\\
      {round(history_ns_evals['0'].iloc[2],5)} &  \\\
      {round(demog_ns_evals['0'].iloc[2],5)} &  \\\
      {round(geog_ns_evals['0'].iloc[2],5)} &  \\\
      {round(econ_ns_evals['0'].iloc[2],5)} &  \\\
      {round(pol_ns_evals['0'].iloc[2],5)}&  \\\
      {round(evals_ns_ensemble_df['0'].iloc[2],5)}")  
 
print(f"{round(base_osv_evals['0'].iloc[2],5)} &  \\\
      {round(history_osv_evals['0'].iloc[2],5)} &  \\\
      {round(demog_osv_evals['0'].iloc[2],5)} &  \\\
      {round(geog_osv_evals['0'].iloc[2],5)} &  \\\
      {round(econ_osv_evals['0'].iloc[2],5)} &  \\\
      {round(pol_osv_evals['0'].iloc[2],5)}&  \\\
      {round(evals_osv_ensemble_df['0'].iloc[2],5)}")    
      
###################
### Final Plots ###
###################

colors=["black","gray","forestgreen","lightgreen","steelblue","lightblue","purple","violet"]

fig, ax = plt.subplots(figsize=(12,8))

plt.scatter(history_war_evals["0"].iloc[1],history_war_evals["0"].iloc[2],c=colors[0],s=70,marker="o") 
plt.scatter(demog_war_evals["0"].iloc[1],demog_war_evals["0"].iloc[2],c=colors[0],s=70,marker="v") 
plt.scatter(geog_war_evals["0"].iloc[1],geog_war_evals["0"].iloc[2],c=colors[0],s=70,marker="s") 
#ax.add_patch(Rectangle((econ_war_evals["0"].iloc[1],econ_war_evals["0"].iloc[2]), width=0.02, height=0.004, color=colors[0]))
plt.scatter(econ_war_evals["0"].iloc[1],econ_war_evals["0"].iloc[2],c=colors[0],s=120,marker="_") 
plt.scatter(pol_war_evals["0"].iloc[1],pol_war_evals["0"].iloc[2],c=colors[0],s=70,marker="D") 
plt.scatter(evals_war_ensemble_df["0"].iloc[1],evals_war_ensemble_df["0"].iloc[2],c=colors[0],s=120,marker="x") 

plt.scatter(history_conflict_evals["0"].iloc[1],history_conflict_evals["0"].iloc[2],c=colors[1],s=70,marker="o") 
plt.scatter(demog_conflict_evals["0"].iloc[1],demog_conflict_evals["0"].iloc[2],c=colors[1],s=70,marker="v") 
plt.scatter(geog_conflict_evals["0"].iloc[1],geog_conflict_evals["0"].iloc[2],c=colors[1],s=70,marker="s") 
#ax.add_patch(Rectangle((econ_conflict_evals["0"].iloc[1],econ_conflict_evals["0"].iloc[2]), width=0.02, height=0.004, color=colors[1]))
plt.scatter(econ_conflict_evals["0"].iloc[1],econ_conflict_evals["0"].iloc[2],c=colors[1],s=120,marker="_") 
plt.scatter(pol_conflict_evals["0"].iloc[1],pol_conflict_evals["0"].iloc[2],c=colors[1],s=70,marker="D") 
plt.scatter(evals_conflict_ensemble_df["0"].iloc[1],evals_conflict_ensemble_df["0"].iloc[2],c=colors[1],s=120,marker="x") 

plt.scatter(history_protest_evals["0"].iloc[1],history_protest_evals["0"].iloc[2],c=colors[2],s=70,marker="o",edgecolors="gray") 
plt.scatter(demog_protest_evals["0"].iloc[1],demog_protest_evals["0"].iloc[2],c=colors[2],s=70,marker="v") 
plt.scatter(geog_protest_evals["0"].iloc[1],geog_protest_evals["0"].iloc[2],c=colors[2],s=70,marker="s") 
#ax.add_patch(Rectangle((econ_protest_evals["0"].iloc[1],econ_protest_evals["0"].iloc[2]), width=0.02, height=0.004, color=colors[2]))
plt.scatter(econ_protest_evals["0"].iloc[1],econ_protest_evals["0"].iloc[2],c=colors[2],s=120,marker="_") 
plt.scatter(pol_protest_evals["0"].iloc[1],pol_protest_evals["0"].iloc[2],c=colors[2],s=70,marker="D") 
plt.scatter(evals_protest_ensemble_df["0"].iloc[1],evals_protest_ensemble_df["0"].iloc[2],c=colors[2],s=120,marker="x") 

plt.scatter(history_riot_evals["0"].iloc[1],history_riot_evals["0"].iloc[2],c=colors[3],s=70,marker="o") 
plt.scatter(demog_riot_evals["0"].iloc[1],demog_riot_evals["0"].iloc[2],c=colors[3],s=70,marker="v") 
plt.scatter(geog_riot_evals["0"].iloc[1],geog_riot_evals["0"].iloc[2],c=colors[3],s=70,marker="s") 
#ax.add_patch(Rectangle((econ_riot_evals["0"].iloc[1],econ_riot_evals["0"].iloc[2]), width=0.02, height=0.004, color=colors[3]))
plt.scatter(econ_riot_evals["0"].iloc[1],econ_riot_evals["0"].iloc[2],c=colors[3],s=120,marker="_") 
plt.scatter(pol_riot_evals["0"].iloc[1],pol_riot_evals["0"].iloc[2],c=colors[3],s=70,marker="D") 
plt.scatter(evals_riot_ensemble_df["0"].iloc[1],evals_riot_ensemble_df["0"].iloc[2],c=colors[3],s=120,marker="x") 

plt.scatter(history_terror_evals["0"].iloc[1],history_terror_evals["0"].iloc[2],c=colors[4],s=70,marker="o") 
plt.scatter(demog_terror_evals["0"].iloc[1],demog_terror_evals["0"].iloc[2],c=colors[4],s=70,marker="v") 
plt.scatter(geog_terror_evals["0"].iloc[1],geog_terror_evals["0"].iloc[2],c=colors[4],s=70,marker="s") 
#ax.add_patch(Rectangle((econ_terror_evals["0"].iloc[1],econ_terror_evals["0"].iloc[2]), width=0.02, height=0.004, color=colors[4]))
plt.scatter(econ_terror_evals["0"].iloc[1],econ_terror_evals["0"].iloc[2],c=colors[4],s=120,marker="_") 
plt.scatter(pol_terror_evals["0"].iloc[1],pol_terror_evals["0"].iloc[2],c=colors[4],s=70,marker="D") 
plt.scatter(evals_terror_ensemble_df["0"].iloc[1],evals_terror_ensemble_df["0"].iloc[2],c=colors[4],s=120,marker="x") 

plt.scatter(history_sb_evals["0"].iloc[1],history_sb_evals["0"].iloc[2],c=colors[5],s=70,marker="o") 
plt.scatter(demog_sb_evals["0"].iloc[1],demog_sb_evals["0"].iloc[2],c=colors[5],s=70,marker="v") 
plt.scatter(geog_sb_evals["0"].iloc[1],geog_sb_evals["0"].iloc[2],c=colors[5],s=70,marker="s") 
#ax.add_patch(Rectangle((econ_sb_evals["0"].iloc[1],econ_sb_evals["0"].iloc[2]), width=0.02, height=0.004, color=colors[5]))
plt.scatter(econ_sb_evals["0"].iloc[1],econ_sb_evals["0"].iloc[2],c=colors[5],s=120,marker="_") 
plt.scatter(pol_sb_evals["0"].iloc[1],pol_sb_evals["0"].iloc[2],c=colors[5],s=70,marker="D") 
plt.scatter(evals_sb_ensemble_df["0"].iloc[1],evals_sb_ensemble_df["0"].iloc[2],c=colors[5],s=120,marker="x") 

plt.scatter(history_ns_evals["0"].iloc[1],history_ns_evals["0"].iloc[2],c=colors[6],s=70,marker="o") 
plt.scatter(demog_ns_evals["0"].iloc[1],demog_ns_evals["0"].iloc[2],c=colors[6],s=70,marker="v") 
plt.scatter(geog_ns_evals["0"].iloc[1],geog_ns_evals["0"].iloc[2],c=colors[6],s=70,marker="s") 
#ax.add_patch(Rectangle((econ_ns_evals["0"].iloc[1],econ_ns_evals["0"].iloc[2]), width=0.02, height=0.004, color=colors[6]))
plt.scatter(econ_ns_evals["0"].iloc[1],econ_ns_evals["0"].iloc[2],c=colors[6],s=120,marker="_") 
plt.scatter(pol_ns_evals["0"].iloc[1],pol_ns_evals["0"].iloc[2],c=colors[6],s=70,marker="D") 
plt.scatter(evals_ns_ensemble_df["0"].iloc[1],evals_ns_ensemble_df["0"].iloc[2],c=colors[6],s=120,marker="x") 

plt.scatter(history_osv_evals["0"].iloc[1],history_osv_evals["0"].iloc[2],c=colors[7],s=70,marker="o") 
plt.scatter(demog_osv_evals["0"].iloc[1],demog_osv_evals["0"].iloc[2],c=colors[7],s=70,marker="v") 
plt.scatter(geog_osv_evals["0"].iloc[1],geog_osv_evals["0"].iloc[2],c=colors[7],s=70,marker="s") 
#ax.add_patch(Rectangle((econ_osv_evals["0"].iloc[1],econ_osv_evals["0"].iloc[2]), width=0.02, height=0.004, color=colors[7]))
plt.scatter(econ_osv_evals["0"].iloc[1],econ_osv_evals["0"].iloc[2],c=colors[7],s=120,marker="_") 
plt.scatter(pol_osv_evals["0"].iloc[1],pol_osv_evals["0"].iloc[2],c=colors[7],s=70,marker="D") 
plt.scatter(evals_osv_ensemble_df["0"].iloc[1],evals_osv_ensemble_df["0"].iloc[2],c=colors[7],s=120,marker="x") 

# Create custom legend
handles = [
    mpatches.Patch(color='black', label='1,000 fatalities'),
    mpatches.Patch(color='gray', label='25 fatalities'),
    mpatches.Patch(color='forestgreen', label='Protest'),
    mpatches.Patch(color='lightgreen', label='Riots'),
    mpatches.Patch(color='steelblue', label='Remote Violence'),
    mpatches.Patch(color='lightblue', label='Civil Conflict'),
    mpatches.Patch(color='purple', label='Non-state'),
    mpatches.Patch(color='violet', label='One-sided')]

marker_handles = [
    mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Conflict History'),
    mlines.Line2D([], [], color='black', marker='v', linestyle='None', markersize=10, label='Demography'),
    mlines.Line2D([], [], color='black', marker='s', linestyle='None', markersize=10, label='Geography'),
    mlines.Line2D([], [], color='black', marker='_', linestyle='None', markersize=12, label='Economy'),
    mlines.Line2D([], [], color='black', marker='D', linestyle='None', markersize=10, label='Regime'),
    mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=10, label='Ensemble'),

]

legend1=ax.legend(handles=handles, title='Outcomes', loc='center left', bbox_to_anchor=(1, 0.7),frameon=False,fontsize=15,title_fontsize=15)
ax.add_artist(legend1) 
legend2 = ax.legend(handles=marker_handles,title='Thematic Models',loc='lower left',frameon=False,fontsize=15,title_fontsize=15,bbox_to_anchor=(1, 0.1))
ax.set_yticks([0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],[0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],size=20)
ax.set_xticks([0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],[0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],size=20)

ax.set_xlabel("Area Under Precivion-Recall Curve",size=20)
ax.set_ylabel("Area Under Receiver Operating Characteristic Cruve",size=20)

plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/evals_final.png",dpi=400,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/evals_final.png",dpi=400,bbox_inches="tight")
plt.savefig("out/evals_final.png",dpi=400,bbox_inches="tight")
plt.show()
    
#################   
### Bar Plots ###
#################   

### Conflict History ###

s_protest = pd.DataFrame(list(zip(['d_protest_lag1',"d_protest_zeros_decay","d_neighbors_proteset_event_counts_lag1","regime_duration",'pop_refugee','pop_refugee_id'], np.abs(history_protest_shap[:, :, 1]).mean(0))),columns=['Feature','Protest'])
s_protest.loc[s_protest["Feature"]=="d_protest_lag1","Feature"]="d_lag1"
s_protest.loc[s_protest["Feature"]=="d_protest_zeros_decay","Feature"]="d_zeros_decay"
s_protest.loc[s_protest["Feature"]=="d_neighbors_proteset_event_counts_lag1","Feature"]="d_neighbors_lag1"
shap_conflict_hist = s_protest[~s_protest['Feature'].str.contains('_id')]

s_riot = pd.DataFrame(list(zip(['d_riot_lag1',"d_riot_zeros_decay","d_neighbors_riot_event_counts_lag1",'regime_duration', 'pop_refugee', 'pop_refugee_id'], np.abs(history_riot_shap[:, :, 1]).mean(0))),columns=['Feature','Riot'])
s_riot.loc[s_riot["Feature"]=="d_riot_lag1","Feature"]="d_lag1"
s_riot.loc[s_riot["Feature"]=="d_riot_zeros_decay","Feature"]="d_zeros_decay"
s_riot.loc[s_riot["Feature"]=="d_neighbors_riot_event_counts_lag1","Feature"]="d_neighbors_lag1"
shap_conflict_hist = pd.merge(shap_conflict_hist, s_riot[~s_riot['Feature'].str.contains('_id')],on=["Feature"])

s_remote = pd.DataFrame(list(zip(['d_remote_lag1',"d_remote_zeros_decay","d_neighbors_remote_event_counts_lag1",'regime_duration', 'pop_refugee', 'pop_refugee_id'], np.abs(history_terror_shap[:, :, 1]).mean(0))),columns=['Feature','Remote'])
s_remote.loc[s_remote["Feature"]=="d_remote_lag1","Feature"]="d_lag1"
s_remote.loc[s_remote["Feature"]=="d_remote_zeros_decay","Feature"]="d_zeros_decay"
s_remote.loc[s_remote["Feature"]=="d_neighbors_remote_event_counts_lag1","Feature"]="d_neighbors_lag1"
shap_conflict_hist = pd.merge(shap_conflict_hist, s_remote[~s_remote['Feature'].str.contains('_id')],on=["Feature"])

s_sb = pd.DataFrame(list(zip(['d_sb_lag1',"d_sb_zeros_decay","d_neighbors_sb_event_counts_lag1",'regime_duration', 'pop_refugee', 'pop_refugee_id'], np.abs(history_sb_shap[:, :, 1]).mean(0))),columns=['Feature','State-based'])
s_sb.loc[s_sb["Feature"]=="d_sb_lag1","Feature"]="d_lag1"
s_sb.loc[s_sb["Feature"]=="d_sb_zeros_decay","Feature"]="d_zeros_decay"
s_sb.loc[s_sb["Feature"]=="d_neighbors_sb_event_counts_lag1","Feature"]="d_neighbors_lag1"
shap_conflict_hist = pd.merge(shap_conflict_hist, s_sb[~s_sb['Feature'].str.contains('_id')],on=["Feature"])

s_osv = pd.DataFrame(list(zip(['d_osv_lag1',"d_osv_zeros_decay","d_neighbors_osv_event_counts_lag1",'regime_duration', 'pop_refugee', 'pop_refugee_id'], np.abs(history_osv_shap[:, :, 1]).mean(0))),columns=['Feature','One-sided'])
s_osv.loc[s_osv["Feature"]=="d_osv_lag1","Feature"]="d_lag1"
s_osv.loc[s_osv["Feature"]=="d_osv_zeros_decay","Feature"]="d_zeros_decay"
s_osv.loc[s_osv["Feature"]=="d_neighbors_osv_event_counts_lag1","Feature"]="d_neighbors_lag1"
shap_conflict_hist = pd.merge(shap_conflict_hist, s_osv[~s_osv['Feature'].str.contains('_id')],on=["Feature"])

s_ns = pd.DataFrame(list(zip(['d_ns_lag1',"d_ns_zeros_decay","d_neighbors_ns_event_counts_lag1",'regime_duration', 'pop_refugee', 'pop_refugee_id'], np.abs(history_ns_shap[:, :, 1]).mean(0))),columns=['Feature','Non-state'])
s_ns.loc[s_ns["Feature"]=="d_ns_lag1","Feature"]="d_lag1"
s_ns.loc[s_ns["Feature"]=="d_ns_zeros_decay","Feature"]="d_zeros_decay"
s_ns.loc[s_ns["Feature"]=="d_neighbors_ns_event_counts_lag1","Feature"]="d_neighbors_lag1"
shap_conflict_hist = pd.merge(shap_conflict_hist, s_ns[~s_ns['Feature'].str.contains('_id')],on=["Feature"])

shap_conflict_hist_s=shap_conflict_hist.iloc[:, 1:]
colors=["forestgreen","lightgreen","steelblue","lightblue","purple","violet"]


y = np.arange(len(shap_conflict_hist_s.index))
bar_height = 0.8
bottom = np.zeros(len(shap_conflict_hist_s.index))

fig, ax = plt.subplots(figsize=(10, 6))

for i, col in enumerate(shap_conflict_hist_s.columns):
    ax.barh(y, shap_conflict_hist_s[col], left=bottom, height=bar_height, label=col, color=colors[i])
    bottom += shap_conflict_hist_s[col]

# Create custom legend
handles = [
    mpatches.Patch(color='forestgreen', label='Protest'),
    mpatches.Patch(color='lightgreen', label='Riots'),
    mpatches.Patch(color='steelblue', label='Remote Violence'),
    mpatches.Patch(color='lightblue', label='Civil Conflict'),
    mpatches.Patch(color='purple', label='Non-state'),
    mpatches.Patch(color='violet', label='One-sided')]

legend1=ax.legend(handles=handles, title='Outcomes', loc='center left', bbox_to_anchor=(1, 0.7),frameon=False,fontsize=15,title_fontsize=15)
ax.set_yticks([0,1,2,3,4],["t-1 lag","Time since","Neighborhood","Regime duration","Refugee population"],size=15)
ax.set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6],[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6],size=15)

ax.set_xlabel("SHAP Values",size=20)

plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/imp_conflict_history.png",dpi=400,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/imp_conflict_history.png",dpi=400,bbox_inches="tight")
plt.savefig("out/imp_conflict_history.png",dpi=400,bbox_inches="tight")
plt.show()

base=pd.read_csv("data/data_out/acled_cy_protest.csv",index_col=0)
base = base[["year","gw_codes"]][~base['gw_codes'].isin(list(exclude.values()))]
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

fig,ax = plt.subplots(figsize=(12, 8))
x_vals=x[["d_protest_lag1"]].reset_index(drop=True)
y_vals=pd.DataFrame(history_protest_shap[:,:,1])[0].reset_index(drop=True)
violin_parts=ax.violinplot(y_vals.loc[x_vals["d_protest_lag1"] == 0],positions=[0], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
for pc in violin_parts['bodies']:
    pc.set_facecolor(colors[0])
violin_parts=ax.violinplot(y_vals.loc[x_vals["d_protest_lag1"] == 1],positions=[1], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
for pc in violin_parts['bodies']:
    pc.set_facecolor(colors[0])
    
x_vals=x[["d_riot_lag1"]].reset_index(drop=True)
y_vals=pd.DataFrame(history_riot_shap[:,:,1])[0].reset_index(drop=True)
violin_parts=ax.violinplot(y_vals.loc[x_vals["d_riot_lag1"] == 0],positions=[0], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
for pc in violin_parts['bodies']:
    pc.set_facecolor(colors[1])
violin_parts=ax.violinplot(y_vals.loc[x_vals["d_riot_lag1"] == 1],positions=[1], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
for pc in violin_parts['bodies']:
    pc.set_facecolor(colors[1])
    
x_vals=x[["d_remote_lag1"]].reset_index(drop=True)
y_vals=pd.DataFrame(history_terror_shap[:,:,1])[0].reset_index(drop=True)
violin_parts=ax.violinplot(y_vals.loc[x_vals["d_remote_lag1"] == 0],positions=[0], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
for pc in violin_parts['bodies']:
    pc.set_facecolor(colors[2])
violin_parts=ax.violinplot(y_vals.loc[x_vals["d_remote_lag1"] == 1],positions=[1], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
for pc in violin_parts['bodies']:
    pc.set_facecolor(colors[2])
        
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
x_vals=x[["d_sb_lag1"]].reset_index(drop=True)
y_vals=pd.DataFrame(history_sb_shap[:,:,1])[0].reset_index(drop=True)
violin_parts=ax.violinplot(y_vals.loc[x_vals["d_sb_lag1"] == 0],positions=[0], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
for pc in violin_parts['bodies']:
    pc.set_facecolor(colors[3])
violin_parts=ax.violinplot(y_vals.loc[x_vals["d_sb_lag1"] == 1],positions=[1], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
for pc in violin_parts['bodies']:
    pc.set_facecolor(colors[3])
    
x_vals=x[["d_ns_lag1"]].reset_index(drop=True)
y_vals=pd.DataFrame(history_ns_shap[:,:,1])[0].reset_index(drop=True)
violin_parts=ax.violinplot(y_vals.loc[x_vals["d_ns_lag1"] == 0],positions=[0], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
for pc in violin_parts['bodies']:
    pc.set_facecolor(colors[4])
violin_parts=ax.violinplot(y_vals.loc[x_vals["d_ns_lag1"] == 1],positions=[1], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
for pc in violin_parts['bodies']:
    pc.set_facecolor(colors[4])
                
x_vals=x[["d_osv_lag1"]].reset_index(drop=True)
y_vals=pd.DataFrame(history_osv_shap[:,:,1])[0].reset_index(drop=True)
violin_parts=ax.violinplot(y_vals.loc[x_vals["d_osv_lag1"] == 0],positions=[0], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
for pc in violin_parts['bodies']:
    pc.set_facecolor(colors[5])
violin_parts=ax.violinplot(y_vals.loc[x_vals["d_osv_lag1"] == 1],positions=[1], widths=0.5,showmeans=False, showextrema=False, showmedians=False)
for pc in violin_parts['bodies']:
    pc.set_facecolor(colors[5])
                    
# Create custom legend
handles = [
    mpatches.Patch(color='forestgreen', label='Protest'),
    mpatches.Patch(color='lightgreen', label='Riots'),
    mpatches.Patch(color='steelblue', label='Remote Violence'),
    mpatches.Patch(color='lightblue', label='Civil Conflict'),
    mpatches.Patch(color='purple', label='Non-state'),
    mpatches.Patch(color='violet', label='One-sided')]

legend1=ax.legend(handles=handles, title='Outcomes', loc='center left', bbox_to_anchor=(1, 0.7),frameon=False,fontsize=15,title_fontsize=15)
                
plt.xlabel("t-1 lag",size=20)
plt.ylabel("SHAP value",size=20)
plt.yticks(size=20)
plt.xticks([0,1],["0","1"],size=20)
plt.xlim(-0.5, 1.5)  
   
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/shap_scatter_hist.png",dpi=400,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/shap_scatter_hist.png",dpi=400,bbox_inches="tight")
plt.savefig("out/shap_scatter_hist.png",dpi=400,bbox_inches="tight")
plt.show()             


### Demography ###


s_protest = pd.DataFrame(list(zip(demog_theme, np.abs(demog_protest_shap[:, :, 1]).mean(0))),columns=['Feature','Protest'])
shap_demog_hist = s_protest[~s_protest['Feature'].str.contains('_id')]

s_riot = pd.DataFrame(list(zip(demog_theme, np.abs(demog_riot_shap[:, :, 1]).mean(0))),columns=['Feature','Riot'])
shap_demog_hist = pd.merge(shap_demog_hist, s_riot[~s_riot['Feature'].str.contains('_id')],on=["Feature"])

s_remote = pd.DataFrame(list(zip(demog_theme, np.abs(demog_terror_shap[:, :, 1]).mean(0))),columns=['Feature','Remote'])
shap_demog_hist = pd.merge(shap_demog_hist, s_remote[~s_remote['Feature'].str.contains('_id')],on=["Feature"])

s_sb = pd.DataFrame(list(zip(demog_theme, np.abs(demog_sb_shap[:, :, 1]).mean(0))),columns=['Feature','State-based'])
shap_demog_hist = pd.merge(shap_demog_hist, s_sb[~s_sb['Feature'].str.contains('_id')],on=["Feature"])

s_osv = pd.DataFrame(list(zip(demog_theme, np.abs(demog_osv_shap[:, :, 1]).mean(0))),columns=['Feature','One-sided'])
shap_demog_hist = pd.merge(shap_demog_hist, s_osv[~s_osv['Feature'].str.contains('_id')],on=["Feature"])

s_ns = pd.DataFrame(list(zip(demog_theme, np.abs(demog_ns_shap[:, :, 1]).mean(0))),columns=['Feature','Non-state'])
shap_demog_hist = pd.merge(shap_demog_hist, s_ns[~s_ns['Feature'].str.contains('_id')],on=["Feature"])

colors=["forestgreen","lightgreen","steelblue","lightblue","purple","violet"]
shap_demog_hist=shap_demog_hist.loc[shap_demog_hist.iloc[:, 1:].max(axis=1).sort_values()[-6:].index]
shap_demog_hist_s=shap_demog_hist.iloc[:, 1:]


y = np.arange(len(shap_demog_hist_s.index))
bar_height = 0.8
bottom = np.zeros(len(shap_demog_hist_s.index))

fig, ax = plt.subplots(figsize=(10, 6))

for i, col in enumerate(shap_demog_hist_s.columns):
    ax.barh(y, shap_demog_hist_s[col], left=bottom, height=bar_height, label=col, color=colors[i])
    bottom += shap_demog_hist_s[col]

# Create custom legend
handles = [
    mpatches.Patch(color='forestgreen', label='Protest'),
    mpatches.Patch(color='lightgreen', label='Riots'),
    mpatches.Patch(color='steelblue', label='Remote Violence'),
    mpatches.Patch(color='lightblue', label='Civil Conflict'),
    mpatches.Patch(color='purple', label='Non-state'),
    mpatches.Patch(color='violet', label='One-sided')]

legend1=ax.legend(handles=handles, title='Outcomes', loc='center left', bbox_to_anchor=(1, 0.7),frameon=False,fontsize=15,title_fontsize=15)
ax.set_yticks([0,1,2,3,4,5],["Population density","Male population, 20-24","Male population, 0-14","Religious fractionalization","Population size","Male population, 15-19"],size=15)
ax.set_xticks([0,0.1,0.2,0.3,0.4,0.5],[0,0.1,0.2,0.3,0.4,0.5],size=15)

ax.set_xlabel("SHAP Values",size=20)

plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/imp_demog.png",dpi=400,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/imp_demog.png",dpi=400,bbox_inches="tight")
plt.savefig("out/imp_demog.png",dpi=400,bbox_inches="tight")
plt.show()

base=pd.read_csv("data/data_out/acled_cy_protest.csv",index_col=0)
base = base[["year","gw_codes"]][~base['gw_codes'].isin(list(exclude.values()))]
x=pd.read_csv("out/df_demog_full.csv",index_col=0)
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")
x["pop"]=np.log(x["pop"]+1)

fig,ax = plt.subplots(figsize=(12, 8))
plt.scatter(x[["pop"]],pd.DataFrame(demog_protest_shap[:,:,1])[0],color=colors[0],s=60)
sns.rugplot(x["pop"], color=colors[0],height=0.06)
plt.scatter(x[["pop"]],pd.DataFrame(demog_riot_shap[:,:,1])[0],color=colors[1],s=60)
sns.rugplot(x["pop"], color=colors[1],height=0.05)
plt.scatter(x[["pop"]],pd.DataFrame(demog_terror_shap[:,:,1])[0],color=colors[2],s=60)
sns.rugplot(x["pop"], color=colors[2],height=0.04)

x=pd.read_csv("out/df_demog_full.csv",index_col=0)
x["pop"]=np.log(x["pop"]+1)
plt.scatter(x[["pop"]],pd.DataFrame(demog_sb_shap[:,:,1])[0],color=colors[3],s=60)
sns.rugplot(x["pop"], color=colors[3],height=0.03)
plt.scatter(x[["pop"]],pd.DataFrame(demog_ns_shap[:,:,1])[0],color=colors[4],s=60)
sns.rugplot(x["pop"], color=colors[4],height=0.02)
plt.scatter(x[["pop"]],pd.DataFrame(demog_osv_shap[:,:,1])[0],color=colors[5],s=60)
sns.rugplot(x["pop"], color=colors[5],height=0.01)

plt.xlabel("Population size",size=20)
plt.ylabel("SHAP value",size=20)
# Create custom legend
handles = [
    mpatches.Patch(color='forestgreen', label='Protest'),
    mpatches.Patch(color='lightgreen', label='Riots'),
    mpatches.Patch(color='steelblue', label='Remote Violence'),
    mpatches.Patch(color='lightblue', label='Civil Conflict'),
    mpatches.Patch(color='purple', label='Non-state'),
    mpatches.Patch(color='violet', label='One-sided')]

legend1=ax.legend(handles=handles, title='Outcomes', loc='center left', bbox_to_anchor=(1, 0.7),frameon=False,fontsize=15,title_fontsize=15)
ax.set_yticks([-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6],[-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6],size=15)
ax.set_xticks([12,13,14,15,16,17,18,19,20,21,22],[12,13,14,15,16,17,18,19,20,21,22],size=15)
ax.set_ylim(-0.3, 0.6)                
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/shap_scatter_pop.png",dpi=400,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/shap_scatter_pop.png",dpi=400,bbox_inches="tight")
plt.savefig("out/shap_scatter_pop.png",dpi=400,bbox_inches="tight")
plt.show()

base=pd.read_csv("data/data_out/acled_cy_protest.csv",index_col=0)
base = base[["year","gw_codes"]][~base['gw_codes'].isin(list(exclude.values()))]
x=pd.read_csv("out/df_demog_full.csv",index_col=0)
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")

fig,ax = plt.subplots(figsize=(12, 8))
plt.scatter(x[["pop_male_share_0_14"]],pd.DataFrame(demog_protest_shap[:,:,1])[6],color=colors[0],s=60)
sns.rugplot(x["pop_male_share_0_14"], color=colors[0],height=0.06)
plt.scatter(x[["pop_male_share_0_14"]],pd.DataFrame(demog_riot_shap[:,:,1])[6],color=colors[1],s=60)
sns.rugplot(x["pop_male_share_0_14"], color=colors[1],height=0.05)


plt.scatter(x[["pop_male_share_0_14"]],pd.DataFrame(demog_terror_shap[:,:,1])[6],color=colors[2],s=60)
sns.rugplot(x["pop_male_share_0_14"], color=colors[2],height=0.04)

x=pd.read_csv("out/df_demog_full.csv",index_col=0)
plt.scatter(x[["pop_male_share_0_14"]],pd.DataFrame(demog_sb_shap[:,:,1])[6],color=colors[3],s=60)
sns.rugplot(x["pop_male_share_0_14"], color=colors[3],height=0.03)
plt.scatter(x[["pop_male_share_0_14"]],pd.DataFrame(demog_ns_shap[:,:,1])[6],color=colors[4],s=60)
sns.rugplot(x["pop_male_share_0_14"], color=colors[4],height=0.02)
plt.scatter(x[["pop_male_share_0_14"]],pd.DataFrame(demog_osv_shap[:,:,1])[6],color=colors[5],s=60)
sns.rugplot(x["pop_male_share_0_14"], color=colors[5],height=0.01)

plt.xlabel("Male population, 0-14",size=20)
plt.ylabel("SHAP value",size=20)
# Create custom legend
handles = [
    mpatches.Patch(color='forestgreen', label='Protest'),
    mpatches.Patch(color='lightgreen', label='Riots'),
    mpatches.Patch(color='steelblue', label='Remote Violence'),
    mpatches.Patch(color='lightblue', label='Civil Conflict'),
    mpatches.Patch(color='purple', label='Non-state'),
    mpatches.Patch(color='violet', label='One-sided')]

legend1=ax.legend(handles=handles, title='Outcomes', loc='center left', bbox_to_anchor=(1, 0.7),frameon=False,fontsize=15,title_fontsize=15)
#ax.set_yticks([-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6],[-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6],size=15)
#ax.set_xticks([12,13,14,15,16,17,18,19,20,21,22],[12,13,14,15,16,17,18,19,20,21,22],size=15)
ax.set_ylim(-0.2, 0.3)                
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/shap_scatter_pop_male.png",dpi=400,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/shap_scatter_pop_male.png",dpi=400,bbox_inches="tight")
plt.savefig("out/shap_scatter_pop_male.png",dpi=400,bbox_inches="tight")
plt.show()



### Environment ###





s_protest = pd.DataFrame(list(zip(geog_theme, np.abs(geog_protest_shap[:, :, 1]).mean(0))),columns=['Feature','Protest'])
shap_geog_hist = s_protest[~s_protest['Feature'].str.contains('_id')]

s_riot = pd.DataFrame(list(zip(geog_theme, np.abs(geog_riot_shap[:, :, 1]).mean(0))),columns=['Feature','Riot'])
shap_geog_hist = pd.merge(shap_geog_hist, s_riot[~s_riot['Feature'].str.contains('_id')],on=["Feature"])

s_remote = pd.DataFrame(list(zip(geog_theme, np.abs(geog_terror_shap[:, :, 1]).mean(0))),columns=['Feature','Remote'])
shap_geog_hist = pd.merge(shap_geog_hist, s_remote[~s_remote['Feature'].str.contains('_id')],on=["Feature"])

s_sb = pd.DataFrame(list(zip(geog_theme, np.abs(geog_sb_shap[:, :, 1]).mean(0))),columns=['Feature','State-based'])
shap_geog_hist = pd.merge(shap_geog_hist, s_sb[~s_sb['Feature'].str.contains('_id')],on=["Feature"])

s_osv = pd.DataFrame(list(zip(geog_theme, np.abs(geog_osv_shap[:, :, 1]).mean(0))),columns=['Feature','One-sided'])
shap_geog_hist = pd.merge(shap_geog_hist, s_osv[~s_osv['Feature'].str.contains('_id')],on=["Feature"])

s_ns = pd.DataFrame(list(zip(geog_theme, np.abs(geog_ns_shap[:, :, 1]).mean(0))),columns=['Feature','Non-state'])
shap_geog_hist = pd.merge(shap_geog_hist, s_ns[~s_ns['Feature'].str.contains('_id')],on=["Feature"])

colors=["forestgreen","lightgreen","steelblue","lightblue","purple","violet"]
shap_geog_hist=shap_geog_hist.loc[shap_geog_hist.iloc[:, 1:].max(axis=1).sort_values()[-6:].index]
shap_geog_hist_s=shap_geog_hist.iloc[:, 1:]


y = np.arange(len(shap_geog_hist_s.index))
bar_height = 0.8
bottom = np.zeros(len(shap_geog_hist_s.index))

fig, ax = plt.subplots(figsize=(10, 6))

for i, col in enumerate(shap_geog_hist_s.columns):
    ax.barh(y, shap_geog_hist_s[col], left=bottom, height=bar_height, label=col, color=colors[i])
    bottom += shap_geog_hist_s[col]

# Create custom legend
handles = [
    mpatches.Patch(color='forestgreen', label='Protest'),
    mpatches.Patch(color='lightgreen', label='Riots'),
    mpatches.Patch(color='steelblue', label='Remote Violence'),
    mpatches.Patch(color='lightblue', label='Civil Conflict'),
    mpatches.Patch(color='purple', label='Non-state'),
    mpatches.Patch(color='violet', label='One-sided')]

legend1=ax.legend(handles=handles, title='Outcomes', loc='center left', bbox_to_anchor=(1, 0.7),frameon=False,fontsize=15,title_fontsize=15)
ax.set_yticks([0,1,2,3,4,5],["Africa","Land area","Neighbor non-democratic","CO2","Waterstress","Percipitation"],size=15)
ax.set_xticks([0,0.1,0.2,0.3],[0,0.1,0.2,0.3],size=15)

ax.set_xlabel("SHAP Values",size=20)

plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/imp_geog.png",dpi=400,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/imp_geog.png",dpi=400,bbox_inches="tight")
plt.savefig("out/imp_geog.png",dpi=400,bbox_inches="tight")
plt.show()



base=pd.read_csv("data/data_out/acled_cy_protest.csv",index_col=0)
base = base[["year","gw_codes"]][~base['gw_codes'].isin(list(exclude.values()))]
x=pd.read_csv("out/df_geog_full.csv",index_col=0)
x=pd.merge(left=base,right=x,on=["year","gw_codes"],how="left")
x["waterstress"]=np.log(x["waterstress"]+1)

fig,ax = plt.subplots(figsize=(12, 8))
plt.scatter(x[["waterstress"]],pd.DataFrame(geog_protest_shap[:,:,1])[9],color=colors[0],s=60)
sns.rugplot(x["waterstress"], color=colors[0],height=0.06)
plt.scatter(x[["waterstress"]],pd.DataFrame(geog_riot_shap[:,:,1])[9],color=colors[1],s=60)
sns.rugplot(x["waterstress"], color=colors[1],height=0.05)
plt.scatter(x[["waterstress"]],pd.DataFrame(geog_terror_shap[:,:,1])[9],color=colors[2],s=60)
sns.rugplot(x["waterstress"], color=colors[2],height=0.04)

x=pd.read_csv("out/df_geog_full.csv",index_col=0)
x["waterstress"]=np.log(x["waterstress"]+1)

plt.scatter(x[["waterstress"]],pd.DataFrame(geog_sb_shap[:,:,1])[9],color=colors[3],s=60)
sns.rugplot(x["waterstress"], color=colors[3],height=0.03)
plt.scatter(x[["waterstress"]],pd.DataFrame(geog_ns_shap[:,:,1])[9],color=colors[4],s=60)
sns.rugplot(x["waterstress"], color=colors[4],height=0.02)
plt.scatter(x[["waterstress"]],pd.DataFrame(geog_osv_shap[:,:,1])[9],color=colors[5],s=60)
sns.rugplot(x["waterstress"], color=colors[5],height=0.01)

plt.xlabel("Male population, 0-14",size=20)
plt.ylabel("SHAP value",size=20)
# Create custom legend
handles = [
    mpatches.Patch(color='forestgreen', label='Protest'),
    mpatches.Patch(color='lightgreen', label='Riots'),
    mpatches.Patch(color='steelblue', label='Remote Violence'),
    mpatches.Patch(color='lightblue', label='Civil Conflict'),
    mpatches.Patch(color='purple', label='Non-state'),
    mpatches.Patch(color='violet', label='One-sided')]

legend1=ax.legend(handles=handles, title='Outcomes', loc='center left', bbox_to_anchor=(1, 0.7),frameon=False,fontsize=15,title_fontsize=15)
#ax.set_yticks([-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6],[-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6],size=15)
#ax.set_xticks([12,13,14,15,16,17,18,19,20,21,22],[12,13,14,15,16,17,18,19,20,21,22],size=15)
ax.set_ylim(-0.2, 0.3)                
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/structural/out/shap_scatter_geog.png",dpi=400,bbox_inches="tight")
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_Dissertation/out/shap_scatter_geog.png",dpi=400,bbox_inches="tight")
plt.savefig("out/shap_scatter_geog.png",dpi=400,bbox_inches="tight")
plt.show()







