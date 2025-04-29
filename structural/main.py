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
import matplotlib.gridspec as gridspec

import os
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'

# Simple random forest
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)],
               'max_features': ["sqrt", "log2", None],
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],
               'min_samples_split': [2,5,10],
               'min_samples_leaf': [1,2,4],
               }

names={'d_protest_lag1':"t-1 lag of at least one protest event",
       "d_protest_zeros_decay":"Time since last protest event",
       "d_neighbors_proteset_event_counts_lag1":"One neighbour had one protest event in previous year (t-1 lag)",
       'd_riot_lag1':"t-1 lag of at least one riot event",
       "d_riot_zeros_decay":"Time since last riot event",
       "d_neighbors_riot_event_counts_lag1":"One neighbour had one riot event in previous year (t-1 lag)",
       'd_terror_lag1':"t-1 lag of at least one fatality from terrorism",
       "d_terror_zeros_decay":"Time since last terrorism fatalities",
       "d_neighbors_terrorism_fatalities_lag1":"One neighbour had one fatality from terrorism in previous year (t-1 lag)",
       'd_sb_lag1':"t-1 lag of at least one fatality in state-based violence",
       "d_sb_zeros_decay":"Time since last state-based violence fatalities",
       "d_neighbors_sb_fatalities_lag1":"One neighbour had one fatality from state-based violence in previous year (t-1 lag)",
       'd_ns_lag1':"t-1 lag of at least one fatality in non-state violence",
       "d_ns_zeros_decay":"Time since last non-state violence fatalities in years",
       "d_neighbors_ns_fatalities_lag1":"One neighbour had one fatality from non-state violence in previous year (t-1 lag)",
       'd_osv_lag1':"t-1 lag of at least one fatality in one-sided violence",
       "d_osv_zeros_decay":"Time since last one-sided violence fatalities",
       "d_neighbors_osv_fatalities_lag1":"One neighbour had one fatality from one-sided violence in previous year (t-1 lag)", 
       "regime_duration":"Years since independence",
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
       'lastelection_id':"Time since the last election for leadership (nan imp)",
       "d_zeros_decay":"Time since last event/fatality",
       "d_lag1":"t-1 lag of outcome",
       "d_neighbors_lag1":"One neighbour had one fatality/event"
       }


# Inputs
demog_theme=['pop','pop_density','pop_density_id','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','group_counts_id','monopoly_share','monopoly_share_id','discriminated_share','discriminated_share_id','powerless_share','powerless_share_id','dominant_share','dominant_share_id','ethnic_frac','ethnic_frac_id','rel_frac','rel_frac_id','lang_frac','lang_frac_id','race_frac','race_frac_id']
geog_theme=['land','forest','forest_id','temp_norm','temp_id','co2','co2_id','percip','percip_id','waterstress','waterstress_id','agri_land','agri_land_id','arable_land','arable_land_id','rugged','soil','desert','tropical','cont_africa','cont_asia','d_neighbors_con','no_neigh','d_neighbors_non_dem','libdem_id_neigh']
econ_theme=['oil_deposits','oil_deposits_id','oil_production','oil_production_id','oil_exports','oil_exports_id','natres_share','natres_share_id','oil_share','oil_share_id','gas_share','gas_share_id','coal_share','coal_share_id','forest_share','forest_share_id','minerals_share','minerals_share_id','gdp','gdp_id','gni','gni_id','gdp_growth_norm','gdp_growth_id','unemploy','unemploy_id','unemploy_male','unemploy_male_id','inflat','inflat_id','conprice','conprice_id','undernour','undernour_id','foodprod_norm','foodprod_id','water_rural','water_rural_id','water_urb','water_urb_id','agri_share','agri_share_id','trade_share','trade_share_id','fert','lifeexp_female','lifeexp_male','pop_growth_norm','pop_growth_id','inf_mort','mig_norm','exports','exports_id','imports','imports_id','primary_female','primary_female_id','primary_male','primary_male_id','second_female','second_female_id','second_male','second_male_id','tert_female','tert_female_id','tert_male','tert_male_id','eys','eys_id','eys_male','eys_male_id','eys_female','eys_female_id','mys','mys_id','mys_male','mys_male_id','mys_female','mys_female_id']
#econ_theme=['oil_deposits','oil_deposits_id','oil_production','oil_production_id','oil_exports','oil_exports_id','natres_share','natres_share_id','oil_share','oil_share_id','gas_share','gas_share_id','coal_share','coal_share_id','forest_share','forest_share_id','minerals_share','minerals_share_id','gdp','gdp_id','gni','gni_id','gdp_growth_norm','gdp_growth_id','unemploy','unemploy_id','unemploy_male','unemploy_male_id','inflat','inflat_id','conprice','conprice_id','undernour','undernour_id','foodprod_norm','foodprod_id','water_rural','water_rural_id','water_urb','water_urb_id','trade_share','trade_share_id','fert','lifeexp_female','lifeexp_male','pop_growth_norm','pop_growth_id','inf_mort','exports','exports_id','imports','imports_id','primary_female','primary_female_id','primary_male','primary_male_id','second_female','second_female_id','second_male','second_male_id','tert_female','tert_female_id','tert_male','tert_male_id','mys','mys_id','mys_male','mys_male_id','mys_female','mys_female_id']
pol_theme=['armedforces_share','armedforces_share_id','milex_share','milex_share_id','corruption','corruption_id', 'effectiveness', 'effectiveness_id', 'polvio','polvio_id','regu','regu_id','law','law_id','account','account_id','tax','tax_id','broadband','broadband_id','telephone','telephone_id','internet_use','internet_use_id','mobile','mobile_id','polyarchy','libdem','libdem_id','partipdem','delibdem','egaldem','civlib','phyvio','pollib','privlib','execon','execon_id','exgender','exgender_id','exgeo','exgeo_id','expol','expol_id','exsoc','exsoc_id','shutdown','shutdown_id','filter','filter_id','tenure_months','tenure_months_id','dem_duration','dem_duration_id','elections','elections_id','lastelection','lastelection_id']
#pol_theme=['armedforces_share','armedforces_share_id','milex_share','milex_share_id','corruption','corruption_id', 'effectiveness', 'effectiveness_id','regu','regu_id','law','law_id','account','account_id','tax','tax_id','broadband','broadband_id','telephone','telephone_id','internet_use','internet_use_id','mobile','mobile_id','polyarchy','libdem','libdem_id','partipdem','delibdem','egaldem','civlib','pollib','privlib','execon','execon_id','exgender','exgender_id','exgeo','exgeo_id','expol','expol_id','exsoc','exsoc_id','shutdown','shutdown_id','filter','filter_id','tenure_months','tenure_months_id','dem_duration','dem_duration_id','elections','elections_id','lastelection','lastelection_id']

# Check downsampling
y=pd.read_csv("out/df_out_full.csv",index_col=0)
fig,ax = plt.subplots()
y["d_civil_war"].hist()
fig,ax = plt.subplots()
y["d_civil_conflict"].hist()
fig,ax = plt.subplots()
y["d_terror"].hist()
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
for var in ['d_civil_war_lag1','d_civil_war_lag1',"d_civil_war_zeros_decay","d_neighbors_civil_war_lag1",'d_civil_conflict_lag1',"d_civil_conflict_zeros_decay","d_neighbors_civil_conflict_lag1",'d_protest_lag1',"d_protest_zeros_decay","d_neighbors_proteset_event_counts_lag1",'d_riot_lag1',"d_riot_zeros_decay","d_neighbors_riot_event_counts_lag1",'d_terror_lag1',"d_terror_zeros_decay","d_neighbors_terrorism_fatalities_lag1",'d_sb_lag1',"d_sb_zeros_decay","d_neighbors_sb_fatalities_lag1",'d_ns_lag1',"d_ns_zeros_decay","d_neighbors_ns_fatalities_lag1",'d_osv_lag1',"d_osv_zeros_decay","d_neighbors_osv_fatalities_lag1","regime_duration",'pop_refugee','pop_refugee_id']:
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
#inputs=['d_civil_war_lag1',"d_civil_war_zeros_decay","d_neighbors_civil_war_lag1",]
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
ensemble_war.to_csv("out/ensemble_war.csv")

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
#inputs=['d_civil_conflict_lag1',"d_civil_conflict_zeros_decay","d_neighbors_civil_conflict_lag1"]
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
ensemble_conflict.to_csv("out/ensemble_conflict.csv")

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
#inputs=['d_protest_lag1',"d_protest_zeros_decay","d_neighbors_proteset_event_counts_lag1"]
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
ensemble_protest.to_csv("out/ensemble_protest.csv")

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
#inputs=['d_riot_lag1',"d_riot_zeros_decay","d_neighbors_riot_event_counts_lag1"]
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
ensemble_riot.to_csv("out/ensemble_riot.csv")

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
      
                                #################
                                ### Terrorism ###
                                #################

###################
### A. Baseline ###
###################

# Load dataset
y=pd.read_csv("out/df_out_full.csv",index_col=0)
x=pd.read_csv("out/df_conf_hist_full.csv",index_col=0)
y=y.loc[y["year"]<2021]
x=x.loc[x["year"]<2021]

# Transforms
x["pop_refugee"]=np.log(x["pop_refugee"]+1)

# Categorical
target="d_terror"
inputs=['d_terror_lag1']
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
y=y.loc[y["year"]<2021]
x=x.loc[x["year"]<2021]

# Categorical
target="d_terror"
inputs=['d_terror_lag1',"d_terror_zeros_decay","d_neighbors_terrorism_fatalities_lag1",'regime_duration','pop_refugee','pop_refugee_id']
history_terror_df,history_terror_evals,history_terror_val,history_terror_shap=gen_model(y,x,target,inputs,names=names,name="history_terror",model_fit=RandomForestClassifier(random_state=0),opti_grid=random_grid,outcome="categorical",int_methods=True,inputs_plot=['d_terror_lag1',"d_terror_zeros_decay","d_neighbors_terrorism_event_counts_lag1","regime_duration",'pop_refugee'])
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
y=y.loc[y["year"]<2021]
x=x.loc[x["year"]<2021]

# Transforms
x["pop"]=np.log(x["pop"]+1)

# Categorical
target="d_terror"
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
y=y.loc[y["year"]<2021]
x=x.loc[x["year"]<2021]

# Transforms
x["land"]=np.log(x["land"]+1)
x["co2"]=np.log(x["co2"]+1)
x["waterstress"]=np.log(x["waterstress"]+1)
preprocess_min_max_group(x,"temp","country")

# Categorical
target="d_terror"
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
y=y.loc[y["year"]<2021]
x=x.loc[x["year"]<2021]

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
y=y.loc[y["year"]<2021]
x=x.loc[x["year"]<2021]

# Transforms
x["milex_share"]=np.log(x["milex_share"]+1)

# Categorical
target="d_terror"
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
weights=[1-history_terror_evals["brier"],1-demog_terror_evals["brier"],1-geog_terror_evals["brier"],1-econ_terror_evals["brier"],1-pol_terror_evals["brier"]]
weights_terror_n = [(x - min(weights)) / (max(weights) - min(weights)) for x in weights]
weights_terror_n = [x / sum(weights_terror_n) for x in weights_terror_n]
ensemble = (history_terror_df.preds_proba*weights_terror_n[0])+(demog_terror_df.preds_proba*weights_terror_n[1])+(geog_terror_df.preds_proba*weights_terror_n[2])+(econ_terror_df.preds_proba*weights_terror_n[3])+(pol_terror_df.preds_proba*weights_terror_n[4])
ensemble_terror=pd.concat([history_terror_df[["country","year","d_terror","test"]],pd.DataFrame(ensemble)],axis=1)
ensemble_terror.columns=["country","year","d_terror","test","preds_proba"]
ensemble_terror=ensemble_terror.reset_index(drop=True)
ensemble_terror.to_csv("out/ensemble_terror.csv")

# Evals
brier = brier_score_loss(ensemble_terror.loc[ensemble_terror["test"]==1].d_terror, ensemble_terror.loc[ensemble_terror["test"]==1].preds_proba)
precision, recall, _ = precision_recall_curve(ensemble_terror.loc[ensemble_terror["test"]==1].d_terror, ensemble_terror.loc[ensemble_terror["test"]==1].preds_proba)
aupr = auc(recall, precision)
auroc = roc_auc_score(ensemble_terror.loc[ensemble_terror["test"]==1].d_terror, ensemble_terror.loc[ensemble_terror["test"]==1].preds_proba)
evals_terror_ensemble = {"brier": brier, "aupr": aupr, "auroc": auroc}
evals_terror_ensemble_df = pd.DataFrame.from_dict(evals_terror_ensemble, orient='index').reset_index()
evals_terror_ensemble_df.to_csv("out/evals_terror_ensemble_df.csv")
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
#inputs=['d_sb_lag1',"d_sb_zeros_decay","d_neighbors_sb_fatalities_lag1"]
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
ensemble_sb.to_csv("out/ensemble_sb.csv")

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
#inputs=['d_osv_lag1',"d_osv_zeros_decay","d_neighbors_osv_fatalities_lag1",'regime_duration']
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
ensemble_osv.to_csv("out/ensemble_osv.csv")

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
#inputs=['d_ns_lag1',"d_ns_zeros_decay","d_neighbors_ns_fatalities_lag1"]
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
ensemble_ns.to_csv("out/ensemble_ns.csv")

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
base=base.loc[base["year"]<2021]
ensemble_sb_short=pd.merge(base, ensemble_sb,on=["year","country"],how="left")

base=ensemble_protest[["year","country"]]
base=base.loc[base["year"]<2021]
ensemble_ns_short=pd.merge(base, ensemble_ns,on=["year","country"],how="left")

base=ensemble_protest[["year","country"]]
base=base.loc[base["year"]<2021]
ensemble_osv_short=pd.merge(base, ensemble_osv,on=["year","country"],how="left")

base=ensemble_protest[["year","country"]]
base=base.loc[base["year"]<2021]
ensemble_terror_short=pd.merge(base, ensemble_terror,on=["year","country"],how="left")

ensemble_protest_s=ensemble_protest.loc[ensemble_protest["year"]<2021]
ensemble_protest_s=ensemble_protest_s.reset_index(drop=True)
ensemble_riot_s=ensemble_riot.loc[ensemble_riot["year"]<2021]
ensemble_riot_s=ensemble_riot_s.reset_index(drop=True)

ensemble = (ensemble_protest_s.preds_proba*weights_n[0])+(ensemble_riot_s.preds_proba*weights_n[1])+(ensemble_terror_short.preds_proba*weights_n[2])+(ensemble_sb_short.preds_proba*weights_n[3])+(ensemble_ns_short.preds_proba*weights_n[4])+(ensemble_osv_short.preds_proba*weights_n[5])
ensemble_ens=pd.concat([ensemble_protest_s[["country","year"]],pd.DataFrame(ensemble)],axis=1)
ensemble_ens.columns=["country","year","preds_proba"]

### sb, ns, osv, terrorism  ###
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

### Year 2021-2023 sb, ns, osv, protest, riots ###

base=ensemble_protest[["year","country"]]
base=base.loc[base["year"]<=2021]
ensemble_sb_short=pd.merge(base, ensemble_sb,on=["year","country"],how="left")

base=ensemble_protest[["year","country"]]
base=base.loc[base["year"]<=2021]
ensemble_ns_short=pd.merge(base, ensemble_ns,on=["year","country"],how="left")

base=ensemble_protest[["year","country"]]
base=base.loc[base["year"]<=2021]
ensemble_osv_short=pd.merge(base, ensemble_osv,on=["year","country"],how="left")

base=ensemble_protest[["year","country"]]
base=base.loc[base["year"]<=2021]
ensemble_protest_short=pd.merge(base, ensemble_protest,on=["year","country"],how="left")

base=ensemble_protest[["year","country"]]
base=base.loc[base["year"]<=2021]
ensemble_riot_short=pd.merge(base, ensemble_riot,on=["year","country"],how="left")

ensemble = (ensemble_protest_short.preds_proba*weights_n[0])+(ensemble_riot_short.preds_proba*weights_n[1])+(ensemble_sb_short.preds_proba*weights_n[3])+(ensemble_ns_short.preds_proba*weights_n[4])+(ensemble_osv_short.preds_proba*weights_n[5])
ensemble_ens_2023=pd.concat([ensemble_protest_short[["country","year"]],pd.DataFrame(ensemble)],axis=1)
ensemble_ens_2023.columns=["country","year","preds_proba"]

ensemble_ens=pd.concat([ensemble_ens,ensemble_ens_2023],axis=0)
ensemble_ens=ensemble_ens.sort_values(by=["country","year"])

ensemble_ens["dummy"]=0
ensemble_ens.loc[ensemble_ens["preds_proba"]>=0.5,"dummy"]=1
ensemble_ens.to_csv("out/ensemble_ens_df.csv")
