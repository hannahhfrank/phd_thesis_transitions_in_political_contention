import pandas as pd
import os
import numpy as np
from functions import dichotomize,lag_groupped,consec_zeros_grouped,get_wb,simple_imp_grouped,linear_imp_grouped
import matplotlib.pyplot as plt
import requests
import pyreadstat

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

############
### UCDP ###
############

ucdp_sb=pd.read_csv("data/data_out/ucdp_cy_sb.csv",index_col=0)
df = ucdp_sb[["year","gw_codes","country","best"]][~ucdp_sb['gw_codes'].isin(list(micro_states.values())+list(exclude.values()))]
df.columns=["year","gw_codes","country","sb_fatalities"]

######################
### Peace duration ###
######################

# Time since last civil conflict

dichotomize(df,"sb_fatalities","d_civil_conflict",25)
df['time_since_civil_conflict'] = consec_zeros_grouped(df,'country','d_civil_conflict')
df = df.drop('d_civil_conflict', axis=1)

#########################
### Previous conflict ### 
#########################

# t-1 lag of fatalities

df["sb_fatalities_lag1"]=lag_groupped(df,"country","sb_fatalities",1)

#############################
### Simultaneous conflict ### 
#############################

# t-1 lag of fatalities of all other types of violence

ucdp_sb_gov=pd.read_csv("data/data_out/ucdp_cy_sb_gov.csv",index_col=0)
ucdp_sb_gov=ucdp_sb_gov[["year","country","gw_codes","best"]][~ucdp_sb['gw_codes'].isin(list(micro_states.values())+list(exclude.values()))]
ucdp_sb_gov.columns=["year","country","gw_codes","sb_gov_fatalities"]

ucdp_osv=pd.read_csv("data/data_out/ucdp_cy_osv.csv",index_col=0)
ucdp_osv = ucdp_osv[["year","gw_codes","best"]][~ucdp_osv['gw_codes'].isin(list(micro_states.values())+list(exclude.values()))]
ucdp_osv.columns=["year","gw_codes","osv_fatalities"]

ucdp_ns=pd.read_csv("data/data_out/ucdp_cy_ns.csv",index_col=0)
ucdp_ns = ucdp_ns[["year","gw_codes","best"]][~ucdp_ns['gw_codes'].isin(list(micro_states.values())+list(exclude.values()))]
ucdp_ns.columns=["year","gw_codes","ns_fatalities"]

ucdp_other=pd.merge(left=ucdp_sb_gov,right=ucdp_osv[["year","gw_codes","osv_fatalities"]],on=["year","gw_codes"],how="left")
ucdp_other=pd.merge(left=ucdp_other,right=ucdp_ns[["year","gw_codes","ns_fatalities"]],on=["year","gw_codes"],how="left")
ucdp_other=ucdp_other.reset_index()

ucdp_other['total_fatalities'] = ucdp_other[["sb_gov_fatalities","osv_fatalities","ns_fatalities"]].sum(axis=1)
ucdp_other["total_fatalities_lag1"]=lag_groupped(ucdp_other,"country","total_fatalities",1)
df=pd.merge(left=df,right=ucdp_other[["year","gw_codes","total_fatalities_lag1"]],on=["year","gw_codes"],how="left")

##########################
### Recent independent ### 
##########################

# F&L: "we mark countries in their first and second years of independence".

d={"country":[],"gw_codes":[],"start":[],"end":[]}

response = requests.get("http://ksgleditsch.com/data/iisystem.dat")
file_content = response.text
lines = file_content.splitlines()

for i in range(len(lines)):
    split = lines[i].split("\t")
    d["gw_codes"].append(int(split[0]))
    d["country"].append(split[2])
    d["start"].append(int(split[3][6:]))
    d["end"].append(int(split[4][6:]))


response = requests.get("http://ksgleditsch.com/data/microstatessystem.dat")
file_content = response.text
lines = file_content.splitlines()

for i in range(len(lines)):
    split = lines[i].split("\t")
    d["gw_codes"].append(int(split[0]))
    d["country"].append(split[2])
    d["start"].append(int(split[3][6:]))
    d["end"].append(int(split[4][6:]))

all_countries=pd.DataFrame(d)
all_countries=all_countries.loc[all_countries["end"]>=1989]

df=pd.merge(left=df,right=all_countries[["gw_codes","start"]],on=["gw_codes"],how="left")
df["time_since_independ"]=df["year"]-df["start"]
df = df.drop('start', axis=1)

####################################
### Economic performance: growth ### 
####################################

df_econ=df[["year","gw_codes","country"]].copy()
feat_dev = ["NY.GDP.MKTP.KD.ZG", # GDP growth (annual %) 
            "NY.GDP.PCAP.CD", # GDP per capita (current US$)
            "EN.POP.DNST", # Population density (people per sq. km of land area)
            "SP.POP.TOTL", # Population size
            "NY.GDP.TOTL.RT.ZS", # Total natural resources rents (% of GDP)
            "NY.GDP.NGAS.RT.ZS", # Natural gas rents (% of GDP)
            "NY.GDP.PETR.RT.ZS", # Oil rents (% of GDP)
            "NY.GDP.COAL.RT.ZS", # Coal rents (% of GDP)
            "AG.LND.PRCP.MM", # Average precipitation in depth (mm per year)
            "AG.LND.AGRI.ZS", # Agricultural land (% of land area)
            ]

# Import country codes  
df_ccodes = pd.read_csv("data/df_ccodes.csv")
c_list=list(df_ccodes.iso_alpha3)
c_list = [char for char in c_list if char != "XYZ"]

# Load data 
economy = get_wb(list(range(1989, 2023, 1)),c_list,feat_dev)

# GDP growth (annual %)

# Taiwan, Democratic Peoples Republic of Korea, Djibouti

# Merge
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.MKTP.KD.ZG"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["NY.GDP.MKTP.KD.ZG"])
base_imp=simple_imp_grouped(base_imp,"country",["NY.GDP.MKTP.KD.ZG"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.MKTP.KD.ZG"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["NY.GDP.MKTP.KD.ZG"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()
    
base_imp = base_imp[~base_imp['country'].isin(["Taiwan", "Democratic Peoples Republic of Korea","Djibouti"])] 
df = df[~df['country'].isin(["Taiwan", "Democratic Peoples Republic of Korea","Djibouti"])] 
df=pd.merge(left=df,right=base_imp[["year","gw_codes","NY.GDP.MKTP.KD.ZG"]],on=["year","gw_codes"],how="left")
df.rename(columns={"NY.GDP.MKTP.KD.ZG": 'growth'}, inplace=True)

##############################################
### Living standards: (esp. GDP, GDP p.c.) ### 
##############################################

# GDP per capita 

# Merge
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["NY.GDP.PCAP.CD"])
base_imp=simple_imp_grouped(base_imp,"country",["NY.GDP.PCAP.CD"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.PCAP.CD"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["NY.GDP.PCAP.CD"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()
    
df=pd.merge(left=df,right=base_imp[["year","gw_codes","NY.GDP.PCAP.CD"]],on=["year","gw_codes"],how="left")
df.rename(columns={"NY.GDP.PCAP.CD": 'gdp'}, inplace=True)

##########################
### Population density ### 
##########################

# Kosovo

# Population density (people per sq. km of land area)

# Merge
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","EN.POP.DNST"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["EN.POP.DNST"])
base_imp=simple_imp_grouped(base_imp,"country",["EN.POP.DNST"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["EN.POP.DNST"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["EN.POP.DNST"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

base_imp = base_imp[~base_imp['country'].isin(["Kosovo"])] 
df = df[~df['country'].isin(["Kosovo"])] 
df=pd.merge(left=df,right=base_imp[["year","gw_codes","EN.POP.DNST"]],on=["year","gw_codes"],how="left")
df.rename(columns={"EN.POP.DNST": 'pop_density'}, inplace=True)

#######################
### Population size ### 
#######################

# Population size

# Merge
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","SP.POP.TOTL"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["SP.POP.TOTL"])
base_imp=simple_imp_grouped(base_imp,"country",["SP.POP.TOTL"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SP.POP.TOTL"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["SP.POP.TOTL"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df=pd.merge(left=df,right=base_imp[["year","gw_codes","SP.POP.TOTL"]],on=["year","gw_codes"],how="left")
df.rename(columns={"SP.POP.TOTL": 'pop'}, inplace=True)

#################################
### Primary commodity exports ### 
#################################

# Total natural resources rents (% of GDP)

# Merge
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["NY.GDP.TOTL.RT.ZS"])
base_imp=simple_imp_grouped(base_imp,"country",["NY.GDP.TOTL.RT.ZS"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.TOTL.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["NY.GDP.TOTL.RT.ZS"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df=pd.merge(left=df,right=base_imp[["year","gw_codes","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")
df.rename(columns={"NY.GDP.TOTL.RT.ZS": 'nat_res'}, inplace=True)

##########################
### Hydrocarbon export ### 
##########################

# Natural gas rents (% of GDP)

# Sierra Leone

# Merge
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.NGAS.RT.ZS"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["NY.GDP.NGAS.RT.ZS"])
base_imp=simple_imp_grouped(base_imp,"country",["NY.GDP.NGAS.RT.ZS"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.NGAS.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["NY.GDP.NGAS.RT.ZS"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

base_imp = base_imp[~base_imp['country'].isin(["Sierra Leone"])] 
df = df[~df['country'].isin(["Sierra Leone"])] 
df=pd.merge(left=df,right=base_imp[["year","gw_codes","NY.GDP.NGAS.RT.ZS"]],on=["year","gw_codes"],how="left")

# Oil rents (% of GDP)

# Merge
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["NY.GDP.PETR.RT.ZS"])
base_imp=simple_imp_grouped(base_imp,"country",["NY.GDP.PETR.RT.ZS"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.PETR.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["NY.GDP.PETR.RT.ZS"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df=pd.merge(left=df,right=base_imp[["year","gw_codes","NY.GDP.PETR.RT.ZS"]],on=["year","gw_codes"],how="left")

# Coal rents (% of GDP)

# Merge
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.COAL.RT.ZS"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["NY.GDP.COAL.RT.ZS"])
base_imp=simple_imp_grouped(base_imp,"country",["NY.GDP.COAL.RT.ZS"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.COAL.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["NY.GDP.COAL.RT.ZS"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df=pd.merge(left=df,right=base_imp[["year","gw_codes","NY.GDP.COAL.RT.ZS"]],on=["year","gw_codes"],how="left")
df['hydro_carb'] = df[["NY.GDP.NGAS.RT.ZS","NY.GDP.PETR.RT.ZS","NY.GDP.COAL.RT.ZS"]].sum(axis=1)
df = df.drop("NY.GDP.NGAS.RT.ZS", axis=1)
df = df.drop("NY.GDP.PETR.RT.ZS", axis=1)
df = df.drop("NY.GDP.COAL.RT.ZS", axis=1)

################
### Rainfall ### 
################

# Montenegro

# Merge
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","AG.LND.PRCP.MM"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["AG.LND.PRCP.MM"])
base_imp=simple_imp_grouped(base_imp,"country",["AG.LND.PRCP.MM"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["AG.LND.PRCP.MM"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["AG.LND.PRCP.MM"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

base_imp = base_imp[~base_imp['country'].isin(["Montenegro","Serbia"])] 
df = df[~df['country'].isin(["Montenegro","Serbia"])] 
df=pd.merge(left=df,right=base_imp[["year","gw_codes","AG.LND.PRCP.MM"]],on=["year","gw_codes"],how="left")
df.rename(columns={"AG.LND.PRCP.MM": 'percip'}, inplace=True)

########################
### Soil degradation ###
########################

# Agricultural land (% of land area)

# Merge
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","AG.LND.AGRI.ZS"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["AG.LND.AGRI.ZS"])
base_imp=simple_imp_grouped(base_imp,"country",["AG.LND.AGRI.ZS"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["AG.LND.AGRI.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["AG.LND.AGRI.ZS"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df=pd.merge(left=df,right=base_imp[["year","gw_codes","AG.LND.AGRI.ZS"]],on=["year","gw_codes"],how="left")
df.rename(columns={"AG.LND.AGRI.ZS": 'agri'}, inplace=True)

########################
### Ethnic diversity ###
########################

erp=pd.read_csv("data/data_out/epr_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=erp[["year","gw_codes","ethnic_frac"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["ethnic_frac"])
base_imp=simple_imp_grouped(base_imp,"country",["ethnic_frac"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["ethnic_frac"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["ethnic_frac"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df=pd.merge(left=df,right=base_imp[["year","gw_codes","ethnic_frac"]],on=["year","gw_codes"],how="left")

########################
### Ethnic dominance ### 
########################

erp=pd.read_csv("data/data_out/epr_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=erp[["year","gw_codes","dominant_share"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["dominant_share"])
base_imp=simple_imp_grouped(base_imp,"country",["dominant_share"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["dominant_share"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["dominant_share"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df=pd.merge(left=df,right=base_imp[["year","gw_codes","dominant_share"]],on=["year","gw_codes"],how="left")


###########################
### Religious diversity ### 
###########################

erp=pd.read_csv("data/data_out/epr_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=erp[["year","gw_codes","rel_frac"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["rel_frac"])
base_imp=simple_imp_grouped(base_imp,"country",["rel_frac"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["rel_frac"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["rel_frac"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df=pd.merge(left=df,right=base_imp[["year","gw_codes","rel_frac"]],on=["year","gw_codes"],how="left")

################################
### Political discrimination ### 
################################

erp=pd.read_csv("data/data_out/epr_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=erp[["year","gw_codes","discriminated_share"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["discriminated_share"])
base_imp=simple_imp_grouped(base_imp,"country",["discriminated_share"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["discriminated_share"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["discriminated_share"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df=pd.merge(left=df,right=base_imp[["year","gw_codes","discriminated_share"]],on=["year","gw_codes"],how="left")

# Alternative from v-dem

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2xpe_exlecon","v2xpe_exlpol"]],on=["year","gw_codes"],how="left")

# Exclusion by Socio-Economic Group

### Simple ###

base_imp=linear_imp_grouped(base,"country",["v2xpe_exlecon"])
base_imp=simple_imp_grouped(base_imp,"country",["v2xpe_exlecon"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2xpe_exlecon"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["v2xpe_exlecon"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

base_imp = base_imp[~base_imp['country'].isin(['Bahamas', 'Belize', 'Brunei Darussalam'])] 
df = df[~df['country'].isin(['Bahamas', 'Belize', 'Brunei Darussalam'])] 
df=pd.merge(left=df,right=base_imp[["year","gw_codes","v2xpe_exlecon"]],on=["year","gw_codes"],how="left")

# Exclusion by Political Group index

### Simple ###

base_imp=linear_imp_grouped(base,"country",["v2xpe_exlpol"])
base_imp=simple_imp_grouped(base_imp,"country",["v2xpe_exlpol"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2xpe_exlpol"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["v2xpe_exlpol"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df=pd.merge(left=df,right=base_imp[["year","gw_codes","v2xpe_exlpol"]],on=["year","gw_codes"],how="left")

######################
### Mass education ### 
######################

# Mean years of schooling

# Merge
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys_male']],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["eys_male"])
base_imp=simple_imp_grouped(base_imp,"country",["eys_male"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["eys_male"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["eys_male"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

base_imp = base_imp[~base_imp['country'].isin(["Somalia"])] 
df = df[~df['country'].isin(['Somalia'])]
df=pd.merge(left=df,right=base_imp[["year","gw_codes","eys_male"]],on=["year","gw_codes"],how="left")

#################################
### Regime level: curvilinear ### 
#################################

# Bahamas, Barbados, Belize, Malte, Serbia, Kosovo, Iceland, Maldives, Vietnam, Brunei Darussalam

polity = pd.read_excel("data/p5v2018.xls")
polity=polity[["year","ccode","country",'polity2']].loc[polity["year"]>=1989]
polity.columns=["year","gw_codes","country",'polity2']
polity=polity.reset_index(drop=True)
base=pd.merge(left=df,right=polity[["year","gw_codes","polity2"]],on=["year","gw_codes"],how="left")

### Simple ###

polity_imp=linear_imp_grouped(base,"country",["polity2"])
polity_imp=simple_imp_grouped(polity_imp,"country",["polity2"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["polity2"].loc[base["country"]==c])
#    axs[1].plot(polity_imp["year"].loc[polity_imp["country"]==c], polity_imp["polity2"].loc[polity_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

polity_imp = polity_imp[~polity_imp['country'].isin(["Bahamas","Barbados","Belize","Malta","Serbia","Kosovo","Iceland","Maldives","Vietnam","Brunei Darussalam"])] 
df = df[~df['country'].isin(["Bahamas","Barbados","Belize","Malta","Serbia","Kosovo","Iceland","Maldives","Vietnam","Brunei Darussalam"])] 
df=pd.merge(left=df,right=polity_imp[["year","gw_codes","polity2"]],on=["year","gw_codes"],how="left")

df["polity2_sqr"]=df["polity2"]**2

# Electoral democracy index

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["v2x_polyarchy"])
base_imp=simple_imp_grouped(base_imp,"country",["v2x_polyarchy"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2x_polyarchy"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["v2x_polyarchy"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df=pd.merge(left=df,right=base_imp[["year","gw_codes","v2x_polyarchy"]],on=["year","gw_codes"],how="left")
df["v2x_polyarchy_sqr"]=df["v2x_polyarchy"]**2

# Liberal democracy index

vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp=linear_imp_grouped(base,"country",["v2x_libdem"])
base_imp=simple_imp_grouped(base_imp,"country",["v2x_libdem"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2x_libdem"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["v2x_libdem"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df=pd.merge(left=df,right=base_imp[["year","gw_codes","v2x_libdem"]],on=["year","gw_codes"],how="left")
df["v2x_libdem_sqr"]=df["v2x_libdem"]**2

#####################
### Regime change ### ---> change
#####################

# F&L: “political instability at the center”; "we use a dummy variable indicating whether the country had a three-orgreater 
# change on the Polity IV regime index in any of the three years prior to the country-year 
# in question"

diff=pd.DataFrame()
for c in df.country.unique():
    df_s=df.loc[df["country"]==c]
    df_s["diff"]=np.insert(np.diff(df_s["polity2"]), 0, 0) 
    df_s['polity2_change'] = abs(df_s['diff'].rolling(window=3, min_periods=1).sum()) > 3
    df_s['polity2_change']=df_s['polity2_change'].astype(int)
    diff=pd.concat([diff,df_s])
df["polity2_change"]=diff['polity2_change']

diff=pd.DataFrame()
for c in df.country.unique():
    df_s=df.loc[df["country"]==c]
    df_s["diff"]=np.insert(np.diff(df_s["v2x_polyarchy"]), 0, 0) 
    df_s['v2x_polyarchy_change'] = df_s['diff'].rolling(window=3, min_periods=1).max() > 3
    df_s['v2x_polyarchy_change']=df_s['v2x_polyarchy_change'].astype(int)
    diff=pd.concat([diff,df_s])
df["v2x_polyarchy_change"]=diff['v2x_polyarchy_change']

diff=pd.DataFrame()
for c in df.country.unique():
    df_s=df.loc[df["country"]==c]
    df_s["diff"]=np.insert(np.diff(df_s["v2x_libdem"]), 0, 0) 
    df_s['v2x_libdem_change'] = df_s['diff'].rolling(window=3, min_periods=1).max() > 3
    df_s['v2x_libdem_change']=df_s['v2x_libdem_change'].astype(int)
    diff=pd.concat([diff,df_s])
df["v2x_libdem_change"]=diff['v2x_libdem_change']

################################
### Hybrid regime (anocracy) ### 
################################

# F&L: "we mark regimes that score between −5 and 5 on the difference between Polity 
# IV’s democracy and autocracy measures (the difference ranges from −10 to 10)"

df["polity2_anoc"]=1
df.loc[df["polity2"]>5,"polity2_anoc"]=0
df.loc[df["polity2"]<-5,"polity2_anoc"]=0

df["v2x_polyarchy_anoc"]=1
df.loc[df["v2x_polyarchy"]>0.75,"v2x_polyarchy_anoc"]=0
df.loc[df["v2x_polyarchy"]<0.25,"v2x_polyarchy_anoc"]=0

df["v2x_libdem_anoc"]=1
df.loc[df["v2x_libdem"]>0.75,"v2x_libdem_anoc"]=0
df.loc[df["v2x_libdem"]<0.25,"v2x_libdem_anoc"]=0

#####################
### Rough terrain ### 
#####################

# Fearon and Laitin
fl, meta = pyreadstat.read_dta('data/repdata.dta')
fl=fl[["ccode","country","year","mtnest"]]
terrain = fl[['ccode','mtnest']].drop_duplicates()
terrain.columns=["gw_codes","terrain_fl"]

# Missing: 'Suriname', 'Luxembourg', 'Cabo Verde', 'Equatorial Guinea','Comoros', 'South Sudan', 'Qatar', 'East Timor', 'Solomon Islands'
# Fill with zero
df=pd.merge(left=df,right=terrain[["gw_codes","terrain_fl"]],on=["gw_codes"],how="left")
df['terrain_fl'].fillna(0, inplace=True) 

# Terrain Ruggedness Index
base=df[["year","gw_codes","country"]].copy()
rug=pd.read_csv("data/data_out/rug_cy.csv",index_col=0)
base=pd.merge(left=base,right=rug[["year","gw_codes","rugged"]],on=["year","gw_codes"],how="left")
df=pd.merge(left=df,right=base[["year","gw_codes","rugged"]],on=["year","gw_codes"],how="left")

df.rename(columns={"v2xpe_exlecon": 'exclu_socecon'}, inplace=True)
df.rename(columns={"v2xpe_exlpol": 'exclu_pol'}, inplace=True)
df.rename(columns={"v2x_polyarchy": 'polyarchy'}, inplace=True)
df.rename(columns={"v2x_polyarchy_sqr": 'polyarchy_sqr'}, inplace=True)
df.rename(columns={"v2x_libdem": 'ibdem'}, inplace=True)
df.rename(columns={"v2x_libdem_sqr": 'libdem_sqr'}, inplace=True)
df.rename(columns={"v2x_polyarchy_change": 'polyarchy_change'}, inplace=True)
df.rename(columns={"v2x_libdem_change": 'libdem_change'}, inplace=True)
df.rename(columns={"v2x_polyarchy_anoc": 'polyarchy_anoc'}, inplace=True)
df.rename(columns={"v2x_libdem_anoc": 'libdem_anoc'}, inplace=True)

# Save
df.isnull().any()
df.to_csv("out/df_consensus.csv")  

df.duplicated(subset=['gw_codes','year']).any()






