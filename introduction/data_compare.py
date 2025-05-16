import pandas as pd
from functions import dichotomize,lag_groupped,consec_zeros_grouped,apply_decay,get_wb,simple_imp_grouped,linear_imp_grouped
import matplotlib.pyplot as plt

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

############
### UCDP ###
############

ucdp_sb=pd.read_csv("data/data_out/ucdp_cy_sb.csv",index_col=0)
df = ucdp_sb[["year","gw_codes","country","best"]][~ucdp_sb['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
df.columns=["year","gw_codes","country","sb_fatalities"]

# t-1 model 
df["sb_fatalities_lag1"]=lag_groupped(df,"country","sb_fatalities",1)

# Time since
dichotomize(df,"sb_fatalities","d_civil_war",1000)
dichotomize(df,"sb_fatalities","d_civil_conflict",25)
df['d_civil_conflict_zeros'] = consec_zeros_grouped(df,'country','d_civil_conflict')
df['d_civil_conflict_zeros_decay'] = apply_decay(df,'d_civil_conflict_zeros')
df = df.drop('d_civil_conflict', axis=1)
df = df.drop('d_civil_conflict_zeros', axis=1)
df['d_civil_war_zeros'] = consec_zeros_grouped(df,'country','d_civil_war')
df['d_civil_war_zeros_decay'] = apply_decay(df,'d_civil_war_zeros')
df = df.drop('d_civil_war', axis=1)
df = df.drop('d_civil_war_zeros', axis=1)

# Neighbor conflict history sb fatalities 
neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df[["year","country","gw_codes","sb_fatalities"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_fat"]=0
for i in range(len(df_neighbors)):
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:   
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            if df_neighbors["sb_fatalities"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["sb_fatalities"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_fat')] = counts

dichotomize(df_neighbors,"neighbors_fat","d_neighbors_sb_fatalities",0)
df_neighbors['d_neighbors_sb_fatalities_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_sb_fatalities',1)
df=pd.merge(left=df,right=df_neighbors[["year","gw_codes","d_neighbors_sb_fatalities_lag1"]],on=["year","gw_codes"],how="left")

#######################
### World Bank data ###
#######################

df_econ=df[["year","gw_codes","country"]].copy()
feat_dev = ["NY.GDP.PCAP.CD", # GDP per capita (current US$)
            "NY.GDP.MKTP.KD.ZG", # GDP growth (annual %) 
            "NY.GDP.PETR.RT.ZS", # Oil rents (% of GDP)
            "SP.POP.TOTL", # Population size
            "SP.DYN.IMRT.IN", # Mortality rate, infant (per 1,000 live births)
            'SP.POP.2024.MA.5Y', # Population ages 20-24, male (% of male population)
            "ER.H2O.FWTL.ZS", # Annual freshwater withdrawals, total (% of internal resources)          
            ]

# Load data 
df_ccodes = pd.read_csv("data/df_ccodes.csv")
c_list=list(df_ccodes.iso_alpha3)
c_list = [char for char in c_list if char != "XYZ"]
economy=get_wb(list(range(1989, 2023, 1)),c_list,feat_dev)

# GDP per capita
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD"]],on=["year","gw_codes"],how="left")
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

# GDP growth (annual %)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.MKTP.KD.ZG"]],on=["year","gw_codes"],how="left")
base_imp=linear_imp_grouped(base,"country",["NY.GDP.MKTP.KD.ZG"])
base_imp=simple_imp_grouped(base_imp,"country",["NY.GDP.MKTP.KD.ZG"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.MKTP.KD.ZG"].loc[base["country"]==c])
#    axs[1].plot(base_imp["year"].loc[base_imp["country"]==c], base_imp["NY.GDP.MKTP.KD.ZG"].loc[base_imp["country"]==c])
#    axs[0].set_title(c)
#    plt.show()
    
df=pd.merge(left=df,right=base_imp[["year","gw_codes","NY.GDP.MKTP.KD.ZG"]],on=["year","gw_codes"],how="left")
df.rename(columns={"NY.GDP.MKTP.KD.ZG": 'growth'}, inplace=True)

# Oil rents (% of GDP)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS"]],on=["year","gw_codes"],how="left")
base_imp=linear_imp_grouped(base,"country",["NY.GDP.PETR.RT.ZS"])
base_imp=simple_imp_grouped(base_imp,"country",["NY.GDP.PETR.RT.ZS"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.PETR.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["oil_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df=pd.merge(left=df,right=base_imp[["year","gw_codes","NY.GDP.PETR.RT.ZS"]],on=["year","gw_codes"],how="left")
df.rename(columns={"NY.GDP.PETR.RT.ZS": 'oil_share'}, inplace=True)

# D. Population size
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","SP.POP.TOTL"]],on=["year","gw_codes"],how="left")
base_imp=linear_imp_grouped(base,"country",["SP.POP.TOTL"])
base_imp=simple_imp_grouped(base_imp,"country",["SP.POP.TOTL"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SP.POP.TOTL"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["pop"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp[["year","gw_codes","SP.POP.TOTL"]],on=["year","gw_codes"],how="left")
df.rename(columns={"SP.POP.TOTL": 'pop'}, inplace=True)

# Mortality rate, infant (per 1,000 live births)

base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SP.POP.TOTL","NY.GDP.PETR.RT.ZS","SP.DYN.IMRT.IN",'SP.POP.2024.MA.5Y',"ER.H2O.FWTL.ZS"]],on=["year","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["SP.DYN.IMRT.IN"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SP.DYN.IMRT.IN"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SP.DYN.IMRT.IN"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["inf_mort"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp_final[["year","gw_codes","SP.DYN.IMRT.IN"]],on=["year","gw_codes"],how="left")
df.rename(columns={"SP.DYN.IMRT.IN": 'inf_mort'}, inplace=True)

# Male total population 15-19
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SP.POP.TOTL","NY.GDP.PETR.RT.ZS","SP.DYN.IMRT.IN",'SP.POP.2024.MA.5Y',"ER.H2O.FWTL.ZS"]],on=["year","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["SP.POP.2024.MA.5Y"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SP.POP.2024.MA.5Y"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SP.POP.2024.MA.5Y"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["male_youth_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp_final[["year","gw_codes","SP.POP.2024.MA.5Y"]],on=["year","gw_codes"],how="left")
df.rename(columns={"SP.POP.2024.MA.5Y": 'male_youth_share'}, inplace=True)

# Average Mean Surface Air Temperature 
temp=pd.read_csv("data/data_out/temp_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SP.POP.TOTL","NY.GDP.PETR.RT.ZS","SP.DYN.IMRT.IN",'SP.POP.2024.MA.5Y',"ER.H2O.FWTL.ZS"]],on=["year","gw_codes"],how="left")
base=pd.merge(left=base,right=temp[["gw_codes","year","temp"]],on=["year","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["temp"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["temp"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["temp"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["temp"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp_final[["year","gw_codes","temp"]],on=["year","gw_codes"],how="left")

# Annual freshwater withdrawals, total (% of internal resources)  

base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SP.POP.TOTL","NY.GDP.PETR.RT.ZS","SP.DYN.IMRT.IN",'SP.POP.2024.MA.5Y',"ER.H2O.FWTL.ZS"]],on=["year","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["ER.H2O.FWTL.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["ER.H2O.FWTL.ZS"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["ER.H2O.FWTL.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["ER.H2O.FWTL.ZS"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp_final[["year","gw_codes","ER.H2O.FWTL.ZS"]],on=["year","gw_codes"],how="left")
df.rename(columns={"ER.H2O.FWTL.ZS": 'withdrawl'}, inplace=True)

###########
### EPR ###
###########

# Ethnic fractionalization
base=df[["year","gw_codes","country"]].copy()
erp=pd.read_csv("data/data_out/epr_cy.csv",index_col=0)
base=pd.merge(left=base,right=erp[["year","gw_codes","ethnic_frac"]],on=["year","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["ethnic_frac"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["ethnic_frac"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["ethnic_frac"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["ethnic_frac"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df=pd.merge(left=df,right=base_imp_final[["year","gw_codes","ethnic_frac"]],on=["year","gw_codes"],how="left")

############
### UNDP ###
############

# Expected years of schooling, male

# Merge
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["eys_male"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["eys_male"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["eys_male"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["eys_male"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp_final[["year","gw_codes","eys_male"]],on=["year","gw_codes"],how="left")

#############
### V-Dem ###
#############

# Electoral democracy index 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
df=pd.merge(left=df,right=vdem[["year","gw_codes","v2x_polyarchy"]],on=["year","gw_codes"],how="left")
df = df.rename(columns={"v2x_polyarchy": 'polyarchy'})

# Liberal democracy index
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_egaldem","v2x_civlib","v2xpe_exlsocgr"]],on=["year","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["v2x_libdem"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["v2x_libdem"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2x_libdem"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["libdem"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp_final[["year","gw_codes","v2x_libdem"]],on=["year","gw_codes"],how="left")
df.rename(columns={"v2x_libdem": 'libdem'}, inplace=True)

# Egalitarian democracy index  
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
df=pd.merge(left=df,right=vdem[["year","gw_codes","v2x_egaldem"]],on=["year","gw_codes"],how="left")
df = df.rename(columns={"v2x_egaldem": 'egaldem'})

# Civil liberties index  
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
df=pd.merge(left=df,right=vdem[["year","gw_codes","v2x_civlib"]],on=["year","gw_codes"],how="left")
df = df.rename(columns={"v2x_civlib": 'civlib'})

# Exclusion by Social Group index
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_egaldem","v2x_civlib","v2xpe_exlsocgr"]],on=["year","gw_codes"],how="left")
base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlsocgr"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["v2xpe_exlsocgr"])

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2xpe_exlsocgr"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["exlsocgr"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df=pd.merge(left=df,right=base_imp_final[["year","gw_codes","v2xpe_exlsocgr"]],on=["year","gw_codes"],how="left")
df.rename(columns={"v2xpe_exlsocgr": 'exlsocgr'}, inplace=True)

# Save 
df = df[~df['country'].isin(['Montenegro', 'Somalia', 'Kuwait', 'Solomon Islands'])] 
print(df.isnull().any())
df=df.reset_index(drop=True)
df.to_csv("out/data_examples.csv")

