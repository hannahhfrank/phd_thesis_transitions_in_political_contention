import pandas as pd
import numpy as np
from functions import dichotomize,lag_groupped,consec_zeros_grouped,apply_decay,imp_opti,calibrate_imp,get_wb,simple_imp_grouped,linear_imp_grouped
import matplotlib.pyplot as plt


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

############
### UCDP ###
############

ucdp_sb=pd.read_csv("data/data_out/ucdp_cy_sb.csv",index_col=0)
ucdp_sb_s = ucdp_sb[["year","gw_codes","country","best","count"]][~ucdp_sb['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
ucdp_sb_s.columns=["year","gw_codes","country","sb_fatalities","sb_event_counts"]

ucdp_osv=pd.read_csv("data/data_out/ucdp_cy_osv.csv",index_col=0)
ucdp_osv_s = ucdp_osv[["year","gw_codes","best","count"]][~ucdp_osv['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
ucdp_osv_s.columns=["year","gw_codes","osv_fatalities","osv_event_counts"]

ucdp_ns=pd.read_csv("data/data_out/ucdp_cy_ns.csv",index_col=0)
ucdp_ns_s = ucdp_ns[["year","gw_codes","best","count"]][~ucdp_ns['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
ucdp_ns_s.columns=["year","gw_codes","ns_fatalities","ns_event_counts"]

#############
### ACLED ###
#############

acled_protest=pd.read_csv("data/data_out/acled_cy_protest.csv",index_col=0)
acled_protest_s = acled_protest[["year","gw_codes","n_protest_events","fatalities"]][~acled_protest['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
acled_protest_s.columns=["year","gw_codes","protest_event_counts","protest_fatalities"]

acled_riots=pd.read_csv("data/data_out/acled_cy_riots.csv",index_col=0)
acled_riots_s = acled_riots[["year","gw_codes","n_riot_events","fatalities"]][~acled_riots['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
acled_riots_s.columns=["year","gw_codes","riot_event_counts","riot_fatalities"]

acled_remote=pd.read_csv("data/data_out/acled_cy_remote.csv",index_col=0)
acled_remote_s = acled_remote[["year","gw_codes","n_remote_events","fatalities"]][~acled_riots['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
acled_remote_s.columns=["year","gw_codes","remote_event_counts","remote_fatalities"]

# Additional countries not included in ACLED: 
# 265 German Democratic Republic	
# 315 Czechoslovakia
# 345 Yugoslavia
# 396 Abkhazia
# 397 South Ossetia
# 680 Yemen, People's Republic of

# Temporal coverage: 1970--2022

# Merge
df=pd.merge(left=ucdp_sb_s,right=ucdp_ns_s,on=["year","gw_codes"],how="left")
df=pd.merge(left=df,right=ucdp_osv_s,on=["year","gw_codes"],how="left")
df=pd.merge(left=df,right=acled_protest_s,on=["year","gw_codes"],how="left")
df=pd.merge(left=df,right=acled_riots_s,on=["year","gw_codes"],how="left")
df=pd.merge(left=df,right=acled_remote_s,on=["year","gw_codes"],how="left")
df=df.fillna(0)

###############
### Outcome ###
###############

# Dichotomize
dichotomize(df,"protest_event_counts","d_protest",25)
dichotomize(df,"riot_event_counts","d_riot",25)
dichotomize(df,"sb_fatalities","d_sb",0)
dichotomize(df,"osv_fatalities","d_osv",0)
dichotomize(df,"ns_fatalities","d_ns",0)
dichotomize(df,"remote_fatalities","d_remote",0)
dichotomize(df,"sb_fatalities","d_civil_war",1000)
dichotomize(df,"sb_fatalities","d_civil_conflict",25)

# Final df
df_out=df[["year","gw_codes","country","d_protest","d_riot","d_sb","d_osv","d_ns","d_remote","protest_event_counts","riot_event_counts","sb_fatalities","osv_fatalities","ns_fatalities","remote_fatalities","d_civil_conflict","d_civil_war"]].copy()
print(df_out.isna().any())
df_out.to_csv("out/df_out_full.csv") 
df_complete=df[["year","gw_codes","country","d_protest","d_riot","d_sb","d_osv","d_ns","d_remote","protest_event_counts","riot_event_counts","sb_fatalities","osv_fatalities","ns_fatalities","remote_fatalities","d_civil_conflict","d_civil_war"]].copy()

#####################
### History theme ###
#####################

# A. t-1 model 

df_conf_hist=df_out[["year","gw_codes","country","d_protest","d_riot","d_sb","d_osv","d_ns","d_remote","protest_event_counts","riot_event_counts","sb_fatalities","osv_fatalities","ns_fatalities","remote_fatalities"]].copy()
df_conf_hist["d_protest_lag1"]=lag_groupped(df,"country","d_protest",1)
df_conf_hist["d_riot_lag1"]=lag_groupped(df,"country","d_riot",1)
df_conf_hist["d_sb_lag1"]=lag_groupped(df,"country","d_sb",1)
df_conf_hist["d_osv_lag1"]=lag_groupped(df,"country","d_osv",1)
df_conf_hist["d_ns_lag1"]=lag_groupped(df,"country","d_ns",1)
df_conf_hist["d_remote_lag1"]=lag_groupped(df,"country","d_remote",1)
df_conf_hist["d_civil_war_lag1"]=lag_groupped(df,"country","d_civil_war",1)
df_conf_hist["d_civil_conflict_lag1"]=lag_groupped(df,"country","d_civil_conflict",1)
df_conf_hist["protest_event_counts_lag1"]=lag_groupped(df,"country","protest_event_counts",1)
df_conf_hist["riot_event_counts_lag1"]=lag_groupped(df,"country","riot_event_counts",1)
df_conf_hist["sb_fatalities_lag1"]=lag_groupped(df,"country","sb_fatalities",1)
df_conf_hist["osv_fatalities_lag1"]=lag_groupped(df,"country","osv_fatalities",1)
df_conf_hist["ns_fatalities_lag1"]=lag_groupped(df,"country","ns_fatalities",1)
df_conf_hist["remote_fatalities_lag1"]=lag_groupped(df,"country","remote_fatalities",1)

# B. Time since 

df_conf_hist['d_protest_zeros'] = consec_zeros_grouped(df,'country','d_protest')
df_conf_hist['d_protest_zeros_decay'] = apply_decay(df_conf_hist,'d_protest_zeros')
df_conf_hist['d_riot_zeros'] = consec_zeros_grouped(df,'country','d_riot')
df_conf_hist['d_riot_zeros_decay'] = apply_decay(df_conf_hist,'d_riot_zeros')
df_conf_hist['d_sb_zeros'] = consec_zeros_grouped(df,'country','d_sb')
df_conf_hist['d_sb_zeros_decay'] = apply_decay(df_conf_hist,'d_sb_zeros')
df_conf_hist['d_osv_zeros'] = consec_zeros_grouped(df,'country','d_osv')
df_conf_hist['d_osv_zeros_decay'] = apply_decay(df_conf_hist,'d_osv_zeros')
df_conf_hist['d_ns_zeros'] = consec_zeros_grouped(df,'country','d_ns')
df_conf_hist['d_ns_zeros_decay'] = apply_decay(df_conf_hist,'d_ns_zeros')
df_conf_hist['d_remote_zeros'] = consec_zeros_grouped(df,'country','d_remote')
df_conf_hist['d_remote_zeros_decay'] = apply_decay(df_conf_hist,'d_remote_zeros')
df_conf_hist['d_civil_war_zeros'] = consec_zeros_grouped(df,'country','d_civil_war')
df_conf_hist['d_civil_war_zeros_decay'] = apply_decay(df_conf_hist,'d_civil_war_zeros')
df_conf_hist['d_civil_conflict_zeros'] = consec_zeros_grouped(df,'country','d_civil_conflict')
df_conf_hist['d_civil_conflict_zeros_decay'] = apply_decay(df_conf_hist,'d_civil_conflict_zeros')

#def lag_groupped_ts(df, group_var, var, lag):
#    return df.groupby(group_var)[var].shift(lag).fillna(35)

#grouped = df.groupby('gw_codes')['d_protest'].apply(lambda x: measure_time_since_last_civil_war(x))
#df["time_since_protest"] = [item for sublist in grouped for item in sublist]
#df.loc[df["time_since_protest"]==-1,"time_since_protest"]=35
#df_conf_hist["time_since_protest_lag1"]=lag_groupped_ts(df,"country","time_since_protest",1)
#grouped = df.groupby('gw_codes')['d_riot'].apply(lambda x: measure_time_since_last_civil_war(x))
#df["time_since_riot"] = [item for sublist in grouped for item in sublist]
#df.loc[df["time_since_riot"]==-1,"time_since_riot"]=35
#df_conf_hist["time_since_riot_lag1"]=lag_groupped_ts(df,"country","time_since_riot",1)
#grouped = df.groupby('gw_codes')['d_sb'].apply(lambda x: measure_time_since_last_civil_war(x))
#df["time_since_sb"] = [item for sublist in grouped for item in sublist]
#df.loc[df["time_since_sb"]==-1,"time_since_sb"]=35
#df_conf_hist["time_since_sb_lag1"]=lag_groupped_ts(df,"country","time_since_sb",1)
#grouped = df.groupby('gw_codes')['d_osv'].apply(lambda x: measure_time_since_last_civil_war(x))
#df["time_since_osv"] = [item for sublist in grouped for item in sublist]
#df.loc[df["time_since_osv"]==-1,"time_since_osv"]=35
#df_conf_hist["time_since_osv_lag1"]=lag_groupped_ts(df,"country","time_since_osv",1)
#grouped = df.groupby('gw_codes')['d_ns'].apply(lambda x: measure_time_since_last_civil_war(x))
#df["time_since_ns"] = [item for sublist in grouped for item in sublist]
#df.loc[df["time_since_ns"]==-1,"time_since_ns"]=35
#df_conf_hist["time_since_ns_lag1"]=lag_groupped_ts(df,"country","time_since_ns",1)
#grouped = df.groupby('gw_codes')['d_terror'].apply(lambda x: measure_time_since_last_civil_war(x))
#df["time_since_terror"] = [item for sublist in grouped for item in sublist]
#df.loc[df["time_since_terror"]==-1,"time_since_terror"]=35
#df_conf_hist["time_since_terror_lag1"]=lag_groupped_ts(df,"country","time_since_terror",1)
#grouped = df.groupby('gw_codes')['d_civil_war'].apply(lambda x: measure_time_since_last_civil_war(x))
#df["time_since_civil_war"] = [item for sublist in grouped for item in sublist]
#df.loc[df["time_since_civil_war"]==-1,"time_since_civil_war"]=35
#df_conf_hist["time_since_civil_war_lag1"]=lag_groupped_ts(df,"country","time_since_civil_war",1)
#grouped = df.groupby('gw_codes')['d_civil_conflict'].apply(lambda x: measure_time_since_last_civil_war(x))
#df["time_since_civil_conflict"] = [item for sublist in grouped for item in sublist]
#df.loc[df["time_since_civil_conflict"]==-1,"time_since_civil_conflict"]=35
#df_conf_hist["ttime_since_civil_conflict_lag1"]=lag_groupped_ts(df,"country","time_since_civil_conflict",1)

# C. Neighbor history protest counts 

neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","protest_event_counts"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_protest"]=0
for i in range(len(df_neighbors)):
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:   
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            if df_neighbors["protest_event_counts"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["protest_event_counts"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_protest')] = counts

dichotomize(df_neighbors,"neighbors_protest","d_neighbors_proteset_event_counts",0)
df_neighbors['d_neighbors_proteset_event_counts_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_proteset_event_counts',1)
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_proteset_event_counts_lag1"]],on=["year","gw_codes"],how="left")

# D. Neighbor history riot counts 

neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","riot_event_counts"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_riot"]=0
for i in range(len(df_neighbors)):
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:   
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            if df_neighbors["riot_event_counts"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["riot_event_counts"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_riot')] = counts

dichotomize(df_neighbors,"neighbors_riot","d_neighbors_riot_event_counts",0)
df_neighbors['d_neighbors_riot_event_counts_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_riot_event_counts',1)
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_riot_event_counts_lag1"]],on=["year","gw_codes"],how="left")

# E. Neighbor conflict history sb fatalities 

neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","sb_fatalities"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

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
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_sb_fatalities_lag1"]],on=["year","gw_codes"],how="left")

# F. Neighbor conflict history ns fatalities 

neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","ns_fatalities"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_ns"]=0
for i in range(len(df_neighbors)):
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:   
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            if df_neighbors["ns_fatalities"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["ns_fatalities"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_ns')] = counts

dichotomize(df_neighbors,"neighbors_ns","d_neighbors_ns_fatalities",0)
df_neighbors['d_neighbors_ns_fatalities_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_ns_fatalities',1)
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_ns_fatalities_lag1"]],on=["year","gw_codes"],how="left")

# G. Neighbor conflict history osv fatalities 

neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","osv_fatalities"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_osv"]=0
for i in range(len(df_neighbors)):
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:   
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            if df_neighbors["osv_fatalities"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["osv_fatalities"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_osv')] = counts

dichotomize(df_neighbors,"neighbors_osv","d_neighbors_osv_fatalities",0)
df_neighbors['d_neighbors_osv_fatalities_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_osv_fatalities',1)
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_osv_fatalities_lag1"]],on=["year","gw_codes"],how="left")

# H. Neighbor conflict history fatalities terrorism 

neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","remote_fatalities"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_remote"]=0
for i in range(len(df_neighbors)):
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:   
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            if df_neighbors["remote_fatalities"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["remote_fatalities"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_remote')] = counts

dichotomize(df_neighbors,"neighbors_remote","d_neighbors_remote_fatalities",0)
df_neighbors['d_neighbors_remote_fatalities_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_remote_fatalities',1)
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_remote_fatalities_lag1"]],on=["year","gw_codes"],how="left")

# H. Neighbor conflict history civil war 
neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","d_civil_war"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_civil_war"]=0
for i in range(len(df_neighbors)):
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:   
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            if df_neighbors["d_civil_war"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["d_civil_war"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_civil_war')] = counts

dichotomize(df_neighbors,"neighbors_civil_war","d_neighbors_civil_war",0)
df_neighbors['d_neighbors_civil_war_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_civil_war',1)
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_civil_war_lag1"]],on=["year","gw_codes"],how="left")

# H. Neighbor conflict history civil conflict 

neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=df_out[["year","country","gw_codes","d_civil_conflict"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

df_neighbors["neighbors_civil_conflict"]=0
for i in range(len(df_neighbors)):
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:   
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            if df_neighbors["d_civil_conflict"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["d_civil_conflict"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_civil_conflict')] = counts

dichotomize(df_neighbors,"neighbors_civil_conflict","d_neighbors_civil_conflict",0)
df_neighbors['d_neighbors_civil_conflict_lag1'] = lag_groupped(df_neighbors,'country','d_neighbors_civil_conflict',1)
df_conf_hist=pd.merge(left=df_conf_hist,right=df_neighbors[["year","gw_codes","d_neighbors_civil_conflict_lag1"]],on=["year","gw_codes"],how="left")

# I. New state 
# Years since independence
base=df_out[["year","gw_codes","country"]].copy()
base['regime_duration']=0

# http://ksgleditsch.com/data/iisystem.dat
# http://ksgleditsch.com/data/microstatessystem.dat
d={"country":[],"gw_codes":[],"start":[],"end":[]}

import requests
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
all_countries_s=all_countries.loc[all_countries["end"]>=1989]

for c in base.gw_codes.unique():
    star_year=all_countries_s.start.loc[all_countries_s["gw_codes"]==c].iloc[0]
    for y in base.year.loc[base["gw_codes"]==c].unique():
        base.loc[(base["year"]==y)&(base["gw_codes"]==c),"regime_duration"]=y-star_year 
        
df_conf_hist=pd.merge(left=df_conf_hist,right=base[["year","gw_codes","regime_duration"]],on=["year","gw_codes"],how="left")

# J. Refugees 

base=df_out[["year","gw_codes","country"]].copy()
feat_dev = ["SM.POP.REFG","SP.POP.TOTL"]

# Import country codes  
df_ccodes = pd.read_csv("data/df_ccodes.csv")
c_list=list(df_ccodes.iso_alpha3)
c_list = [char for char in c_list if char != "XYZ"]

# Load data 
demog=get_wb(list(range(1989, 2024, 1)),c_list,feat_dev)

# Merge
base=pd.merge(left=base,right=demog[["year","gw_codes","SM.POP.REFG","SP.POP.TOTL"]],on=["year","gw_codes"],how="left")
    
### Multiple ###

base_imp=imp_opti(base,"country",["SM.POP.REFG"],vars_add=["SP.POP.TOTL"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "SM.POP.REFG")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SM.POP.REFG"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SM.POP.REFG"])

# Merge
base_imp_final['SM.POP.REFG'] = base_imp_final['SM.POP.REFG'].fillna(base_imp_calib['SM.POP.REFG'])
base_imp_final['pop_refugee_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={'SM.POP.REFG': 'pop_refugee'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SM.POP.REFG"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["pop_refugee"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_conf_hist=pd.merge(left=df_conf_hist,right=base_imp_final[["year","gw_codes","pop_refugee",'pop_refugee_id']],on=["year","gw_codes"],how="left")

# Final df
df_conf_hist=df_conf_hist[["year","gw_codes","country","d_protest_lag1","d_riot_lag1","d_riot_lag1","d_sb_lag1","d_osv_lag1","d_ns_lag1","d_remote_lag1","d_civil_war_lag1","d_civil_conflict_lag1","protest_event_counts_lag1","riot_event_counts_lag1","sb_fatalities_lag1","osv_fatalities_lag1","ns_fatalities_lag1","remote_fatalities_lag1","d_protest_zeros_decay","d_riot_zeros_decay","d_sb_zeros_decay","d_osv_zeros_decay","d_ns_zeros_decay","d_remote_zeros_decay",'d_civil_conflict_zeros_decay','d_civil_war_zeros_decay',"d_neighbors_proteset_event_counts_lag1","d_neighbors_riot_event_counts_lag1","d_neighbors_sb_fatalities_lag1","d_neighbors_osv_fatalities_lag1","d_neighbors_ns_fatalities_lag1","d_neighbors_remote_fatalities_lag1","d_neighbors_civil_war_lag1","d_neighbors_civil_conflict_lag1","regime_duration","pop_refugee",'pop_refugee_id']].copy()
print(df_conf_hist.isna().any())
print(df_conf_hist.min())
df_conf_hist.to_csv("out/df_conf_hist_full.csv")  

# Merge df
df_complete=pd.merge(left=df_complete,right=df_conf_hist[["year","gw_codes","d_protest_lag1","d_riot_lag1","d_sb_lag1","d_osv_lag1","d_ns_lag1","d_remote_lag1","d_civil_war_lag1","d_civil_conflict_lag1","protest_event_counts_lag1","riot_event_counts_lag1","sb_fatalities_lag1","osv_fatalities_lag1","ns_fatalities_lag1","remote_fatalities_lag1","d_protest_zeros_decay","d_protest_zeros_decay","d_sb_zeros_decay","d_osv_zeros_decay","d_ns_zeros_decay","d_remote_zeros_decay",'d_civil_conflict_zeros_decay','d_civil_war_zeros_decay',"d_neighbors_proteset_event_counts_lag1","d_neighbors_riot_event_counts_lag1","d_neighbors_sb_fatalities_lag1","d_neighbors_osv_fatalities_lag1","d_neighbors_ns_fatalities_lag1","d_neighbors_remote_fatalities_lag1","d_neighbors_civil_war_lag1","d_neighbors_civil_conflict_lag1","regime_duration"]],on=["year","gw_codes"],how="left")

########################
### Demography theme ###
########################

df_demog=df_out[["year","gw_codes","country"]].copy()

# Initiate
feat_dev = ["SP.POP.TOTL","EN.POP.DNST","SP.URB.TOTL.IN.ZS","SP.URB.TOTL","SP.RUR.TOTL.ZS","SP.RUR.TOTL"]

# Import country codes  
df_ccodes = pd.read_csv("data/df_ccodes.csv")
c_list=list(df_ccodes.iso_alpha3)
c_list = [char for char in c_list if char != "XYZ"]

# Load data 
demog=get_wb(list(range(1989, 2024, 1)),c_list,feat_dev)
demog = demog.rename(columns={'SP.POP.TOTL': 'pop'})

### A. Population size ### 

df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop"]],on=["year","gw_codes"],how="left")

### B. Population density ###

# Initiate
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=demog[["year","gw_codes","pop","EN.POP.DNST","SP.URB.TOTL.IN.ZS","SP.URB.TOTL","SP.RUR.TOTL.ZS","SP.RUR.TOTL"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["EN.POP.DNST"],vars_add=["pop","SP.URB.TOTL.IN.ZS","SP.URB.TOTL","SP.RUR.TOTL.ZS","SP.RUR.TOTL"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "EN.POP.DNST")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["EN.POP.DNST"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["EN.POP.DNST"])

# Merge
base_imp_final['EN.POP.DNST'] = base_imp_final['EN.POP.DNST'].fillna(base_imp_calib['EN.POP.DNST'])
base_imp_final['pop_density_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={'EN.POP.DNST': 'pop_density'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["EN.POP.DNST"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["pop_density"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","pop_density",'pop_density_id']],on=["year","gw_codes"],how="left")

### C. Urbanization ### 

# Initiate
demog = demog.rename(columns={"SP.URB.TOTL.IN.ZS": 'urb_share'})

# Merge
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","urb_share"]],on=["year","gw_codes"],how="left")

### D. Rural ### 

demog = demog.rename(columns={"SP.RUR.TOTL.ZS": 'rural_share'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","rural_share"]],on=["year","gw_codes"],how="left")

### E. Male population ###

# Initiate
base=df_out[["year","gw_codes","country"]].copy()
feat_dev = ["SP.POP.TOTL.MA.ZS","SP.POP.0014.MA.ZS","SP.POP.1519.MA.5Y","SP.POP.2024.MA.5Y","SP.POP.2529.MA.5Y","SP.POP.3034.MA.5Y"]

# Import country codes  
df_ccodes = pd.read_csv("data/df_ccodes.csv")
c_list=list(df_ccodes.iso_alpha3)
c_list = [char for char in c_list if char != "XYZ"]

# Load data 
demog=get_wb(list(range(1989, 2024, 1)),c_list,feat_dev)

demog = demog.rename(columns={"SP.POP.TOTL.MA.ZS": 'pop_male_share'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop_male_share"]],on=["year","gw_codes"],how="left")

### Male total population 0-14 ###  

demog = demog.rename(columns={"SP.POP.0014.MA.ZS": 'pop_male_share_0_14'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop_male_share_0_14"]],on=["year","gw_codes"],how="left")

### Male total population 15-19 ### 

demog = demog.rename(columns={"SP.POP.1519.MA.5Y": 'pop_male_share_15_19'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop_male_share_15_19"]],on=["year","gw_codes"],how="left")

### Male total population 20-24 ###  

demog = demog.rename(columns={"SP.POP.2024.MA.5Y": 'pop_male_share_20_24'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop_male_share_20_24"]],on=["year","gw_codes"],how="left")

### Male total population 25-29 ###  

demog = demog.rename(columns={"SP.POP.2529.MA.5Y": 'pop_male_share_25_29'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop_male_share_25_29"]],on=["year","gw_codes"],how="left")

### Male total population 30-34 ### 

demog = demog.rename(columns={"SP.POP.3034.MA.5Y": 'pop_male_share_30_34'})
df_demog=pd.merge(left=df_demog,right=demog[["year","gw_codes","pop_male_share_30_34"]],on=["year","gw_codes"],how="left")

### B. Ethnic fractionalization, dominance, religious, linguistical, racial fractionalization ###  ---> No missing values 
base=df_out[["year","gw_codes","country"]].copy()
erp=pd.read_csv("data/data_out/epr_cy.csv",index_col=0)
base=pd.merge(left=base,right=erp[["year","gw_codes","group_counts","monopoly_share","discriminated_share","powerless_share","dominant_share","ethnic_frac","rel_frac","lang_frac","race_frac"]],on=["year","gw_codes"],how="left")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["group_counts"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["group_counts"])
base_imp_final['group_counts_id'] = base["group_counts"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["group_counts"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["group_counts"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","group_counts",'group_counts_id']],on=["year","gw_codes"],how="left")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["monopoly_share"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["monopoly_share"])
base_imp_final['monopoly_share_id'] = base["monopoly_share"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["monopoly_share"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["monopoly_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","monopoly_share",'monopoly_share_id']],on=["year","gw_codes"],how="left")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["discriminated_share"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["discriminated_share"])
base_imp_final['discriminated_share_id'] = base["discriminated_share"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["discriminated_share"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["discriminated_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","discriminated_share",'discriminated_share_id']],on=["year","gw_codes"],how="left")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["powerless_share"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["powerless_share"])
base_imp_final['powerless_share_id'] = base["powerless_share"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["powerless_share"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["powerless_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","powerless_share",'powerless_share_id']],on=["year","gw_codes"],how="left")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["dominant_share"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["dominant_share"])
base_imp_final['dominant_share_id'] = base["dominant_share"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["dominant_share"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["dominant_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","dominant_share",'dominant_share_id']],on=["year","gw_codes"],how="left")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["ethnic_frac"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["ethnic_frac"])
base_imp_final['ethnic_frac_id'] = base["ethnic_frac"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["ethnic_frac"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["ethnic_frac"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","ethnic_frac",'ethnic_frac_id']],on=["year","gw_codes"],how="left")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["rel_frac"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["rel_frac"])
base_imp_final['rel_frac_id'] = base["rel_frac"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["rel_frac"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["rel_frac"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","rel_frac",'rel_frac_id']],on=["year","gw_codes"],how="left")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["lang_frac"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["lang_frac"])
base_imp_final['lang_frac_id'] = base["lang_frac"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["lang_frac"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["lang_frac"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","lang_frac",'lang_frac_id']],on=["year","gw_codes"],how="left")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["race_frac"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["race_frac"])
base_imp_final['race_frac_id'] = base["race_frac"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["race_frac"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["race_frac"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df_demog=pd.merge(left=df_demog,right=base_imp_final[["year","gw_codes","race_frac",'race_frac_id']],on=["year","gw_codes"],how="left")

# Final df
df_demog=df_demog[["year","gw_codes","country",'pop','pop_density','pop_density_id','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','group_counts_id','monopoly_share','monopoly_share_id','discriminated_share','discriminated_share_id','powerless_share','powerless_share_id','dominant_share','dominant_share_id','ethnic_frac','ethnic_frac_id','rel_frac','rel_frac_id','lang_frac','lang_frac_id','race_frac','race_frac_id']].copy()
print(df_demog.isna().any())
print(df_demog.min())
df_demog.to_csv("out/df_demog_full.csv") 
df_complete=pd.merge(left=df_complete,right=df_demog[["year","gw_codes",'pop','pop_density','pop_density_id','urb_share','rural_share','pop_male_share','pop_male_share_0_14','pop_male_share_15_19','pop_male_share_20_24','pop_male_share_25_29','pop_male_share_30_34','group_counts','group_counts_id','monopoly_share','monopoly_share_id','discriminated_share','discriminated_share_id','powerless_share','powerless_share_id','dominant_share','dominant_share_id','ethnic_frac','ethnic_frac_id','rel_frac','rel_frac_id','lang_frac','lang_frac_id','race_frac','race_frac_id']],on=["year","gw_codes"],how="left")

###################################
### Economy & development theme ###
###################################

# A. Oil deposits
oil=pd.read_csv("data/data_out/eia_cy.csv",index_col=0)

# Initiate
base=df_out[["year","gw_codes","country"]].copy()

# Merge
base=pd.merge(left=base,right=oil[['gw_codes','year','oil_deposits','oil_production','oil_exports','gas_deposits','gas_production','gas_exports']],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["oil_deposits"],vars_add=['oil_production','oil_exports','gas_deposits','gas_production','gas_exports'],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "oil_deposits")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["oil_deposits"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["oil_deposits"])

# Merge
base_imp_final['oil_deposits'] = base_imp_final['oil_deposits'].fillna(base_imp_calib['oil_deposits'])
base_imp_final['oil_deposits_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["oil_deposits"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["oil_deposits"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=df_out[["year","gw_codes","country"]].copy()
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","oil_deposits",'oil_deposits_id']],on=["year","gw_codes"],how="left")

## B. Oil production
# Takes values that are negative, cap at 0.

# Initiate
base=df_out[["year","gw_codes","country"]].copy()

# Merge
base=pd.merge(left=base,right=oil[['gw_codes','year','oil_deposits','oil_production','oil_exports','gas_deposits','gas_production','gas_exports']],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",['oil_production'],vars_add=["oil_deposits",'oil_exports','gas_deposits','gas_production','gas_exports'],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "oil_production")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["oil_production"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["oil_production"])

# Merge
base_imp_final['oil_production'] = base_imp_final['oil_production'].fillna(base_imp_calib['oil_production'])
base_imp_final['oil_production_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["oil_production"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["oil_production"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()
    
# Merge
base_imp_final['oil_production'] = base_imp_final['oil_production'].apply(lambda x: max(0, x))
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","oil_production",'oil_production_id']],on=["year","gw_codes"],how="left")

# C. Oil exports

# Initiate
base=df_out[["year","gw_codes","country"]].copy()

# Merge
base=pd.merge(left=base,right=oil[['gw_codes','year','oil_deposits','oil_production','oil_exports','gas_deposits','gas_production','gas_exports']],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",['oil_exports'],vars_add=["oil_deposits",'oil_production','gas_deposits','gas_production','gas_exports'],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "oil_exports")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["oil_exports"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["oil_exports"])

# Merge
base_imp_final['oil_exports'] = base_imp_final['oil_exports'].fillna(base_imp_calib['oil_exports'])
base_imp_final['oil_exports_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["oil_exports"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["oil_exports"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()
    
# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","oil_exports",'oil_exports_id']],on=["year","gw_codes"],how="left")

# L. Total natual resource rents % of GDP

feat_dev = ["NY.GDP.TOTL.RT.ZS", # Total natual resource rents % of GDP
            "NY.GDP.PETR.RT.ZS", # 	Oil rents (% of GDP)
            "NY.GDP.NGAS.RT.ZS", # Natural gas rents (% of GDP)
            "NY.GDP.COAL.RT.ZS", # 	Coal rents (% of GDP)
            "NY.GDP.FRST.RT.ZS", # Forest rents (% of GDP) 
            "NY.GDP.MINR.RT.ZS", # Mineral rents (% of GDP)
            "NY.GDP.PCAP.CD", # GDP per capita (current US$)
            "NY.GNP.PCAP.CD", # GNI per capita, Atlas method (current US$)
            "NY.GDP.MKTP.KD.ZG", # GDP growth (annual %) 
            "SL.UEM.TOTL.NE.ZS", # Unemployment, total (% of total labor force)
            "SL.UEM.TOTL.MA.NE.ZS", # Unemployment, male (% of male labor force)
            "FP.CPI.TOTL.ZG", # Inflation, consumer prices (annual %)
            "SN.ITK.DEFC.ZS", # Prevalence of undernourishment (% of population)
            "SP.DYN.IMRT.IN", # Mortality rate, infant (per 1,000 live births)
            "AG.PRD.FOOD.XD", # Food production index (2014-2016 = 100)
            "SM.POP.NETM", # Net migration
            "NV.AGR.TOTL.ZS", # Agriculture % of GDP
            "NE.TRD.GNFS.ZS", # Trade % of GDP
            "SH.H2O.BASW.RU.ZS", # People using at least basic drinking water services, rural (% of rural population)
            "SH.H2O.BASW.UR.ZS", # People using at least basic drinking water services, urban (% of urban population)
            "FP.CPI.TOTL", # Consumer price index (2010 = 100)
            "SP.DYN.TFRT.IN", # Fertility rate, total (births per woman)
            "SP.DYN.LE00.FE.IN", # Life expectancy at birth, female (years) 
            "SP.DYN.LE00.MA.IN", # Life expectancy at birth, male (years)
            "SP.POP.GROW", # Population growth (annual %)
            "NE.EXP.GNFS.ZS", # Exports of goods and services (% of GDP)
            "NE.IMP.GNFS.ZS", # Imports of goods and services (% of GDP)
            "SE.PRM.ENRR.FE", # School enrollment, primary, female (% gross)
            "SE.PRM.ENRR.MA", # School enrollment, primary, male (% gross)
            "SE.SEC.ENRR.FE", # School enrollment, secondary, female (% gross)
            "SE.SEC.ENRR.MA", # School enrollment, secondary, male (% gross)
            "SE.TER.ENRR.FE", # School enrollment, tertiary, female (% gross)
            "SE.TER.ENRR.MA", # School enrollment, tertiary, male (% gross)
            ]

# Import country codes  
df_ccodes = pd.read_csv("data/df_ccodes.csv")
c_list=list(df_ccodes.iso_alpha3)
c_list = [char for char in c_list if char != "XYZ"]

# Load data 
economy=get_wb(list(range(1989, 2024, 1)),c_list,feat_dev)

# A. Total natual resource rents % of GDP

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["NY.GDP.TOTL.RT.ZS"],vars_add=["NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "NY.GDP.TOTL.RT.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.TOTL.RT.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["NY.GDP.TOTL.RT.ZS"])

# Merge
base_imp_final["NY.GDP.TOTL.RT.ZS"] = base_imp_final["NY.GDP.TOTL.RT.ZS"].fillna(base_imp_calib["NY.GDP.TOTL.RT.ZS"])
base_imp_final['natres_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.TOTL.RT.ZS": 'natres_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.TOTL.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["natres_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","natres_share",'natres_share_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"NY.GDP.TOTL.RT.ZS": 'natres_share'})

# B. Oil rents (% of GDP)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["NY.GDP.PETR.RT.ZS"],vars_add=["NY.GDP.TOTL.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "NY.GDP.PETR.RT.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.PETR.RT.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["NY.GDP.PETR.RT.ZS"])

# Merge
base_imp_final["NY.GDP.PETR.RT.ZS"] = base_imp_final["NY.GDP.PETR.RT.ZS"].fillna(base_imp_calib["NY.GDP.PETR.RT.ZS"])
base_imp_final['oil_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.PETR.RT.ZS": 'oil_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.PETR.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["oil_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","oil_share",'oil_share_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"NY.GDP.PETR.RT.ZS": 'oil_share'})

# C. Natural gas rents (% of GDP)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["NY.GDP.NGAS.RT.ZS"],vars_add=["NY.GDP.TOTL.RT.ZS","NY.GDP.PETR.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "NY.GDP.NGAS.RT.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.NGAS.RT.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["NY.GDP.NGAS.RT.ZS"])

# Merge
base_imp_final["NY.GDP.NGAS.RT.ZS"] = base_imp_final["NY.GDP.NGAS.RT.ZS"].fillna(base_imp_calib["NY.GDP.NGAS.RT.ZS"])
base_imp_final['gas_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.NGAS.RT.ZS": 'gas_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.NGAS.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["gas_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","gas_share",'gas_share_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"NY.GDP.NGAS.RT.ZS": 'gas_share'})

# D. Coal rents (% of GDP)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["NY.GDP.COAL.RT.ZS"],vars_add=["NY.GDP.TOTL.RT.ZS","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "NY.GDP.COAL.RT.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.COAL.RT.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["NY.GDP.COAL.RT.ZS"])

# Merge
base_imp_final["NY.GDP.COAL.RT.ZS"] = base_imp_final["NY.GDP.COAL.RT.ZS"].fillna(base_imp_calib["NY.GDP.COAL.RT.ZS"])
base_imp_final['coal_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.COAL.RT.ZS": 'coal_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.COAL.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["coal_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","coal_share",'coal_share_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"NY.GDP.COAL.RT.ZS": 'coal_share'})

# E. Forest rents (% of GDP)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["NY.GDP.FRST.RT.ZS"],vars_add=["NY.GDP.TOTL.RT.ZS","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.MINR.RT.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "NY.GDP.FRST.RT.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.FRST.RT.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["NY.GDP.FRST.RT.ZS"])

# Merge
base_imp_final["NY.GDP.FRST.RT.ZS"] = base_imp_final["NY.GDP.FRST.RT.ZS"].fillna(base_imp_calib["NY.GDP.FRST.RT.ZS"])
base_imp_final['forest_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.FRST.RT.ZS": 'forest_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.FRST.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["forest_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","forest_share",'forest_share_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"NY.GDP.FRST.RT.ZS": 'forest_share'})

# F. Minerals rents (% of GDP)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.TOTL.RT.ZS"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["NY.GDP.MINR.RT.ZS"],vars_add=["NY.GDP.TOTL.RT.ZS","NY.GDP.PETR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "NY.GDP.MINR.RT.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.MINR.RT.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["NY.GDP.MINR.RT.ZS"])

# Merge
base_imp_final["NY.GDP.MINR.RT.ZS"] = base_imp_final["NY.GDP.MINR.RT.ZS"].fillna(base_imp_calib["NY.GDP.MINR.RT.ZS"])
base_imp_final['minerals_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.MINR.RT.ZS": 'minerals_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.MINR.RT.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["minerals_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","minerals_share",'minerals_share_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"NY.GDP.MINR.RT.ZS": 'minerals_share'})


# A. GDP per capita 

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["NY.GDP.PCAP.CD"],vars_add=["NY.GDP.MKTP.KD.ZG","NY.GNP.PCAP.CD"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "NY.GDP.PCAP.CD")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.PCAP.CD"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["NY.GDP.PCAP.CD"])

# Merge
base_imp_final["NY.GDP.PCAP.CD"] = base_imp_final["NY.GDP.PCAP.CD"].fillna(base_imp_calib["NY.GDP.PCAP.CD"])
base_imp_final['gdp_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.PCAP.CD": 'gdp'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.PCAP.CD"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["gdp"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","gdp",'gdp_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"NY.GDP.PCAP.CD": 'gdp'})

# A. GNI per capita 

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["NY.GNP.PCAP.CD"],vars_add=["NY.GDP.MKTP.KD.ZG","NY.GDP.PCAP.CD"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "NY.GNP.PCAP.CD")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["NY.GNP.PCAP.CD"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["NY.GNP.PCAP.CD"])

# Merge
base_imp_final["NY.GNP.PCAP.CD"] = base_imp_final["NY.GNP.PCAP.CD"].fillna(base_imp_calib["NY.GNP.PCAP.CD"])
base_imp_final['gni_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GNP.PCAP.CD": 'gni'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GNP.PCAP.CD"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["gni"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","gni",'gni_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"NY.GNP.PCAP.CD": 'gni'})

# B. GDP growth

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["NY.GDP.MKTP.KD.ZG"],vars_add=["NY.GDP.PCAP.CD","NY.GNP.PCAP.CD"],max_iter=10,min_val=-100000000)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "NY.GDP.MKTP.KD.ZG")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["NY.GDP.MKTP.KD.ZG"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["NY.GDP.MKTP.KD.ZG"])

# Merge
base_imp_final["NY.GDP.MKTP.KD.ZG"] = base_imp_final["NY.GDP.MKTP.KD.ZG"].fillna(base_imp_calib["NY.GDP.MKTP.KD.ZG"])
base_imp_final['gdp_growth_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NY.GDP.MKTP.KD.ZG": 'gdp_growth'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NY.GDP.MKTP.KD.ZG"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["gdp_growth"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","gdp_growth",'gdp_growth_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"NY.GDP.MKTP.KD.ZG": 'gdp_growth'})

# C. Unemployment, total

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SL.UEM.TOTL.NE.ZS"],vars_add=["SL.UEM.TOTL.MA.NE.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "SL.UEM.TOTL.NE.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SL.UEM.TOTL.NE.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SL.UEM.TOTL.NE.ZS"])

# Merge
base_imp_final["SL.UEM.TOTL.NE.ZS"] = base_imp_final["SL.UEM.TOTL.NE.ZS"].fillna(base_imp_calib["SL.UEM.TOTL.NE.ZS"])
base_imp_final['unemploy_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SL.UEM.TOTL.NE.ZS": 'unemploy'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SL.UEM.TOTL.NE.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["unemploy"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","unemploy",'unemploy_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SL.UEM.TOTL.NE.ZS": 'unemploy'})

# D. Unemployment, men

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SL.UEM.TOTL.MA.NE.ZS"],vars_add=["SL.UEM.TOTL.NE.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "SL.UEM.TOTL.MA.NE.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SL.UEM.TOTL.MA.NE.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SL.UEM.TOTL.MA.NE.ZS"])

# Merge
base_imp_final["SL.UEM.TOTL.MA.NE.ZS"] = base_imp_final["SL.UEM.TOTL.MA.NE.ZS"].fillna(base_imp_calib["SL.UEM.TOTL.MA.NE.ZS"])
base_imp_final['unemploy_male_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SL.UEM.TOTL.MA.NE.ZS": 'unemploy_male'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SL.UEM.TOTL.MA.NE.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["unemploy_male"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","unemploy_male",'unemploy_male_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SL.UEM.TOTL.MA.NE.ZS": 'unemploy_male'})

# E. Inflation rate

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["FP.CPI.TOTL.ZG"],vars_add=["FP.CPI.TOTL"],max_iter=10,min_val=-100000000)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "FP.CPI.TOTL.ZG")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["FP.CPI.TOTL.ZG"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["FP.CPI.TOTL.ZG"])

# Merge
base_imp_final["FP.CPI.TOTL.ZG"] = base_imp_final["FP.CPI.TOTL.ZG"].fillna(base_imp_calib["FP.CPI.TOTL.ZG"])
base_imp_final['inflat_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"FP.CPI.TOTL.ZG": 'inflat'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["FP.CPI.TOTL.ZG"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["inflat"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","inflat",'inflat_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"FP.CPI.TOTL.ZG": 'inflat'})

# U. Consumer price index (2010 = 100)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["FP.CPI.TOTL"],vars_add=["FP.CPI.TOTL.ZG"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "FP.CPI.TOTL")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["FP.CPI.TOTL"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["FP.CPI.TOTL"])

# Merge
base_imp_final["FP.CPI.TOTL"] = base_imp_final["FP.CPI.TOTL"].fillna(base_imp_calib["FP.CPI.TOTL"])
base_imp_final['conprice_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"FP.CPI.TOTL": 'conprice'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["FP.CPI.TOTL"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["conprice"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","conprice",'conprice_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"FP.CPI.TOTL": 'conprice'})

# F. Prevalence of undernourishment (% of population)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SN.ITK.DEFC.ZS"],vars_add=["AG.PRD.FOOD.XD","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "SN.ITK.DEFC.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SN.ITK.DEFC.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SN.ITK.DEFC.ZS"])

# Merge
base_imp_final["SN.ITK.DEFC.ZS"] = base_imp_final["SN.ITK.DEFC.ZS"].fillna(base_imp_calib["SN.ITK.DEFC.ZS"])
base_imp_final['undernour_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SN.ITK.DEFC.ZS": 'undernour'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SN.ITK.DEFC.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["undernour"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","undernour",'undernour_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SN.ITK.DEFC.ZS": 'undernour'})

# H. Food production

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["AG.PRD.FOOD.XD"],vars_add=["SN.ITK.DEFC.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "AG.PRD.FOOD.XD")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["AG.PRD.FOOD.XD"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["AG.PRD.FOOD.XD"])

# Merge
base_imp_final["AG.PRD.FOOD.XD"] = base_imp_final["AG.PRD.FOOD.XD"].fillna(base_imp_calib["AG.PRD.FOOD.XD"])
base_imp_final['foodprod_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"AG.PRD.FOOD.XD": 'foodprod'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["AG.PRD.FOOD.XD"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["foodprod"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","foodprod",'foodprod_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"AG.PRD.FOOD.XD": 'foodprod'})

# P. People using at least basic drinking water services, rural (% of rural population)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SH.H2O.BASW.RU.ZS"],vars_add=["SN.ITK.DEFC.ZS","AG.PRD.FOOD.XD","SH.H2O.BASW.UR.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "SH.H2O.BASW.RU.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SH.H2O.BASW.RU.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SH.H2O.BASW.RU.ZS"])

# Merge
base_imp_final["SH.H2O.BASW.RU.ZS"] = base_imp_final["SH.H2O.BASW.RU.ZS"].fillna(base_imp_calib["SH.H2O.BASW.RU.ZS"])
base_imp_final['water_rural_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SH.H2O.BASW.RU.ZS": 'water_rural'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SH.H2O.BASW.RU.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["water_rural"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","water_rural",'water_rural_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SH.H2O.BASW.RU.ZS": 'water_rural'})

# Q. People using at least basic drinking water services, urban (% of urban population)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SH.H2O.BASW.UR.ZS"],vars_add=["SN.ITK.DEFC.ZS","AG.PRD.FOOD.XD","SH.H2O.BASW.RU.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "SH.H2O.BASW.UR.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SH.H2O.BASW.UR.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SH.H2O.BASW.UR.ZS"])

# Merge
base_imp_final["SH.H2O.BASW.UR.ZS"] = base_imp_final["SH.H2O.BASW.UR.ZS"].fillna(base_imp_calib["SH.H2O.BASW.UR.ZS"])
base_imp_final['water_urb_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SH.H2O.BASW.UR.ZS": 'water_urb'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SH.H2O.BASW.UR.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["water_urb"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","water_urb",'water_urb_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SH.H2O.BASW.UR.ZS": 'water_urb'})

# I. Agriculture % of GDP

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["NV.AGR.TOTL.ZS"],vars_add=["NE.TRD.GNFS.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "NV.AGR.TOTL.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["NV.AGR.TOTL.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["NV.AGR.TOTL.ZS"])

# Merge
base_imp_final["NV.AGR.TOTL.ZS"] = base_imp_final["NV.AGR.TOTL.ZS"].fillna(base_imp_calib["NV.AGR.TOTL.ZS"])
base_imp_final['agri_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NV.AGR.TOTL.ZS": 'agri_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NV.AGR.TOTL.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["agri_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","agri_share",'agri_share_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"NV.AGR.TOTL.ZS": 'agri_share'})

# K. Trade % of GDP

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["NE.TRD.GNFS.ZS"],vars_add=["NV.AGR.TOTL.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "NE.TRD.GNFS.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["NE.TRD.GNFS.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["NE.TRD.GNFS.ZS"])

# Merge
base_imp_final["NE.TRD.GNFS.ZS"] = base_imp_final["NE.TRD.GNFS.ZS"].fillna(base_imp_calib["NE.TRD.GNFS.ZS"])
base_imp_final['trade_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NE.TRD.GNFS.ZS": 'trade_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NE.TRD.GNFS.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["trade_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","trade_share",'trade_share_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"NE.TRD.GNFS.ZS": 'trade_share'})

# V. Fertility rate, total (births per woman)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SP.DYN.TFRT.IN"],vars_add=["SP.POP.GROW","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.DYN.IMRT.IN"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","SP.DYN.TFRT.IN")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SP.DYN.TFRT.IN"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SP.DYN.TFRT.IN"])

# Merge
base_imp_final["SP.DYN.TFRT.IN"] = base_imp_final["SP.DYN.TFRT.IN"].fillna(base_imp_calib["SP.DYN.TFRT.IN"])
base_imp_final['fert_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SP.DYN.TFRT.IN": 'fert'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SP.DYN.TFRT.IN"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["fert"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","fert",'fert_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SP.DYN.TFRT.IN": 'fert'})

# W. Life expectancy at birth, female (years) 

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SP.DYN.LE00.FE.IN"],vars_add=["SP.DYN.TFRT.IN","SP.POP.GROW","SP.DYN.LE00.MA.IN","SP.DYN.IMRT.IN"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","SP.DYN.LE00.FE.IN")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SP.DYN.LE00.FE.IN"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SP.DYN.LE00.FE.IN"])

# Merge
base_imp_final["SP.DYN.LE00.FE.IN"] = base_imp_final["SP.DYN.LE00.FE.IN"].fillna(base_imp_calib["SP.DYN.LE00.FE.IN"])
base_imp_final['lifeexp_female_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SP.DYN.LE00.FE.IN": 'lifeexp_female'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SP.DYN.LE00.FE.IN"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["lifeexp_female"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","lifeexp_female",'lifeexp_female_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SP.DYN.LE00.FE.IN": 'lifeexp_female'})

# X. Life expectancy at birth, male (years)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SP.DYN.LE00.MA.IN"],vars_add=["SP.DYN.TFRT.IN","SP.POP.GROW","SP.DYN.LE00.FE.IN","SP.DYN.IMRT.IN"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","SP.DYN.LE00.MA.IN")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SP.DYN.LE00.MA.IN"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SP.DYN.LE00.MA.IN"])

# Merge
base_imp_final["SP.DYN.LE00.MA.IN"] = base_imp_final["SP.DYN.LE00.MA.IN"].fillna(base_imp_calib["SP.DYN.LE00.MA.IN"])
base_imp_final['lifeexp_male_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SP.DYN.LE00.MA.IN": 'lifeexp_male'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SP.DYN.LE00.MA.IN"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["lifeexp_male"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","lifeexp_male",'lifeexp_male_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SP.DYN.LE00.MA.IN": 'lifeexp_male'})

# Z. Population growth (annual %) 

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SP.POP.GROW"],vars_add=["SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.DYN.IMRT.IN"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","SP.POP.GROW")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SP.POP.GROW"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SP.POP.GROW"])

# Merge
base_imp_final["SP.POP.GROW"] = base_imp_final["SP.POP.GROW"].fillna(base_imp_calib["SP.POP.GROW"])
base_imp_final['pop_growth_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SP.POP.GROW": 'pop_growth'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SP.POP.GROW"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["pop_growth"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","pop_growth","pop_growth_id"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SP.POP.GROW": 'pop_growth'})

# G. Infant mortality 

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SP.DYN.IMRT.IN"],vars_add=["SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","SP.DYN.IMRT.IN")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SP.DYN.IMRT.IN"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SP.DYN.IMRT.IN"])

# Merge
base_imp_final["SP.DYN.IMRT.IN"] = base_imp_final["SP.DYN.IMRT.IN"].fillna(base_imp_calib["SP.DYN.IMRT.IN"])
base_imp_final['inf_mort_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SP.DYN.IMRT.IN": 'inf_mort'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SP.DYN.IMRT.IN"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["inf_mort"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","inf_mort","inf_mort_id"]],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SP.DYN.IMRT.IN": 'inf_mort'})

# J. Net migration 

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NY.GDP.PCAP.CD","NY.GNP.PCAP.CD","NY.GDP.MKTP.KD.ZG","SL.UEM.TOTL.NE.ZS","SL.UEM.TOTL.MA.NE.ZS","FP.CPI.TOTL.ZG","SN.ITK.DEFC.ZS","SP.DYN.IMRT.IN","AG.PRD.FOOD.XD","SM.POP.NETM","NV.AGR.TOTL.ZS","NE.TRD.GNFS.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","FP.CPI.TOTL","SP.DYN.TFRT.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.MA.IN","SP.POP.GROW"]],on=["year","gw_codes"],how="left")

base = base.rename(columns={"SM.POP.NETM": 'mig'})
df_econ=pd.merge(left=df_econ,right=base[["year","gw_codes","mig"]],on=["year","gw_codes"],how="left")

# A. Exports of goods and services (% of GDP)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["NE.EXP.GNFS.ZS"],vars_add=["NE.IMP.GNFS.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "NE.EXP.GNFS.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["NE.EXP.GNFS.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["NE.EXP.GNFS.ZS"])

# Merge
base_imp_final["NE.EXP.GNFS.ZS"] = base_imp_final["NE.EXP.GNFS.ZS"].fillna(base_imp_calib["NE.EXP.GNFS.ZS"])
base_imp_final['exports_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NE.EXP.GNFS.ZS": 'exports'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NE.EXP.GNFS.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["exports"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","exports",'exports_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"NE.EXP.GNFS.ZS": 'exports'})

# B. Imports of goods and services (% of GDP)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["NE.IMP.GNFS.ZS"],vars_add=["NE.EXP.GNFS.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "NE.IMP.GNFS.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["NE.IMP.GNFS.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["NE.IMP.GNFS.ZS"])

# Merge
base_imp_final["NE.IMP.GNFS.ZS"] = base_imp_final["NE.IMP.GNFS.ZS"].fillna(base_imp_calib["NE.IMP.GNFS.ZS"])
base_imp_final['imports_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"NE.IMP.GNFS.ZS": 'imports'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["NE.IMP.GNFS.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["imports"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","imports",'imports_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"NE.IMP.GNFS.ZS": 'imports'})

# D. School enrollment, primary, female (% gross)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SE.PRM.ENRR.FE"],vars_add=["SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "SE.PRM.ENRR.FE")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SE.PRM.ENRR.FE"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SE.PRM.ENRR.FE"])

# Merge
base_imp_final["SE.PRM.ENRR.FE"] = base_imp_final["SE.PRM.ENRR.FE"].fillna(base_imp_calib["SE.PRM.ENRR.FE"])
base_imp_final['primary_female_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.PRM.ENRR.FE": 'primary_female'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SE.PRM.ENRR.FE"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["primary_female"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","primary_female",'primary_female_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SE.PRM.ENRR.FE": 'primary_female'})

# E. School enrollment, primary, male (% gross)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SE.PRM.ENRR.MA"],vars_add=["SE.PRM.ENRR.FE","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "SE.PRM.ENRR.MA")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SE.PRM.ENRR.MA"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SE.PRM.ENRR.MA"])

# Merge
base_imp_final["SE.PRM.ENRR.MA"] = base_imp_final["SE.PRM.ENRR.MA"].fillna(base_imp_calib["SE.PRM.ENRR.MA"])
base_imp_final['primary_male_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.PRM.ENRR.MA": 'primary_male'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SE.PRM.ENRR.MA"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["primary_male"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","primary_male",'primary_male_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SE.PRM.ENRR.MA": 'primary_male'})

# F. School enrollment, secondary, female (% gross)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SE.SEC.ENRR.FE"],vars_add=["SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "SE.SEC.ENRR.FE")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SE.SEC.ENRR.FE"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SE.SEC.ENRR.FE"])

# Merge
base_imp_final["SE.SEC.ENRR.FE"] = base_imp_final["SE.SEC.ENRR.FE"].fillna(base_imp_calib["SE.SEC.ENRR.FE"])
base_imp_final['second_female_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.SEC.ENRR.FE": 'second_female'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SE.SEC.ENRR.FE"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["second_female"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","second_female",'second_female_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SE.SEC.ENRR.FE": 'second_female'})

# F. School enrollment, secondary, male (% gross)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SE.SEC.ENRR.MA"],vars_add=["SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.TER.ENRR.FE","SE.TER.ENRR.MA"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "SE.SEC.ENRR.MA")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SE.SEC.ENRR.MA"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SE.SEC.ENRR.MA"])

# Merge
base_imp_final["SE.SEC.ENRR.MA"] = base_imp_final["SE.SEC.ENRR.MA"].fillna(base_imp_calib["SE.SEC.ENRR.MA"])
base_imp_final['second_male_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.SEC.ENRR.MA": 'second_male'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SE.SEC.ENRR.MA"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["second_male"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","second_male",'second_male_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SE.SEC.ENRR.MA": 'second_male'})

# G. School enrollment, tertiary, female (% gross)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SE.TER.ENRR.FE"],vars_add=["SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.MA"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "SE.TER.ENRR.FE")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SE.TER.ENRR.FE"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SE.TER.ENRR.FE"])

# Merge
base_imp_final["SE.TER.ENRR.FE"] = base_imp_final["SE.TER.ENRR.FE"].fillna(base_imp_calib["SE.TER.ENRR.FE"])
base_imp_final['tert_female_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.TER.ENRR.FE": 'tert_female'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SE.TER.ENRR.FE"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["tert_female"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","tert_female",'tert_female_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SE.TER.ENRR.FE": 'tert_female'})

# H. School enrollment, tertiary, male (% gross)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=economy[["year","gw_codes","NE.EXP.GNFS.ZS","NE.IMP.GNFS.ZS","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE","SE.TER.ENRR.MA"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["SE.TER.ENRR.MA"],vars_add=["SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.TER.ENRR.FE"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "SE.TER.ENRR.MA")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["SE.TER.ENRR.MA"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["SE.TER.ENRR.MA"])

# Merge
base_imp_final["SE.TER.ENRR.MA"] = base_imp_final["SE.TER.ENRR.MA"].fillna(base_imp_calib["SE.TER.ENRR.MA"])
base_imp_final['tert_male_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"SE.TER.ENRR.MA": 'tert_male'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["SE.TER.ENRR.FE"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["tert_male"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","tert_male",'tert_male_id']],on=["year","gw_codes"],how="left")
base = base.rename(columns={"SE.TER.ENRR.MA": 'tert_male'})

# A. Expected years of schooling

# Merge
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["eys"],vars_add=['eys_male','eys_female'],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","eys")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["eys"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["eys"])

# Merge
base_imp_final["eys"] = base_imp_final["eys"].fillna(base_imp_calib["eys"])
base_imp_final['eys_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["eys"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["eys"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","eys",'eys_id']],on=["year","gw_codes"],how="left")

# B. Expected years of schooling, male

# Merge
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["eys_male"],vars_add=['eys','eys_female'],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","eys_male")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["eys_male"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["eys_male"])

# Merge
base_imp_final["eys_male"] = base_imp_final["eys_male"].fillna(base_imp_calib["eys_male"])
base_imp_final['eys_male_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["eys_male"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["eys_male"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","eys_male",'eys_male_id']],on=["year","gw_codes"],how="left")

# C. Expected years of schooling, female

# Merge
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["eys_female"],vars_add=['eys','eys_male'],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","eys_female")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["eys_female"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["eys_female"])

# Merge
base_imp_final["eys_female"] = base_imp_final["eys_female"].fillna(base_imp_calib["eys_female"])
base_imp_final['eys_female_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["eys_female"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["eys_female"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","eys_female",'eys_female_id']],on=["year","gw_codes"],how="left")

# D. Mean years of schooling

# Merge
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["mys"],vars_add=['mys_male','mys_male'],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","mys")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["mys"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["mys"])

# Merge
base_imp_final["mys"] = base_imp_final["mys"].fillna(base_imp_calib["mys"])
base_imp_final['mys_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["mys"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["mys"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","mys",'mys_id']],on=["year","gw_codes"],how="left")

# E. Mean years of schooling, male

# Merge
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["mys_male"],vars_add=['mys','mys_female'],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","mys_male")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["mys_male"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["mys_male"])

# Merge
base_imp_final["mys_male"] = base_imp_final["mys_male"].fillna(base_imp_calib["mys_male"])
base_imp_final['mys_male_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["mys_male"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["mys_male"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","mys_male",'mys_male_id']],on=["year","gw_codes"],how="left")

# F. Mean years of schooling, female

# Merge
hdi=pd.read_csv("data/data_out/hdi_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=hdi[["year","gw_codes",'eys','eys_male','eys_female','mys','mys_male','mys_female']],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["mys_female"],vars_add=['mys','mys_male'],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","mys_female")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["mys_female"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["mys_female"])

# Merge
base_imp_final["mys_female"] = base_imp_final["mys_female"].fillna(base_imp_calib["mys_female"])
base_imp_final['mys_female_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["mys_female"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["mys_female"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_econ=pd.merge(left=df_econ,right=base_imp_final[["year","gw_codes","mys_female",'mys_female_id']],on=["year","gw_codes"],how="left")

print(df_econ.isna().any().any())
print(df_econ.min())
df_econ.to_csv("out/df_econ_full.csv")  
df_complete=pd.merge(left=df_complete,right=df_econ[['year','gw_codes','oil_deposits','oil_deposits_id','oil_production','oil_production_id','oil_exports','oil_exports_id','natres_share','natres_share_id','oil_share','oil_share_id','gas_share','gas_share_id','coal_share','coal_share_id','forest_share','forest_share_id','minerals_share','minerals_share_id','gdp','gdp_id','gni','gni_id','gdp_growth','gdp_growth_id','unemploy','unemploy_id','unemploy_male','unemploy_male_id','inflat','inflat_id','conprice','conprice_id','undernour','undernour_id','foodprod','foodprod_id','water_rural','water_rural_id','water_urb','water_urb_id','agri_share','agri_share_id','trade_share','trade_share_id','fert','fert_id','lifeexp_female','lifeexp_female_id','lifeexp_male','lifeexp_male_id','pop_growth','pop_growth_id','inf_mort','inf_mort_id','mig','exports','exports_id','imports','imports_id','primary_female','primary_female_id','primary_male','primary_male_id','second_female','second_female_id','second_male','second_male_id','tert_female','tert_female_id','tert_male','tert_male_id','eys','eys_id','eys_male','eys_male_id','eys_female','eys_female_id','mys','mys_id','mys_male','mys_male_id','mys_female','mys_female_id']],on=["year","gw_codes"],how="left")

###############################
### Regime and policy theme ###
###############################

df_pol=df_out[["year","gw_codes","country"]].copy()

# World Bank
feat_dev = ["MS.MIL.TOTL.TF.ZS", # Armed forces personnel (% of total labor force)
            "MS.MIL.XPND.GD.ZS", # Military expenditure (% of GDP)
            "CC.EST", # Control of Corruption: Estimate
            "GE.EST", # Government Effectiveness: Estimate
            "PV.EST", # Political Stability and Absence of Violence/Terrorism: Estimate
            "RQ.EST", # Regulatory Quality: Estimate
            "RL.EST", # Rule of Law: Estimate
            "VA.EST", # Voice and Accountability: Estimate
            "GC.TAX.TOTL.GD.ZS", # Tax revenue (% of GDP)
            "IT.NET.BBND.P2", # Fixed broadband subscriptions (per 100 people)
            "IT.MLT.MAIN.P2", # Fixed telephone subscriptions (per 100 people)
            "IT.NET.USER.ZS", # Individuals using the Internet (% of population)
            "IT.CEL.SETS.P2" # Mobile cellular subscriptions (per 100 people)
            ]
df_ccodes = pd.read_csv("data/df_ccodes.csv")
c_list=list(df_ccodes.iso_alpha3)
c_list = [char for char in c_list if char != "XYZ"]

# Load data 
pol=get_wb(list(range(1989, 2024, 1)),c_list,feat_dev)

# G. Armed forces personnel (% of total labor force)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["MS.MIL.TOTL.TF.ZS"],vars_add=["MS.MIL.XPND.GD.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","MS.MIL.TOTL.TF.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["MS.MIL.TOTL.TF.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["MS.MIL.TOTL.TF.ZS"])

# Merge
base_imp_final["MS.MIL.TOTL.TF.ZS"] = base_imp_final["MS.MIL.TOTL.TF.ZS"].fillna(base_imp_calib["MS.MIL.TOTL.TF.ZS"])
base_imp_final['armedforces_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"MS.MIL.TOTL.TF.ZS": 'armedforces_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["MS.MIL.TOTL.TF.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["armedforces_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","armedforces_share",'armedforces_share_id']],on=["year","gw_codes"],how="left")

# H. Military expenditure (% of GDP)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["MS.MIL.XPND.GD.ZS"],vars_add=["MS.MIL.TOTL.TF.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","MS.MIL.XPND.GD.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["MS.MIL.XPND.GD.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["MS.MIL.XPND.GD.ZS"])

# Merge
base_imp_final["MS.MIL.XPND.GD.ZS"] = base_imp_final["MS.MIL.XPND.GD.ZS"].fillna(base_imp_calib["MS.MIL.XPND.GD.ZS"])
base_imp_final['milex_share_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"MS.MIL.XPND.GD.ZS": 'milex_share'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["MS.MIL.XPND.GD.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["milex_share"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","milex_share",'milex_share_id']],on=["year","gw_codes"],how="left")

# I. Control of Corruption: Estimate

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["CC.EST"],vars_add=["GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-100000000)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","CC.EST")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["CC.EST"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["CC.EST"])

# Merge
base_imp_final["CC.EST"] = base_imp_final["CC.EST"].fillna(base_imp_calib["CC.EST"])
base_imp_final['corruption_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"CC.EST": 'corruption'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["CC.EST"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["corruption"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","corruption",'corruption_id']],on=["year","gw_codes"],how="left")

# J. Government Effectiveness: Estimate

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["GE.EST"],vars_add=["CC.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-100000000)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","GE.EST")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["GE.EST"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["GE.EST"])

# Merge
base_imp_final["GE.EST"] = base_imp_final["GE.EST"].fillna(base_imp_calib["GE.EST"])
base_imp_final['effectiveness_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"GE.EST": 'effectiveness'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["GE.EST"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["effectiveness"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","effectiveness",'effectiveness_id']],on=["year","gw_codes"],how="left")

# K. Political Stability and Absence of Violence/Terrorism: Estimate

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["PV.EST"],vars_add=["CC.EST","GE.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-100000000)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","PV.EST")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["PV.EST"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["PV.EST"])

# Merge
base_imp_final["PV.EST"] = base_imp_final["PV.EST"].fillna(base_imp_calib["PV.EST"])
base_imp_final['polvio_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"PV.EST": 'polvio'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["PV.EST"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["polvio"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","polvio",'polvio_id']],on=["year","gw_codes"],how="left")

# L. Regulatory Quality: Estimate

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["RQ.EST"],vars_add=["CC.EST","GE.EST","PV.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-100000000)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","RQ.EST")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["RQ.EST"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["RQ.EST"])

# Merge
base_imp_final["RQ.EST"] = base_imp_final["RQ.EST"].fillna(base_imp_calib["RQ.EST"])
base_imp_final['regu_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"RQ.EST": 'regu'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["RQ.EST"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["regu"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","regu",'regu_id']],on=["year","gw_codes"],how="left")

# M. Rule of Law: Estimate

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["RL.EST"],vars_add=["CC.EST","GE.EST","PV.EST","RQ.EST","VA.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-100000000)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","RL.EST")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["RL.EST"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["RL.EST"])

# Merge
base_imp_final["RL.EST"] = base_imp_final["RL.EST"].fillna(base_imp_calib["RL.EST"])
base_imp_final['law_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"RL.EST": 'law'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["RL.EST"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["law"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","law",'law_id']],on=["year","gw_codes"],how="left")

# N. Voice and Accountability: Estimate

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["VA.EST"],vars_add=["CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","GC.TAX.TOTL.GD.ZS"],max_iter=10,min_val=-100000000)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","VA.EST")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["VA.EST"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["VA.EST"])

# Merge
base_imp_final["VA.EST"] = base_imp_final["VA.EST"].fillna(base_imp_calib["VA.EST"])
base_imp_final['account_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"VA.EST": 'account'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["VA.EST"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["account"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","account",'account_id']],on=["year","gw_codes"],how="left")

# O. Tax revenue (% of GDP)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["GC.TAX.TOTL.GD.ZS"],vars_add=["CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","GC.TAX.TOTL.GD.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["GC.TAX.TOTL.GD.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["GC.TAX.TOTL.GD.ZS"])

# Merge
base_imp_final["GC.TAX.TOTL.GD.ZS"] = base_imp_final["GC.TAX.TOTL.GD.ZS"].fillna(base_imp_calib["GC.TAX.TOTL.GD.ZS"])
base_imp_final['tax_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"GC.TAX.TOTL.GD.ZS": 'tax'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["GC.TAX.TOTL.GD.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["tax"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","tax",'tax_id']],on=["year","gw_codes"],how="left")

# Fixed broadband subscriptions (per 100 people)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["IT.NET.BBND.P2"],vars_add=["IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","IT.NET.BBND.P2")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["IT.NET.BBND.P2"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["IT.NET.BBND.P2"])

# Merge
base_imp_final["IT.NET.BBND.P2"] = base_imp_final["IT.NET.BBND.P2"].fillna(base_imp_calib["IT.NET.BBND.P2"])
base_imp_final['broadband_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"IT.NET.BBND.P2": 'broadband'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["IT.NET.BBND.P2"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["broadband"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","broadband",'broadband_id']],on=["year","gw_codes"],how="left")

# Fixed telephone subscriptions (per 100 people)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["IT.MLT.MAIN.P2"],vars_add=["IT.NET.BBND.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","IT.MLT.MAIN.P2")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["IT.MLT.MAIN.P2"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["IT.MLT.MAIN.P2"])

# Merge
base_imp_final["IT.MLT.MAIN.P2"] = base_imp_final["IT.MLT.MAIN.P2"].fillna(base_imp_calib["IT.MLT.MAIN.P2"])
base_imp_final['telephone_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"IT.MLT.MAIN.P2": 'telephone'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["IT.MLT.MAIN.P2"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["telephone"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","telephone",'telephone_id']],on=["year","gw_codes"],how="left")

# Individuals using the Internet (% of population)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["IT.NET.USER.ZS"],vars_add=["IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.CEL.SETS.P2"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","IT.NET.USER.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["IT.NET.USER.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["IT.NET.USER.ZS"])

# Merge
base_imp_final["IT.NET.USER.ZS"] = base_imp_final["IT.NET.USER.ZS"].fillna(base_imp_calib["IT.NET.USER.ZS"])
base_imp_final['internet_use_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"IT.NET.USER.ZS": 'internet_use'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["IT.NET.USER.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["internet_use"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","internet_use",'internet_use_id']],on=["year","gw_codes"],how="left")

# Mobile cellular subscriptions (per 100 people)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=pol[["year","gw_codes","MS.MIL.TOTL.TF.ZS","MS.MIL.XPND.GD.ZS","CC.EST","GE.EST","PV.EST","RQ.EST","RL.EST","VA.EST","GC.TAX.TOTL.GD.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS","IT.CEL.SETS.P2"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["IT.CEL.SETS.P2"],vars_add=["IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.NET.USER.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","IT.CEL.SETS.P2")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["IT.CEL.SETS.P2"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["IT.CEL.SETS.P2"])

# Merge
base_imp_final["IT.CEL.SETS.P2"] = base_imp_final["IT.CEL.SETS.P2"].fillna(base_imp_calib["IT.CEL.SETS.P2"])
base_imp_final['mobile_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"IT.CEL.SETS.P2": 'mobile'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["IT.CEL.SETS.P2"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["mobile"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","mobile",'mobile_id']],on=["year","gw_codes"],how="left")

# A. Electoral democracy index

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year",
                                    "gw_codes",
                                    "v2x_polyarchy", # Electoral democracy index
                                    "v2x_libdem", # Liberal democracy index
                                    "v2x_partipdem", # Participatory democracy index
                                    "v2x_delibdem", # Deliberative democracy index
                                    "v2x_egaldem", # Egalitarian democracy index                              
                                    "v2x_civlib", # Civil liberties index
                                    "v2x_clphy", # Physical violence index
                                    "v2x_clpol", # Political civil liberties index
                                    "v2x_clpriv", # Private civil liberties index                               
                                    "v2xpe_exlecon", # Exclusion by Socio-Economic Group
                                    "v2xpe_exlgender", # Exclusion by Gender index
                                    "v2xpe_exlgeo", # Exclusion by Urban-Rural Location index
                                    "v2xpe_exlpol", # Exclusion by Political Group index
                                    "v2xpe_exlsocgr", # Exclusion by Social Group index
                                    "v2smgovshut", # Government Internet shut down in practice
                                    "v2smgovfilprc" # Government Internet filtering in practice
                                    ]],
              on=["year","gw_codes"],how="left")

# Merge
base = base.rename(columns={"v2x_polyarchy": 'polyarchy'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","polyarchy"]],on=["year","gw_codes"],how="left")

# B. Liberal democracy index

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["v2x_libdem"],vars_add=["v2x_polyarchy","v2x_partipdem","v2x_delibdem","v2x_egaldem"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","v2x_libdem")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["v2x_libdem"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["v2x_libdem"])

# Merge
base_imp_final["v2x_libdem"] = base_imp_final["v2x_libdem"].fillna(base_imp_calib["v2x_libdem"])
base_imp_final['libdem_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2x_libdem": 'libdem'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2x_libdem"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["libdem"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","libdem",'libdem_id']],on=["year","gw_codes"],how="left")

# C. Participatory democracy index  --> no missing values

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

# Merge
base = base.rename(columns={"v2x_partipdem": 'partipdem'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","partipdem"]],on=["year","gw_codes"],how="left")

# D. Deliberative democracy index --> no missing values

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

# Merge
base = base.rename(columns={"v2x_delibdem": 'delibdem'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","delibdem"]],on=["year","gw_codes"],how="left")

# E. Egalitarian democracy index  --> no missing values

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

# Merge
base = base.rename(columns={"v2x_egaldem": 'egaldem'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","egaldem"]],on=["year","gw_codes"],how="left")

# A. Civil liberties index --> no missing values

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

# Merge
base = base.rename(columns={"v2x_civlib": 'civlib'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","civlib"]],on=["year","gw_codes"],how="left")

# B. Physical violence index  --> no missing values

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

# Merge
base = base.rename(columns={"v2x_clphy": 'phyvio'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","phyvio"]],on=["year","gw_codes"],how="left")

# C. Political civil liberties index  --> no missing values

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

# Merge
base = base.rename(columns={"v2x_clpol": 'pollib'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","pollib"]],on=["year","gw_codes"],how="left")

# D. Private civil liberties index  --> no missing values

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

# Merge
base = base.rename(columns={"v2x_clpriv": 'privlib'})
df_pol=pd.merge(left=df_pol,right=base[["year","gw_codes","privlib"]],on=["year","gw_codes"],how="left")

# A. Exclusion by Socio-Economic Group index

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["v2xpe_exlecon"],vars_add=["v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","v2xpe_exlecon")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlecon"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["v2xpe_exlecon"])

# Merge
base_imp_final["v2xpe_exlecon"] = base_imp_final["v2xpe_exlecon"].fillna(base_imp_calib["v2xpe_exlecon"])
base_imp_final['execon_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2xpe_exlecon": 'execon'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2xpe_exlecon"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["execon"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","execon",'execon_id']],on=["year","gw_codes"],how="left")

# B. Exclusion by Gender index

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["v2xpe_exlgender"],vars_add=["v2xpe_exlecon","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","v2xpe_exlgender")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlgender"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["v2xpe_exlgender"])

# Merge
base_imp_final["v2xpe_exlgender"] = base_imp_final["v2xpe_exlgender"].fillna(base_imp_calib["v2xpe_exlgender"])
base_imp_final['exgender_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2xpe_exlgender": 'exgender'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2xpe_exlgender"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["exgender"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","exgender",'exgender_id']],on=["year","gw_codes"],how="left")

# C. Exclusion by Urban-Rural Location index

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["v2xpe_exlgeo"],vars_add=["v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlpol","v2xpe_exlsocgr"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","v2xpe_exlgeo")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlgeo"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["v2xpe_exlgeo"])

# Merge
base_imp_final["v2xpe_exlgeo"] = base_imp_final["v2xpe_exlgeo"].fillna(base_imp_calib["v2xpe_exlgeo"])
base_imp_final['exgeo_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2xpe_exlgeo": 'exgeo'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2xpe_exlgeo"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["exgeo"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","exgeo",'exgeo_id']],on=["year","gw_codes"],how="left")

# D. Exclusion by Political Group index

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["v2xpe_exlpol"],vars_add=["v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlsocgr"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","v2xpe_exlpol")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlpol"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["v2xpe_exlpol"])

# Merge
base_imp_final["v2xpe_exlpol"] = base_imp_final["v2xpe_exlpol"].fillna(base_imp_calib["v2xpe_exlpol"])
base_imp_final['expol_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2xpe_exlpol": 'expol'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2xpe_exlpol"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["expol"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","expol",'expol_id']],on=["year","gw_codes"],how="left")

# E. Exclusion by Social Group index

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["v2xpe_exlsocgr"],vars_add=["v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","v2xpe_exlsocgr")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["v2xpe_exlsocgr"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["v2xpe_exlsocgr"])

# Merge
base_imp_final["v2xpe_exlsocgr"] = base_imp_final["v2xpe_exlsocgr"].fillna(base_imp_calib["v2xpe_exlsocgr"])
base_imp_final['exsoc_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2xpe_exlsocgr": 'exsoc'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2xpe_exlsocgr"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["exsoc"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","exsoc",'exsoc_id']],on=["year","gw_codes"],how="left")

# A. Government Internet shut down in practice

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["v2smgovshut"],vars_add=["v2smgovfilprc"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","v2smgovshut")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["v2smgovshut"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["v2smgovshut"])

# Merge
base_imp_final["v2smgovshut"] = base_imp_final["v2smgovshut"].fillna(base_imp_calib["v2smgovshut"])
base_imp_final['shutdown_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2smgovshut": 'shutdown'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2smgovshut"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["shutdown"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","shutdown",'shutdown_id']],on=["year","gw_codes"],how="left")

# B. Government Internet shut down in practice

# Merge
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_polyarchy","v2x_libdem","v2x_partipdem","v2x_delibdem","v2x_egaldem","v2x_civlib","v2x_clphy","v2x_clpol","v2x_clpriv","v2xpe_exlecon","v2xpe_exlgender","v2xpe_exlgeo","v2xpe_exlpol","v2xpe_exlsocgr","v2smgovshut","v2smgovfilprc"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["v2smgovfilprc"],vars_add=["v2smgovshut"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","v2smgovfilprc")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["v2smgovfilprc"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["v2smgovfilprc"])

# Merge
base_imp_final["v2smgovfilprc"] = base_imp_final["v2smgovfilprc"].fillna(base_imp_calib["v2smgovfilprc"])
base_imp_final['filter_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"v2smgovfilprc": 'filter'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2smgovfilprc"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["filter"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","filter",'filter_id']],on=["year","gw_codes"],how="left")

# A. Number of months that leader has been in power 

# Merge
reign=pd.read_csv("data/data_out/reign_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=reign[["year",
                                    "gw_codes",
                                    "tenure_months", # Number of months that leader has been in power 
                                    "dem_duration", # Logged number of months that a country is democratic
                                    "elections", # Election for leadership taking place in that year
                                    "lastelection" # Time since the last election for leadership (decay function)
                                    ]],on=["year","gw_codes"],how="left")


       
### Multiple ###

base_imp=imp_opti(base,"country",["tenure_months"],vars_add=["dem_duration","elections","lastelection"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","tenure_months")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["tenure_months"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["tenure_months"])

# Merge
base_imp_final["tenure_months"] = base_imp_final["tenure_months"].fillna(base_imp_calib["tenure_months"])
base_imp_final['tenure_months_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["tenure_months"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["tenure_months"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","tenure_months",'tenure_months_id']],on=["year","gw_codes"],how="left")
  
# B. Logged number of months that a country is democratic

# Merge
reign=pd.read_csv("data/data_out/reign_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=reign[["year","gw_codes","tenure_months","dem_duration","elections","lastelection"]],on=["year","gw_codes"],how="left")
        
### Multiple ###

base_imp=imp_opti(base,"country",["dem_duration"],vars_add=["tenure_months","elections","lastelection"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","dem_duration")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["dem_duration"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["dem_duration"])

# Merge
base_imp_final["dem_duration"] = base_imp_final["dem_duration"].fillna(base_imp_calib["dem_duration"])
base_imp_final['dem_duration_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["dem_duration"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["dem_duration"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","dem_duration",'dem_duration_id']],on=["year","gw_codes"],how="left")

# C. Election for leadership taking place in that year

# Merge
reign=pd.read_csv("data/data_out/reign_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=reign[["year","gw_codes","tenure_months","dem_duration","elections","lastelection"]],on=["year","gw_codes"],how="left")
        
### Multiple ###

base_imp=imp_opti(base,"country",["elections"],vars_add=["tenure_months","dem_duration","lastelection"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","elections")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["elections"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["elections"])

# Merge
base_imp_final["elections"] = base_imp_final["elections"].fillna(base_imp_calib["elections"])
base_imp_final['elections_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["elections"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["elections"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","elections",'elections_id']],on=["year","gw_codes"],how="left")

# D. Time since the last election for leadership (decay function)

# Merge
reign=pd.read_csv("data/data_out/reign_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=reign[["year","gw_codes","tenure_months","dem_duration","elections","lastelection"]],on=["year","gw_codes"],how="left")
        
### Multiple ###

base_imp=imp_opti(base,"country",["lastelection"],vars_add=["tenure_months","dem_duration","elections"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","lastelection")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["lastelection"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["lastelection"])

# Merge
base_imp_final["lastelection"] = base_imp_final["lastelection"].fillna(base_imp_calib["lastelection"])
base_imp_final['lastelection_id'] = base_imp["missing_id"]

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["lastelection"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["lastelection"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_pol=pd.merge(left=df_pol,right=base_imp_final[["year","gw_codes","lastelection",'lastelection_id']],on=["year","gw_codes"],how="left")
  
print(df_pol.isna().any().any())
print(df_pol.min())
df_pol.to_csv("out/df_pol_full.csv") 
df_complete=pd.merge(left=df_complete,right=df_pol[['year','gw_codes','armedforces_share','armedforces_share_id','milex_share','milex_share_id','corruption','corruption_id','effectiveness','effectiveness_id','polvio','polvio_id','regu','regu_id','law','law_id','account','account_id','tax','tax_id','broadband','broadband_id','telephone','telephone_id','internet_use','internet_use_id','mobile','mobile_id','polyarchy','libdem','libdem_id','partipdem','delibdem','egaldem','civlib','phyvio','pollib','privlib','execon','execon_id','exgender','exgender_id','exgeo','exgeo_id','expol','expol_id','exsoc','exsoc_id','shutdown','shutdown_id','filter','filter_id','tenure_months','tenure_months_id','dem_duration','dem_duration_id','elections','elections_id','lastelection','lastelection_id']],on=["year","gw_codes"],how="left")

##############################################
### Geography, environment & climate theme ###
##############################################

df_geog=df_out[["year","gw_codes","country"]].copy()
feat_dev = ["AG.LND.TOTL.K2", # Land area (sq. km)
            "AG.LND.FRST.ZS", # Forest area (% of land area)
            "AG.LND.PRCP.MM", # Average precipitation in depth (mm per year)
            "ER.H2O.FWST.ZS", # Level of water stress: freshwater withdrawal as a proportion of available freshwater resources
            "EN.CLC.MDAT.ZS", # Droughts, floods, extreme temperatures (% of population, average 1990-2009)
            #"EN.ATM.CO2E.KT", # CO2 emissions (kt) EN.ATM.CO2E.KT
            "EN.GHG.CO2.MT.CE.AR5", # Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)
            "AG.LND.AGRI.ZS", # Agricultural land (% of land area)
            "AG.LND.ARBL.ZS", # Arable land (% of land area)
            "AG.LND.IRIG.AG.ZS", # Agricultural irrigated land (% of total agricultural land)
            ]

df_ccodes = pd.read_csv("data/df_ccodes.csv")
c_list=list(df_ccodes.iso_alpha3)
c_list = [char for char in c_list if char != "XYZ"]
geog=get_wb(list(range(1989, 2024, 1)),c_list,feat_dev)

# A. Land area (sq. km)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=geog[["year","gw_codes","AG.LND.TOTL.K2","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.CLC.MDAT.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS","AG.LND.IRIG.AG.ZS"]],on=["year","gw_codes"],how="left")

# Impute missings
base['AG.LND.TOTL.K2'] = base.groupby('country')['AG.LND.TOTL.K2'].transform(lambda x: x.fillna(x.mean()))
# Source: https://en.wikipedia.org/wiki/Geography_of_Kosovo
base.loc[base["country"]=="Kosovo",'AG.LND.TOTL.K2']=10910
# Source: https://en.wikipedia.org/wiki/Geography_of_Taiwan
base.loc[base["country"]=="Taiwan",'AG.LND.TOTL.K2']=35808

base = base.rename(columns={"AG.LND.TOTL.K2": 'land'})
geog = geog.rename(columns={"AG.LND.TOTL.K2": 'land'})

# Merge
df_geog=pd.merge(left=df_geog,right=base[["year","gw_codes","land"]],on=["year","gw_codes"],how="left")

# F. Average Mean Surface Air Temperature 
temp=pd.read_csv("data/data_out/temp_cy.csv",index_col=0)
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=temp[["year","gw_codes","temp"]],on=["year","gw_codes"],how="left")  

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["temp"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["temp"])
base_imp_final['temp_id'] = base["temp"].isnull().astype(int)

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["temp"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["temp"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","temp",'temp_id']],on=["year","gw_codes"],how="left")

# B. Forest area (% of land area)  

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=geog[["year","gw_codes","land","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.CLC.MDAT.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS","AG.LND.IRIG.AG.ZS"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["AG.LND.FRST.ZS"],vars_add=["land","EN.GHG.CO2.MT.CE.AR5"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","AG.LND.FRST.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["AG.LND.FRST.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["AG.LND.FRST.ZS"])

# Merge
base_imp_final["AG.LND.FRST.ZS"] = base_imp_final["AG.LND.FRST.ZS"].fillna(base_imp_calib["AG.LND.FRST.ZS"])
base_imp_final['forest_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"AG.LND.FRST.ZS": 'forest'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["AG.LND.FRST.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["forest"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","forest",'forest_id']],on=["year","gw_codes"],how="left")

# C. CO2 emissions (kt) 

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=geog[["year","gw_codes","land","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.CLC.MDAT.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS","AG.LND.IRIG.AG.ZS"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["EN.GHG.CO2.MT.CE.AR5"],vars_add=["land","AG.LND.FRST.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","EN.GHG.CO2.MT.CE.AR5")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["EN.GHG.CO2.MT.CE.AR5"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["EN.GHG.CO2.MT.CE.AR5"])

# Merge
base_imp_final["EN.GHG.CO2.MT.CE.AR5"] = base_imp_final["EN.GHG.CO2.MT.CE.AR5"].fillna(base_imp_calib["EN.GHG.CO2.MT.CE.AR5"])
base_imp_final['co2_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"EN.GHG.CO2.MT.CE.AR5": 'co2'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["EN.GHG.CO2.MT.CE.AR5"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["co2"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","co2",'co2_id']],on=["year","gw_codes"],how="left")

# D. Average precipitation in depth (mm per year)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=geog[["year","gw_codes","land","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.CLC.MDAT.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS","AG.LND.IRIG.AG.ZS"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["AG.LND.PRCP.MM"],vars_add=["ER.H2O.FWST.ZS","EN.CLC.MDAT.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","AG.LND.PRCP.MM")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["AG.LND.PRCP.MM"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["AG.LND.PRCP.MM"])

# Merge
base_imp_final["AG.LND.PRCP.MM"] = base_imp_final["AG.LND.PRCP.MM"].fillna(base_imp_calib["AG.LND.PRCP.MM"])
base_imp_final['percip_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"AG.LND.PRCP.MM": 'percip'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["AG.LND.PRCP.MM"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["percip"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","percip",'percip_id']],on=["year","gw_codes"],how="left")

# E. Level of water stress: freshwater withdrawal as a proportion of available freshwater resources

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=geog[["year","gw_codes","land","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.CLC.MDAT.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS","AG.LND.IRIG.AG.ZS"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["ER.H2O.FWST.ZS"],vars_add=["AG.LND.PRCP.MM","EN.CLC.MDAT.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","ER.H2O.FWST.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["ER.H2O.FWST.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["ER.H2O.FWST.ZS"])

# Merge
base_imp_final["ER.H2O.FWST.ZS"] = base_imp_final["ER.H2O.FWST.ZS"].fillna(base_imp_calib["ER.H2O.FWST.ZS"])
base_imp_final['waterstress_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"ER.H2O.FWST.ZS": 'waterstress'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["ER.H2O.FWST.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["waterstress"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","waterstress",'waterstress_id']],on=["year","gw_codes"],how="left")

# H. Agricultural land (% of land area) 

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=geog[["year","gw_codes","land","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.CLC.MDAT.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS","AG.LND.IRIG.AG.ZS"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["AG.LND.AGRI.ZS"],vars_add=["AG.LND.ARBL.ZS","AG.LND.IRIG.AG.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","AG.LND.AGRI.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["AG.LND.AGRI.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["AG.LND.AGRI.ZS"])

# Merge
base_imp_final["AG.LND.AGRI.ZS"] = base_imp_final["AG.LND.AGRI.ZS"].fillna(base_imp_calib["AG.LND.AGRI.ZS"])
base_imp_final['agri_land_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"AG.LND.AGRI.ZS": 'agri_land'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["AG.LND.AGRI.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["agri_land"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","agri_land",'agri_land_id']],on=["year","gw_codes"],how="left")

# I. Arable land (% of land area)

# Merge
base=df_out[["year","gw_codes","country"]].copy()
base=pd.merge(left=base,right=geog[["year","gw_codes","land","AG.LND.FRST.ZS","AG.LND.PRCP.MM","ER.H2O.FWST.ZS","EN.CLC.MDAT.ZS","EN.GHG.CO2.MT.CE.AR5","AG.LND.AGRI.ZS","AG.LND.ARBL.ZS","AG.LND.IRIG.AG.ZS"]],on=["year","gw_codes"],how="left")

### Multiple ###

base_imp=imp_opti(base,"country",["AG.LND.ARBL.ZS"],vars_add=["AG.LND.AGRI.ZS","AG.LND.IRIG.AG.ZS"],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country","AG.LND.ARBL.ZS")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["AG.LND.ARBL.ZS"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["AG.LND.ARBL.ZS"])

# Merge
base_imp_final["AG.LND.ARBL.ZS"] = base_imp_final["AG.LND.ARBL.ZS"].fillna(base_imp_calib["AG.LND.ARBL.ZS"])
base_imp_final['arable_land_id'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={"AG.LND.ARBL.ZS": 'arable_land'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["AG.LND.ARBL.ZS"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["arable_land"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","arable_land",'arable_land_id']],on=["year","gw_codes"],how="left")

# D. Terrain Ruggedness Index --> no missings
base=df_out[["year","gw_codes","country"]].copy()
rug=pd.read_csv("data/data_out/rug_cy.csv",index_col=0)
base=pd.merge(left=base,right=rug[["year","gw_codes","rugged","soil","desert","tropical"]],on=["year","gw_codes"],how="left")
df_geog=pd.merge(left=df_geog,right=base[["year","gw_codes","rugged","soil","desert","tropical"]],on=["year","gw_codes"],how="left")

### A. Asia or Africa ###
rug=pd.read_csv("data/data_out/rug_cy.csv",index_col=0)
df_geog=pd.merge(left=df_geog,right=rug[["year","gw_codes","cont_africa","cont_asia"]],on=["year","gw_codes"],how="left")
df_geog.loc[df_geog["country"]=="Kosovo","cont_africa"]=0
df_geog.loc[df_geog["country"]=="Kosovo","cont_asia"]=0
df_geog.loc[df_geog["country"]=="Montenegro","cont_africa"]=0
df_geog.loc[df_geog["country"]=="Montenegro","cont_asia"]=0
df_geog.loc[df_geog["country"]=="Serbia (Yugoslavia)","cont_africa"]=0
df_geog.loc[df_geog["country"]=="Serbia (Yugoslavia)","cont_asia"]=0
df_geog.loc[df_geog["country"]=="South Sudan","cont_africa"]=1
df_geog.loc[df_geog["country"]=="South Sudan","cont_asia"]=0

###  B. Neighbor at war ###

neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
gw_codes=pd.read_csv("data/df_ccodes_gw.csv")
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]
df_neighbors=pd.merge(left=ucdp_sb[["year","country","gw_codes","best"]],right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")
df_neighbors["neighbors_fat"]=0
for i in range(len(df_neighbors)):
    if pd.isna(df_neighbors["neighbors"].iloc[i]): 
        pass
    else:   
        lst=df_neighbors["neighbors"].iloc[i].split(';')
        counts=0
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            if df_neighbors["best"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].empty==False:
                counts+=int(df_neighbors["best"].loc[(df_neighbors["year"]==df_neighbors["year"].iloc[i])&(df_neighbors["gw_codes"]==c)].iloc[0])
        if counts>0:
            df_neighbors.iloc[i, df_neighbors.columns.get_loc('neighbors_fat')] = counts

dichotomize(df_neighbors,"neighbors_fat","d_neighbors_con",0)
df_geog=pd.merge(left=df_geog,right=df_neighbors[["year","gw_codes","d_neighbors_con"]],on=["year","gw_codes"],how="left")
            
### C. No neighbors ###

neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
neighbors['neighbors'].fillna(0, inplace=True)
neighbors["no_neigh"]=0
neighbors.loc[neighbors["neighbors"]==0,"no_neigh"]=1
df_geog=pd.merge(left=df_geog,right=neighbors[["year","gw_codes","no_neigh"]],on=["year","gw_codes"],how="left")

### D. Neighbor democratic ###
# Countries missing, Bahamas, Belize, Brunei Darissalam

base=df_out[["year","gw_codes","country"]].copy() 
vdem=pd.read_csv("data/data_out/vdem_cy.csv",index_col=0)
base=pd.merge(left=base,right=vdem[["year","gw_codes","v2x_libdem",'v2x_polyarchy','v2x_partipdem','v2x_delibdem','v2x_egaldem']],on=["year","gw_codes"],how="left")

feat_dev = ["v2x_libdem"]

### Multiple ###

base_imp=imp_opti(base,"country",["v2x_libdem"],vars_add=['v2x_polyarchy','v2x_partipdem','v2x_delibdem','v2x_egaldem'],max_iter=10)

# Calibrate 
base_imp_calib=calibrate_imp(base_imp, "country", "v2x_libdem")

### Simple ###

base_imp_final=linear_imp_grouped(base,"country",["v2x_libdem"])
base_imp_final=simple_imp_grouped(base_imp_final,"country",["v2x_libdem"])

# Merge
base_imp_final['v2x_libdem'] = base_imp_final['v2x_libdem'].fillna(base_imp_calib['v2x_libdem'])
base_imp_final['libdem_id_neigh'] = base_imp["missing_id"]
base_imp_final = base_imp_final.rename(columns={'v2x_libdem': 'libdem'})

# Validate
#for c in base.country.unique():
#    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#    axs[0].plot(base["year"].loc[base["country"]==c], base["v2x_libdem"].loc[base["country"]==c])
#    axs[1].plot(base_imp_final["year"].loc[base_imp_final["country"]==c], base_imp_final["libdem"].loc[base_imp_final["country"]==c])
#    axs[0].set_title(c)
#    plt.show()

# Merge
dichotomize(base_imp_final,"libdem","d_libdem",0.5)
neighbors=pd.read_csv("data/data_out/cy_neighbors.csv",index_col=0)
base_imp_final=pd.merge(left=base_imp_final,right=neighbors[["gw_codes","year","neighbors"]],on=["year","gw_codes"],how="left")

# Check for each neighbor
base_imp_final["neighbors_dem"]=0
for i in range(len(base_imp_final)):
    if pd.isna(base_imp_final["neighbors"].iloc[i]): 
        pass
    else:   
        lst=base_imp_final["neighbors"].iloc[i].split(';')
        counts=0
        for x in lst:
            c=int(gw_codes_s["gw_codes"].loc[gw_codes_s["country"]==x].iloc[0])
            if base_imp_final["d_libdem"].loc[(base_imp_final["year"]==base_imp_final["year"].iloc[i])&(base_imp_final["gw_codes"]==c)].empty==False:
                counts+=int(base_imp_final["d_libdem"].loc[(base_imp_final["year"]==base_imp_final["year"].iloc[i])&(base_imp_final["gw_codes"]==c)].iloc[0])
        if counts>0:
            base_imp_final.iloc[i, base_imp_final.columns.get_loc('neighbors_dem')] = counts
            
# Dichotomize and merge
base_imp_final["d_neighbors_non_dem"]=0
base_imp_final.loc[base_imp_final["neighbors_dem"]<0.5,"d_neighbors_non_dem"]=1
df_geog=pd.merge(left=df_geog,right=base_imp_final[["year","gw_codes","d_neighbors_non_dem",'libdem_id_neigh']],on=["year","gw_codes"],how="left")

# Final df
base=df_geog[["year","gw_codes","country","cont_africa","cont_asia","d_neighbors_con","d_neighbors_non_dem",'libdem_id_neigh',"no_neigh"]]
base.loc[base["libdem_id_neigh"]==1, "d_neighbors_non_dem"]=np.nan
df=pd.merge(left=df,right=base[["year","gw_codes","cont_africa","cont_asia","d_neighbors_con","no_neigh","d_neighbors_non_dem"]],on=["year","gw_codes"],how="left")

# Save
print(df_geog.isna().any())
print(df_geog.min())
df_geog.to_csv("out/df_geog_full.csv")


df_complete=pd.merge(left=df_complete,right=df_geog[['year','gw_codes','land','temp','temp_id','forest','forest_id','co2','co2_id','percip','percip_id','waterstress','waterstress_id','agri_land','agri_land_id','arable_land','arable_land_id','rugged','soil','desert','tropical','cont_africa','cont_asia','d_neighbors_con','no_neigh','d_neighbors_non_dem','libdem_id_neigh']],on=["year","gw_codes"],how="left")
df_complete.to_csv("out/df_complete.csv")


