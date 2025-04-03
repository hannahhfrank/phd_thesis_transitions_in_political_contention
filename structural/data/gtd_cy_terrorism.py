### This file generates a csv file containing GTD data on series level ###

### Load libraries -------
import pandas as pd
import pickle
import numpy as np
        
### GTD data  ----------
# Codebook: https://www.start.umd.edu/gtd/downloads/Codebook.pdf
df = pd.read_excel("globalterrorismdb_0522dist.xlsx") #### check input <------
print("Loaded data")
print(df.head())

### Subset relevant case ----
#gtd_suicide = df[(df["suicide"]==1)]
gtd_suicide = df[(df["imonth"]!=0)]
gtd_suicide = gtd_suicide[(gtd_suicide["doubtterr"]!=1)]

### Subset columns -----
gtd_s = gtd_suicide[[
            #"eventid"
            "iyear",
            #"imonth",
            #"iday",
            #"approxdate",
            #"extended",
            #"resolution",
            "country",
            "country_txt",
            #"region",
            #"region_txt",
            #"provstate",
            #"city",
            #"latitude", # <-----
            #"longitude", # <-----
            #"specificity",
            #"vicinity",
            #"location",
            #"summary",
            #"crit1",
            #"crit2",
            #"crit3",
            #"doubtterr",
            #"alternative",
            #"alternative_txt",
            #"multiple",
            #"success",
            #"suicide",
            #"attacktype1",
            #"attacktype1_txt",
            #"attacktype2",
            #"attacktype2_txt",
            #"attacktype3",
            #"attacktype3_txt",
            #"targtype1",
            #"targtype1_txt",
            #"targsubtype1"
            #"targsubtype1_txt",
            #"corp1",
            #"target1",
            #"natlty1",
            #"natlty1_txt",
            #"targtype2",
            #"targtype2_txt",
            #"targsubtype2",
            #"targsubtype2_txt",
            #"corp2",
            #"target2",
            #"natlty2",
            #"natlty2_txt",
            #"targtype3",
            #"targtype3_txt",
            #"targsubtype3",
            #"targsubtype3_txt",
            #"corp3",
            #"target3",
            #"natlty3",
            #"natlty3_txt",
            #"gname",
            #"gsubname",
            #"gname2",
            #"gsubname2",
            #"gname3",
            #"gsubname3",
            #"motive",
            #"guncertain1",
            #"guncertain2",
            #"guncertain3",
            #"individual",
            #"nperps",
            #"nperpcap",
            #"claimed",
            #"claimmode",
            #"claimmode_txt",
            #"claim2",
            #"claimmode2",
            #"claimmode2_txt",
            #"claim3",
            #"claimmode3",
            #"claimmode3_txt",
            #"compclaim",
            #"weaptype1",
            #"weaptype1_txt",
            #"weapsubtype1",
            #"weapsubtype1_txt",
            #"weaptype2",
            #"weaptype2_txt",
            #"weapsubtype2",
            #"weapsubtype2_txt",
            #"weaptype3",
            #"weaptype3_txt",
            #"weapsubtype3",
            #"weapsubtype3_txt",
            #"weaptype4",
            #"weaptype4_txt",
            #"weapsubtype4",
            #"weapsubtype4_txt",
            #"weapdetail",
            "nkill",
            #"nkillus",
            #"nkillter",
            #"nwound",
            #"nwoundus",
            #"nwoundte",
            #"property",
            #"propextent",
            #"propextent_txt",
            #"propvalue",
            #"propcomment",
            #"ishostkid",
            #"nhostkid",
            #"nhostkidus",
            #"nhours",
            #"ndays",
            #"divert",
            #"kidhijcountry",
            #"ransom",
            #"ransomamt",
            #"ransomamtus",
            #"ransompaid",
            #"ransompaidus",
            #"ransomnote",
            #"hostkidoutcome",
            #"hostkidoutcome_txt",
            #"nreleased",
            #"addnotes",
            #"scite1",
            #"scite2",
            #"scite3",
            #"dbsource",
            #"INT_LOG",
            #"INT_IDEO",
            #"INT_MISC",
            #"INT_ANY",
            #"related",
            ]].copy()

### Aggregate to year level -------
agg_month = pd.DataFrame(gtd_s.groupby(["iyear","country","country_txt"]).size())
agg_month = agg_month.reset_index()
agg_month.rename(columns={0:"count"},inplace=True)

### Add fatalities -----
fat = pd.DataFrame(gtd_s.groupby(["iyear","country"])['nkill'].sum())
agg_month = pd.merge(left=agg_month,right=fat,left_on=["iyear","country"],right_on=["iyear","country"])
agg_month = agg_month.reset_index(drop=True)
agg_month.columns = ["year","gtd_codes","country","n_attack","fatalities"]
print("Aggregated data")
print(agg_month.head())

### Get countries and years --------
years = list(agg_month.year.unique())
countries = list(agg_month.gtd_codes.unique())

### Add missing observation to time series, those have zero fatalities --------

# Loop through every country-month
for i in range(0, len(countries)):
    print(countries[i])
    for x in years:
        
        # Check if country-month in data, if False add
        if ((agg_month['year'] == x) 
            & (agg_month['gtd_codes'] == countries[i])).any() == False:
                
                # Subset data to add
                s = {'year':x,'country':agg_month['country'].loc[(agg_month["gtd_codes"]==countries[i])].iloc[0],'gtd_codes':countries[i],'n_attack':0,'fatalities':0}
                s = pd.DataFrame(data=s,index=[0])
                agg_month = pd.concat([agg_month,s])  
                
agg_month = agg_month.sort_values(by=["country","year"])
agg_month.reset_index(drop=True,inplace=True)
print("Added missing obersvationes")

### Import country codes  -----
df_ccodes = pd.read_csv("df_ccodes.csv")
agg_month = pd.merge(left=agg_month,right=df_ccodes[["gw_codes","gtd_codes"]],left_on=["gtd_codes"],right_on=["gtd_codes"],how="left")

agg_month = agg_month.loc[agg_month["gw_codes"]<1000]

### Add missing country-months for countries completely missing ------
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

### Manually add to df_codes -----
obs={"country":"Yugoslavia", 
     "gw_codes":345,
     "iso_alpha3":99999999,
     "M49":"XYZ", 
     "StateAbb":99999999,
     "acled_codes":99999999,
     "gtd_codes":99999999,
     "vdem_codes":99999999}
df_ccodes = pd.concat([df_ccodes, pd.DataFrame(obs,index=[0])])

obs={"country":"Yemen, Peoples Republic of", 
     "gw_codes":680,
     "iso_alpha3":99999999,
     "M49":"XYZ", 
     "StateAbb":99999999,
     "acled_codes":99999999,
     "gtd_codes":99999999,
     "vdem_codes":99999999}
df_ccodes = pd.concat([df_ccodes, pd.DataFrame(obs,index=[0])])

obs={"country":"German Democratic Republic", 
     "gw_codes":265,
     "iso_alpha3":99999999,
     "M49":"XYZ", 
     "StateAbb":99999999,
     "acled_codes":99999999,
     "gtd_codes":99999999,
     "vdem_codes":99999999}
df_ccodes = pd.concat([df_ccodes, pd.DataFrame(obs,index=[0])])

obs={"country":"Czechoslovakia", 
     "gw_codes":315,
     "iso_alpha3":99999999,
     "M49":"XYZ", 
     "StateAbb":99999999,
     "acled_codes":99999999,
     "gtd_codes":99999999,
     "vdem_codes":99999999}
df_ccodes = pd.concat([df_ccodes, pd.DataFrame(obs,index=[0])])

obs={"country":"Abkhazia", 
     "gw_codes":396,
     "iso_alpha3":99999999,
     "M49":"XYZ", 
     "StateAbb":99999999,
     "acled_codes":99999999,
     "gtd_codes":99999999,
     "vdem_codes":99999999}
df_ccodes = pd.concat([df_ccodes, pd.DataFrame(obs,index=[0])])

obs={"country":"South Ossetia", 
     "gw_codes":397,
     "iso_alpha3":99999999,
     "M49":"XYZ", 
     "StateAbb":99999999,
     "acled_codes":99999999,
     "gtd_codes":99999999,
     "vdem_codes":99999999}
df_ccodes = pd.concat([df_ccodes, pd.DataFrame(obs,index=[0])])

obs={"country":"Vietnam, Republic of", 
     "gw_codes":817,
     "iso_alpha3":99999999,
     "M49":"XYZ", 
     "StateAbb":99999999,
     "acled_codes":99999999,
     "gtd_codes":99999999,
     "vdem_codes":99999999}
df_ccodes = pd.concat([df_ccodes, pd.DataFrame(obs,index=[0])])

all_countries_s=all_countries.loc[all_countries["end"]>=1970]
countries_acled=agg_month["gw_codes"].unique()
countries=all_countries_s["gw_codes"].unique()
add = list(filter(lambda x: x not in countries_acled,countries))


for i in range(0, len(add)):
    print(df_ccodes[df_ccodes["gw_codes"]==add[i]]["country"].values[0])
    for d in years:
        s = {'year':d,'country':df_ccodes[df_ccodes["gw_codes"]==add[i]]["country"].values[0],'gw_codes':add[i],'gtd_codes':df_ccodes[df_ccodes["gw_codes"]==add[i]]["gtd_codes"].values[0],'n_attack':0,'fatalities':0}
        s = pd.DataFrame(data=s,index=[0])
        agg_month = pd.concat([agg_month, s])  
print("Added missing countries")
print(agg_month.head())


### Sort and reset index -------
ucdp_final = agg_month[["year","country","gw_codes","n_attack","fatalities"]]
ucdp_final = ucdp_final.sort_values(by=["country","year"])
ucdp_final.reset_index(drop=True,inplace=True)
print(ucdp_final.head(3))


### Check independence and remove obs ----------


### Dissolution ###

# Czechoslovakia, split 1 January 1993
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==315)&(ucdp_final["year"]>1992))]

# Yemen, People's Republic of, unification 22 May 1990
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==680)&(ucdp_final["year"]>1990))]

# German Democratic Republic, reunification 3 October 1990
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==265)&(ucdp_final["year"]>1990))]

# Yugoslavia, Serbia independence decalartion 5 June 2006 
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==345)&(ucdp_final["year"]>2006))]

# Viertnam, Republic of, end vietnam war 30 April 1975
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==817)&(ucdp_final["year"]>1975))]

### New states ###

# Namibia 21 March 1990 independence
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==565)&(ucdp_final["year"]<1990))]

# Turkmenistan 26 December 1991, remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==701)&(ucdp_final["year"]<=1991))]

# Tajikistan 26 December 1991, remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==702)&(ucdp_final["year"]<=1991))]

# Kyrgyztan 26 December 1991, remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==703)&(ucdp_final["year"]<=1991))]

# Uzbekistan 26 December 1991, remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==704)&(ucdp_final["year"]<=1991))]

# Kazakhstan 26 December 1991, remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==705)&(ucdp_final["year"]<=1991))]

# Ukraine 26 December 1991, remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==369)&(ucdp_final["year"]<=1991))]

# Armenia 26 December 1991, remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==371)&(ucdp_final["year"]<=1991))]

# Gerogia 26 December 1991, remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==372)&(ucdp_final["year"]<=1991))]

# Azerbeijan 26 December 1991, remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==373)&(ucdp_final["year"]<=1991))]

# Belarus 26 December 1991, remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==370)&(ucdp_final["year"]<=1991))]

# Moldova 26 December 1991, remove complete 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==359)&(ucdp_final["year"]<1991))]

# Latvia 6 Sepmetmeber 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==367)&(ucdp_final["year"]<1991))]

# Estonia 6 Sepmetmeber 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==366)&(ucdp_final["year"]<1991))]

# Lithuanua 6 Sepmetmeber 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==368)&(ucdp_final["year"]<1991))]

# Macedonia independence referendum 8 September 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==343)&(ucdp_final["year"]<1991))]

# Bosnia-Herzegovina independence 3 March 1992
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==346)&(ucdp_final["year"]<1992))]

# Montenegro independence restored 3 June 2006
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==341)&(ucdp_final["year"]<2006))]

# Kosovo, declaration of independence 17 February 2008
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==347)&(ucdp_final["year"]<2008))]

# Cezch republic establishment 1 January 1993
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==316)&(ucdp_final["year"]<1993))]

# Slovakia establishment 1 January 1993
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==317)&(ucdp_final["year"]<1993))]

# Slovenia, admitted to UN 22 May 1992
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==349)&(ucdp_final["year"]<1992))]

# Croatia, recognized 22 May 1992
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==344)&(ucdp_final["year"]<1992))]

# Eritrea, independence 24 May 1993
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==531)&(ucdp_final["year"]<1993))]

# Palau, independence 1 October 1994
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==986)&(ucdp_final["year"]<1994))]

# East Timor, independence restored 20 May 2002
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==860)&(ucdp_final["year"]<2002))]

# Serbia, independence decalartion 5 June 2006 
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==340)&(ucdp_final["year"]<2006))]

# South Ossetia, 26 August 2008 recognition by Russia
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==397)&(ucdp_final["year"]<2008))]

# Abkhazia, 28 August 2008 declared as occupied by Russia
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==396)&(ucdp_final["year"]<2008))]

# South Sudan, 9 July 2011
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==626)&(ucdp_final["year"]<2011))]

### Save data ---------
ucdp_final = ucdp_final.sort_values(by=["gw_codes","year"])
ucdp_final.reset_index(drop=True,inplace=True)
ucdp_final["gw_codes"] = ucdp_final["gw_codes"].astype(int)
ucdp_final["n_attack"] = ucdp_final["n_attack"].astype(int)
ucdp_final["fatalities"] = ucdp_final["fatalities"].astype(int)
ucdp_final.to_csv("data_out/gtd_cy_attacks.csv",sep=',')
print("Saved DataFrame!")











