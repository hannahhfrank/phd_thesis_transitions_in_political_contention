### This file generates a csv file containing UCDP GED data on the country-year level of analysis ###

### Load libraries -------
import pandas as pd

### UCDP Georeferenced Event Dataset ----------
# Codebook: https://ucdp.uu.se/downloads/ged/ged211.pdf
ucdp = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged241-csv.zip",low_memory=False)

### Only use state-based violence ---------
ucdp_s = ucdp[(ucdp["type_of_violence"]==3)].copy()

### Exclude state-based between governments ----------
ucdp_ss = ucdp_s.loc[(ucdp_s["dyad_name"] != "Government of Afghanistan - Government of United Kingdom, Government of United States of America") &
                    (ucdp_s["dyad_name"] != "Government of Cambodia (Kampuchea) - Government of Thailand") &
                    (ucdp_s["dyad_name"] != "Government of Cameroon - Government of Nigeria") &
                    (ucdp_s["dyad_name"] != "Government of Djibouti - Government of Eritrea") &
                    (ucdp_s["dyad_name"] != "Government of Ecuador - Government of Peru") &
                    (ucdp_s["dyad_name"] != "Government of Eritrea - Government of Ethiopia") &
                    (ucdp_s["dyad_name"] != "Government of India - Government of Pakistan") &
                    (ucdp_s["dyad_name"] != "Government of China - Government of India") &  
                    (ucdp_s["dyad_name"] != "Government of Iran - Government of Israel") &                    
                    (ucdp_s["dyad_name"] != "Government of Iraq - Government of Kuwait") &
                    (ucdp_s["dyad_name"] != "Government of Australia, Government of United Kingdom, Government of United States of America - Government of Iraq") &
                    (ucdp_s["dyad_name"] != "Government of Kyrgyzstan - Government of Tajikistan") &                    
                    (ucdp_s["dyad_name"] != "Government of Panama - Government of United States of America") &
                    (ucdp_s["dyad_name"] != "Government of Russia (Soviet Union) - Government of Ukraine") &                   
                    (ucdp_s["dyad_name"] != "Government of South Sudan - Government of Sudan") ].copy(deep=True)

### Aggregate to month level -------
agg_month = pd.DataFrame(ucdp_ss.groupby(["year","country_id"]).size())
agg_month = agg_month.reset_index()
agg_month.rename(columns={0:"count"},inplace=True)
print("Aggregated data")

### Aggregate fatality variables by country-year ---------
best = pd.DataFrame(ucdp_ss.groupby(["year","country_id","country"])['best'].sum())
high = pd.DataFrame(ucdp_ss.groupby(["year","country_id","country"])['high'].sum())
low = pd.DataFrame(ucdp_ss.groupby(["year","country_id","country"])['low'].sum())

### Merge fatality variables and reset index ----------------
best_high = pd.concat([best, high], axis=1)
fat =  pd.concat([best_high, low], axis=1)
if "year" in fat == False:
    fat["year"] = fat.index.get_level_values(0)
if "country_id" in fat == False:
    fat["country_id"] = fat.index.get_level_values(1)   
fat = fat.reset_index()
print("Load data and arregate to country-year")

### Merge ----
fat = pd.merge(left=fat,right=agg_month[["year","country_id","count"]],left_on=["year","country_id"],right_on=["year","country_id"])
print(fat.head())

### Get countries and years -------
countries = ucdp_ss.country_id.unique()
years = ucdp_ss.year.unique()

### Add missing observations (those with zero fatalities) --------
# Loop through every year for every country
for i in range(0, len(countries)):
    for x in range(0, len(years)):        
        # Check if country-year in data, if False add
        if ((fat['year'] == years[x]) 
            & (fat['country_id'] == countries[i])).any() == False:
                s = {'year':fat['year'].loc[(fat["year"]==years[x])].iloc[0],'country':fat['country'].loc[(fat["country_id"]==countries[i])].iloc[0],'country_id':[countries[i]],'best':0,'high':0,'low':0,"count":0}
                s = pd.DataFrame(data=s)
                fat = pd.concat([fat,s]) 

### Import country codes  -----
fat.rename(columns = {'country_id':'gw_codes'},inplace = True)
df_ccodes = pd.read_csv("df_ccodes.csv")

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
obs={"country":"Yemen, Peoples Republic of", 
     "gw_codes":680,
     "iso_alpha3":99999999,
     "M49":"XYZ", 
     "StateAbb":99999999,
     "acled_codes":99999999,
     "gtd_codes":99999999,
     "vdem_codes":99999999}
df_ccodes = pd.concat([df_ccodes,pd.DataFrame(obs,index=[0])])

obs={"country":"German Democratic Republic", 
     "gw_codes":265,
     "iso_alpha3":99999999,
     "M49":"XYZ", 
     "StateAbb":99999999,
     "acled_codes":99999999,
     "gtd_codes":99999999,
     "vdem_codes":99999999}
df_ccodes = pd.concat([df_ccodes,pd.DataFrame(obs,index=[0])])

obs={"country":"Czechoslovakia", 
     "gw_codes":315,
     "iso_alpha3":99999999,
     "M49":"XYZ", 
     "StateAbb":99999999,
     "acled_codes":99999999,
     "gtd_codes":99999999,
     "vdem_codes":99999999}
df_ccodes = pd.concat([df_ccodes,pd.DataFrame(obs,index=[0])])

obs={"country":"Abkhazia", 
     "gw_codes":396,
     "iso_alpha3":99999999,
     "M49":"XYZ", 
     "StateAbb":99999999,
     "acled_codes":99999999,
     "gtd_codes":99999999,
     "vdem_codes":99999999}
df_ccodes = pd.concat([df_ccodes,pd.DataFrame(obs,index=[0])])

obs={"country":"South Ossetia", 
     "gw_codes":397,
     "iso_alpha3":99999999,
     "M49":"XYZ", 
     "StateAbb":99999999,
     "acled_codes":99999999,
     "gtd_codes":99999999,
     "vdem_codes":99999999}
df_ccodes = pd.concat([df_ccodes,pd.DataFrame(obs,index=[0])])

all_countries_s=all_countries.loc[all_countries["end"]>=1989]
countries_acled=fat["gw_codes"].unique()
countries=all_countries_s["gw_codes"].unique()
add = list(filter(lambda x: x not in countries_acled,countries))
                
### Add missing country-years for countries completely missing ------
for i in range(0, len(add)):
        # Check if country in data, if False add
        if (fat['gw_codes'] == add[i]).any() == False:
            for x in years:
                s = {'year':x,'country': df_ccodes[df_ccodes["gw_codes"]==add[i]]["country"].values[0],'gw_codes':add[i],'best':0,'high':0,'low':0,"count":0}
                s = pd.DataFrame(data=s,index=[0])
                fat = pd.concat([fat,s])  

### Sort and reset index -------
ucdp_final = fat.sort_values(by=["gw_codes","year"])
ucdp_final.reset_index(drop=True,inplace=True)
print("Added missing  ountries")
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
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==359)&(ucdp_final["year"]<=1991))]

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
ucdp_final.to_csv("data_out/ucdp_cy_osv.csv",sep=',')
print("Saved DataFrame!")








