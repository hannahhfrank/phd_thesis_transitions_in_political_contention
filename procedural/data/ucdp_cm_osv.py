### This file generates a csv file containing UCDP GED data on the country-month level of analysis ###

### Load libraries -------
import pandas as pd
        
### UCDP Georeferenced Event Dataset ----------
# Codebook: https://ucdp.uu.se/downloads/ged/ged211.pdf
ucdp = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged231-csv.zip",low_memory=False)

### Only use onse-sided violence violence ---------
ucdp_s = ucdp[(ucdp["type_of_violence"]==3)].copy(deep=True)

### Add dates ----
ucdp_s["dd_date_start"] = pd.to_datetime(ucdp_s['date_start'],format='%Y-%m-%d %H:%M:%S.000')
ucdp_s["dd_date_end"] = pd.to_datetime(ucdp_s['date_end'],format='%Y-%m-%d %H:%M:%S.000')

# Only store month
ucdp_s["month_date_start"] = ucdp_s["dd_date_start"].dt.strftime('%m')
ucdp_s["month_date_end"] = ucdp_s["dd_date_end"].dt.strftime('%m')
ucdp_date = ucdp_s[["year","dd_date_start","dd_date_end","active_year","country","country_id","date_prec","best","high","low","month_date_start","month_date_end"]].copy(deep=True)

# Reset index 
ucdp_date = ucdp_date.sort_values(by=["country", "year"],ascending=True)
ucdp_date.reset_index(drop=True, inplace=True)

### Loop through data and delete observations which comprise more than one month ------
ucdp_final = ucdp_date.copy()
for i in range(0,len(ucdp_date)):
    if ucdp_date["month_date_start"].loc[i]!=ucdp_date["month_date_end"].loc[i]:
        ucdp_final = ucdp_final.drop(index=i, axis=0)       
print("Added dates and deleted observations which comprise more than on month")

### Generate year_month variable --------
ucdp_final['dd'] = pd.to_datetime(ucdp_final['dd_date_start'],format='%Y-%m').dt.to_period('M')

### Aggregate to month level -------
agg_month = pd.DataFrame(ucdp_final.groupby(["dd","year","country_id"]).size())
agg_month = agg_month.reset_index()
agg_month.rename(columns={0:"count"},inplace=True)
print("Aggregated data")

### Aggregate fatality variables ----------
best = pd.DataFrame(ucdp_final.groupby(["dd","country_id"])['best'].sum())
high = pd.DataFrame(ucdp_final.groupby(["dd","country_id"])['high'].sum())
low = pd.DataFrame(ucdp_final.groupby(["dd","country_id"])['low'].sum())

### Merge fatality variables and reset index ---------
best_high = pd.concat([best, high], axis=1)
fat =  pd.concat([best_high, low], axis=1)
fat.head()
if "year" in fat == False:
    fat["year"] = fat.index.get_level_values(0)
if "country_id" in fat == False:
    fat["country_id"] = fat.index.get_level_values(1)  
ucdp_fat = fat.reset_index()

### Re-obtain year variable ------------
ucdp_fat["year"] = None
for i in ucdp_fat.index:
    ucdp_fat.loc[i, "year"] = ucdp_fat.loc[i, 'dd'].year
    
### Re-obtain country variables --------
ucdp_cc = ucdp[["country_id", "country"]].drop_duplicates().reset_index(drop=True)
ucdp_final = pd.merge(ucdp_fat,ucdp_cc,how='left',on='country_id')

### Merge ----
ucdp_final = pd.merge(left=ucdp_final,right=agg_month[["dd","country_id","count"]],left_on=["dd","country_id"],right_on=["dd","country_id"])
print(ucdp_final.head())

### Get countries and years --------
years = list(ucdp_final.year.unique())
countries = list(ucdp_final.country_id.unique())

### Make range of time stamps to add missing observations -------
date = list(pd.date_range(start="1989-01",end="2022-12",freq="MS"))
date = pd.to_datetime(date, format='%Y-%m').to_period('M')

### Add missing observation to time series, those have zero fatalities --------

# Loop through every country-month
for i in range(0, len(countries)):
    print(countries[i])
    for x in range(0, len(date)):
        
        # Check if country-month in data, if False add
        if ((ucdp_final['dd'] == date[x]) 
            & (ucdp_final['country_id'] == countries[i])).any() == False:
                
                # Subset data to add
                s = {'dd':date[x],'year':date[x].year,'country':ucdp_final['country'].loc[(ucdp_final["country_id"]==countries[i])].iloc[0],'country_id':[countries[i]],'best':0,'high':0,'low':0,'count':0}
                s = pd.DataFrame(data=s)
                ucdp_final = pd.concat([ucdp_final,s])  
print("Added missing obersvationes")

### Import country codes  -----
ucdp_final.rename(columns = {'country_id':'gw_codes'},inplace = True)
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
countries_acled=ucdp_final["gw_codes"].unique()
countries=all_countries_s["gw_codes"].unique()
add = list(filter(lambda x: x not in countries_acled,countries))

for i in range(0, len(add)):
    print(i)
    print(df_ccodes[df_ccodes["gw_codes"]==add[i]]["country"].values[0])
    for d in range(0, len(date)):
        s = {'dd':date[d],'year':date[d].year,'country':df_ccodes[df_ccodes["gw_codes"]==add[i]]["country"].values[0],'gw_codes':add[i],'best':0,'high':0,'low':0,'count':0}
        s = pd.DataFrame(data=s,index=[0])
        ucdp_final = pd.concat([ucdp_final, s])  
print("Added missing countries")
print(ucdp_final.head())

### Sort and reset index -------
ucdp_final = ucdp_final[["dd","year","country","gw_codes","best","high","low","count"]]
ucdp_final = ucdp_final.sort_values(by=["country","year","dd"])
ucdp_final.reset_index(drop=True,inplace=True)
print(ucdp_final.head(3))

### Check independence and remove obs ----------


### Dissolution ###

# Czechoslovakia, split 1 January 1993
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==315)&(ucdp_final["year"]>1992))]

# Yemen, People's Republic of, unification 22 May 1990
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==680)&(ucdp_final["year"]>1990))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==680)&(ucdp_final["dd"]==date[17]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==680)&(ucdp_final["dd"]==date[18]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==680)&(ucdp_final["dd"]==date[19]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==680)&(ucdp_final["dd"]==date[20]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==680)&(ucdp_final["dd"]==date[21]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==680)&(ucdp_final["dd"]==date[22]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==680)&(ucdp_final["dd"]==date[23]))]

# German Democratic Republic, reunification 3 October 1990
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==265)&(ucdp_final["year"]>1990))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==265)&(ucdp_final["dd"]==date[21]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==265)&(ucdp_final["dd"]==date[22]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==265)&(ucdp_final["dd"]==date[23]))]

# Yugoslavia, Serbia independence decalartion 5 June 2006 
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==345)&(ucdp_final["year"]>2006))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==345)&(ucdp_final["dd"]==date[209]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==345)&(ucdp_final["dd"]==date[210]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==345)&(ucdp_final["dd"]==date[211]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==345)&(ucdp_final["dd"]==date[212]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==345)&(ucdp_final["dd"]==date[213]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==345)&(ucdp_final["dd"]==date[214]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==345)&(ucdp_final["dd"]==date[215]))]

### New states ###

# Namibia 21 March 1990 independence
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==565)&(ucdp_final["year"]<1990))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==565)&(ucdp_final["dd"]==date[12]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==565)&(ucdp_final["dd"]==date[13]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==565)&(ucdp_final["dd"]==date[14]))]

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
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==367)&(ucdp_final["dd"]==date[24]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==367)&(ucdp_final["dd"]==date[25]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==367)&(ucdp_final["dd"]==date[26]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==367)&(ucdp_final["dd"]==date[27]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==367)&(ucdp_final["dd"]==date[28]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==367)&(ucdp_final["dd"]==date[29]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==367)&(ucdp_final["dd"]==date[30]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==367)&(ucdp_final["dd"]==date[31]))]

# Estonia 6 Sepmetmeber 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==366)&(ucdp_final["year"]<1991))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==366)&(ucdp_final["dd"]==date[24]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==366)&(ucdp_final["dd"]==date[25]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==366)&(ucdp_final["dd"]==date[26]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==366)&(ucdp_final["dd"]==date[27]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==366)&(ucdp_final["dd"]==date[28]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==366)&(ucdp_final["dd"]==date[29]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==366)&(ucdp_final["dd"]==date[30]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==366)&(ucdp_final["dd"]==date[31]))]

# Lithuanua 6 Sepmetmeber 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==368)&(ucdp_final["year"]<1991))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==368)&(ucdp_final["dd"]==date[24]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==368)&(ucdp_final["dd"]==date[25]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==368)&(ucdp_final["dd"]==date[26]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==368)&(ucdp_final["dd"]==date[27]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==368)&(ucdp_final["dd"]==date[28]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==368)&(ucdp_final["dd"]==date[29]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==368)&(ucdp_final["dd"]==date[30]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==368)&(ucdp_final["dd"]==date[31]))]

# Macedonia independence referendum 8 September 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==343)&(ucdp_final["year"]<1991))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==343)&(ucdp_final["dd"]==date[24]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==343)&(ucdp_final["dd"]==date[25]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==343)&(ucdp_final["dd"]==date[26]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==343)&(ucdp_final["dd"]==date[27]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==343)&(ucdp_final["dd"]==date[28]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==343)&(ucdp_final["dd"]==date[29]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==343)&(ucdp_final["dd"]==date[30]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==343)&(ucdp_final["dd"]==date[31]))]

# Bosnia-Herzegovina independence 3 March 1992
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==346)&(ucdp_final["year"]<1992))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==346)&(ucdp_final["dd"]==date[36]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==346)&(ucdp_final["dd"]==date[37]))]

# Montenegro independence restored 3 June 2006
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==341)&(ucdp_final["year"]<2006))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==341)&(ucdp_final["dd"]==date[208]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==341)&(ucdp_final["dd"]==date[207]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==341)&(ucdp_final["dd"]==date[206]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==341)&(ucdp_final["dd"]==date[205]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==341)&(ucdp_final["dd"]==date[204]))]

# Kosovo, declaration of independence 17 February 2008
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==347)&(ucdp_final["year"]<2008))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==347)&(ucdp_final["dd"]==date[228]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==347)&(ucdp_final["dd"]==date[229]))]

# Cezch republic establishment 1 January 1993
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==316)&(ucdp_final["year"]<1993))]

# Slovakia establishment 1 January 1993
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==317)&(ucdp_final["year"]<1993))]

# Slovenia, independence 25 June 1991
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==349)&(ucdp_final["year"]<1991))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==349)&(ucdp_final["dd"]==date[24]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==349)&(ucdp_final["dd"]==date[25]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==349)&(ucdp_final["dd"]==date[26]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==349)&(ucdp_final["dd"]==date[27]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==349)&(ucdp_final["dd"]==date[28]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==349)&(ucdp_final["dd"]==date[29]))]

# Croatia, recognized 22 May 1992
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==344)&(ucdp_final["year"]<1992))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==344)&(ucdp_final["dd"]==date[36]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==344)&(ucdp_final["dd"]==date[37]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==344)&(ucdp_final["dd"]==date[38]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==344)&(ucdp_final["dd"]==date[39]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==344)&(ucdp_final["dd"]==date[40]))]

# Eritrea, independence 24 May 1993
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==531)&(ucdp_final["year"]<1993))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==531)&(ucdp_final["dd"]==date[48]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==531)&(ucdp_final["dd"]==date[49]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==531)&(ucdp_final["dd"]==date[50]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==531)&(ucdp_final["dd"]==date[51]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==531)&(ucdp_final["dd"]==date[52]))]

# Palau, independence 1 October 1994
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==986)&(ucdp_final["year"]<1994))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==986)&(ucdp_final["dd"]==date[60]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==986)&(ucdp_final["dd"]==date[61]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==986)&(ucdp_final["dd"]==date[62]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==986)&(ucdp_final["dd"]==date[63]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==986)&(ucdp_final["dd"]==date[64]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==986)&(ucdp_final["dd"]==date[65]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==986)&(ucdp_final["dd"]==date[66]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==986)&(ucdp_final["dd"]==date[67]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==986)&(ucdp_final["dd"]==date[68]))]

# East Timor, independence restored 20 May 2002
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==860)&(ucdp_final["year"]<2002))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==860)&(ucdp_final["dd"]==date[156]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==860)&(ucdp_final["dd"]==date[157]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==860)&(ucdp_final["dd"]==date[158]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==860)&(ucdp_final["dd"]==date[159]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==860)&(ucdp_final["dd"]==date[160]))]

# Serbia, independence decalartion 5 June 2006 
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==340)&(ucdp_final["year"]<2006))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==340)&(ucdp_final["dd"]==date[204]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==340)&(ucdp_final["dd"]==date[205]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==340)&(ucdp_final["dd"]==date[206]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==340)&(ucdp_final["dd"]==date[207]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==340)&(ucdp_final["dd"]==date[208]))]

# South Ossetia, 26 August 2008 recognition by Russia
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==397)&(ucdp_final["year"]<2008))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==397)&(ucdp_final["dd"]==date[228]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==397)&(ucdp_final["dd"]==date[229]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==397)&(ucdp_final["dd"]==date[230]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==397)&(ucdp_final["dd"]==date[231]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==397)&(ucdp_final["dd"]==date[232]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==397)&(ucdp_final["dd"]==date[233]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==397)&(ucdp_final["dd"]==date[234]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==397)&(ucdp_final["dd"]==date[235]))]

# Abkhazia, 28 August 2008 declared as occupied by Russia
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==396)&(ucdp_final["year"]<2008))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==396)&(ucdp_final["dd"]==date[228]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==396)&(ucdp_final["dd"]==date[229]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==396)&(ucdp_final["dd"]==date[230]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==396)&(ucdp_final["dd"]==date[231]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==396)&(ucdp_final["dd"]==date[232]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==396)&(ucdp_final["dd"]==date[233]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==396)&(ucdp_final["dd"]==date[234]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==396)&(ucdp_final["dd"]==date[235]))]

# South Sudan, 9 July 2011
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==626)&(ucdp_final["year"]<2011))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==626)&(ucdp_final["dd"]==date[269]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==626)&(ucdp_final["dd"]==date[268]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==626)&(ucdp_final["dd"]==date[267]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==626)&(ucdp_final["dd"]==date[266]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==626)&(ucdp_final["dd"]==date[265]))]
ucdp_final=ucdp_final.loc[~((ucdp_final["gw_codes"]==626)&(ucdp_final["dd"]==date[264]))]

### Save data ---------
ucdp_final = ucdp_final.sort_values(by=["gw_codes","year","dd"])
ucdp_final.reset_index(drop=True,inplace=True)
ucdp_final.to_csv("data_out/ucdp_cm_osv.csv",sep=',')
print("Saved DataFrame!")



