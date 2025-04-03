### This file generates a csv and dictionary file containing ACLED data on series level ###

### Load libraries -------
import pandas as pd

### Load ACLED data from local folder --------
acled = pd.read_csv("acled_all_events.csv",low_memory=False)

### Only include protest events -----
df_protest = acled.loc[acled['event_type']=="Riots"].copy(deep=True)

# Palestine not included in UCDP GED
df_protest.loc[df_protest["country"]=="Palestine","country"]="Israel"
df_protest.loc[df_protest["country"]=="Palestine","iso"]=376

### Aggregate to month level -------
agg_month = pd.DataFrame(df_protest.groupby(["year","iso","country"]).size())
agg_month = agg_month.reset_index()
agg_month.rename(columns={0:"count"},inplace=True)
print("Aggregated data")
print(agg_month.head())

### Add fatalities -----
fat = pd.DataFrame(df_protest.groupby(["year","iso"])['fatalities'].sum())
fat = fat.reset_index()
agg_month = pd.merge(left=agg_month,right=fat,left_on=["year","iso"],right_on=["year","iso"])

### Drop countries which are not in Gleditsch and Ward data----
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

### And remove countrues which are not in ACLED ---------
df_ccodes = pd.read_csv("df_ccodes.csv")
all_countries_s = pd.merge(left=all_countries_s,right=df_ccodes[["acled_codes","gw_codes"]],left_on=["gw_codes"],right_on=["gw_codes"],how="left")
all_countries_ss=all_countries_s.loc[all_countries_s["acled_codes"]<1000]
countries=all_countries_ss.acled_codes.unique()
agg_month = agg_month[agg_month['iso'].isin(countries)]

### Get dates and countries --------
# Check coverage here:
# https://acleddata.com/acleddatanew/wp-content/uploads/dlm_uploads/2019/01/ACLED_Country-and-Time-Period-coverage_updatedFeb2022.pdf
country_dates={840:[2020,2023],
               124:[2021,2023],
               44:[2018,2023],
               192:[2018,2023],
               332:[2018,2023],
               214:[2018,2023],
               388:[2018,2023],
               780:[2018,2023],
               52:[2018,2023],
               212:[2018,2023],
               308:[2018,2023],
               662:[2018,2023],
               670:[2018,2023],
               28:[2018,2023],
               659:[2018,2023],       
               484:[2018,2023],
               84:[2018,2023],
               320:[2018,2023],
               340:[2018,2023],
               222:[2018,2023],
               558:[2018,2023],
               188:[2018,2023],
               591:[2018,2023],
               170:[2018,2023],
               862:[2018,2023],
               328:[2018,2023],
               740:[2018,2023],
               218:[2018,2023],
               604:[2018,2023],
               76:[2018,2023],
               68:[2018,2023],
               600:[2018,2023],
               152:[2018,2023],
               32:[2018,2023],
               858:[2018,2023],
               826:[2020,2023],
               372:[2020,2023],
               528:[2020,2023],
               56:[2020,2023],
               442:[2020,2023],
               250:[2020,2023],
               492:[2020,2023],
               438:[2020,2023],
               756:[2020,2023],
               724:[2020,2023],
               20:[2020,2023],
               620:[2020,2023],
               276:[2020,2023],
               616:[2020,2023],
               40:[2020,2023],
               348:[2020,2023],
               203:[2020,2023],
               703:[2020,2023],
               380:[2020,2023],
               674:[2020,2023],
               470:[2020,2023],
               8:[2018,2023],
               688:[2018,2023],
               499:[2018,2023],
               807:[2018,2023],
               191:[2018,2023],
               70:[2018,2023],
               0:[2018,2023],
               705:[2020,2023],
               300:[2018,2023],
               196:[2018,2023],
               100:[2018,2023],
               498:[2018,2023],
               642:[2018,2023],
               643:[2018,2023],
               233:[2020,2023],
               428:[2020,2023],
               440:[2020,2023],
               804:[2018,2023],
               112:[2018,2023],
               51:[2018,2023],
               268:[2018,2023],
               31:[2018,2023],
               246:[2020,2023],
               752:[2020,2023],
               578:[2020,2023],
               208:[2020,2023],
               352:[2020,2023],
               132:[2020,2023],
               678:[2020,2023],
               624:[1997,2023],
               226:[1997,2023],
               270:[1997,2023],
               466:[1997,2023],
               686:[1997,2023],
               204:[1997,2023],
               478:[1997,2023],
               562:[1997,2023],
               384:[1997,2023],
               324:[1997,2023],
               854:[1997,2023],
               430:[1997,2023],
               694:[1997,2023],
               288:[1997,2023],
               768:[1997,2023],
               120:[1997,2023],
               566:[1997,2023],
               266:[1997,2023],
               140:[1997,2023],
               148:[1997,2023],
               178:[1997,2023],
               180:[1997,2023],
               800:[1997,2023],
               404:[1997,2023],
               834:[1997,2023],
               108:[1997,2023],
               646:[1997,2023],
               706:[1997,2023],
               262:[1997,2023],
               231:[1997,2023],
               232:[1997,2023],
               24:[1997,2023],
               508:[1997,2023],
               894:[1997,2023],
               716:[1997,2023],
               454:[1997,2023],
               710:[1997,2023],
               516:[1997,2023],
               426:[1997,2023],
               72:[1997,2023],
               748:[1997,2023],
               450:[1997,2023],
               174:[2020,2023],
               480:[2020,2023],
               690:[2020,2023],
               504:[1997,2023],
               12:[1997,2023],
               788:[1997,2023],
               434:[1997,2023],
               729:[1997,2023],
               728:[2011,2023],
               364:[2016,2023],
               792:[2016,2023],
               368:[2016,2023],
               818:[1997,2023],
               760:[2017,2023],
               422:[2016,2023],
               400:[2016,2023],
               376:[2016,2023],
               682:[2015,2023],
               887:[2015,2023],
               414:[2016,2023],
               48:[2016,2023],
               634:[2016,2023],
               784:[2016,2023],
               512:[2016,2023],
               4:[2017,2023],
               795:[2018,2023],
               762:[2018,2023],
               417:[2018,2023],
               860:[2018,2023],
               398:[2018,2023],
               156:[2018,2023],
               496:[2018,2023],
               158:[2018,2023],
               408:[2018,2023],
               410:[2018,2023],
               392:[2018,2023],
               356:[2016,2023],
               64:[2020,2023],
               586:[2010,2023],
               50:[2010,2023],
               104:[2010,2023],
               144:[2010,2023],
               462:[2020,2023],
               524:[2010,2023],
               764:[2010,2023],
               116:[2010,2023],
               418:[2010,2023],
               704:[2010,2023],
               458:[2018,2023],
               702:[2020,2023],
               96:[2020,2023], 
               608:[2016,2023],
               360:[2015,2023],
               626:[2020,2023],
               36:[2021,2023],
               598:[2021,2023],
               554:[2021,2023],
               548:[2021,2023],
               90:[2021,2023],
               242:[2021,2023],
               296:[2021,2023], 
               520:[2021,2023], 
               776:[2021,2023],
               798:[2021,2023], 
               584:[2021,2023], 
               585:[2021,2023], 
               583:[2021,2023], 
               882:[2021,2023],
               }

# Loop through every country-month
for i in range(0, len(countries)):
    if agg_month['country'].loc[(agg_month["iso"]==countries[i])].any()==False:
        date = list(range(country_dates[countries[i]][0],country_dates[countries[i]][1]+1))
        
        for d in date:
            s = {'year':d,'country':all_countries_ss["country"].loc[all_countries_ss["acled_codes"]==countries[i]].iloc[0],'iso':countries[i],'count':0,'fatalities':0}
            s = pd.DataFrame(data=s,index=[0])
            agg_month = pd.concat([agg_month, s]) 
            
    else: 
        date = list(range(country_dates[countries[i]][0],country_dates[countries[i]][1]+1))

        for d in date:
            
            # Check if country-year in data, if False add
            if ((agg_month['year'] == d) 
                & (agg_month['iso'] == countries[i])).any() == False:
                    
                    # Subset data to add
                    s = {'year':d,'country':agg_month['country'].loc[(agg_month["iso"]==countries[i])].iloc[0],'iso':countries[i],'count':0,'fatalities':0}
                    s = pd.DataFrame(data=s,index=[0])
                    agg_month = pd.concat([agg_month,s])      
    
    
print("Added missing obersvationes")
agg_month = agg_month.sort_values(by=["country","year"])
agg_month.reset_index(drop=True,inplace=True)
print(agg_month.head(3))

### Convert data type -------
agg_month = pd.merge(left=agg_month,right=df_ccodes[["acled_codes","gw_codes"]],left_on=["iso"],right_on=["acled_codes"])
agg_month["gw_codes"] = agg_month["gw_codes"].astype(int)
agg_month["acled_codes"] = agg_month["acled_codes"].astype(int)
agg_month = agg_month[["year","country","gw_codes","acled_codes","count","fatalities"]]
agg_month.columns = ["year","country","gw_codes","acled_codes","n_riot_events","fatalities"]

### Save data ---------
agg_month = agg_month.sort_values(by=["country","year"])
agg_month.reset_index(drop=True,inplace=True)
agg_month.to_csv("data_out/acled_cy_riots.csv",sep=',')
print("Saved DataFrame!")



    
    
    
    
    
    
    
    