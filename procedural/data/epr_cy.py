import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
     
### Ethnic Power Relations (EPR) Dataset Family 2021 ----------
# Codebook: https://icr.ethz.ch/data/epr/core/EPR_2021_Codebook_EPR.pdf
erp = pd.read_csv("https://icr.ethz.ch/data/epr/core/EPR-2021.csv")

### Add missing observations ------
df=pd.DataFrame()
for i in range(len(erp)):
    date = list(range(erp['from'].iloc[i],erp['to'].iloc[i]+1))
    for x in range(0, len(date)):
        s = {'year':date[x],'gw_codes':erp['gwid'].iloc[i],'country':erp['statename'].iloc[i],'group':erp['group'].iloc[i],'group_id':erp['groupid'].iloc[i],'group_size':erp['size'].iloc[i],'group_status':erp['status'].iloc[i]}
        s = pd.DataFrame(data=s,index=[i])
        df = pd.concat([df,s])  
print("Added missing obersvationes") 

# Add religion, language, race
erp2 = pd.read_csv("https://icr.ethz.ch/data/epr/ed/ED-2021.csv")

# Make id
df['group_id'] = df['group_id'].apply(lambda x: '{:05d}'.format(x))
df['gwgroupid'] = df['gw_codes'].astype(str) + df['group_id'].astype(str)
df['gwgroupid'] = df['gwgroupid'].astype(int)

df=pd.merge(df,erp2[["gwgroupid","rel1_size","rel2_size","rel3_size","lang1_size","lang2_size","lang3_size","pheno1_size","pheno2_size","pheno3_size"]],on=["gwgroupid"],how="left")
df = df.replace(0, np.nan)
df['rel_frac'] = np.nan
df['lang_frac'] = np.nan
df['race_frac'] = np.nan


# Within group fractionalization
for i in range(len(df)):
    df['rel_frac'].iloc[i]=1-(np.sum(np.square(df[['rel1_size','rel2_size','rel3_size']].iloc[i].dropna().values)))
    df['lang_frac'].iloc[i]=1-(np.sum(np.square(df[['lang1_size','lang2_size','lang3_size']].iloc[i].dropna().values)))
    df['race_frac'].iloc[i]=1-(np.sum(np.square(df[['pheno1_size','pheno2_size','pheno3_size']].iloc[i].dropna().values)))

df['rel_frac'] = df['rel_frac'].apply(lambda x: max(0, x))
df['race_frac'] = df['race_frac'].apply(lambda x: max(0, x))

### Aggregate data ------
df_agg=pd.DataFrame()
date = list(range(1989, 2022, 1))

for c in df.gw_codes.unique():
    for d in date:
        s = {'year':d,
             'gw_codes':c,
             'country':df["country"].loc[(df["gw_codes"]==c)].iloc[0],
             'group_counts':len(df.loc[(df["gw_codes"]==c)&(df["year"]==d)]),
             'group_names':"-".join(df["group"].loc[(df["gw_codes"]==c)&(df["year"]==d)]),
             'monopoly_share':df["group_size"].loc[(df["gw_codes"]==c)&(df["year"]==d)&(df["group_status"]=="MONOPOLY")].sum(),
             'discriminated_share':df["group_size"].loc[(df["gw_codes"]==c)&(df["year"]==d)&(df["group_status"]=="DISCRIMINATED")].sum(),
             'powerless_share':df["group_size"].loc[(df["gw_codes"]==c)&(df["year"]==d)&(df["group_status"]=="POWERLESS")].sum(),
             'dominant_share':df["group_size"].loc[(df["gw_codes"]==c)&(df["year"]==d)&(df["group_status"]=="DOMINANT")].sum(),             
             'ethnic_frac':1-(np.sum(np.square(df["group_size"].loc[(df["gw_codes"]==c)&(df["year"]==d)].dropna().values))),
             'rel_frac':df["rel_frac"].loc[(df["gw_codes"]==c)&(df["year"]==d)].mean(),
             'lang_frac':df["lang_frac"].loc[(df["gw_codes"]==c)&(df["year"]==d)].mean(),
             'race_frac':df["race_frac"].loc[(df["gw_codes"]==c)&(df["year"]==d)].mean(),
             }
        s = pd.DataFrame(data=s,index=[0])
        df_agg = pd.concat([df_agg,s])  
    
### Save data ---------
df_agg = df_agg.sort_values(by=["gw_codes","year"])
df_agg.reset_index(drop=True,inplace=True)

df_agg.to_csv("data_out/epr_cy.csv",sep=',')
print("Saved DataFrame!")



# Validate
for c in df.country.unique():
    fig, axs = plt.subplots(figsize=(10, 5))
    plt.plot(df_agg["year"].loc[df_agg["country"]==c], df_agg["ethnic_frac"].loc[df_agg["country"]==c])








