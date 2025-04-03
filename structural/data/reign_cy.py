### This file generates a csv file containing reign data on the country-month level of analysis ###

### Load libraries -------
import pandas as pd

### Rulers, Elections and Irregular Governance Dataset (REIGN) ----------
# Codebook: https://raw.githubusercontent.com/OEFDataScience/REIGN.github.io/gh-pages/documents/REIGN_CODEBOOK.pdf
reign = pd.read_csv("reign_8_21.csv", encoding='latin1')

# Drop leaders
#reign = reign.drop([51772,51773,51835,51836,123045,135911,135914,135915])

### Import country codes  -----
df_ccodes = pd.read_csv("df_ccodes.csv")

### Add ucdp codes -----
df_ccodes_s = df_ccodes[["gw_codes","iso_alpha3","acled_codes"]]

### Merge country codes -----
reign = pd.merge(reign,df_ccodes_s,how='left',left_on=['ccode'],right_on=['gw_codes'])

### Add dates ----
reign['dd'] = pd.to_datetime(reign['month'].astype(str)+reign['year'].astype(str),format='%m%Y').dt.to_period('M')

reign = reign[["country",
               #"ccode",
               "gw_codes",
               "iso_alpha3",
               "acled_codes",
               "dd",
               "year",
               "month",
               #"leader", # Provides the de-facto leader’s name.
               #"elected", # whether the de facto leader had previously been elected
               #"age", # leader’s age
               #"male", # ex of the de facto leader
               #"militarycareer", # career in the military, police force or defense ministry.
               "tenure_months", # months that a leader has been in power
               #"government", # regime type
               #"gov_democracy", # either a parliamentary democracy or presidential democrac
               "dem_duration", # logged number of months that a country has had a democratic government
               #"anticipation", # here is an election for the de facto leadership position coming within the next six-months.
               #"ref_ant", # here is a constitutional referendum coming within the next six-months
               #"leg_ant", # here is a legislative election to determine the de facto leader coming within the next six-months
               #"exec_ant", # here is an executive election to determine the de facto leader coming within the next six-months
               #"irreg_lead_ant", # an irregular election to determine the de facto leader is expected within the next six months
               "election_now", # there is an election for the de facto leadership position taking place in that country-month
               #"election_recent", # there is an election for the de facto leadership position that took place in the previous six months.
               #"leg_recent", # here is a legislative election took place in the previous six months
               #"exec_recent", # here is an executive election took place in the previous six months
               #"lead_recent", # if any electoral opportunity (non-referendum) to change leadership took place in the previous six months
               #"ref_recent", # there is a constitutional referendum took place in the previous six months
               #"direct_recent", # a direct (popular) election took place in the previous six months.
               #"indirect_recent", # an indirect (elite) election took place in the previous six months
               #"victory_recent", # an incumbent political party/leader won an election in the previous six months
               #"defeat_recent", #  an incumbent political party/leader won an election in the previous six months
               #"change_recent", # the de facto leader changed due to an election in the previous six months
               #"nochange_recent", # the de facto leader did not change following an election in the previous six months.
               #"delayed", # previously scheduled/expected election is cancelled by choice or through exogenous factors 
               "lastelection", # time since the last election
               #"loss", # ???
               #"irregular", # ???
               #"political_violence", # elative level (z-score) of political violence 
               #"prev_conflict", # umber of on-going violent civil and inter-state conflicts that the country was involved in during the previous month. 
               #"pt_suc", # a successful coup event took place in that month
               #"pt_attempt", #  coup attempt, regardless of success, took place in that month
               #"precip", #  measures the Standardized Precipitation Index (SPI) for each country month
               #"couprisk", # estimated probability of the risk of a military coup attempt taking place in the country-month.
               #"pctile_risk" # the percentile risk for each country’s estimated risk of a military coup attempt that month.
                  ]]
### Fix countries manually -----
reign.loc[reign["country"]=="Czechoslovakia", "gw_codes"] = 315
reign.loc[reign["country"]=="Czechoslovakia", "acled_codes"] = 99999999

reign.loc[reign["country"]=="Germany", "gw_codes"] = 260
reign.loc[reign["country"]=="Germany", "acled_codes"] = 276

reign.loc[reign["country"]=="Germany East", "gw_codes"] = 265
reign.loc[reign["country"]=="Germany East", "acled_codes"] = 99999999

reign.loc[reign["country"]=="Kiribati", "gw_codes"] = 970
reign.loc[reign["country"]=="Kiribati", "acled_codes"] = 296

reign.loc[reign["country"]=="Liechtenstein", "gw_codes"] = 99999999
reign.loc[reign["country"]=="Liechtenstein", "acled_codes"] = 438

reign.loc[reign["country"]=="Serbia", "gw_codes"] = 340
reign.loc[reign["country"]=="Serbia", "acled_codes"] = 688

reign.loc[reign["country"]=="Tonga", "gw_codes"] = 972
reign.loc[reign["country"]=="Tonga", "acled_codes"] = 776

reign.loc[reign["country"]=="Tuvalu", "gw_codes"] = 973
reign.loc[reign["country"]=="Tuvalu", "acled_codes"] = 99999999

reign.loc[reign["country"]=="Vietnam South", "gw_codes"] = 817
reign.loc[reign["country"]=="Vietnam South", "acled_codes"] = 99999999

reign.loc[reign["country"]=="Yemen South", "gw_codes"] = 680
reign.loc[reign["country"]=="Yemen South", "acled_codes"] = 99999999
reign.loc[reign["country"]=="Yemen", "acled_codes"] = 887

reign.loc[reign["country"]=="Yugoslavia", "gw_codes"] = 345
reign.loc[reign["country"]=="Yugoslavia", "acled_codes"] = 99999999

### Sort 
reign = reign.sort_values(by=["country","year","dd"])
reign.reset_index(drop=True,inplace=True)
print(reign.head(3))

reign.loc[reign["country"]=="Yemen", "gw_codes"]=678

group=pd.DataFrame(reign.groupby(["year","country"])['election_now'].max())
group=group.reset_index()
group.columns=["year","country","elections"]

reign=pd.merge(reign,group,on=["year","country"],how="left")



reign=reign.loc[reign["month"]==12]


### Save data ---------
reign=reign[['country', 'gw_codes', 'year', 'tenure_months', 'dem_duration', 'elections',
       'lastelection']]
reign["gw_codes"]=reign["gw_codes"].astype(int)
reign=reign.reset_index(drop=True)
reign=reign.sort_values(by=['gw_codes', 'year'])

# Remove duplicates
duplicates = reign.duplicated(subset=['country', 'year'],keep=False)
duplicate_rows = reign[duplicates]
reign=reign[~reign.index.isin(duplicate_rows.loc[duplicate_rows["tenure_months"]==1].index)]
duplicates = reign.duplicated(subset=['country', 'year'],keep=False)
duplicate_rows = reign[duplicates]

reign.to_csv("data_out/reign_cy.csv",sep=',')
print("Saved DataFrame!")



