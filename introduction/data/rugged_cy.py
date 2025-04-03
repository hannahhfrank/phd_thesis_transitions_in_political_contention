### This file generates a csv file containing Ruggedness data on the country-month level of analysis ###

### Load libraries -------
import pandas as pd
import numpy as np


### Data and replication files for 'Ruggedness: The blessing of bad geography in Africa'
# https://diegopuga.org/data/rugged/

rug=pd.read_csv("rugged_data.csv", encoding='latin-1')
rug_s=rug[["isonum","country","rugged","land_area","soil","desert","tropical","cont_africa","cont_asia"]]

# Get country codes 
gw_codes=pd.read_csv("/Users/hannahfrank/phd_thesis_transitions_pol_contention/data/data_out/df_ccodes_gw.csv")
gw_codes_s=gw_codes.loc[gw_codes["end"]>=1989]

# Country codes 
ccodes={
"Abkhazia":[396,99999],      	
"Afghanistan":[700,4],      	
"Albania":[339,8],      	
"Algeria":[615,12],      	
"Andorra":[232,20],      	
"Angola":[540,24],      	
"Antigua & Barbuda":[58,28],      
"Argentina":[160,32],      	
"Armenia":[371,51], 
"Australia":[900,36],   
"Austria":[305,40],
"Azerbaijan":[373,31],      	
"Bahamas":[31,44],      
"Bahrain":[692,48], 
"Bangladesh":[771,50],  	
"Barbados":[53,52],  
"Belarus (Byelorussia)":[370,112],
"Belgium":[211,56],   
"Belize":[80,84],  
"Benin":[434,204],      
"Bhutan":[760,64],      	
"Bolivia":[145,68],   
"Bosnia-Herzegovina":[346,70],  
"Botswana":[571,72],
"Brazil":[140,76],
"Brunei":[835,96], 
"Bulgaria":[355,100], 
"Burkina Faso (Upper Volta)":[439,854],      	
"Burundi":[516,108],      	
"Cambodia (Kampuchea)":[811,116],
"Cameroon":[471,120],
"Canada":[20,124],
"Cape Verde":[402,132],
"Central African Republic":[482,140],
"Chad":[483,148],
"Chile":[155,152],
"China":[710,156],
"Colombia":[100,170],
"Comoros":[581,174],
"Congo":[484,178],
"Congo, Democratic Republic of (Zaire)"	:[490,180],
"Costa Rica":[94,188],
"Cote D’Ivoire":[437,384],
"Croatia":[344,191],
"Cuba":[40,192],
"Cyprus":[352,196],
"Czech Republic":[316,203],
"Czechoslovakia":[315,99999],
"Denmark":[390,208],
"Djibouti":[522,262],
"Dominica":[54,212],
"Dominican Republic":[42,214],
"East Timor":[860,626],
"Ecuador":[130,218],
"Egypt":[651,818],
"El Salvador":[92,222],
"Equatorial Guinea":[411,226],
"Eritrea":[531,232],
"Estonia":[366,233],
"Ethiopia":[530,231],
"Federated States of Micronesia":[987,583],
"Fiji":[950,242],
"Finland":[375,246],
"France":[220,250],
"Gabon":[481,266],
"Gambia":[420,270], 
"Georgia":[372,268],
"German Democratic Republic":[265,99999],
"German Federal Republic":[260,276],
"Ghana":[452,288],
"Greece":[350,300],
"Grenada":[55,308],
"Guatemala":[90,320],
"Guinea":[438,324],
"Guinea-Bissau":[404,624],
"Guyana":[110,328],
"Haiti":[41,332],
"Honduras":[91,340],
"Hungary":[310,348],
"Iceland":[395,352],
"India":[750,356],
"Indonesia":[850,360],
"Iran (Persia)":[630,364],
"Iraq":[645,368],
"Ireland":[205,372],
"Israel":[666,376],
"Italy/Sardinia":[325,380],
"Jamaica":[51,388],
"Japan":[740,392],
"Jordan":[663,400],
"Kazakhstan":[705,398],
"Kenya":[501,404],
"Kiribati":[970,296],
"Korea, People's Republic of":[731,408],
"Korea, Republic of":[732,410],
"Kosovo":[347,99999], # <----- Serbia and Montenegro
"Kuwait":[690,414],
"Kyrgyz Republic":[703,417],
"Laos":[812,418],
"Latvia":[367,428],
"Lebanon":[660,422],
"Lesotho":[570,426],
"Liberia":[450,430],
"Libya":[620,434],
"Liechtenstein":[223,438],
"Lithuania":[368,440],
"Luxembourg":[212,442],
"Macedonia (Former Yugoslav Republic of)":[343,807],
"Madagascar":[580,450],
"Malawi":[553,454],
"Malaysia":[820,458],
"Maldives":[781,462],
"Mali":[432,466],
"Malta":[338,470],
"Marshall Islands":[983,584],
"Mauritania":[435,478],
"Mauritius":[590,480],
"Mexico":[70,484],
"Moldova":[359,498],
"Monaco":[221,492],
"Mongolia":[712,496],
"Montenegro":[341,99999], # <----- Serbia and Montenegro
"Morocco":[600,504],
"Mozambique":[541,508],
"Myanmar (Burma)":[775,104],
"Namibia":[565,516],
"Nauru":[971,520],
"Nepal":[790,524],
"Netherlands":[210,528],
"New Zealand":[920,554],
"Nicaragua":[93,558],
"Niger":[436,562],
"Nigeria":[475,566],
"Norway":[385,578],
"Oman":[698,512],
"Pakistan":[770,586],
"Palau":[986,585],
"Panama":[95,591],
"Papua New Guinea":[910,598],
"Paraguay":[150,600],
"Peru":[135,604],
"Philippines":[840,608],
"Poland":[290,616],
"Portugal":[235,620],
"Qatar":[694,634],
"Rumania":[360,642],
"Russia (Soviet Union)":[365,643],
"Rwanda":[517,646],
"Saint Kitts and Nevis":[60,659],
"Saint Lucia":[56,662],
"Saint Vincent and the Grenadines":[57,670],
"Samoa/Western Samoa":[990,882],
"San Marino":[331,674],
"Saudi Arabia":[670,682],
"Senegal":[433,686],
"Serbia":[340,891], ###<---Serbia and Montenegro
"Seychelles":[591,690],
"Sierra Leone":[451,694],
"Singapore":[830,702],
"Slovakia":[317,703],
"Slovenia":[349,705],
"Solomon Islands":[940,90],
"Somalia":[520,706],
"South Africa":[560,710],
"South Ossetia":[397,99999],
"South Sudan":[626,99999], ###<------- Sudan
"Spain":[230,724],
"Sri Lanka (Ceylon)":[780,144],
"Sudan":[625,736],
"Surinam":[115,740],
"Swaziland":[572,748],
"Sweden":[380,752],
"Switzerland":[225,756],
"Syria":[652,760],
"São Tomé and Principe":[403,678],
"Taiwan":[713,158],
"Tajikistan":[702,762],
"Tanzania/Tanganyika":[510,834],
"Thailand":[800,764],
"Togo":[461,768],
"Tonga":[972,776],
"Trinidad and Tobago":[52,780],
"Tunisia":[616,788],
"Turkey (Ottoman Empire)":[640,792],
"Turkmenistan":[701,795],
"Tuvalu":[973,798],
"Uganda":[500,800],
"Ukraine":[369,804],
"United Arab Emirates":[696,784],
"United Kingdom":[200,826],
"United States of America":[2,840],
"Uruguay":[165,858],
"Uzbekistan":[704,860],
"Vanuatu":[935,548],
"Venezuela":[101,862],
"Vietnam, Democratic Republic of":[816,704],
"Yemen (Arab Republic of Yemen)":[678,887,],
"Yemen, People's Republic of":[680,99999],
"Yugoslavia":[345,99999],
"Zambia":[551,894],
"Zimbabwe (Rhodesia)":[552,716],  
}


### Convert dictionary with country codes to df ----
df_ccodes = pd.DataFrame.from_dict(ccodes,orient='index')
df_ccodes = df_ccodes.reset_index()
df_ccodes.columns = ['country','gw_codes','isonum']
rug_s=pd.merge(left=rug_s,right=df_ccodes[['gw_codes','isonum']],on=["isonum"],how="outer")                       
rug_s=rug_s[["country","gw_codes","rugged","land_area","soil","desert","tropical","cont_africa","cont_asia"]]

### Add ------
df_add=pd.DataFrame()
for c in ["Abkhazia","Czechoslovakia","German Democratic Republic","Kosovo","Montenegro","South Sudan","South Ossetia","Yemen (Arab Republic of Yemen)"]:
    print(c)
    s = {'country': c,
         "gw_codes":df_ccodes["gw_codes"].loc[df_ccodes["country"]==c].values[0],
         "rugged":np.nan,
         "land_area":np.nan,
         "soil":np.nan,
         "desert":np.nan,
         "tropical":np.nan,
         "cont_africa":np.nan,
         "cont_asia":np.nan}
    s = pd.DataFrame(data=s,index=[0])
    df_add = pd.concat([df_add,s])    
      
df_add.sort_values(by=["gw_codes"])
rug_s = rug_s.dropna(subset=["gw_codes"])
rug_s = rug_s.dropna(subset=["country"])
rug_s = pd.concat([rug_s,df_add])
rug_s = rug_s.reset_index(drop=True)
rug_s=rug_s.sort_values(by=["country"])

### Manual fixed, impute missing countries with 'closest' countries
rug_s.loc[rug_s["country"]=="South Sudan", "rugged"]=rug_s["rugged"].loc[rug_s["country"]=="Sudan"].values[0]
rug_s.loc[rug_s["country"]=="South Sudan", "soil"]=rug_s["soil"].loc[rug_s["country"]=="Sudan"].values[0]
rug_s.loc[rug_s["country"]=="South Sudan", "desert"]=rug_s["desert"].loc[rug_s["country"]=="Sudan"].values[0]
rug_s.loc[rug_s["country"]=="South Sudan", "tropical"]=rug_s["tropical"].loc[rug_s["country"]=="Sudan"].values[0]

rug_s.loc[rug_s["country"]=="Montenegro", "rugged"]=rug_s["rugged"].loc[rug_s["country"]=="Serbia and Montenegro"].values[0]
rug_s.loc[rug_s["country"]=="Montenegro", "soil"]=rug_s["soil"].loc[rug_s["country"]=="Serbia and Montenegro"].values[0]
rug_s.loc[rug_s["country"]=="Montenegro", "desert"]=rug_s["desert"].loc[rug_s["country"]=="Serbia and Montenegro"].values[0]
rug_s.loc[rug_s["country"]=="Montenegro", "tropical"]=rug_s["tropical"].loc[rug_s["country"]=="Serbia and Montenegro"].values[0]

rug_s.loc[rug_s["country"]=="Kosovo", "rugged"]=rug_s["rugged"].loc[rug_s["country"]=="Serbia and Montenegro"].values[0]
rug_s.loc[rug_s["country"]=="Kosovo", "soil"]=rug_s["soil"].loc[rug_s["country"]=="Serbia and Montenegro"].values[0]
rug_s.loc[rug_s["country"]=="Kosovo", "desert"]=rug_s["desert"].loc[rug_s["country"]=="Serbia and Montenegro"].values[0]
rug_s.loc[rug_s["country"]=="Kosovo", "tropical"]=rug_s["tropical"].loc[rug_s["country"]=="Serbia and Montenegro"].values[0]

### Agg data -------
df_agg=pd.DataFrame()
date = list(range(1989,2024))

for c in rug_s.gw_codes.unique():
    print(c)
    for d in date:

        s = {'year':d,
             'gw_codes':c,
             "country":df_ccodes["country"].loc[df_ccodes["gw_codes"]==c].values[0],
             "rugged":rug_s["rugged"].loc[rug_s["gw_codes"]==c].values[0],
             "land_area":rug_s["land_area"].loc[rug_s["gw_codes"]==c].values[0],
             "soil":rug_s["soil"].loc[rug_s["gw_codes"]==c].values[0],
             "desert":rug_s["desert"].loc[rug_s["gw_codes"]==c].values[0],
             "tropical":rug_s["tropical"].loc[rug_s["gw_codes"]==c].values[0],
             "cont_africa":rug_s["cont_africa"].loc[rug_s["gw_codes"]==c].values[0],
             "cont_asia":rug_s["cont_asia"].loc[rug_s["gw_codes"]==c].values[0]}
        s = pd.DataFrame(data=s,index=[0])
        df_agg = pd.concat([df_agg,s])  
df_agg.sort_values(by=["year","gw_codes"])

df_agg.to_csv("data_out/rug_cy.csv",sep=',')
print("Saved DataFrame!")  

