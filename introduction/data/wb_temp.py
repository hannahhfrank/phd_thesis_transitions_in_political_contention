import pandas as pd

df = pd.read_excel("temp.xlsx")

df_s = df[df['code'].str.len() == 3]
df_s = df_s[df_s["code"]!="TWN"]
df_s = df_s[df_s["code"]!="ALA"]
df_s = df_s[df_s["code"]!="GLP"]
df_s = df_s[df_s["code"]!="GUF"]
df_s = df_s[df_s["code"]!="MTQ"]
df_s = df_s[df_s["code"]!="MYT"]
df_s = df_s[df_s["code"]!="REU"]

df_s.columns=['code', 'country', '1950', '1951', '1952', '1953', '1954',
           '1955', '1956', '1957', '1958', '1959', '1960',
           '1961', '1962', '1963', '1964', '1965', '1966',
           '1967', '1968', '1969', '1970', '1971', '1972',
           '1973', '1974', '1975', '1976', '1977', '1978',
           '1979', '1980', '1981', '1982', '1983', '1984',
           '1985', '1986', '1987', '1988', '1989', '1990',
           '1991', '1992', '1993', '1994', '1995', '1996',
           '1997', '1998', '1999', '2000', '2001', '2002',
           '2003', '2004', '2005', '2006', '2007', '2008',
           '2009', '2010', '2011', '2012', '2013', '2014',
           '2015', '2016', '2017', '2018', '2019', '2020',
           '2021', '2022']
df_ss=df_s[['country', '1950', '1951', '1952', '1953', '1954',
           '1955', '1956', '1957', '1958', '1959', '1960',
           '1961', '1962', '1963', '1964', '1965', '1966',
           '1967', '1968', '1969', '1970', '1971', '1972',
           '1973', '1974', '1975', '1976', '1977', '1978',
           '1979', '1980', '1981', '1982', '1983', '1984',
           '1985', '1986', '1987', '1988', '1989', '1990',
           '1991', '1992', '1993', '1994', '1995', '1996',
           '1997', '1998', '1999', '2000', '2001', '2002',
           '2003', '2004', '2005', '2006', '2007', '2008',
           '2009', '2010', '2011', '2012', '2013', '2014',
           '2015', '2016', '2017', '2018', '2019', '2020',
           '2021', '2022']]
df_long = df_ss.melt(id_vars='country')
df_long.columns=["country","year","temp"]
df_long["year"]=df_long["year"].astype(int)
dep_long=df_long.loc[df_long["year"]>=1989]

### Country codes #####
dep_long.country.unique()

dep_long["gw_codes"]=999999
dep_long.loc[dep_long["country"]=="Afghanistan","gw_codes"]=700
dep_long.loc[dep_long["country"]=="Albania","gw_codes"]=339
dep_long.loc[dep_long["country"]=="Algeria","gw_codes"]=615
dep_long.loc[dep_long["country"]=="Angola","gw_codes"]=540
dep_long.loc[dep_long["country"]=="Antigua and Barbuda","gw_codes"]=58
dep_long.loc[dep_long["country"]=="Argentina","gw_codes"]=160
dep_long.loc[dep_long["country"]=="Armenia","gw_codes"]=371
dep_long.loc[dep_long["country"]=="Australia","gw_codes"]=900
dep_long.loc[dep_long["country"]=="Austria","gw_codes"]=305
dep_long.loc[dep_long["country"]=="Azerbaijan","gw_codes"]=373

dep_long.loc[dep_long["country"]=="Bahrain","gw_codes"]=692
dep_long.loc[dep_long["country"]=="Bangladesh","gw_codes"]=771
dep_long.loc[dep_long["country"]=="Barbados","gw_codes"]=53
dep_long.loc[dep_long["country"]=="Belarus","gw_codes"]=370
dep_long.loc[dep_long["country"]=="Belgium","gw_codes"]=211
dep_long.loc[dep_long["country"]=="Belize","gw_codes"]=80
dep_long.loc[dep_long["country"]=="Benin","gw_codes"]=434
dep_long.loc[dep_long["country"]=="Bhutan","gw_codes"]=760
dep_long.loc[dep_long["country"]=="Bolivia","gw_codes"]=145
dep_long.loc[dep_long["country"]=="Bosnia and Herzegovina","gw_codes"]=346
dep_long.loc[dep_long["country"]=="Botswana","gw_codes"]=571
dep_long.loc[dep_long["country"]=="Brazil","gw_codes"]=140
dep_long.loc[dep_long["country"]=="Brunei Darussalam","gw_codes"]=835
dep_long.loc[dep_long["country"]=="Bulgaria","gw_codes"]=355
dep_long.loc[dep_long["country"]=="Burkina Faso","gw_codes"]=439
dep_long.loc[dep_long["country"]=="Myanmar","gw_codes"]=775
dep_long.loc[dep_long["country"]=="Burundi","gw_codes"]=516

dep_long.loc[dep_long["country"]=="Cabo Verde","gw_codes"]=402
dep_long.loc[dep_long["country"]=="Cambodia","gw_codes"]=811
dep_long.loc[dep_long["country"]=="Cameroon","gw_codes"]=471
dep_long.loc[dep_long["country"]=="Canada","gw_codes"]=20
dep_long.loc[dep_long["country"]=="Central African Republic","gw_codes"]=482
dep_long.loc[dep_long["country"]=="Chad","gw_codes"]=483
dep_long.loc[dep_long["country"]=="Chile","gw_codes"]=155
dep_long.loc[dep_long["country"]=="China","gw_codes"]=710
dep_long.loc[dep_long["country"]=="Colombia","gw_codes"]=100
dep_long.loc[dep_long["country"]=="Comoros","gw_codes"]=581
dep_long.loc[dep_long["country"]=="Congo","gw_codes"]=484
dep_long.loc[dep_long["country"]=="Democratic Republic of Congo","gw_codes"]=490
dep_long.loc[dep_long["country"]=="Costa Rica","gw_codes"]=94
dep_long.loc[dep_long["country"]=="Côte d'Ivoire","gw_codes"]=437
dep_long.loc[dep_long["country"]=="Croatia","gw_codes"]=344
dep_long.loc[dep_long["country"]=="Cuba","gw_codes"]=40
dep_long.loc[dep_long["country"]=="Cyprus","gw_codes"]=352
dep_long.loc[dep_long["country"]=="Czech Republic","gw_codes"]=316

dep_long.loc[dep_long["country"]=="Denmark","gw_codes"]=390
dep_long.loc[dep_long["country"]=="Djibouti","gw_codes"]=522
dep_long.loc[dep_long["country"]=="Dominica","gw_codes"]=54
dep_long.loc[dep_long["country"]=="Dominican Republic","gw_codes"]=42

dep_long.loc[dep_long["country"]=="Ecuador","gw_codes"]=130
dep_long.loc[dep_long["country"]=="Arab Republic of Egypt","gw_codes"]=651
dep_long.loc[dep_long["country"]=="El Salvador","gw_codes"]=92
dep_long.loc[dep_long["country"]=="Equatorial Guinea","gw_codes"]=411
dep_long.loc[dep_long["country"]=="Eritrea","gw_codes"]=531
dep_long.loc[dep_long["country"]=="Estonia","gw_codes"]=366
dep_long.loc[dep_long["country"]=="Eswatini","gw_codes"]=572
dep_long.loc[dep_long["country"]=="Ethiopia","gw_codes"]=530

dep_long.loc[dep_long["country"]=="Fiji","gw_codes"]=950
dep_long.loc[dep_long["country"]=="Finland","gw_codes"]=375
dep_long.loc[dep_long["country"]=="France","gw_codes"]=220

dep_long.loc[dep_long["country"]=="Gabon","gw_codes"]=481
dep_long.loc[dep_long["country"]=="The Gambia","gw_codes"]=420
dep_long.loc[dep_long["country"]=="Georgia","gw_codes"]=372
dep_long.loc[dep_long["country"]=="Germany","gw_codes"]=260
dep_long.loc[dep_long["country"]=="Ghana","gw_codes"]=452
dep_long.loc[dep_long["country"]=="Greece","gw_codes"]=350
dep_long.loc[dep_long["country"]=="Grenada","gw_codes"]=55
dep_long.loc[dep_long["country"]=="Guatemala","gw_codes"]=90
dep_long.loc[dep_long["country"]=="Guinea","gw_codes"]=438
dep_long.loc[dep_long["country"]=="Guinea-Bissau","gw_codes"]=404
dep_long.loc[dep_long["country"]=="Guyana","gw_codes"]=110

dep_long.loc[dep_long["country"]=="Haiti","gw_codes"]=41
dep_long.loc[dep_long["country"]=="Honduras","gw_codes"]=91
dep_long.loc[dep_long["country"]=="Hungary","gw_codes"]=310

dep_long.loc[dep_long["country"]=="Iceland","gw_codes"]=395
dep_long.loc[dep_long["country"]=="India","gw_codes"]=750
dep_long.loc[dep_long["country"]=="Indonesia","gw_codes"]=850
dep_long.loc[dep_long["country"]=="Islamic Republic of Iran","gw_codes"]=630
dep_long.loc[dep_long["country"]=="Iraq","gw_codes"]=645
dep_long.loc[dep_long["country"]=="Ireland","gw_codes"]=205
dep_long.loc[dep_long["country"]=="Israel","gw_codes"]=666
dep_long.loc[dep_long["country"]=="Italy","gw_codes"]=325

dep_long.loc[dep_long["country"]=="Jamaica","gw_codes"]=51
dep_long.loc[dep_long["country"]=="Japan","gw_codes"]=740
dep_long.loc[dep_long["country"]=="Jordan","gw_codes"]=663

dep_long.loc[dep_long["country"]=="Kazakhstan","gw_codes"]=705
dep_long.loc[dep_long["country"]=="Kenya","gw_codes"]=501
dep_long.loc[dep_long["country"]=="Kiribati","gw_codes"]=970
dep_long.loc[dep_long["country"]=="Kosovo","gw_codes"]=347
dep_long.loc[dep_long["country"]=="Kuwait","gw_codes"]=690
dep_long.loc[dep_long["country"]=="Kyrgyz Republic","gw_codes"]=703

dep_long.loc[dep_long["country"]=="Lao People's Democratic Republic","gw_codes"]=812
dep_long.loc[dep_long["country"]=="Latvia","gw_codes"]=367
dep_long.loc[dep_long["country"]=="Lebanon","gw_codes"]=660
dep_long.loc[dep_long["country"]=="Lesotho","gw_codes"]=570
dep_long.loc[dep_long["country"]=="Liberia","gw_codes"]=450
dep_long.loc[dep_long["country"]=="Libya","gw_codes"]=620
dep_long.loc[dep_long["country"]=="Lithuania","gw_codes"]=368
dep_long.loc[dep_long["country"]=="Luxembourg","gw_codes"]=212

dep_long.loc[dep_long["country"]=="Madagascar","gw_codes"]=580
dep_long.loc[dep_long["country"]=="Malawi","gw_codes"]=553
dep_long.loc[dep_long["country"]=="Malaysia","gw_codes"]=820
dep_long.loc[dep_long["country"]=="Maldives","gw_codes"]=781
dep_long.loc[dep_long["country"]=="Mali","gw_codes"]=432
dep_long.loc[dep_long["country"]=="Malta","gw_codes"]=338
dep_long.loc[dep_long["country"]=="Mauritius","gw_codes"]=590
dep_long.loc[dep_long["country"]=="Mauritania","gw_codes"]=435
dep_long.loc[dep_long["country"]=="Mexico","gw_codes"]=70
dep_long.loc[dep_long["country"]=="Micronesia","gw_codes"]=987
dep_long.loc[dep_long["country"]=="Moldova","gw_codes"]=359
dep_long.loc[dep_long["country"]=="Mongolia","gw_codes"]=712
dep_long.loc[dep_long["country"]=="Montenegro","gw_codes"]=341
dep_long.loc[dep_long["country"]=="Morocco","gw_codes"]=600
dep_long.loc[dep_long["country"]=="Mozambique","gw_codes"]=541

dep_long.loc[dep_long["country"]=="Namibia","gw_codes"]=565
dep_long.loc[dep_long["country"]=="Nauru","gw_codes"]=971
dep_long.loc[dep_long["country"]=="Nepal","gw_codes"]=790
dep_long.loc[dep_long["country"]=="Netherlands","gw_codes"]=210
dep_long.loc[dep_long["country"]=="New Zealand","gw_codes"]=920
dep_long.loc[dep_long["country"]=="Nicaragua","gw_codes"]=93
dep_long.loc[dep_long["country"]=="Niger","gw_codes"]=436
dep_long.loc[dep_long["country"]=="Nigeria","gw_codes"]=475
dep_long.loc[dep_long["country"]=="D. P. R. of Korea","gw_codes"]=731
dep_long.loc[dep_long["country"]=="North Macedonia","gw_codes"]=343
dep_long.loc[dep_long["country"]=="Norway","gw_codes"]=385

dep_long.loc[dep_long["country"]=="Oman","gw_codes"]=698

dep_long.loc[dep_long["country"]=="Pakistan","gw_codes"]=770
dep_long.loc[dep_long["country"]=="Panama","gw_codes"]=95
dep_long.loc[dep_long["country"]=="Papua New Guinea","gw_codes"]=910
dep_long.loc[dep_long["country"]=="Paraguay","gw_codes"]=150
dep_long.loc[dep_long["country"]=="Peru","gw_codes"]=135
dep_long.loc[dep_long["country"]=="Philippines","gw_codes"]=840
dep_long.loc[dep_long["country"]=="Poland","gw_codes"]=290
dep_long.loc[dep_long["country"]=="Portugal","gw_codes"]=235

dep_long.loc[dep_long["country"]=="Qatar","gw_codes"]=694

dep_long.loc[dep_long["country"]=="Romania","gw_codes"]=360
dep_long.loc[dep_long["country"]=="Russian Federation","gw_codes"]=365
dep_long.loc[dep_long["country"]=="Rwanda","gw_codes"]=517

dep_long.loc[dep_long["country"]=="Saint Kitts and Nevis","gw_codes"]=60
dep_long.loc[dep_long["country"]=="Saint Lucia","gw_codes"]=56
dep_long.loc[dep_long["country"]=="Saint Vincent and the Grenadines","gw_codes"]=57
dep_long.loc[dep_long["country"]=="São Tomé and Príncipe","gw_codes"]=403
dep_long.loc[dep_long["country"]=="Saudi Arabia","gw_codes"]=670
dep_long.loc[dep_long["country"]=="Senegal","gw_codes"]=433
dep_long.loc[dep_long["country"]=="Serbia","gw_codes"]=340
dep_long.loc[dep_long["country"]=="Seychelles","gw_codes"]=591
dep_long.loc[dep_long["country"]=="Sierra Leone","gw_codes"]=451
dep_long.loc[dep_long["country"]=="Singapore","gw_codes"]=830
dep_long.loc[dep_long["country"]=="Slovak Republic","gw_codes"]=317
dep_long.loc[dep_long["country"]=="Slovenia","gw_codes"]=349
dep_long.loc[dep_long["country"]=="Solomon Islands","gw_codes"]=940
dep_long.loc[dep_long["country"]=="Somalia","gw_codes"]=520
dep_long.loc[dep_long["country"]=="South Africa","gw_codes"]=560
dep_long.loc[dep_long["country"]=="Republic of Korea","gw_codes"]=732
dep_long.loc[dep_long["country"]=="South Sudan","gw_codes"]=626
dep_long.loc[dep_long["country"]=="Spain","gw_codes"]=230
dep_long.loc[dep_long["country"]=="Sri Lanka","gw_codes"]=780
dep_long.loc[dep_long["country"]=="Sudan","gw_codes"]=625
dep_long.loc[dep_long["country"]=="Suriname","gw_codes"]=115
dep_long.loc[dep_long["country"]=="Sweden","gw_codes"]=380
dep_long.loc[dep_long["country"]=="Switzerland","gw_codes"]=225
dep_long.loc[dep_long["country"]=="Syrian Arab Republic","gw_codes"]=652

dep_long.loc[dep_long["country"]=="Taiwan","gw_codes"]=713
dep_long.loc[dep_long["country"]=="Tajikistan","gw_codes"]=702
dep_long.loc[dep_long["country"]=="Tanzania","gw_codes"]=510
dep_long.loc[dep_long["country"]=="Thailand","gw_codes"]=800
dep_long.loc[dep_long["country"]=="The Bahamas","gw_codes"]=31
dep_long.loc[dep_long["country"]=="Timor-Leste","gw_codes"]=860
dep_long.loc[dep_long["country"]=="Togo","gw_codes"]=461
dep_long.loc[dep_long["country"]=="Tonga","gw_codes"]=972
dep_long.loc[dep_long["country"]=="Trinidad and Tobago","gw_codes"]=52
dep_long.loc[dep_long["country"]=="Tunisia","gw_codes"]=616
dep_long.loc[dep_long["country"]=="Türkiye","gw_codes"]=640
dep_long.loc[dep_long["country"]=="Turkmenistan","gw_codes"]=701
dep_long.loc[dep_long["country"]=="Tuvalu","gw_codes"]=973

dep_long.loc[dep_long["country"]=="Uganda","gw_codes"]=500
dep_long.loc[dep_long["country"]=="Ukraine","gw_codes"]=369
dep_long.loc[dep_long["country"]=="United Arab Emirates","gw_codes"]=696
dep_long.loc[dep_long["country"]=="United Kingdom","gw_codes"]=200
dep_long.loc[dep_long["country"]=="United States of America","gw_codes"]=2
dep_long.loc[dep_long["country"]=="Uruguay","gw_codes"]=165
dep_long.loc[dep_long["country"]=="Uzbekistan","gw_codes"]=704

dep_long.loc[dep_long["country"]=="Vanuatu","gw_codes"]=935
dep_long.loc[dep_long["country"]=="R. B. de Venezuela","gw_codes"]=101
dep_long.loc[dep_long["country"]=="Vietnam","gw_codes"]=816

dep_long.loc[dep_long["country"]=="Republic of Yemen","gw_codes"]=678
dep_long.loc[dep_long["country"]=="Zambia","gw_codes"]=551
dep_long.loc[dep_long["country"]=="Zimbabwe","gw_codes"]=552

dep_long=dep_long.loc[dep_long["gw_codes"]<999999]

dep_long.reset_index(drop=True,inplace=True)
dep_long=dep_long[['country', 'gw_codes', 'year', 'temp']]
dep_long.to_csv("data_out/temp_cy.csv",sep=',')
print("Saved DataFrame!")
