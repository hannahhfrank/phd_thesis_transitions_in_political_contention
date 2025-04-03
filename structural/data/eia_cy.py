import pandas as pd
import numpy as np
     
### US Energy Infomration Administration
# https://www.eia.gov/international/data/world

dep = pd.read_csv("INT-Export-04-10-2024_15-06-49.csv", skiprows=1) # Oil deposits
dep.columns = ["index","country",1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
dep = dep.drop([0, 1])
dep = dep.drop(dep.columns[0], axis=1)

dep_long = dep.melt(id_vars='country')

dep_long.columns=["country","year","oil_deposits"]
dep_long=dep_long.sort_values(by=["country","year"])
dep_long=dep_long.reset_index(drop=True)

dep_long=dep_long.loc[dep_long["year"]>=1989]
dep_long["year"]=dep_long["year"].astype(int)
dep_long.loc[dep_long["oil_deposits"]=="--","oil_deposits"]=np.nan
dep_long.loc[dep_long["oil_deposits"]=="ie","oil_deposits"]=np.nan
dep_long["oil_deposits"]=dep_long["oil_deposits"].astype(float)
dep_long["country"]=dep_long["country"].str.strip()

dep2 = pd.read_csv("INT-Export-04-10-2024_17-25-37.csv", skiprows=1) # Oil production
dep2["Unnamed: 1"]=dep2["Unnamed: 1"].str.strip()
dep2.columns = ["index","country",'1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2023']
dep2_s=dep2.loc[dep2["country"]=="Total petroleum and other liquids (Mb/d)"]

dep2_s["country"]=["World",
'Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Angola',
       'Antarctica', 'Antigua and Barbuda', 'Argentina', 'Armenia',
       'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain',
       'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin',
       'Bermuda', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina',
       'Botswana', 'Brazil', 'British Virgin Islands', 'Brunei',
       'Bulgaria', 'Burkina Faso', 'Burma', 'Burundi', 'Cabo Verde',
       'Cambodia', 'Cameroon', 'Canada', 'Cayman Islands',
       'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia',
       'Comoros', 'Congo-Brazzaville', 'Congo-Kinshasa', 'Cook Islands',
       'Costa Rica', "Cote d'Ivoire", 'Croatia', 'Cuba', 'Cyprus',
       'Czechia', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic',
       'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea',
       'Estonia', 'Eswatini', 'Ethiopia', 'Falkland Islands',
       'Faroe Islands', 'Fiji', 'Finland', 'Former Czechoslovakia',
       'Former Serbia and Montenegro', 'Former U.S.S.R.',
       'Former Yugoslavia', 'France', 'French Guiana', 'French Polynesia',
       'Gabon', 'Gambia, The', 'Georgia', 'Germany', 'Germany, East',
       'Germany, West', 'Ghana', 'Gibraltar', 'Greece', 'Greenland',
       'Grenada', 'Guadeloupe', 'Guam', 'Guatemala', 'Guinea',
       'Guinea-Bissau', 'Guyana', 'Haiti', 'Hawaiian Trade Zone',
       'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India',
       'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy',
       'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati',
       'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon',
       'Lesotho', 'Liberia', 'Libya', 'Lithuania', 'Luxembourg', 'Macau',
       'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta',
       'Martinique', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia',
       'Moldova', 'Mongolia', 'Montenegro', 'Montserrat', 'Morocco',
       'Mozambique', 'Namibia', 'Nauru', 'Nepal', 'Netherlands',
       'Netherlands Antilles', 'New Caledonia', 'New Zealand',
       'Nicaragua', 'Niger', 'Nigeria', 'Niue', 'North Korea',
       'North Macedonia', 'Northern Mariana Islands', 'Norway', 'Oman',
       'Pakistan', 'Palestinian Territories', 'Panama',
       'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland',
       'Portugal', 'Puerto Rico', 'Qatar', 'Reunion', 'Romania', 'Russia',
       'Rwanda', 'Saint Helena', 'Saint Kitts and Nevis', 'Saint Lucia',
       'Saint Pierre and Miquelon', 'Saint Vincent/Grenadines', 'Samoa',
       'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia',
       'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia',
       'Solomon Islands', 'Somalia', 'South Africa', 'South Korea',
       'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden',
       'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania',
       'Thailand', 'The Bahamas', 'Timor-Leste', 'Togo', 'Tonga',
       'Trinidad and Tobago', 'Tunisia', 'Turkiye', 'Turkmenistan',
       'Turks and Caicos Islands', 'Tuvalu', 'U.S. Pacific Islands',
       'U.S. Territories', 'U.S. Virgin Islands', 'Uganda', 'Ukraine',
       'United Arab Emirates', 'United Kingdom', 'United States',
       'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam',
       'Wake Island', 'Western Sahara', 'Yemen', 'Zambia', 'Zimbabwe']
dep2_s = dep2_s.drop([2])
dep2_s = dep2_s.drop(dep2_s.columns[0], axis=1)

dep2_s_long = dep2_s.melt(id_vars='country')

dep2_s_long.columns=["country","year","oil_production"]
dep2_s_long=dep2_s_long.sort_values(by=["country","year"])
dep2_s_long=dep2_s_long.reset_index(drop=True)

dep2_s_long["year"]=dep2_s_long["year"].astype(int)
dep2_s_long=dep2_s_long.loc[dep2_s_long["year"]>=1989]
dep2_s_long.loc[dep2_s_long["oil_production"]=="--","oil_production"]=np.nan
dep2_s_long["oil_production"]=dep2_s_long["oil_production"].astype(float)

dep_long=pd.merge(dep_long,dep2_s_long,on=["country","year"],how="outer")

dep3 = pd.read_csv("INT-Export-04-10-2024_18-01-25.csv", skiprows=1) # Oil exports
dep3.columns = ["index","country",1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
dep3 = dep3.drop([0, 1])
dep3 = dep3.drop(dep3.columns[0], axis=1)

dep3_long = dep3.melt(id_vars='country')

dep3_long.columns=["country","year","oil_exports"]
dep3_long=dep3_long.sort_values(by=["country","year"])
dep3_long=dep3_long.reset_index(drop=True)

dep3_long=dep3_long.loc[dep3_long["year"]>=1989]
dep3_long["year"]=dep3_long["year"].astype(int)
dep3_long.loc[dep3_long["oil_exports"]=="--","oil_exports"]=np.nan
dep3_long["oil_exports"]=dep3_long["oil_exports"].astype(float)
dep3_long["country"]=dep3_long["country"].str.strip()

dep_long=pd.merge(dep_long,dep3_long,on=["country","year"],how="outer")

dep4 = pd.read_csv("INT-Export-04-10-2024_18-09-29.csv", skiprows=1) # Gas deposits
dep4.columns = ["index","country",1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
dep4 = dep4.drop([0, 1])
dep4 = dep4.drop(dep4.columns[0], axis=1)

dep4_long = dep4.melt(id_vars='country')

dep4_long.columns=["country","year","gas_deposits"]
dep4_long=dep4_long.sort_values(by=["country","year"])
dep4_long=dep4_long.reset_index(drop=True)

dep4_long=dep4_long.loc[dep4_long["year"]>=1989]
dep4_long["year"]=dep4_long["year"].astype(int)
dep4_long.loc[dep4_long["gas_deposits"]=="--","gas_deposits"]=np.nan
dep4_long.loc[dep4_long["gas_deposits"]=="ie","gas_deposits"]=np.nan
dep4_long["gas_deposits"]=dep4_long["gas_deposits"].astype(float)
dep4_long["country"]=dep4_long["country"].str.strip()

dep_long=pd.merge(dep_long,dep4_long,on=["country","year"],how="outer")

dep5 = pd.read_csv("INT-Export-04-10-2024_18-18-55.csv", skiprows=1) # Gas production
dep5.columns = ["index","country",1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
dep5 = dep5.drop([0, 1])
dep5 = dep5.drop(dep5.columns[0], axis=1)

dep5_long = dep5.melt(id_vars='country')

dep5_long.columns=["country","year","gas_production"]
dep5_long=dep5_long.sort_values(by=["country","year"])
dep5_long=dep5_long.reset_index(drop=True)

dep5_long=dep5_long.loc[dep5_long["year"]>=1989]
dep5_long["year"]=dep5_long["year"].astype(int)
dep5_long.loc[dep5_long["gas_production"]=="--","gas_production"]=np.nan
dep5_long.loc[dep5_long["gas_production"]=="ie","gas_production"]=np.nan
dep5_long["gas_production"]=dep5_long["gas_production"].astype(float)
dep5_long["country"]=dep5_long["country"].str.strip()

dep_long=pd.merge(dep_long,dep5_long,on=["country","year"],how="outer")

dep6 = pd.read_csv("INT-Export-04-10-2024_18-26-02.csv", skiprows=1) # Gas exports
dep6.columns = ["index","country",1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021,2022]
dep6 = dep6.drop([0, 1])
dep6 = dep6.drop(dep6.columns[0], axis=1)

dep6_long = dep6.melt(id_vars='country')

dep6_long.columns=["country","year","gas_exports"]
dep6_long=dep6_long.sort_values(by=["country","year"])
dep6_long=dep6_long.reset_index(drop=True)

dep6_long=dep6_long.loc[dep6_long["year"]>=1989]
dep6_long["year"]=dep6_long["year"].astype(int)
dep6_long.loc[dep6_long["gas_exports"]=="--","gas_exports"]=np.nan
dep6_long.loc[dep6_long["gas_exports"]=="ie","gas_exports"]=np.nan
dep6_long["gas_exports"]=dep6_long["gas_exports"].astype(float)
dep6_long["country"]=dep6_long["country"].str.strip()

dep_long=pd.merge(dep_long,dep6_long,on=["country","year"],how="outer")


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
dep_long.loc[dep_long["country"]=="Brunei","gw_codes"]=835
dep_long.loc[dep_long["country"]=="Bulgaria","gw_codes"]=355
dep_long.loc[dep_long["country"]=="Burkina Faso","gw_codes"]=439
dep_long.loc[dep_long["country"]=="Burma","gw_codes"]=775
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
dep_long.loc[dep_long["country"]=="Congo-Brazzaville","gw_codes"]=484
dep_long.loc[dep_long["country"]=="Congo-Kinshasa","gw_codes"]=490
dep_long.loc[dep_long["country"]=="Costa Rica","gw_codes"]=94
dep_long.loc[dep_long["country"]=="Cote d'Ivoire","gw_codes"]=437
dep_long.loc[dep_long["country"]=="Croatia","gw_codes"]=344
dep_long.loc[dep_long["country"]=="Cuba","gw_codes"]=40
dep_long.loc[dep_long["country"]=="Cyprus","gw_codes"]=352
dep_long.loc[dep_long["country"]=="Czechia","gw_codes"]=316

dep_long.loc[dep_long["country"]=="Denmark","gw_codes"]=390
dep_long.loc[dep_long["country"]=="Djibouti","gw_codes"]=522
dep_long.loc[dep_long["country"]=="Dominica","gw_codes"]=54
dep_long.loc[dep_long["country"]=="Dominican Republic","gw_codes"]=42

dep_long.loc[dep_long["country"]=="Ecuador","gw_codes"]=130
dep_long.loc[dep_long["country"]=="Egypt","gw_codes"]=651
dep_long.loc[dep_long["country"]=="El Salvador","gw_codes"]=92
dep_long.loc[dep_long["country"]=="Equatorial Guinea","gw_codes"]=411
dep_long.loc[dep_long["country"]=="Eritrea","gw_codes"]=531
dep_long.loc[dep_long["country"]=="Estonia","gw_codes"]=366
dep_long.loc[dep_long["country"]=="Eswatini","gw_codes"]=572
dep_long.loc[dep_long["country"]=="Ethiopia","gw_codes"]=530

dep_long.loc[dep_long["country"]=="Fiji","gw_codes"]=950
dep_long.loc[dep_long["country"]=="Finland","gw_codes"]=375
dep_long.loc[dep_long["country"]=="Former Czechoslovakia","gw_codes"]=315
dep_long.loc[dep_long["country"]=="Former Serbia and Montenegro","gw_codes"]=999999 ## <--- check with Serbia
dep_long.loc[dep_long["country"]=="Former U.S.S.R.","gw_codes"]=999999 ## <--- check with Russia
dep_long.loc[dep_long["country"]=="Former Yugoslavia","gw_codes"]=345
dep_long.loc[dep_long["country"]=="France","gw_codes"]=220

dep_long.loc[dep_long["country"]=="Gabon","gw_codes"]=481
dep_long.loc[dep_long["country"]=="Gambia, The","gw_codes"]=420
dep_long.loc[dep_long["country"]=="Georgia","gw_codes"]=372
dep_long.loc[dep_long["country"]=="Germany","gw_codes"]=260
dep_long.loc[dep_long["country"]=="Germany, East","gw_codes"]=265
dep_long.loc[dep_long["country"]=="Germany, West","gw_codes"]=999999 ## <--- check with Germany
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
dep_long.loc[dep_long["country"]=="Iran","gw_codes"]=630
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
dep_long.loc[dep_long["country"]=="Kyrgyzstan","gw_codes"]=703

dep_long.loc[dep_long["country"]=="Laos","gw_codes"]=812
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
dep_long.loc[dep_long["country"]=="North Korea","gw_codes"]=731
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
dep_long.loc[dep_long["country"]=="Russia","gw_codes"]=365
dep_long.loc[dep_long["country"]=="Rwanda","gw_codes"]=517

dep_long.loc[dep_long["country"]=="Saint Kitts and Nevis","gw_codes"]=60
dep_long.loc[dep_long["country"]=="Saint Lucia","gw_codes"]=56
dep_long.loc[dep_long["country"]=="Saint Vincent/Grenadines","gw_codes"]=57
dep_long.loc[dep_long["country"]=="Sao Tome and Principe","gw_codes"]=403
dep_long.loc[dep_long["country"]=="Saudi Arabia","gw_codes"]=670
dep_long.loc[dep_long["country"]=="Senegal","gw_codes"]=433
dep_long.loc[dep_long["country"]=="Serbia","gw_codes"]=340
dep_long.loc[dep_long["country"]=="Seychelles","gw_codes"]=591
dep_long.loc[dep_long["country"]=="Sierra Leone","gw_codes"]=451
dep_long.loc[dep_long["country"]=="Singapore","gw_codes"]=830
dep_long.loc[dep_long["country"]=="Slovakia","gw_codes"]=317
dep_long.loc[dep_long["country"]=="Slovenia","gw_codes"]=349
dep_long.loc[dep_long["country"]=="Solomon Islands","gw_codes"]=940
dep_long.loc[dep_long["country"]=="Somalia","gw_codes"]=520
dep_long.loc[dep_long["country"]=="South Africa","gw_codes"]=560
dep_long.loc[dep_long["country"]=="South Korea","gw_codes"]=732
dep_long.loc[dep_long["country"]=="South Sudan","gw_codes"]=626
dep_long.loc[dep_long["country"]=="Spain","gw_codes"]=230
dep_long.loc[dep_long["country"]=="Sri Lanka","gw_codes"]=780
dep_long.loc[dep_long["country"]=="Sudan","gw_codes"]=625
dep_long.loc[dep_long["country"]=="Suriname","gw_codes"]=115
dep_long.loc[dep_long["country"]=="Sweden","gw_codes"]=380
dep_long.loc[dep_long["country"]=="Switzerland","gw_codes"]=225
dep_long.loc[dep_long["country"]=="Syria","gw_codes"]=652

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
dep_long.loc[dep_long["country"]=="Turkiye","gw_codes"]=640
dep_long.loc[dep_long["country"]=="Turkmenistan","gw_codes"]=701
dep_long.loc[dep_long["country"]=="Tuvalu","gw_codes"]=973

dep_long.loc[dep_long["country"]=="Uganda","gw_codes"]=500
dep_long.loc[dep_long["country"]=="Ukraine","gw_codes"]=369
dep_long.loc[dep_long["country"]=="United Arab Emirates","gw_codes"]=696
dep_long.loc[dep_long["country"]=="United Kingdom","gw_codes"]=200
dep_long.loc[dep_long["country"]=="United States","gw_codes"]=2
dep_long.loc[dep_long["country"]=="Uruguay","gw_codes"]=165
dep_long.loc[dep_long["country"]=="Uzbekistan","gw_codes"]=704

dep_long.loc[dep_long["country"]=="Vanuatu","gw_codes"]=935
dep_long.loc[dep_long["country"]=="Venezuela","gw_codes"]=101
dep_long.loc[dep_long["country"]=="Vietnam","gw_codes"]=816

dep_long.loc[dep_long["country"]=="Yemen","gw_codes"]=678
dep_long.loc[dep_long["country"]=="Zambia","gw_codes"]=551
dep_long.loc[dep_long["country"]=="Zimbabwe","gw_codes"]=552

#####################
### Fix countries ###
#####################

### Oil deposits ###
dep_long.loc[dep_long["country"]=="Serbia"]
dep_long.loc[dep_long["country"]=="Former Serbia and Montenegro"]
dep_long.loc[dep_long["country"]=="Former Yugoslavia"]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2005),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2005)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2004),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2004)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2003),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2003)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2002),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2002)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2001),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2001)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2000),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2000)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1999),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1999)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1998),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1998)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1997),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1997)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1996),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1996)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1995),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1995)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1994),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1994)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1993),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1993)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1991),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1991)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1990),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1989),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1989)].iloc[0]

dep_long.loc[dep_long["country"]=="Germany"]
dep_long.loc[dep_long["country"]=="Germany, West"]
dep_long.loc[(dep_long["country"]=="Germany")&(dep_long["year"]==1990),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Germany, West")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Germany")&(dep_long["year"]==1989),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Germany, West")&(dep_long["year"]==1989)].iloc[0]


dep_long.loc[dep_long["country"]=="Russia"]
dep_long.loc[dep_long["country"]=="Former U.S.S.R."]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1989),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1989)].iloc[0]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1990),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1991),"oil_deposits"]=dep_long["oil_deposits"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1991)].iloc[0]

### Gas deposits ###
dep_long.loc[dep_long["country"]=="Serbia"]
dep_long.loc[dep_long["country"]=="Former Serbia and Montenegro"]
dep_long.loc[dep_long["country"]=="Former Yugoslavia"]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2005),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2005)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2004),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2004)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2003),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2003)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2002),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2002)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2001),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2001)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2000),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2000)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1999),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1999)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1998),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1998)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1997),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1997)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1996),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1996)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1995),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1995)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1994),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1994)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1993),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1993)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1991),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1991)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1990),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1989),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1989)].iloc[0]

dep_long.loc[dep_long["country"]=="Germany"]
dep_long.loc[dep_long["country"]=="Germany, West"]
dep_long.loc[(dep_long["country"]=="Germany")&(dep_long["year"]==1990),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Germany, West")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Germany")&(dep_long["year"]==1989),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Germany, West")&(dep_long["year"]==1989)].iloc[0]

dep_long.loc[dep_long["country"]=="Russia"]
dep_long.loc[dep_long["country"]=="Former U.S.S.R."]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1989),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1989)].iloc[0]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1990),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1991),"gas_deposits"]=dep_long["gas_deposits"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1991)].iloc[0]

### Oil production ###
dep_long.loc[dep_long["country"]=="Serbia"]
dep_long.loc[dep_long["country"]=="Former Serbia and Montenegro"]
dep_long.loc[dep_long["country"]=="Former Yugoslavia"]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2005),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2005)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2004),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2004)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2003),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2003)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2002),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2002)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2001),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2001)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2000),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2000)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1999),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1999)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1998),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1998)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1997),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1997)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1996),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1996)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1995),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1995)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1994),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1994)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1993),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1993)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1991),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1991)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1990),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1989),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1989)].iloc[0]

dep_long.loc[dep_long["country"]=="Germany"]
dep_long.loc[dep_long["country"]=="Germany, West"]
dep_long.loc[(dep_long["country"]=="Germany")&(dep_long["year"]==1990),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Germany, West")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Germany")&(dep_long["year"]==1989),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Germany, West")&(dep_long["year"]==1989)].iloc[0]

dep_long.loc[dep_long["country"]=="Russia"]
dep_long.loc[dep_long["country"]=="Former U.S.S.R."]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1989),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1989)].iloc[0]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1990),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1991),"oil_production"]=dep_long["oil_production"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1991)].iloc[0]

### Gas production ###
dep_long.loc[dep_long["country"]=="Serbia"]
dep_long.loc[dep_long["country"]=="Former Serbia and Montenegro"]
dep_long.loc[dep_long["country"]=="Former Yugoslavia"]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2005),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2005)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2004),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2004)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2003),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2003)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2002),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2002)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2001),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2001)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2000),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2000)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1999),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1999)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1998),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1998)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1997),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1997)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1996),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1996)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1995),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1995)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1994),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1994)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1993),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1993)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1991),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1991)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1990),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1989),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1989)].iloc[0]

dep_long.loc[dep_long["country"]=="Germany"]
dep_long.loc[dep_long["country"]=="Germany, West"]
dep_long.loc[(dep_long["country"]=="Germany")&(dep_long["year"]==1990),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Germany, West")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Germany")&(dep_long["year"]==1989),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Germany, West")&(dep_long["year"]==1989)].iloc[0]

dep_long.loc[dep_long["country"]=="Russia"]
dep_long.loc[dep_long["country"]=="Former U.S.S.R."]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1989),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1989)].iloc[0]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1990),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1991),"gas_production"]=dep_long["gas_production"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1991)].iloc[0]

### Oil esports ###
dep_long.loc[dep_long["country"]=="Serbia"]
dep_long.loc[dep_long["country"]=="Former Serbia and Montenegro"]
dep_long.loc[dep_long["country"]=="Former Yugoslavia"]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2005),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2005)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2004),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2004)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2003),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2003)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2002),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2002)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2001),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2001)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2000),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2000)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1999),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1999)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1998),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1998)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1997),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1997)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1996),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1996)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1995),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1995)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1994),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1994)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1993),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1993)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1991),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1991)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1990),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1989),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1989)].iloc[0]

dep_long.loc[dep_long["country"]=="Germany"]
dep_long.loc[dep_long["country"]=="Germany, West"]
dep_long.loc[(dep_long["country"]=="Germany")&(dep_long["year"]==1990),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Germany, West")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Germany")&(dep_long["year"]==1989),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Germany, West")&(dep_long["year"]==1989)].iloc[0]

dep_long.loc[dep_long["country"]=="Russia"]
dep_long.loc[dep_long["country"]=="Former U.S.S.R."]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1989),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1989)].iloc[0]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1990),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1991),"oil_exports"]=dep_long["oil_exports"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1991)].iloc[0]

### Gas esports ###
dep_long.loc[dep_long["country"]=="Serbia"]
dep_long.loc[dep_long["country"]=="Former Serbia and Montenegro"]
dep_long.loc[dep_long["country"]=="Former Yugoslavia"]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2005),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2005)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2004),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2004)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2003),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2003)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2002),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2002)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2001),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2001)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==2000),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==2000)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1999),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1999)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1998),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1998)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1997),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1997)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1996),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1996)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1995),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1995)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1994),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1994)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1993),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Serbia and Montenegro")&(dep_long["year"]==1993)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1991),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1991)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1990),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Serbia")&(dep_long["year"]==1989),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former Yugoslavia")&(dep_long["year"]==1989)].iloc[0]

dep_long.loc[dep_long["country"]=="Germany"]
dep_long.loc[dep_long["country"]=="Germany, West"]
dep_long.loc[(dep_long["country"]=="Germany")&(dep_long["year"]==1990),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Germany, West")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Germany")&(dep_long["year"]==1989),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Germany, West")&(dep_long["year"]==1989)].iloc[0]

dep_long.loc[dep_long["country"]=="Russia"]
dep_long.loc[dep_long["country"]=="Former U.S.S.R."]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1989),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1989)].iloc[0]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1990),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1990)].iloc[0]
dep_long.loc[(dep_long["country"]=="Russia")&(dep_long["year"]==1991),"gas_exports"]=dep_long["gas_exports"].loc[(dep_long["country"]=="Former U.S.S.R.")&(dep_long["year"]==1991)].iloc[0]



dep_long.reset_index(drop=True,inplace=True)
dep_long=dep_long[['country', 'gw_codes', 'year', 'oil_deposits','oil_production','oil_exports','gas_deposits','gas_production','gas_exports']]
dep_long.to_csv("data_out/eia_cy.csv",sep=',')
print("Saved DataFrame!")

