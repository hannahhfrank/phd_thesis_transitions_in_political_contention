### This file generates a csv and dictionary file containing country codes ###

### Load libraries -------
import pandas as pd
import os
import pickle

# Check here: <------
#https://unstats.un.org/unsd/methodology/m49/

#http://ksgleditsch.com/statelist.html
#https://dornsife.usc.edu/assets/sites/298/docs/country_to_gwno_PUBLIC_6-5-2015.txt
#https://wits.worldbank.org/wits/wits/witshelp/content/codes/country_codes.htm
#https://correlatesofwar.org/data-sets/cow-country-codes
#https://acleddata.com/resources/
#https://www.start.umd.edu/gtd/downloads/Codebook.pdf
#https://v-dem.net/documents/25/countryunit_v13.pdf

# Columns: 
# (1) Gleditsch & Ward country codes
# (2) ISO-alpha3 code / World Bank codes
# (3) M49 code
# (4) StateAbb
# (5) ACLED
# (6) GTD
# (7) V-dem

c_codes = { 'Abkhazia': [396,'XYZ','XYZ','XYZ',99999999,99999999,99999999],
            'Afghanistan': [700, 'AFG', '004', 'AFG', 4, 4, 36],
            ###'Åland Islands': [99999999, 'ALA', '248', 'xyz', 99999999, 99999999, 99999999],
            'Albania': [339, 'ALB', '008', 'ALB', 8, 5, 12],
            'Algeria': [615, 'DZA', '012', 'ALG', 12, 6, 103],
            ###'American Samoa': [99999999, 'ASM', '016', 'ASM', 16, 99999999, 99999999], 
            'Andorra': [232, 'AND', '020', 'AND', 20, 7, 142], 
            'Angola': [540, 'AGO', '024', 'ANG', 24, 8, 104], 
            ###'Anguilla': [99999999, 'AGO', '024', 'AGO', 660, 99999999, 99999999], 
            ###'Antarctica': [99999999, 'ATA', '010', 'ATA', 10, 99999999, 99999999],
            'Antigua and Barbuda': [58, 'ATG', '028', 'AAB', 28, 10, 99999999], 
            'Argentina': [160, 'ARG', '032', 'ARG', 32, 11, 37], 
            'Armenia': [371, 'ARM', '051', 'ARM', 51, 12, 105], 
            ###'Aruba': [99999999, 'ABW', '533', 'ABW', 533, 553, 99999999, 99999999], 
            'Australia': [900, 'AUS', '036', 'AUL', 36, 14, 67],
            'Austria': [305, 'AUT', '040', 'AUS', 40, 15, 144], 
            'Azerbaijan': [373, 'AZE', '031', 'AZE', 31, 16, 106],
            'Bahamas': [31, 'BHS', '044', 'BHM', 44, 17, 99999999],
            'Bahrain': [692, 'BHR', '048', 'BAH', 48, 18, 146],
            'Bangladesh': [771, 'BGD', '050', 'BNG', 50, 19, 24],
            'Barbados': [53, 'BRB', '052', 'BAR', 52, 20, 147],
            'Belarus': [370, 'BLR', '112', 'BLR', 112, 35, 107],
            'Belgium': [211, 'BEL', '056', 'BEL', 56, 21, 148],
            'Belize': [80, 'BLZ', '084', 'BLZ', 84, 22, 99999999],
            'Benin': [434, 'BEN', '204', 'BEN', 204, 23, 52],
            ###'Bermuda': [99999999, 'BMU', '060', 'BMU', 60, 24,99999999],
            'Bhutan': [760, 'BTN', '064', 'BHU', 64, 25, 53],
            'Bolivia': [145, 'BOL', '064', 'BOL', 68, 26, 25],
            ###'Bonaire, Sint Eustatius and Saba': [99999999, 'BES', '535', 'BES', 99999999, 99999999, 99999999],
            'Bosnia and Herzegovina': [346, 'BIH', '070', 'BOS', 70, 28, 150],
            'Botswana': [571, 'BWA', '072', 'BOT', 72, 29, 68],
            ###'Bouvet Island': [99999999, 'BVT', '074', 'BVT', 99999999, 99999999, 99999999],
            'Brazil': [140, 'BRA', '076', 'BRA', 76, 30, 19],
            ###'British Indian Ocean Territory': [99999999, 'IOT', '086', 'IOT', 99999999, 99999999, 99999999],
            ###'British Virgin Islands': [99999999, 'VGB', '092', 'VGB', 92, 99999999, 99999999],
            'Brunei Darussalam': [835, 'BRN', '096', 'BRU', 96, 31, 151],
            'Bulgaria': [355, 'BGR', '100', 'BUL', 100, 32, 152],
            'Burkina Faso': [439, 'BFA', '854', 'BFO', 854, 33, 54],
            'Burundi': [516, 'BDI', '108', 'BUI', 108, 34, 69],
            'Cabo Verde': [402, 'CPV', '132', 'CAP', 132, 99999999, 70],
            'Cambodia (Kampuchea)': [811, 'KHM', '116', 'CAM', 116, 36, 55],
            'Cameroon': [471, 'CMR', '120', 'CAO', 120, 37, 108],
            'Canada': [20, 'CAN', '124', 'CAN', 124, 38, 66],
            ###'Cayman Islands': [99999999, 'CYM', '136', 'CYM', 136, 99999999, 99999999],
            'Central African Republic': [482, 'CAF', '140', 'CEN', 140, 41, 71],
            'Chad': [483, 'TCD', '148', 'CHA', 148, 42, 109],
            'Chile': [155, 'CHL', '152', 'CHL', 152, 43, 72],
            'China': [710, 'CHN', '156', 'CHN', 156, 44, 110],
            ###'China, Hong Kong Special Administrative Region': [99999999, 'HKG', '344', 'HKG', 99999999, 89, 167],
            ###'China, Macao Special Administrative Region': [99999999, 'MAC', '446', 'MAC', 99999999, 99999999, 99999999],
            ###'Christmas Island': [99999999, 'CXR', '162', 'CXR', 162, 99999999, 99999999],
            ###'Cocos (Keeling) Islands': [99999999, 'CCK', '166', 'CCK', 184, 99999999, 99999999],
            'Colombia': [100, 'COL', '170', 'COL', 170, 45, 15],
            'Comoros': [581, 'COM', '174', 'COM', 174, 46, 153],
            'Congo': [484, 'COG', '178', 'CON', 178, 47, 112],
            ###'Cook Islands': [99999999, 'COK', '184', 'COK', 99999999, 99999999, 99999999],
            'Costa Rica': [94, 'CRI', '188', 'COS', 188, 49, 73],
            'Croatia': [344, 'HRV', '191', 'CRO', 191, 50, 154],
            'Cuba': [40, 'CUB', '192', 'CUB', 192, 51, 155],
            ###'Curaçao': [99999999, 'CUW', '531', 'CUW', 531, 99999999, 99999999],
            'Cyprus': [352, 'CYP', '196', 'CYP', 196, 53, 156],
            'Czechia': [316, 'CZE', '203', 'CZR', 203, 54, 157],
            "Czechoslovakia":[315,'XYZ','XYZ','XYZ',99999999,99999999,99999999],
            'Democratic Peoples Republic of Korea': [731, 'PRK', '408', 'PRK', 408, 149, 41],
            'DR Congo (Zaire)': [490, 'COD', '180', 'DRC', 180, 229, 111],
            'Denmark': [390, 'DNK', '208', 'DEN', 208, 55, 158],
            'Djibouti': [522, 'DJI', '262', 'DJI', 262, 56, 113],
            'Dominica': [54, 'DMA', '212', 'DMA', 212, 57, 99999999],
            'Dominican Republic': [42, 'DOM', '214', 'DRC', 214, 58, 114],
            'East Timor': [860, 'TLS', '626', 'ETM', 626, 347, 74], # <-------- TLS in UN source or TMP
            'Ecuador': [130, 'ECU', '218', 'ECU', 218, 59, 75],
            'Egypt': [651, 'EGY', '818', 'EGY', 818, 60, 13],
            'El Salvador': [92, 'SLV', '222', 'SAL', 222, 61, 22],
            'Equatorial Guinea': [411, 'GNQ', '226', 'EQG', 226, 62, 160],
            'Eritrea': [531, 'ERI', '232', 'ERI', 232, 63, 115],
            'Estonia': [366, 'EST', '233', 'EST', 233, 64, 161],
            'eSwatini': [572, 'SWZ', '748', 'SWA', 748, 197, 132],
            'Ethiopia': [530, 'ETH', '231', 'ETH', 231, 65, 38],
            ###'Falkland Islands (Malvinas)': [99999999, 'FLK', '238', 'FLK', 99999999, 66, 99999999],
            ###'Faroe Islands': [99999999, 'FRO', '234', 'FRO, 234', 99999999, 99999999],
            'Fiji': [950, 'FJI', '242', 'FIJ', 242, 67, 162],
            'Finland': [375, 'FIN', '246', 'FIN', 246, 68, 163],
            'France': [220, 'FRA', '250', 'FRN', 250, 69, 76],
            ###'French Guiana': [99999999, 'GUF', '254', 'GUF', 254, 70, 99999999],
            ###'French Polynesia': [99999999, 'PYF', '258', 'PYF', 258, 71, 99999999],
            ###'French Southern Territories': [99999999, 'ATF', '260', 'ATF', 99999999, 99999999, 99999999],
            'Gabon': [481, 'GAB', '266', 'GAB', 266, 72, 116],
            'Gambia': [420, 'GMB', '270', 'GAM', 270, 73, 117],
            'Georgia': [372, 'GEO', '268', 'GRG', 268, 74, 118],
            'German Democratic Republic': [265,'XYZ','XYZ','XYZ',99999999,99999999,99999999],
            'Germany': [260, 'DEU', '276', 'GMY', 276, 75, 77],
            'Ghana': [452, 'GHA', '288', 'GHA', 288, 76, 7],
            ###'Gibraltar': [99999999, 'GIB', '292', 'GIB', 292, 99999999, 99999999],
            'Greece': [350, 'GRC', '300', 'GRC', 300, 78, 164],
            ###'Greenland': [99999999, 'GRL', '304', 'GRL', 304, 79, 99999999],
            'Grenada': [55, 'GRD', '308', 'GRD', 308, 80, 99999999],
            ####'Guadeloupe': [99999999, 'GLP', '312', 'GLP', 312, 81, 99999999],
            ###'Guam': [99999999, 'GUM', '316', 'GUM', 316, 99999999, 99999999],
            'Guatemala': [90, 'GTM', '320', 'GUA', 320, 83, 78],
            ###'Guernsey': [99999999, 'GGY', '831', 'GGY', 831, 99999999, 99999999],
            'Guinea': [438, 'GIN', '324', 'GUI', 324, 84, 63],
            'Guinea-Bissau': [404, 'GNB', '624', 'GNB', 624, 85, 119],
            'Guyana': [110, 'GUY', '328', 'GUY', 328, 86, 166],
            'Haiti': [41, 'HTI', '332', 'HAI', 332, 87, 26],
            ###'Heard Island and McDonald Islands': [99999999, 'HMD', '334', 'HMD', 99999999, 99999999],
            ###'Holy See': [99999999, 'VAT', '336', 'VAT', 99999999, 99999999],
            'Honduras': [91, 'HND', '340', 'HON', 340, 88, 27],
            'Hungary': [310, 'HUN', '348', 'HUN', 348, 90, 210],
            'Iceland': [395, 'ISL', '352', 'ICE', 352, 91, 168],
            'India': [750, 'IND', '356', 'IND', 356, 92, 39],
            'Indonesia': [850, 'IDN', '360', 'INS', 360, 93, 56],
            'Iran': [630, 'IRN', '364', 'IRN', 364, 94, 79],
            'Iraq': [645, 'IRQ', '368', 'IRQ', 368, 95, 80],
            'Ireland': [205, 'IRL', '372', 'IRE', 372, 96, 81],
            ###'Isle of Man': [99999999, 'IMN', '833', 'IMN', 833, 125, 99999999],
            'Israel': [666, 'ISR', '376', 'ISR', 376, 97, 169],
            'Italy': [325, 'ITA', '380', 'ITA', 380, 98, 82],
            'Ivory Coast': [437, 'CIV', '384', 'CDI', 384, 99, 64],
            'Jamaica': [51, 'JAM', '388', 'JAM', 388, 100, 120],
            'Japan': [740, 'JPN', '392', 'JPN', 392, 101, 9],
            ###'Jersey': [99999999, 'JEY', '832', 'JEY', 832, 99999999, 99999999],
            'Jordan': [663, 'JOR', '400', 'JOR', 400, 102, 83],
            'Kazakhstan': [705, 'KAZ', '398', 'KZK', 398, 103, 121],
            'Kenya': [501, 'KEN', '404', 'KEN', 404, 104, 40],
            'Kiribati': [970, 'KIR', '296', 'KIR', 296, 99999999, 99999999],
            'Kuwait': [690, 'KWT', '414', 'KUW', 414, 106, 171],
            'Kosovo': [347, 'XKX', 'XKX', 'KOS', 0, 1003, 43],
            'Kyrgyzstan': [703, 'KGZ', '417', 'KYR', 417, 107, 122],
            'Laos': [812, 'LAO', '418', 'LAO', 418, 108, 123],
            'Latvia': [367, 'LVA', '428', 'LAT', 428, 109, 84],
            'Lebanon': [660, 'LBN', '422', 'LEB', 422, 110, 44],
            'Lesotho': [570, 'LSO', '426', 'LES', 426, 111, 85],
            'Liberia': [450, 'LBR', '430', 'LBR', 430, 112, 86],
            'Libya': [620, 'LBY', '434', 'LIB', 434, 113, 124],
            'Liechtenstein': [223, 'LIE', '438', 'LIE', 438, 114, 99999999],
            'Lithuania': [368, 'LTU', '440', 'LIT', 440, 115, 173],
            'Luxembourg': [212, 'LUX', '442', 'LUX', 442, 116, 174],
            'Madagascar': [580, 'MDG', '450', 'MAG', 450, 119, 125],
            'Malawi': [553, 'MWI', '454', 'MAW', 454, 120, 87],
            'Malaysia': [820, 'MYS', '458', 'MAL', 458, 121, 177],
            'Maldives': [781, 'MDV', '462', 'MAD', 462, 122, 88],
            'Mali': [432, 'MLI', '466', 'MLI', 466, 123, 28],
            'Malta': [338, 'MLT', '470', 'MLT', 470, 124, 178],
            'Marshall Islands': [983, 'MHL', '584', 'MSI', 584, 126, 99999999],
            ####'Martinique': [99999999, 'MTQ', '474', 'MTQ', 474, 127, 99999999],
            'Mauritania': [435, 'MRT', '478', 'MAA', 478, 128, 65],                  
            'Mauritius': [590, 'MUS', '480', 'MAS', 480, 129, 180],   
            ###'Mayotte': [99999999, 'MYT', '175', 'MYT', 175, 99999999, 99999999], 
            'Mexico': [70, 'MEX', '484', 'MEX', 484, 130, 3],        
            'Micronesia (Federated States of)': [987, 'FSM', '583', 'FSM', 583, 99999999, 99999999],        
            'Monaco': [221, 'MCO', '492', 'MNC', 492, 99999999, 182], 
            'Mongolia': [712, 'MNG', '496', 'MON', 496, 134, 89],
            'Montenegro': [341, 'MNE', '499', 'MNG', 499, 1002, 183],
            ###'Montserrat': [99999999, 'MSR', '500', 'MSR', 500, 99999999, 99999999],
            'Morocco': [600, 'MAR', '504', 'MOR', 504, 136, 90],
            'Mozambique': [541, 'MOZ', '508', 'MZM', 508, 137, 57],
            'Myanmar (Burma)': [775, 'MMR', '104', 'MYA', 104, 138, 10],
            'Namibia': [565, 'NAM', '516', 'NAM', 516, 139, 127],
            'Nauru': [971, 'NRU', '520', 'NAU', 520, 99999999, 99999999],
            'Nepal': [790, 'NPL', '524', 'NEP', 524, 141, 58],
            'Netherlands': [210, 'NLD', '528', 'NTH', 528, 142, 91],
            ###'New Caledonia': [99999999, 'NCL', '540', 'NCL', 540, 143, 99999999],
            'New Zealand': [920, 'NZL', '554', 'NEW', 554, 144, 185],
            'Nicaragua': [93, 'NIC', '558', 'NIC', 558, 145, 59], 
            'Niger': [436, 'NER', '562', 'NIR', 562, 146, 60],
            'Nigeria': [475, 'NGA', '566', 'NIG', 566, 147, 45],
            ###'Niue': [99999999, 'NIU', '570', 'NIU', 99999999, 99999999, 99999999],
            ###'Norfolk Island': [99999999, 'NFK', '574', 'NFK', 99999999, 99999999, 99999999],
            'North Macedonia': [343, 'MKD', '807', 'MAC', 807, 118, 176],
            ###'Northern Mariana Islands': [99999999, 'MNP', '580', 'MNP', 580, 99999999, 99999999],
            'Norway': [385, 'NOR', '578', 'NOR', 578, 151, 186],
            'Oman': [698, 'OMN', '512', 'OMA', 512, 152, 187],
            'Pakistan': [770, 'PAK', '586', 'PAK', 586, 153, 29],
            'Palau': [986, 'PLW', '585', 'PAL', 585, 99999999, 99999999], 
            'Panama': [95, 'PAN', '591', 'PAN', 591, 156, 92], 
            'Papua New Guinea': [910, 'PNG', '598', 'PNG', 598, 157, 93],
            'Paraguay': [150, 'PRY', '600', 'PAR', 600, 158, 189],
            'Peru': [135, 'PER', '604', 'PER', 604, 159, 30],
            'Philippines': [840, 'PHL', '608', 'PHI', 608, 160, 46],
            ###'Pitcairn': [99999999, 'PCN', '612', 'PCN', 99999999, 99999999, 99999999],
            'Poland': [290, 'POL', '616', 'POL', 616, 161, 17],
            'Portugal': [235, 'PRT', '620', 'POR', 620, 162, 21],
            ###'Puerto Rico': [99999999, 'PRI', '630', 'PRI', 630, 163, 99999999],
            'Qatar': [694, 'QAT', '634', 'QAT', 634, 164, 94],
            'Republic of Moldova': [359, 'MDA', '498', 'MLD', 498, 132, 126],
            ###'Réunion': [99999999, 'REU', '638', 'REU', 638, 99999999, 99999999],
            'Romania': [360, 'ROU', '642', 'ROM', 642, 166, 190],
            'Russia': [365, 'RUS', '643', 'RUS', 643, 167, 11],
            'Rwanda': [517, 'RWA', '646', 'RWA', 646, 168, 129],
            ###'Saint Barthélemy': [99999999, 'BLM', '652', 'BLM', 652, 99999999, 99999999],
            ###'Saint Helena': [99999999, 'SHN', '654', 'SHN', 654, 99999999, 99999999],
            'Saint Kitts and Nevis': [60, 'KNA', '659', 'SKN', 659, 189, 99999999],
            'Saint Lucia': [56, 'LCA', '662', 'SLU', 662, 190, 99999999],
            ###'Saint Martin (French Part)': [99999999, 'MAF', '663', 'MAF', 663, 192, 99999999],
            ###'Saint Pierre and Miquelon': [99999999, 'SPM', '666', 'SPM', 666, 99999999, 99999999],
            'Saint Vincent and the Grenadines': [57, 'VCT', '670', 'SVG', 670, 99999999, 99999999],
            'Samoa': [990, 'WSM', '882', 'WSM', 882, 99999999, 99999999],
            'San Marino': [331, 'SMR', '674', 'SNM', 674, 99999999, 195],
            'Sao Tome and Principe': [403, 'STP', '678', 'STP', 678, 99999999, 196],
            ###'Sark': [99999999, 'XKX', '680', 'XKX', 99999999, 99999999, 99999999],   
            'Saudi Arabia': [670, 'SAU', '682', 'SAU', 682, 173, 197],
            'Senegal': [433, 'SEN', '686', 'SEN', 686, 174, 31],
            'Serbia': [340, 'SRB', '688', 'SRB', 688, 1001, 198],
            'Seychelles': [591, 'SYC', '690', 'SEY', 690, 176, 199],
            'Sierra Leone': [451, 'SLE', '694', 'SIE', 694, 177, 95],
            'Singapore': [830, 'SGP', '702', 'SIN', 702, 178, 200],
            ###'Sint Maarten (Dutch part)': [99999999, 'SXM', '534', 'SXM', 534, 99999999, 99999999],
            'Slovakia': [317, 'SVK', '703', 'SLO', 703, 179, 201],
            'Slovenia': [349, 'SVN', '705', 'SLV', 705, 180, 202],
            'Solomon Islands': [940, 'SLB', '090', 'SOL', 90, 181, 203],
            'Somalia': [520, 'SOM', '706', 'SOM', 706, 182, 130],
            'South Africa': [560, 'ZAF', '710', 'SAF', 710, 183, 8],
            ###'South Georgia and the South Sandwich Islands': [99999999, 'SGS', '239', 'SGS', 99999999, 99999999, 99999999],
            'South Korea': [732, 'KOR', '410', 'ROK', 410, 184, 42],
            'South Sudan': [626, 'SSD', '728', 'SSD', 728, 1004, 32], 
            'South Ossetia': [397, 'XYZ', 'XYZ', 'XZY', 99999999,99999999,99999999], 
            'Spain': [230, 'ESP', '724', 'SPN', 724, 185, 96], 
            'Sri Lanka': [780, 'LKA', '144', 'SRI', 144, 186, 131],
            ###'State of Palestine': [99999999, 'PSE', '275', 'PSE', 275, 99999999, 128],
            'Sudan': [625, 'SDN', '729', 'SUD', 729, 195, 33],
            'Suriname': [115, 'SUR', '740', 'SUR', 740, 196, 4],
            ###'Svalbard and Jan Mayen Islands': [99999999, 'SJM', '744', 'SJM', 99999999, 99999999, 99999999],
            'Sweden': [380, 'SWE', '752', 'SWD', 752, 198, 5],
            'Switzerland': [225, 'CHE', '756', 'SWZ', 756, 199, 6],
            'Syria': [652, 'SYR', '760', 'SYR', 760, 200, 97],
            'Tajikistan': [702, 'TJK', '762', 'TAJ', 762, 202, 133],
            'Tanzania': [510, 'TZA', '834', 'TAZ', 834, 203, 47],
            'Thailand': [800, 'THA', '764', 'THI', 764, 205, 49],
            'Taiwan': [713, 'TWN', '158', 'TAW', 158, 201, 48],
            'Togo': [461, 'TGO', '768', 'TOG', 768, 204],
            ###'Tokelau': [99999999, 'TKL', '772', 'TKL', 99999999, 99999999, 99999999],
            'Tonga': [972, 'TON', '776', 'TON', 776, 206, 134],
            'Trinidad and Tobago': [52, 'TTO', '780', 'TRI', 780, 207, 135],
            'Tunisia': [616, 'TUN', '788', 'TUN', 788, 208, 98],
            'Turkey': [640, 'TUR', '792', 'TUR', 792, 209, 99],
            'Turkmenistan': [701, 'TKM', '795', 'TKM', 795, 210, 136],
            ###'Turks and Caicos Islands': [99999999, 'TCA', '796', 'TCA', 796, 99999999, 99999999],
            'Tuvalu': [973, 'TUV', '798', 'TUV', 798, 212, 99999999],
            'Uganda': [500, 'UGA', '800', 'UGA', 800, 213, 50],
            'Ukraine': [369, 'UKR', '804', 'UKR', 804, 214, 100],
            'United Arab Emirates': [696, 'ARE', '784', 'UAE', 784, 215, 207],
            'United Kingdom': [200, 'GBR', '826', 'UKG', 826, 603, 101],
            ###'United States Minor Outlying Islands': [99999999, 'UMI', '581', 'UMI', 99999999, 99999999, 99999999],
            'United States': [2, 'USA', '840', 'USA', 840, 217, 20],
            ###'United States Virgin Islands': [99999999, 'VIR', '850', 'VIR', 850, 99999999, 99999999],
            'Uruguay': [165, 'URY', '858', 'URU', 858, 218, 102],
            'Uzbekistan': [704, 'UZB', '860', 'UZB', 860, 219, 140],
            'Vanuatu': [935, 'VUT', '548', 'VAN', 548, 220, 206],
            'Venezuela': [101, 'VEN', '862', 'VEN', 862, 220, 51],
            'Vietnam': [816, 'VNM', '704', 'DRV', 704, 223, 34],
            ###'Wallis and Futuna Islands': [99999999, 'WLF', '876', 'WLF', 876, 99999999, 99999999],
            ###'Western Sahara': [99999999, 'ESH', '732', 'ESH', 99999999, 99999999, 99999999],
            'Yemen (North Yemen)': [678, 'YEM', '887', 'YEM', 887, 228, 14],
            "Yemen, People's Republic of": [680, 'XYZ', 'XYZ', 'XYZ', 99999999, 99999999, 99999999],
            'Yugoslavia': [345, 'XYZ', 'XYZ', 'XYZ', 99999999, 99999999, 99999999],
            'Zambia': [551, 'ZMB', '894', 'ZAM', 894, 230, 61],
            'Zimbabwe': [552, 'ZWE', '716', 'ZIM', 716, 231, 62],
            }

### Convert dictionary with country codes to df ----
df_ccodes = pd.DataFrame.from_dict(c_codes,orient='index')
df_ccodes = df_ccodes.reset_index()
df_ccodes.columns = ['country','gw_codes','iso_alpha3','M49','StateAbb','acled_codes','gtd_codes',"vdem_codes"]

### Save data ---------
df_ccodes.to_csv("df_ccodes.csv",index=False,sep=',')
print("Saved DataFrame!")

# GW country codes
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
all_countries=all_countries.reset_index()
all_countries=all_countries[["country","gw_codes","start","end"]]

### Save data ---------
all_countries.to_csv("df_ccodes_gw.csv",index=False,sep=',')
print("Saved DataFrame!")






