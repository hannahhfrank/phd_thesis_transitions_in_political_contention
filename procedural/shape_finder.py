import pandas as pd
import numpy as np
from dtaidistance import dtw,ed
import bisect
import pickle
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt 

################
### Get data ###
################

# List of microstates: 
micro_states={"Dominica":54,
              "Grenada":55,
              "Saint Lucia":56,
              "Saint Vincent and the Grenadines":57,
              "Antigua & Barbuda":58,
              "Saint Kitts and Nevis":60,
              "Monaco":221,
              "Liechtenstein":223,
              "San Marino":331,
              "Andorra":232,
              "Abkhazia":396,
              "South Ossetia":397,
              "São Tomé and Principe":403,
              "Seychelles":591,
              "Vanuatu":935,
              "Kiribati":970,
              "Nauru":971,
              "Tonga":972,
              "Tuvalu":973,
              "Marshall Islands":983,
              "Palau":986,
              "Micronesia":987,
              "Samoa":990}

# Additional countries not included in ACLED: 
# 265 German Democratic Republic	
# 315 Czechoslovakia
# 345 Yugoslavia
# 396 Abkhazia
# 397 South Ossetia
# 680 Yemen, People's Republic of ---> EXCLUDE

# Temporal coverage: 1989--2022

exclude={"German Democratic Republic":265,
         "Czechoslovakia":315,
         "Yugoslavia":345,
         "Abkhazia":396,
         "South Ossetia":397,
         "Yemen, People's Republic of":680}

# Countries not included in World Bank: 
# Or have mostly missing values    
exclude2 ={"Taiwan":713, # Not included in WDI
           "Bahamas":31, # Not included in vdem
           "Belize":80, # Not included in vdem
           "Brunei Darussalam":835, # Not included in vdem
           "Kosovo":347, # Mostly missing in WDI
           "Democratic Peoples Republic of Korea":731} # Mostly missing in WDI

df=pd.read_csv("data/data_out/ucdp_cm_sb.csv",index_col=0)
df = df[~df['gw_codes'].isin(list(micro_states.values())+list(exclude.values())+list(exclude2.values()))]
df=df.sort_values(by=["country","year","dd"])


### Shape finder simplified ###

def shape_finder_simple(df,min_d=0.1,dtw_sel=2,win=10,horizon=12,h_train=10,outcome="best",input_year=2021):
    dict_m={i :[] for i in df.country.unique()} 
    pred_tot_pr=[]
        
    for c in df.country.unique():
        # Get input shape
        df_s=df[["year","dd","best"]].loc[df["country"]==c]
        input_shape = df_s.loc[df_s["year"]==input_year][2:]
        input_shape = input_shape.set_index('dd').drop('year', axis=1)
        input_shape = input_shape["best"]
        input_shape.name = c
        input_shape_copy=input_shape
        
        # If input shape is not flat, otherwise predict 0
        if not (input_shape==0).all():
        
            # Get subsequences and normalize
            df_input_sub=df.loc[df["year"]<=input_year]
            df_input_sub = df_input_sub.pivot(index='dd', columns='country', values='best')
            df_input_sub = df_input_sub.fillna(0)
            seq = []
            for i in range(len(df_input_sub.columns)): 
                seq.append(df_input_sub.iloc[:, i]) 
            seq_n = []
            for i in seq:
                seq_n.append((i - i.mean()) / i.std())
                    
            # Min-max normalize input shape
            if input_shape.var() != 0.0:
                input_shape = (input_shape - input_shape.min()) / (input_shape.max() - input_shape.min())
            else : # if input is plat, set to 0.5
                input_shape= [0.5]*len(input_shape)
                input_shape = np.array(input_shape)
                
            # Get a df with all distances between input and all references
            tot = []
            for lop in range(int(-dtw_sel), int(dtw_sel) + 1): 
                n_test = [] 
                to = 0  
                exclude = []  
                interv = [0]  
                for i in seq_n:
                    n_test = np.concatenate([n_test, i])  
                    to = to + len(i) 
                    exclude = exclude + [*range(to - (win+lop), to)]  
                    interv.append(to)  
                for i in range(len(n_test)):
                    if i not in exclude:
                        seq2 = n_test[i:i + int(10 + lop)]
                        if seq2.var() != 0.0:
                            seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                        else:
                            seq2 = np.array([0.5]*len(seq2))
                        try:
                            dist = dtw.distance(input_shape.values, seq2, use_c=True)
                            tot.append([i, dist, 10 + lop])
                        except:
                            pass
            # Stores indes, distance, and window length
            tot = pd.DataFrame(tot)
                
            # Get filtered repository         
            min_d_d=min_d
            sequences=[]
            while len(sequences)<5:
                matches=[]    
                tot_more = tot.sort_values([1])
                tot_more = tot_more[tot_more[1] < min_d_d]
                toti = tot_more[0]
                n = len(toti)
                diff_data = {f'diff{i}': toti.diff(i) for i in range(1, n + 1)}
                diff_df = pd.DataFrame(diff_data).fillna(10)
                diff_df = abs(diff_df)
                tot_more = tot_more[diff_df.min(axis=1) >= (10/ 2)]
                
                if len(tot_more) > 0:
                    for c_lo in range(len(tot_more)):
                        i = tot_more.iloc[c_lo, 0]
                        win_l = int(tot_more.iloc[c_lo, 2])
                        
                        n_test = [] 
                        to = 0  
                        exclude = []  
                        interv = [0]  
                        for x in seq_n:
                            n_test = np.concatenate([n_test, x])  
                            to = to + len(x)  
                            exclude = exclude + [*range(to - (win+lop), to)]  
                            interv.append(to)  
                            
                        col = seq[bisect.bisect_right(interv, i) - 1].name
                        index_obs = seq[bisect.bisect_right(interv, i) - 1].index[i - interv[bisect.bisect_right(interv, i) - 1]]
                        obs = df_input_sub.loc[index_obs:, col].iloc[:win_l]
                        matches.append([obs, tot_more.iloc[c_lo, 1]])
                        sequences=matches
                min_d_d += 0.05
    
            dict_m[c]=sequences
                           
            ########################
            ### Make predictions ###
            ########################
            
            df_input = df.pivot(index='dd', columns='country', values=outcome)
            df_input = df_input.fillna(0)
            
            # Extract references
            l_find=dict_m[c]
            tot_seq = [[series.name, series.index[-1], series.min(),series.max()] for series, weight in l_find]
            
            pred_seq=[]
            co=[]
            deca=[]
            scale=[]
            # For each reference, get index of last observed value
            for col,last_date,mi,ma in tot_seq:
                date=df_input.loc[:f"{input_year}-12"].index.get_loc(last_date)
                # If future of reference does not lie in testing window, add min-max normalized future
                if date+horizon<len(df_input.loc[:f"{input_year}-12"]):
                    seq=df_input.loc[:f"{input_year}-12"].iloc[date+1:date+1+horizon,df_input.loc[:f"{input_year}-12"].columns.get_loc(col)].reset_index(drop=True)
                    seq = (seq - mi) / (ma - mi)
                    pred_seq.append(seq.tolist())
                
            # Apply clustering allgorithm to futures
            tot_seq=pd.DataFrame(pred_seq)
            linkage_matrix = linkage(tot_seq, method='ward')
            clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
            tot_seq['Cluster'] = clusters
            
            # Get centroid for each cluster
            val_sce = tot_seq.groupby('Cluster').mean()
            # Calculate how many observations are in cluster and select majority cluster
            pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
            pred_ori=val_sce.loc[pr==pr.max(),:]
            pred_ori=pred_ori.mean(axis=0)
            # Reverse min-max normalization and append to predictions
            preds=pred_ori*(input_shape_copy.max()-input_shape_copy.min())+input_shape_copy.min()
            pred_tot_pr.append(preds)
              
        else:
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
                        
    # Save        
    preds = pd.concat(pred_tot_pr,axis=1)
    preds.columns=df.country.unique()
    preds.index=[f"{input_year+1}-01",f"{input_year+1}-02",f"{input_year+1}-03",f"{input_year+1}-04",f"{input_year+1}-05",f"{input_year+1}-06",f"{input_year+1}-07",f"{input_year+1}-08",f"{input_year+1}-09",f"{input_year+1}-10",f"{input_year+1}-11",f"{input_year+1}-12"]
    preds.index.name = 'dd'
    preds_out = preds.reset_index().melt(id_vars='dd', var_name='country', value_name='best')
    preds_out=preds_out.rename(columns={'best': 'preds'})
    #preds_out.to_csv(f'out/preds{input_year+1}.csv')  

    return preds_out     

preds_2022=shape_finder_simple(df)
preds_2023=shape_finder_simple(df,input_year=2022)

shape_finder = pd.concat([preds_2022, preds_2023], axis=0, ignore_index=True)
shape_finder=shape_finder.sort_values(by=["country","dd"])
shape_finder=shape_finder.reset_index(drop=True)
shape_finder["year"] = shape_finder["dd"].str[:4] 
shape_finder["year"]=shape_finder["year"].astype(int)

# Merge
shape_finder=pd.merge(shape_finder,df[["country","dd","gw_codes","best"]],on=["dd","country"],how="left")
shape_finder=shape_finder[["year","dd","country","gw_codes","preds","best"]]
shape_finder=shape_finder.rename(columns={'best': 'sb_fatalities'})
shape_finder.to_csv('out/shape_finder.csv') 


































###################
### Thomas code ###
###################


def int_exc(seq_n, win):
    """
    Create intervals and exclude list for the given normalized sequences.

    Args:
        seq_n (list): A list of normalized sequences.
        win (int): The window size for pattern matching.

    Returns:
        tuple: A tuple containing the exclude list, intervals, and the concatenated testing sequence.
    """
    n_test = []  # List to store the concatenated testing sequence
    to = 0  # Variable to keep track of the total length of concatenated sequences
    exclude = []  # List to store the excluded indices
    interv = [0]  # List to store the intervals

    for i in seq_n:
        n_test = np.concatenate([n_test, i])  # Concatenate each normalized sequence to create the testing sequence
        to = to + len(i)  # Calculate the total length of the concatenated sequence
        exclude = exclude + [*range(to - win, to)]  # Add the excluded indices to the list
        interv.append(to)  # Add the interval (end index) for each sequence to the list

    # Return the exclude list, intervals, and the concatenated testing sequence as a tuple
    return exclude, interv, n_test

class Shape():
    """
    A class to set custom shape using a graphical interface, user-provided values or random values.

    Attributes:
        time (list): List of x-coordinates representing time.
        values (list): List of y-coordinates representing values.
        window (int): The window size for the graphical interface.
    """

    def __init__(self, time=len(range(10)), values=[0.5]*10, window=10):
        """
        Args:
            time (int): The initial number of time points.
            values (list): The initial values corresponding to each time point.
            window (int): The window size for the graphical interface.
        """
        self.time = time
        self.values = values
        self.window = window

    def set_shape(self,input_shape):
        try:
            input_shape=pd.Series(input_shape)
            if input_shape.var() != 0.0 :
                input_shape=(input_shape-input_shape.min())/(input_shape.max()-input_shape.min())
            else:
                input_shape = np.array([0.5]*len(input_shape))
            self.time=list(range(len(input_shape)))
            self.values = input_shape.tolist()
            self.window=len(input_shape.tolist())
        except: 
            print('Wrong format, please provide a compatible input.')
        
        
class finder():
    """
    A class to find and predict custom patterns in a given dataset using an interactive shape finder.

    Attributes:
        data (DataFrame): The dataset containing time series data.
        Shape (Shape): An instance of the Shape class used for interactive shape finding.
        sequences (list): List to store the found sequences matching the custom shape.
    """
    def __init__(self,data,Shape=Shape(),sequences=[],sce=None,val_sce=None):
        """
        Initializes the finder object with the given dataset and Shape instance.

        Args:
            data (DataFrame): The dataset containing time series data.
            Shape (Shape, optional): An instance of the Shape class for shape finding. Defaults to Shape().
            sequences (list, optional): List to store the found sequences matching the custom shape. Defaults to [].
        """
        self.data=data
        self.Shape=Shape
        self.sequences=sequences
        self.sce = sce
        self.val_sce = val_sce
        
    def find_patterns(self, metric='euclidean', min_d=0.5, dtw_sel=0, select=True):
        """
        Finds custom patterns in the given dataset using the interactive shape finder.
    
        Args:
            metric (str, optional): The distance metric to use for shape matching. 'euclidean' or 'dtw'. Defaults to 'euclidean'.
            min_d (float, optional): The minimum distance threshold for a matching sequence. Defaults to 0.5.
            dtw_sel (int, optional): The window size variation for dynamic time warping (Only for 'dtw' mode). Defaults to 0.
            select (bool, optional): Whether to include overlapping patterns. Defaults to True.
        """
        # Clear any previously stored sequences
        self.sequences = []
        
        # Check if dtw_sel is zero when metric is 'euclidean'
        if metric=='euclidean':
            dtw_sel=0
    
        # Extract individual columns (time series) from the data
        seq = []
        for i in range(len(self.data.columns)): 
            seq.append(self.data.iloc[:, i])
    
        # Normalize each column (time series)
        seq_n = []
        for i in seq:
            seq_n.append((i - i.mean()) / i.std())
    
        # Get exclude list, intervals, and a testing sequence for pattern matching
        exclude, interv, n_test = int_exc(seq, self.Shape.window)
    
        # Convert custom shape values to a pandas Series and normalize it
        seq1 = pd.Series(data=self.Shape.values)
        if seq1.var() != 0.0:
            seq1 = (seq1 - seq1.min()) / (seq1.max() - seq1.min())
        else :    
            seq1 = [0.5]*len(seq1)
        seq1 = np.array(seq1)
    
        # Initialize the list to store the found sequences that match the custom shape
        tot = []
    
        if dtw_sel == 0:
            # Loop through the testing sequence
            for i in range(len(n_test)):
                # Check if the current index is not in the exclude list
                if i not in exclude:
                    seq2 = n_test[i:i + self.Shape.window]
                    if seq2.var() != 0.0:
                        seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                    else:
                        seq2 = np.array([0.5]*len(seq2))
                    try:
                        if metric == 'euclidean':
                            # Calculate the Euclidean distance between the custom shape and the current window
                            dist = ed.distance(seq1, seq2)
                        elif metric == 'dtw':
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2, use_c=True)
                        tot.append([i, dist, self.Shape.window])
                    except:
                        # Ignore any exceptions (e.g., divide by zero)
                        pass
        else:
            # Loop through the range of window size variations (dtw_sel)
            for lop in range(int(-dtw_sel), int(dtw_sel) + 1):
                # Get exclude list, intervals, and a testing sequence for pattern matching with the current window size
                exclude, interv, n_test = int_exc(seq_n, self.Shape.window + lop)
                for i in range(len(n_test)):
                    # Check if the current index is not in the exclude list
                    if i not in exclude:
                        seq2 = n_test[i:i + int(self.Shape.window + lop)]
                        if seq2.var() != 0.0:
                            seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                        else:
                            seq2 = np.array([0.5]*len(seq2))
                        try:
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2, use_c=True)
                            tot.append([i, dist, self.Shape.window + lop])
                        except:
                            # Ignore any exceptions (e.g., divide by zero)
                            pass
    
        # Create a DataFrame from the list of sequences and distances, sort it by distance, and filter based on min_d
        tot = pd.DataFrame(tot)
        tot = tot.sort_values([1])
        tot = tot[tot[1] < min_d]
        toti = tot[0]
    
        if select:
            n = len(toti)
            diff_data = {f'diff{i}': toti.diff(i) for i in range(1, n + 1)}
            diff_df = pd.DataFrame(diff_data).fillna(self.Shape.window)
            diff_df = abs(diff_df)
            tot = tot[diff_df.min(axis=1) >= (self.Shape.window / 2)]
    
        if len(tot) > 0:
            # If there are selected patterns, store them along with their distances in the 'sequences' list
            for c_lo in range(len(tot)):
                i = tot.iloc[c_lo, 0]
                win_l = int(tot.iloc[c_lo, 2])
                exclude, interv, n_test = int_exc(seq_n, win_l)
                col = seq[bisect.bisect_right(interv, i) - 1].name
                index_obs = seq[bisect.bisect_right(interv, i) - 1].index[i - interv[bisect.bisect_right(interv, i) - 1]]
                obs = self.data.loc[index_obs:, col].iloc[:win_l]
                self.sequences.append([obs, tot.iloc[c_lo, 1]])
        else:
            print('No patterns found')


    
############
### 2022 ###
############

df_input = df.pivot(index='dd', columns='country', values='best')
df_input = df_input.fillna(0)

h_train=10
dict_m={i :[] for i in df_input.columns} # until 2020
# Remove last two years in data to get training data
df_input_sub=df_input.iloc[:-24]
# For each country in df
for coun in range(len(df_input_sub.columns)):
    # If the last h_train observations of training data are not flat, run Shape finder, else pass    
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        shape = Shape()
        # Set last h_train observations of training data as shape
        shape.set_shape(df_input_sub.iloc[-h_train:,coun]) 
        # Find matches in training data        
        find = finder(df_input.iloc[:-24],shape)
        find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2)
        min_d_d=0.1
        # If there are fewer than 5 observations in reference, increase max distance until 5 observations are matched
        while len(find.sequences)<5:
            min_d_d += 0.05
            find.find_patterns(min_d=min_d_d,select=True,metric='dtw',dtw_sel=2)
        # Save matches and distances                
        dict_m[df_input.columns[coun]]=find.sequences
    else :
        pass
    
# For saving        
with open('out/shape2022_thoma.pkl', 'wb') as f:
     pickle.dump(dict_m, f) 
       
     
with open('out/shape2022_thoma.pkl', 'rb') as f:
   dict_m = pickle.load(f)
    
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
df_input_sub=df_input.iloc[:-24]
cluster_dist=[]

for coun in range(len(df_input_sub.columns)):
    # If the last h_train observations of training data are not flat
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        # For each case, get region, year and magnitude as the log(total fatalities)

        # Get reference reporitory for case                
        l_find=dict_m[df_input.columns[coun]]
        # For each case in repository, get country name, last time point               
        tot_seq = [[series.name, series.index[-1], series.min(),series.max()] for series, weight in l_find]
        
        # For each case in reference repository                       
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma in tot_seq:
            # Get views id for last month            
            date=df_input.iloc[:-24].index.get_loc(last_date)
            # If reference + horizon is in training data                        
            if date+horizon<len(df_input.iloc[:-24]):
                # Extract sequence for reference, for the next 12 months                                
                seq=df_input.iloc[:-24].iloc[date+1:date+1+horizon,df_input.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())

                
        # Sequences for all case in the reference repository, if they belong to training data,
        # every row is one sequence                 
        tot_seq=pd.DataFrame(pred_seq)
        
        ### Apply hierachical clustering to sequences in reference repository ###        
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        
        # Calculate mean sequence for each cluster        
        val_sce = tot_seq.groupby('Cluster').mean()
        
        # Proportions for each cluster        
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        #cluster_dist.append(pr.max())
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        
        # A. Get mean sequence with lowest intensity
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        # Adjust by range (*max-min) and add min value
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
       
        # B. Get mean sequence for cluster with highest number of observations                
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        # Adjust by range (*max-min) and add min value
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        # Append predictions
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        
    else:
        # Add zeros
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
        
# Save        
df_sf_1 = pd.concat(pred_tot_pr,axis=1)
df_sf_1.columns=df_input.columns
df_sf_1=df_sf_1.set_index(df_input.iloc[-24:-12].index)
df_sf_1_out = df_sf_1.reset_index().melt(id_vars='dd', var_name='country', value_name='best')
df_sf_1_out.to_csv('out/shape2022_thoma.csv')  
    
     
############
### 2023 ###
############   
     
df_input = df.pivot(index='dd', columns='country', values='best')
df_input = df_input.fillna(0)

h_train=10
dict_m={i :[] for i in df_input.columns} # until 2020
# Remove last two years in data to get training data
df_input_sub=df_input.iloc[:-12]
# For each country in df
for coun in range(len(df_input_sub.columns)):
    # If the last h_train observations of training data are not flat, run Shape finder, else pass    
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        shape = Shape()
        # Set last h_train observations of training data as shape
        shape.set_shape(df_input_sub.iloc[-h_train:,coun]) 
        # Find matches in training data        
        find = finder(df_input.iloc[:-12],shape)
        find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2)
        min_d_d=0.1
        # If there are fewer than 5 observations in reference, increase max distance until 5 observations are matched
        while len(find.sequences)<5:
            min_d_d += 0.05
            find.find_patterns(min_d=min_d_d,select=True,metric='dtw',dtw_sel=2)
        # Save matches and distances                
        dict_m[df_input.columns[coun]]=find.sequences
    else :
        pass
    
# For saving        
with open('out/shape2023_thoma.pkl', 'wb') as f:
     pickle.dump(dict_m, f) 
       
     
with open('out/shape2023_thoma.pkl', 'rb') as f:
   dict_m = pickle.load(f)
    
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
df_input_sub=df_input.iloc[:-12]
cluster_dist=[]


for coun in range(len(df_input_sub.columns)):
    # If the last h_train observations of training data are not flat
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        # For each case, get region, year and magnitude as the log(total fatalities)

        # Get reference reporitory for case                
        l_find=dict_m[df_input.columns[coun]]
        # For each case in repository, get country name, last time point               
        tot_seq = [[series.name, series.index[-1], series.min(),series.max()] for series, weight in l_find]
        
        # For each case in reference repository                       
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma in tot_seq:
            # Get views id for last month            
            date=df_input.iloc[:-12].index.get_loc(last_date)
            # If reference + horizon is in training data                        
            if date+horizon<len(df_input.iloc[:-12]):
                # Extract sequence for reference, for the next 12 months                                
                seq=df_input.iloc[:-12].iloc[date+1:date+1+horizon,df_input.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())

                
        # Sequences for all case in the reference repository, if they belong to training data,
        # every row is one sequence                 
        tot_seq=pd.DataFrame(pred_seq)
        
        ### Apply hierachical clustering to sequences in reference repository ###        
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        
        # Calculate mean sequence for each cluster        
        val_sce = tot_seq.groupby('Cluster').mean()
        
        # Proportions for each cluster        
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        #cluster_dist.append(pr.max())
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        
        # A. Get mean sequence with lowest intensity
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        # Adjust by range (*max-min) and add min value
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
       
        # B. Get mean sequence for cluster with highest number of observations                
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        # Adjust by range (*max-min) and add min value
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        # Append predictions
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        
    else:
        # Add zeros
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
        
# Save        
df_sf_2 = pd.concat(pred_tot_pr,axis=1)
df_sf_2.columns=df_input.columns
df_sf_2=df_sf_2.set_index(df_input.iloc[-12:].index)
df_sf_2_out = df_sf_2.reset_index().melt(id_vars='dd', var_name='country', value_name='best')
df_sf_2_out.to_csv('out/shape2023_thoma.csv')  
         
     
     
 
     
     