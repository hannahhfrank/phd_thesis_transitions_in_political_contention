import pandas as pd
import numpy as np
from functions import earth_mover_distance,preprocess_min_max_group
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
from PyALE import ale
import lime
import seaborn as sns
from sklearn.dummy import DummyRegressor,DummyClassifier
import geopandas as gpd
import matplotlib as mpl
from matplotlib import cm 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import os 
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
from sklearn.svm import SVR
import random
random.seed(42)
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib as mpl
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'
from alibi.explainers import CounterfactualProto
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from alibi.explainers import AnchorTabular
from alibi.explainers import KernelShap,TreeShap
from scipy.cluster.hierarchy import linkage, fcluster
from io import BytesIO
from matplotlib.image import imread
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.tree import plot_tree

random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)],
               'max_features': ["sqrt", "log2", None],
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],
               'min_samples_split': [2,5,10],
               'min_samples_leaf': [1,2,4]
               }

df=pd.read_csv("data/df_interpret.csv",index_col=0)
df["country"] = df["country"].str.replace(r"\s*\(.*?\)", "", regex=True)  # Remove everything inside ()

#for var in ['sb_fatalities','onset','sb_fatalities_lag1','d_civil_conflict_zeros_decay','d_neighbors_sb_fatalities_lag1','gdp','growth','pop','oil','percip','temp','ethnic_frac','powerless_share','eys_male','libdem','libdem_change','rugged']:
#    fig,ax = plt.subplots()
#    df[var].hist()
#    plt.title(var)

# Transforms
df["sb_fatalities_log"]=np.log(df["sb_fatalities"]+1)
df["d_neighbors_sb_fatalities_lag1_log"]=np.log(df["d_neighbors_sb_fatalities_lag1"]+1)
df["sb_fatalities_lag1_log"]=np.log(df["sb_fatalities_lag1"]+1)

df["gdp"]=np.log(df["gdp"])
df["pop"]=np.log(df["pop"])

df.columns=['year', 
            'gw_codes', 
            'country', 
            'sb_fatalities', 
            'side_b', 
            'onset',
            'onset2',
            "sb_fatalities_lag1",
            "d_civil_conflict_zeros_decay",
            'neighbors_sb_fatalities_lag1', 
            'GDP', 
            'Growth', 
            'Population', 
            'Oil rents',
            'Percipitation', 
            'Male education',   
            "Regulatory quality",
            'Temperature', 
            'Ethnic fractionalization', 
            'Powerless, share', 
            'Liberal democracy', 
            'Electoral democracy', 
            "Political stability" ,
            "Polity2",
            'Terrain (rugg)', 
            "Terrain",
            'sb_fatalities_log',
            "Fatalities in neighborhood",
            "t-1 fatalities"]

target='onset2'
inputs=['Fatalities in neighborhood', 
        #"t-1 fatalities",
        'GDP', 
        'Growth', 
        'Population', 
        'Oil rents',
        #'Percipitation', 
        'Temperature', 
        #'Ethnic fractionalization', 
        'Powerless, share', 
        'Male education',
        'Liberal democracy',
         "Regulatory quality",
        #'Terrain'
        #"Terrain",
        ]
y=df[["year",'country','onset2']]

x=df[["year",
      'country',
      'Fatalities in neighborhood', 
      #"t-1 fatalities",
      'GDP', 
      'Growth', 
      'Population',
      'Oil rents',
      #'Percipitation', 
      'Temperature', 
      #'Ethnic fractionalization', 
      'Powerless, share', 
      'Male education',
      'Liberal democracy', 
      "Regulatory quality",
      #'Terrain'
      #"Terrain",
      ]]

#################################
### Out-of-sample Performance ###
#################################

# Data split
#train_y = pd.DataFrame()
##test_y = pd.DataFrame()
#train_x = pd.DataFrame()
#test_x = pd.DataFrame()
    
#val_train_index = []
#val_test_index = []
    
#for c in y.country.unique():
#    y_s = y.loc[y["country"] == c]
#    x_s = x.loc[x["country"] == c]
#    
#    # Train, test
#    y_train = y_s[["country","year"]+[target]][:int(0.8*len(y_s))]
#    x_train = x_s[["country","year"]+inputs][:int(0.8*len(x_s))]
#    y_test = y_s[["country","year"]+[target]][int(0.8*len(y_s)):]
#    x_test = x_s[["country","year"]+inputs][int(0.8*len(x_s)):]
#    # Merge
#    train_y = pd.concat([train_y, y_train])
#    test_y = pd.concat([test_y, y_test])
#    train_x = pd.concat([train_x, x_train])
#    test_x = pd.concat([test_x, x_test])
#    
#    # Validation
#    val_train_index += list(y_train[:int(0.8*len(y_train))].index)
#    val_test_index += list(y_train[int(0.8*len(y_train)):].index)
#    
#splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
    
# Train model 
#train_y_d=train_y.drop(columns=["country","year"])
#train_x_d=train_x.drop(columns=["country","year"])
#test_y_d=test_y.drop(columns=["country","year"])
#test_x_d=test_x.drop(columns=["country","year"])
     
# Train model  
#ps = PredefinedSplit(test_fold=splits)
#grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=1), param_grid=random_grid, cv=ps, verbose=0, n_jobs=-1)
#grid_search.fit(train_x_d.values, train_y_d.values.ravel())
#best_params = grid_search.best_params_
#model=RandomForestClassifier(random_state=1,**best_params)
#model.fit(train_x_d.values, train_y_d.values.ravel())
#pred = pd.DataFrame(model.predict_proba(test_x_d)[:, 1])##

#pred_b = pd.DataFrame(model.predict(test_x_d)) 

#cm = confusion_matrix(test_y_d, pred_b)
#plt.figure(figsize=(6,4))
#sns.heatmap(cm, annot=True,fmt="d", cmap="Blues", xticklabels=["No onset", "Onset"], yticklabels=["No onset", "Onset"])
#plt.xlabel("Predicted Label")
#plt.ylabel("True Label")
#plt.show()
   
#bier = brier_score_loss(test_y_d, pred)
#aupr = average_precision_score(test_y_d, pred)
#auroc = roc_auc_score(test_y_d, pred)
#evals = {"bier": bier, "aupr": aupr, "auroc": auroc}
#print(round(evals["bier"],6))
#print(round(evals["aupr"],6))
#print(round(evals["auroc"],6))#

# Baseline
#base=DummyClassifier()
#base.fit(train_x_d.values, train_y_d.values.ravel())
#pred_base = pd.DataFrame(base.predict(test_x_d))
       
#brier = brier_score_loss(test_y_d, pred_base)
#aupr = average_precision_score(test_y_d, pred_base)
#auroc = roc_auc_score(test_y_d, pred_base)
#evals_base = {"bier": brier, "aupr": aupr, "auroc": auroc}
#print(round(evals_base["bier"],6))
#print(round(evals_base["aupr"],6))
#print(round(evals_base["auroc"],6))

#############################
### In-sample predictions ###
#############################


val_train_index = []
val_test_index = []
    
for c in y.country.unique():
    y_s = y.loc[y["country"] == c]
    x_s = x.loc[x["country"] == c]
    
    # Train, test
    y_train = y_s[["country","year"]+[target]][:int(0.7*len(y_s))]
    y_test = y_s[["country","year"]+[target]][int(0.7*len(y_s)):]
    
    # Validation
    val_train_index += list(y_train.index)
    val_test_index += list(y_test.index)
    
splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
    
# Decision tree example

model = DecisionTreeClassifier(random_state=42,max_depth=3)
model.fit(df[['GDP', 'Fatalities in neighborhood',
        'Growth', 
        'Population', 
        'Oil rents',
        'Powerless, share', 
         "Regulatory quality",]], df[[target]].values.ravel())    

plt.figure(figsize=(23,10))

plot_tree(model, feature_names=df[['GDP', 'Fatalities in neighborhood',
        'Growth', 
        'Population', 
        'Oil rents',
        'Powerless, share', 
         "Regulatory quality",]].columns, filled=False,fontsize=25)
plt.savefig("out/decion_tree.png",dpi=300,bbox_inches='tight')
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/interpretability/out/decion_tree.eps",dpi=300,bbox_inches='tight')        
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/decion_tree.eps",dpi=300,bbox_inches='tight')
plt.show()

# Export as DOT data
#dot_data = export_graphviz(
#    model,
#    out_file=None,
#    feature_names=df[inputs].columns,
#    class_names=True,
#    filled=True,
#    rounded=True,
#    special_characters=True
#)

# Create graph
#graph = graphviz.Source(dot_data)
#graph.render("tree")  # saves as tree.pdf
#graph.view()


# Train model  
ps = PredefinedSplit(test_fold=splits)
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=1), param_grid=random_grid, cv=ps, verbose=0, n_jobs=-1)
grid_search.fit(df[inputs], df[[target]].values.ravel())
best_params = grid_search.best_params_
model=RandomForestClassifier(random_state=1,**best_params)
model.fit(df[inputs], df[[target]].values.ravel())

#pred_b = pd.DataFrame(model.predict(test_x_d)) 

model.fit(df[inputs].values, df[target].values)
pred_b = pd.DataFrame(model.predict(df[inputs])) 

cm = confusion_matrix(df[target], pred_b)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True,fmt="d", cmap="Blues", xticklabels=["No onset", "Onset"], yticklabels=["No onset", "Onset"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

################################
### Interpretability methods ###
################################

x_d=df[inputs]
y_d=df[[target]]

onsets=df.loc[list(y_d.loc[y_d["onset2"]==1].index)]
onsets=onsets.reset_index(drop=True)
onsets.index = range(1, len(onsets) + 1)
onsets[["country","year"]].to_latex(f"out/onsets.tex")

                            ############
                            ### SHAP ###
                            ############


#explainer = shap.Explainer(model)
#shap_values = explainer.shap_values(x_d)
#shap.summary_plot(shap_values[:,:,1],x_d)
#explainer = shap.TreeExplainer(model)
#expected_value = explainer.expected_value[1]

#shaps=pd.DataFrame()
#for i in list(y_d.loc[y_d["onset2"]==1].index):
#    #plt.title(f"Government of {df['country'].iloc[i]}---{df['side_b'].iloc[i]}")
#    #shap.decision_plot(expected_value, shap_values[i,:,1],x_d.iloc[i],highlight=0)
#    d=np.array(shap_values[i,:,1])
#    d = pd.DataFrame(d.flatten(), columns=[i])
#    shaps=pd.concat([shaps,d],axis=1)
#shaps.index=x_d.columns

predict_fn = lambda x_d: model.predict_proba(x_d)[:, 1]
explainer = KernelShap(predict_fn,feature_names=list(x_d.columns),link='logit')
explainer.feature_names
explainer.fit(x_d)

shaps=pd.DataFrame()
for i in list(y_d.loc[y_d["onset2"]==1].index):
   X_explain = x_d.iloc[i].values.reshape(1, -1)
   explanation = explainer.explain(X_explain)  
   d=np.array(explanation.shap_values)
   d = pd.DataFrame(d.flatten(), columns=[i])
   shaps=pd.concat([shaps,d],axis=1)
shaps.index=x_d.columns
shaps.columns=onsets.index

cluster_map=sns.clustermap(shaps,method="ward",metric="euclidean",col_cluster=True,row_cluster=False)
linkage_matrix = linkage(shaps.T, method="ward",metric="euclidean")
cluster_labels = fcluster(linkage_matrix, 8, criterion="maxclust")
ordered_columns = [shaps.columns[i] for i in cluster_map.dendrogram_col.reordered_ind]



cluster_labels = np.sort(cluster_labels)
shaps_plot = shaps.loc[inputs]
maps1=sns.clustermap(shaps_plot, cmap="bone_r",linewidths=0,method="ward",metric="euclidean",col_cluster=True,row_cluster=False,yticklabels=12,figsize=(20,5),cbar_pos=(0.99, 0.1, 0.013, 0.65))
#maps1.cax.set_visible(False)

# Get the image object from the heatmap axis
#im = maps1.ax_heatmap.collections[0]

# Manually create colorbar on the right
#cbar_ax = maps1.fig.add_axes([1, 0.2, 0.01, 0.5])
#cbar = maps1.fig.colorbar(im, cax=cbar_ax)
#cbar.ax.tick_params(labelsize=12)
maps1.ax_heatmap.yaxis.set_ticks_position('left')
maps1.ax_heatmap.yaxis.set_label_position('left')
maps1.ax_heatmap.tick_params(axis='y', labelleft=True)
maps1.ax_heatmap.set_yticks(ticks=np.arange(len(shaps_plot))+0.5, labels=inputs,size=25,rotation=0,ha="right")
maps1.ax_heatmap.set_xticks(ticks=np.arange(len(ordered_columns))+0.5, labels=ordered_columns,size=15)
ax = maps1.ax_heatmap
change_points_col = np.where(np.diff(cluster_labels) != 0)[0] + 1
for x in change_points_col:
    ax.vlines(x=x, ymin=0, ymax=len(df), colors="black", linewidth=1)
   
#plt.tight_layout()
plt.savefig("out/shaps.eps",dpi=300,bbox_inches='tight') 
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/interpretability/out/inter_shaps.eps",dpi=300,bbox_inches='tight')        
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/inter_shaps.eps",dpi=300,bbox_inches='tight')
plt.show()

#buf = BytesIO()
#maps1.savefig(buf, format='png')
#buf.seek(0)
#plt.close(maps1.fig)  # Close the original figure to avoid duplication

# Create a new figure with two subplots
#fig, ax = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1.7]})

# Insert the clustermap image on the left
#axins = inset_axes(ax[0], width="100%", height="100%", bbox_to_anchor=(-0.32, -0.085, 1.46, 1.36), bbox_transform=ax[0].transAxes)
#axins.imshow(imread(buf))
#axins.axis('off')  # Hide axes from the inset
#ax[0].axis('off')  # Hide original subplot borders

# Plot the heatmap on the right
#df_normalized = (x_d - x_d.min()) / (x_d.max() - x_d.min())
corr=x_d.loc[list(y_d.loc[y_d["onset2"]==1].index)].T
corr.columns=onsets.index
corr_n=corr[ordered_columns]

#sns.heatmap(corr_n, cmap="seismic", linewidths=0,cbar=False, ax=ax[1])
#ax[1].set_xticks(ticks=np.arange(len(ordered_columns))+0.5, labels=ordered_columns,size=13)
#for x in change_points_col:
#    ax[1].vlines(x=x, ymin=0, ymax=len(df), colors="black", linewidth=1)
#ax[1].tick_params(left=False, labelleft=False)


#from pandas.plotting import parallel_coordinates
#corr_n=corr_n.T
#corr_n=corr_n.reset_index()
#parallel_coordinates(corr_n,"index")
start_indices = [0] + list(change_points_col)
end_indices = list(change_points_col) + [None]
for i, (start, end) in enumerate(zip(start_indices, end_indices), start=1):
    sub_df = corr_n.iloc[:, start:end]
    sub_df["Means"]=x_d.mean()
    sub_df.to_latex(f"out/vars_{start}.tex", float_format="%.2f")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Simulated dataset: 9 variables (rows), 58 observations (columns)
corr_n_v=corr_n.values
colormaps = ['bone_r', 'bone_r', 'seismic', 'bone_r', 'bone_r', 'bone_r', 'bone_r', 'bone_r', 'bone_r',"seismic"]
vmin = corr_n_v.min(axis=1)
vmax = corr_n_v.max(axis=1)

# Setup figure and GridSpec layout
fig = plt.figure(figsize=(20,5))
gs = gridspec.GridSpec(
    10, 2,
    width_ratios=[20, 0.3],  # Make colorbars narrower and closer
    height_ratios=[10]*10,
    hspace=0.1,
    wspace=0.02  # Reduce horizontal space between heatmap and colorbar
)
axes = []
for i in range(10):
    print(i)
    ax = fig.add_subplot(gs[i, 0])
    cax = fig.add_subplot(gs[i, 1])  # Colorbar axes

    # Reshape row to 2D for imshow
    row_data = corr_n_v[i, :].reshape(1, -1)
    
    im = ax.imshow(row_data, aspect='auto', cmap=colormaps[i], vmin=vmin[i], vmax=vmax[i])
    
    for x in change_points_col:
        ax.vlines(x=x - 0.5, ymin=-0.5, ymax=0.5, color='black', linewidth=2)
        
    # Clean axis ticks
    ax.set_yticks([])
    ax.set_ylabel(inputs[i], rotation=0, ha='right', fontsize=25,va="center")
    ax.set_xticks([])

    if i==9:
        ax.set_xticks(np.arange(0, 58, 1),corr_n.columns,size=15)

    
    # Add colorbar to the right of each row
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7)
    

    axes.append(ax)
plt.tight_layout()

plt.savefig("out/vars.eps",dpi=300,bbox_inches='tight')
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/interpretability/out/inter_vars.eps",dpi=300,bbox_inches='tight')        
plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/inter_vars.eps",dpi=300,bbox_inches='tight')
plt.show()


                                ############
                                ### LIME ###
                                ############

limes=lime.lime_tabular.LimeTabularExplainer(x_d.values,feature_names=x_d.columns,class_names=['No Onset', 'Onset'])

limes_out=[]

for i in list(y_d.loc[y_d["onset2"]==1].index):
    exp=limes.explain_instance(x_d[x_d.index==i].values[0],model.predict_proba).as_list()
    limes_out.append(exp)
    #plt.title(f"Government of {df['country'].iloc[i]}---{df['side_b'].iloc[i]}")



case_feature_sets = [set(f for f, w in exp) for exp in limes_out]

from itertools import combinations
from collections import Counter

combo_counter = Counter()

for features in case_feature_sets:
    if len(features) >=4:
        combos = combinations(sorted(features), 4)
        combo_counter.update(combos)

for combo, count in combo_counter.most_common(10):
    print(f"{combo}: {count}")


                            ####################
                            ### Scoped rules ###
                            ####################


explainer = AnchorTabular(model.predict, x_d.columns)
explainer.fit(x_d.values)  

rule_list = []

for i in list(y_d.loc[y_d["onset2"]==1].index):
    X_explain = x_d.iloc[i].values.reshape(1, -1)  
    explanation = explainer.explain(X_explain)     
    
    rule_list.append(explanation.anchor)
    
   

from itertools import combinations
from collections import Counter

combo_counter = Counter()

for features in rule_list:
    if len(features) >=3:
        combos = combinations(sorted(features), 3)
        combo_counter.update(combos)

subset=[]
for combo, count in combo_counter.most_common(10):
    print(f"{combo}: {count}")
    subset.append(combo)
    
memberships = list(map(tuple, rule_list))


#rule_list_s = [rule_list[i-1] for i in ordered_columns]

#start_indices = [0] + list(change_points_col)
#end_indices = list(change_points_col) + [None]
#for i, (start, end) in enumerate(zip(start_indices, end_indices), start=1):
#    sub_df = rule_list_s[start:end]
#    sub_df.to_latex(f"out/scoped_{start}.tex", float_format="%.2f")


#for i in change_points_col[:1]:

#    l=rule_list_s[change_points_col[i]: change_points_col[i+1]] 

from upsetplot import from_memberships
memberships = list(map(tuple, rule_list))
data = from_memberships(memberships, data=[1]*len(memberships))
import matplotlib.pyplot as plt
from upsetplot import UpSet

plt.figure()
UpSet(
    data,
    show_counts=True,        # print counts on the bars
    sort_by='cardinality'    # largest intersections first
).plot()

plt.title("UpSet plot of scoped‑rule conditions")
plt.show()


from collections import defaultdict
from typing import List, Tuple

# ------------- helpers --------------------------------------------------
def make_trie():
    """Recursively default‑to‑dict-of-dicts."""
    return defaultdict(make_trie)

def insert_rule(trie, conditions: Tuple[str, ...], leaf_info: str):
    """Insert a rule (sequence of conditions) into the trie."""
    node = trie
    for cond in conditions:
        node = node[cond]          # walk / create each level
    node['_leaf'] = leaf_info      # store metrics at the leaf

def print_tree(trie, depth=0, bullet='•', indent='    '):
    """Recursively pretty‑print the trie."""
    for cond, subtree in trie.items():
        if cond == '_leaf':        # reached a leaf
            continue
        print(f"{indent*depth}{bullet} {cond}")
        print_tree(subtree, depth+1)          # descend
        
    # print leaf info *after* children so it’s aligned under last cond
    if '_leaf' in trie and depth:             # root itself has no bullet
        print(f"{indent*depth}{trie['_leaf']}")

# ------------- example data ---------------------------------------------
rules = [
    ("GDP < 3000", "Polity ≤ –5", "EthFrac > 0.40",      "(prec 0.91 | cov 0.07)"),
    ("GDP < 3000", "PriorWar = Yes",                     "(prec 0.87 | cov 0.10)"),
    ("Age ≤ 27",   "Rating ≥ 4.5",                       "(prec 0.93 | cov 0.18)")
]

# ------------- build & print -------------------------------------------
trie = make_trie()
for *conds, metrics in subset:            # last element = metrics string
    insert_rule(trie, tuple(conds), metrics)

print_tree(trie)







                    ###################################
                    ### Counterfactual explanations ###
                    ###################################

shape = (1,) + x_d.shape[1:]
explainer = CounterfactualProto(model.predict_proba,shape=shape)
explainer.fit(x_d.values)

idx = 45  # Select a sample index from the test set
X_explain = x_d.iloc[idx].values.reshape(1, -1)  # Reshape for prediction
original_prediction = model.predict(X_explain)[0]  # Model's original prediction

# Define the target class (choose a different class)
target_class = [0] # Cycle through classes (Iris has 3)

# Generate the counterfactual explanation
explanation = explainer.explain(X_explain, target_class=target_class)

# Print results
print(f"Original Prediction: {original_prediction}")
print(f"Counterfactual Target: {target_class}")
print(f"Counterfactual Instance: {explanation.cf['X']}")
print(f"Feature Changes: {explanation.cf['X'] - X_explain}")


# Extract feature differences
feature_changes = (explanation.cf['X'] - X_explain).flatten()

plt.figure(figsize=(8, 5))
plt.barh(x_d.columns, feature_changes, color='blue')
plt.xlabel("Feature Change")
plt.ylabel("Features")
plt.title("Counterfactual Feature Changes")
plt.axvline(x=0, color='black', linestyle='--')
plt.show()

target_class = 1  # Change this to any class you want to explain
positive_cases = [i for i in range(len(x_d)) if model.predict(x_d.iloc[i].values.reshape(1, -1))[0] == target_class]
print(f"Number of Positive Cases: {len(positive_cases)}")

for idx in positive_cases:  # Explain first 5 samples
    X_explain = x_d.iloc[idx].values.reshape(1, -1)
    original_prediction = model.predict(X_explain)[0]
    target_class = [0]  # Cycle through classes
    
    explanation = explainer.explain(X_explain, target_class=target_class)
    
    print(f"Instance {idx + 1}:")
    print(f"Original Class: {original_prediction}")
    print(f"Counterfactual Class: {target_class}")
    print(f"Feature Changes: {explanation.cf['X'] - X_explain}")
    print("-" * 40)