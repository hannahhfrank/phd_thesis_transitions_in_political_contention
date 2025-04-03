import pandas as pd
import numpy as np
from functions import data_split,earth_mover_distance,preprocess_min_max_group
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


random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)],
               #'max_features': ["sqrt", "log2", None],
               #'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],
               #'min_samples_split': [2,5,10],
               #'min_samples_leaf': [1,2,4]
               }

df=pd.read_csv("data/df_interpret.csv",index_col=0)
df["country"] = df["country"].str.replace(r"\s*\(.*?\)", "", regex=True)  # Remove everything inside ()

#for var in ['sb_fatalities','onset','sb_fatalities_lag1','d_civil_conflict_zeros_decay','d_neighbors_sb_fatalities_lag1','gdp','growth','pop','oil','percip','temp','ethnic_frac','powerless_share','eys_male','libdem','libdem_change','rugged']:
#    fig,ax = plt.subplots()
#    df[var].hist()
#    plt.title(var)

# Transforms
df["sb_fatalities_log"]=np.log(df["sb_fatalities"]+1)
df["gdp"]=np.log(df["gdp"])
df["pop"]=np.log(df["pop"])

df.columns=['year', 'gw_codes', 'country', 'sb_fatalities', 'side_b', 'onset',
       'onset2',"Time since civil conflict",
       'Fatalities in neighborhood', 'GDP', 'Growth', 'Population', 'Oil rents',
       'Percipitation', 'Temperature', 'Ethnic fractionalization', 'Powerless, share', 'Male education',
       'Liberal democracy', 'Electoral democracy', "Political stability" ,'Terrain', 'sb_fatalities_log']

target='onset2'
inputs=['Fatalities in neighborhood', 'GDP', 'Growth', 'Population', 'Oil rents',
'Percipitation', 'Temperature', 'Ethnic fractionalization', 'Powerless, share', 'Male education',
'Liberal democracy', 'Terrain']
y=df[["year",'country','onset2']]
x=df[["year",'country','Fatalities in neighborhood', 'GDP', 'Growth', 'Population', 'Oil rents',
'Percipitation', 'Temperature', 'Ethnic fractionalization', 'Powerless, share', 'Male education',
'Liberal democracy', 'Terrain']]

#################################
### Out-of-sample Performance ###
#################################

# Data split
train_y = pd.DataFrame()
test_y = pd.DataFrame()
train_x = pd.DataFrame()
test_x = pd.DataFrame()
    
val_train_index = []
val_test_index = []
    
for c in y.country.unique():
    y_s = y.loc[y["country"] == c]
    x_s = x.loc[x["country"] == c]
    
    # Train, test
    y_train = y_s[["country","year"]+[target]][:int(0.8*len(y_s))]
    x_train = x_s[["country","year"]+inputs][:int(0.8*len(x_s))]
    y_test = y_s[["country","year"]+[target]][int(0.8*len(y_s)):]
    x_test = x_s[["country","year"]+inputs][int(0.8*len(x_s)):]
    # Merge
    train_y = pd.concat([train_y, y_train])
    test_y = pd.concat([test_y, y_test])
    train_x = pd.concat([train_x, x_train])
    test_x = pd.concat([test_x, x_test])
    
    # Validation
    val_train_index += list(y_train[:int(0.8*len(y_train))].index)
    val_test_index += list(y_train[int(0.8*len(y_train)):].index)
    
splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
    
# Train model 
train_y_d=train_y.drop(columns=["country","year"])
train_x_d=train_x.drop(columns=["country","year"])
test_y_d=test_y.drop(columns=["country","year"])
test_x_d=test_x.drop(columns=["country","year"])
     
# Train model  
ps = PredefinedSplit(test_fold=splits)
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=1), param_grid=random_grid, cv=ps, verbose=0, n_jobs=-1)
grid_search.fit(train_x_d.values, train_y_d.values.ravel())
best_params = grid_search.best_params_
model=RandomForestClassifier(random_state=1,**best_params)
model.fit(train_x_d.values, train_y_d.values.ravel())
pred = pd.DataFrame(model.predict_proba(test_x_d)[:, 1])

pred_b = pd.DataFrame(model.predict(test_x_d)) 

cm = confusion_matrix(test_y_d, pred_b)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True,fmt="d", cmap="Blues", xticklabels=["No onset", "Onset"], yticklabels=["No onset", "Onset"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
   
bier = brier_score_loss(test_y_d, pred)
aupr = average_precision_score(test_y_d, pred)
auroc = roc_auc_score(test_y_d, pred)
evals = {"bier": bier, "aupr": aupr, "auroc": auroc}
print(round(evals["bier"],6))
print(round(evals["aupr"],6))
print(round(evals["auroc"],6))

# Baseline
base=DummyClassifier()
base.fit(train_x_d.values, train_y_d.values.ravel())
pred_base = pd.DataFrame(base.predict(test_x_d))
       
brier = brier_score_loss(test_y_d, pred_base)
aupr = average_precision_score(test_y_d, pred_base)
auroc = roc_auc_score(test_y_d, pred_base)
evals_base = {"bier": brier, "aupr": aupr, "auroc": auroc}
print(round(evals_base["bier"],6))
print(round(evals_base["aupr"],6))
print(round(evals_base["auroc"],6))

#############################
### In-sample predictions ###
#############################

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

# (1) SHAP A
explainer = shap.Explainer(model)
shap_values = explainer.shap_values(x_d)
shap.summary_plot(shap_values[:,:,1],x_d)
explainer = shap.TreeExplainer(model)
expected_value = explainer.expected_value[1]

shaps=pd.DataFrame()
for i in list(y_d.loc[y_d["onset2"]==1].index):
    plt.title(f"Government of {df['country'].iloc[i]}---{df['side_b'].iloc[i]}")
    shap.decision_plot(expected_value, shap_values[i,:,1],x_d.iloc[i],highlight=0)
    d=np.array(shap_values[i,:,1])
    d = pd.DataFrame(d.flatten(), columns=[i])
    shaps=pd.concat([shaps,d],axis=1)
shaps.index=x_d.columns

cluster_map=sns.clustermap(shaps,method="ward",metric="euclidean",col_cluster=True,row_cluster=False)
linkage_matrix = linkage(shaps.T, method="ward",metric="euclidean")
cluster_labels = fcluster(linkage_matrix, 9, criterion="maxclust")
ordered_columns = [shaps.columns[i] for i in cluster_map.dendrogram_col.reordered_ind]

plt.figure(figsize=(10, 8))  
cluster_labels = np.sort(cluster_labels)
shaps_plot = shaps.loc[inputs]
maps=sns.clustermap(shaps_plot, cmap="bone_r",linewidths=0,method="ward",metric="euclidean",col_cluster=True,row_cluster=False,yticklabels=12,figsize=(10,8))
maps.ax_heatmap.set_yticks(ticks=np.arange(len(shaps_plot))+0.5, labels=inputs,size=12,rotation=0,ha="left")
maps.ax_heatmap.set_xticks(ticks=np.arange(len(ordered_columns))+0.5, labels=ordered_columns,size=12)
ax = maps.ax_heatmap
change_points_col = np.where(np.diff(cluster_labels) != 0)[0] + 1
for x in change_points_col:
    ax.vlines(x=x, ymin=0, ymax=len(df), colors="black", linewidth=1)
plt.savefig("out/cluster_map.png",dpi=300,bbox_inches='tight')
plt.show()

    
df_normalized = (x_d - x_d.min()) / (x_d.max() - x_d.min())
corr=df_normalized.loc[list(y_d.loc[y_d["onset2"]==1].index)].T
corr_n=corr[ordered_columns]
plt.figure(figsize=(10,8))  
sns.heatmap(corr_n, cmap="Purples", linewidths=0)
plt.xticks(ticks=np.arange(len(ordered_columns)), labels=ordered_columns,size=8)
for x in change_points_col:
    plt.vlines(x=x, ymin=0, ymax=len(df), colors="black", linewidth=1)
plt.savefig("out/input_map.png",dpi=300,bbox_inches='tight')
plt.show()

# (1) SHAP B
explainer = TreeShap(model)
explainer.fit(x_d)

shaps=pd.DataFrame()
for i in list(y_d.loc[y_d["onset2"]==1].index):
    X_explain = x_d.iloc[i].values.reshape(1, -1)
    explanation = explainer.explain(X_explain)  
    d=np.array(explanation.shap_values)
    d = pd.DataFrame(d.flatten(), columns=[i])
    shaps=pd.concat([shaps,d],axis=1)
shaps.index=x_d.columns

cluster_map=sns.clustermap(shaps,method="ward",metric="euclidean",col_cluster=True,row_cluster=False)
linkage_matrix = linkage(shaps.T, method="ward",metric="euclidean")
cluster_labels = fcluster(linkage_matrix, 7, criterion="maxclust")
ordered_columns = [shaps.columns[i] for i in cluster_map.dendrogram_col.reordered_ind]

plt.figure(figsize=(10, 8))  
cluster_labels = np.sort(cluster_labels)
shaps_plot = shaps.loc[inputs]
maps=sns.clustermap(shaps_plot, cmap="bwr",linewidths=0,method="ward",metric="euclidean",col_cluster=True,row_cluster=False,yticklabels=12,figsize=(10,8))
maps.ax_heatmap.set_yticks(ticks=np.arange(len(shaps_plot))+0.5, labels=inputs,size=12,rotation=0,ha="left")
maps.ax_heatmap.set_xticks(ticks=np.arange(len(ordered_columns))+0.5, labels=ordered_columns,size=12)
ax = maps.ax_heatmap
change_points_col = np.where(np.diff(cluster_labels) != 0)[0] + 1
for x in change_points_col:
    ax.vlines(x=x, ymin=0, ymax=len(df), colors="black", linewidth=1)
plt.savefig("out/cluster_map.png",dpi=300,bbox_inches='tight')
plt.show()


# (1) SHAP C
predict_fn = lambda x_d: model.predict_proba(x_d)[:, 1]
explainer = KernelShap(predict_fn,feature_names=list(x_d.columns),link='logit')
explainer.feature_names
explainer.fit(x_d)

shaps=pd.DataFrame()[:1]
for i in list(y_d.loc[y_d["onset2"]==1].index)[:1]:
    X_explain = x_d.iloc[i].values.reshape(1, -1)
    explanation = explainer.explain(X_explain)  
    d=np.array(explanation.shap_values)
    d = pd.DataFrame(d.flatten(), columns=[i])
    shaps=pd.concat([shaps,d],axis=1)
shaps.index=x_d.columns

cluster_map=sns.clustermap(shaps,method="ward",metric="euclidean",col_cluster=True,row_cluster=False)
linkage_matrix = linkage(shaps.T, method="ward",metric="euclidean")
cluster_labels = fcluster(linkage_matrix, 7, criterion="maxclust")
ordered_columns = [shaps.columns[i] for i in cluster_map.dendrogram_col.reordered_ind]

plt.figure(figsize=(10, 8))  
cluster_labels = np.sort(cluster_labels)
shaps_plot = shaps.loc[inputs]
maps=sns.clustermap(shaps_plot, cmap="bwr",linewidths=0,method="ward",metric="euclidean",col_cluster=True,row_cluster=False,yticklabels=12,figsize=(10,8))
maps.ax_heatmap.set_yticks(ticks=np.arange(len(shaps_plot))+0.5, labels=inputs,size=12,rotation=0,ha="left")
maps.ax_heatmap.set_xticks(ticks=np.arange(len(ordered_columns))+0.5, labels=ordered_columns,size=12)
ax = maps.ax_heatmap
change_points_col = np.where(np.diff(cluster_labels) != 0)[0] + 1
for x in change_points_col:
    ax.vlines(x=x, ymin=0, ymax=len(df), colors="black", linewidth=1)
plt.savefig("out/cluster_map.png",dpi=300,bbox_inches='tight')
plt.show()

# (2) LIME
lime=lime.lime_tabular.LimeTabularExplainer(x_d.values,feature_names=x_d.columns,class_names=['No Onset', 'Onset'])

for i in list(y_d.loc[y_d["onset2"]==1].index):
    lime.explain_instance(x_d[x_d.index==i].values[0],model.predict_proba).as_pyplot_figure()
    plt.title(f"Government of {df['country'].iloc[i]}---{df['side_b'].iloc[i]}")

# (3) Counterfactual explanations

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


# (4) Scoped rules


explainer = AnchorTabular(model.predict, x_d.columns)
explainer.fit(x_d.values)  

rule_list = []

for i in list(y_d.loc[y_d["onset2"]==1].index)[:1]:
    X_explain = x_d.iloc[i].values.reshape(1, -1)  # Select the instance
    explanation = explainer.explain(X_explain)  # Explain with high precision
    
    rule_list.append(explanation.anchor)
    


# Print and visualize the rules
print("Scoped Rules:", explanation.anchor)

plt.figure(figsize=(8, 4))
plt.barh(explanation.anchor, explanation.precision, color="green")
plt.xlabel("Precision")
plt.title("Scoped Rules from Anchor Explainer")
plt.show()


# Display all rules
for i, rule in enumerate(rule_list):
    print(f"Case {i+1}: {rule}")

from collections import Counter

# Flatten all rules into a single list
all_rules = [rule for rule_set in rule_list for rule in rule_set]

# Count occurrences of each rule
rule_counts = Counter(all_rules)

# Display top rules
print("\nMost Common Scoped Rules for Positive Cases:")
for rule, count in rule_counts.most_common():
    print(f"{rule}: {count} times")


# Extract top 10 rules
top_rules = rule_counts.most_common(10)
rule_labels, rule_frequencies = zip(*top_rules)

# Plot
plt.figure(figsize=(8, 5))
plt.barh(rule_labels, rule_frequencies, color='blue')
plt.xlabel("Frequency")
plt.ylabel("Rules")
plt.title(f"Top 10 Scoped Rules for Class {target_class}")
plt.gca().invert_yaxis()  # Flip the y-axis for better readability
plt.show()
