import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import gridspec
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

plot_params = {"text.usetex":True,"font.family":"serif","font.size":20,"xtick.labelsize":20,"ytick.labelsize":20,"axes.labelsize":20,"figure.titlesize":20,"figure.figsize":(8,5)}
plt.rcParams.update(plot_params)

def evals(y_true, y_pred, countries, onset_tolerance=3, alpha=0.7, beta=0.3, horizon=24):

    unique_countries = countries.unique()
    onset_scores_by_country = []
    mse_values = []
    d_nn=[]
    final_scores_by_country = []
    normalized_mse_by_country = []

    for country in unique_countries:
     # Filter data for the current country
        country_mask = countries == country
        y_true_country = y_true[country_mask]
        y_pred_country = y_pred[country_mask]

        # Detect Onsets in Ground Truth and Predictions for the Country
        true_onsets = y_true_country[(y_true_country.shift(1) == 0) & (y_true_country > 0)].index
        pred_onsets = y_pred_country[(y_pred_country.shift(1) == 0) & (y_pred_country > 0)].index
        
        # Calculate Onset Score
        correct_onsets = []
        for true_onset in list(true_onsets):
            # Find the closest predicted onset within the tolerance
            closest_pred = pred_onsets[np.abs(pred_onsets - true_onset) <= onset_tolerance]
            if not closest_pred.empty:
                delta_t = np.abs(closest_pred[0] - true_onset)
                correct_onsets.append(np.exp(-0.1*delta_t))  # Exponential penalty for timing error

        onset_score = np.sum(correct_onsets) / len(true_onsets) if len(true_onsets) > 0 else 0
        onset_scores_by_country.append(onset_score)    

        # Compute MSE for the current country
        mse = np.mean((y_true_country.values-y_pred_country.values) ** 2)
        mse_values.append(mse)
            
        # Normalize MSE for the Current Country
        # Min and max are calculated for the country's data
        if y_true_country.max() == y_true_country.min() and y_pred_country.max() == y_pred_country.min():
            normalized_true = pd.Series([0] * len(y_true_country))  # or assign 1 if you prefer
            normalized_pred = pd.Series([0] * len(y_pred_country))  # or assign 1 if you prefer
    
        elif y_true_country.max() == y_true_country.min() and y_pred_country.max() != y_pred_country.min():
            normalized_true = pd.Series([0] * len(y_true_country))  # or assign 1 if you prefer
            normalized_pred = (y_pred_country - y_pred_country.min()) / (y_pred_country.max() - y_pred_country.min()) 
        
        elif y_true_country.max() != y_true_country.min() and y_pred_country.max() == y_pred_country.min():
            normalized_true = (y_true_country - y_true_country.min()) / (y_true_country.max() - y_true_country.min()) 
            normalized_pred = pd.Series([0] * len(y_pred_country))  # or assign 1 if you prefer
          
        elif y_true_country.max() != y_true_country.min() and y_pred_country.max() != y_pred_country.min():
            normalized_true = (y_true_country - y_true_country.min()) / (y_true_country.max() - y_true_country.min()) 
            normalized_pred = (y_pred_country - y_pred_country.min()) / (y_pred_country.max() - y_pred_country.min()) 
        normalized_mse = np.mean((normalized_true.values - normalized_pred.values) ** 2)
        normalized_mse_by_country.append(normalized_mse)
        
        # Calculate Final Score for the Current Country
        final_score = alpha * onset_score + beta * (1 - normalized_mse)
        final_scores_by_country.append(final_score)
              
        # DE
        real = y_true_country
        real=real.reset_index(drop=True)
        sf = y_pred_country
        sf=sf.reset_index(drop=True)

        max_s=0
        if (real==0).all()==False:
            for value in real[1:].index:
                if (real[value]==real[value-1]):
                    1
                else:
                    max_exp=0
                    if (real[value]-real[value-1])/(sf[value]-sf[value-1])>0 and sf[value]-sf[value-1] != 0:
                        t=abs(((real[value]-real[value-1])-(sf[value]-sf[value-1]))/(real[value]-real[value-1]))
                        max_exp = np.exp(-(5*t))
                    else:
                        if value==horizon-1:
                            if (real[value]-real[value-1])/(sf[value-1]-sf[value-2])>0 and sf[value-1]-sf[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value-1]-sf[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(5*t)):
                                    max_exp = np.exp(-(5*t))
                        elif value==1:
                            if (real[value]-real[value-1])/(sf[value+1]-sf[value])>0 and sf[value+1]-sf[value] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value+1]-sf[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(5*t)):
                                    max_exp = np.exp(-(5*t))
                        else : 
                            if (real[value]-real[value-1])/(sf[value-1]-sf[value-2])>0 and sf[value-1]-sf[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value-1]-sf[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(5*t)):
                                    max_exp = np.exp(-(5*t))
                            if (real[value]-real[value-1])/(sf[value+1]-sf[value])>0 and sf[value+1]-sf[value] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value+1]-sf[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(5*t)):
                                    max_exp = np.exp(-(5*t))
                    max_s=max_s+max_exp 
            d_nn.append(max_s)
        else:
            d_nn.append(0)   
         
    # Calculate the mean MSE across all countries
    mean_mse = np.mean(mse_values)
    de_mean = np.mean(d_nn)
    normalized_mse=np.mean(normalized_mse_by_country)
    onset_score=np.mean(onset_scores_by_country)
    final_score=np.mean(final_scores_by_country)

    return {
        "Difference Explained": de_mean,
        "Onset Score": onset_score,
        "Mean MSE": mean_mse,
        "Normalized MSE": normalized_mse,
        "Final Score": final_score,
        "DE by Country": d_nn,
        "Onset Scores by Country": onset_scores_by_country,
        "MSE by Country": mse_values,
        "Normalized MSE by Country": normalized_mse_by_country,
        "Final Scores by Country": final_scores_by_country,

    }


shape_finder=pd.read_csv("out/shape_finder.csv",index_col=0)
evals_shape_finder = evals(shape_finder.sb_fatalities, shape_finder.preds, shape_finder.country)
print(f"Difference Explained {evals_shape_finder['Difference Explained']}")
print(f"Onset Score {evals_shape_finder['Onset Score']}")
print(f"Mean MSE {evals_shape_finder['Mean MSE']}")
print(f"Normalized MSE {evals_shape_finder['Normalized MSE']}")
print(f"Final Score {evals_shape_finder['Final Score']}")

views=pd.read_csv("out/views.csv",index_col=0)
evals_views = evals(views.sb_fatalities, views.preds, views.country)
print(f"Difference Explained {evals_views['Difference Explained']}")
print(f"Onset Score {evals_views['Onset Score']}")
print(f"Mean MSE {evals_views['Mean MSE']}")
print(f"Normalized MSE {evals_views['Normalized MSE']}")
print(f"Final Score {evals_views['Final Score']}")

zinb=pd.read_csv("out/zinb.csv",index_col=0)
evals_zinb = evals(zinb.sb_fatalities, zinb.preds, zinb.country)
print(f"Difference Explained {evals_zinb['Difference Explained']}")
print(f"Onset Score {evals_zinb['Onset Score']}")
print(f"Mean MSE {evals_zinb['Mean MSE']}")
print(f"Normalized MSE {evals_zinb['Normalized MSE']}")
print(f"Final Score {evals_zinb['Final Score']}")

catcher=pd.read_csv("out/catcher.csv",index_col=0)
catcher=catcher.loc[catcher["dd"]>="2022-01"]
evals_catcher = evals(catcher.sb_fatalities, catcher.preds, catcher.country)
print(f"Difference Explained {evals_catcher['Difference Explained']}")
print(f"Onset Score {evals_catcher['Onset Score']}")
print(f"Mean MSE {evals_catcher['Mean MSE']}")
print(f"Normalized MSE {evals_catcher['Normalized MSE']}")
print(f"Final Score {evals_catcher['Final Score']}")

# Ensemble
#catcher_countries = pd.DataFrame({'country': catcher.country.unique(),"onsets":evals_catcher["Onset Scores by Country"]})
#catcher_countries=list(catcher_countries.loc[catcher_countries["onsets"]>0].country)
#catcher_countries = catcher[(catcher["preds"].shift(1) == 0) & (catcher["preds"] > 0)].country.unique()
catcher_countries=catcher[(catcher["preds"].shift(1) == 0) & (catcher["preds"] > 0)].groupby("country")["dd"].count()
catcher_countries=catcher_countries[catcher_countries>1].index
catcher_s = catcher[catcher['country'].isin(catcher_countries)]
base = catcher[["country","dd","sb_fatalities"]]
ensemble=pd.merge(base,catcher_s[["country","dd","preds"]],on=["country","dd"], how="left")
ensemble['preds'] = ensemble['preds'].fillna(shape_finder['preds'])

evals_ensemble = evals(ensemble.sb_fatalities, ensemble.preds, ensemble.country)
print(f"Difference Explained {evals_ensemble['Difference Explained']}")
print(f"Onset Score {evals_ensemble['Onset Score']}")
print(f"Mean MSE {evals_ensemble['Mean MSE']}")
print(f"Normalized MSE {evals_ensemble['Normalized MSE']}")
print(f"Final Score {evals_ensemble['Final Score']}")


### GLobal evaluation ###
def bootstrap_se(data, n_bootstrap=100):
    bootstrap_statistics = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_statistics.append(np.mean(bootstrap_sample))
    bootstrap_se = np.std(bootstrap_statistics)
    return bootstrap_se

# Difference explained
means = [np.mean(evals_shape_finder['DE by Country']),np.mean(evals_views['DE by Country']),np.mean(evals_zinb['DE by Country']),np.mean(evals_catcher['DE by Country']),np.mean(evals_ensemble['DE by Country'])]
#std_error = [2*np.std(evals_shape_finder['DE by Country'])/np.sqrt(len(evals_shape_finder['DE by Country'])),2*np.std(evals_views['DE by Country'])/np.sqrt(len(evals_views['DE by Country'])),2*np.std(evals_zinb['DE by Country'])/np.sqrt(len(evals_zinb['DE by Country'])),2*np.std(evals_catcher['DE by Country'])/np.sqrt(len(evals_catcher['DE by Country'])),2*np.std(evals_ensemble['DE by Country'])/np.sqrt(len(evals_ensemble['DE by Country']))]
std_error = [3*(bootstrap_se(evals_shape_finder['DE by Country'])/np.sqrt(len(evals_shape_finder['DE by Country']))),
             3*(bootstrap_se(evals_views['DE by Country']/np.sqrt(len(evals_views['DE by Country'])))),
             3*(bootstrap_se(evals_zinb['DE by Country']/np.sqrt(len(evals_zinb['DE by Country'])))),
             3*(bootstrap_se(evals_catcher['DE by Country']/np.sqrt(len(evals_catcher['DE by Country'])))),
             3*(bootstrap_se(evals_ensemble['DE by Country']/np.sqrt(len(evals_ensemble['DE by Country']))))]
mean_de = pd.DataFrame({'mean': means,'std': std_error})

# Onset score
means = [np.mean(evals_shape_finder["Final Scores by Country"]),np.mean(evals_views["Final Scores by Country"]),np.mean(evals_zinb["Final Scores by Country"]),np.mean(evals_catcher["Final Scores by Country"]),np.mean(evals_ensemble["Final Scores by Country"])]
#std_error = [2*np.std(evals_shape_finder["Final Scores by Country"])/np.sqrt(len(evals_shape_finder["Final Scores by Country"])),2*np.std(evals_views["Final Scores by Country"])/np.sqrt(len(evals_views["Onset Scores by Country"])),2*np.std(evals_zinb["Onset Scores by Country"])/np.sqrt(len(evals_zinb["Onset Scores by Country"])),2*np.std(evals_catcher["Onset Scores by Country"])/np.sqrt(len(evals_catcher["Onset Scores by Country"])),2*np.std(evals_ensemble["Onset Scores by Country"])/np.sqrt(len(evals_ensemble["Onset Scores by Country"]))]
std_error = [3*(bootstrap_se(evals_shape_finder['Final Scores by Country'])/np.sqrt(len(evals_shape_finder['Final Scores by Country']))),
             3*(bootstrap_se(evals_views['Final Scores by Country']/np.sqrt(len(evals_views['Final Scores by Country'])))),
             3*(bootstrap_se(evals_zinb['Final Scores by Country']/np.sqrt(len(evals_zinb['Final Scores by Country'])))),
             3*(bootstrap_se(evals_catcher['Final Scores by Country']/np.sqrt(len(evals_catcher['Final Scores by Country'])))),
             3*(bootstrap_se(evals_ensemble['Final Scores by Country']/np.sqrt(len(evals_ensemble['Final Scores by Country']))))]
mean_onset = pd.DataFrame({'mean': means,'std': std_error})

fig,ax = plt.subplots(figsize=(12,8))

plt.scatter(mean_onset["mean"][0],mean_de["mean"][0],color="black",s=50)
plt.plot([mean_onset["mean"][0],mean_onset["mean"][0]],[mean_de["mean"][0]-mean_de["std"][0],mean_de["mean"][0]+mean_de["std"][0]],linewidth=3,color="black")
plt.plot([mean_onset["mean"][0]-mean_onset["std"][0],mean_onset["mean"][0]+mean_onset["std"][0]],[mean_de["mean"][0],mean_de["mean"][0]],linewidth=3,color="black")

plt.scatter(mean_onset["mean"][1],mean_de["mean"][1],color="black",s=50)
plt.plot([mean_onset["mean"][1],mean_onset["mean"][1]],[mean_de["mean"][1]-mean_de["std"][1],mean_de["mean"][1]+mean_de["std"][1]],linewidth=3,color="black")
plt.plot([mean_onset["mean"][1]-mean_onset["std"][1],mean_onset["mean"][1]+mean_onset["std"][1]],[mean_de["mean"][1],mean_de["mean"][1]],linewidth=3,color="black")

plt.scatter(mean_onset["mean"][2],mean_de["mean"][2],color="black",s=50)
plt.plot([mean_onset["mean"][2],mean_onset["mean"][2]],[mean_de["mean"][2]-mean_de["std"][2],mean_de["mean"][2]+mean_de["std"][2]],linewidth=3,color="black")
plt.plot([mean_onset["mean"][2]-mean_onset["std"][2],mean_onset["mean"][2]+mean_onset["std"][2]],[mean_de["mean"][2],mean_de["mean"][2]],linewidth=3,color="black")

plt.scatter(mean_onset["mean"][3],mean_de["mean"][3],color="black",s=50)
plt.plot([mean_onset["mean"][3],mean_onset["mean"][3]],[mean_de["mean"][3]-mean_de["std"][3],mean_de["mean"][3]+mean_de["std"][3]],linewidth=3,color="black")
plt.plot([mean_onset["mean"][3]-mean_onset["std"][3],mean_onset["mean"][3]+mean_onset["std"][3]],[mean_de["mean"][3],mean_de["mean"][3]],linewidth=3,color="black")

plt.scatter(mean_onset["mean"][4],mean_de["mean"][4],color="gray",s=50)
plt.plot([mean_onset["mean"][4],mean_onset["mean"][4]],[mean_de["mean"][4]-mean_de["std"][4],mean_de["mean"][4]+mean_de["std"][4]],linewidth=3,color="gray")
plt.plot([mean_onset["mean"][4]-mean_onset["std"][4],mean_onset["mean"][4]+mean_onset["std"][4]],[mean_de["mean"][4],mean_de["mean"][4]],linewidth=3,color="gray")

plt.ylabel("Difference explained (DE)")
plt.xlabel("Onset score, penalized by MSE")

plt.text(mean_onset["mean"][0]+0.004, mean_de["mean"][0]+0.004, "Shape finder", size=20, color='black')
plt.text(mean_onset["mean"][1]+0.004, mean_de["mean"][1]+0.004, "ViEWS", size=20, color='black')
plt.text(mean_onset["mean"][2]+0.004, mean_de["mean"][2]+0.004, "ZINB", size=20, color='black')
plt.text(mean_onset["mean"][3]+0.004, mean_de["mean"][3]+0.004, "Onset catcher", size=20, color='black')
plt.text(mean_onset["mean"][4]+0.004, mean_de["mean"][4]+0.004, "Ensemble", size=20, color='gray')

ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
ax.set_xticks([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4])

plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/procedural/out/proc_main.png",dpi=300,bbox_inches='tight')        
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/proc_main.png",dpi=300,bbox_inches='tight')
plt.savefig("out/proc_main.png",dpi=300,bbox_inches='tight')        
        

### By country ###
catcher_countries = pd.DataFrame({'country': catcher.country.unique(),'de_catcher': evals_catcher["DE by Country"],'onset_catcher': evals_catcher["Final Scores by Country"]})
views_countries = pd.DataFrame({'country': views.country.unique(),'de_views': evals_views["DE by Country"],'onset_views': evals_views["Final Scores by Country"]})
sf_countries = pd.DataFrame({'country': shape_finder.country.unique(),'de_sf': evals_shape_finder["DE by Country"],'onset_sf': evals_shape_finder["Final Scores by Country"]})

evals_country=pd.merge(catcher_countries,views_countries,on=["country"])
evals_country=pd.merge(evals_country,sf_countries,on=["country"])

summary = catcher.groupby('country').agg({'sb_fatalities': 'mean'})
#summary["sb_fatalities"]=np.log(summary["sb_fatalities"]+1)

fig,ax = plt.subplots(figsize=(12,8))
plt.scatter(evals_country["onset_catcher"],evals_country["de_catcher"],c="black",s=summary,alpha=0.6)
plt.scatter(evals_country["onset_views"],evals_country["de_views"],c="forestgreen",s=summary,alpha=0.6)
plt.scatter(evals_country["onset_sf"],evals_country["de_sf"],c="steelblue",s=summary,alpha=0.6)
plt.ylabel("Difference explained (DE)")
plt.xlabel("Onset score, penalized by MSE")


evals_country.loc[evals_country["country"]=="Russia (Soviet Union)","country"]="Russia"
evals_country.loc[evals_country["country"]=="Yemen (North Yemen)","country"]="Yemen"
evals_country.loc[evals_country["country"]=="Central African Republic","country"]="CAR"
evals_country.loc[evals_country["country"]=="DR Congo (Zaire)","country"]="DRC"

add=evals_country.sort_values(by=["onset_catcher"])[-1:]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))

#add=evals_country.sort_values(by=["onset_catcher"])[-2:-1]
#for i in range(len(add)):
#    point=add.iloc[i]
#    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]-0.17), ha='center',size=12,textcoords="offset points",xytext=(0,4))

add=evals_country.sort_values(by=["onset_catcher"])[-3:-2]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))

add=evals_country.sort_values(by=["onset_catcher"])[-4:-3]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))
 
add=evals_country.sort_values(by=["onset_catcher"])[-5:-4]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))

add=evals_country.sort_values(by=["onset_catcher"])[-6:-5]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))

add=evals_country.sort_values(by=["onset_catcher"])[-7:-6]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))

add=evals_country.sort_values(by=["onset_catcher"])[-8:-7]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))
                                       
add=evals_country.sort_values(by=["onset_catcher"])[-9:-8]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))
                                       
add=evals_country.sort_values(by=["onset_catcher"])[-10:-9]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))
                                                                                      
add=evals_country.sort_values(by=["onset_sf"])[-2:]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_sf"], point["de_sf"]), ha='center',size=12,textcoords="offset points",xytext=(0,4),color="steelblue")
                                                  
add=evals_country.sort_values(by=["onset_catcher"])[-11:-10]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))
                                                  
add=evals_country.sort_values(by=["onset_catcher"])[-12:-11]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))
                                                                
add=evals_country.sort_values(by=["onset_catcher"])[-13:-12]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))

add=evals_country.sort_values(by=["onset_catcher"])[-14:-13]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))
 
add=evals_country.sort_values(by=["onset_catcher"])[-15:-14]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))

add=evals_country.sort_values(by=["onset_catcher"])[-16:-15]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))
                                                                                         
add=evals_country.sort_values(by=["onset_catcher"])[-17:-16]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4))
                                                                                         
                                                                                       
add=evals_country.sort_values(by=["de_sf"])[-6:]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_sf"], point["de_sf"]), ha='center',size=12,textcoords="offset points",xytext=(0,4),color="steelblue")
                                                                                                               
add=evals_country.sort_values(by=["de_views"])[-6:]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_views"], point["de_views"]), ha='center',size=12,textcoords="offset points",xytext=(0,4),color="forestgreen")
                                                                                                               
add=evals_country.sort_values(by=["de_catcher"])[-6:]
for i in range(len(add)):
    point=add.iloc[i]
    plt.annotate(f'{point["country"]}', (point["onset_catcher"], point["de_catcher"]), ha='center',size=12,textcoords="offset points",xytext=(0,4),color="black")
                                                                                                               
                                                                                                         
plt.savefig("out/proc_scatter.png",dpi=300,bbox_inches='tight')    
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/procedural/out/proc_scatter.png",dpi=300,bbox_inches='tight')        
plt.savefig("/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/proc_scatter.png",dpi=300,bbox_inches='tight')    


########################
### Prediction plots ###
########################

catcher=pd.read_csv("out/catcher.csv",index_col=0)
df=pd.read_csv("data/data_out/ucdp_cm_sb.csv",index_col=0)
codes=df[["country","gw_codes"]].drop_duplicates()
catcher=pd.merge(left=catcher,right=codes,on=["country"],how="left")
catcher['year'] = catcher['dd'].str[:4]
catcher['year'] = catcher['year'].astype(int)

for c in views.gw_codes.unique():
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    df_s=views.loc[views["gw_codes"]==c]
    df_ss=shape_finder.loc[shape_finder["gw_codes"]==c]
    df_sss=catcher.loc[catcher["gw_codes"]==c]
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df_s["dd"].loc[df_s["year"]==2022],df_s["sb_fatalities"].loc[df_s["year"]==2022],color="black")
    #ax1.plot(df_s["dd"].loc[df_s["year"]==2022],df_s["preds"].loc[df_s["year"]==2022],color="black",linestyle="dotted")
    #ax1.plot(df_ss["dd"].loc[df_ss["year"]==2022],df_ss["preds"].loc[df_ss["year"]==2022],color="black",linestyle="dashed")   
    ax1.plot(df_sss["dd"].loc[df_sss["year"]==2022],df_sss["preds"].loc[df_sss["year"]==2022],color="black",marker="x",markersize=9,linestyle="dashed")   
    ax1.set_xticks(["2022-01","2022-03","2022-05","2022-07","2022-09","2022-11"],["01-22","03-22","05-22","07-22","09-22","11-22"])
    
    ax2 = fig.add_subplot(gs[1])    
    ax2.plot(df_s["dd"].loc[df_s["year"]==2023],df_s["sb_fatalities"].loc[df_s["year"]==2023],color="black")
    #ax2.plot(df_s["dd"].loc[df_s["year"]==2023],df_s["preds"].loc[df_s["year"]==2023],color="black",linestyle="dotted")
    #ax2.plot(df_ss["dd"].loc[df_ss["year"]==2023],df_ss["preds"].loc[df_ss["year"]==2023],color="black",linestyle="dashed")
    ax2.plot(df_sss["dd"].loc[df_sss["year"]==2023],df_sss["preds"].loc[df_sss["year"]==2023],color="black",marker="x",markersize=9,linestyle="dashed") 
    ax2.set_xticks(["2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],["01-23","03-23","05-23","07-23","09-23","11-23"])
    ax2.yaxis.set_ticks_position('right')
    ax2.yaxis.set_label_position('right')
    gs.update(wspace=0)    
    
    #fig.suptitle(views["country"].loc[views["gw_codes"]==c].iloc[0],size=30)
    if df_sss.preds.max()!=0:
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/procedural/out/proc_preds_{views['country'].loc[views['gw_codes']==c].iloc[0]}.png",dpi=300,bbox_inches='tight')        
        plt.savefig(f"/Users/hannahfrank/Dropbox/Apps/Overleaf/PhD_dissertation/out/proc_preds_{views['country'].loc[views['gw_codes']==c].iloc[0]}.png",dpi=300,bbox_inches='tight')
        plt.savefig(f"out/proc_preds_{views['country'].loc[views['gw_codes']==c].iloc[0]}.png",dpi=300,bbox_inches='tight')    
    plt.show()   






