import numpy as np
from scipy import stats, optimize, interpolate
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, LassoLarsCV, LassoLarsIC, RidgeCV, ElasticNetCV

import netCDF4 # module that reads in .nc files (built on top of HDF5 format)
import pandas as pd
import geopandas as gpd
from geopandas.tools import sjoin
import xarray
import rioxarray

from shapely.geometry import Point, mapping
from shapely.geometry.polygon import Polygon
from pyproj import CRS, Transformer # for transforming projected coordinates to elliptical coordinates
import cartopy.crs as ccrs # for defining and transforming coordinate systems
import cartopy.feature as cfeature # to add features to a cartopy map
import cartopy.io.shapereader as shpreader

#self-library
from fire_utils import seas_burnarea, clim_pred_var 

from datetime import datetime, timedelta
from cftime import num2date, date2num, DatetimeGregorian
from tqdm import tqdm
import matplotlib.pyplot as plt

# Specifying the paths for various data directories

data_dir= "../data"
pred_input_path= "/12km/"
resp_input_path= "/firelist/"
outfilepath= "../plots/"

def uni_lsq_regression_model(sum_burn_var, pred_file_indx, pred_seas_indx, regindx, freq_flag= False):
    
    # returns the predictions from the best-fit linear regression model and the correlation coefficient
     
    burnvar_len= len(sum_burn_var) # to account for differences between vegetation and burned area data set lengths
    pred_season_data= clim_pred_var(pred_file_indx= pred_file_indx, pred_seas_indx= pred_seas_indx, regindx= regindx, \
                                                                                                        burnarr_len= burnvar_len)
    
    if freq_flag:
        rescaled_burn_var= sum_burn_var
    else:
        rescaled_burn_var= np.log10(sum_burnarea)
        
    reg_pred= LinearRegression().fit(pred_season_data.reshape(-1, 1), rescaled_burn_var)
    pred_reg_burnvar= reg_pred.predict(pred_season_data.reshape(-1, 1))
    r_pred= reg_pred.score(pred_season_data.reshape(-1, 1), rescaled_burn_var)
    
    return pred_season_data, pred_reg_burnvar, r_pred

def reg_uni_climate_fire_corr(sum_burnarea, regression, regindx):
    
    # cycles through the ~9 predictor variables, inputs them from their respective files, and returns the predictions from the best-fit 
    # (univariate or multivariate) linear regression model and the correlation coefficient
    
    pred_var_arr= {1: r"Mar-Oct $T_{\rm max} \ $", 2: r"Mar-Oct VPD", 3: r"Mar-Oct Precip", 4: r"Antecedent precip", 5: r"Mar-Oct PET", 6: r"Forest", \
                   7: r"Deadbiomass litter", 8: r"Livebiomass leaf", 9: r"Mean connectivity"}
    pred_units_arr= {1: r"$\ [ ^\circ{\rm C}]$", 2: r"$\ [{\rm hPa}]$", 3: r"$\ [{\rm mm}]$", 4: r"$\ [{\rm mm}]$", 5: r"$\ [{\rm mm}]$", 6: None, \
                     7: r"$\ [{\rm kg}/{\rm ha}]$", 8: r"$\ [{\rm kg}/{\rm ha}]$", 9: None}
    
    reg_name= {1: "Sierra Nevada", 5: "Pacific NW Mts.", 8: "Middle Rockies", 11: "AZ/NM Mts.", 12: "IM Semidesert"}
    plt_name= {1: "ca_sierra", 2: "ca_north_coast", 3: "ca_cent_coast", 4: "ca_south_coast", 5: "pnw_mts", 6: "columbia_plateau", 7:"northern_rockies", \
              8: "middle_rockies", 9: "southern_rockies", 10: "ut_mts", 11: "aznm_mts", 12: "im_semidesert", 13: "im_desert", 14: "northern_great_plains", \
               15: "ca_total"}
    
    if regression == "uni_lsq":
        fig, ax= plt.subplots(3, 3, figsize=(14, 14));
        plt.suptitle(r'%s'%reg_name[regindx], fontsize= 18, y= 0.95);
        fig.text(0.04, 0.5, r'Burned area [in ${\rm km}^2$]', va='center', rotation='vertical', fontsize= 16);
        
        x_arr= np.arange(len(pred_var_arr))//3
        y_arr= np.arange(len(pred_var_arr))%3
    
    for i in tqdm(range(len(pred_var_arr))):
        
        pred_file_indx= i+1
        pred_seas_indx= i+1
        
        if regression == "uni_lsq":
            pred_season_data, logpred_reg_burnarea, r_pred= uni_lsq_regression_model(sum_burnarea, pred_file_indx, pred_seas_indx, regindx)
            
        ax[x_arr[i], y_arr[i]].plot(pred_season_data, np.log10(sum_burnarea), 'o', markersize= 10, 
                                                markerfacecolor= 'orchid', 
                                                markeredgecolor= 'orchid',
                                                linestyle= 'None');
        ax[x_arr[i], y_arr[i]].plot(pred_season_data, logpred_reg_burnarea, color= 'black', lw= 2, label=r'$r = %.2f$'%np.sqrt(r_pred));
            
        ax[x_arr[i], y_arr[i]].set_ylim(.8, 4.0);
        if y_arr[i] > 0:
            ax[x_arr[i], y_arr[i]].set_yticklabels([]);
            
        if pred_units_arr[i+1] == None:
            ax[x_arr[i], y_arr[i]].set_xlabel('%s'%pred_var_arr[i+1], fontsize= 14);
        else:
            ax[x_arr[i], y_arr[i]].set_xlabel('%s'%pred_var_arr[i+1] +  '%s'%pred_units_arr[i+1], fontsize= 14);
        #ax[0, 0].set_ylabel(r'${\rm log}_{10}$ (Burned area) [in ${\rm km}^2$]', fontsize= 14);
            
        fig.subplots_adjust(hspace= 0.3)
        ax[x_arr[i], y_arr[i]].tick_params(labeltop=False, top=True, labelright=False, right=True, which='both', labelsize= 12);
        ax[x_arr[i], y_arr[i]].grid(b=True, which='major', color='black', alpha=0.05, linestyle='-');
        ax[x_arr[i], y_arr[i]].grid(b=True, which='minor', color='black', alpha=0.01, linestyle='-');
        ax[x_arr[i], y_arr[i]].legend(loc='best', frameon=True, fontsize=12);
        
    #plt.savefig(outfilepath + '%s_'%plt_name[regindx] + 'climate_fire_corr.pdf', bbox_inches='tight');
    
def multi_regression_model(sum_burn_var, regression, regindx, freq_flag= False):
    
    # returns the normalized coefficients for a multivariate regression model
    
    pred_var_arr= ["Tmax", "VPD", "Precip", "Antecedent precip", "PET", "Forest"]
    burnvar_len= len(sum_burn_var)
    
    #print("Calculating the normalized climate-fire correlation coefficients for %s ..."%reg_name[regindx])
    
    pred_season_data_arr= [clim_pred_var(pred_file_indx= i+1, pred_seas_indx= i+1, regindx= regindx, burnarr_len= burnvar_len) \
                                                                                                        for i in range(len(pred_var_arr))]
    des_mat= np.vstack(pred_season_data_arr).T
    std_arr= np.std(des_mat, axis= 0) #np.sqrt(np.sum(des_mat ** 2, axis=0))
    norm_des_mat= (des_mat - np.mean(des_mat, axis= 0))/std_arr
    if freq_flag:
        rescaled_burn_var= (sum_burn_var - np.mean(sum_burn_var))/np.std(sum_burn_var)
    else:
        rescaled_burn_var= (np.log10(sum_burn_var) - np.mean(np.log10(sum_burn_var)))/(np.std(np.log10(sum_burn_var)))
    
    if regression == "lassoCV":
        model= LassoCV(alphas= [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], eps= 1e-4, cv= 10).fit(norm_des_mat, rescaled_burn_var)
        hyperparam_val= model.alpha_
    elif regression == "enetCV":
        model= ElasticNetCV(l1_ratio= [0.01, 0.1, 0.3, 0.7, 0.9, 0.99], eps= 1e-4, n_alphas= 1000, cv= 10).fit(norm_des_mat, rescaled_burn_var)
        hyperparam_val= model.l1_ratio_

    r2_mod= model.score(norm_des_mat, rescaled_burn_var)
    coeff_arr= model.coef_ #*std_arr
    
    return coeff_arr, r2_mod, hyperparam_val

def reg_multi_climate_fire_corr(firefile, fireseason, regression):
    
    reg_name= {1: r"Sierra Nevada", 5: r"Pacific NW Mts.", 8: r"Middle Rockies", 11: r"AZ/NM Mts.", 12: r"IM Semidesert", 13: r"IM Desert"}
    reg_keys= list(reg_name.keys())
    
    fig, ax= plt.subplots(2, 3, figsize=(16, 14));
    #plt.suptitle(r"", fontsize= 18, y= 0.95);
    fig.text(0.04, 0.5, r'Burned area [in ${\rm km}^2$]', va='center', rotation='vertical', fontsize= 16);
    x_arr= np.arange(len(reg_name))//3
    y_arr= np.arange(len(reg_name))%3
    pred_var_arr= ["Tmax", "VPD", "Precip", "Antecedent precip", "PET", "Forest"]
    ypos= np.arange(len(pred_var_arr))
    
    for i in tqdm(range(len(reg_name))):
        
        print("Calculating the normalized climate-fire correlation coefficients for %s ..."%reg_name[reg_keys[i]])
        burnarea_arr= seas_burnarea(firefile, fireseason, regindx= reg_keys[i])
        coeff_arr, r2_mod, hyperparam_val= multi_regression_model(burnarea_arr, regression, regindx= reg_keys[i])
        
        ax[x_arr[i], y_arr[i]].barh(ypos, coeff_arr, align= "center");
        ax[x_arr[i], y_arr[i]].set_xlim(-1.2, 1.2);
        ax[x_arr[i], y_arr[i]].set_xlabel(r"Normalized coefficients", fontsize= 14);
        ax[x_arr[i], y_arr[i]].set_yticks(ypos);
        if y_arr[i] > 0:
            ax[x_arr[i], y_arr[i]].set_yticklabels([]);
        else:
            ax[x_arr[i], y_arr[i]].set_yticklabels(pred_var_arr, fontsize= 14);
            
        fig.subplots_adjust(hspace= 0.4, wspace= 0.4)
        ax[x_arr[i], y_arr[i]].tick_params(labeltop=False, top=True, labelright=False, right=True, which='both', labelsize= 12);
        ax[x_arr[i], y_arr[i]].grid(b=True, which='major', color='black', alpha=0.05, linestyle='-');
        ax[x_arr[i], y_arr[i]].grid(b=True, which='minor', color='black', alpha=0.01, linestyle='-');
        ax[x_arr[i], y_arr[i]].set_title("%s"%reg_name[reg_keys[i]] + r"$ \ ({\rm R}^2 = %.2f)$"%r2_mod, fontsize= 16);
    
    #plt.savefig(outfilepath + '%s_'%regression + 'climate_fire_coeffs.pdf', bbox_inches='tight');