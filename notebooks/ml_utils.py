import numpy as np
import pandas as pd
#from time import clock
from datetime import date, datetime, timedelta
from cftime import num2date, date2num, DatetimeGregorian
from tqdm import tqdm
from copy import deepcopy

#Import and write files
import csv
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

#self-libraries
from fire_utils import ncdump, coord_transform, bailey_ecoprovince_shp, bailey_ecoprovince_mask, update_reg_indx, mon_fire_freq, mon_burned_area, tindx_func, clim_pred_var, init_eff_clim_fire_df 
from stats_utils import uni_lsq_regression_model, multi_regression_model

#Helper functions
from math import factorial
from scipy.special import gamma
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from tensorflow.python import ops
from tensorflow.python import debug as tf_debug

#Plot modules
import matplotlib.pyplot as plt

#Stats modules
from scipy import stats
from scipy.stats import norm, pareto, genpareto
import statsmodels.api as sm
from scipy import stats, interpolate
from scipy.optimize import minimize

# Data processing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, SplineTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

#modules for Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression

#modules for Neural Network
import tensorflow as tf
import tensorflow_probability as tfp
tfd= tfp.distributions
from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import regularizers

## ----------------------------------------------------------------- MDN functions ----------------------------------------------------------------------------

def zinb_model(parameter_vector):
    
    pi, mu, delta= tf.split(parameter_vector, 3, axis= 1)
    mu= tf.squeeze(mu)
    delta= tf.squeeze(delta)
    
    p_rescaled= 1 - mu/(mu + delta) #1 - tf.exp(mu)/(tf.exp(mu) + delta) #also play with normal mu
    probs_tf = tf.concat([pi, 1-pi], axis=1)
    
    zinb_mix= tfd.Mixture(cat=tfd.Categorical(probs= probs_tf),
    components=[tfd.Deterministic(loc= tf.zeros_like(mu)), tfd.NegativeBinomial(total_count= delta, probs= p_rescaled),])
    
    return zinb_mix

def zinb_loss(y, parameter_vector):
    
    zinb_mix= zinb_model(parameter_vector)
    log_likelihood= zinb_mix.log_prob(tf.transpose(y))

    return(-tf.reduce_mean(log_likelihood, axis= -1))

def zinb_accuracy(y, parameter_vector):
    
    y= tf.squeeze(y)
    zinb_mix= zinb_model(parameter_vector)    
    cdf_mod= zinb_mix.cdf(y)
    
    empcdf= tfd.Empirical(y)
    cdf_emp= empcdf.cdf(y)
    
    err= 100 * (-tf.reduce_mean(tf.math.log(cdf_mod/cdf_emp), axis= -1))
    
    return(100 - tf.abs(err))

def zipd_model(parameter_vector):
    
    pi, mu, delta= tf.split(parameter_vector, 3, axis= 1)
    mu= tf.squeeze(mu)
    delta= tf.squeeze(delta)
    
    rateparam= tf.exp(mu + delta)
    probs_tf = tf.concat([pi, 1-pi], axis=1)
    
    zipd_mix= tfd.Mixture(
    cat=tfd.Categorical(probs= probs_tf),
    components=[tfd.Deterministic(loc= tf.zeros_like(mu)), tfd.Poisson(rate= rateparam),])
    
    return zipd_mix

def zipd_model_shap(mdn_mod, X):
    
    parameter_vector= mdn_mod.predict(x= X)
    pi, mu, delta= tf.split(parameter_vector, 3, axis= 1)
    mu= tf.squeeze(mu)
    delta= tf.squeeze(delta)
    
    rateparam= tf.exp(mu + delta)
    probs_tf= tf.concat([pi, 1-pi], axis=1) #tf.stack([pi, 1-pi], -1)
    
    zipd_mix= tfd.Mixture(
    cat=tfd.Categorical(probs= tf.squeeze(probs_tf)),
    components=[tfd.Deterministic(loc= tf.zeros_like(mu)), tfd.Poisson(rate= rateparam),])
    
    return pd.Series(tf.cast(tf.reduce_mean(zipd_mix.sample(1000, seed= 99), axis= 0), tf.int64).numpy())

def zipd_loss(y, parameter_vector, cdf_flag= False):
    
    zipd_mix= zipd_model(parameter_vector)
    log_likelihood= zipd_mix.log_prob(tf.transpose(y))

    return(-tf.reduce_mean(log_likelihood, axis= -1))

def zipd_accuracy(y, parameter_vector):
    
    y= tf.squeeze(y)
    zipd_mix= zipd_model(parameter_vector)
    cdf_mod= zipd_mix.cdf(y)
    
    empcdf= tfd.Empirical(y)
    cdf_emp= empcdf.cdf(y)
    
    err= 100 * (-tf.reduce_mean(tf.math.log(cdf_mod/cdf_emp), axis= -1))
    
    return(100 - tf.abs(err))

def gpd_model(parameter_vector):
    
    alpha, scale, shape= tf.split(parameter_vector, 3, axis= 1)
    loc_arr= tf.zeros_like(scale)
    
    gpd_mix= tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.GeneralizedPareto(
            loc=loc_arr, scale= scale, concentration= shape))
    
    return gpd_mix

def gpd_loss(y, parameter_vector):
    
    gpd_mix= gpd_model(parameter_vector)
    log_likelihood= gpd_mix.log_prob(tf.transpose(y))

    return(-tf.reduce_mean(log_likelihood, axis= -1))

def gpd_accuracy(y, parameter_vector):
    
    gpd_mix= gpd_model(parameter_vector)
    cdf_mod= gpd_mix.cdf(tf.transpose(y))
    
    empcdf= tfd.Empirical(tf.transpose(y))
    cdf_emp= empcdf.cdf(tf.transpose(y))
    
    err= 100 * (-tf.reduce_mean(tf.math.log(cdf_mod/cdf_emp), axis= -1))
    #print(err.shape)
    
    return(100 - tf.abs(err))

def lognorm_model(parameter_vector):
    
    alpha, mu, sigma= tf.split(parameter_vector, 3, axis= 1)
    
    lognorm_mix= tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.LogNormal(
            loc= mu, scale= sigma))
    
    return lognorm_mix

def lognorm_loss(y, parameter_vector):
    
    lognorm_mix= lognorm_model(parameter_vector)
    log_likelihood= lognorm_mix.log_prob(tf.transpose(y))

    return(-tf.reduce_mean(log_likelihood, axis= -1))

def lognorm_accuracy(y, parameter_vector):
    
    lognorm_mix= lognorm_model(parameter_vector)
    cdf_mod= lognorm_mix.cdf(tf.transpose(y))
    
    empcdf= tfd.Empirical(tf.transpose(y))
    cdf_emp= empcdf.cdf(tf.transpose(y))
    
    err= 100 * (-tf.reduce_mean(tf.math.log(cdf_mod/cdf_emp), axis= -1))
    #print(err.shape)
    
    return(100 - tf.abs(err))

def lognorm_gpd_model(parameter_vector):
    
    alpha, scale, shape= tf.split(parameter_vector, 3, axis= 1)
    mu_lognorm, scale_gpd= tf.split(scale, 2, axis= 1)
    sigma_lognorm, shape_gpd= tf.split(shape, 2, axis= 1)
    
    xmin= 0 #threshold burned area in km^2
    loc_arr= tf.ones_like(scale_gpd)*xmin
    
    lognorm_gpd_mix= tfd.Mixture(
    cat=tfd.Categorical(probs= alpha),
    components=[tfd.LogNormal(loc= tf.squeeze(mu_lognorm), scale= tf.squeeze(sigma_lognorm)), 
                tfd.GeneralizedPareto(loc= tf.squeeze(loc_arr), scale= tf.squeeze(scale_gpd), concentration= tf.squeeze(shape_gpd)),])
    
    return lognorm_gpd_mix

def lognorm_gpd_model_predict(parameter_vector):
    
    alpha, scale, shape= tf.split(parameter_vector, 3, axis= 1)
    mu_lognorm, scale_gpd= tf.split(scale, 2, axis= 1)
    sigma_lognorm, shape_gpd= tf.split(shape, 2, axis= 1)
    
    xmin= 0 #threshold burned area in km^2
    loc_arr= tf.ones_like(scale_gpd)*xmin
    
    lognorm_gpd_mix= tfd.Mixture(
    cat=tfd.Categorical(probs= tf.squeeze(alpha)),
    components=[tfd.LogNormal(loc= tf.squeeze(mu_lognorm), scale= tf.squeeze(sigma_lognorm)), 
                tfd.GeneralizedPareto(loc= tf.squeeze(loc_arr), scale= tf.squeeze(scale_gpd), concentration= tf.squeeze(shape_gpd)),])
    
    return lognorm_gpd_mix

def lognorm_gpd_loss(y, parameter_vector):
    
    lognorm_gpd_mix= lognorm_gpd_model(parameter_vector)
    log_likelihood= lognorm_gpd_mix.log_prob(tf.transpose(y))

    return(-tf.reduce_mean(log_likelihood, axis= -1))

def lognorm_gpd_accuracy(y, parameter_vector):
    
    lognorm_gpd_mix= lognorm_gpd_model(parameter_vector)
    cdf_mod= lognorm_gpd_mix.cdf(tf.transpose(y))
    
    empcdf= tfd.Empirical(tf.transpose(y))
    cdf_emp= empcdf.cdf(tf.transpose(y))
    
    err= 100 * (-tf.reduce_mean(tf.math.log(cdf_mod/cdf_emp), axis= -1))
    #print(err.shape)
    
    return(100 - tf.abs(err))

class SeqBlock(tf.keras.layers.Layer):
    
    def __init__(self, hidden_l= 2, n_neurs=100, initializer= "glorot_uniform", reg= False, regrate= None, dropout= False):
        super(SeqBlock, self).__init__(name="SeqBlock")
        self.nnmodel= tf.keras.Sequential()
        for l in range(hidden_l):
            if reg:
                self.nnmodel.add(Dense(n_neurs, activation="relu",
                                    kernel_initializer= initializer,
                                    kernel_regularizer=tf.keras.regularizers.l2(regrate),
                                    name="h_%d"%(l+1))) 
            else:
                self.nnmodel.add(Dense(n_neurs, activation="relu",
                                    kernel_initializer= initializer, 
                                    name="h_%d"%(l+1)))
            #self.nnmodel.add(tf.keras.layers.LayerNormalization())
            if dropout:
                self.nnmodel.add(tf.keras.layers.Dropout(0.3))

    def call(self, inputs):
        return self.nnmodel(inputs)
    
class MDN_size(tf.keras.Model):

    def __init__(self, layers= 2, neurons=10, components = 2, initializer= "glorot_uniform", reg= False, regrate= None, dropout= False):
        super(MDN_size, self).__init__(name="MDN_size")
        self.neurons = neurons
        self.components = components
        self.n_hidden_layers= layers
        
        #hidden layers
        self.seqblock= SeqBlock(layers, neurons, initializer, reg, regrate, dropout)
        
        #output layer
        if reg:
            self.outlayer= Dense(3*components, activation="relu",
                                    kernel_initializer= initializer,
                                    kernel_regularizer=tf.keras.regularizers.l2(regrate),
                                    name="output_layer")
        else:
            self.outlayer= Dense(3*components, activation="relu",
                                    kernel_initializer= initializer,
                                    name="output_layer")
        #self.linreg= Dense(neurons, activation="linear", kernel_initializer= initializer, name="linear")
        
        self.alphas = Dense(components, activation="softmax", kernel_initializer= initializer, name="alphas")
        self.distparam1 = Dense(components, activation="softplus", kernel_initializer= initializer, name="distparam1")
        self.distparam2 = Dense(components, activation="softplus", kernel_initializer= initializer, name="distparam2")
        self.pvec = Concatenate(name="pvec")
        
    def call(self, inputs):
        x = self.outlayer(self.seqblock(inputs)) # + self.linreg(inputs)
        #x0, x1, x2= tf.split(x, 3, axis= 1)
        
        alpha_v = self.alphas(x) 
        distparam1_v = self.distparam1(x)
        distparam2_v = self.distparam2(x)
        
        return self.pvec([alpha_v, distparam1_v, distparam2_v])

class MDN_freq(tf.keras.Model):

    def __init__(self, layers= 2, neurons=10, components = 1, initializer= "glorot_uniform", reg= False, regrate= None, dropout= False, func_type= 'zinb'):
        super(MDN_freq, self).__init__(name="MDN_freq")
        self.neurons = neurons
        self.components = components
        self.n_hidden_layers= layers
        
        #hidden layers
        self.seqblock= SeqBlock(layers, neurons, initializer, reg, regrate, dropout)
        
        #output layer
        if reg:
            self.outlayer= Dense(3*components, activation="relu",
                                    kernel_initializer= initializer,
                                    kernel_regularizer=tf.keras.regularizers.l2(regrate),
                                    name="output_layer")
        else:
            self.outlayer= Dense(3*components, activation="relu",
                                    kernel_initializer= initializer,
                                    name="output_layer")
        #self.linreg= Dense(neurons, activation="linear", kernel_initializer= initializer, name="linear")
        
        self.pi = Dense(components, activation="sigmoid", kernel_initializer= initializer, name="pi")
        self.mu = Dense(components, activation="softplus", kernel_initializer= initializer, name="mu")
        if func_type == 'zipd':
            self.delta= Dense(components, activation="gelu", kernel_initializer= initializer, name="delta") #ensures gaussian error in delta
        else:
            self.delta= Dense(components, activation="softplus", kernel_initializer= initializer, name="delta")
        self.pvec = Concatenate(name="pvec")
        
    def call(self, inputs):
        x = self.outlayer(self.seqblock(inputs)) #+ self.linreg(inputs)
        #x0, x1, x2= tf.split(x, 3, axis= 1)
        
        pi_v = self.pi(x) 
        mu_v = self.mu(x)
        delta_v = self.delta(x)
        
        return self.pvec([pi_v, mu_v, delta_v])

class DNN(tf.keras.Model):
    def __init__(self, layers= 2, neurons= 16, initializer= "he_normal", reg= False, regrate= None, dropout= False):
        super(DNN, self).__init__(name="DNN")
        self.neurons= neurons
        self.n_hidden_layers= layers
        
        self.seqblock= SeqBlock(layers, neurons, initializer, reg, regrate, dropout)
        self.out = Dense(1, activation="sigmoid", name="out")
        
    def call(self, inputs):
        x = self.seqblock(inputs)
        return self.out(x)

def hyperparam_tuning(n_layers, n_neurons, n_components= None, bs= 128, epochs= 1000, lr= 1e-4, X_dat= None, y_dat= None, fire_tag= 'size', func_flag= 'gpd', samp_weights= False, samp_weight_arr= None):
    
    # Function for tuning the hyperparamters of the MDNs to determine fire properties
    
    opt= tf.keras.optimizers.Adam(learning_rate= lr)
    if func_flag == 'gpd':
        loss_metric= gpd_loss
        acc_metric= gpd_accuracy
    elif func_flag == 'lognorm':
        loss_metric= lognorm_loss
        acc_metric= lognorm_accuracy
    elif func_flag == 'zinb':
        if fire_tag == 'size':
            print("This setting only works for determining fire frequency")
        loss_metric= zinb_loss
        acc_metric= zinb_accuracy
    elif func_flag == 'zipd':
        if fire_tag == 'size':
            print("This setting only works for determining fire frequency")
        loss_metric= zipd_loss
        acc_metric= zipd_accuracy
    
    list_of_lists = []
    
    if fire_tag == 'size':
        for i in tqdm(range(len(n_layers))):
            print("Constructing a MDN for fire %s"%fire_tag + " w/ %d hidden layers"%n_layers[i])
            for nn in n_neurons:
                for c in n_components:
                    hp= MDN_size(layers= n_layers[i], neurons= nn, components= c)
                    hp.compile(loss=loss_metric, optimizer=opt, metrics=[acc_metric])
                    hp.fit(x=X_dat, y=y_dat, batch_size= bs, epochs= epochs, verbose=0)

                    loss, accuracy= hp.evaluate(X_dat, y_dat, verbose=0)
                    list_of_lists.append([c, n_layers[i], nn, loss, accuracy])
                    hp= hp.reset_states()
        
        hp_df= pd.DataFrame(list_of_lists, columns=["n_components", "n_layers", "n_neurons", "Loss", "Accuracy"])

    elif fire_tag == 'freq':
        for i in tqdm(range(len(n_layers))):
            print("Constructing a MDN for fire %s"%fire_tag + " w/ %d hidden layers"%n_layers[i])
            for nn in n_neurons:
                    hp= MDN_freq(layers= n_layers[i], neurons= nn)
                    hp.compile(loss=loss_metric, optimizer=opt, metrics=[acc_metric])
                    if samp_weights == False:
                        hp.fit(x=X_dat, y=y_dat, batch_size= bs, epochs= epochs, verbose=0)
                    else:
                        hp.fit(x=X_dat, y=y_dat, batch_size= bs, epochs= epochs, sample_weight= samp_weight_arr, verbose=0)

                    loss, accuracy= hp.evaluate(X_dat, y_dat, verbose=0)
                    list_of_lists.append([n_layers[i], nn, loss, accuracy])
                    hp= hp.reset_states()
        
        hp_df= pd.DataFrame(list_of_lists, columns=["n_layers", "n_neurons", "Loss", "Accuracy"])                 

    hp_df.to_hdf('../sav_files/%s_'%fire_tag + '%s_'%func_flag +'hp_tuning.h5', key='df', mode= 'w')
    
    return hp_df


def validation_cycle(n_layers, n_neurons, n_components= None, num_iterations= 5, bs= 128, epochs= 100, lr= 1e-4, X_dat= None, y_dat= None, X_val_dat= None, y_val_dat= None, fire_tag= 'size', func_flag= 'gpd', samp_weights= False, samp_weight_arr= None):

    # Function for calculating training and validation accuracy over multiple iterations
    
    #tf.random.set_seed(99) --> seed for MDN_size

    Acc_List = [] #metric per epoch
    Val_Acc_List = []
    Loss_List = []
    Val_Loss_List = []

    Accuracy_train = [] #metric per iteration
    Accuracy_val = []
    Loss_train = []
    Loss_val = []
    
    opt= tf.keras.optimizers.Adam(learning_rate= lr)
    if func_flag == 'gpd':
        loss_metric= gpd_loss
        acc_metric= gpd_accuracy
    elif func_flag == 'lognorm':
        loss_metric= lognorm_loss
        acc_metric= lognorm_accuracy
    elif func_flag == 'zinb':
        if fire_tag == 'size':
            print("This setting only works for determining fire frequency")
        loss_metric= zinb_loss
        acc_metric= zinb_accuracy
    elif func_flag == 'zipd':
        if fire_tag == 'size':
            print("This setting only works for determining fire frequency")
        loss_metric= zipd_loss
        acc_metric= zipd_accuracy

    for i in tqdm(range(num_iterations)):
        print("Validation iteration %d"%(i+1) + " for fire %s MDN with "%fire_tag + "%s loss function"%func_flag)
        if fire_tag == 'size':
            mdn_val= MDN_size(layers= n_layers, neurons= n_neurons, components= n_components, initializer= 'he_normal')
        elif fire_tag == 'freq':
            mdn_val= MDN_freq(layers= n_layers, neurons= n_neurons, initializer= 'he_normal')
        
        mdn_val.compile(loss= loss_metric, optimizer= opt, metrics=[acc_metric])
        if samp_weights == False:
            mdnhist= mdn_val.fit(x= X_dat, y= y_dat, batch_size= bs, epochs= epochs, validation_data=(X_val_dat, y_val_dat), verbose=0)
        else:
            mdnhist= mdn_val.fit(x= X_dat, y= y_dat, batch_size= bs, epochs= epochs, validation_data=(X_val_dat, y_val_dat), sample_weight= samp_weight_arr, verbose=0)
        loss_list, acc_list, val_loss_list, val_acc_list= [mdnhist.history[k] for k in mdnhist.history.keys()]

        Acc_List.append(acc_list)
        Val_Acc_List.append(val_acc_list)
        Loss_List.append(loss_list)
        Val_Loss_List.append(val_loss_list)

        loss_train, acc_train= mdn_val.evaluate(X_dat, y_dat, verbose= 0)
        loss_val, acc_val= mdn_val.evaluate(X_val_dat, y_val_dat, verbose= 0)
        Accuracy_train.append(acc_train)
        Accuracy_val.append(acc_val)
        Loss_train.append(loss_train)
        Loss_val.append(loss_val)

        mdn_val= mdn_val.reset_states()
    
    return Acc_List, Val_Acc_List, Loss_List, Val_Loss_List, Accuracy_train, Accuracy_val, Loss_train, Loss_val

## ----------------------------------------------------------------- Fire freq functions ----------------------------------------------------------------------------

def fire_freq_data(fire_freq_df, dropcols= ['index', 'Tmin', 'Solar', 'Ant_Tmax', 'RH', 'Ant_RH', 'Elev', 'Camp_dist']): 
    
    # Returns the train/val/test data given an initial fire frequency df
    
    fire_freq_train= fire_freq_df[fire_freq_df.month < 372].reset_index().drop(columns=['index']) # for Training and Testing; 372 ==> Jan. 2015; ensures ~80/20 split
    fire_freq_test= fire_freq_df[fire_freq_df.month >= 372].reset_index().drop(columns=['index']) # for Prediction
    tmp_freq_df= fire_freq_df[fire_freq_df.iloc[:, 0:22].columns] #CAUTION: number changes when adding new variables
    X_freq_df= pd.DataFrame({})
    scaler= StandardScaler().fit(fire_freq_train.iloc[:, 0:22])
    X_freq_df[tmp_freq_df.columns]= scaler.transform(tmp_freq_df)

    X_freq_train_df= X_freq_df.iloc[0:len(fire_freq_train)].reset_index().drop(columns= dropcols)
    y_freq_train= np.array(fire_freq_train.fire_freq, dtype=np.float32)

    X_freqs_test= X_freq_df.iloc[-len(fire_freq_test):].reset_index().drop(columns= dropcols) 
    y_freqs_test= np.array(fire_freq_test.fire_freq, dtype=np.float32)

    #splitting only the training data set
    X_freqs_train, X_freqs_val, y_freqs_train, y_freqs_val = train_test_split(X_freq_train_df, y_freq_train, test_size=0.2, random_state=87)
    freq_samp_weight_arr= fire_freq_train.iloc[X_freqs_train.index]['sample_weight'].to_numpy()
    
    return X_freqs_train, X_freqs_val, y_freqs_train, y_freqs_val, fire_freq_test, X_freqs_test, y_freqs_test, freq_samp_weight_arr

def freq_pred_func(mdn_model, X_test_dat, func_flag= 'zinb', l4_flag= False, reg_len_arr= None, modsave= False, nsig= 2):
    
    # Calculates the mean and 2 sigma uncertainties of fire frequencies given a NN model
    
    n_regions= 18
    if func_flag == 'zinb':
        stat_model= zinb_model
    elif func_flag == 'zipd':
        stat_model= zipd_model
    
    if not l4_flag:
        freq_test_size= np.int(len(X_test_dat)/n_regions)
        freq_arr_1= np.linspace(0, len(X_test_dat) - freq_test_size, n_regions, dtype= int)
        freq_arr_2= freq_arr_1 + freq_test_size
        
        reg_freq_df= pd.DataFrame({'mean_freq': pd.Series(dtype= 'int'), 'low_%dsig_freq'%nsig: pd.Series(dtype= 'int'), 'high_%dsig_freq'%nsig: pd.Series(dtype= 'int'), 'reg_indx': pd.Series(dtype= 'int')})
        if not modsave:
            for i in tqdm(range(n_regions)): 
                param_vec= mdn_model.predict(x= tf.constant(X_test_dat[freq_arr_1[i]:freq_arr_2[i]]))
                freq_samp= stat_model(param_vec).sample(10000, seed= 99)
                reg_freq= tf.cast(tf.reduce_mean(freq_samp, axis= 0), tf.int64)
                reg_freq_sig= tf.cast(tf.math.reduce_std(freq_samp, axis= 0), tf.int64)
                #reg_freq_med= tfp.stats.percentile(sierra_freq_samp, 50.0, interpolation='midpoint', axis= 0)

                reg_freq_low= (reg_freq - nsig*reg_freq_sig).numpy()
                reg_freq_low[reg_freq_low < 0]= 0
                reg_freq_high= (reg_freq + nsig*reg_freq_sig).numpy()
                reg_indx_arr= (i+1)*np.ones(len(reg_freq), dtype= int)

                reg_freq_df= reg_freq_df.append(pd.DataFrame({'mean_freq': reg_freq.numpy(), 'low_%dsig_freq'%nsig: reg_freq_low, 'high_%dsig_freq'%nsig: reg_freq_high,'reg_indx': reg_indx_arr}), ignore_index=True)
        else:
            for i in range(n_regions): 
                param_vec= mdn_model.predict(x= tf.constant(X_test_dat[freq_arr_1[i]:freq_arr_2[i]]))
                freq_samp= stat_model(param_vec).sample(10000, seed= 99)
                reg_freq= tf.cast(tf.reduce_mean(freq_samp, axis= 0), tf.int64)
                reg_freq_sig= tf.cast(tf.math.reduce_std(freq_samp, axis= 0), tf.int64)
                #reg_freq_med= tfp.stats.percentile(sierra_freq_samp, 50.0, interpolation='midpoint', axis= 0)

                reg_freq_low= (reg_freq - 2*reg_freq_sig).numpy()
                reg_freq_low[reg_freq_low < 0]= 0
                reg_freq_high= (reg_freq + 2*reg_freq_sig).numpy()
                reg_indx_arr= (i+1)*np.ones(len(reg_freq), dtype= int)

                reg_freq_df= reg_freq_df.append(pd.DataFrame({'mean_freq': reg_freq.numpy(), 'low_%dsig_freq'%nsig: reg_freq_low, 'high_%dsig_freq'%nsig: reg_freq_high,'reg_indx': reg_indx_arr}), ignore_index=True) 
        
    else:
        cumlenarr= np.insert(np.cumsum(reg_len_arr), 0, 0)
        reg_freq_df= pd.DataFrame({'mean_freq': pd.Series(dtype= 'int'), 'std_freq': pd.Series(dtype= 'int'), 'reg_indx': pd.Series(dtype= 'int')})
        if not modsave:
            for i in tqdm(range(n_regions)): 
                param_vec= mdn_model.predict(x= tf.constant(X_test_dat[cumlenarr[i]:cumlenarr[i+1]]))
                freq_samp= stat_model(param_vec).sample(10000, seed= 99)
                reg_freq= tf.cast(tf.reduce_mean(freq_samp, axis= 0), tf.int64)
                reg_freq_sig= tf.cast(tf.math.reduce_std(freq_samp, axis= 0), tf.int64)

                reg_indx_arr= (i+1)*np.ones(len(reg_freq), dtype= int)

                reg_freq_df= reg_freq_df.append(pd.DataFrame({'mean_freq': reg_freq.numpy(), 'std_freq': reg_freq_sig.numpy(), 'reg_indx': reg_indx_arr}), \
                                                                                                                                            ignore_index=True)
        else:
            for i in range(n_regions): #n_regions
                param_vec= mdn_model.predict(x= tf.constant(X_test_dat[cumlenarr[i]:cumlenarr[i+1]]))
                freq_samp= stat_model(param_vec).sample(10000, seed= 99)
                reg_freq= tf.cast(tf.reduce_mean(freq_samp, axis= 0), tf.int64)
                reg_freq_sig= tf.cast(tf.math.reduce_std(freq_samp, axis= 0), tf.int64)
                #reg_freq_med= tfp.stats.percentile(sierra_freq_samp, 50.0, interpolation='midpoint', axis= 0)

                reg_indx_arr= (i+1)*np.ones(len(reg_freq), dtype= int)

                reg_freq_df= reg_freq_df.append(pd.DataFrame({'mean_freq': reg_freq.numpy(), 'std_freq': reg_freq_sig.numpy(), 'reg_indx': reg_indx_arr}), \
                                                                                                                                                ignore_index=True)
            
    return reg_freq_df

def reg_fire_freq_func(X_train_dat, y_train_dat, X_val_dat, y_val_dat, X_test_dat, lr= 1e-4, n_layers= 2, n_neurons= 16, epochs= 500, bs= 32, func_flag= 'zinb', rseed= None, samp_weights= False, samp_weight_arr= None, modsave= False):
    
    # Calculates the predicted fire frequency as well as its 1 sigma uncertainty for all regions
    
    if rseed == None:
        rseed= np.random.randint(100)
    tf.random.set_seed(rseed)

    if func_flag == 'zinb':
        stat_model= zinb_model
        loss_func= zinb_loss
        acc_func= zinb_accuracy
    elif func_flag == 'zipd':
        stat_model= zipd_model
        loss_func= zipd_loss
        acc_func= zipd_accuracy
    
    print("Initialized a MDN with %d layers"%n_layers + " and %d neurons"%n_neurons)

    mon= EarlyStopping(monitor='val_loss', min_delta=0, patience= 10, verbose=0, mode='auto')
    mdn= MDN_freq(layers= n_layers, neurons= n_neurons)
    mdn.compile(loss=loss_func, optimizer= tf.keras.optimizers.Adam(learning_rate= lr), metrics=[acc_func])
    #h= mdn.fit(x= X_train_dat, y= y_train_dat, epochs= epochs, validation_data=(X_val_dat, y_val_dat), callbacks= [mon], batch_size= bs, verbose=0)
    if samp_weights == False:
        h= mdn.fit(x= X_train_dat, y= y_train_dat, epochs= epochs, validation_data=(X_val_dat, y_val_dat), callbacks= [mon], batch_size= bs, verbose=0)
    else:
        h= mdn.fit(x= X_train_dat, y= y_train_dat, epochs= epochs, validation_data=(X_val_dat, y_val_dat), callbacks= [mon], batch_size= bs, \
                                                                                                        sample_weight= samp_weight_arr, verbose=0) 
    print("MDN trained for %d epochs"%len(h.history['loss']))
    
    if not modsave:
        reg_freq_df= freq_pred_func(mdn_model= mdn, X_test_dat= X_test_dat, func_flag= func_flag, l4_flag= False, reg_len_arr= None, modsave= False)
        return reg_freq_df, h
    
    else:
        reg_freq_df= freq_pred_func(mdn_model= mdn, X_test_dat= X_test_dat, func_flag= func_flag, l4_flag= False, reg_len_arr= None, modsave= True)
        return reg_freq_df, h, mdn


def reg_fire_freq_L4_func(X_train_dat, y_train_dat, X_val_dat, y_val_dat, X_test_dat, reg_len_arr, lr= 1e-4, n_layers= 2, n_neurons= 16, epochs= 500, bs= 32, func_flag= 'zinb', rseed= None, samp_weights= False, samp_weight_arr= None, modsave= False):
    
    # Calculates the predicted fire frequency as well as its n sigma uncertainty for all regions
    
    if rseed == None:
        rseed= np.random.randint(100)
    tf.random.set_seed(rseed)

    if func_flag == 'zinb':
        stat_model= zinb_model
        loss_func= zinb_loss
        acc_func= zinb_accuracy
    elif func_flag == 'zipd':
        stat_model= zipd_model
        loss_func= zipd_loss
        acc_func= zipd_accuracy
    
    n_regions= 18
    cumlenarr= np.insert(np.cumsum(reg_len_arr), 0, 0)
    
    print("Initialized a MDN with %d layers"%n_layers + " and %d neurons"%n_neurons)
    
    mon= EarlyStopping(monitor='val_loss', min_delta=0, patience= 10, verbose=0, mode='auto')
    mdn= MDN_freq(layers= n_layers, neurons= n_neurons)
    mdn.compile(loss=loss_func, optimizer= tf.keras.optimizers.Adam(learning_rate= lr), metrics=[acc_func])
    if samp_weights == False:
        h= mdn.fit(x= X_train_dat, y= y_train_dat, epochs= epochs, validation_data=(X_val_dat, y_val_dat), callbacks= [mon], batch_size= bs, verbose=0)
    else:
        h= mdn.fit(x= X_train_dat, y= y_train_dat, epochs= epochs, validation_data=(X_val_dat, y_val_dat), callbacks= [mon], batch_size= bs, \
                                                                                                        sample_weight= samp_weight_arr, verbose=0) #callbacks= [mon],
    print("MDN trained for %d epochs"%len(h.history['loss']))
    
    if not modsave:
        reg_freq_df= freq_pred_func(mdn_model= mdn, X_test_dat= X_test_dat, func_flag= func_flag, l4_flag= True, reg_len_arr= reg_len_arr, modsave= False)
        return reg_freq_df, h
    
    else:
        reg_freq_df= freq_pred_func(mdn_model= mdn, X_test_dat= X_test_dat, func_flag= func_flag, l4_flag= True, reg_len_arr= reg_len_arr, modsave= True)
        return reg_freq_df, h, mdn
    
def freq_acc_func(pvec, obs_freqs, func_flag= 'zinb'):
    
    if func_flag == 'zinb':
        stat_model= zinb_model
    elif func_flag == 'zipd':
        stat_model= zipd_model
        
    pmf_pred= stat_model(pvec).prob(obs_freqs)
    obspmf= tfd.Empirical(obs_freqs)
    pmf_obs= obspmf.cdf(obs_freqs)
    
    acc= 100 - 100*stats.entropy(pmf_obs, qk= pmf_pred)  #converting convex KL divergence to concave equivalent 
    
    return acc  

def freq_crps_func(mdn_model, obs_input, obs_freqs, func_flag= 'zinb'):
    
    if func_flag == 'zinb':
        stat_model= zinb_model
    elif func_flag == 'zipd':
        stat_model= zipd_model
    
    param_vec= mdn_model.predict(x= tf.constant(obs_input))
    X_1= stat_model(param_vec).sample(10000)
    X_2= stat_model(param_vec).sample(10000)

    crps= 0.5*tf.reduce_mean(tf.abs(X_1 - X_2), axis= 0) - tf.reduce_mean(tf.abs(X_1 - obs_freqs), axis= 0)
    return (100 - 100*tf.abs(crps)).numpy()
    
def fire_freq_predict(fire_L3_freq_df, fire_L4_freq_df, dropcols= ['index', 'Tmin', 'Solar', 'Ant_Tmax', 'RH', 'Ant_RH', 'Elev', 'Camp_dist'], n_iters= 5, n_epochs= 10, bs= 32):
    
    # Evaluates the chisq and Pearson's correlation for observed and predicted fire frequencies for a variety of hyperparameters
    
    X_L3_freqs_train, X_L3_freqs_val, y_L3_freqs_train, y_L3_freqs_val, fire_L3_freq_test, X_L3_freqs_test, y_L3_freqs_test, L3_freq_samp_weight_arr= fire_freq_data(fire_L3_freq_df, dropcols= dropcols)
    X_L4_freqs_train, X_L4_freqs_val, y_L4_freqs_train, y_L4_freqs_val, fire_L4_freq_test, X_L4_freqs_test, y_L4_freqs_test, L4_freq_samp_weight_arr= fire_freq_data(fire_L4_freq_df, dropcols= dropcols)
    
    n_regions= 18
    tot_months= 60
    freqtestgrps= fire_L4_freq_test.groupby('reg_indx')
    reglenarr= np.asarray([len(freqtestgrps.get_group(k)) for k in freqtestgrps.groups.keys()])
    cumreglen= np.insert(np.cumsum(reglenarr), 0, 0)
    freq_test_size= np.int64(len(y_L3_freqs_test)/n_regions)
    freq_arr_1= np.linspace(0, len(y_L3_freqs_test) - freq_test_size, n_regions, dtype= int)
    freq_arr_2= freq_arr_1 + freq_test_size
    
    #n_iters= 5
    reg_flag= ['L3', 'L4']
    func_flag= ['zinb', 'zipd']
    list_of_lists = []

    
    for it in tqdm(range(n_iters)):
        #add model save option
        rseed= np.random.randint(100) 
        reg_L4_freq_zipd_df, _ , mdn_L4_freq_zipd= reg_fire_freq_L4_func(X_train_dat= X_L4_freqs_train, y_train_dat= y_L4_freqs_train, X_val_dat= X_L4_freqs_val, \
                                    y_val_dat= y_L4_freqs_val, X_test_dat= X_L4_freqs_test, reg_len_arr= reglenarr, epochs= n_epochs, bs= bs, 
                                    func_flag= 'zipd', rseed= rseed, samp_weights= True, samp_weight_arr= L4_freq_samp_weight_arr, modsave= True)
        reg_L4_freq_zipd_groups= reg_L4_freq_zipd_df.groupby('reg_indx')
        mdn_L4_freq_zipd.save('../sav_files/iter_runs_%s'%date.today().strftime("%y_%m_%d")+ '/mdn_L4_zipd' + '_iter_%d'%(it+1))

        reg_L4_freq_zinb_df, _, mdn_L4_freq_zinb= reg_fire_freq_L4_func(X_train_dat= X_L4_freqs_train, y_train_dat= y_L4_freqs_train, X_val_dat= X_L4_freqs_val, \
                                    y_val_dat= y_L4_freqs_val, X_test_dat= X_L4_freqs_test, reg_len_arr= reglenarr, epochs= n_epochs, bs= bs, 
                                    func_flag= 'zinb', rseed= rseed, samp_weights= True, samp_weight_arr= L4_freq_samp_weight_arr, modsave= True)
        reg_L4_freq_zinb_groups= reg_L4_freq_zinb_df.groupby('reg_indx')
        mdn_L4_freq_zinb.save('../sav_files/iter_runs_%s'%date.today().strftime("%y_%m_%d")+ '/mdn_L4_zinb' + '_iter_%d'%(it+1))

        reg_L3_freq_zipd_df, _, mdn_L3_freq_zipd= reg_fire_freq_func(X_train_dat= X_L3_freqs_train, y_train_dat= y_L3_freqs_train, X_val_dat= X_L3_freqs_val, \
                                    y_val_dat= y_L3_freqs_val, X_test_dat= X_L3_freqs_test, epochs= n_epochs, bs= bs, func_flag= 'zipd', rseed= rseed, \
                                    samp_weights= True, samp_weight_arr= L3_freq_samp_weight_arr, modsave= True) 
        reg_L3_freq_zipd_groups= reg_L3_freq_zipd_df.groupby('reg_indx')
        mdn_L3_freq_zipd.save('../sav_files/iter_runs_%s'%date.today().strftime("%y_%m_%d")+ '/mdn_L3_zipd' + '_iter_%d'%(it+1))

        reg_L3_freq_zinb_df, _, mdn_L3_freq_zinb= reg_fire_freq_func(X_train_dat= X_L3_freqs_train, y_train_dat= y_L3_freqs_train, X_val_dat= X_L3_freqs_val, \
                                    y_val_dat= y_L3_freqs_val, X_test_dat= X_L3_freqs_test, epochs= n_epochs, bs= bs, func_flag= 'zinb', rseed= rseed, \
                                    samp_weights= True, samp_weight_arr= L3_freq_samp_weight_arr, modsave= True)
        reg_L3_freq_zinb_groups= reg_L3_freq_zinb_df.groupby('reg_indx')
        mdn_L3_freq_zinb.save('../sav_files/iter_runs_%s'%date.today().strftime("%y_%m_%d")+ '/mdn_L3_zinb' + '_iter_%d'%(it+1))

        for regindx in range(n_regions):
            for l in reg_flag:
                for f in func_flag:
                    if l == 'L3':
                        obs_freqs= y_L3_freqs_test[freq_arr_1[regindx]:freq_arr_2[regindx]]
                        obs_input= X_L3_freqs_test[freq_arr_1[regindx]:freq_arr_2[regindx]]
                        if f == 'zinb':
                            reg_L3_freq_groups= reg_L3_freq_zinb_groups
                            param_vec= mdn_L3_freq_zinb.predict(x= tf.constant(obs_input))
                            emp_accuracy= freq_acc_func(pvec= param_vec, obs_freqs= obs_freqs, func_flag= 'zinb')
                            mod_accuracy= zinb_accuracy(obs_freqs, param_vec)
                            mod_loss= zinb_loss(obs_freqs, param_vec)
                        else:
                            reg_L3_freq_groups= reg_L3_freq_zipd_groups
                            param_vec= mdn_L3_freq_zipd.predict(x= tf.constant(obs_input))
                            emp_accuracy= freq_acc_func(pvec= param_vec, obs_freqs= obs_freqs, func_flag= 'zipd')
                            mod_accuracy= zipd_accuracy(obs_freqs, param_vec)
                            mod_loss= zipd_loss(obs_freqs, param_vec)

                        mean_freqs= reg_L3_freq_groups.get_group(regindx + 1)['mean_freq']
                        high_freqs= reg_L3_freq_groups.get_group(regindx + 1)['high_2sig_freq']
                        low_freqs= reg_L3_freq_groups.get_group(regindx + 1)['low_2sig_freq']

                        pearson_r= stats.pearsonr(obs_freqs, mean_freqs)
                        errarr_1= 16*(mean_freqs - obs_freqs)**2/(high_freqs - low_freqs)**2
                        errarr_2= 4*abs(mean_freqs - obs_freqs)/(high_freqs - low_freqs)
                        chisq_1= np.sum(errarr_1[np.isfinite(errarr_1)])
                        chisq_2= np.sum(errarr_2[np.isfinite(errarr_2)])
                        dof= len(errarr_1[np.isfinite(errarr_1)]) #+ mdn.count_params() - 1
                        
                        list_of_lists.append([it + 1, regindx + 1, l, f, pearson_r[0], chisq_1/dof, chisq_2/dof, emp_accuracy, mod_accuracy.numpy(), mod_loss.numpy()])

                    elif l == 'L4':
                        l4_freqs= y_L4_freqs_test[cumreglen[regindx]:cumreglen[regindx + 1]]
                        obs_input= X_L4_freqs_test[cumreglen[regindx]:cumreglen[regindx + 1]]
                        if f == 'zinb':
                            reg_L4_freq_groups= reg_L4_freq_zinb_groups
                            param_vec= mdn_L4_freq_zinb.predict(x= tf.constant(obs_input))
                            emp_accuracy= freq_acc_func(pvec= param_vec, obs_freqs= l4_freqs, func_flag= 'zinb')
                            mod_accuracy= zinb_accuracy(l4_freqs, param_vec)
                            mod_loss= zinb_loss(l4_freqs, param_vec)
                        else:
                            reg_L4_freq_groups= reg_L4_freq_zipd_groups
                            param_vec= mdn_L4_freq_zipd.predict(x= tf.constant(obs_input))
                            emp_accuracy= freq_acc_func(pvec= param_vec, obs_freqs= l4_freqs, func_flag= 'zipd')
                            mod_accuracy= zipd_accuracy(l4_freqs, param_vec)
                            mod_loss= zipd_loss(l4_freqs, param_vec)

                        obs_freqs= np.asarray([np.sum(y_L4_freqs_test[cumreglen[regindx]:cumreglen[regindx + 1]][np.arange(m, reglenarr[regindx], tot_months)]) \
                                                                                            for m in range(tot_months)])

                        mean_freqs= np.asarray([np.sum(reg_L4_freq_groups.get_group(regindx + 1)['mean_freq'].iloc[np.arange(m, reglenarr[regindx], tot_months)]) \
                                                                                                            for m in range(tot_months)])
                        high_freqs= mean_freqs + 2*np.sqrt([np.sum(reg_L4_freq_groups.get_group(regindx + 1)['std_freq'].iloc[np.arange(m, reglenarr[regindx], tot_months)]**2) \
                                                    for m in range(tot_months)])
                        low_freqs= mean_freqs - 2*np.sqrt([np.sum(reg_L4_freq_groups.get_group(regindx + 1)['std_freq'].iloc[np.arange(m, reglenarr[regindx], tot_months)]**2) \
                                                    for m in range(tot_months)])
                        low_freqs[low_freqs < 0]= 0

                        pearson_r= stats.pearsonr(obs_freqs, mean_freqs)
                        errarr_1= 16*(mean_freqs - obs_freqs)**2/(high_freqs - low_freqs)**2
                        errarr_2= 4*abs(mean_freqs - obs_freqs)/(high_freqs - low_freqs)
                        chisq_1= np.sum(errarr_1[np.isfinite(errarr_1)])
                        chisq_2= np.sum(errarr_2[np.isfinite(errarr_2)])
                        dof= len(errarr_1[np.isfinite(errarr_1)]) #+ mdn.count_params() - 1 #might be misleading due to high number of NaNs

                        list_of_lists.append([it + 1, regindx + 1, l, f, pearson_r[0], chisq_1/dof, chisq_2/dof, emp_accuracy, mod_accuracy.numpy(), mod_loss.numpy()])

    hp_df= pd.DataFrame(list_of_lists, columns=["Iteration", "reg_indx", "reg_flag", "func_flag", "Pearson_r", "Red_ChiSq_1", "Red_ChiSq_2", "Emp_Accuracy", "Mod_Accuracy", "Loss"])
    return hp_df

def load_ml_freq(fire_L4_freq_df, fire_L3_freq_df, dropcols= ['index', 'Tmin', 'Solar', 'Ant_Tmax', 'RH', 'Ant_RH', 'Elev', 'Camp_dist'], run_id= None): 
    
    # Loads the frequency distribution from prior saved 'best fit' runs --> used for plotting CCDF plots
    
    n_regions= 18
    tot_months= 60
    
    X_L3_freqs_train, X_L3_freqs_val, y_L3_freqs_train, y_L3_freqs_val, fire_L3_freq_test, X_L3_freqs_test, y_L3_freqs_test, \
        L3_freq_samp_weight_arr= fire_freq_data(fire_L3_freq_df, dropcols= dropcols)
    X_L4_freqs_train, X_L4_freqs_val, y_L4_freqs_train, y_L4_freqs_val, fire_L4_freq_test, X_L4_freqs_test, y_L4_freqs_test, \
        L4_freq_samp_weight_arr= fire_freq_data(fire_L4_freq_df, dropcols= dropcols)

    freqtestgrps= fire_L4_freq_test.groupby('reg_indx')
    reglenarr= np.asarray([len(freqtestgrps.get_group(k)) for k in freqtestgrps.groups.keys()])
    cumreglen= np.insert(np.cumsum(reglenarr), 0, 0)

    freq_test_size= np.int64(len(y_L3_freqs_test)/n_regions)
    freq_arr_1= np.linspace(0, len(y_L3_freqs_test) - freq_test_size, n_regions, dtype= int)
    freq_arr_2= freq_arr_1 + freq_test_size
    
    print("Loading ML frequency data from ../sav_files/iter_runs_%s"%run_id)
    hp_df= pd.read_hdf('../sav_files/iter_runs_%s'%run_id + '/hyperparams_iter_runs_%s.h5'%run_id)
    hp_df= hp_df[hp_df.Pearson_r >= 0.2]
    hp_df['tot_metric']= hp_df['Emp_Accuracy']/hp_df['Red_ChiSq']
    opt_freq_ind= np.asarray([hp_df.groupby('reg_indx').get_group(i+1).dropna().sort_values(by= ['tot_metric'], ascending= False).iloc[[0]].index for i in range(n_regions)]).flatten()

    regmodels= []
    reg_freq_df= pd.DataFrame({'mean_freq': pd.Series(dtype= 'int'), 'low_1sig_freq': pd.Series(dtype= 'int'), 'high_1sig_freq': pd.Series(dtype= 'int'), \
                                                                                                                        'reg_indx': pd.Series(dtype= 'int')})
    for regindx in tqdm(range(n_regions)):

        mod_params= hp_df.loc[opt_freq_ind[regindx]].to_dict()
        if mod_params['reg_flag'] == 'L4':
            if mod_params['func_flag'] == 'zipd':
                mdn_L4_zipd= tf.keras.models.load_model('../sav_files/iter_runs_%s'%run_id + '/mdn_L4_zipd_iter_%d'%(mod_params['Iteration']), \
                                                                                        custom_objects= {'zipd_loss': zipd_loss, 'zipd_accuracy': zipd_accuracy})
                reg_L4_freq_df= freq_pred_func(mdn_model= mdn_L4_zipd, X_test_dat= X_L4_freqs_test, func_flag= 'zipd', l4_flag= True, reg_len_arr= reglenarr, modsave= True)
                regmodels.append(zipd_model(mdn_L4_zipd.predict(x= tf.constant(X_L4_freqs_test[cumreglen[regindx]:cumreglen[regindx + 1]]))))
            else:
                mdn_L4_zinb= tf.keras.models.load_model('../sav_files/iter_runs_%s'%run_id + '/mdn_L4_zinb_iter_%d'%(mod_params['Iteration']), \
                                                                                        custom_objects= {'zinb_loss': zinb_loss, 'zinb_accuracy': zinb_accuracy})
                reg_L4_freq_df= freq_pred_func(mdn_model= mdn_L4_zinb, X_test_dat= X_L4_freqs_test, func_flag= 'zipd', l4_flag= True, reg_len_arr= reglenarr, modsave= True)
                regmodels.append(zinb_model(mdn_L4_zinb.predict(x= tf.constant(X_L4_freqs_test[cumreglen[regindx]:cumreglen[regindx + 1]]))))

            reg_L4_freq_groups= reg_L4_freq_df.groupby('reg_indx')
            fire_l3_mean_freqs= np.asarray([np.sum(reg_L4_freq_groups.get_group(regindx + 1)['mean_freq'].iloc[np.arange(m, reglenarr[regindx], tot_months)]) \
                                                                                                for m in range(tot_months)])
            fire_l3_high_freqs= fire_l3_mean_freqs + np.sqrt([np.sum(reg_L4_freq_groups.get_group(regindx + 1)['std_freq'].iloc[np.arange(m, reglenarr[regindx], tot_months)]**2) \
                                        for m in range(tot_months)])
            fire_l3_low_freqs= fire_l3_mean_freqs - np.sqrt([np.sum(reg_L4_freq_groups.get_group(regindx + 1)['std_freq'].iloc[np.arange(m, reglenarr[regindx], tot_months)]**2) \
                                        for m in range(tot_months)])
            fire_l3_low_freqs[fire_l3_low_freqs < 0]= 0

            reg_indx_arr= (regindx + 1)*np.ones(len(fire_l3_mean_freqs), dtype= int)
            reg_freq_df= reg_freq_df.append(pd.DataFrame({'mean_freq': fire_l3_mean_freqs.astype(int), 'low_1sig_freq': np.rint(fire_l3_low_freqs).astype(int), \
                                                          'high_1sig_freq': np.rint(fire_l3_high_freqs).astype(int), 'reg_indx': reg_indx_arr}), ignore_index=True)
        else:
            if mod_params['func_flag'] == 'zipd':
                mdn_L3_zipd= tf.keras.models.load_model('../sav_files/iter_runs_%s'%run_id + '/mdn_L3_zipd_iter_%d'%(mod_params['Iteration']), \
                                                            custom_objects= {'zipd_loss': zipd_loss, 'zipd_accuracy': zipd_accuracy})
                reg_L3_freq_df= freq_pred_func(mdn_model= mdn_L3_zipd, X_test_dat= X_L3_freqs_test, func_flag= 'zipd', l4_flag= False, modsave= True, nsig= 1)
                regmodels.append(zipd_model(mdn_L3_zipd.predict(x= tf.constant(X_L3_freqs_test[freq_arr_1[regindx]:freq_arr_2[regindx]]))))
            else:
                mdn_L3_zinb= tf.keras.models.load_model('../sav_files/iter_runs_%s'%run_id + '/mdn_L3_zinb_iter_%d'%(mod_params['Iteration']), \
                                                            custom_objects= {'zinb_loss': zinb_loss, 'zinb_accuracy': zinb_accuracy})
                reg_L3_freq_df= freq_pred_func(mdn_model= mdn_L3_zinb, X_test_dat= X_L3_freqs_test, func_flag= 'zinb', l4_flag= False, modsave= True, nsig= 1)
                regmodels.append(zinb_model(mdn_L3_zinb.predict(x= tf.constant(X_L3_freqs_test[freq_arr_1[regindx]:freq_arr_2[regindx]]))))

            reg_L3_freq_groups= reg_L3_freq_df.groupby('reg_indx')
            reg_freq_df= pd.concat([reg_freq_df, reg_L3_freq_groups.get_group(regindx + 1)], ignore_index= True)
        
    return reg_freq_df

def fire_freq_loco(fire_L3_freq_df, fire_L4_freq_df, n_iters= 10, n_epochs= 10, bs= 32, run_id= None):
    
    # Calculates the variable importance for a fire frequency MDN using a LOCO approach
    
    n_regions= 18
    tot_months= 60
    locoarr= ['VPD', 'Tmax', 'Prec', 'Forest', 'FM1000', 'Ant_VPD', 'Antprec', 'Avgprec', 'Urban', 'FFWI']
    list_of_lists= []
    
    for it in tqdm(range(n_iters)):
        rseed= np.random.randint(100)
        for var in range(len(locoarr) + 1):
            if var == 0: # 0 corresponds to all variables
                dropvarlist= ['index', 'Tmin', 'Solar', 'Ant_Tmax', 'RH', 'Ant_RH', 'Elev', 'Camp_dist']
            else:
                print("Loading predictor variable data without %s"%locoarr[var - 1])
                dropvarlist= ['index', 'Tmin', 'Solar', 'Ant_Tmax', 'RH', 'Ant_RH', 'Elev', 'Camp_dist', locoarr[var - 1]]

            X_L4_freqs_train, X_L4_freqs_val, y_L4_freqs_train, y_L4_freqs_val, fire_L4_freq_test, X_L4_freqs_test, y_L4_freqs_test, \
            L4_freq_samp_weight_arr= fire_freq_data(fire_L4_freq_df, dropcols= dropvarlist)
            freqtestgrps= fire_L4_freq_test.groupby('reg_indx')
            reglenarr= np.asarray([len(freqtestgrps.get_group(k)) for k in freqtestgrps.groups.keys()])
            cumreglen= np.insert(np.cumsum(reglenarr), 0, 0)

            reg_L4_freq_zipd_df, _ , mdn_L4_freq_zipd= reg_fire_freq_L4_func(X_train_dat= X_L4_freqs_train, y_train_dat= y_L4_freqs_train, X_val_dat= X_L4_freqs_val, \
                                        y_val_dat= y_L4_freqs_val, X_test_dat= X_L4_freqs_test, reg_len_arr= reglenarr, epochs= n_epochs, bs= bs, 
                                        func_flag= 'zipd', rseed= rseed, samp_weights= True, samp_weight_arr= L4_freq_samp_weight_arr, modsave= True)
            mdn_L4_freq_zipd.save('../sav_files/loco_runs_%s'%run_id + '/mdn_L4_zipd_iter_run_%d'%(it+1) + '_var_%d'%(var))
            reg_L4_freq_zipd_groups= reg_L4_freq_zipd_df.groupby('reg_indx')

            for regindx in range(n_regions):
                l4_freqs= y_L4_freqs_test[cumreglen[regindx]:cumreglen[regindx + 1]]
                obs_input= X_L4_freqs_test[cumreglen[regindx]:cumreglen[regindx + 1]]
                reg_L4_freq_groups= reg_L4_freq_zipd_groups
                param_vec= mdn_L4_freq_zipd.predict(x= tf.constant(obs_input))
                emp_accuracy= freq_acc_func(pvec= param_vec, obs_freqs= l4_freqs, func_flag= 'zipd')
                mod_accuracy= zipd_accuracy(l4_freqs, param_vec)
                mod_loss= zipd_loss(l4_freqs, param_vec)            

                obs_freqs= np.asarray([np.sum(y_L4_freqs_test[cumreglen[regindx]:cumreglen[regindx + 1]][np.arange(m, reglenarr[regindx], tot_months)]) \
                                                                                                for m in range(tot_months)])

                mean_freqs= np.asarray([np.sum(reg_L4_freq_groups.get_group(regindx + 1)['mean_freq'].iloc[np.arange(m, reglenarr[regindx], tot_months)]) \
                                                                                                                for m in range(tot_months)])
                high_freqs= mean_freqs + np.sqrt([np.sum(reg_L4_freq_groups.get_group(regindx + 1)['std_freq'].iloc[np.arange(m, reglenarr[regindx], tot_months)]**2) \
                                                        for m in range(tot_months)])
                low_freqs= mean_freqs - np.sqrt([np.sum(reg_L4_freq_groups.get_group(regindx + 1)['std_freq'].iloc[np.arange(m, reglenarr[regindx], tot_months)]**2) \
                                                        for m in range(tot_months)])
                low_freqs[low_freqs < 0]= 0

                pearson_r= stats.pearsonr(obs_freqs, mean_freqs)
                errarr_1= 4*(mean_freqs - obs_freqs)**2/(high_freqs - low_freqs)**2
                chisq_1= np.sum(errarr_1[np.isfinite(errarr_1)])
                dof= len(errarr_1[np.isfinite(errarr_1)]) #+ mdn.count_params() - 1 #might be misleading due to high number of NaNs

                list_of_lists.append([it + 1, var, regindx + 1, pearson_r[0], chisq_1/dof, emp_accuracy, mod_accuracy.numpy(), mod_loss.numpy()])

    var_df= pd.DataFrame(list_of_lists, columns=["Iteration", "Variable", "reg_indx", "Pearson_r", "Red_ChiSq_1", "Emp_Accuracy", "Mod_Accuracy", "Loss"])
    return var_df

## ----------------------------------------------------------------- Fire size functions ----------------------------------------------------------------------------

def fire_size_data(res= '12km', dropcols= ['CAPE', 'Solar', 'Ant_Tmax', 'RH', 'Ant_RH'], start_month= 372, tot_test_months= 60, threshold= None, scaled= False, rseed= None, tflag= False, hyp_flag= False):
    
    # Returns the train/val/test data for fire sizes given a grid resolution and threshold (in km^2); hyp_flag= True when performing hyperparameter search
    
    final_month= start_month + tot_test_months
    if rseed == None:
        rseed= np.random.randint(1000)
    
    if scaled:
        clim_fire_gdf= pd.read_hdf('../data/clim_fire_size_%s_rescaled_data.h5'%res) #should be replaced by a function that does the scaling properly and quickly
    else:
        if tflag:
            clim_fire_gdf= pd.read_hdf('../data/clim_fire_size_%s_w2020_data.h5'%res)
        else:
            clim_fire_gdf= pd.read_hdf('../data/clim_fire_size_%s_data.h5'%res) #saved clim_fire_gdf with geolocated fire + climate data at 12km res
    
    fire_size_train= init_eff_clim_fire_df(clim_fire_gdf, start_month, tot_test_months, hyp_flag= hyp_flag) #pd.read_hdf(data_dir + 'clim_fire_size_%s_train_data.h5'%res)
    #fire_size_train.loc[:, 'Tmin']= fire_size_train['Tmax'] - fire_size_train['Tmin']
    
    testfiregroups= clim_fire_gdf[(clim_fire_gdf['fire_month'] >= start_month) & (clim_fire_gdf['fire_month'] < final_month)].groupby('fire_indx')
    testdf= pd.DataFrame({})
    if hyp_flag:
        for k in testfiregroups.groups.keys():
            testdf= testdf.append(testfiregroups.get_group(k).loc[[testfiregroups.get_group(k)['cell_frac'].idxmax()]])
    else:
        for k in tqdm(testfiregroups.groups.keys()):
            testdf= testdf.append(testfiregroups.get_group(k).loc[[testfiregroups.get_group(k)['cell_frac'].idxmax()]]) 
    fire_size_test= testdf.reset_index().drop(columns= ['index', 'cell_frac']) #pd.read_hdf(data_dir + 'clim_fire_size_%s_test_data.h5'%res)
    #fire_size_test.loc[:, 'Tmin']= fire_size_test['Tmax'] - fire_size_test['Tmin']
    
    if threshold != None:
        fire_size_train= fire_size_train[fire_size_train['fire_size']/1e6 > threshold].reset_index().drop(columns= ['index'])
        fire_size_test= fire_size_test[fire_size_test['fire_size']/1e6 > threshold].reset_index().drop(columns= ['index'])
    
    fire_size_df= pd.concat([fire_size_train, fire_size_test], axis= 0)
    
    if scaled:
        X_size_train_df= fire_size_train.iloc[:, 7:].drop(columns= dropcols) 
        X_sizes_test= fire_size_test.iloc[:, 7:].drop(columns= dropcols)
    else:
        tmp_size_df= fire_size_df[fire_size_df.iloc[:, 7:].columns]
        X_size_df= pd.DataFrame({})
        scaler= StandardScaler().fit(fire_size_train.iloc[:, 7:])
        X_size_df[tmp_size_df.columns]= scaler.transform(tmp_size_df) 

        X_size_train_df= X_size_df.iloc[fire_size_train.index].drop(columns= dropcols) 
        X_sizes_test= X_size_df.iloc[fire_size_test.index + len(fire_size_train)].drop(columns= dropcols) 

    y_size_train= np.array(fire_size_train.fire_size/1e6, dtype=np.float32) #1e6 converts from m^2 to km^2
    y_sizes_test= np.array(fire_size_test.fire_size/1e6, dtype=np.float32)

    #splitting only the training data set
    X_sizes_train, X_sizes_val, y_sizes_train, y_sizes_val = train_test_split(X_size_train_df, y_size_train, test_size=0.2, random_state= rseed)
    
    if scaled:
        return X_sizes_train, X_sizes_val, y_sizes_train, y_sizes_val, fire_size_train, fire_size_test, X_sizes_test, y_sizes_test
    else:
        return X_sizes_train, X_sizes_val, y_sizes_train, y_sizes_val, fire_size_train, fire_size_test, X_sizes_test, y_sizes_test, scaler


def size_pred_func(mdn_model, stat_model, size_test_df, X_test_dat, max_size_arr, sum_size_arr, ncomps= 2, freq_flag= 'ml', regmlfreq= None, freqs_data= None, \
                                                                                                    debug= False, regindx= None, seed= None):
    
    # Given a NN model, the function returns the monthly burned area time series for all L3 regions
    
    tf.random.set_seed(seed)
    #X_test_dat= np.array(X_test_dat, dtype= np.float32)
    
    if debug:
        n_regions= 1 #18
    else:
        n_regions= 18
    tot_months= 60
    reg_size_df= pd.DataFrame({'mean_size': pd.Series(dtype= 'int'), 'low_1sig_size': pd.Series(dtype= 'int'), 'high_1sig_size': pd.Series(dtype= 'int'), \
                                                                                           'reg_indx': pd.Series(dtype= 'int')})

    for i in range(n_regions): 
        if debug:
            size_ind_df= size_test_df.reset_index()[['fire_size', 'fire_month', 'reg_indx']]
            reg_ind_df= size_ind_df.groupby('reg_indx').get_group(regindx).groupby('fire_month')
        else:
            size_ind_df= size_test_df.reset_index()[['fire_size', 'fire_month', 'reg_indx']]
            reg_ind_df= size_ind_df.groupby('reg_indx').get_group(i+1).groupby('fire_month')

        mean_burnarea_tot= np.zeros(tot_months)
        high_1sig_burnarea_tot= np.zeros(tot_months)
        low_1sig_burnarea_tot= np.zeros(tot_months)
        if debug:
            fire_ind_grid= []
            ml_param_grid= []

        for m in range(tot_months):
            mindx= m + 372
            samp_arr= tf.zeros([10000, 0])
            if freq_flag == 'ml':
                if debug:
                    reg_freqs= regmlfreq.groupby('reg_indx').get_group(regindx)  #replace with model instead of df and try with one region first
                else:
                    reg_freqs= regmlfreq.groupby('reg_indx').get_group(i+1)
                mean_freqs= reg_freqs['mean_freq'].iloc[[m]].to_numpy()[0] #iloc maintains month index for every region
                high_freqs= reg_freqs['high_1sig_freq'].iloc[[m]].to_numpy()[0]
                low_freqs= reg_freqs['low_1sig_freq'].iloc[[m]].to_numpy()[0]
            elif freq_flag == 'data':
                freq_size= np.int64(len(freqs_data)/18)
                tmparr_1= np.linspace(0, len(freqs_data) - freq_size, 18, dtype= np.int64)
                #tmparr_2= tmparr_1 + freq_size
                if debug:
                    freqs= freqs_data.astype(np.int64)[tmparr_1[regindx - 1] + m]
                else:
                    freqs= freqs_data.astype(np.int64)[tmparr_1[i] + m]
            
            # for sampling from frequency distribution, create additional function from here
            if mean_freqs == 0: #if mean freqs from distribution is zero, then set burned area to be zero
                mean_burnarea_tot[m]= 0
                high_1sig_burnarea_tot[m]= 0
                low_1sig_burnarea_tot[m]= 0
                #if debug:
                    #fire_ind_grid.append(np.array([0]))
                    #ml_param_grid.append(np.array([np.zeros(3*ncomps, dtype= np.float32)]))

            else:
                try:
                    fire_ind_arr= reg_ind_df.get_group(mindx).index.to_numpy() #replace with random draws of grid points from a RF learned 'fire potential' map
                    #print(m, mean_freqs, high_freqs, fire_ind_arr)
                    freqsarr= [mean_freqs, high_freqs, low_freqs] #low freqs are usually 0, so find a fix for that
                    for freqs in freqsarr:
                        if freqs > 0:
                            if freqs <= len(fire_ind_arr):
                                fire_ind_arr= np.random.choice(fire_ind_arr, freqs, replace= False)
                            else:
                                fire_ind_arr= np.append(fire_ind_arr, np.random.choice(fire_ind_arr, freqs - len(fire_ind_arr), replace= True)) #False might imply we run out of fires

                            ml_param_vec= mdn_model.predict(x= np.array(X_test_dat.iloc[fire_ind_arr], dtype= np.float32)) #note: different indexing than the fire_size_test df
                            samp_arr= tf.concat([samp_arr, stat_model(ml_param_vec).sample(10000, seed= 99)], axis= 1)
                            if debug:
                                fire_ind_grid.append(fire_ind_arr)
                                ml_param_grid.append(ml_param_vec)

                    size_samp_arr= tf.reduce_mean(samp_arr, axis= 0).numpy()
                    std_size_arr= tf.math.reduce_std(samp_arr, axis= 0).numpy()
                    high_1sig_err= deepcopy(std_size_arr)
                    tot_l1sig_arr= np.sqrt(np.sum(std_size_arr**2))

                    size_samp_arr[size_samp_arr > 2*max_size_arr[i]]= max_size_arr[i]
                    high_1sig_err[high_1sig_err > max_size_arr[i]]= max_size_arr[i] 
                    tot_h1sig_arr= np.sqrt(np.sum(high_1sig_err**2))

                    if np.sum(size_samp_arr) > 2*sum_size_arr[i]:
                        mean_burnarea_tot[m]= sum_size_arr[i]
                    else:
                        mean_burnarea_tot[m]= np.sum(size_samp_arr)

                    high_1sig_burnarea_tot[m]= mean_burnarea_tot[m] + tot_h1sig_arr
                    low_1sig_burnarea_tot[m]= mean_burnarea_tot[m] - tot_l1sig_arr
                    if (mean_burnarea_tot[m] - tot_l1sig_arr) < 0: 
                        low_1sig_burnarea_tot[m]= 0

                    #if np.max(size_samp_arr) > max_size_arr[i]:
                    #    max_size_arr[i]= np.max(size_samp_arr)

                    #while np.sum(size_samp_arr) > 2*sum_size_arr[i]:
                    #    rseed= np.random.randint(10000)
                    #    size_samp_arr= tf.reduce_mean(stat_model(ml_param_vec).sample(10000, seed= tfp.random.sanitize_seed(rseed)), axis= 0).numpy()
                    #    std_size_arr= tf.math.reduce_std(stat_model(ml_param_vec).sample(10000, seed= tfp.random.sanitize_seed(rseed)), axis= 0).numpy()
                    #if np.sum(size_samp_arr) > sum_size_arr[i]:
                    #    sum_size_arr[i]= np.sum(size_samp_arr)

                except KeyError:
                    if mean_freqs == 0:
                        mean_burnarea_tot[m]= 0 #current kludge and needs to be fixed
                        high_1sig_burnarea_tot[m]= 0
                        low_1sig_burnarea_tot[m]= 0
                        #if debug:
                        #    fire_ind_grid.append(np.array([0]))
                        #    ml_param_grid.append(np.array([np.zeros(3*ncomps, dtype= np.float32)]))

        reg_indx_arr= (i+1)*np.ones(tot_months, dtype= np.int64)
        reg_size_df= reg_size_df.append(pd.DataFrame({'mean_size': mean_burnarea_tot, 'low_1sig_size': low_1sig_burnarea_tot, 'high_1sig_size': high_1sig_burnarea_tot, \
                                                                                           'reg_indx': reg_indx_arr}), ignore_index=True)

    if debug:
        return reg_size_df, fire_ind_grid, ml_param_grid
    else:
        return reg_size_df
    
def reg_fire_size_func(X_train_dat, y_train_dat, X_val_dat, y_val_dat, size_test_df, X_test_dat, custom_ml_model= None, max_size_arr= None, sum_size_arr= None, \
                                                        func_flag= 'gpd', lnc_arr= [2, 16, 2], initializer= "he_normal", regflag= True, regrate= 0.001, doflag= True,\
                                                        epochs= 500, bs= 32, freq_flag= 'ml', regmlfreq= None, freqs_data= None, samp_weights= False, \
                                                        samp_weight_arr= None, loco= False, debug= False, regindx= None, rseed= None):
    
    # Calculates the predicted fire burned areas as well as its 1 sigma uncertainty for all regions
    
    if rseed == None:
        rseed= np.random.randint(100)
    tf.random.set_seed(rseed)
    X_train_dat= np.array(X_train_dat, dtype= np.float32)
    X_val_dat= np.array(X_val_dat, dtype= np.float32)

    if func_flag == 'gpd':
        n_layers, n_neurons, n_comps= lnc_arr
        stat_model= gpd_model
        loss_func= gpd_loss
        acc_func= gpd_accuracy
         
    elif func_flag == 'lognorm':
        n_layers, n_neurons, n_comps= lnc_arr     
        stat_model= lognorm_model
        loss_func= lognorm_loss
        acc_func= lognorm_accuracy
    
    elif func_flag == 'lognorm_gpd':
        n_layers, n_neurons, n_comps= lnc_arr     
        stat_model= lognorm_gpd_model
        loss_func= lognorm_gpd_loss
        acc_func= lognorm_gpd_accuracy
    
    print("Initialized a MDN with %d layers"%n_layers + " and %d neurons"%n_neurons)

    es_mon = EarlyStopping(monitor='val_loss', min_delta=0, patience= 10, verbose=0, mode='auto', restore_best_weights=True)
    mdn= MDN_size(layers= n_layers, neurons= n_neurons, components= n_comps, initializer= initializer, reg= regflag, regrate= regrate, dropout= doflag)
    mdn.compile(loss=loss_func, optimizer= tf.keras.optimizers.Adam(learning_rate= 1e-4), metrics=[acc_func])
    if samp_weights:
        h= mdn.fit(x= X_train_dat, y= y_train_dat, epochs= epochs, validation_data=(X_val_dat, y_val_dat), callbacks= [es_mon], batch_size= bs, sample_weight= samp_weight_arr, verbose=0)
    else:
        h= mdn.fit(x= X_train_dat, y= y_train_dat, epochs= epochs, validation_data=(X_val_dat, y_val_dat), callbacks= [es_mon], batch_size= bs, verbose=0)

    print("MDN trained for %d epochs"%len(h.history['loss']))

    if loco:
        return mdn, h

    else: 
        if debug:
            burnarea_df, fire_ind_grid, ml_param_grid= size_pred_func(mdn, stat_model, size_test_df, X_test_dat, max_size_arr, sum_size_arr, ncomps= n_comps, \
                                      freq_flag= freq_flag, regmlfreq= regmlfreq, freqs_data= freqs_data, debug= True, regindx= regindx, seed= rseed)
            return burnarea_df, fire_ind_grid, ml_param_grid
        else:
            burnarea_df= size_pred_func(mdn, stat_model, size_test_df, X_test_dat, max_size_arr, sum_size_arr, ncomps= n_comps, freq_flag= freq_flag, \
                                            regmlfreq= regmlfreq, freqs_data= freqs_data, debug= False, regindx= regindx, seed= rseed)
            return burnarea_df
        
def size_acc_func(pvec, obs_sizes, func_flag= 'gpd'):
    
    if func_flag == 'gpd':
        stat_model= gpd_model
    elif func_flag == 'lognormal':
        stat_model= lognormal_model
        
    pmf_pred= stat_model(pvec).prob(obs_sizes)
    obspmf= tfd.Empirical(obs_sizes)
    pmf_obs= obspmf.cdf(obs_sizes)
    
    acc= 100 - stats.entropy(pmf_obs, qk= pmf_pred)  #converting convex KL divergence to concave equivalent 
    
    return acc  
        
def fire_size_loco(firefile, reg_freq_df, res= '12km', n_iters= 10, n_epochs= 10, bs= 32, run_id= None):
    
    # Calculates the variable importance for a fire frequency MDN using a LOCO approach
    
    n_regions= 18
    locoarr= ['VPD', 'Tmax', 'Forest', 'Urban', 'FM1000', 'Prec', 'Antprec', 'Ant_VPD', 'Avgprec', 'FFWI']
    list_of_lists= []
    
    size_train_df= pd.read_hdf('../data/clim_fire_size_%s_train_data.h5'%res)
    max_fire_train_arr= []
    sum_fire_train_arr= []
    for r in range(n_regions):
        max_fire_train_arr.append(np.max(np.concatenate(\
                            [size_train_df.groupby('reg_indx').get_group(r+1).groupby('fire_month').get_group(k).fire_size.to_numpy()/1e6  \
                            for k in size_train_df.groupby('reg_indx').get_group(r+1).groupby('fire_month').groups.keys()])))
        sum_fire_train_arr.append(np.max([np.sum(\
                            size_train_df.groupby('reg_indx').get_group(r+1).groupby('fire_month').get_group(k).fire_size.to_numpy()/1e6) \
                            for k in size_train_df.groupby('reg_indx').get_group(r+1).groupby('fire_month').groups.keys()]))
    max_fire_train_arr= np.asarray(max_fire_train_arr)
    sum_fire_train_arr= np.asarray(sum_fire_train_arr)
    
    for it in tqdm(range(n_iters)):
        rseed= np.random.randint(100)
        for var in range(len(locoarr) + 1):
            if var == 0: # 0 corresponds to all variables
                dropvarlist= ['CAPE', 'Solar', 'Ant_Tmax', 'RH', 'Ant_RH']
            else:
                print("Loading predictor variable data without %s"%locoarr[var - 1])
                dropvarlist= ['CAPE', 'Solar', 'Ant_Tmax', 'RH', 'Ant_RH', locoarr[var - 1]]

            X_sizes_train, X_sizes_val, y_sizes_train, y_sizes_val, fire_size_train, fire_size_test, X_sizes_test, \
                                y_sizes_test= fire_size_data(res= res, dropcols= dropvarlist)

            mdn_size_gpd, _= reg_fire_size_func(X_train_dat= X_sizes_train, y_train_dat= y_sizes_train, X_val_dat= X_sizes_val, y_val_dat= y_sizes_val, \
                                    size_test_df= fire_size_test, X_test_dat= X_sizes_test, max_size_arr= max_fire_train_arr, sum_size_arr= sum_fire_train_arr, \
                                    func_flag= 'gpd', lnc_arr= [2, 8, 2], epochs= n_epochs, bs= bs, loco= True, rseed= rseed)
            mdn_size_gpd.save('../sav_files/loco_size_runs_%s'%run_id + '/mdn_%s'%res + '_gpd_iter_run_%d'%(it+1) + '_var_%d'%(var))
            
            for reg in range(n_regions):
                reg_ml_size_df, fire_ind_arr, ml_param_arr= size_pred_func(mdn_model= mdn_size_gpd, stat_model= gpd_model, size_test_df= fire_size_test, \
                                     X_test_dat= X_sizes_test, max_size_arr= max_fire_train_arr, sum_size_arr= sum_fire_train_arr, ncomps= 2, freq_flag= 'ml', \
                                     regmlfreq= reg_freq_df, debug= True, regindx= (reg + 1), seed= rseed)
                reg_sizes= y_sizes_test[np.concatenate(fire_ind_arr)]
                param_vec= [item for sublist in ml_param_arr for item in sublist] # neat hack from: https://stackoverflow.com/questions/952914/
                obs_sizes= mon_burned_area(firefile, (reg + 1))[372:]
                mean_sizes= reg_ml_size_df['mean_size']
                
                emp_accuracy= size_acc_func(pvec= param_vec, obs_sizes= reg_sizes, func_flag= 'gpd')
                mod_accuracy= gpd_accuracy(reg_sizes, param_vec)
                mod_loss= gpd_loss(reg_sizes, param_vec)  
                pearson_r= stats.pearsonr(obs_sizes, mean_sizes)

                list_of_lists.append([it + 1, var, reg + 1, pearson_r[0], emp_accuracy, mod_accuracy.numpy(), mod_loss.numpy()])

    var_df= pd.DataFrame(list_of_lists, columns=["Iteration", "Variable", "reg_indx", "Pearson_r", "Emp_Accuracy", "Mod_Accuracy", "Loss"])
    return var_df

## ----------------------------------------------------------------- Random forest functions ----------------------------------------------------------------------------


def rf_fire_grid_run(clim_grid_train_df, rb_frac, n_features= 36, dropcols= ['RH', 'Ant_RH'], n_trees= 100, threshold= 0.4, criterion= 'gini', \
                                                                                                                    test_data= True, clim_grid_test_df= None):
    
    # Creates a RF classifier instance for predicting fire probabilities at the grid scale
    
    rseed= np.random.randint(100)
    df1= clim_grid_train_df[clim_grid_train_df['fire_freq']==1] #.iloc[0:10000]
    n1= len(df1)
    n2= rb_frac*n1
    df2= clim_grid_train_df[clim_grid_train_df['fire_freq']==0]
    df2= df2.sample(n= int(n2))
    df_train= pd.concat([df1, df2], sort= False).sample(frac= 1).reset_index(drop=True) #shuffling the rows
    
    y_r = np.array(df_train.fire_freq)
    X_r = df_train.iloc[:, 0:n_features].drop(columns= dropcols)
    scaler= StandardScaler()
    X_r= scaler.fit_transform(X_r)
    X_r= np.array(X_r, dtype=np.float32)
    X_train, X_val, y_train, y_val = train_test_split(X_r, y_r, test_size=0.3, random_state= 99)
    
    rf= RandomForestClassifier(n_estimators= n_trees, criterion= criterion, random_state= rseed)
    forest= rf.fit(X_train, y_train)
    print("Trained the RF classifer on %d data points."%len(X_train))
    
    if threshold == None:
        predictions= rf.predict(X_val)
        errors= abs(predictions - y_val)  
    else:
        predicted_thresh= rf.predict_proba(X_val)
        predictions= (predicted_thresh[:, 1] >= threshold).astype('int')
        errors= abs(predictions - y_val)
    print('Training MAE:', round(np.mean(errors), 6))
    train_accuracy= metrics.accuracy_score(y_val, predictions)
    train_f1_score= metrics.f1_score(y_val, predictions)
    train_recall= metrics.recall_score(y_val, predictions)
    train_metrics= [train_accuracy, train_f1_score, train_recall]
    
    if test_data:
        df3= clim_grid_test_df[clim_grid_test_df['fire_freq']==1] #.iloc[10000:]
        df4= clim_grid_test_df[clim_grid_test_df['fire_freq']==0]
        df4= df4.sample(1000000)
        df_test= pd.concat([df3, df4], sort= False).sample(frac= 1).reset_index(drop=True) #shuffling the rows

        y= np.array(df_test.fire_freq)
        X= df_test.iloc[:, 0:n_features].drop(columns= dropcols)
        X= scaler.fit_transform(X) #same scaler as training data
        X= np.array(X, dtype=np.float32)

        if threshold == None:
            predictions= rf.predict(X)
            errors= abs(predictions - y)  
        else:
            predicted_thresh= rf.predict_proba(X)
            predictions= (predicted_thresh[:, 1] >= threshold).astype('int')
            errors= abs(predictions - y)
        print('Test MAE:', round(np.mean(errors), 6))
        test_accuracy= metrics.accuracy_score(y, predictions)
        test_f1_score= metrics.f1_score(y, predictions)
        test_recall= metrics.recall_score(y, predictions)
        test_metrics= [test_accuracy, test_f1_score, test_recall]

        return rf, forest, train_metrics, test_metrics
    else:
        return rf, forest, train_metrics
    
def rf_hyperparam_tuning(clim_grid_train_df, dropcols= ['Solar', 'Ant_Tmax', 'RH', 'Ant_RH'], rb_frac_arr= [10, 4, 3, 7/3, 3/2, 1], \
                         n_trees_arr= [50, 100, 250, 500, 1000], thresh_arr= [0.4, 0.5, None], n_iters= 5, run_id= None, modsave= False, \
                         test_data= True, clim_grid_test_df= None):
    
    list_of_lists= []
    
    for it in tqdm(range(n_iters)):
        for n_trees in n_trees_arr:
            for rb_frac in rb_frac_arr:
                for thresh in thresh_arr:
                    rf, forest, train_metrics, test_metrics= rf_fire_grid_run(clim_grid_train_df, rb_frac= rb_frac, dropcols= dropcols, \
                                                                    n_trees= n_trees, threshold= thresh, test_data= True, clim_grid_test_df= clim_grid_test_df)
                    list_of_lists.append([it + 1, n_trees, rb_frac, np.nan_to_num(thresh), train_metrics[0], train_metrics[1], test_metrics[0], test_metrics[2]])
            if modsave:
                joblib.dump(rf, "../sav_files/rf_runs_%s"%run_id + "/rf_mod_%d"%(it+1) + "_%d"%n_trees + "_.joblib")
        
    param_df= pd.DataFrame(list_of_lists, columns=["Iteration", "Trees", "Rebalance frac", "Threshold", "Train Accuracy", "Train Recall", \
                                                                                                                    "Test Accuracy", "Test Recall"])
    return param_df


## ----------------------------------------------------------------- Calibration and prediction functions for fire frequency ----------------------------------------------------------------------------

def ml_fire_freq_hyperparam_tuning(clim_df, negfrac= 0.3, n_iters= 5, bs_arr= [2048, 4096, 8192], pfrac_arr= [0.2, 0.3, 0.5, 0.7], \
                            dropcols= ['index', 'Solar', 'Ant_Tmax', 'RH', 'Ant_RH', 'FFWI_max7', 'Avgprec_4mo', 'Avgprec_2mo', 'AvgVPD_4mo', 'AvgVPD_2mo', 'Tmax_max7', 'VPD_max7', 'Tmin_max7'], \
                            start_month= 372, test_tot_months= 60, ml_model= 'mdn', loro_ind= None, run_id= None):
    
    list_of_lists = []
    
    for it in tqdm(range(n_iters)):
        rseed= np.random.randint(1000)
        n_features= 36
        end_month= start_month + test_tot_months
        
        clim_df= clim_df.dropna().reset_index().drop(columns=['index'])
        
        fire_freq_test_df= clim_df[(clim_df.month >= start_month) & (clim_df.month < end_month)]
        fire_freq_train_df= clim_df.drop(fire_freq_test_df.index)
        if loro_ind != None:
            fire_freq_train_df[fire_freq_train_df != loro_ind]

        tmp_freq_df= clim_df[clim_df.iloc[:, 0:n_features].columns]
        X_freq_df= pd.DataFrame({})
        scaler= StandardScaler().fit(fire_freq_train_df.iloc[:, 0:n_features])
        X_freq_df[tmp_freq_df.columns]= scaler.transform(tmp_freq_df)
        
        df1= fire_freq_train_df[fire_freq_train_df.fire_freq == 1]
        df2= fire_freq_train_df[fire_freq_train_df.fire_freq == 0].sample(frac= negfrac, random_state= rseed)
        fire_freq_train_df= pd.concat([df1, df2], sort= False) #.sample(frac= 1) .reset_index().drop(columns=['index'])

        X_train_df= X_freq_df.iloc[fire_freq_train_df.index].reset_index().drop(columns= dropcols)
        y_train_arr= np.array(fire_freq_train_df.fire_freq, dtype=np.float32)

        X_test_df= X_freq_df.iloc[fire_freq_test_df.index]
        X_test_df.loc[:, 'reg_indx']= fire_freq_test_df.reg_indx
        X_test_df.loc[:, 'month']= fire_freq_test_df.month
        X_test_df= X_test_df.reset_index().drop(columns= dropcols)
        y_test_arr= np.array(fire_freq_test_df.fire_freq)

        X_train, X_val, y_train, y_val= train_test_split(X_train_df, y_train_arr, test_size= 0.3, random_state= rseed)
        bool_train_labels= y_train != 0
        
        X_train_pos= X_train[bool_train_labels]
        X_train_neg= X_train[~bool_train_labels]
        y_train_pos= y_train[bool_train_labels]
        y_train_neg= y_train[~bool_train_labels]
        
        BUFFER_SIZE= len(X_train) #100000
        def make_ds(features, labels):
            ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()
            ds = ds.shuffle(BUFFER_SIZE).repeat()
            return ds

        pos_ds = make_ds(X_train_pos, y_train_pos)
        neg_ds = make_ds(X_train_neg, y_train_neg)

        for bs in bs_arr:
            for p_frac in pfrac_arr:
                
                resampled_ds= tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[p_frac, 1 - p_frac])
                resampled_ds= resampled_ds.batch(bs).prefetch(2)
                val_ds= tf.data.Dataset.from_tensor_slices((X_val, y_val)) #.cache()
                val_ds= val_ds.batch(bs).prefetch(2) 
                
                tf.random.set_seed(rseed)
                mon= EarlyStopping(monitor='val_loss', min_delta=0, patience= 5, verbose=0, mode='auto', restore_best_weights=True)
                if ml_model == 'mdn':
                    mdn= MDN_freq(layers= 2, neurons= 16)
                    mdn.compile(loss= zipd_loss, optimizer= tf.keras.optimizers.Adam(learning_rate= 1e-4), metrics=[zipd_accuracy])
                    h_ml= mdn.fit(resampled_ds, steps_per_epoch= 32, epochs= 500, validation_data= val_ds, callbacks= [mon], verbose= 0) # sample_weight= freq_samp_weight_arr #callbacks= [mon],
                    
                    print("MDN trained for %d epochs"%len(h_ml.history['loss']))
                    mdn.save('../sav_files/grid_freq_runs_%s'%run_id + '/mdn_%s'%bs + '_pfrac_%s'%str(p_frac) + '_iter_run_%d'%(it+1))
                    list_of_lists.append([it+1, bs, p_frac, len(h_ml.history['loss']), np.nanmean(h_ml.history['val_zipd_accuracy'])])
                
                elif ml_model == 'dnn':
                    dnn= DNN(layers= 2, neurons= 16)
                    dnn.compile(loss= "binary_crossentropy", optimizer= tf.keras.optimizers.Adam(learning_rate= 1e-4), metrics= ['binary_accuracy', tf.keras.metrics.Precision(name='precision'), \
                                                                                                                                 tf.keras.metrics.Recall(name='recall')])
                    h_ml= dnn.fit(resampled_ds, steps_per_epoch= 32, epochs= 500, validation_data= val_ds, callbacks= [mon], verbose= 0) # sample_weight= freq_samp_weight_arr #callbacks= [mon],
                    
                    print("DNN trained for %d epochs"%len(h_ml.history['loss']))
                    dnn.save('../sav_files/grid_freq_runs_%s'%run_id + '/dnn_%s'%bs + '_pfrac_%s'%str(p_frac) + '_iter_run_%d'%(it+1))
                    list_of_lists.append([it+1, bs, p_frac, len(h_ml.history['loss']), np.nanmean(h_ml.history['val_recall'])])
    
    
    hp_df= pd.DataFrame(list_of_lists, columns=["Iteration", "Batch size", "Fire fraction", "Epochs", "Val Accuracy/Recall"])
    
    return hp_df

def reg_pred_freq(X_test_dat, freq_test_df, nregs, start_month, func_flag, run_id, it, bs, p_frac):
    
    if func_flag == 'zipd':
        mdn_grid_zipd= tf.keras.models.load_model('../sav_files/grid_freq_runs_%s'%run_id + '/mdn_%s'%bs + '_pfrac_%s'%str(p_frac) + '_iter_run_%d'%it, \
                                                                                        custom_objects= {'zipd_loss': zipd_loss, 'zipd_accuracy': zipd_accuracy})
        mdn_freq_df= grid_freq_predict(X_test_dat, freq_test_df, nregs, ml_model= mdn_grid_zipd, start_month= start_month, func_flag= func_flag)
        mdn_freq_df.to_hdf('../sav_files/mdn_mon_fire_freq_%s'%run_id + '_it_%d'%it + '_%s'%bs + '_%s.h5'%str(p_frac), key= 'df', mode= 'w')

        print('Saved monthly grid predictions at: mdn_mon_fire_freq_%s'%run_id + '_it_%d'%it + '_%s'%bs + '_%s.h5'%str(p_frac))
    
    elif func_flag == 'logistic':
        dnn_grid_mod= tf.keras.models.load_model('../sav_files/grid_freq_runs_%s'%run_id + '/dnn_%s'%bs + '_pfrac_%s'%str(p_frac) + '_iter_run_%d'%it)
        dnn_freq_df= grid_freq_predict(X_test_dat, freq_test_df, nregs, ml_model= dnn_grid_mod, start_month= start_month, func_flag= func_flag)
        dnn_freq_df.to_hdf('../sav_files/dnn_mon_fire_freq_%s'%run_id + '_it_%d'%it + '_%s'%bs + '_%s.h5'%str(p_frac), key= 'df', mode= 'w')

        print('Saved monthly grid predictions at: dnn_mon_fire_freq_%s'%run_id + '_it_%d'%it + '_%s'%bs + '_%s.h5'%str(p_frac))
        

def rescale_factor_model(ml_freq_groups, regindx, tot_months, test_start, test_tot, input_type= 'std', pred_type= 'std', regtype= 'polynomial'):
    
    # Uses a linear model to predict the std/mean of annual observed frequencies using mean annual predicted frequencies
    
    ann_arr= np.arange(0, tot_months + 1, 12)
    ann_test_arr= np.arange(test_start, test_start + test_tot + 1, 12)
    
    pred_freqs_train= np.delete(np.array(ml_freq_groups.get_group(regindx)['pred_mean_freq']), np.arange(ann_test_arr[0]+1, ann_test_arr[-1])) #deleting elements corresponding to test years
    obs_freqs_train= np.delete(np.array(ml_freq_groups.get_group(regindx)['obs_freq']), np.arange(ann_test_arr[0]+1, ann_test_arr[-1])) #test years may start arbitrarily but must be continuous in range
    train_ind_arr= np.arange(0, len(pred_freqs_train), 12) #once test elements are removed, iterate over the new indices
    
    if input_type == 'std':
        X_mat_train= np.array([np.std(pred_freqs_train[train_ind_arr[t]:train_ind_arr[t+1]]) \
                        for t in range(len(train_ind_arr) - 1)])
        X_mat= np.array([np.std(ml_freq_groups.get_group(regindx)['pred_mean_freq'][ann_arr[t]:ann_arr[t+1]]) \
                        for t in range(len(ann_arr) - 1)])
    else:
        X_mat_train= np.array([np.mean(pred_freqs_train[train_ind_arr[t]:train_ind_arr[t+1]]) \
                        for t in range(len(train_ind_arr) - 1)])
        X_mat= np.array([np.mean(ml_freq_groups.get_group(regindx)['pred_mean_freq'][ann_arr[t]:ann_arr[t+1]]) \
                        for t in range(len(ann_arr) - 1)])
    
    if pred_type == 'std':
        Y_arr_train= np.array([np.std(obs_freqs_train[train_ind_arr[t]:train_ind_arr[t+1]]) \
                        for t in range(len(train_ind_arr) - 1)])
        Y_arr= np.array([np.std(ml_freq_groups.get_group(regindx)['obs_freq'][ann_arr[t]:ann_arr[t+1]]) \
                        for t in range(len(ann_arr) - 1)])
    else:
        Y_arr_train= np.array([np.mean(obs_freqs_train[train_ind_arr[t]:train_ind_arr[t+1]]) \
                        for t in range(len(train_ind_arr) - 1)])
        Y_arr= np.array([np.mean(ml_freq_groups.get_group(regindx)['obs_freq'][ann_arr[t]:ann_arr[t+1]]) \
                        for t in range(len(ann_arr) - 1)])
        
    if regtype == 'linear':
        reg_pred= LinearRegression().fit(X_mat_train.reshape(-1, 1), Y_arr_train)
        pred_norm= reg_pred.predict(X_mat.reshape(-1, 1))
        r_pred= reg_pred.score(X_mat.reshape(-1, 1), Y_arr)
    else:
        if regtype == 'polynomial':
            poly_feat= PolynomialFeatures(3)
            ann_poly_x= poly_feat.fit_transform(X_mat.reshape(-1, 1))
        elif regtype == 'spline':
            spline_feat= SplineTransformer(degree= 3, n_knots= 5)
            scaler= spline_feat.fit(X_mat_train.reshape(-1, 1))
            ann_poly_x_train= scaler.transform(X_mat_train.reshape(-1, 1))
            ann_poly_x= scaler.transform(X_mat.reshape(-1, 1))
        
        model= make_pipeline(SplineTransformer(n_knots= 5, degree= 3), LinearRegression())
        reg_pred= model.fit(X_mat_train.reshape(-1, 1), Y_arr_train)
        pred_norm= reg_pred.predict(X_mat.reshape(-1, 1))
        r_pred= reg_pred.score(X_mat.reshape(-1, 1), Y_arr)
    
    return pred_norm, r_pred

def grid_freq_metrics(ml_freq_df, n_regs, tot_months, test_start, test_tot):
    
    # Loads a data frame with grid scale fire frequencies and outputs the monthly and annual time series with two accuracy metrics 
    
    ml_freq_groups= ml_freq_df.groupby('reg_indx')
    ann_arr= np.arange(0, tot_months + 1, 12)
    
    regtype_arr= ['linear', 'spline']
    inp_type_arr= ['std', 'mean']
    pred_type_arr= ['std', 'mean']
    list_of_lists= []
    acc_df= pd.DataFrame([])
    
    for r in range(n_regs): #tqdm
        for reg in regtype_arr:
            for i in inp_type_arr:
                for p in pred_type_arr:
                    rfac_norm, r_pred= rescale_factor_model(ml_freq_groups, regindx= (r+1), tot_months= tot_months, test_start= test_start, test_tot= test_tot, input_type= i, pred_type= p, regtype= reg)
                    norm_fac= []
                    if p == 'std':
                        for t in range(len(ann_arr) - 1):
                            tmpnorm= rfac_norm[t]/np.ceil(np.std(ml_freq_groups.get_group(r+1)['pred_mean_freq'][ann_arr[t]:ann_arr[t+1]]))
                            if np.isinf(tmpnorm):
                                norm_fac.append(rfac_norm[t]/np.std(ml_freq_groups.get_group(r+1)['pred_mean_freq']))
                            else:
                                norm_fac.append(tmpnorm)
                        pred_calib_arr= np.concatenate([np.ceil(np.array(ml_freq_groups.get_group(r+1)['pred_mean_freq'][ann_arr[t]:ann_arr[t+1]])*norm_fac[t]) for t in range(len(ann_arr) - 1)])
                        pred_calib_high2sig_arr= np.concatenate([np.ceil(np.array(ml_freq_groups.get_group(r+1)['pred_high_2sig'][ann_arr[t]:ann_arr[t+1]])*norm_fac[t]) for t in range(len(ann_arr) - 1)])
                        pred_calib_low2sig_arr= np.concatenate([np.ceil(np.array(ml_freq_groups.get_group(r+1)['pred_low_2sig'][ann_arr[t]:ann_arr[t+1]])*norm_fac[t]) for t in range(len(ann_arr) - 1)])
                        ann_pred_std_arr= np.asarray([np.sqrt(np.sum((((ml_freq_groups.get_group(r+1)['pred_high_2sig'] - \
                                                            ml_freq_groups.get_group(r+1)['pred_low_2sig'])*norm_fac[t]/4)**2)[ann_arr[t]:ann_arr[t+1]])) for t in range(len(ann_arr) - 1)]) 
                    else:
                        for t in range(len(ann_arr) - 1):
                            tmpnorm= rfac_norm[t]/np.ceil(np.mean(ml_freq_groups.get_group(r+1)['pred_mean_freq'][ann_arr[t]:ann_arr[t+1]]))
                            if np.isinf(tmpnorm):
                                norm_fac.append(rfac_norm[t]/np.mean(ml_freq_groups.get_group(r+1)['pred_mean_freq']))
                            else:
                                norm_fac.append(tmpnorm)
                        pred_calib_arr= np.concatenate([np.ceil(np.array(ml_freq_groups.get_group(r+1)['pred_mean_freq'][ann_arr[t]:ann_arr[t+1]])*norm_fac[t]) for t in range(len(ann_arr) - 1)])
                        pred_calib_high2sig_arr= np.concatenate([np.ceil(np.array(ml_freq_groups.get_group(r+1)['pred_high_2sig'][ann_arr[t]:ann_arr[t+1]])*norm_fac[t]) for t in range(len(ann_arr) - 1)])
                        pred_calib_low2sig_arr= np.concatenate([np.ceil(np.array(ml_freq_groups.get_group(r+1)['pred_low_2sig'][ann_arr[t]:ann_arr[t+1]])*norm_fac[t]) for t in range(len(ann_arr) - 1)])
                        ann_pred_std_arr= np.asarray([np.sqrt(np.sum((((ml_freq_groups.get_group(r+1)['pred_high_2sig'] - \
                                                            ml_freq_groups.get_group(r+1)['pred_low_2sig'])*norm_fac[t]/4)**2)[ann_arr[t]:ann_arr[t+1]])) for t in range(len(ann_arr) - 1)]) 
                        
                    obs_freqs_tot= np.array(ml_freq_groups.get_group(r+1)['obs_freq'])
                    obs_freqs_test= np.array(ml_freq_groups.get_group(r+1)['obs_freq'])[test_start:(test_start + test_tot)]
                    pred_freqs_test= pred_calib_arr[test_start:(test_start + test_tot)]
                    pred_std_tot= (pred_calib_high2sig_arr - pred_calib_low2sig_arr)/4
                    pred_std_test= pred_std_tot[test_start:(test_start + test_tot)]
                    ann_pred_calib_arr= np.array([np.sum(pred_calib_arr[ann_arr[t]:ann_arr[t+1]]) for t in range(len(ann_arr) - 1)])
                    ann_obs_freq_tot= np.array([np.sum(obs_freqs_tot[ann_arr[t]:ann_arr[t+1]]) for t in range(len(ann_arr) - 1)])
                                        
                    r_calib_tot= stats.pearsonr(obs_freqs_tot, pred_calib_arr)[0]
                    r_calib_test= stats.pearsonr(obs_freqs_test, pred_freqs_test)[0]
                    r_ann_calib_tot= stats.pearsonr(ann_obs_freq_tot, ann_pred_calib_arr)[0]
                    
                    errarr_tot= (pred_calib_arr - obs_freqs_tot)**2/pred_std_tot**2
                    chisq_tot= np.sum(errarr_tot[np.isfinite(errarr_tot)])
                    dof_tot= len(errarr_tot[np.isfinite(errarr_tot)])
                    errarr_test= (pred_freqs_test - obs_freqs_test)**2/pred_std_test**2
                    chisq_test= np.sum(errarr_test[np.isfinite(errarr_test)])
                    dof_test= len(errarr_test[np.isfinite(errarr_test)])
                    errarr_ann_tot= (ann_pred_calib_arr - ann_obs_freq_tot)**2/ann_pred_std_arr**2
                    chisq_ann_tot= np.sum(errarr_ann_tot[np.isfinite(errarr_ann_tot)])
                    dof_ann_tot= len(errarr_ann_tot[np.isfinite(errarr_ann_tot)])
                    
                    list_of_lists.append([r+1, reg, i, p, r_calib_tot, chisq_tot/dof_tot, r_calib_test, chisq_test/dof_test, r_ann_calib_tot, chisq_ann_tot/dof_ann_tot])
    
    acc_df= pd.DataFrame(list_of_lists, columns=["reg_indx", "Regression", "Input type", "Pred. type", "r_total", "red_chisq_total", "r_test", "red_chisq_test", "r_ann_total", "red_chisq_ann_total"])
    return acc_df

def grid_freq_predict(X_test_dat, freq_test_df= None, n_regs= 18, ml_model= None, start_month= 0, final_month= 432, func_flag= 'zipd', shap_flag= False, regindx= None, rseed= 99):
    
    # Predicts the fire frequency for each L3 ecoregion
    
    tot_months= final_month - start_month
    ml_freq_df= pd.DataFrame([])
    
    if shap_flag:
        freq_arr= []
        X_arr= np.array(X_test_dat, dtype= np.float32)
        param_vec= ml_model.predict(x= tf.constant(X_arr))
        freq_samp= zipd_model(param_vec).sample(1000, seed= rseed)
        freq_arr.append(tf.cast(tf.reduce_mean(freq_samp, axis= 0), tf.int64).numpy())
        
        reg_indx_arr= (regindx+1)*np.ones(len(freq_arr), dtype= np.int64)
        ml_freq_df= ml_freq_df.append(pd.DataFrame({'pred_mean_freq': freq_arr, 'reg_indx': reg_indx_arr}))
    
    else:
        tot_rfac_arr= []
        for r in tqdm(range(n_regs)):  
            pred_freq= []
            pred_freq_sig= []
            freq_arr= []
            for m in np.linspace(start_month, final_month - 1, tot_months, dtype= np.int64):
                X_arr= np.array(X_test_dat.groupby('reg_indx').get_group(r+1).groupby('month').get_group(m).dropna().drop(columns= ['reg_indx', 'month']), dtype= np.float32)
                if func_flag == 'zipd':
                    param_vec= ml_model.predict(x= tf.constant(X_arr))
                    freq_samp= zipd_model(param_vec).sample(10000, seed= rseed)
                    pred_freq.append(tf.reduce_sum(tf.cast(tf.reduce_mean(freq_samp, axis= 0), tf.int64)).numpy())
                    pred_freq_sig.append(np.sqrt(tf.reduce_sum(tf.pow(tf.cast(tf.math.reduce_std(freq_samp, axis= 0), tf.int64), 2)).numpy()).astype(np.int64))

                    freq_arr.append(tf.cast(tf.reduce_mean(freq_samp, axis= 0), tf.int64).numpy())

                elif func_flag == 'logistic':
                    reg_predictions= ml_model.predict(x= tf.constant(X_arr)).flatten()
                    freq_arr.append([1 if p > 0.5 else 0 for p in reg_predictions])
                    pred_freq.append(np.sum([1 if p > 0.5 else 0 for p in reg_predictions]))

            obs_freqs= [np.sum(freq_test_df.groupby('reg_indx').get_group(r+1).groupby('month').get_group(m).fire_freq) \
                                                                              for m in np.linspace(start_month, final_month - 1, tot_months, dtype= np.int64)]
            #tot_rfac_arr.append((np.std(obs_freqs)/np.std(pred_freq)))
            pred_freq_arr= [np.sum(freq_arr[m - start_month]) for m in np.linspace(start_month, final_month - 1, tot_months, dtype= np.int64)]
            reg_indx_arr= np.ones(tot_months, dtype= np.int64)*(r+1)

            if func_flag == 'zipd':
                pred_high_2sig= np.ceil((np.array(pred_freq) + 2*np.array(pred_freq_sig)))
                pred_low= np.array(pred_freq) - 2*np.array(pred_freq_sig)
                pred_low[pred_low < 0]= 0
                pred_low_2sig= np.floor(pred_low)

                ml_freq_df= ml_freq_df.append(pd.DataFrame({'obs_freq': obs_freqs, 'pred_mean_freq': pred_freq_arr, 'pred_high_2sig': pred_high_2sig, 'pred_low_2sig': pred_low_2sig, \
                                                                                                                                      'reg_indx': reg_indx_arr}))
            if func_flag == 'logistic':
                ml_freq_df= ml_freq_df.append(pd.DataFrame({'obs_freq': obs_freqs, 'pred_mean_freq': pred_freq_arr, 'reg_indx': reg_indx_arr}))
            
    return ml_freq_df #, tot_rfac_arr

def calib_freq_predict(ml_freq_df, n_regs, tot_months, test_start, test_tot, ml_model= 'mdn', debug= False, regindx= None, arg_arr= None, manarg= None):
    
    # returns the calibrated predicted frequencies with a rescaled factor optimized for both monthly and annual metrics
    
    ml_freq_groups= ml_freq_df.groupby('reg_indx')
    ann_arr= np.arange(0, tot_months + 1, 12)
    pred_mon_freq_df= pd.DataFrame([])
    pred_ann_freq_df= pd.DataFrame([])
    
    if debug:
        norm_fac= []
        reg, inp, pred= arg_arr
        rfac_norm, r_pred= rescale_factor_model(ml_freq_groups, regindx= regindx, tot_months= tot_months, test_start= test_start, test_tot= test_tot, input_type= inp, pred_type= pred, regtype= reg)

        if pred == 'mean':
            for t in range(len(ann_arr) - 1):
                tmpnorm= rfac_norm[t]/np.ceil(np.mean(ml_freq_groups.get_group(regindx)['pred_mean_freq'][ann_arr[t]:ann_arr[t+1]]))
                if np.isinf(tmpnorm):
                    norm_fac.append(rfac_norm[t]/np.mean(ml_freq_groups.get_group(regindx)['pred_mean_freq']))
                else:
                    norm_fac.append(tmpnorm)
        else:
            for t in range(len(ann_arr) - 1):
                tmpnorm= rfac_norm[t]/np.ceil(np.std(ml_freq_groups.get_group(regindx)['pred_mean_freq'][ann_arr[t]:ann_arr[t+1]]))
                if np.isinf(tmpnorm):
                    norm_fac.append(rfac_norm[t]/np.std(ml_freq_groups.get_group(regindx)['pred_mean_freq']))
                else:
                    norm_fac.append(tmpnorm)

        pred_calib_arr= np.concatenate([np.ceil(np.array(ml_freq_groups.get_group(regindx)['pred_mean_freq'][ann_arr[t]:ann_arr[t+1]])*norm_fac[t]) for t in range(len(ann_arr) - 1)])
        ann_pred_calib_arr= [np.sum(pred_calib_arr[ann_arr[t]:ann_arr[t+1]]) for t in range(len(ann_arr) - 1)]
        ann_obs_freq_arr= [np.sum(ml_freq_groups.get_group(regindx)['obs_freq'][ann_arr[t]:ann_arr[t+1]]) for t in range(len(ann_arr) - 1)]

        mon_reg_indx_arr= np.ones(tot_months, dtype= np.int64)*(regindx)
        ann_reg_indx_arr= np.ones(len(ann_arr[1:]), dtype= np.int64)*(regindx)

        if ml_model == 'mdn':
            pred_calib_high2sig_arr= np.concatenate([np.ceil(np.array(ml_freq_groups.get_group(regindx)['pred_high_2sig'][ann_arr[t]:ann_arr[t+1]])*norm_fac[t]) for t in range(len(ann_arr) - 1)])
            pred_calib_low2sig_arr= np.concatenate([np.ceil(np.array(ml_freq_groups.get_group(regindx)['pred_low_2sig'][ann_arr[t]:ann_arr[t+1]])*norm_fac[t]) for t in range(len(ann_arr) - 1)])

            ann_pred_std_arr= np.asarray([np.sqrt(np.sum((((ml_freq_groups.get_group(regindx)['pred_high_2sig'] - \
                                                                ml_freq_groups.get_group(regindx)['pred_low_2sig'])*norm_fac[t]/4)**2)[ann_arr[t]:ann_arr[t+1]])) for t in range(len(ann_arr) - 1)])

            pred_mon_freq_df= pred_mon_freq_df.append(pd.DataFrame({'obs_freq': ml_freq_groups.get_group(regindx)['obs_freq'], 'pred_mean_freq': pred_calib_arr, 'pred_high_2sig': pred_calib_high2sig_arr, \
                                                                        'pred_low_2sig': pred_calib_low2sig_arr, 'reg_indx': mon_reg_indx_arr}))
            pred_ann_freq_df= pred_ann_freq_df.append(pd.DataFrame({'obs_freq': ann_obs_freq_arr, 'pred_mean_freq': ann_pred_calib_arr, 'pred_high_2sig': (ann_pred_calib_arr + 2*ann_pred_std_arr), \
                                                                        'pred_low_2sig': (ann_pred_calib_arr - 2*ann_pred_std_arr), 'reg_indx': ann_reg_indx_arr}))
        elif ml_model == 'dnn':
            pred_mon_freq_df= pred_mon_freq_df.append(pd.DataFrame({'obs_freq': ml_freq_groups.get_group(regindx)['obs_freq'], 'pred_mean_freq': pred_calib_arr, 'reg_indx': mon_reg_indx_arr}))
            pred_ann_freq_df= pred_ann_freq_df.append(pd.DataFrame({'obs_freq': ann_obs_freq_arr, 'pred_mean_freq': ann_pred_calib_arr, 'reg_indx': ann_reg_indx_arr}))

        return pred_mon_freq_df, pred_ann_freq_df
    
    else:
        ml_acc_df= grid_freq_metrics(ml_freq_df= ml_freq_df, n_regs= n_regs, tot_months= tot_months, test_start= test_start, test_tot= test_tot)
        ml_acc_df['tot_metric']= ml_acc_df['r_total']*ml_acc_df['r_ann_total']/(ml_acc_df['red_chisq_total'] + ml_acc_df['red_chisq_ann_total'])
        for r in range(n_regs): #tqdm
            norm_fac= []
            if manarg != None:
                reg, inp, pred= ml_acc_df.groupby('reg_indx').get_group(r+1).sort_values(by= ['tot_metric'], ascending= False).iloc[[manarg]][['Regression', 'Input type', 'Pred. type']].to_numpy()[0]
            else:
                reg, inp, pred= ml_acc_df.groupby('reg_indx').get_group(r+1).sort_values(by= ['tot_metric'], ascending= False).iloc[[0]][['Regression', 'Input type', 'Pred. type']].to_numpy()[0]
            rfac_norm, r_pred= rescale_factor_model(ml_freq_groups, regindx= r+1, tot_months= tot_months, test_start= test_start, test_tot= test_tot, input_type= inp, pred_type= pred, regtype= reg)

            if pred == 'mean':
                for t in range(len(ann_arr) - 1):
                    tmpnorm= rfac_norm[t]/np.ceil(np.mean(ml_freq_groups.get_group(r+1)['pred_mean_freq'][ann_arr[t]:ann_arr[t+1]]))
                    if np.isinf(tmpnorm):
                        norm_fac.append(rfac_norm[t]/np.mean(ml_freq_groups.get_group(r+1)['pred_mean_freq']))
                    else:
                        norm_fac.append(tmpnorm)
            else:
                for t in range(len(ann_arr) - 1):
                    tmpnorm= rfac_norm[t]/np.ceil(np.std(ml_freq_groups.get_group(r+1)['pred_mean_freq'][ann_arr[t]:ann_arr[t+1]]))
                    if np.isinf(tmpnorm):
                        norm_fac.append(rfac_norm[t]/np.std(ml_freq_groups.get_group(r+1)['pred_mean_freq']))
                    else:
                        norm_fac.append(tmpnorm)

            pred_calib_arr= np.concatenate([np.ceil(np.array(ml_freq_groups.get_group(r+1)['pred_mean_freq'][ann_arr[t]:ann_arr[t+1]])*norm_fac[t]) for t in range(len(ann_arr) - 1)])
            ann_pred_calib_arr= [np.sum(pred_calib_arr[ann_arr[t]:ann_arr[t+1]]) for t in range(len(ann_arr) - 1)]
            ann_obs_freq_arr= [np.sum(ml_freq_groups.get_group(r+1)['obs_freq'][ann_arr[t]:ann_arr[t+1]]) for t in range(len(ann_arr) - 1)]

            mon_reg_indx_arr= np.ones(tot_months, dtype= np.int64)*(r+1)
            ann_reg_indx_arr= np.ones(len(ann_arr[1:]), dtype= np.int64)*(r+1)

            if ml_model == 'mdn':
                pred_calib_high2sig_arr= np.concatenate([np.ceil(np.array(ml_freq_groups.get_group(r+1)['pred_high_2sig'][ann_arr[t]:ann_arr[t+1]])*norm_fac[t]) for t in range(len(ann_arr) - 1)])
                pred_calib_low2sig_arr= np.concatenate([np.ceil(np.array(ml_freq_groups.get_group(r+1)['pred_low_2sig'][ann_arr[t]:ann_arr[t+1]])*norm_fac[t]) for t in range(len(ann_arr) - 1)])

                ann_pred_std_arr= np.asarray([np.sqrt(np.sum((((ml_freq_groups.get_group(r+1)['pred_high_2sig'] - \
                                                                ml_freq_groups.get_group(r+1)['pred_low_2sig'])*norm_fac[t]/4)**2)[ann_arr[t]:ann_arr[t+1]])) for t in range(len(ann_arr) - 1)])

                pred_mon_freq_df= pred_mon_freq_df.append(pd.DataFrame({'obs_freq': ml_freq_groups.get_group(r+1)['obs_freq'], 'pred_mean_freq': pred_calib_arr, 'pred_high_2sig': pred_calib_high2sig_arr, \
                                                                        'pred_low_2sig': pred_calib_low2sig_arr, 'reg_indx': mon_reg_indx_arr}))
                pred_ann_freq_df= pred_ann_freq_df.append(pd.DataFrame({'obs_freq': ann_obs_freq_arr, 'pred_mean_freq': ann_pred_calib_arr, 'pred_high_2sig': (ann_pred_calib_arr + 2*ann_pred_std_arr), \
                                                                        'pred_low_2sig': (ann_pred_calib_arr - 2*ann_pred_std_arr), 'reg_indx': ann_reg_indx_arr}))
            elif ml_model == 'dnn':
                pred_mon_freq_df= pred_mon_freq_df.append(pd.DataFrame({'obs_freq': ml_freq_groups.get_group(r+1)['obs_freq'], 'pred_mean_freq': pred_calib_arr, 'reg_indx': mon_reg_indx_arr}))
                pred_ann_freq_df= pred_ann_freq_df.append(pd.DataFrame({'obs_freq': ann_obs_freq_arr, 'pred_mean_freq': ann_pred_calib_arr, 'reg_indx': ann_reg_indx_arr}))

        return pred_mon_freq_df, pred_ann_freq_df
    
def grid_freq_loc_predict(X_test_dat, n_regs, ml_model, start_month, final_month= 432, func_flag= 'zipd', rseed= None):
    
    # Predicts the grid scale location of fire frequencies/probabilites for each L3 ecoregion
    
    if rseed == None:
        rseed= np.random.randint(1000)
        
    tot_months= final_month - start_month
    freq_df= pd.DataFrame([])
    
    for r in tqdm(range(n_regs)): #range(n_regs)
        pred_freq= []
        pred_freq_sig= []
        freq_arr= []
        for m in np.linspace(start_month, final_month - 1, tot_months, dtype= np.int64):
            X_arr= np.array(X_test_dat.groupby('reg_indx').get_group(r+1).groupby('month').get_group(m).dropna().drop(columns= ['reg_indx', 'month']), dtype= np.float32)
            if func_flag == 'zipd':
                param_vec= ml_model.predict(x= tf.constant(X_arr))
                freq_samp= zipd_model(param_vec).sample(10000, seed= rseed)
                freq_arr.append(tf.cast(tf.reduce_mean(freq_samp, axis= 0), tf.int64).numpy()) #associate climate grid index
            
            elif func_flag == 'logistic':
                reg_predictions= ml_model.predict(x= tf.constant(X_arr)).flatten()
                freq_arr.append([1 if p > 0.5 else 0 for p in reg_predictions])
        
        mon_reg_indx_arr= np.ones(tot_months, dtype= np.int64)*(r+1)
        freq_df= freq_df.append(pd.DataFrame({'freq_arr': freq_arr, 'reg_indx': mon_reg_indx_arr}))
        
    return freq_df

def fire_loc_func(loc_df, ml_freq_df, X_test_dat, regindx, start_month, final_month= 432, loc_flag= 'ml'):
    
    # Predicts the grid scale location of fire frequencies/probabilites for each L3 ecoregion
    
    tot_months= final_month - start_month
    ml_freq_groups= ml_freq_df.groupby('reg_indx')
    if loc_flag == 'ml':
        freqlabel= 'pred_mean_freq'
    else:
        freqlabel= 'obs_freq'
    fire_loc_arr= []
    
    for m in np.linspace(start_month, final_month - 1, tot_months, dtype= np.int64):
        if not np.nonzero(loc_df.groupby('reg_indx').get_group(regindx)['freq_arr'].loc[[m - start_month]].to_numpy()[0])[0].size:
            fire_loc_arr.append(np.array([0]))
        else:
            indarr= np.random.choice(np.nonzero(loc_df.groupby('reg_indx').get_group(regindx)['freq_arr'].loc[[m - start_month]].to_numpy()[0])[0], \
                                     ml_freq_groups.get_group(regindx)[freqlabel].loc[[m]].to_numpy()[0].astype(int))
            fire_loc_arr.append(X_test_dat.groupby('reg_indx').get_group(regindx).groupby('month').get_group(m).index.to_numpy()[indarr])
    
    return fire_loc_arr

def fire_loc_func_obs(size_test_df, regindx, start_month, final_month= 432, ml_freq_flag= False, ml_freq_df= None):
    
    #note the outputs are indices in the size_test_df data frame, not the grid cells in a particular month as in the previous case
    
    tot_months= final_month - start_month
    reg_ind_groups= size_test_df[['fire_size', 'fire_month', 'reg_indx']].groupby('reg_indx').get_group(regindx).groupby('fire_month')
    fire_loc_arr= []
    
    if not ml_freq_flag:
        for m in np.linspace(start_month, final_month - 1, tot_months, dtype= np.int64):
            try:
                indarr= reg_ind_groups.get_group(m).index.to_numpy()
                fire_loc_arr.append(indarr)
            except KeyError:
                fire_loc_arr.append(np.array([0]))
    else:
        ml_freq_groups= ml_freq_df.groupby('reg_indx')
        for m in np.linspace(start_month, final_month - 1, tot_months, dtype= np.int64):
            predfreq= ml_freq_groups.get_group(regindx)['pred_mean_freq'].loc[[m]].to_numpy()[0].astype(int)
            try:
                indarr= reg_ind_groups.get_group(m).sort_values(by= ['fire_size'], ascending= False).index.to_numpy()
                if len(indarr) >= predfreq:
                    fire_loc_arr.append(indarr[:predfreq])
                else:
                    indarr= np.append(indarr, np.random.choice(indarr, predfreq - len(indarr), replace= True))
                    fire_loc_arr.append(indarr)
            except KeyError:
                fire_loc_arr.append(np.array([0]))
    
    return fire_loc_arr

def loc_ind_func(loc_df, ml_freq_df, X_test_dat, n_regs, start_month= 0, final_month= 432):
    
    # Returns an array of grid indices corresponding to predicted frequencies
    
    pred_loc_arr= []
    for r in tqdm(range(n_regs)): #range(n_regs)
        tmplocarr= fire_loc_func(loc_df, ml_freq_df, X_test_dat, regindx= r+1, start_month= start_month, final_month= final_month)
        tmplocarr= np.hstack(tmplocarr)
        pred_loc_arr.append(tmplocarr[tmplocarr != 0])

    return pred_loc_arr

## ----------------------------------------------------------------- Calibration and prediction functions for fire size ----------------------------------------------------------------------------


def grid_size_pred_func(mdn_model, stat_model, max_size_arr, sum_size_arr, start_month= 0, final_month= 432, freq_flag= 'ml', nsamps= 1000, \
                            loc_df= None, loc_flag= 'ml', ml_freq_flag= False, ml_freq_df= None, X_freq_test_dat= None, size_test_df= None, X_size_test_dat= None, \
                            debug= False, shap_flag= False, regindx= None, seed= None):
    
    # Given a NN model, the function returns the monthly burned area time series for all L3 regions
    # TODO: include effect of frequency uncertainty
    
    #tf.random.set_seed(seed)
    if seed == None:
        seed= np.random.randint(1000)
    
    if debug:
        n_regions= 1 #18
    else:
        n_regions= 18
    tot_months= final_month - start_month
    
    if shap_flag:
        reg_size_df= pd.DataFrame({'mean_size': pd.Series(dtype= 'int'), 'reg_indx': pd.Series(dtype= 'int')})
        samp_arr= tf.zeros([nsamps, 0])
        ml_param_vec= mdn_model.predict(x= np.array(X_size_test_dat, dtype= np.float32))
        samp_arr= tf.concat([samp_arr, tf.reshape(stat_model(ml_param_vec).sample(nsamps, seed= seed), (nsamps, ml_param_vec.shape[0]))], axis= 1)
        size_samp_arr= tf.reduce_mean(samp_arr, axis= 0).numpy()
        size_samp_arr[size_samp_arr > max_size_arr[regindx]]= max_size_arr[regindx]
        reg_indx_arr= (regindx+1)*np.ones(len(size_samp_arr), dtype= np.int64)
        reg_size_df= reg_size_df.append(pd.DataFrame({'mean_size': size_samp_arr, 'reg_indx': reg_indx_arr}), ignore_index=True) #add month index
    
    else:
        reg_size_df= pd.DataFrame({'mean_size': pd.Series(dtype= 'int'), 'low_1sig_size': pd.Series(dtype= 'int'), 'high_1sig_size': pd.Series(dtype= 'int'), \
                                                                                           'reg_indx': pd.Series(dtype= 'int')})
        for r in range(n_regions): #tqdm --> removed for hyperparameter runs
            mean_burnarea_tot= np.zeros(tot_months)
            high_1sig_burnarea_tot= np.zeros(tot_months)
            low_1sig_burnarea_tot= np.zeros(tot_months)

            if freq_flag == 'ml':
                if debug:
                    fire_loc_arr= fire_loc_func(loc_df, ml_freq_df, X_freq_test_dat, regindx= regindx, start_month= start_month, final_month= final_month, loc_flag= loc_flag)  #replace with model instead of df and try with one region first
                    fire_ind_grid= []
                    ml_param_grid= []
                else:
                    fire_loc_arr= fire_loc_func(loc_df, ml_freq_df, X_freq_test_dat, regindx= r+1, start_month= start_month, final_month= final_month, loc_flag= loc_flag)
            elif freq_flag == 'data':
                if debug:
                    fire_loc_arr= fire_loc_func_obs(size_test_df, regindx= regindx, start_month= start_month, final_month= final_month, ml_freq_flag= ml_freq_flag, ml_freq_df= ml_freq_df)
                    fire_ind_grid= []
                    ml_param_grid= []
                else:
                    fire_loc_arr= fire_loc_func_obs(size_test_df, regindx= r+1, start_month= start_month, final_month= final_month, ml_freq_flag= ml_freq_flag, ml_freq_df= ml_freq_df)

            for m in np.linspace(start_month, final_month - 1, tot_months, dtype= np.int64):
                mindx= m - start_month
                samp_arr= tf.zeros([nsamps, 0])

                # for sampling from frequency distribution, create additional function from here
                if np.nonzero(fire_loc_arr[mindx])[0].size == 0: #if mean freqs from distribution is zero, then set burned area to be zero
                    mean_burnarea_tot[mindx]= 0
                    high_1sig_burnarea_tot[mindx]= 0
                    low_1sig_burnarea_tot[mindx]= 0
                    if debug:
                        fire_ind_grid.append([0])
                        ml_param_grid.append([0])
                else:
                    if freq_flag == 'ml':
                        ml_param_vec= mdn_model.predict(x= np.array(X_freq_test_dat.iloc[fire_loc_arr[mindx]].drop(columns= ['CAPE', 'reg_indx', 'month']), dtype= np.float32)) #note: different indexing than the fire_size_test df
                    elif freq_flag == 'data':
                        ml_param_vec= mdn_model.predict(x= np.array(X_size_test_dat.iloc[fire_loc_arr[mindx]], dtype= np.float32))
                    samp_arr= tf.concat([samp_arr, tf.reshape(stat_model(ml_param_vec).sample(nsamps, seed= seed), (nsamps, ml_param_vec.shape[0]))], axis= 1)
                    if debug:
                        fire_ind_grid.append(fire_loc_arr[mindx])
                        ml_param_grid.append(ml_param_vec)

                    size_samp_arr= tf.reduce_mean(samp_arr, axis= 0).numpy()
                    std_size_arr= tf.math.reduce_std(samp_arr, axis= 0).numpy()
                    high_1sig_err= deepcopy(std_size_arr)
                    tot_l1sig_arr= np.sqrt(np.sum(std_size_arr**2))

                    if debug:
                        size_samp_arr[size_samp_arr > max_size_arr[regindx]]= max_size_arr[regindx]
                        high_1sig_err[high_1sig_err > max_size_arr[regindx]]= max_size_arr[regindx] 
                        tot_h1sig_arr= np.sqrt(np.sum(high_1sig_err**2))

                        if np.sum(size_samp_arr) > 3*sum_size_arr[regindx][m]:
                            mean_burnarea_tot[mindx]= sum_size_arr[regindx][m]
                        else:
                            mean_burnarea_tot[mindx]= np.sum(size_samp_arr)
                    else:
                        size_samp_arr[size_samp_arr > max_size_arr[r]]= max_size_arr[r]
                        high_1sig_err[high_1sig_err > max_size_arr[r]]= max_size_arr[r] 
                        tot_h1sig_arr= np.sqrt(np.sum(high_1sig_err**2))

                        if np.sum(size_samp_arr) > 3*sum_size_arr[r][m]:
                            mean_burnarea_tot[mindx]= sum_size_arr[r][m]
                        else:
                            mean_burnarea_tot[mindx]= np.sum(size_samp_arr)

                    high_1sig_burnarea_tot[mindx]= mean_burnarea_tot[mindx] + tot_h1sig_arr
                    low_1sig_burnarea_tot[mindx]= mean_burnarea_tot[mindx] - tot_l1sig_arr
                    if (mean_burnarea_tot[mindx] - tot_l1sig_arr) < 0: 
                        low_1sig_burnarea_tot[mindx]= 0

                    #if np.max(size_samp_arr) > max_size_arr[i]:
                    #    max_size_arr[i]= np.max(size_samp_arr)

                    #while np.sum(size_samp_arr) > 2*sum_size_arr[i]:
                    #    rseed= np.random.randint(10000)
                    #    size_samp_arr= tf.reduce_mean(stat_model(ml_param_vec).sample(10000, seed= tfp.random.sanitize_seed(rseed)), axis= 0).numpy()
                    #    std_size_arr= tf.math.reduce_std(stat_model(ml_param_vec).sample(10000, seed= tfp.random.sanitize_seed(rseed)), axis= 0).numpy()
                    #if np.sum(size_samp_arr) > sum_size_arr[i]:
                    #    sum_size_arr[i]= np.sum(size_samp_arr)

            if debug:
                reg_indx_arr= (regindx)*np.ones(tot_months, dtype= np.int64)
            else:
                reg_indx_arr= (r+1)*np.ones(tot_months, dtype= np.int64)
            reg_size_df= reg_size_df.append(pd.DataFrame({'mean_size': mean_burnarea_tot, 'low_1sig_size': low_1sig_burnarea_tot, 'high_1sig_size': high_1sig_burnarea_tot, \
                                                                                               'reg_indx': reg_indx_arr}), ignore_index=True)

    if debug:
        return reg_size_df, fire_ind_grid, ml_param_grid
    else:
        return reg_size_df

def sampling_func(size_arr):
    
    # Calculates an interpolated pdf from a given fire size data set
    
    s= pd.Series(size_arr)
    s= s[~s.duplicated()]
    
    empcdf_sizes= tfd.Empirical(s) 
    cdf_emp_sizes= empcdf_sizes.cdf(s)
    kde_density= stats.gaussian_kde(size_arr, bw_method= "silverman")
    curve_1= 1 - np.sort(cdf_emp_sizes)
    curve_2= kde_density.evaluate(np.sort(s))[1:]
    curve_1= curve_1[curve_1!= 0]

    eff_curve= 10**(0.5*(np.log10(curve_1) + np.log10(curve_2)))
    eff_curve_func= interpolate.interp1d(np.sort(s)[1:], eff_curve, kind= 'linear', fill_value= 'extrapolate')
    
    return eff_curve_func

def max_fire_size_sum_func(fire_size_df, final_month= 432):
    monthstep= [120, 240, 360, final_month]
    reg_sum_size_arr= []
    for r in range(18): #tqdm
        tmparr= np.array([np.sum(fire_size_df.groupby('reg_indx').get_group(r+1).groupby('fire_month').get_group(k).fire_size.to_numpy()/1e6) \
                                        for k in fire_size_df.groupby('reg_indx').get_group(r+1).groupby('fire_month').groups.keys()])
        indarr= np.array(list(fire_size_df.groupby('reg_indx').get_group(r+1).groupby('fire_month').groups.keys()), dtype= int)

        maxsumsize= np.max(tmparr[0:len(indarr[indarr <= monthstep[0]])])
        sumsizearr= np.repeat(maxsumsize, monthstep[0])
        totarrlen= len(indarr[indarr <= monthstep[0]])

        for i in range(len(monthstep) - 1):
            if np.max(tmparr[totarrlen:totarrlen+len(indarr[(indarr > monthstep[i])&(indarr <= monthstep[i+1])])]) > maxsumsize:
                maxsumsize= np.max(tmparr[totarrlen:totarrlen+len(indarr[(indarr > monthstep[i])&(indarr <= monthstep[i+1])])])

            sumsizearr= np.append(sumsizearr, np.repeat(maxsumsize, monthstep[i+1] - monthstep[i]))
            totarrlen+= len(indarr[(indarr > monthstep[i])&(indarr <= monthstep[i+1])])

        reg_sum_size_arr.append(sumsizearr)
        
    return np.asarray(reg_sum_size_arr)

def cumm_fire_freq_func(mdn_mon_freq_df, mdn_ann_freq_df, optflag= False, regarr= None):
    
    #Returns the cummulative monthly and annual fire frequencies across the entire study region
    
    n_regions= 18
    mdn_mon_freq_groups= mdn_mon_freq_df.groupby('reg_indx')
    mdn_ann_freq_groups= mdn_ann_freq_df.groupby('reg_indx')
    reg_mon_r_calib_arr= np.array([stats.pearsonr(mdn_mon_freq_groups.get_group(r+1)['obs_freq'].to_numpy(), mdn_mon_freq_groups.get_group(r+1)['pred_mean_freq'].to_numpy())[0] for r in range(n_regions)])
    
    tot_mon_obs_freq_arr= []
    tot_mon_pred_freq_arr= []
    tot_mon_pred_high_2sig_arr= []
    tot_mon_pred_low_2sig_arr= []
    tot_ann_obs_freq_arr= []
    tot_ann_pred_freq_arr= []
    tot_ann_pred_high_2sig_arr= []
    tot_ann_pred_low_2sig_arr= []

    if optflag:
        if regarr == None:
            reglist= np.arange(n_regions)[reg_mon_r_calib_arr >= 0.6]
        else:
            reglist= regarr
    else:
        reglist= np.arange(n_regions)
    
    for r in reglist:
        tot_mon_obs_freq_arr.append(mdn_mon_freq_groups.get_group(r+1)['obs_freq'].to_numpy())
        tot_mon_pred_freq_arr.append(mdn_mon_freq_groups.get_group(r+1)['pred_mean_freq'].to_numpy())
        tot_mon_pred_high_2sig_arr.append(mdn_mon_freq_groups.get_group(r+1)['pred_high_2sig'].to_numpy())
        tot_mon_pred_low_2sig_arr.append(mdn_mon_freq_groups.get_group(r+1)['pred_low_2sig'].to_numpy())
        tot_ann_obs_freq_arr.append(mdn_ann_freq_groups.get_group(r+1)['obs_freq'].to_numpy())
        tot_ann_pred_freq_arr.append(mdn_ann_freq_groups.get_group(r+1)['pred_mean_freq'].to_numpy())
        tot_ann_pred_high_2sig_arr.append(mdn_ann_freq_groups.get_group(r+1)['pred_high_2sig'].to_numpy())
        tot_ann_pred_low_2sig_arr.append(mdn_ann_freq_groups.get_group(r+1)['pred_low_2sig'].to_numpy())
    
    tot_mon_pred_2sig_arr= (np.sqrt(np.sum(np.power(tot_mon_pred_high_2sig_arr, 2), axis= 0)) - np.sqrt(np.sum(np.power(tot_mon_pred_low_2sig_arr, 2), axis= 0)))
    tot_mon_pred_high_2sig_arr= np.sum(tot_mon_pred_freq_arr, axis= 0) + tot_mon_pred_2sig_arr
    tot_mon_pred_low_2sig_arr= np.sum(tot_mon_pred_freq_arr, axis= 0) - tot_mon_pred_2sig_arr
    
    tot_ann_pred_2sig_arr= (np.sqrt(np.sum(np.power(tot_ann_pred_high_2sig_arr, 2), axis= 0)) - np.sqrt(np.sum(np.power(tot_ann_pred_low_2sig_arr, 2), axis= 0)))
    tot_ann_pred_high_2sig_arr= np.sum(tot_ann_pred_freq_arr, axis= 0) + tot_ann_pred_2sig_arr
    tot_ann_pred_low_2sig_arr= np.sum(tot_ann_pred_freq_arr, axis= 0) - tot_ann_pred_2sig_arr

    return tot_mon_obs_freq_arr, tot_mon_pred_freq_arr, tot_mon_pred_high_2sig_arr, tot_mon_pred_low_2sig_arr, \
                                                                    tot_ann_obs_freq_arr, tot_ann_pred_freq_arr, tot_ann_pred_high_2sig_arr, tot_ann_pred_low_2sig_arr

def cumm_fire_size_func(firefile, reg_size_df, tot_months= 432, optflag= False, regarr= None, timebreak= False, breakmon= 252, reg_gpd_ext_size_df= None):
    
    #Returns the cummulative monthly and annual fire sizes across the entire study region
    
    n_regions= 18
    final_yr= int(tot_months/12) + 1983
    yr_arr= np.arange(0, tot_months + 1, 12)
    reg_mon_r_calib_arr= np.array([stats.pearsonr(mon_burned_area(firefile, r+1, final_year= final_yr), reg_size_df.groupby('reg_indx').get_group(r+1)['mean_size'][0:tot_months])[0] for r in range(n_regions)])
    
    tot_mon_obs_size_arr= []
    tot_mon_pred_size_arr= []
    tot_mon_pred_1sig_arr= []
    tot_ann_obs_size_arr= []
    tot_ann_pred_size_arr= []
    tot_ann_pred_1sig_arr= []
    #tot_ann_pred_low_1sig_arr= []

    iind= 0
    if optflag:
        if regarr == None:
            reglist= np.arange(n_regions)[reg_mon_r_calib_arr >= 0.6]
        else:
            reglist= regarr
    else:
        reglist= np.arange(n_regions)
    
    for r in reglist: #tqdm
        tot_mon_obs_size_arr.append(mon_burned_area(firefile, r+1, final_year= final_yr).values)
        tot_mon_pred_size_arr.append(reg_size_df.groupby('reg_indx').get_group(r+1)['mean_size'].to_numpy()[0:tot_months])
        tot_mon_pred_1sig_arr.append(0.5*(reg_size_df.groupby('reg_indx').get_group(r+1)['high_1sig_size'].to_numpy() - reg_size_df.groupby('reg_indx').get_group(r+1)['low_1sig_size'].to_numpy())[0:tot_months])
        tot_ann_obs_size_arr.append(np.array([np.sum(tot_mon_obs_size_arr[iind][yr_arr[i]:yr_arr[i+1]]) for i in range(len(yr_arr) - 1)]))
        tot_ann_pred_size_arr.append(np.array([np.sum(tot_mon_pred_size_arr[iind][yr_arr[i]:yr_arr[i+1]]) for i in range(len(yr_arr) - 1)]))
        tot_ann_pred_1sig_arr.append(np.array([np.sqrt(np.sum(np.power(tot_mon_pred_1sig_arr[iind][yr_arr[i]:yr_arr[i+1]], 2))) for i in range(len(yr_arr) - 1)]))
        iind+= 1
        #tot_ann_pred_low_1sig_arr.append(np.array([np.sqrt(np.sum(np.power(tot_mon_pred_low_1sig_arr[r][yr_arr[i]:yr_arr[i+1]], 2))) for i in range(len(yr_arr) - 1)]))
    
    if timebreak:
        reg_mon_ext_r_calib_arr= np.array([stats.pearsonr(mon_burned_area(firefile, r+1, final_year= final_yr), reg_gpd_ext_size_df.groupby('reg_indx').get_group(r+1)['mean_size'][0:tot_months])[0] for r in range(n_regions)])

        tot_mon_pred_ext_size_arr= []
        tot_mon_pred_ext_1sig_arr= []
        tot_ann_pred_ext_size_arr= []
        tot_ann_pred_ext_1sig_arr= []
        
        iind= 0
        if optflag:
            if regarr == None:
                reglist= np.arange(n_regions)[reg_mon_ext_r_calib_arr >= 0.6]
            else:
                reglist= regarr
        else:
            reglist= np.arange(n_regions)
    
        for r_ext in reglist: #tqdm
            tot_mon_pred_ext_size_arr.append(reg_gpd_ext_size_df.groupby('reg_indx').get_group(r_ext+1)['mean_size'].to_numpy()[0:tot_months])
            tot_mon_pred_ext_1sig_arr.append(0.5*(reg_gpd_ext_size_df.groupby('reg_indx').get_group(r_ext+1)['high_1sig_size'].to_numpy() - \
                                                  reg_gpd_ext_size_df.groupby('reg_indx').get_group(r_ext+1)['low_1sig_size'].to_numpy())[0:tot_months])
            tot_ann_pred_ext_size_arr.append(np.array([np.sum(tot_mon_pred_ext_size_arr[iind][yr_arr[i]:yr_arr[i+1]]) for i in range(len(yr_arr) - 1)]))
            tot_ann_pred_ext_1sig_arr.append(np.array([np.sqrt(np.sum(np.power(tot_mon_pred_ext_1sig_arr[iind][yr_arr[i]:yr_arr[i+1]], 2))) for i in range(len(yr_arr) - 1)]))
            iind+= 1
        
        breakyr= int(breakmon/12) - 1
        tot_mon_fire_size_arr= np.append(np.sum(tot_mon_pred_size_arr, axis= 0)[0:breakmon], np.sum(tot_mon_pred_ext_size_arr, axis= 0)[breakmon:])
        tot_mon_fire_1sig_arr= np.append(np.sqrt(np.sum(np.power(tot_mon_pred_1sig_arr, 2), axis= 0))[0:breakmon], np.sqrt(np.sum(np.power(tot_mon_pred_ext_1sig_arr, 2), axis= 0))[breakmon:])
        tot_ann_fire_size_arr= np.append(np.sum(tot_ann_pred_size_arr, axis= 0)[0:breakyr], np.sum(tot_ann_pred_ext_size_arr, axis= 0)[breakyr:])
        tot_ann_fire_1sig_arr= np.sqrt(np.sum(np.power(tot_ann_pred_ext_1sig_arr, 2), axis= 0)) #np.append(np.sqrt(np.sum(np.power(tot_ann_pred_1sig_arr, 2), axis= 0))[0:20], np.sqrt(np.sum(np.power(tot_ann_pred_ext_1sig_arr, 2), axis= 0))[20:])
        
        return tot_mon_obs_size_arr, tot_mon_fire_size_arr, tot_mon_fire_1sig_arr, tot_ann_obs_size_arr, tot_ann_fire_size_arr, tot_ann_fire_1sig_arr
    
    else:
        return tot_mon_obs_size_arr, np.sum(tot_mon_pred_size_arr, axis= 0), np.sqrt(np.sum(np.power(tot_mon_pred_1sig_arr, 2), axis= 0)), tot_ann_obs_size_arr, \
                                                                                    np.sum(tot_ann_pred_size_arr, axis= 0), np.sqrt(np.sum(np.power(tot_ann_pred_1sig_arr, 2), axis= 0))
    
def mon_to_ann_size_func(mon_size_arr, mon_high_1sig_arr, mon_low_1sig_arr, yrarr):
    
    pred_sizes= np.array([np.sum(mon_size_arr[yrarr[i]:yrarr[i+1]]) for i in range(len(yrarr) - 1)])
    pred_mon_std= 0.5*(mon_high_1sig_arr - mon_low_1sig_arr)
    pred_std= np.asarray([np.sqrt(np.sum(np.power(pred_mon_std, 2)[yrarr[i]:yrarr[i+1]])) for i in range(len(yrarr) - 1)])
    pred_high_1sig= pred_sizes + pred_std
    pred_low_1sig= pred_sizes - pred_std
    pred_low_1sig[pred_low_1sig < 0]= 0
    
    return pred_sizes, pred_high_1sig, pred_low_1sig

def ml_fire_size_hyperparam_tuning(firefilepath, n_iters= 5, lnc_arr= [[2, 8, 2], [1, 16, 4]], func_flag_arr= ['gpd', 'lognorm_gpd'], rwt_flag_arr= [False, True], \
                            dropcols= ['CAPE', 'Solar', 'Ant_Tmax', 'RH', 'Ant_RH', 'FFWI_max3', 'Avgprec_4mo', 'Avgprec_2mo', 'AvgVPD_4mo', 'AvgVPD_2mo', 'Tmax_max7', 'VPD_max7', 'Tmin_max7'], \
                            start_month= 420, tot_test_months= 12, threshold= 4, scaled= False, loro_ind= None, run_id= None):
    
    list_of_lists = []
    
    for it in tqdm(range(n_iters)):
        rseed= np.random.randint(1000)
        n_features= 36
        end_month= start_month + tot_test_months
        
        if scaled:
            X_sizes_train, X_sizes_val, y_sizes_train, y_sizes_val, fire_size_train, fire_size_test, X_sizes_test, y_sizes_test= fire_size_data(res= '12km', \
                                                dropcols= dropcols, start_month= start_month, tot_test_months= tot_test_months, threshold= threshold, scaled= True, hyp_flag= True) #size_scaler
        else:
             X_sizes_train, X_sizes_val, y_sizes_train, y_sizes_val, fire_size_train, fire_size_test, X_sizes_test, y_sizes_test, size_scaler= fire_size_data(res= '12km', \
                                                dropcols= dropcols, start_month= start_month, tot_test_months= tot_test_months, threshold= threshold, scaled= False, hyp_flag= True)

        X_sizes_train_df= pd.concat([X_sizes_train, X_sizes_val], sort= False).reset_index().drop(columns=['index'])
        X_sizes_tot= pd.concat([X_sizes_train_df, X_sizes_test], sort= False).reset_index().drop(columns=['index'])
        fire_size_tot= pd.concat([fire_size_train, fire_size_test], sort= False).reset_index().drop(columns=['index'])

        fire_size_arr= np.concatenate([y_sizes_train, y_sizes_val])
        eff_freq_func= sampling_func(y_sizes_train)
        norm_fac= 1e-6
        samp_weight_arr= norm_fac/eff_freq_func(y_sizes_train)
        
        run_id_ind, itind, bs, p_frac= ['02_27_22', 2, 8192, '0.3'] #['03_01_22', 12, 8192, '0.3'] #['02_27_22', 2, 8192, '0.3']
        freq_loc_df= pd.read_hdf('../sav_files/fire_freq_pred_dfs/freq_loc_df_%s'%run_id_ind + '_%s'%bs + '_pfrac_%s'%str(p_frac) + '_iter_run_%d.h5'%itind)

        mdn_freq_df= pd.read_hdf('../sav_files/fire_freq_pred_dfs/mdn_mon_fire_freq_02_27_22_it_2_8192_0.3.h5')
        mdn_mon_freq_df, mdn_ann_freq_df= calib_freq_predict(ml_freq_df= mdn_freq_df, n_regs= 18, tot_months= 432, test_start= 372, test_tot= 60, ml_model= 'mdn')
        mdn_mon_freq_groups= mdn_mon_freq_df.groupby('reg_indx')
        mdn_ann_freq_groups= mdn_ann_freq_df.groupby('reg_indx')
        
        nregions= 18
        max_fire_train_arr= []
        sum_fire_train_arr= []

        for r in range(nregions):
            max_fire_train_arr.append(np.max(np.concatenate([fire_size_train.groupby('reg_indx').get_group(r+1).groupby('fire_month').get_group(k).fire_size.to_numpy()/1e6 \
                                            for k in fire_size_train.groupby('reg_indx').get_group(r+1).groupby('fire_month').groups.keys()])))
            #sum_fire_train_arr.append(np.max([np.sum(fire_size_train.groupby('reg_indx').get_group(r+1).groupby('fire_month').get_group(k).fire_size.to_numpy()/1e6) \
            #                                for k in fire_size_train.groupby('reg_indx').get_group(r+1).groupby('fire_month').groups.keys()]))

        max_fire_train_arr= np.asarray(max_fire_train_arr)
        sum_fire_train_arr= max_fire_size_sum_func(fire_size_df= fire_size_tot)

        for lnc in lnc_arr:
            for f in func_flag_arr:
                for rwt in rwt_flag_arr:
                    if rwt == True:
                        mdn_size_model, h_mdn= reg_fire_size_func(X_train_dat= X_sizes_train, y_train_dat= y_sizes_train, X_val_dat= X_sizes_val, y_val_dat= y_sizes_val, \
                                size_test_df= fire_size_test, X_test_dat= X_sizes_test, max_size_arr= max_fire_train_arr, sum_size_arr= sum_fire_train_arr, \
                                epochs= 1000, bs= 32, func_flag= f, lnc_arr= lnc, samp_weights= True, samp_weight_arr= samp_weight_arr, loco= True, rseed= rseed)
                        mdn_size_model.save('../sav_files/grid_size_runs_%s'%run_id + '/mdn_%s_'%f + '%d_layers_'%lnc[0] + '%d_comps_'%lnc[2] + 'ext_iter_run_%d'%(it+1))
                    else:
                        mdn_size_model, h_mdn= reg_fire_size_func(X_train_dat= X_sizes_train, y_train_dat= y_sizes_train, X_val_dat= X_sizes_val, y_val_dat= y_sizes_val, \
                                size_test_df= fire_size_test, X_test_dat= X_sizes_test, max_size_arr= max_fire_train_arr, sum_size_arr= sum_fire_train_arr, \
                                epochs= 1000, bs= 32, func_flag= f, lnc_arr= lnc, samp_weights= False, loco= True, rseed= rseed)
                        mdn_size_model.save('../sav_files/grid_size_runs_%s'%run_id + '/mdn_%s_'%f + '%d_layers_'%lnc[0] + '%d_comps_'%lnc[2] + 'iter_run_%d'%(it+1))
                    
                    if f == 'gpd':
                        stat_model= gpd_model
                    elif f == 'lognorm':
                        stat_model= lognorm_model
                    elif f == 'lognorm_gpd':
                        stat_model= lognorm_gpd_model_predict

                    reg_data_size_df= grid_size_pred_func(mdn_model= mdn_size_model, stat_model= stat_model, max_size_arr= max_fire_train_arr, sum_size_arr= sum_fire_train_arr, \
                                        start_month= 0, freq_flag= 'data', size_test_df= fire_size_tot, X_size_test_dat= X_sizes_tot, \
                                        debug= False, seed= 99)

                    tot_mon_obs_size_arr, tot_mon_pred_size_arr, tot_mon_pred_1sig_arr, tot_ann_obs_size_arr, \
                            tot_ann_pred_size_arr, tot_ann_pred_1sig_arr= cumm_fire_size_func(firefile= firefilepath, reg_size_df= reg_data_size_df, optflag= True)
                    if len(tot_mon_obs_size_arr) == 0:
                        tot_mon_obs_size_arr, tot_mon_pred_size_arr, tot_mon_pred_1sig_arr, tot_ann_obs_size_arr, \
                            tot_ann_pred_size_arr, tot_ann_pred_1sig_arr= cumm_fire_size_func(firefile= firefilepath, reg_size_df= reg_data_size_df, optflag= False)
                    tot_mon_size_r= stats.pearsonr(np.sum(tot_mon_obs_size_arr, axis= 0), tot_mon_pred_size_arr)[0]
                    tot_ann_size_r= stats.pearsonr(np.sum(tot_ann_obs_size_arr, axis= 0), tot_ann_pred_size_arr)[0]

                    list_of_lists.append([it+1, lnc[0], f, rwt, np.nanmean(h_mdn.history['%s_accuracy'%f]), tot_mon_size_r, tot_ann_size_r])

    hp_df= pd.DataFrame(list_of_lists, columns=["Iteration", "n_layers", "func_flag", "rwt_flag", "Val Accuracy/Recall", "Monthly corr", "Annual corr"])
    
    return hp_df

def theoretical_cdf_func(fire_size_df, X_size_df, mdn_mod, start_month, final_month, regmode= False, regindx= None):
    
    # Computes the analytic cdf of fire sizes with parameters predicted from a trained NN model
    
    totmonths= final_month - start_month
    regmodels= []
    tot_fires= 0
    if not regmode:
        for r in tqdm(range(18)):
            ml_param_vec= []
            fire_loc_arr= fire_loc_func_obs(fire_size_df, regindx= r+1, start_month= start_month, final_month= final_month)

            for m in np.linspace(0, totmonths - 1, totmonths, dtype= np.int64):
                if np.nonzero(fire_loc_arr[m])[0].size != 0:
                    ml_param_vec.append(mdn_mod.predict(x= np.array(X_size_df.iloc[fire_loc_arr[m]], dtype= np.float32)))

            regmodels.append(gpd_model(np.vstack(ml_param_vec)))
            tot_fires+= len(np.vstack(ml_param_vec))

        return regmodels, tot_fires
    else:
        ml_param_vec= []
        fire_loc_arr= fire_loc_func_obs(fire_size_df, regindx= regindx, start_month= start_month, final_month= final_month)

        for m in np.linspace(0, totmonths - 1, totmonths, dtype= np.int64):
            if np.nonzero(fire_loc_arr[m])[0].size != 0:
                ml_param_vec.append(mdn_mod.predict(x= np.array(X_size_df.iloc[fire_loc_arr[m]], dtype= np.float32)))

        regmodels.append(gpd_model(np.vstack(ml_param_vec)))
        tot_fires+= len(np.vstack(ml_param_vec))

        return regmodels, tot_fires
    
def mon_size_percentile_func(mdn_model, X_sizes_dat, fire_size_df, regindx, mindx, start_month= 0, final_month= 444, rseed= 99):
    
    # Calculates the mean and higher percentiles of the fire size distribution for monthly area burned in a given region
    
    params= mdn_model.predict(x= np.array(X_sizes_dat.iloc[fire_loc_func_obs(fire_size_df, regindx= regindx, start_month= start_month, \
                                                                                                 final_month= final_month)[mindx]], dtype= np.float32))
    samps= gpd_model(params).sample(1000, seed= rseed)
    #re_samps= tf.clip_by_value(samps, clip_value_min= 0, clip_value_max= max_fire_train_arr[regindx - 1], name=None)
    
    mean_size= tf.reduce_sum(tf.reduce_mean(samps, axis= 0)).numpy()
    p95_size= 1/1000*np.sum([tfp.stats.percentile(tf.reduce_sum(samps, axis= 1), 95, axis= 0) for i in range(1000)])
    p99_size= 1/1000*np.sum([tfp.stats.percentile(tf.reduce_sum(samps, axis= 1), 99, axis= 0) for i in range(1000)])
    p995_size= 1/1000*np.sum([tfp.stats.percentile(tf.reduce_sum(samps, axis= 1), 99.5, axis= 0) for i in range(1000)])
    
    return [mean_size, p95_size, p99_size, p995_size]

def ann_size_percentile_func(mdn_model, firefile, fire_size_df, X_sizes_dat, extyeararr, reglist= None, inf_fac= 3.0, start_month= 0, final_month= 444, rseed= 99):
    
    # Calculates the mean and higher percentiles of the fire size distribution for annual area burned across different regions
    
    n_regions= 18
    emp_burned_area= [] 
    pred_burned_area= []
    if reglist == None:
        reglist= range(n_regions)

    for m in tqdm(range(len(extyeararr))):
        smon= (extyeararr[m] - 1984)*12 
        tmpregarr= np.hstack([np.hstack(fire_loc_func_obs(fire_size_df, regindx= r+1, start_month= start_month, final_month= final_month)[smon:smon+12]) for r in reglist])
        emp_burned_area.append(np.sum([np.sum(mon_burned_area(firefile, regindx= r+1, final_year= 2020)[smon:smon+12]) for r in reglist]))

        tmpregarr= tmpregarr[tmpregarr!=0]
        params= mdn_model.predict(x= np.array(X_sizes_dat.iloc[tmpregarr], dtype= np.float32))
        samps= tf.clip_by_value(gpd_model(params).sample(1000, seed= rseed), clip_value_min= 0, clip_value_max= inf_fac*fire_size_df[fire_size_df.fire_month <= smon]['fire_size'].max()/1e6)

        pred_burned_area.append(tf.reduce_sum(samps, axis= 1).numpy())
    
    return pred_burned_area, emp_burned_area