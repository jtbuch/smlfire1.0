import numpy as np
import pandas as pd
#from time import clock
from datetime import datetime, timedelta
from cftime import num2date, date2num, DatetimeGregorian
from tqdm import tqdm

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
from fire_utils import ncdump, coord_transform, bailey_ecoprovince_shp, bailey_ecoprovince_mask, update_reg_indx, mon_fire_freq, mon_burned_area, tindx_func, clim_pred_var  
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

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

class SeqBlock(tf.keras.layers.Layer):
    
<<<<<<< HEAD
    def __init__(self, hidden_l= 2, n_neurs=100, initializer= "glorot_uniform", reg= False, regrate= None, dropout= False):
=======
    def __init__(self, hidden_l= 2, n_neurs=100, initializer= "glorot_uniform", reg= False, l2rate= None, dropout= False):
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3
        super(SeqBlock, self).__init__(name="SeqBlock")
        self.nnmodel= tf.keras.Sequential()
        for l in range(hidden_l):
            if reg:
                self.nnmodel.add(Dense(n_neurs, activation="relu",
                                    kernel_initializer= initializer,
<<<<<<< HEAD
                                    kernel_regularizer=tf.keras.regularizers.l2(regrate),
=======
                                    kernel_regularizer=tf.keras.regularizers.l2(l2rate),
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3
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

<<<<<<< HEAD
    def __init__(self, layers= 2, neurons=10, components = 2, initializer= "glorot_uniform", reg= False, regrate= None, dropout= False):
=======
    def __init__(self, layers= 2, neurons=10, components = 2, initializer= "glorot_uniform", reg= False, l2rate= None, dropout= False):
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3
        super(MDN_size, self).__init__(name="MDN_size")
        self.neurons = neurons
        self.components = components
        self.n_hidden_layers= layers
        
        #hidden layers
<<<<<<< HEAD
        self.seqblock= SeqBlock(layers, neurons, initializer, reg, regrate, dropout)
=======
        self.seqblock= SeqBlock(layers, neurons, initializer, reg, l2rate, dropout)
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3
        
        #output layer
        if reg:
            self.outlayer= Dense(3*components, activation="relu",
                                    kernel_initializer= initializer,
<<<<<<< HEAD
                                    kernel_regularizer=tf.keras.regularizers.l2(regrate),
=======
                                    kernel_regularizer=tf.keras.regularizers.l2(l2rate),
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3
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
<<<<<<< HEAD
        #x0, x1, x2= tf.split(x, 3, axis= 1)
=======
        x0, x1, x2= tf.split(x, 3, axis= 1)
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3
        
        alpha_v = self.alphas(x0) 
        distparam1_v = self.distparam1(x1)
        distparam2_v = self.distparam2(x2)
        
        return self.pvec([alpha_v, distparam1_v, distparam2_v])

class MDN_freq(tf.keras.Model):

<<<<<<< HEAD
    def __init__(self, layers= 2, neurons=10, components = 1, initializer= "glorot_uniform", reg= False, regrate= None, dropout= False, func_type= 'zinb'):
=======
    def __init__(self, layers= 2, neurons=10, components = 1, initializer= "glorot_uniform", reg= False, l2rate= None, dropout= False, func_type= 'zinb'):
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3
        super(MDN_freq, self).__init__(name="MDN_freq")
        self.neurons = neurons
        self.components = components
        self.n_hidden_layers= layers
        
        #hidden layers
<<<<<<< HEAD
        self.seqblock= SeqBlock(layers, neurons, initializer, reg, regrate, dropout)
=======
        self.seqblock= SeqBlock(layers, neurons, initializer, reg, l2rate, dropout)
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3
        
        #output layer
        if reg:
            self.outlayer= Dense(3*components, activation="relu",
                                    kernel_initializer= initializer,
<<<<<<< HEAD
                                    kernel_regularizer=tf.keras.regularizers.l2(regrate),
=======
                                    kernel_regularizer=tf.keras.regularizers.l2(l2rate),
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3
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
<<<<<<< HEAD
        #x0, x1, x2= tf.split(x, 3, axis= 1)
=======
        x0, x1, x2= tf.split(x, 3, axis= 1)
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3
        
        pi_v = self.pi(x0) 
        mu_v = self.mu(x1)
        delta_v = self.delta(x2)
        
        return self.pvec([pi_v, mu_v, delta_v])

<<<<<<< HEAD
def hyperparam_tuning(n_layers, n_neurons, n_components= None, bs= 128, epochs= 1000, lr= 1e-4, X_dat= None, y_dat= None, fire_tag= 'size', func_flag= 'gpd', samp_weights= False, samp_weight_arr= None):
=======
def hyperparam_tuning(n_layers, n_neurons, n_components= None, bs= 128, epochs= 1000, lr= 1e-4, X_dat= None, y_dat= None, fire_tag= 'size', func_flag= 'gpd'):
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3
    
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
<<<<<<< HEAD
                    if samp_weights == False:
                        hp.fit(x=X_dat, y=y_dat, batch_size= bs, epochs= epochs, verbose=0)
                    else:
                        hp.fit(x=X_dat, y=y_dat, batch_size= bs, epochs= epochs, sample_weight= samp_weight_arr, verbose=0)
=======
                    hp.fit(x=X_dat, y=y_dat, batch_size= bs, epochs= epochs, verbose=0)
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3

                    loss, accuracy= hp.evaluate(X_dat, y_dat, verbose=0)
                    list_of_lists.append([n_layers[i], nn, loss, accuracy])
                    hp= hp.reset_states()
        
        hp_df= pd.DataFrame(list_of_lists, columns=["n_layers", "n_neurons", "Loss", "Accuracy"])                 

    hp_df.to_hdf('../sav_files/%s_'%fire_tag + '%s_'%func_flag +'hp_tuning.h5', key='df', mode= 'w')
    
    return hp_df


<<<<<<< HEAD
def validation_cycle(n_layers, n_neurons, n_components= None, num_iterations= 5, bs= 128, epochs= 100, lr= 1e-4, X_dat= None, y_dat= None, X_val_dat= None, y_val_dat= None, fire_tag= 'size', func_flag= 'gpd', samp_weights= False, samp_weight_arr= None):
=======
def validation_cycle(n_layers, n_neurons, n_components= None, num_iterations= 5, bs= 128, epochs= 100, lr= 1e-4, X_dat= None, y_dat= None, X_val_dat= None, y_val_dat= None, fire_tag= 'size', func_flag= 'gpd'):
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3

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
<<<<<<< HEAD
        if samp_weights == False:
            mdnhist= mdn_val.fit(x= X_dat, y= y_dat, batch_size= bs, epochs= epochs, validation_data=(X_val_dat, y_val_dat), verbose=0)
        else:
            mdnhist= mdn_val.fit(x= X_dat, y= y_dat, batch_size= bs, epochs= epochs, validation_data=(X_val_dat, y_val_dat), sample_weight= samp_weight_arr, verbose=0)
=======
        mdnhist= mdn_val.fit(x= X_dat, y= y_dat, batch_size= bs, epochs= epochs, validation_data=(X_val_dat, y_val_dat), verbose=0)
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3
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

<<<<<<< HEAD
def reg_fire_freq_func(X_train_dat, y_train_dat, X_val_dat, y_val_dat, X_test_dat, n_layers= 2, lr= 1e-4, n_neurons= 16, bs= 32, epochs= 500, func_flag= 'zinb', rseed= None, samp_weights= False, samp_weight_arr= None):
    
    # Calculates the predicted fire frequency as well as its 1 sigma uncertainty for all regions
    
    if rseed == None:
        rseed= np.random.randint(100)
    tf.random.set_seed(rseed)
=======
def reg_fire_freq_func(X_train_dat, y_train_dat, X_val_dat, y_val_dat, X_test_dat, n_layers= 2, n_neurons= 16, bs= 32, func_flag= 'zinb'):
    
    # Calculates the predicted fire frequency as well as its 1 sigma uncertainty for all regions
    
    tf.random.set_seed(56)
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3

    if func_flag == 'zinb':
        stat_model= zinb_model
        loss_func= zinb_loss
        acc_func= zinb_accuracy
    elif func_flag == 'zipd':
        stat_model= zipd_model
        loss_func= zipd_loss
        acc_func= zipd_accuracy
    
    n_regions= 18
    freq_test_size= np.int(len(X_test_dat)/n_regions)
    freq_arr_1= np.linspace(0, len(X_test_dat) - freq_test_size, n_regions, dtype= int)
    freq_arr_2= freq_arr_1 + freq_test_size
    
    mon= EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    mdn= MDN_freq(layers= n_layers, neurons= n_neurons)
<<<<<<< HEAD
    mdn.compile(loss=loss_func, optimizer= tf.keras.optimizers.Adam(learning_rate= lr), metrics=[acc_func])
    #h= mdn.fit(x= X_train_dat, y= y_train_dat, epochs= epochs, validation_data=(X_val_dat, y_val_dat), callbacks= [mon], batch_size= bs, verbose=0)
    if samp_weights == False:
        h= mdn.fit(x= X_train_dat, y= y_train_dat, epochs= epochs, validation_data=(X_val_dat, y_val_dat), callbacks= [mon], batch_size= bs, verbose=0)
    else:
        h= mdn.fit(x= X_train_dat, y= y_train_dat, epochs= epochs, validation_data=(X_val_dat, y_val_dat), callbacks= [mon], batch_size= bs, \
                                                                                                             sample_weight= samp_weight_arr, verbose=0)
=======
    mdn.compile(loss=loss_func, optimizer= tf.keras.optimizers.Adam(learning_rate= 1e-4), metrics=[acc_func])
    h= mdn.fit(x= X_train_dat, y= y_train_dat, epochs= 500, validation_data=(X_val_dat, y_val_dat), callbacks= [mon], batch_size= bs, verbose=0) #callbacks= [mon]
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3
    print("MDN trained for %d epochs"%len(h.history['loss']))
    
    reg_freq_df= pd.DataFrame({'mean_freq': pd.Series(dtype= 'int'), 'low_1sig_freq': pd.Series(dtype= 'int'), 'high_1sig_freq': pd.Series(dtype= 'int'), \
                                                                                       'reg_indx': pd.Series(dtype= 'int')})
    
    for i in tqdm(range(n_regions)): #n_regions
        param_vec= mdn.predict(x= tf.constant(X_test_dat[freq_arr_1[i]:freq_arr_2[i]]))
        freq_samp= stat_model(param_vec).sample(10000)
        reg_freq= tf.cast(tf.reduce_mean(freq_samp, axis= 0), tf.int64)
        reg_freq_sig= tf.cast(tf.math.reduce_std(freq_samp, axis= 0), tf.int64)
        #reg_freq_med= tfp.stats.percentile(sierra_freq_samp, 50.0, interpolation='midpoint', axis= 0)

        reg_freq_low= (reg_freq - 2*reg_freq_sig).numpy()
        reg_freq_low[reg_freq_low < 0]= 0
        reg_freq_high= (reg_freq + 2*reg_freq_sig).numpy()
        reg_indx_arr= (i+1)*np.ones(len(reg_freq), dtype= int)
        
        reg_freq_df= reg_freq_df.append(pd.DataFrame({'mean_freq': reg_freq.numpy(), 'low_1sig_freq': reg_freq_low, 'high_1sig_freq': reg_freq_high, \
                                                                                   'reg_indx': reg_indx_arr}), ignore_index=True)
    
    return reg_freq_df, h


<<<<<<< HEAD
def reg_fire_freq_L4_func(X_train_dat, y_train_dat, X_val_dat, y_val_dat, X_test_dat, reg_len_arr, lr= 1e-4, n_layers= 2, n_neurons= 16, epochs= 500, bs= 32, func_flag= 'zinb', rseed= None, samp_weights= False, samp_weight_arr= None):
    
    # Calculates the predicted fire frequency as well as its n sigma uncertainty for all regions
    
    if rseed == None:
        rseed= np.random.randint(100)
    tf.random.set_seed(rseed)
=======
def reg_fire_freq_L4_func(X_train_dat, y_train_dat, X_val_dat, y_val_dat, X_test_dat, reg_len_arr, n_layers= 2, n_neurons= 16, bs= 32, func_flag= 'zinb'):
    
    # Calculates the predicted fire frequency as well as its n sigma uncertainty for all regions
    
    #tf.random.set_seed(99)
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3

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
    #freq_test_size= np.int(len(X_test_dat)/n_regions)
    #freq_arr_1= np.linspace(0, len(X_test_dat) - freq_test_size, n_regions, dtype= int)
    #freq_arr_2= freq_arr_1 + freq_test_size
    
    print("Initialized a MDN with %d layers"%n_layers + " and %d neurons"%n_neurons)
    
    mon= EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    mdn= MDN_freq(layers= n_layers, neurons= n_neurons)
<<<<<<< HEAD
    mdn.compile(loss=loss_func, optimizer= tf.keras.optimizers.Adam(learning_rate= lr), metrics=[acc_func])
    if samp_weights == False:
        h= mdn.fit(x= X_train_dat, y= y_train_dat, epochs= epochs, validation_data=(X_val_dat, y_val_dat), callbacks= [mon], batch_size= bs, verbose=0)
    else:
        h= mdn.fit(x= X_train_dat, y= y_train_dat, epochs= epochs, validation_data=(X_val_dat, y_val_dat), batch_size= bs, \
                                                                                                             sample_weight= samp_weight_arr, verbose=0) #callbacks= [mon]
=======
    mdn.compile(loss=loss_func, optimizer= tf.keras.optimizers.Adam(learning_rate= 1e-4), metrics=[acc_func])
    h= mdn.fit(x= X_train_dat, y= y_train_dat, epochs= 500, validation_data=(X_val_dat, y_val_dat), callbacks= [mon], batch_size= bs, verbose=0) #callbacks= [mon]
>>>>>>> 17d6a8e194e84f3c3372867017c8550cb4ff7cc3
    print("MDN trained for %d epochs"%len(h.history['loss']))
    
    reg_freq_df= pd.DataFrame({'mean_freq': pd.Series(dtype= 'int'), 'low_1sig_freq': pd.Series(dtype= 'int'), 'high_1sig_freq': pd.Series(dtype= 'int'), \
                                                                                       'reg_indx': pd.Series(dtype= 'int')})
    
    for i in tqdm(range(n_regions)): #n_regions
        param_vec= mdn.predict(x= tf.constant(X_test_dat[cumlenarr[i]:cumlenarr[i+1]]))
        freq_samp= stat_model(param_vec).sample(10000)
        reg_freq= tf.cast(tf.reduce_mean(freq_samp, axis= 0), tf.int64)
        reg_freq_sig= tf.cast(tf.math.reduce_std(freq_samp, axis= 0), tf.int64)
        #reg_freq_med= tfp.stats.percentile(sierra_freq_samp, 50.0, interpolation='midpoint', axis= 0)

        reg_indx_arr= (i+1)*np.ones(len(reg_freq), dtype= int)
        
        reg_freq_df= reg_freq_df.append(pd.DataFrame({'mean_freq': reg_freq.numpy(), 'std_freq': reg_freq_sig.numpy(), 'reg_indx': reg_indx_arr}), \
                                                                                                                                        ignore_index=True)
    
    return reg_freq_df, h