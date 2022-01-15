import numpy as np
import pandas as pd
#from time import clock
from datetime import date, datetime, timedelta
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

def fire_freq_data(fire_freq_df, dropcols= ['index', 'Tmin', 'Solar', 'Ant_Tmax', 'RH', 'Ant_RH', 'Elev', 'Camp_dist']): #'Road_dist'
    
    # Returns the train/val/test data given an initial fire frequency df
    
    fire_freq_train= fire_freq_df[fire_freq_df.month < 372].reset_index().drop(columns=['index']) # for Training and Testing; 372 ==> Jan. 2015; ensures ~80/20 split
    fire_freq_test= fire_freq_df[fire_freq_df.month >= 372].reset_index().drop(columns=['index']) # for Prediction
    tmp_freq_df= fire_freq_df[fire_freq_df.iloc[:, 0:22].columns] #20 --> if dropping VPD/RH and their antecdent version
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

def freq_pred_func(mdn_model, X_test_dat, func_flag= 'zinb', l4_flag= False, reg_len_arr= None, modsave= False):
    
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
        
        reg_freq_df= pd.DataFrame({'mean_freq': pd.Series(dtype= 'int'), 'low_2sig_freq': pd.Series(dtype= 'int'), 'high_2sig_freq': pd.Series(dtype= 'int'), \
                                                                                       'reg_indx': pd.Series(dtype= 'int')})
        if not modsave:
            for i in tqdm(range(n_regions)): 
                param_vec= mdn_model.predict(x= tf.constant(X_test_dat[freq_arr_1[i]:freq_arr_2[i]]))
                freq_samp= stat_model(param_vec).sample(10000)
                reg_freq= tf.cast(tf.reduce_mean(freq_samp, axis= 0), tf.int64)
                reg_freq_sig= tf.cast(tf.math.reduce_std(freq_samp, axis= 0), tf.int64)
                #reg_freq_med= tfp.stats.percentile(sierra_freq_samp, 50.0, interpolation='midpoint', axis= 0)

                reg_freq_low= (reg_freq - 2*reg_freq_sig).numpy()
                reg_freq_low[reg_freq_low < 0]= 0
                reg_freq_high= (reg_freq + 2*reg_freq_sig).numpy()
                reg_indx_arr= (i+1)*np.ones(len(reg_freq), dtype= int)

                reg_freq_df= reg_freq_df.append(pd.DataFrame({'mean_freq': reg_freq.numpy(), 'low_2sig_freq': reg_freq_low, 'high_2sig_freq': reg_freq_high, \
                                                                                           'reg_indx': reg_indx_arr}), ignore_index=True)
        else:
            for i in range(n_regions): 
                param_vec= mdn_model.predict(x= tf.constant(X_test_dat[freq_arr_1[i]:freq_arr_2[i]]))
                freq_samp= stat_model(param_vec).sample(10000)
                reg_freq= tf.cast(tf.reduce_mean(freq_samp, axis= 0), tf.int64)
                reg_freq_sig= tf.cast(tf.math.reduce_std(freq_samp, axis= 0), tf.int64)
                #reg_freq_med= tfp.stats.percentile(sierra_freq_samp, 50.0, interpolation='midpoint', axis= 0)

                reg_freq_low= (reg_freq - 2*reg_freq_sig).numpy()
                reg_freq_low[reg_freq_low < 0]= 0
                reg_freq_high= (reg_freq + 2*reg_freq_sig).numpy()
                reg_indx_arr= (i+1)*np.ones(len(reg_freq), dtype= int)

                reg_freq_df= reg_freq_df.append(pd.DataFrame({'mean_freq': reg_freq.numpy(), 'low_2sig_freq': reg_freq_low, 'high_2sig_freq': reg_freq_high, \
                                                                                           'reg_indx': reg_indx_arr}), ignore_index=True) 
        
    else:
        cumlenarr= np.insert(np.cumsum(reg_len_arr), 0, 0)
        reg_freq_df= pd.DataFrame({'mean_freq': pd.Series(dtype= 'int'), 'std_freq': pd.Series(dtype= 'int'), 'reg_indx': pd.Series(dtype= 'int')})
        if not modsave:
            for i in tqdm(range(n_regions)): 
                param_vec= mdn_model.predict(x= tf.constant(X_test_dat[cumlenarr[i]:cumlenarr[i+1]]))
                freq_samp= stat_model(param_vec).sample(10000)
                reg_freq= tf.cast(tf.reduce_mean(freq_samp, axis= 0), tf.int64)
                reg_freq_sig= tf.cast(tf.math.reduce_std(freq_samp, axis= 0), tf.int64)

                reg_indx_arr= (i+1)*np.ones(len(reg_freq), dtype= int)

                reg_freq_df= reg_freq_df.append(pd.DataFrame({'mean_freq': reg_freq.numpy(), 'std_freq': reg_freq_sig.numpy(), 'reg_indx': reg_indx_arr}), \
                                                                                                                                            ignore_index=True)
        else:
            for i in range(n_regions): #n_regions
                param_vec= mdn_model.predict(x= tf.constant(X_test_dat[cumlenarr[i]:cumlenarr[i+1]]))
                freq_samp= stat_model(param_vec).sample(10000)
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
    
def freq_acc_func(mdn_model, obs_input, obs_freqs, func_flag= 'zinb'):
    
    if func_flag == 'zinb':
        acc_func= zinb_accuracy
    elif func_flag == 'zipd':
        acc_func= zipd_accuracy
    param_vec= mdn_model.predict(x= tf.constant(obs_input))
    
    return acc_func(obs_freqs, param_vec).numpy()
    
def fire_freq_predict(fire_L3_freq_df, fire_L4_freq_df, n_iters= 5, n_epochs= 10, bs= 32):
    
    # Evaluates the chisq and Pearson's correlation for observed and predicted fire frequencies for a variety of hyperparameters
    
    X_L3_freqs_train, X_L3_freqs_val, y_L3_freqs_train, y_L3_freqs_val, fire_L3_freq_test, X_L3_freqs_test, y_L3_freqs_test, L3_freq_samp_weight_arr= fire_freq_data(fire_L3_freq_df)
    X_L4_freqs_train, X_L4_freqs_val, y_L4_freqs_train, y_L4_freqs_val, fire_L4_freq_test, X_L4_freqs_test, y_L4_freqs_test, L4_freq_samp_weight_arr= fire_freq_data(fire_L4_freq_df)
    
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
                            accuracy= freq_acc_func(mdn_model= mdn_L3_freq_zinb, obs_input= obs_input, obs_freqs= obs_freqs, func_flag= 'zinb')
                        else:
                            reg_L3_freq_groups= reg_L3_freq_zipd_groups
                            accuracy= freq_acc_func(mdn_model= mdn_L3_freq_zipd, obs_input= obs_input, obs_freqs= obs_freqs, func_flag= 'zipd')

                        mean_freqs= reg_L3_freq_groups.get_group(regindx + 1)['mean_freq']
                        high_freqs= reg_L3_freq_groups.get_group(regindx + 1)['high_2sig_freq']
                        low_freqs= reg_L3_freq_groups.get_group(regindx + 1)['low_2sig_freq']

                        pearson_r= stats.pearsonr(obs_freqs, mean_freqs)
                        errarr= 16*(mean_freqs - obs_freqs)**2/(high_freqs - low_freqs)**2
                        chisq= np.sum(errarr[np.isfinite(errarr)])
                        #dof= len(errarr[np.isfinite(errarr)]) #+ mdn.count_params() - 1
                        
                        list_of_lists.append([it + 1, regindx + 1, l, f, pearson_r[0], chisq, accuracy])

                    elif l == 'L4':
                        l4_freqs= y_L4_freqs_test[cumreglen[regindx]:cumreglen[regindx + 1]]
                        obs_input= X_L4_freqs_test[cumreglen[regindx]:cumreglen[regindx + 1]]
                        if f == 'zinb':
                            reg_L4_freq_groups= reg_L4_freq_zinb_groups
                            accuracy= freq_acc_func(mdn_model= mdn_L4_freq_zinb, obs_input= obs_input, obs_freqs= l4_freqs, func_flag= 'zinb')
                        else:
                            reg_L4_freq_groups= reg_L4_freq_zipd_groups
                            accuracy= freq_acc_func(mdn_model= mdn_L4_freq_zipd, obs_input= obs_input, obs_freqs= l4_freqs, func_flag= 'zipd')

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
                        errarr= 16*(mean_freqs - obs_freqs)**2/(high_freqs - low_freqs)**2
                        chisq= np.sum(errarr[np.isfinite(errarr)])
                        #dof= len(errarr[np.isfinite(errarr)]) #+ mdn.count_params() - 1 #might be misleading due to high number of NaNs

                        list_of_lists.append([it + 1, regindx + 1, l, f, pearson_r[0], chisq, accuracy])

    hp_df= pd.DataFrame(list_of_lists, columns=["Iteration", "reg_indx", "reg_flag", "func_flag", "Pearson_r", "Red_ChiSq", "Accuracy"])
    return hp_df