import numpy as np
import pandas as pd
from time import clock
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
from fire_utils import ncdump, coord_transform, bailey_ecoprovince_shp, bailey_ecoprovince_mask, update_reg_indx, mon_fire_freq, tindx_func, clim_pred_var  
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
import seaborn as sns
%matplotlib inline
%config IPython.matplotlib.backend = 'retina'
%config InlineBackend.figure_format = 'retina'

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


class SeqBlock(tf.keras.layers.Layer):
    
    def __init__(self, hidden_l= 2, n_neurs=100, initializer= "glorot_uniform"):
        super(SeqBlock, self).__init__(name="SeqBlock")
        self.nnmodel= tf.keras.Sequential()
        for l in range(hidden_l):
            self.nnmodel.add(Dense(n_neurs, activation="relu", kernel_initializer= initializer, name="h_%d"%(l+1)))
            self.nnmodel.add(tf.keras.layers.LayerNormalization())

    def call(self, inputs):
        return self.nnmodel(inputs)
    
class MDN_size(tf.keras.Model):

    def __init__(self, layers= 2, neurons=10, components = 2, initializer= "glorot_uniform"):
        super(MDN_size, self).__init__(name="MDN_size")
        self.neurons = neurons
        self.components = components
        self.n_hidden_layers= layers
        
        self.seqblock= SeqBlock(layers, neurons, initializer)
        self.linreg= Dense(neurons, activation="linear", kernel_initializer= initializer, name="linear")
        
        self.alphas = Dense(components, activation="softmax", kernel_initializer= initializer, name="alphas")
        self.distparam1 = Dense(components, activation="softplus", kernel_initializer= initializer, name="distparam1")
        self.distparam2 = Dense(components, activation="softplus", kernel_initializer= initializer, name="distparam2")
        self.pvec = Concatenate(name="pvec")
        
    def call(self, inputs):
        x = self.seqblock(inputs) + self.linreg(inputs)
        
        alpha_v = self.alphas(x) 
        distparam1_v = self.distparam1(x)
        distparam2_v = self.distparam2(x)
        
        return self.pvec([alpha_v, distparam1_v, distparam2_v])

class MDN_freq(tf.keras.Model):

    def __init__(self, layers= 2, neurons=10, components = 1, initializer= "glorot_uniform", func_type= 'zinb'):
        super(MDN_freq, self).__init__(name="MDN_freq")
        self.neurons = neurons
        self.components = components
        self.n_hidden_layers= layers
        
        self.seqblock= SeqBlock(layers, neurons, initializer)
        self.linreg= Dense(neurons, activation="linear", kernel_initializer= initializer, name="linear")
        
        self.pi = Dense(components, activation="sigmoid", kernel_initializer= initializer, name="pi")
        self.mu = Dense(components, activation="softplus", kernel_initializer= initializer, name="mu")
        if func_type == 'zipd':
            self.delta= Dense(components, activation="gelu", kernel_initializer= initializer, name="delta") #ensures gaussian error in delta
        else:
            self.delta= Dense(components, activation="softplus", kernel_initializer= initializer, name="delta")
        self.pvec = Concatenate(name="pvec")
        
    def call(self, inputs):
        x = self.seqblock(inputs) + self.linreg(inputs)
        
        pi_v = self.pi(x) 
        mu_v = self.mu(x)
        delta_v = self.delta(x)
        
        return self.pvec([pi_v, mu_v, delta_v])

def hyperparam_tuning(n_layers, n_neurons, n_components= None, X_dat= None, y_dat= None, fire_tag= 'size', func_flag= 'gpd'):
    
    # Function for tuning the hyperparamters of the MDNs to determine fire properties
    
    opt= tf.keras.optimizers.Adam(learning_rate= 1e-4)
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
                    hp.fit(x=X_dat, y=y_dat, epochs=50, verbose=0)

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
                    hp.fit(x=X_dat, y=y_dat, epochs=50, verbose=0)

                    loss, accuracy= hp.evaluate(X_dat, y_dat, verbose=0)
                    list_of_lists.append([n_layers[i], nn, loss, accuracy])
                    hp= hp.reset_states()
        
        hp_df= pd.DataFrame(list_of_lists, columns=["n_layers", "n_neurons", "Loss", "Accuracy"])                 

    hp_df.to_hdf('../sav_files/%s_'%fire_tag + '%s_'%func_flag +'hp_tuning.h5', key='df', mode= 'w')
    
    return hp_df


def validation_cycle(n_layers, n_neurons, n_components= None, num_iterations= 5, X_dat= None, y_dat= None, X_val_dat= None, y_val_dat= None, fire_tag= 'size', func_flag= 'gpd'):

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
    
    opt= tf.keras.optimizers.Adam(learning_rate= 1e-4)
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
        mdnhist= mdn_val.fit(x= X_dat, y= y_dat, epochs=100, validation_data=(X_val_dat, y_val_dat), batch_size= 128, verbose=0)
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