import numpy as np
import pickle

import os
import pickle
import sys
#from data_generator import Dataset

import pandas as pd

sys.path.append('..')
sys.path.append('../..')
#import tensorflow as tf
import urllib

import numpy as np


class Concrete(object):

    def __init__(self):
        self.load_raw_data()

    def load_raw_data(self):
        data_file='concrete_data.csv'
        dirname, _ = os.path.split(os.path.abspath(__file__))
        data = pd.read_csv(os.path.realpath(os.path.join(dirname,"concrete", data_file)), sep=",", header=None)
        X = data.values[:,0:-1]
        Y = -data.values[:,-1].reshape((-1,1))
        self.X= X
        self.Y=Y

    def get_normalised_data(self):
        self.load_raw_data()

        self.X1 = (self.X - np.mean(self.X, axis=0))/np.std(self.X, axis=0)
        self.Y1 = (self.Y - np.mean(self.Y, axis=0))/np.std(self.Y, axis=0)
        return self.X1, self.Y1


class Wine(object):

    def __init__(self):
        self.load_raw_data()

    def load_raw_data(self):
        data_file='wine_red_data.csv'
        #data_file = 'wine_all_data.csv'
        dirname, _ = os.path.split(os.path.abspath(__file__))
        data = pd.read_csv(os.path.realpath(os.path.join("datasets/wine", data_file)), sep=" ", header=None)
        X = data.values[:,0:-1]
        Y = -data.values[:,-1].reshape((-1,1))
        self.X= X
        self.Y=Y

    def get_normalised_data(self):
        self.load_raw_data()

        self.X1 = (self.X - np.mean(self.X, axis=0))/np.std(self.X, axis=0)
        self.Y1 = (self.Y - np.mean(self.Y, axis=0))/np.std(self.Y, axis=0)
        return self.X1, self.Y1


class Yacht(object):

    def __init__(self):
        self.load_raw_data()

    def load_raw_data(self):
        data_file = 'yacht_data.csv'
        dirname, _ = os.path.split(os.path.abspath(__file__))
        data = pd.read_csv(os.path.realpath(os.path.join(dirname,"datasets/yacht", data_file)), sep=" ", header=None)
        X = data.values[:,0:-1]
        Y = -data.values[:,-1].reshape((-1,1))
        self.X= X
        self.Y=Y

    def get_normalised_data(self):
        self.load_raw_data()

        self.X1 = (self.X - np.mean(self.X, axis=0))/np.std(self.X, axis=0)
        self.Y1 = (self.Y - np.mean(self.Y, axis=0))/np.std(self.Y, axis=0)
        return self.X1, self.Y1


class Boston(object):
    def __init__(self):
        self.load_raw_data()

    def load_raw_data(self):
        data_file = 'boston.txt'
        dirname, _ = os.path.split(os.path.abspath(__file__))
        data = pd.read_csv(os.path.realpath(os.path.join(dirname,"boston", data_file)), sep=" ", header=None)
        data = np.loadtxt(os.path.realpath(os.path.join(dirname, "datasets/boston/", data_file)))
        X = data.values[:,0:-1]
        Y = -data.values[:,-1].reshape((-1,1))
        self.X= X
        self.Y=Y

    def get_normalised_data(self):
        self.load_raw_data()

        self.X1 = (self.X - np.mean(self.X, axis=0))/np.std(self.X, axis=0)
        self.Y1 = (self.Y - np.mean(self.Y, axis=0))/np.std(self.Y, axis=0)
        return self.X1, self.Y1


class Boston(object):
    def __init__(self):
        self.load_raw_data()

    def load_raw_data(self):
        data_file = 'boston.txt'
        dirname, _ = os.path.split(os.path.abspath(__file__))
        #data = pd.read_csv(os.path.realpath(os.path.join("datasets/boston", data_file)), sep=" ", header=None)
        data= np.loadtxt(os.path.realpath(os.path.join("datasets/boston/", data_file)))
        X = data[:,0:-1]
        Y = data[:,-1].reshape((-1,1))
        self.X= X
        self.Y=Y

    def get_normalised_data(self):
        self.load_raw_data()

        self.X1 = (self.X - np.mean(self.X, axis=0))/np.std(self.X, axis=0)
        self.Y1 = (self.Y - np.mean(self.Y, axis=0))/np.std(self.Y, axis=0)
        return self.X1, self.Y1



CACHE_DIR = os.path.join(os.sep, 'tmp', 'radon')



def preprocess_radon_dataset(srrs2, cty, state='MN'):
  """Preprocess radon dataset as done in "Bayesian Data Analysis" book."""
  srrs2 = srrs2[srrs2.state == state].copy()
  cty = cty[cty.st == state].copy()
  # We will now join datasets on Federal Information Processing Standards
  # (FIPS) id, ie, codes that link geographic units, counties and county
  # equivalents. http://jeffgill.org/Teaching/rpqm_9.pdf
  srrs2['fips'] = 1000 * srrs2.stfips + srrs2.cntyfips
  cty['fips'] = 1000 * cty.stfips + cty.ctfips

  df = srrs2.merge(cty[['fips', 'Uppm']], on='fips')
  df = df.drop_duplicates(subset='idnum')
  df = df.rename(index=str, columns={'Uppm': 'uranium_ppm'})

  # For any missing or invalid activity readings, we'll use a value of `0.1`.
  df['radon'] = df.activity.apply(lambda x: x if x > 0. else 0.1)

  # Remap categories to start from 0 and end at max(category).
  county_name = sorted(df.county.unique())
  df['county'] = df.county.astype(
      pd.api.types.CategoricalDtype(categories=county_name)).cat.codes
  county_name = list(map(str.strip, county_name))

  df['log_radon'] = df['radon'].apply(np.log)
  df['log_uranium_ppm'] = df['uranium_ppm'].apply(np.log)
  df = df[['idnum', 'log_radon', 'floor', 'county', 'log_uranium_ppm']]

  return df, county_name
