#Author: Sebastian Langan
#Institution: University of Maryland, College Park
#Date: 5/16/2021
#Goals of this Project: Build a ML (or other predictive, Stats/Computational Model) Model and Program to Diagnose an Illness using EEG Data Analysis. 
#5/21/2021 Update: Based on Data Availablity Online, it's likely I'll try and do Major Depressive Disorder as my Illness of choice for this project.

import numpy as np
import scipy as sci
from scipy.fft import fft, fftfreq
import sklearn as sk
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.model_selection import cross_val_score 
from sklearn import svm
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D


# STEP 1: IMPORTING ALL RAW EEG DATA FROM CAVANAGH STUDY PARTICIPANTS AND CONDUCTING INITIAL SIGNAL PROCESSING FROM TIME --> FREQUENCY DOMAIN: #

#Reading in the Participants' Raw EEG Data as Data Frames and then immediately converting them to NumPy arrays that are ready for processing by the FFT, one at a time for now. 
pd.set_option("display.max_rows", None, "display.max_columns", None)

#Participant 507: 
participant507EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/507_Depression_REST_trimmed.csv")
participant507EEGData = participant507EEGDataDF.to_numpy()

#Participant 508: 
participant508EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/508_Depression_REST_trimmed.csv")
participant508EEGData = participant508EEGDataDF.to_numpy()

#Participant 509: 
participant509EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/509_Depression_REST_trimmed.csv")
participant509EEGData = participant509EEGDataDF.to_numpy()

#Participant 510: 
participant510EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/510_Depression_REST_trimmed.csv")
participant510EEGData = participant510EEGDataDF.to_numpy()

#Participant 511: 
participant511EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/511_Depression_REST_trimmed.csv")
participant511EEGData = participant511EEGDataDF.to_numpy()

#Participant 512: 
participant512EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/512_Depression_REST_trimmed.csv")
participant512EEGData = participant512EEGDataDF.to_numpy()

#Participant 513: 
participant513EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/513_Depression_REST_trimmed.csv")
participant513EEGData = participant513EEGDataDF.to_numpy()

#Participant 514: 
participant514EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/514_Depression_REST_trimmed.csv")
participant514EEGData = participant514EEGDataDF.to_numpy()

#Participant 515: 
participant515EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/515_Depression_REST_trimmed.csv")
participant515EEGData = participant515EEGDataDF.to_numpy()

#Participant 516: 
participant516EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/516_Depression_REST_trimmed.csv")
participant516EEGData = participant516EEGDataDF.to_numpy()

#Participant 517: 
participant517EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/517_Depression_REST_trimmed.csv")
participant517EEGData = participant517EEGDataDF.to_numpy()

#Participant 518: 
participant518EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/518_Depression_REST_trimmed.csv")
participant518EEGData = participant518EEGDataDF.to_numpy()

#Participant 519: 
participant519EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/519_Depression_REST_trimmed.csv")
participant519EEGData = participant519EEGDataDF.to_numpy()

#Participant 520: 
participant520EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/520_Depression_REST_trimmed.csv")
participant520EEGData = participant520EEGDataDF.to_numpy()

#Participant 521: 
participant521EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/521_Depression_REST_trimmed.csv")
participant521EEGData = participant521EEGDataDF.to_numpy()

#Participant 522: 
participant522EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/522_Depression_REST_trimmed.csv")
participant522EEGData = participant522EEGDataDF.to_numpy()

#Participant 523: 
participant523EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/523_Depression_REST_trimmed.csv")
participant523EEGData = participant523EEGDataDF.to_numpy()

#Participant 524: 
participant524EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/524_Depression_REST_trimmed.csv")
participant524EEGData = participant524EEGDataDF.to_numpy()

#Participant 526: 
participant526EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/526_Depression_REST_trimmed.csv")
participant526EEGData = participant526EEGDataDF.to_numpy()

#Participant 527: 
participant527EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/513_Depression_REST_trimmed.csv")
participant527EEGData = participant527EEGDataDF.to_numpy()

#Participant 559: 
participant559EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/595_Depression_REST_trimmed.csv")
participant559EEGData = participant559EEGDataDF.to_numpy()

#Participant 561: 
participant561EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/561_Depression_REST_trimmed.csv")
participant561EEGData = participant561EEGDataDF.to_numpy()

#Participant 565: 
participant565EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/565_Depression_REST_trimmed.csv")
participant565EEGData = participant565EEGDataDF.to_numpy()

#Participant 567: 
participant567EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/567_Depression_REST_trimmed.csv")
participant567EEGData = participant567EEGDataDF.to_numpy()

#Participant 587: 
participant587EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/587_Depression_REST_trimmed.csv")
participant587EEGData = participant587EEGDataDF.to_numpy()

#Participant 591: 
participant591EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/591_Depression_REST_trimmed.csv")
participant591EEGData = participant591EEGDataDF.to_numpy()

#Participant 594: 
participant594EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/594_Depression_REST_trimmed.csv")
participant594EEGData = participant594EEGDataDF.to_numpy()

#Participant 595: 
participant595EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/595_Depression_REST_trimmed.csv")
participant595EEGData = participant595EEGDataDF.to_numpy()

#Participant 597: 
participant597EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/597_Depression_REST_trimmed.csv")
participant597EEGData = participant597EEGDataDF.to_numpy()

#Participant 605: 
participant605EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/605_Depression_REST_trimmed.csv")
participant605EEGData = participant605EEGDataDF.to_numpy()

#Participant 607: 
participant607EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/607_Depression_REST_trimmed.csv")
participant607EEGData = participant607EEGDataDF.to_numpy()

#Participant 610: 
participant610EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/610_Depression_REST_trimmed.csv")
participant610EEGData = participant610EEGDataDF.to_numpy()

#Participant 613: 
participant613EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/613_Depression_REST_trimmed.csv")
participant613EEGData = participant613EEGDataDF.to_numpy()

#Participant 614: 
participant614EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/614_Depression_REST_trimmed.csv")
participant614EEGData = participant614EEGDataDF.to_numpy()

#Participant 616: 
participant616EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/616_Depression_REST_trimmed.csv")
participant616EEGData = participant616EEGDataDF.to_numpy()

#Participant 622: 
participant622EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/622_Depression_REST_trimmed.csv")
participant622EEGData = participant622EEGDataDF.to_numpy()

#Participant 624: 
participant624EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/624_Depression_REST_trimmed.csv")
participant624EEGData = participant624EEGDataDF.to_numpy()

#Participant 625: 
participant625EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/625_Depression_REST_trimmed.csv")
participant625EEGData = participant625EEGDataDF.to_numpy()

#Participant 626: 
participant626EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/626_Depression_REST_trimmed.csv")
participant626EEGData = participant626EEGDataDF.to_numpy()

#Participant 627: 
participant627EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/627_Depression_REST_trimmed.csv")
participant627EEGData = participant627EEGDataDF.to_numpy()

#Participant 628: 
participant628EEGDataDF = pd.read_csv("C:/Users/12407/Desktop/Independent Coding Projects/Projects for Summer 2021 Project Portfolio/Diagnosing Illness with EEG Data/Open Source EEG Data/Dep EEG Data/628_Depression_REST_trimmed.csv")
participant628EEGData = participant628EEGDataDF.to_numpy()



#Computing the DFTs of the Participants' EEG Data:
#There will be a total of 42 DFT (Fourier Coefficient) Arrays for both the HC and MDD Groups from the Cavanagh Data Set.
#Immediately after, I'll compute the Power Spectra for each of these 42 DFT arrays. 
#I'll repeat the following participant-specific block of code for all of the participants. 

#Participant 507: 
participant507EEGDataFullDFT = np.zeros(participant507EEGData.shape, dtype = 'complex_')

for n in range(0, participant507EEGDataFullDFT.shape[0] - 1):
    participant507EEGDataFullDFT[n, :] = fft(participant507EEGData[n, :])

#Participant 508: 
participant508EEGDataFullDFT = np.zeros(participant508EEGData.shape, dtype = 'complex_')

for n in range(0, participant508EEGDataFullDFT.shape[0] - 1):
    participant508EEGDataFullDFT[n, :] = fft(participant508EEGData[n, :])

#Participant 509: 
participant509EEGDataFullDFT = np.zeros(participant509EEGData.shape, dtype = 'complex_')

for n in range(0, participant509EEGDataFullDFT.shape[0] - 1):
    participant509EEGDataFullDFT[n, :] = fft(participant509EEGData[n, :])

#Participant 510: 
participant510EEGDataFullDFT = np.zeros(participant510EEGData.shape, dtype = 'complex_')

for n in range(0, participant510EEGDataFullDFT.shape[0] - 1):
    participant510EEGDataFullDFT[n, :] = fft(participant510EEGData[n, :])

#Participant 511: 
participant511EEGDataFullDFT = np.zeros(participant511EEGData.shape, dtype = 'complex_')

for n in range(0, participant511EEGDataFullDFT.shape[0] - 1):
    participant511EEGDataFullDFT[n, :] = fft(participant511EEGData[n, :])

#Participant 512: 
participant512EEGDataFullDFT = np.zeros(participant512EEGData.shape, dtype = 'complex_')

for n in range(0, participant512EEGDataFullDFT.shape[0] - 1):
    participant512EEGDataFullDFT[n, :] = fft(participant512EEGData[n, :])

#Participant 513: 
participant513EEGDataFullDFT = np.zeros(participant513EEGData.shape, dtype = 'complex_')

for n in range(0, participant513EEGDataFullDFT.shape[0] - 1):
    participant513EEGDataFullDFT[n, :] = fft(participant513EEGData[n, :])

#Participant 514: 
participant514EEGDataFullDFT = np.zeros(participant514EEGData.shape, dtype = 'complex_')

for n in range(0, participant514EEGDataFullDFT.shape[0] - 1):
    participant514EEGDataFullDFT[n, :] = fft(participant514EEGData[n, :])

#Participant 515: 
participant515EEGDataFullDFT = np.zeros(participant515EEGData.shape, dtype = 'complex_')

for n in range(0, participant515EEGDataFullDFT.shape[0] - 1):
    participant515EEGDataFullDFT[n, :] = fft(participant515EEGData[n, :])

#Participant 516: 
participant516EEGDataFullDFT = np.zeros(participant516EEGData.shape, dtype = 'complex_')

for n in range(0, participant516EEGDataFullDFT.shape[0] - 1):
    participant516EEGDataFullDFT[n, :] = fft(participant516EEGData[n, :])

#Participant 517: 
participant517EEGDataFullDFT = np.zeros(participant517EEGData.shape, dtype = 'complex_')

for n in range(0, participant517EEGDataFullDFT.shape[0] - 1):
    participant517EEGDataFullDFT[n, :] = fft(participant517EEGData[n, :])

#Participant 518: 
participant518EEGDataFullDFT = np.zeros(participant518EEGData.shape, dtype = 'complex_')

for n in range(0, participant518EEGDataFullDFT.shape[0] - 1):
    participant518EEGDataFullDFT[n, :] = fft(participant518EEGData[n, :])

#Participant 519: 
participant519EEGDataFullDFT = np.zeros(participant519EEGData.shape, dtype = 'complex_')

for n in range(0, participant519EEGDataFullDFT.shape[0] - 1):
    participant519EEGDataFullDFT[n, :] = fft(participant519EEGData[n, :])

#Participant 520: 
participant520EEGDataFullDFT = np.zeros(participant520EEGData.shape, dtype = 'complex_')

for n in range(0, participant520EEGDataFullDFT.shape[0] - 1):
    participant520EEGDataFullDFT[n, :] = fft(participant520EEGData[n, :])

#Participant 521: 
participant521EEGDataFullDFT = np.zeros(participant521EEGData.shape, dtype = 'complex_')

for n in range(0, participant521EEGDataFullDFT.shape[0] - 1):
    participant521EEGDataFullDFT[n, :] = fft(participant521EEGData[n, :])

#Participant 522: 
participant522EEGDataFullDFT = np.zeros(participant522EEGData.shape, dtype = 'complex_')

for n in range(0, participant522EEGDataFullDFT.shape[0] - 1):
    participant522EEGDataFullDFT[n, :] = fft(participant522EEGData[n, :])

#Participant 523: 
participant523EEGDataFullDFT = np.zeros(participant523EEGData.shape, dtype = 'complex_')

for n in range(0, participant523EEGDataFullDFT.shape[0] - 1):
    participant523EEGDataFullDFT[n, :] = fft(participant523EEGData[n, :])

#Participant 524: 
participant524EEGDataFullDFT = np.zeros(participant524EEGData.shape, dtype = 'complex_')

for n in range(0, participant524EEGDataFullDFT.shape[0] - 1):
    participant524EEGDataFullDFT[n, :] = fft(participant524EEGData[n, :])

#Participant 526: 
participant526EEGDataFullDFT = np.zeros(participant526EEGData.shape, dtype = 'complex_')

for n in range(0, participant526EEGDataFullDFT.shape[0] - 1):
    participant526EEGDataFullDFT[n, :] = fft(participant526EEGData[n, :])

#Participant 527: 
participant527EEGDataFullDFT = np.zeros(participant527EEGData.shape, dtype = 'complex_')

for n in range(0, participant527EEGDataFullDFT.shape[0] - 1):
    participant527EEGDataFullDFT[n, :] = fft(participant527EEGData[n, :])

#Participant 559: 
participant559EEGDataFullDFT = np.zeros(participant559EEGData.shape, dtype = 'complex_')

for n in range(0, participant559EEGDataFullDFT.shape[0] - 1):
    participant559EEGDataFullDFT[n, :] = fft(participant559EEGData[n, :])

#Participant 561: 
participant561EEGDataFullDFT = np.zeros(participant561EEGData.shape, dtype = 'complex_')

for n in range(0, participant561EEGDataFullDFT.shape[0] - 1):
    participant561EEGDataFullDFT[n, :] = fft(participant561EEGData[n, :])

#Participant 565: 
participant565EEGDataFullDFT = np.zeros(participant565EEGData.shape, dtype = 'complex_')

for n in range(0, participant507EEGDataFullDFT.shape[0] - 1):
    participant565EEGDataFullDFT[n, :] = fft(participant565EEGData[n, :])

#Participant 567: 
participant567EEGDataFullDFT = np.zeros(participant567EEGData.shape, dtype = 'complex_')

for n in range(0, participant567EEGDataFullDFT.shape[0] - 1):
    participant567EEGDataFullDFT[n, :] = fft(participant567EEGData[n, :])

#Participant 587: 
participant587EEGDataFullDFT = np.zeros(participant587EEGData.shape, dtype = 'complex_')

for n in range(0, participant587EEGDataFullDFT.shape[0] - 1):
    participant587EEGDataFullDFT[n, :] = fft(participant587EEGData[n, :])

#Participant 591: 
participant591EEGDataFullDFT = np.zeros(participant591EEGData.shape, dtype = 'complex_')

for n in range(0, participant591EEGDataFullDFT.shape[0] - 1):
    participant591EEGDataFullDFT[n, :] = fft(participant591EEGData[n, :])

#Participant 594: 
participant594EEGDataFullDFT = np.zeros(participant594EEGData.shape, dtype = 'complex_')

for n in range(0, participant594EEGDataFullDFT.shape[0] - 1):
    participant594EEGDataFullDFT[n, :] = fft(participant594EEGData[n, :])

#Participant 595: 
participant595EEGDataFullDFT = np.zeros(participant595EEGData.shape, dtype = 'complex_')

for n in range(0, participant595EEGDataFullDFT.shape[0] - 1):
    participant595EEGDataFullDFT[n, :] = fft(participant595EEGData[n, :])

#Participant 597: 
participant597EEGDataFullDFT = np.zeros(participant597EEGData.shape, dtype = 'complex_')

for n in range(0, participant597EEGDataFullDFT.shape[0] - 1):
    participant597EEGDataFullDFT[n, :] = fft(participant597EEGData[n, :])

#Participant 605: 
participant605EEGDataFullDFT = np.zeros(participant605EEGData.shape, dtype = 'complex_')

for n in range(0, participant605EEGDataFullDFT.shape[0] - 1):
    participant605EEGDataFullDFT[n, :] = fft(participant605EEGData[n, :])

#Participant 607: 
participant607EEGDataFullDFT = np.zeros(participant607EEGData.shape, dtype = 'complex_')

for n in range(0, participant607EEGDataFullDFT.shape[0] - 1):
    participant607EEGDataFullDFT[n, :] = fft(participant607EEGData[n, :])

#Participant 610: 
participant610EEGDataFullDFT = np.zeros(participant610EEGData.shape, dtype = 'complex_')

for n in range(0, participant610EEGDataFullDFT.shape[0] - 1):
    participant610EEGDataFullDFT[n, :] = fft(participant610EEGData[n, :])

#Participant 613: 
participant613EEGDataFullDFT = np.zeros(participant613EEGData.shape, dtype = 'complex_')

for n in range(0, participant613EEGDataFullDFT.shape[0] - 1):
    participant613EEGDataFullDFT[n, :] = fft(participant613EEGData[n, :])

#Participant 614: 
participant614EEGDataFullDFT = np.zeros(participant614EEGData.shape, dtype = 'complex_')

for n in range(0, participant614EEGDataFullDFT.shape[0] - 1):
    participant614EEGDataFullDFT[n, :] = fft(participant614EEGData[n, :])

#Participant 616: 
participant616EEGDataFullDFT = np.zeros(participant616EEGData.shape, dtype = 'complex_')

for n in range(0, participant616EEGDataFullDFT.shape[0] - 1):
    participant616EEGDataFullDFT[n, :] = fft(participant616EEGData[n, :])

#Participant 622: 
participant622EEGDataFullDFT = np.zeros(participant622EEGData.shape, dtype = 'complex_')

for n in range(0, participant594EEGDataFullDFT.shape[0] - 1):
    participant622EEGDataFullDFT[n, :] = fft(participant622EEGData[n, :])

#Participant 624: 
participant624EEGDataFullDFT = np.zeros(participant624EEGData.shape, dtype = 'complex_')

for n in range(0, participant624EEGDataFullDFT.shape[0] - 1):
    participant624EEGDataFullDFT[n, :] = fft(participant624EEGData[n, :])

#Participant 625: 
participant625EEGDataFullDFT = np.zeros(participant625EEGData.shape, dtype = 'complex_')

for n in range(0, participant625EEGDataFullDFT.shape[0] - 1):
    participant625EEGDataFullDFT[n, :] = fft(participant625EEGData[n, :])
    
#Participant 626: 
participant626EEGDataFullDFT = np.zeros(participant626EEGData.shape, dtype = 'complex_')

for n in range(0, participant594EEGDataFullDFT.shape[0] - 1):
    participant626EEGDataFullDFT[n, :] = fft(participant626EEGData[n, :])

#Participant 627: 
participant627EEGDataFullDFT = np.zeros(participant627EEGData.shape, dtype = 'complex_')

for n in range(0, participant627EEGDataFullDFT.shape[0] - 1):
    participant627EEGDataFullDFT[n, :] = fft(participant627EEGData[n, :])

#Participant 628: 
participant628EEGDataFullDFT = np.zeros(participant628EEGData.shape, dtype = 'complex_')

for n in range(0, participant628EEGDataFullDFT.shape[0] - 1):
    participant628EEGDataFullDFT[n, :] = fft(participant628EEGData[n, :])





##STEP 2: COMPUTING EEG FEATURE VARIABLES FOR EACH PARTICIPANT: ##

#Now that 42 separate Raw EEG Data and Raw DFT Arrays have been computed, one of each type for each participant, it's time to create a single dataframe to contain all of 
#feature variables ((i) Channel-Averaged (Whole Brain) Absolute Alpha Power (ii) Channel-Averaged (Whole Brain) Absolute Theta Power (iii) Channel-Averaged (Whole Brain) Relative Alpha Power 
#(iv) Channel-Averaged (Whole Brain) Relative Theta Power, (v) Theta Band Cordance (either Whole Brain or localized to one region in the brain)). 
#Once this dataframe is constructed and all of the relevant data has been computed from the 42 DFT Arrays and inserted into it, Machine Learning Model Training and Testing can begin.

#Creating EEG Features and Diagnostic Labels DataFrame (for all Participants):
#NOTE: In the last column that contains participants' Depression and Healthy Control Labels, a value of "0" will indicate HC and a value of "1" will indicate an MDD diagnosis. 
EEGFeaturesDF = pd.DataFrame(columns = ['ParticipantID', 'ChannelAvgAbsAlphaPower', 'ChannelAvgAbsThetaPower', 'ChannelAvgRelAlphaPower', 'ChannelAvgRelThetaPower', 'ThetaBandCordance', 'MDDorHC'])
EEGFeaturesDFColumnList = ['ParticipantID', 'ChannelAvgAbsAlphaPower', 'ChannelAvgAbsThetaPower', 'ChannelAvgRelAlphaPower', 'ChannelAvgRelThetaPower', 'ThetaBandCordance', 'MDDorHC']
#7/9/2021: For the C.N.N. (Convolutional Neural Network) model, I'll also create a (Python) List of numpy arrays; each of these arrays will consist of all of the 
#channel-averaged, total frequency bin power values for each of the Cavanagh study participants. 
EEGChannelAvgTotalPowerList = []

#Creating 42 (1x5) Participant-Specfic EEG Feature Vectors. These will be individually placed into the EEGFeaturesDF as rows later on. 
#As such, each of these arrays will follow the same row/column structure as that of EEGFeaturesDF. 

participant507EEGFeatures = np.zeros(shape = (1,7))
participant507EEGFeatures[0, 0] = 507
participant507EEGFeatures[0, 6] = 0

participant508EEGFeatures = np.zeros(shape = (1,7))
participant508EEGFeatures[0, 0] = 508
participant508EEGFeatures[0, 6] = 0

participant509EEGFeatures = np.zeros(shape = (1,7))
participant509EEGFeatures[0, 0] = 509
participant509EEGFeatures[0, 6] = 0

participant510EEGFeatures = np.zeros(shape = (1,7))
participant510EEGFeatures[0, 0] = 510
participant510EEGFeatures[0, 6] = 0

participant511EEGFeatures = np.zeros(shape = (1,7))
participant511EEGFeatures[0, 0] = 511
participant511EEGFeatures[0, 6] = 0

participant512EEGFeatures = np.zeros(shape = (1,7))
participant512EEGFeatures[0, 0] = 512
participant512EEGFeatures[0, 6] = 0

participant513EEGFeatures = np.zeros(shape = (1,7))
participant513EEGFeatures[0, 0] = 513
participant513EEGFeatures[0, 6] = 0

participant514EEGFeatures = np.zeros(shape = (1,7))
participant514EEGFeatures[0, 0] = 514
participant514EEGFeatures[0, 6] = 0

participant515EEGFeatures = np.zeros(shape = (1,7))
participant515EEGFeatures[0, 0] = 515
participant515EEGFeatures[0, 6] = 0

participant516EEGFeatures = np.zeros(shape = (1,7))
participant516EEGFeatures[0, 0] = 516
participant516EEGFeatures[0, 6] = 0

participant517EEGFeatures = np.zeros(shape = (1,7))
participant517EEGFeatures[0, 0] = 517
participant517EEGFeatures[0, 6] = 0

participant518EEGFeatures = np.zeros(shape = (1,7))
participant518EEGFeatures[0, 0] = 518
participant518EEGFeatures[0, 6] = 0

participant519EEGFeatures = np.zeros(shape = (1,7))
participant519EEGFeatures[0, 0] = 519
participant519EEGFeatures[0, 6] = 0

participant520EEGFeatures = np.zeros(shape = (1,7))
participant520EEGFeatures[0, 0] = 520
participant520EEGFeatures[0, 6] = 0

participant521EEGFeatures = np.zeros(shape = (1,7))
participant521EEGFeatures[0, 0] = 521
participant521EEGFeatures[0, 6] = 0

participant522EEGFeatures = np.zeros(shape = (1,7))
participant522EEGFeatures[0, 0] = 522
participant522EEGFeatures[0, 6] = 0

participant523EEGFeatures = np.zeros(shape = (1,7))
participant523EEGFeatures[0, 0] = 523
participant523EEGFeatures[0, 6] = 0

participant524EEGFeatures = np.zeros(shape = (1,7))
participant524EEGFeatures[0, 0] = 524
participant524EEGFeatures[0, 6] = 0

participant526EEGFeatures = np.zeros(shape = (1,7))
participant526EEGFeatures[0, 0] = 526
participant526EEGFeatures[0, 6] = 0

participant527EEGFeatures = np.zeros(shape = (1,7))
participant527EEGFeatures[0, 0] = 527
participant527EEGFeatures[0, 6] = 0

participant559EEGFeatures = np.zeros(shape = (1,7))
participant559EEGFeatures[0, 0] = 559
participant559EEGFeatures[0, 6] = 1

participant561EEGFeatures = np.zeros(shape = (1,7))
participant561EEGFeatures[0, 0] = 561
participant561EEGFeatures[0, 6] = 1

participant565EEGFeatures = np.zeros(shape = (1,7))
participant565EEGFeatures[0, 0] = 565
participant565EEGFeatures[0, 6] = 1

participant567EEGFeatures = np.zeros(shape = (1,7))
participant567EEGFeatures[0, 0] = 567
participant567EEGFeatures[0, 6] = 1

participant587EEGFeatures = np.zeros(shape = (1,7))
participant587EEGFeatures[0, 0] = 587
participant587EEGFeatures[0, 6] = 1

participant591EEGFeatures = np.zeros(shape = (1,7))
participant591EEGFeatures[0, 0] = 591
participant591EEGFeatures[0, 6] = 1

participant594EEGFeatures = np.zeros(shape = (1,7))
participant594EEGFeatures[0, 0] = 594
participant594EEGFeatures[0, 6] = 1

participant595EEGFeatures = np.zeros(shape = (1,7))
participant595EEGFeatures[0, 0] = 595
participant595EEGFeatures[0, 6] = 1

participant597EEGFeatures = np.zeros(shape = (1,7))
participant597EEGFeatures[0, 0] = 597
participant597EEGFeatures[0, 6] = 1

participant605EEGFeatures = np.zeros(shape = (1,7))
participant605EEGFeatures[0, 0] = 605
participant605EEGFeatures[0, 6] = 1

participant607EEGFeatures = np.zeros(shape = (1,7))
participant607EEGFeatures[0, 0] = 607
participant607EEGFeatures[0, 6] = 1

participant610EEGFeatures = np.zeros(shape = (1,7))
participant610EEGFeatures[0, 0] = 610
participant610EEGFeatures[0, 6] = 1

participant613EEGFeatures = np.zeros(shape = (1,7))
participant613EEGFeatures[0, 0] = 613
participant613EEGFeatures[0, 6] = 1

participant614EEGFeatures = np.zeros(shape = (1,7))
participant614EEGFeatures[0, 0] = 614
participant614EEGFeatures[0, 6] = 1

participant616EEGFeatures = np.zeros(shape = (1,7))
participant616EEGFeatures[0, 0] = 616
participant616EEGFeatures[0, 6] = 1

participant622EEGFeatures = np.zeros(shape = (1,7))
participant622EEGFeatures[0, 0] = 622
participant622EEGFeatures[0, 6] = 1

participant624EEGFeatures = np.zeros(shape = (1,7))
participant624EEGFeatures[0, 0] = 624
participant624EEGFeatures[0, 6] = 1

participant625EEGFeatures = np.zeros(shape = (1,7))
participant625EEGFeatures[0, 0] = 625
participant625EEGFeatures[0, 6] = 1

participant626EEGFeatures = np.zeros(shape = (1,7))
participant626EEGFeatures[0, 0] = 626
participant626EEGFeatures[0, 6] = 1

participant627EEGFeatures = np.zeros(shape = (1,7))
participant627EEGFeatures[0, 0] = 627
participant627EEGFeatures[0, 6] = 1

participant628EEGFeatures = np.zeros(shape = (1,7))
participant628EEGFeatures[0, 0] = 628
participant628EEGFeatures[0, 6] = 1



#Creating and Computing Participant-Specific Arrays Containing Channel-Averaged DFT Data:  

#Participant 507: 
participant507EEGDataFullDFTChannelSum = participant507EEGDataFullDFT.sum(axis = 0)

participant507EEGDataFullDFTChannelAvg = participant507EEGDataFullDFTChannelSum / participant507EEGDataFullDFT.shape[0]

#Participant 508: 
participant508EEGDataFullDFTChannelSum = participant508EEGDataFullDFT.sum(axis = 0)

participant508EEGDataFullDFTChannelAvg = participant508EEGDataFullDFTChannelSum / participant508EEGDataFullDFT.shape[0]

#Participant 509: 
participant509EEGDataFullDFTChannelSum = participant509EEGDataFullDFT.sum(axis = 0)

participant509EEGDataFullDFTChannelAvg = participant509EEGDataFullDFTChannelSum / participant509EEGDataFullDFT.shape[0]

#Participant 510: 
participant510EEGDataFullDFTChannelSum = participant510EEGDataFullDFT.sum(axis = 0)

participant510EEGDataFullDFTChannelAvg = participant510EEGDataFullDFTChannelSum / participant510EEGDataFullDFT.shape[0]

#Participant 511: 
participant511EEGDataFullDFTChannelSum = participant511EEGDataFullDFT.sum(axis = 0)

participant511EEGDataFullDFTChannelAvg = participant507EEGDataFullDFTChannelSum / participant507EEGDataFullDFT.shape[0]

#Participant 512: 
participant512EEGDataFullDFTChannelSum = participant512EEGDataFullDFT.sum(axis = 0)

participant512EEGDataFullDFTChannelAvg = participant512EEGDataFullDFTChannelSum / participant512EEGDataFullDFT.shape[0]

#Participant 513: 
participant513EEGDataFullDFTChannelSum = participant513EEGDataFullDFT.sum(axis = 0)

participant513EEGDataFullDFTChannelAvg = participant513EEGDataFullDFTChannelSum / participant513EEGDataFullDFT.shape[0]

#Participant 514: 
participant514EEGDataFullDFTChannelSum = participant514EEGDataFullDFT.sum(axis = 0)

participant514EEGDataFullDFTChannelAvg = participant514EEGDataFullDFTChannelSum / participant514EEGDataFullDFT.shape[0]

#Participant 515: 
participant515EEGDataFullDFTChannelSum = participant515EEGDataFullDFT.sum(axis = 0)

participant515EEGDataFullDFTChannelAvg = participant515EEGDataFullDFTChannelSum / participant515EEGDataFullDFT.shape[0]

#Participant 516: 
participant516EEGDataFullDFTChannelSum = participant516EEGDataFullDFT.sum(axis = 0)

participant516EEGDataFullDFTChannelAvg = participant516EEGDataFullDFTChannelSum / participant516EEGDataFullDFT.shape[0]

#Participant 517: 
participant517EEGDataFullDFTChannelSum = participant517EEGDataFullDFT.sum(axis = 0)

participant517EEGDataFullDFTChannelAvg = participant517EEGDataFullDFTChannelSum / participant517EEGDataFullDFT.shape[0]

#Participant 518: 
participant518EEGDataFullDFTChannelSum = participant518EEGDataFullDFT.sum(axis = 0)

participant518EEGDataFullDFTChannelAvg = participant518EEGDataFullDFTChannelSum / participant518EEGDataFullDFT.shape[0]

#Participant 519: 
participant519EEGDataFullDFTChannelSum = participant519EEGDataFullDFT.sum(axis = 0)

participant519EEGDataFullDFTChannelAvg = participant519EEGDataFullDFTChannelSum / participant519EEGDataFullDFT.shape[0]

#Participant 520: 
participant520EEGDataFullDFTChannelSum = participant520EEGDataFullDFT.sum(axis = 0)

participant520EEGDataFullDFTChannelAvg = participant520EEGDataFullDFTChannelSum / participant520EEGDataFullDFT.shape[0]

#Participant 521: 
participant521EEGDataFullDFTChannelSum = participant521EEGDataFullDFT.sum(axis = 0)

participant521EEGDataFullDFTChannelAvg = participant521EEGDataFullDFTChannelSum / participant521EEGDataFullDFT.shape[0]

#Participant 522: 
participant522EEGDataFullDFTChannelSum = participant522EEGDataFullDFT.sum(axis = 0)

participant522EEGDataFullDFTChannelAvg = participant522EEGDataFullDFTChannelSum / participant522EEGDataFullDFT.shape[0]

#Participant 523: 
participant523EEGDataFullDFTChannelSum = participant523EEGDataFullDFT.sum(axis = 0)

participant523EEGDataFullDFTChannelAvg = participant523EEGDataFullDFTChannelSum / participant523EEGDataFullDFT.shape[0]

#Participant 524: 
participant524EEGDataFullDFTChannelSum = participant524EEGDataFullDFT.sum(axis = 0)

participant524EEGDataFullDFTChannelAvg = participant524EEGDataFullDFTChannelSum / participant524EEGDataFullDFT.shape[0]

#Participant 526: 
participant526EEGDataFullDFTChannelSum = participant526EEGDataFullDFT.sum(axis = 0)

participant526EEGDataFullDFTChannelAvg = participant526EEGDataFullDFTChannelSum / participant526EEGDataFullDFT.shape[0]

#Participant 527: 
participant527EEGDataFullDFTChannelSum = participant527EEGDataFullDFT.sum(axis = 0)

participant527EEGDataFullDFTChannelAvg = participant527EEGDataFullDFTChannelSum / participant527EEGDataFullDFT.shape[0]

#Participant 559: 
participant559EEGDataFullDFTChannelSum = participant559EEGDataFullDFT.sum(axis = 0)

participant559EEGDataFullDFTChannelAvg = participant559EEGDataFullDFTChannelSum / participant559EEGDataFullDFT.shape[0]

#Participant 561: 
participant561EEGDataFullDFTChannelSum = participant561EEGDataFullDFT.sum(axis = 0)

participant561EEGDataFullDFTChannelAvg = participant561EEGDataFullDFTChannelSum / participant561EEGDataFullDFT.shape[0]

#Participant 565: 
participant565EEGDataFullDFTChannelSum = participant565EEGDataFullDFT.sum(axis = 0)

participant565EEGDataFullDFTChannelAvg = participant565EEGDataFullDFTChannelSum / participant565EEGDataFullDFT.shape[0]

#Participant 567: 
participant567EEGDataFullDFTChannelSum = participant567EEGDataFullDFT.sum(axis = 0)

participant567EEGDataFullDFTChannelAvg = participant567EEGDataFullDFTChannelSum / participant567EEGDataFullDFT.shape[0]

#Participant 587: 
participant587EEGDataFullDFTChannelSum = participant587EEGDataFullDFT.sum(axis = 0)

participant587EEGDataFullDFTChannelAvg = participant587EEGDataFullDFTChannelSum / participant587EEGDataFullDFT.shape[0]

#Participant 591: 
participant591EEGDataFullDFTChannelSum = participant591EEGDataFullDFT.sum(axis = 0)

participant591EEGDataFullDFTChannelAvg = participant591EEGDataFullDFTChannelSum / participant591EEGDataFullDFT.shape[0]

#Participant 594: 
participant594EEGDataFullDFTChannelSum = participant594EEGDataFullDFT.sum(axis = 0)

participant594EEGDataFullDFTChannelAvg = participant594EEGDataFullDFTChannelSum / participant594EEGDataFullDFT.shape[0]

#Participant 595: 
participant595EEGDataFullDFTChannelSum = participant595EEGDataFullDFT.sum(axis = 0)

participant595EEGDataFullDFTChannelAvg = participant595EEGDataFullDFTChannelSum / participant595EEGDataFullDFT.shape[0]

#Participant 597: 
participant597EEGDataFullDFTChannelSum = participant597EEGDataFullDFT.sum(axis = 0)

participant597EEGDataFullDFTChannelAvg = participant597EEGDataFullDFTChannelSum / participant597EEGDataFullDFT.shape[0]

#Participant 605: 
participant605EEGDataFullDFTChannelSum = participant605EEGDataFullDFT.sum(axis = 0)

participant605EEGDataFullDFTChannelAvg = participant605EEGDataFullDFTChannelSum / participant605EEGDataFullDFT.shape[0]

#Participant 607: 
participant607EEGDataFullDFTChannelSum = participant607EEGDataFullDFT.sum(axis = 0)

participant607EEGDataFullDFTChannelAvg = participant607EEGDataFullDFTChannelSum / participant607EEGDataFullDFT.shape[0]

#Participant 610: 
participant610EEGDataFullDFTChannelSum = participant610EEGDataFullDFT.sum(axis = 0)

participant610EEGDataFullDFTChannelAvg = participant610EEGDataFullDFTChannelSum / participant610EEGDataFullDFT.shape[0]

#Participant 613: 
participant613EEGDataFullDFTChannelSum = participant613EEGDataFullDFT.sum(axis = 0)

participant613EEGDataFullDFTChannelAvg = participant613EEGDataFullDFTChannelSum / participant613EEGDataFullDFT.shape[0]

#Participant 614: 
participant614EEGDataFullDFTChannelSum = participant614EEGDataFullDFT.sum(axis = 0)

participant614EEGDataFullDFTChannelAvg = participant614EEGDataFullDFTChannelSum / participant614EEGDataFullDFT.shape[0]

#Participant 616: 
participant616EEGDataFullDFTChannelSum = participant616EEGDataFullDFT.sum(axis = 0)

participant616EEGDataFullDFTChannelAvg = participant616EEGDataFullDFTChannelSum / participant616EEGDataFullDFT.shape[0]

#Participant 622: 
participant622EEGDataFullDFTChannelSum = participant622EEGDataFullDFT.sum(axis = 0)

participant622EEGDataFullDFTChannelAvg = participant622EEGDataFullDFTChannelSum / participant622EEGDataFullDFT.shape[0]

#Participant 624: 
participant624EEGDataFullDFTChannelSum = participant624EEGDataFullDFT.sum(axis = 0)

participant624EEGDataFullDFTChannelAvg = participant624EEGDataFullDFTChannelSum / participant624EEGDataFullDFT.shape[0]

#Participant 625: 
participant625EEGDataFullDFTChannelSum = participant625EEGDataFullDFT.sum(axis = 0)

participant625EEGDataFullDFTChannelAvg = participant625EEGDataFullDFTChannelSum / participant625EEGDataFullDFT.shape[0]

#Participant 626: 
participant626EEGDataFullDFTChannelSum = participant626EEGDataFullDFT.sum(axis = 0)

participant626EEGDataFullDFTChannelAvg = participant626EEGDataFullDFTChannelSum / participant626EEGDataFullDFT.shape[0]

#Participant 627: 
participant627EEGDataFullDFTChannelSum = participant627EEGDataFullDFT.sum(axis = 0)

participant627EEGDataFullDFTChannelAvg = participant627EEGDataFullDFTChannelSum / participant627EEGDataFullDFT.shape[0]

#Participant 628: 
participant628EEGDataFullDFTChannelSum = participant628EEGDataFullDFT.sum(axis = 0)

participant628EEGDataFullDFTChannelAvg = participant628EEGDataFullDFTChannelSum / participant628EEGDataFullDFT.shape[0]



#Computing Feature Variable 1: channelAvgAbsAlphaPower (for all Participants) 

#Note: The indices 4012 and 6018 correspond to the beginning and ends of the 8-12 Hz Alpha Band Frequency Range for EEG Recordings within each DFT Vector,
#given the DFT/EEG recording lengths of 250,734 samples @ a sampling frequency of 500 Hz (see the Kavanagh study for details). 
# **Note 2**: I completely forgot that after cutting down the length (and t.f., the number of total EEG samples in my data), I needed to re-do the loop
#indices to calculate the different features to feed into the Logistic Regression/KNN, etc. ML models. For the new channel-averaged Alpha Power Feature, the 
#for loop indices should be (130, 192). 

#Participant 507: 
participant507ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant507ChannelAvgAbsAlphaPower += math.sqrt((participant507EEGDataFullDFTChannelAvg[j].real ** 2) + (participant507EEGDataFullDFTChannelAvg[j].imag ** 2))
participant507EEGFeatures[0, 1] = participant507ChannelAvgAbsAlphaPower

#Participant 508: 
participant508ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant508ChannelAvgAbsAlphaPower += math.sqrt((participant508EEGDataFullDFTChannelAvg[j].real ** 2) + (participant508EEGDataFullDFTChannelAvg[j].imag ** 2))
participant508EEGFeatures[0, 1] = participant508ChannelAvgAbsAlphaPower

#Participant 509: 
participant509ChannelAvgAbsAlphaPower = 0
for j in range(130, 192):  
    participant509ChannelAvgAbsAlphaPower += math.sqrt((participant509EEGDataFullDFTChannelAvg[j].real ** 2) + (participant509EEGDataFullDFTChannelAvg[j].imag ** 2))
participant509EEGFeatures[0, 1] = participant509ChannelAvgAbsAlphaPower

#Participant 510: 
participant510ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant510ChannelAvgAbsAlphaPower += math.sqrt((participant510EEGDataFullDFTChannelAvg[j].real ** 2) + (participant510EEGDataFullDFTChannelAvg[j].imag ** 2))
participant510EEGFeatures[0, 1] = participant510ChannelAvgAbsAlphaPower

#Participant 511: 
participant511ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant511ChannelAvgAbsAlphaPower += math.sqrt((participant511EEGDataFullDFTChannelAvg[j].real ** 2) + (participant511EEGDataFullDFTChannelAvg[j].imag ** 2))
participant511EEGFeatures[0, 1] = participant511ChannelAvgAbsAlphaPower

#Participant 512: 
participant512ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant512ChannelAvgAbsAlphaPower += math.sqrt((participant512EEGDataFullDFTChannelAvg[j].real ** 2) + (participant512EEGDataFullDFTChannelAvg[j].imag ** 2))
participant512EEGFeatures[0, 1] = participant512ChannelAvgAbsAlphaPower

#Participant 513: 
participant513ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant513ChannelAvgAbsAlphaPower += math.sqrt((participant513EEGDataFullDFTChannelAvg[j].real ** 2) + (participant513EEGDataFullDFTChannelAvg[j].imag ** 2))
participant513EEGFeatures[0, 1] = participant513ChannelAvgAbsAlphaPower

#Participant 514: 
participant514ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant514ChannelAvgAbsAlphaPower += math.sqrt((participant514EEGDataFullDFTChannelAvg[j].real ** 2) + (participant514EEGDataFullDFTChannelAvg[j].imag ** 2))
participant514EEGFeatures[0, 1] = participant514ChannelAvgAbsAlphaPower

#Participant 515: 
participant515ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant515ChannelAvgAbsAlphaPower += math.sqrt((participant515EEGDataFullDFTChannelAvg[j].real ** 2) + (participant515EEGDataFullDFTChannelAvg[j].imag ** 2))
participant515EEGFeatures[0, 1] = participant515ChannelAvgAbsAlphaPower

#Participant 516: 
participant516ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant516ChannelAvgAbsAlphaPower += math.sqrt((participant516EEGDataFullDFTChannelAvg[j].real ** 2) + (participant516EEGDataFullDFTChannelAvg[j].imag ** 2))
participant516EEGFeatures[0, 1] = participant516ChannelAvgAbsAlphaPower

#Participant 517: 
participant517ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant517ChannelAvgAbsAlphaPower += math.sqrt((participant517EEGDataFullDFTChannelAvg[j].real ** 2) + (participant517EEGDataFullDFTChannelAvg[j].imag ** 2))
participant517EEGFeatures[0, 1] = participant517ChannelAvgAbsAlphaPower

#Participant 518: 
participant518ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant518ChannelAvgAbsAlphaPower += math.sqrt((participant518EEGDataFullDFTChannelAvg[j].real ** 2) + (participant518EEGDataFullDFTChannelAvg[j].imag ** 2))
participant518EEGFeatures[0, 1] = participant518ChannelAvgAbsAlphaPower

#Participant 519: 
participant519ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant519ChannelAvgAbsAlphaPower += math.sqrt((participant519EEGDataFullDFTChannelAvg[j].real ** 2) + (participant519EEGDataFullDFTChannelAvg[j].imag ** 2))
participant519EEGFeatures[0, 1] = participant519ChannelAvgAbsAlphaPower

#Participant 520: 
participant520ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant520ChannelAvgAbsAlphaPower += math.sqrt((participant520EEGDataFullDFTChannelAvg[j].real ** 2) + (participant520EEGDataFullDFTChannelAvg[j].imag ** 2))
participant520EEGFeatures[0, 1] = participant520ChannelAvgAbsAlphaPower

#Participant 521: 
participant521ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant521ChannelAvgAbsAlphaPower += math.sqrt((participant521EEGDataFullDFTChannelAvg[j].real ** 2) + (participant521EEGDataFullDFTChannelAvg[j].imag ** 2))
participant521EEGFeatures[0, 1] = participant521ChannelAvgAbsAlphaPower

#Participant 522: 
participant522ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant522ChannelAvgAbsAlphaPower += math.sqrt((participant522EEGDataFullDFTChannelAvg[j].real ** 2) + (participant522EEGDataFullDFTChannelAvg[j].imag ** 2))
participant522EEGFeatures[0, 1] = participant522ChannelAvgAbsAlphaPower

#Participant 523: 
participant523ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant523ChannelAvgAbsAlphaPower += math.sqrt((participant523EEGDataFullDFTChannelAvg[j].real ** 2) + (participant523EEGDataFullDFTChannelAvg[j].imag ** 2))
participant523EEGFeatures[0, 1] = participant523ChannelAvgAbsAlphaPower

#Participant 524: 
participant524ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant524ChannelAvgAbsAlphaPower += math.sqrt((participant524EEGDataFullDFTChannelAvg[j].real ** 2) + (participant524EEGDataFullDFTChannelAvg[j].imag ** 2))
participant524EEGFeatures[0, 1] = participant524ChannelAvgAbsAlphaPower

#Participant 526: 
participant526ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant526ChannelAvgAbsAlphaPower += math.sqrt((participant526EEGDataFullDFTChannelAvg[j].real ** 2) + (participant526EEGDataFullDFTChannelAvg[j].imag ** 2))
participant526EEGFeatures[0, 1] = participant526ChannelAvgAbsAlphaPower

#Participant 527: 
participant527ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant527ChannelAvgAbsAlphaPower += math.sqrt((participant527EEGDataFullDFTChannelAvg[j].real ** 2) + (participant527EEGDataFullDFTChannelAvg[j].imag ** 2))
participant527EEGFeatures[0, 1] = participant527ChannelAvgAbsAlphaPower

#Participant 559: 
participant559ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant559ChannelAvgAbsAlphaPower += math.sqrt((participant559EEGDataFullDFTChannelAvg[j].real ** 2) + (participant559EEGDataFullDFTChannelAvg[j].imag ** 2))
participant559EEGFeatures[0, 1] = participant559ChannelAvgAbsAlphaPower

#Participant 561: 
participant561ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant561ChannelAvgAbsAlphaPower += math.sqrt((participant561EEGDataFullDFTChannelAvg[j].real ** 2) + (participant561EEGDataFullDFTChannelAvg[j].imag ** 2))
participant561EEGFeatures[0, 1] = participant561ChannelAvgAbsAlphaPower

#Participant 561: 
participant561ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant561ChannelAvgAbsAlphaPower += math.sqrt((participant561EEGDataFullDFTChannelAvg[j].real ** 2) + (participant561EEGDataFullDFTChannelAvg[j].imag ** 2))
participant561EEGFeatures[0, 1] = participant561ChannelAvgAbsAlphaPower

#Participant 565: 
participant565ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant565ChannelAvgAbsAlphaPower += math.sqrt((participant565EEGDataFullDFTChannelAvg[j].real ** 2) + (participant565EEGDataFullDFTChannelAvg[j].imag ** 2))
participant565EEGFeatures[0, 1] = participant565ChannelAvgAbsAlphaPower

#Participant 567: 
participant567ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant567ChannelAvgAbsAlphaPower += math.sqrt((participant567EEGDataFullDFTChannelAvg[j].real ** 2) + (participant567EEGDataFullDFTChannelAvg[j].imag ** 2))
participant567EEGFeatures[0, 1] = participant567ChannelAvgAbsAlphaPower

#Participant 587: 
participant587ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant587ChannelAvgAbsAlphaPower += math.sqrt((participant587EEGDataFullDFTChannelAvg[j].real ** 2) + (participant587EEGDataFullDFTChannelAvg[j].imag ** 2))
participant587EEGFeatures[0, 1] = participant587ChannelAvgAbsAlphaPower

#Participant 587: 
participant587ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant587ChannelAvgAbsAlphaPower += math.sqrt((participant587EEGDataFullDFTChannelAvg[j].real ** 2) + (participant587EEGDataFullDFTChannelAvg[j].imag ** 2))
participant587EEGFeatures[0, 1] = participant587ChannelAvgAbsAlphaPower

#Participant 591: 
participant591ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant591ChannelAvgAbsAlphaPower += math.sqrt((participant591EEGDataFullDFTChannelAvg[j].real ** 2) + (participant591EEGDataFullDFTChannelAvg[j].imag ** 2))
participant591EEGFeatures[0, 1] = participant591ChannelAvgAbsAlphaPower

#Participant 594: 
participant594ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant594ChannelAvgAbsAlphaPower += math.sqrt((participant594EEGDataFullDFTChannelAvg[j].real ** 2) + (participant594EEGDataFullDFTChannelAvg[j].imag ** 2))
participant594EEGFeatures[0, 1] = participant594ChannelAvgAbsAlphaPower

#Participant 595: 
participant595ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant595ChannelAvgAbsAlphaPower += math.sqrt((participant595EEGDataFullDFTChannelAvg[j].real ** 2) + (participant595EEGDataFullDFTChannelAvg[j].imag ** 2))
participant595EEGFeatures[0, 1] = participant595ChannelAvgAbsAlphaPower

#Participant 597: 
participant597ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant597ChannelAvgAbsAlphaPower += math.sqrt((participant597EEGDataFullDFTChannelAvg[j].real ** 2) + (participant597EEGDataFullDFTChannelAvg[j].imag ** 2))
participant597EEGFeatures[0, 1] = participant597ChannelAvgAbsAlphaPower

#Participant 605: 
participant605ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant605ChannelAvgAbsAlphaPower += math.sqrt((participant605EEGDataFullDFTChannelAvg[j].real ** 2) + (participant605EEGDataFullDFTChannelAvg[j].imag ** 2))
participant605EEGFeatures[0, 1] = participant605ChannelAvgAbsAlphaPower

#Participant 607: 
participant607ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant607ChannelAvgAbsAlphaPower += math.sqrt((participant607EEGDataFullDFTChannelAvg[j].real ** 2) + (participant607EEGDataFullDFTChannelAvg[j].imag ** 2))
participant607EEGFeatures[0, 1] = participant607ChannelAvgAbsAlphaPower

#Participant 610: 
participant610ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant610ChannelAvgAbsAlphaPower += math.sqrt((participant610EEGDataFullDFTChannelAvg[j].real ** 2) + (participant610EEGDataFullDFTChannelAvg[j].imag ** 2))
participant610EEGFeatures[0, 1] = participant610ChannelAvgAbsAlphaPower

#Participant 613: 
participant613ChannelAvgAbsAlphaPower = 0
for j in range(130, 192):  
    participant613ChannelAvgAbsAlphaPower += math.sqrt((participant613EEGDataFullDFTChannelAvg[j].real ** 2) + (participant613EEGDataFullDFTChannelAvg[j].imag ** 2))
participant613EEGFeatures[0, 1] = participant613ChannelAvgAbsAlphaPower

#Participant 614: 
participant614ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant614ChannelAvgAbsAlphaPower += math.sqrt((participant614EEGDataFullDFTChannelAvg[j].real ** 2) + (participant614EEGDataFullDFTChannelAvg[j].imag ** 2))
participant614EEGFeatures[0, 1] = participant614ChannelAvgAbsAlphaPower

#Participant 616: 
participant616ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant616ChannelAvgAbsAlphaPower += math.sqrt((participant616EEGDataFullDFTChannelAvg[j].real ** 2) + (participant616EEGDataFullDFTChannelAvg[j].imag ** 2))
participant616EEGFeatures[0, 1] = participant616ChannelAvgAbsAlphaPower

#Participant 622: 
participant622ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant622ChannelAvgAbsAlphaPower += math.sqrt((participant622EEGDataFullDFTChannelAvg[j].real ** 2) + (participant622EEGDataFullDFTChannelAvg[j].imag ** 2))
participant622EEGFeatures[0, 1] = participant622ChannelAvgAbsAlphaPower

#Participant 624: 
participant624ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant624ChannelAvgAbsAlphaPower += math.sqrt((participant624EEGDataFullDFTChannelAvg[j].real ** 2) + (participant624EEGDataFullDFTChannelAvg[j].imag ** 2))
participant624EEGFeatures[0, 1] = participant624ChannelAvgAbsAlphaPower

#Participant 625: 
participant625ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant625ChannelAvgAbsAlphaPower += math.sqrt((participant625EEGDataFullDFTChannelAvg[j].real ** 2) + (participant625EEGDataFullDFTChannelAvg[j].imag ** 2))
participant625EEGFeatures[0, 1] = participant625ChannelAvgAbsAlphaPower

#Participant 626: 
participant626ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant626ChannelAvgAbsAlphaPower += math.sqrt((participant626EEGDataFullDFTChannelAvg[j].real ** 2) + (participant626EEGDataFullDFTChannelAvg[j].imag ** 2))
participant626EEGFeatures[0, 1] = participant626ChannelAvgAbsAlphaPower

#Participant 627: 
participant627ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant627ChannelAvgAbsAlphaPower += math.sqrt((participant627EEGDataFullDFTChannelAvg[j].real ** 2) + (participant627EEGDataFullDFTChannelAvg[j].imag ** 2))
participant627EEGFeatures[0, 1] = participant627ChannelAvgAbsAlphaPower

#Participant 628: 
participant628ChannelAvgAbsAlphaPower = 0
for j in range(130, 192): 
    participant628ChannelAvgAbsAlphaPower += math.sqrt((participant628EEGDataFullDFTChannelAvg[j].real ** 2) + (participant628EEGDataFullDFTChannelAvg[j].imag ** 2))
participant628EEGFeatures[0, 1] = participant628ChannelAvgAbsAlphaPower



#Computing Feature Variable 2: channelAvgAbsThetaPower (for all Participants) 

#Participant 507: 
participant507ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant507ChannelAvgAbsThetaPower += math.sqrt((participant507EEGDataFullDFTChannelAvg[k].real ** 2) + (participant507EEGDataFullDFTChannelAvg[k].imag ** 2))
participant507EEGFeatures[0, 2] = participant507ChannelAvgAbsThetaPower

#Participant 508: 
participant508ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant508ChannelAvgAbsThetaPower += math.sqrt((participant508EEGDataFullDFTChannelAvg[k].real ** 2) + (participant508EEGDataFullDFTChannelAvg[k].imag ** 2))
participant508EEGFeatures[0, 2] = participant508ChannelAvgAbsThetaPower

#Participant 509: 
participant509ChannelAvgAbsThetaPower = 0
for k in range(65, 129):  
    participant509ChannelAvgAbsThetaPower += math.sqrt((participant509EEGDataFullDFTChannelAvg[k].real ** 2) + (participant509EEGDataFullDFTChannelAvg[k].imag ** 2))
participant509EEGFeatures[0, 2] = participant509ChannelAvgAbsThetaPower

#Participant 510: 
participant510ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant510ChannelAvgAbsThetaPower += math.sqrt((participant510EEGDataFullDFTChannelAvg[k].real ** 2) + (participant510EEGDataFullDFTChannelAvg[k].imag ** 2))
participant510EEGFeatures[0, 2] = participant510ChannelAvgAbsThetaPower

#Participant 511: 
participant511ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant511ChannelAvgAbsThetaPower += math.sqrt((participant511EEGDataFullDFTChannelAvg[k].real ** 2) + (participant511EEGDataFullDFTChannelAvg[k].imag ** 2))
participant511EEGFeatures[0, 2] = participant511ChannelAvgAbsThetaPower

#Participant 512: 
participant512ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant512ChannelAvgAbsThetaPower += math.sqrt((participant512EEGDataFullDFTChannelAvg[k].real ** 2) + (participant512EEGDataFullDFTChannelAvg[k].imag ** 2))
participant512EEGFeatures[0, 2] = participant512ChannelAvgAbsThetaPower

#Participant 513: 
participant513ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant513ChannelAvgAbsThetaPower += math.sqrt((participant513EEGDataFullDFTChannelAvg[k].real ** 2) + (participant513EEGDataFullDFTChannelAvg[k].imag ** 2))
participant513EEGFeatures[0, 2] = participant513ChannelAvgAbsThetaPower

#Participant 514: 
participant514ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant514ChannelAvgAbsThetaPower += math.sqrt((participant514EEGDataFullDFTChannelAvg[k].real ** 2) + (participant514EEGDataFullDFTChannelAvg[k].imag ** 2))
participant514EEGFeatures[0, 2] = participant514ChannelAvgAbsThetaPower

#Participant 515: 
participant515ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant515ChannelAvgAbsThetaPower += math.sqrt((participant515EEGDataFullDFTChannelAvg[k].real ** 2) + (participant515EEGDataFullDFTChannelAvg[k].imag ** 2))
participant515EEGFeatures[0, 2] = participant515ChannelAvgAbsThetaPower

#Participant 516: 
participant516ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant516ChannelAvgAbsThetaPower += math.sqrt((participant516EEGDataFullDFTChannelAvg[k].real ** 2) + (participant516EEGDataFullDFTChannelAvg[k].imag ** 2))
participant516EEGFeatures[0, 2] = participant516ChannelAvgAbsThetaPower

#Participant 517: 
participant517ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant517ChannelAvgAbsThetaPower += math.sqrt((participant517EEGDataFullDFTChannelAvg[k].real ** 2) + (participant517EEGDataFullDFTChannelAvg[k].imag ** 2))
participant517EEGFeatures[0, 2] = participant517ChannelAvgAbsThetaPower

#Participant 518: 
participant518ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant518ChannelAvgAbsThetaPower += math.sqrt((participant518EEGDataFullDFTChannelAvg[k].real ** 2) + (participant518EEGDataFullDFTChannelAvg[k].imag ** 2))
participant518EEGFeatures[0, 2] = participant518ChannelAvgAbsThetaPower

#Participant 519: 
participant519ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant519ChannelAvgAbsThetaPower += math.sqrt((participant519EEGDataFullDFTChannelAvg[k].real ** 2) + (participant519EEGDataFullDFTChannelAvg[k].imag ** 2))
participant519EEGFeatures[0, 2] = participant519ChannelAvgAbsThetaPower

#Participant 520: 
participant520ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant520ChannelAvgAbsThetaPower += math.sqrt((participant520EEGDataFullDFTChannelAvg[k].real ** 2) + (participant520EEGDataFullDFTChannelAvg[k].imag ** 2))
participant520EEGFeatures[0, 2] = participant520ChannelAvgAbsThetaPower

#Participant 521: 
participant521ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant521ChannelAvgAbsThetaPower += math.sqrt((participant521EEGDataFullDFTChannelAvg[k].real ** 2) + (participant521EEGDataFullDFTChannelAvg[k].imag ** 2))
participant521EEGFeatures[0, 2] = participant521ChannelAvgAbsThetaPower

#Participant 522: 
participant522ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant522ChannelAvgAbsThetaPower += math.sqrt((participant522EEGDataFullDFTChannelAvg[k].real ** 2) + (participant522EEGDataFullDFTChannelAvg[k].imag ** 2))
participant522EEGFeatures[0, 2] = participant522ChannelAvgAbsThetaPower

#Participant 523: 
participant523ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant523ChannelAvgAbsThetaPower += math.sqrt((participant523EEGDataFullDFTChannelAvg[k].real ** 2) + (participant523EEGDataFullDFTChannelAvg[k].imag ** 2))
participant523EEGFeatures[0, 2] = participant523ChannelAvgAbsThetaPower

#Participant 524: 
participant524ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant524ChannelAvgAbsThetaPower += math.sqrt((participant524EEGDataFullDFTChannelAvg[k].real ** 2) + (participant524EEGDataFullDFTChannelAvg[k].imag ** 2))
participant524EEGFeatures[0, 2] = participant524ChannelAvgAbsThetaPower

#Participant 526: 
participant526ChannelAvgAbsThetaPower = 0 
for k in range(65, 129):  
    participant526ChannelAvgAbsThetaPower += math.sqrt((participant526EEGDataFullDFTChannelAvg[k].real ** 2) + (participant526EEGDataFullDFTChannelAvg[k].imag ** 2))
participant526EEGFeatures[0, 2] = participant526ChannelAvgAbsThetaPower

#Participant 527: 
participant527ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant527ChannelAvgAbsThetaPower += math.sqrt((participant527EEGDataFullDFTChannelAvg[k].real ** 2) + (participant527EEGDataFullDFTChannelAvg[k].imag ** 2))
participant527EEGFeatures[0, 2] = participant527ChannelAvgAbsThetaPower

#Participant 559: 
participant559ChannelAvgAbsThetaPower = 0
for k in range(65, 129):  
    participant559ChannelAvgAbsThetaPower += math.sqrt((participant559EEGDataFullDFTChannelAvg[k].real ** 2) + (participant559EEGDataFullDFTChannelAvg[k].imag ** 2))
participant559EEGFeatures[0, 2] = participant559ChannelAvgAbsThetaPower

#Participant 561: 
participant561ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant561ChannelAvgAbsThetaPower += math.sqrt((participant561EEGDataFullDFTChannelAvg[k].real ** 2) + (participant561EEGDataFullDFTChannelAvg[k].imag ** 2))
participant561EEGFeatures[0, 2] = participant561ChannelAvgAbsThetaPower

#Participant 565: 
participant565ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant565ChannelAvgAbsThetaPower += math.sqrt((participant565EEGDataFullDFTChannelAvg[k].real ** 2) + (participant565EEGDataFullDFTChannelAvg[k].imag ** 2))
participant565EEGFeatures[0, 2] = participant565ChannelAvgAbsThetaPower

#Participant 567: 
participant567ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant567ChannelAvgAbsThetaPower += math.sqrt((participant567EEGDataFullDFTChannelAvg[k].real ** 2) + (participant567EEGDataFullDFTChannelAvg[k].imag ** 2))
participant567EEGFeatures[0, 2] = participant567ChannelAvgAbsThetaPower

#Participant 587: 
participant587ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant587ChannelAvgAbsThetaPower += math.sqrt((participant587EEGDataFullDFTChannelAvg[k].real ** 2) + (participant587EEGDataFullDFTChannelAvg[k].imag ** 2))
participant587EEGFeatures[0, 2] = participant587ChannelAvgAbsThetaPower

#Participant 591: 
participant591ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant591ChannelAvgAbsThetaPower += math.sqrt((participant591EEGDataFullDFTChannelAvg[k].real ** 2) + (participant591EEGDataFullDFTChannelAvg[k].imag ** 2))
participant591EEGFeatures[0, 2] = participant591ChannelAvgAbsThetaPower

#Participant 594: 
participant594ChannelAvgAbsThetaPower = 0
for k in range(65, 129):  
    participant594ChannelAvgAbsThetaPower += math.sqrt((participant594EEGDataFullDFTChannelAvg[k].real ** 2) + (participant594EEGDataFullDFTChannelAvg[k].imag ** 2))
participant594EEGFeatures[0, 2] = participant594ChannelAvgAbsThetaPower

#Participant 595: 
participant595ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant595ChannelAvgAbsThetaPower += math.sqrt((participant595EEGDataFullDFTChannelAvg[k].real ** 2) + (participant595EEGDataFullDFTChannelAvg[k].imag ** 2))
participant595EEGFeatures[0, 2] = participant595ChannelAvgAbsThetaPower

#Participant 597: 
participant597ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant597ChannelAvgAbsThetaPower += math.sqrt((participant597EEGDataFullDFTChannelAvg[k].real ** 2) + (participant597EEGDataFullDFTChannelAvg[k].imag ** 2))
participant597EEGFeatures[0, 2] = participant597ChannelAvgAbsThetaPower

#Participant 605: 
participant605ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant605ChannelAvgAbsThetaPower += math.sqrt((participant605EEGDataFullDFTChannelAvg[k].real ** 2) + (participant605EEGDataFullDFTChannelAvg[k].imag ** 2))
participant605EEGFeatures[0, 2] = participant605ChannelAvgAbsThetaPower

#Participant 607: 
participant607ChannelAvgAbsThetaPower = 0
for k in range(65, 129):  
    participant607ChannelAvgAbsThetaPower += math.sqrt((participant607EEGDataFullDFTChannelAvg[k].real ** 2) + (participant607EEGDataFullDFTChannelAvg[k].imag ** 2))
participant607EEGFeatures[0, 2] = participant607ChannelAvgAbsThetaPower

#Participant 610: 
participant610ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant610ChannelAvgAbsThetaPower += math.sqrt((participant610EEGDataFullDFTChannelAvg[k].real ** 2) + (participant610EEGDataFullDFTChannelAvg[k].imag ** 2))
participant610EEGFeatures[0, 2] = participant610ChannelAvgAbsThetaPower

#Participant 613: 
participant613ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant613ChannelAvgAbsThetaPower += math.sqrt((participant613EEGDataFullDFTChannelAvg[k].real ** 2) + (participant613EEGDataFullDFTChannelAvg[k].imag ** 2))
participant613EEGFeatures[0, 2] = participant613ChannelAvgAbsThetaPower

#Participant 614: 
participant614ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant614ChannelAvgAbsThetaPower += math.sqrt((participant614EEGDataFullDFTChannelAvg[k].real ** 2) + (participant614EEGDataFullDFTChannelAvg[k].imag ** 2))
participant614EEGFeatures[0, 2] = participant614ChannelAvgAbsThetaPower

#Participant 616: 
participant616ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant616ChannelAvgAbsThetaPower += math.sqrt((participant616EEGDataFullDFTChannelAvg[k].real ** 2) + (participant616EEGDataFullDFTChannelAvg[k].imag ** 2))
participant616EEGFeatures[0, 2] = participant616ChannelAvgAbsThetaPower

#Participant 622: 
participant622ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant622ChannelAvgAbsThetaPower += math.sqrt((participant622EEGDataFullDFTChannelAvg[k].real ** 2) + (participant622EEGDataFullDFTChannelAvg[k].imag ** 2))
participant622EEGFeatures[0, 2] = participant622ChannelAvgAbsThetaPower

#Participant 624: 
participant624ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant624ChannelAvgAbsThetaPower += math.sqrt((participant624EEGDataFullDFTChannelAvg[k].real ** 2) + (participant624EEGDataFullDFTChannelAvg[k].imag ** 2))
participant624EEGFeatures[0, 2] = participant624ChannelAvgAbsThetaPower

#Participant 625: 
participant625ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant625ChannelAvgAbsThetaPower += math.sqrt((participant625EEGDataFullDFTChannelAvg[k].real ** 2) + (participant625EEGDataFullDFTChannelAvg[k].imag ** 2))
participant625EEGFeatures[0, 2] = participant625ChannelAvgAbsThetaPower

#Participant 626: 
participant626ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant626ChannelAvgAbsThetaPower += math.sqrt((participant626EEGDataFullDFTChannelAvg[k].real ** 2) + (participant626EEGDataFullDFTChannelAvg[k].imag ** 2))
participant626EEGFeatures[0, 2] = participant626ChannelAvgAbsThetaPower

#Participant 627: 
participant627ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant627ChannelAvgAbsThetaPower += math.sqrt((participant627EEGDataFullDFTChannelAvg[k].real ** 2) + (participant627EEGDataFullDFTChannelAvg[k].imag ** 2))
participant627EEGFeatures[0, 2] = participant627ChannelAvgAbsThetaPower

#Participant 628: 
participant628ChannelAvgAbsThetaPower = 0
for k in range(65, 129): 
    participant628ChannelAvgAbsThetaPower += math.sqrt((participant628EEGDataFullDFTChannelAvg[k].real ** 2) + (participant628EEGDataFullDFTChannelAvg[k].imag ** 2))
participant628EEGFeatures[0, 2] = participant628ChannelAvgAbsThetaPower



#Computing Feature Variable 3: channelAvgRelAlphaPower (for all Participants) 
#7/9/2021 Edit: For each block of participant-specific code, I should also append an ~8000 element array containing ***each*** frequency bins' power values to the "EEGChannelAvgTotalPowerList". 

#Participant 507: 
#7/9/2021: Code Modified to Also Calculate individual frequency bins' DFT power values to use as tiny feature variables for the Convolutional Neural Network Model.  
participant507ChannelTotalPower = 0
participant507ChannelAvgDFTPowerVals = np.zeros(participant507EEGDataFullDFTChannelAvg.shape) 
for m in range(0, participant507EEGDataFullDFTChannelAvg.size - 1):
    participant507ChannelTotalPower += math.sqrt((participant507EEGDataFullDFTChannelAvg[m].real ** 2) + (participant507EEGDataFullDFTChannelAvg[m].imag ** 2))
    participant507ChannelAvgDFTPowerVals[m] = math.sqrt((participant507EEGDataFullDFTChannelAvg[m].real ** 2) + (participant507EEGDataFullDFTChannelAvg[m].imag ** 2))
participant507ChannelAvgRelAlphaPower = participant507ChannelAvgAbsAlphaPower / participant507ChannelTotalPower
participant507EEGFeatures[0, 3] = participant507ChannelAvgRelAlphaPower

#Participant 508: 
participant508ChannelTotalPower = 0
for m in range(0, participant508EEGDataFullDFTChannelAvg.size - 1):
    participant508ChannelTotalPower += math.sqrt((participant508EEGDataFullDFTChannelAvg[m].real ** 2) + (participant508EEGDataFullDFTChannelAvg[m].imag ** 2))
participant508ChannelAvgRelAlphaPower = participant508ChannelAvgAbsAlphaPower / participant508ChannelTotalPower
participant508EEGFeatures[0, 3] = participant508ChannelAvgRelAlphaPower

#Participant 509: 
participant509ChannelTotalPower = 0
for m in range(0, participant509EEGDataFullDFTChannelAvg.size - 1):
    participant509ChannelTotalPower += math.sqrt((participant509EEGDataFullDFTChannelAvg[m].real ** 2) + (participant509EEGDataFullDFTChannelAvg[m].imag ** 2))
participant509ChannelAvgRelAlphaPower = participant509ChannelAvgAbsAlphaPower / participant509ChannelTotalPower
participant509EEGFeatures[0, 3] = participant509ChannelAvgRelAlphaPower

#Participant 510: 
participant510ChannelTotalPower = 0
for m in range(0, participant510EEGDataFullDFTChannelAvg.size - 1):
    participant510ChannelTotalPower += math.sqrt((participant510EEGDataFullDFTChannelAvg[m].real ** 2) + (participant510EEGDataFullDFTChannelAvg[m].imag ** 2))
participant510ChannelAvgRelAlphaPower = participant510ChannelAvgAbsAlphaPower / participant510ChannelTotalPower
participant510EEGFeatures[0, 3] = participant510ChannelAvgRelAlphaPower

#Participant 511: 
participant511ChannelTotalPower = 0
for m in range(0, participant511EEGDataFullDFTChannelAvg.size - 1):
    participant511ChannelTotalPower += math.sqrt((participant511EEGDataFullDFTChannelAvg[m].real ** 2) + (participant511EEGDataFullDFTChannelAvg[m].imag ** 2))
participant511ChannelAvgRelAlphaPower = participant511ChannelAvgAbsAlphaPower / participant511ChannelTotalPower
participant511EEGFeatures[0, 3] = participant511ChannelAvgRelAlphaPower

#Participant 512: 
participant512ChannelTotalPower = 0
for m in range(0, participant512EEGDataFullDFTChannelAvg.size - 1):
    participant512ChannelTotalPower += math.sqrt((participant512EEGDataFullDFTChannelAvg[m].real ** 2) + (participant512EEGDataFullDFTChannelAvg[m].imag ** 2))
participant512ChannelAvgRelAlphaPower = participant512ChannelAvgAbsAlphaPower / participant512ChannelTotalPower
participant512EEGFeatures[0, 3] = participant512ChannelAvgRelAlphaPower

#Participant 513: 
participant513ChannelTotalPower = 0
for m in range(0, participant513EEGDataFullDFTChannelAvg.size - 1):
    participant513ChannelTotalPower += math.sqrt((participant513EEGDataFullDFTChannelAvg[m].real ** 2) + (participant513EEGDataFullDFTChannelAvg[m].imag ** 2))
participant513ChannelAvgRelAlphaPower = participant513ChannelAvgAbsAlphaPower / participant513ChannelTotalPower
participant513EEGFeatures[0, 3] = participant513ChannelAvgRelAlphaPower

#Participant 514: 
participant514ChannelTotalPower = 0
for m in range(0, participant514EEGDataFullDFTChannelAvg.size - 1):
    participant514ChannelTotalPower += math.sqrt((participant514EEGDataFullDFTChannelAvg[m].real ** 2) + (participant514EEGDataFullDFTChannelAvg[m].imag ** 2))
participant514ChannelAvgRelAlphaPower = participant514ChannelAvgAbsAlphaPower / participant514ChannelTotalPower
participant514EEGFeatures[0, 3] = participant514ChannelAvgRelAlphaPower

#Participant 515: 
participant515ChannelTotalPower = 0
for m in range(0, participant515EEGDataFullDFTChannelAvg.size - 1):
    participant515ChannelTotalPower += math.sqrt((participant515EEGDataFullDFTChannelAvg[m].real ** 2) + (participant515EEGDataFullDFTChannelAvg[m].imag ** 2))
participant515ChannelAvgRelAlphaPower = participant515ChannelAvgAbsAlphaPower / participant515ChannelTotalPower
participant515EEGFeatures[0, 3] = participant515ChannelAvgRelAlphaPower

#Participant 516: 
participant516ChannelTotalPower = 0
for m in range(0, participant516EEGDataFullDFTChannelAvg.size - 1):
    participant516ChannelTotalPower += math.sqrt((participant516EEGDataFullDFTChannelAvg[m].real ** 2) + (participant516EEGDataFullDFTChannelAvg[m].imag ** 2))
participant516ChannelAvgRelAlphaPower = participant516ChannelAvgAbsAlphaPower / participant516ChannelTotalPower
participant516EEGFeatures[0, 3] = participant516ChannelAvgRelAlphaPower

#Participant 517: 
participant517ChannelTotalPower = 0
for m in range(0, participant517EEGDataFullDFTChannelAvg.size - 1):
    participant517ChannelTotalPower += math.sqrt((participant517EEGDataFullDFTChannelAvg[m].real ** 2) + (participant517EEGDataFullDFTChannelAvg[m].imag ** 2))
participant517ChannelAvgRelAlphaPower = participant517ChannelAvgAbsAlphaPower / participant517ChannelTotalPower
participant517EEGFeatures[0, 3] = participant517ChannelAvgRelAlphaPower

#Participant 518: 
participant518ChannelTotalPower = 0
for m in range(0, participant518EEGDataFullDFTChannelAvg.size - 1):
    participant518ChannelTotalPower += math.sqrt((participant518EEGDataFullDFTChannelAvg[m].real ** 2) + (participant518EEGDataFullDFTChannelAvg[m].imag ** 2))
participant518ChannelAvgRelAlphaPower = participant518ChannelAvgAbsAlphaPower / participant518ChannelTotalPower
participant518EEGFeatures[0, 3] = participant518ChannelAvgRelAlphaPower

#Participant 519: 
participant519ChannelTotalPower = 0
for m in range(0, participant519EEGDataFullDFTChannelAvg.size - 1):
    participant519ChannelTotalPower += math.sqrt((participant519EEGDataFullDFTChannelAvg[m].real ** 2) + (participant519EEGDataFullDFTChannelAvg[m].imag ** 2))
participant519ChannelAvgRelAlphaPower = participant519ChannelAvgAbsAlphaPower / participant519ChannelTotalPower
participant519EEGFeatures[0, 3] = participant519ChannelAvgRelAlphaPower

#Participant 520: 
participant520ChannelTotalPower = 0
for m in range(0, participant520EEGDataFullDFTChannelAvg.size - 1):
    participant520ChannelTotalPower += math.sqrt((participant520EEGDataFullDFTChannelAvg[m].real ** 2) + (participant520EEGDataFullDFTChannelAvg[m].imag ** 2))
participant520ChannelAvgRelAlphaPower = participant520ChannelAvgAbsAlphaPower / participant520ChannelTotalPower
participant520EEGFeatures[0, 3] = participant520ChannelAvgRelAlphaPower

#Participant 521: 
participant521ChannelTotalPower = 0
for m in range(0, participant521EEGDataFullDFTChannelAvg.size - 1):
    participant521ChannelTotalPower += math.sqrt((participant521EEGDataFullDFTChannelAvg[m].real ** 2) + (participant521EEGDataFullDFTChannelAvg[m].imag ** 2))
participant521ChannelAvgRelAlphaPower = participant521ChannelAvgAbsAlphaPower / participant521ChannelTotalPower
participant521EEGFeatures[0, 3] = participant521ChannelAvgRelAlphaPower

#Participant 522: 
participant522ChannelTotalPower = 0
for m in range(0, participant522EEGDataFullDFTChannelAvg.size - 1):
    participant522ChannelTotalPower += math.sqrt((participant522EEGDataFullDFTChannelAvg[m].real ** 2) + (participant522EEGDataFullDFTChannelAvg[m].imag ** 2))
participant522ChannelAvgRelAlphaPower = participant522ChannelAvgAbsAlphaPower / participant522ChannelTotalPower
participant522EEGFeatures[0, 3] = participant522ChannelAvgRelAlphaPower

#Participant 523: 
participant523ChannelTotalPower = 0
for m in range(0, participant523EEGDataFullDFTChannelAvg.size - 1):
    participant523ChannelTotalPower += math.sqrt((participant523EEGDataFullDFTChannelAvg[m].real ** 2) + (participant523EEGDataFullDFTChannelAvg[m].imag ** 2))
participant523ChannelAvgRelAlphaPower = participant523ChannelAvgAbsAlphaPower / participant523ChannelTotalPower
participant523EEGFeatures[0, 3] = participant523ChannelAvgRelAlphaPower

#Participant 524: 
participant524ChannelTotalPower = 0
for m in range(0, participant524EEGDataFullDFTChannelAvg.size - 1):
    participant524ChannelTotalPower += math.sqrt((participant524EEGDataFullDFTChannelAvg[m].real ** 2) + (participant524EEGDataFullDFTChannelAvg[m].imag ** 2))
participant524ChannelAvgRelAlphaPower = participant524ChannelAvgAbsAlphaPower / participant524ChannelTotalPower
participant524EEGFeatures[0, 3] = participant524ChannelAvgRelAlphaPower

#Participant 526: 
participant526ChannelTotalPower = 0
for m in range(0, participant526EEGDataFullDFTChannelAvg.size - 1):
    participant526ChannelTotalPower += math.sqrt((participant526EEGDataFullDFTChannelAvg[m].real ** 2) + (participant526EEGDataFullDFTChannelAvg[m].imag ** 2))
participant526ChannelAvgRelAlphaPower = participant526ChannelAvgAbsAlphaPower / participant526ChannelTotalPower
participant526EEGFeatures[0, 3] = participant526ChannelAvgRelAlphaPower

#Participant 527: 
participant527ChannelTotalPower = 0
for m in range(0, participant527EEGDataFullDFTChannelAvg.size - 1):
    participant527ChannelTotalPower += math.sqrt((participant527EEGDataFullDFTChannelAvg[m].real ** 2) + (participant527EEGDataFullDFTChannelAvg[m].imag ** 2))
participant527ChannelAvgRelAlphaPower = participant527ChannelAvgAbsAlphaPower / participant527ChannelTotalPower
participant527EEGFeatures[0, 3] = participant527ChannelAvgRelAlphaPower

#Participant 559: 
participant559ChannelTotalPower = 0
for m in range(0, participant559EEGDataFullDFTChannelAvg.size - 1):
    participant559ChannelTotalPower += math.sqrt((participant559EEGDataFullDFTChannelAvg[m].real ** 2) + (participant559EEGDataFullDFTChannelAvg[m].imag ** 2))
participant559ChannelAvgRelAlphaPower = participant559ChannelAvgAbsAlphaPower / participant559ChannelTotalPower
participant559EEGFeatures[0, 3] = participant559ChannelAvgRelAlphaPower

#Participant 561: 
participant561ChannelTotalPower = 0
for m in range(0, participant561EEGDataFullDFTChannelAvg.size - 1):
    participant561ChannelTotalPower += math.sqrt((participant561EEGDataFullDFTChannelAvg[m].real ** 2) + (participant561EEGDataFullDFTChannelAvg[m].imag ** 2))
participant561ChannelAvgRelAlphaPower = participant561ChannelAvgAbsAlphaPower / participant561ChannelTotalPower
participant561EEGFeatures[0, 3] = participant561ChannelAvgRelAlphaPower

#Participant 565: 
participant565ChannelTotalPower = 0
for m in range(0, participant565EEGDataFullDFTChannelAvg.size - 1):
    participant565ChannelTotalPower += math.sqrt((participant565EEGDataFullDFTChannelAvg[m].real ** 2) + (participant565EEGDataFullDFTChannelAvg[m].imag ** 2))
participant565ChannelAvgRelAlphaPower = participant565ChannelAvgAbsAlphaPower / participant565ChannelTotalPower
participant565EEGFeatures[0, 3] = participant565ChannelAvgRelAlphaPower

#Participant 567: 
participant567ChannelTotalPower = 0
for m in range(0, participant567EEGDataFullDFTChannelAvg.size - 1):
    participant567ChannelTotalPower += math.sqrt((participant567EEGDataFullDFTChannelAvg[m].real ** 2) + (participant567EEGDataFullDFTChannelAvg[m].imag ** 2))
participant567ChannelAvgRelAlphaPower = participant567ChannelAvgAbsAlphaPower / participant567ChannelTotalPower
participant567EEGFeatures[0, 3] = participant567ChannelAvgRelAlphaPower

#Participant 587: 
participant587ChannelTotalPower = 0
for m in range(0, participant587EEGDataFullDFTChannelAvg.size - 1):
    participant587ChannelTotalPower += math.sqrt((participant587EEGDataFullDFTChannelAvg[m].real ** 2) + (participant587EEGDataFullDFTChannelAvg[m].imag ** 2))
participant587ChannelAvgRelAlphaPower = participant587ChannelAvgAbsAlphaPower / participant587ChannelTotalPower
participant587EEGFeatures[0, 3] = participant587ChannelAvgRelAlphaPower

#Participant 591: 
participant591ChannelTotalPower = 0
for m in range(0, participant591EEGDataFullDFTChannelAvg.size - 1):
    participant591ChannelTotalPower += math.sqrt((participant591EEGDataFullDFTChannelAvg[m].real ** 2) + (participant591EEGDataFullDFTChannelAvg[m].imag ** 2))
participant591ChannelAvgRelAlphaPower = participant591ChannelAvgAbsAlphaPower / participant591ChannelTotalPower
participant591EEGFeatures[0, 3] = participant591ChannelAvgRelAlphaPower

#Participant 594: 
participant594ChannelTotalPower = 0
for m in range(0, participant594EEGDataFullDFTChannelAvg.size - 1):
    participant594ChannelTotalPower += math.sqrt((participant594EEGDataFullDFTChannelAvg[m].real ** 2) + (participant594EEGDataFullDFTChannelAvg[m].imag ** 2))
participant594ChannelAvgRelAlphaPower = participant594ChannelAvgAbsAlphaPower / participant594ChannelTotalPower
participant594EEGFeatures[0, 3] = participant594ChannelAvgRelAlphaPower

#Participant 595: 
participant595ChannelTotalPower = 0
for m in range(0, participant595EEGDataFullDFTChannelAvg.size - 1):
    participant595ChannelTotalPower += math.sqrt((participant595EEGDataFullDFTChannelAvg[m].real ** 2) + (participant595EEGDataFullDFTChannelAvg[m].imag ** 2))
participant595ChannelAvgRelAlphaPower = participant595ChannelAvgAbsAlphaPower / participant595ChannelTotalPower
participant595EEGFeatures[0, 3] = participant595ChannelAvgRelAlphaPower

#Participant 597: 
participant597ChannelTotalPower = 0
for m in range(0, participant597EEGDataFullDFTChannelAvg.size - 1):
    participant597ChannelTotalPower += math.sqrt((participant597EEGDataFullDFTChannelAvg[m].real ** 2) + (participant597EEGDataFullDFTChannelAvg[m].imag ** 2))
participant597ChannelAvgRelAlphaPower = participant597ChannelAvgAbsAlphaPower / participant597ChannelTotalPower
participant597EEGFeatures[0, 3] = participant597ChannelAvgRelAlphaPower

#Participant 605: 
participant605ChannelTotalPower = 0
for m in range(0, participant605EEGDataFullDFTChannelAvg.size - 1):
    participant605ChannelTotalPower += math.sqrt((participant605EEGDataFullDFTChannelAvg[m].real ** 2) + (participant605EEGDataFullDFTChannelAvg[m].imag ** 2))
participant605ChannelAvgRelAlphaPower = participant605ChannelAvgAbsAlphaPower / participant605ChannelTotalPower
participant605EEGFeatures[0, 3] = participant605ChannelAvgRelAlphaPower

#Participant 607: 
participant607ChannelTotalPower = 0
for m in range(0, participant607EEGDataFullDFTChannelAvg.size - 1):
    participant607ChannelTotalPower += math.sqrt((participant607EEGDataFullDFTChannelAvg[m].real ** 2) + (participant607EEGDataFullDFTChannelAvg[m].imag ** 2))
participant607ChannelAvgRelAlphaPower = participant607ChannelAvgAbsAlphaPower / participant607ChannelTotalPower
participant607EEGFeatures[0, 3] = participant607ChannelAvgRelAlphaPower

#Participant 610: 
participant610ChannelTotalPower = 0
for m in range(0, participant610EEGDataFullDFTChannelAvg.size - 1):
    participant610ChannelTotalPower += math.sqrt((participant610EEGDataFullDFTChannelAvg[m].real ** 2) + (participant610EEGDataFullDFTChannelAvg[m].imag ** 2))
participant610ChannelAvgRelAlphaPower = participant610ChannelAvgAbsAlphaPower / participant610ChannelTotalPower
participant610EEGFeatures[0, 3] = participant610ChannelAvgRelAlphaPower

#Participant 613: 
participant613ChannelTotalPower = 0
for m in range(0, participant613EEGDataFullDFTChannelAvg.size - 1):
    participant613ChannelTotalPower += math.sqrt((participant613EEGDataFullDFTChannelAvg[m].real ** 2) + (participant613EEGDataFullDFTChannelAvg[m].imag ** 2))
participant613ChannelAvgRelAlphaPower = participant613ChannelAvgAbsAlphaPower / participant613ChannelTotalPower
participant613EEGFeatures[0, 3] = participant613ChannelAvgRelAlphaPower

#Participant 614: 
participant614ChannelTotalPower = 0
for m in range(0, participant614EEGDataFullDFTChannelAvg.size - 1):
    participant614ChannelTotalPower += math.sqrt((participant614EEGDataFullDFTChannelAvg[m].real ** 2) + (participant614EEGDataFullDFTChannelAvg[m].imag ** 2))
participant614ChannelAvgRelAlphaPower = participant614ChannelAvgAbsAlphaPower / participant614ChannelTotalPower
participant614EEGFeatures[0, 3] = participant614ChannelAvgRelAlphaPower

#Participant 616: 
participant616ChannelTotalPower = 0
for m in range(0, participant616EEGDataFullDFTChannelAvg.size - 1):
    participant616ChannelTotalPower += math.sqrt((participant616EEGDataFullDFTChannelAvg[m].real ** 2) + (participant616EEGDataFullDFTChannelAvg[m].imag ** 2))
participant616ChannelAvgRelAlphaPower = participant616ChannelAvgAbsAlphaPower / participant616ChannelTotalPower
participant616EEGFeatures[0, 3] = participant616ChannelAvgRelAlphaPower

#Participant 622: 
participant622ChannelTotalPower = 0
for m in range(0, participant622EEGDataFullDFTChannelAvg.size - 1):
    participant622ChannelTotalPower += math.sqrt((participant622EEGDataFullDFTChannelAvg[m].real ** 2) + (participant622EEGDataFullDFTChannelAvg[m].imag ** 2))
participant622ChannelAvgRelAlphaPower = participant622ChannelAvgAbsAlphaPower / participant622ChannelTotalPower
participant622EEGFeatures[0, 3] = participant622ChannelAvgRelAlphaPower

#Participant 624: 
participant624ChannelTotalPower = 0
for m in range(0, participant624EEGDataFullDFTChannelAvg.size - 1):
    participant624ChannelTotalPower += math.sqrt((participant624EEGDataFullDFTChannelAvg[m].real ** 2) + (participant624EEGDataFullDFTChannelAvg[m].imag ** 2))
participant624ChannelAvgRelAlphaPower = participant624ChannelAvgAbsAlphaPower / participant624ChannelTotalPower
participant624EEGFeatures[0, 3] = participant624ChannelAvgRelAlphaPower

#Participant 625: 
participant625ChannelTotalPower = 0
for m in range(0, participant625EEGDataFullDFTChannelAvg.size - 1):
    participant625ChannelTotalPower += math.sqrt((participant625EEGDataFullDFTChannelAvg[m].real ** 2) + (participant625EEGDataFullDFTChannelAvg[m].imag ** 2))
participant625ChannelAvgRelAlphaPower = participant625ChannelAvgAbsAlphaPower / participant625ChannelTotalPower
participant625EEGFeatures[0, 3] = participant625ChannelAvgRelAlphaPower

#Participant 626: 
participant626ChannelTotalPower = 0
for m in range(0, participant626EEGDataFullDFTChannelAvg.size - 1):
    participant626ChannelTotalPower += math.sqrt((participant626EEGDataFullDFTChannelAvg[m].real ** 2) + (participant626EEGDataFullDFTChannelAvg[m].imag ** 2))
participant626ChannelAvgRelAlphaPower = participant626ChannelAvgAbsAlphaPower / participant626ChannelTotalPower
participant626EEGFeatures[0, 3] = participant626ChannelAvgRelAlphaPower

#Participant 627: 
participant627ChannelTotalPower = 0
for m in range(0, participant627EEGDataFullDFTChannelAvg.size - 1):
    participant627ChannelTotalPower += math.sqrt((participant627EEGDataFullDFTChannelAvg[m].real ** 2) + (participant627EEGDataFullDFTChannelAvg[m].imag ** 2))
participant627ChannelAvgRelAlphaPower = participant627ChannelAvgAbsAlphaPower / participant627ChannelTotalPower
participant627EEGFeatures[0, 3] = participant627ChannelAvgRelAlphaPower

#Participant 628: 
participant628ChannelTotalPower = 0
for m in range(0, participant628EEGDataFullDFTChannelAvg.size - 1):
    participant628ChannelTotalPower += math.sqrt((participant628EEGDataFullDFTChannelAvg[m].real ** 2) + (participant628EEGDataFullDFTChannelAvg[m].imag ** 2))
participant628ChannelAvgRelAlphaPower = participant628ChannelAvgAbsAlphaPower / participant628ChannelTotalPower
participant628EEGFeatures[0, 3] = participant628ChannelAvgRelAlphaPower



#Computing Feature Variable 4: channelAvgRelThetaPower (for all Participants) 

#Participant 507: 
participant507ChannelAvgRelThetaPower = participant507ChannelAvgAbsThetaPower / participant507ChannelTotalPower
participant507EEGFeatures[0, 4] = participant507ChannelAvgRelThetaPower

#Participant 508: 
participant508ChannelAvgRelThetaPower = participant508ChannelAvgAbsThetaPower / participant508ChannelTotalPower
participant508EEGFeatures[0, 4] = participant508ChannelAvgRelThetaPower

#Participant 509: 
participant509ChannelAvgRelThetaPower = participant509ChannelAvgAbsThetaPower / participant509ChannelTotalPower
participant509EEGFeatures[0, 4] = participant509ChannelAvgRelThetaPower

#Participant 510: 
participant510ChannelAvgRelThetaPower = participant510ChannelAvgAbsThetaPower / participant510ChannelTotalPower
participant510EEGFeatures[0, 4] = participant510ChannelAvgRelThetaPower

#Participant 511: 
participant511ChannelAvgRelThetaPower = participant511ChannelAvgAbsThetaPower / participant511ChannelTotalPower
participant511EEGFeatures[0, 4] = participant511ChannelAvgRelThetaPower

#Participant 512: 
participant512ChannelAvgRelThetaPower = participant512ChannelAvgAbsThetaPower / participant512ChannelTotalPower
participant512EEGFeatures[0, 4] = participant512ChannelAvgRelThetaPower

#Participant 513: 
participant513ChannelAvgRelThetaPower = participant513ChannelAvgAbsThetaPower / participant513ChannelTotalPower
participant513EEGFeatures[0, 4] = participant513ChannelAvgRelThetaPower

#Participant 514: 
participant514ChannelAvgRelThetaPower = participant514ChannelAvgAbsThetaPower / participant514ChannelTotalPower
participant514EEGFeatures[0, 4] = participant514ChannelAvgRelThetaPower

#Participant 515: 
participant515ChannelAvgRelThetaPower = participant515ChannelAvgAbsThetaPower / participant515ChannelTotalPower
participant515EEGFeatures[0, 4] = participant515ChannelAvgRelThetaPower

#Participant 516: 
participant516ChannelAvgRelThetaPower = participant516ChannelAvgAbsThetaPower / participant516ChannelTotalPower
participant516EEGFeatures[0, 4] = participant516ChannelAvgRelThetaPower

#Participant 517: 
participant517ChannelAvgRelThetaPower = participant517ChannelAvgAbsThetaPower / participant517ChannelTotalPower
participant517EEGFeatures[0, 4] = participant517ChannelAvgRelThetaPower

#Participant 518: 
participant518ChannelAvgRelThetaPower = participant518ChannelAvgAbsThetaPower / participant518ChannelTotalPower
participant518EEGFeatures[0, 4] = participant518ChannelAvgRelThetaPower

#Participant 519: 
participant519ChannelAvgRelThetaPower = participant519ChannelAvgAbsThetaPower / participant519ChannelTotalPower
participant519EEGFeatures[0, 4] = participant519ChannelAvgRelThetaPower

#Participant 520: 
participant520ChannelAvgRelThetaPower = participant520ChannelAvgAbsThetaPower / participant520ChannelTotalPower
participant520EEGFeatures[0, 4] = participant520ChannelAvgRelThetaPower

#Participant 521: 
participant521ChannelAvgRelThetaPower = participant521ChannelAvgAbsThetaPower / participant521ChannelTotalPower
participant521EEGFeatures[0, 4] = participant521ChannelAvgRelThetaPower

#Participant 522: 
participant522ChannelAvgRelThetaPower = participant522ChannelAvgAbsThetaPower / participant522ChannelTotalPower
participant522EEGFeatures[0, 4] = participant522ChannelAvgRelThetaPower

#Participant 523: 
participant523ChannelAvgRelThetaPower = participant523ChannelAvgAbsThetaPower / participant523ChannelTotalPower
participant523EEGFeatures[0, 4] = participant523ChannelAvgRelThetaPower

#Participant 524: 
participant524ChannelAvgRelThetaPower = participant524ChannelAvgAbsThetaPower / participant524ChannelTotalPower
participant524EEGFeatures[0, 4] = participant524ChannelAvgRelThetaPower

#Participant 526: 
participant526ChannelAvgRelThetaPower = participant526ChannelAvgAbsThetaPower / participant526ChannelTotalPower
participant526EEGFeatures[0, 4] = participant526ChannelAvgRelThetaPower

#Participant 527: 
participant527ChannelAvgRelThetaPower = participant527ChannelAvgAbsThetaPower / participant527ChannelTotalPower
participant527EEGFeatures[0, 4] = participant527ChannelAvgRelThetaPower

#Participant 559: 
participant559ChannelAvgRelThetaPower = participant559ChannelAvgAbsThetaPower / participant559ChannelTotalPower
participant559EEGFeatures[0, 4] = participant559ChannelAvgRelThetaPower

#Participant 561: 
participant561ChannelAvgRelThetaPower = participant561ChannelAvgAbsThetaPower / participant561ChannelTotalPower
participant561EEGFeatures[0, 4] = participant561ChannelAvgRelThetaPower

#Participant 565: 
participant565ChannelAvgRelThetaPower = participant565ChannelAvgAbsThetaPower / participant565ChannelTotalPower
participant565EEGFeatures[0, 4] = participant565ChannelAvgRelThetaPower

#Participant 567: 
participant567ChannelAvgRelThetaPower = participant567ChannelAvgAbsThetaPower / participant567ChannelTotalPower
participant567EEGFeatures[0, 4] = participant567ChannelAvgRelThetaPower

#Participant 587: 
participant587ChannelAvgRelThetaPower = participant587ChannelAvgAbsThetaPower / participant587ChannelTotalPower
participant587EEGFeatures[0, 4] = participant587ChannelAvgRelThetaPower

#Participant 591: 
participant591ChannelAvgRelThetaPower = participant591ChannelAvgAbsThetaPower / participant591ChannelTotalPower
participant591EEGFeatures[0, 4] = participant591ChannelAvgRelThetaPower

#Participant 594: 
participant594ChannelAvgRelThetaPower = participant594ChannelAvgAbsThetaPower / participant594ChannelTotalPower
participant594EEGFeatures[0, 4] = participant594ChannelAvgRelThetaPower

#Participant 595: 
participant595ChannelAvgRelThetaPower = participant595ChannelAvgAbsThetaPower / participant595ChannelTotalPower
participant595EEGFeatures[0, 4] = participant595ChannelAvgRelThetaPower

#Participant 597: 
participant597ChannelAvgRelThetaPower = participant597ChannelAvgAbsThetaPower / participant597ChannelTotalPower
participant597EEGFeatures[0, 4] = participant597ChannelAvgRelThetaPower

#Participant 605: 
participant605ChannelAvgRelThetaPower = participant605ChannelAvgAbsThetaPower / participant605ChannelTotalPower
participant605EEGFeatures[0, 4] = participant605ChannelAvgRelThetaPower

#Participant 607: 
participant607ChannelAvgRelThetaPower = participant607ChannelAvgAbsThetaPower / participant607ChannelTotalPower
participant607EEGFeatures[0, 4] = participant607ChannelAvgRelThetaPower

#Participant 610: 
participant610ChannelAvgRelThetaPower = participant610ChannelAvgAbsThetaPower / participant610ChannelTotalPower
participant610EEGFeatures[0, 4] = participant610ChannelAvgRelThetaPower

#Participant 613: 
participant613ChannelAvgRelThetaPower = participant613ChannelAvgAbsThetaPower / participant613ChannelTotalPower
participant613EEGFeatures[0, 4] = participant613ChannelAvgRelThetaPower

#Participant 614: 
participant614ChannelAvgRelThetaPower = participant614ChannelAvgAbsThetaPower / participant614ChannelTotalPower
participant614EEGFeatures[0, 4] = participant614ChannelAvgRelThetaPower

#Participant 616: 
participant616ChannelAvgRelThetaPower = participant616ChannelAvgAbsThetaPower / participant616ChannelTotalPower
participant616EEGFeatures[0, 4] = participant616ChannelAvgRelThetaPower

#Participant 622: 
participant622ChannelAvgRelThetaPower = participant622ChannelAvgAbsThetaPower / participant622ChannelTotalPower
participant622EEGFeatures[0, 4] = participant622ChannelAvgRelThetaPower

#Participant 624: 
participant624ChannelAvgRelThetaPower = participant624ChannelAvgAbsThetaPower / participant624ChannelTotalPower
participant624EEGFeatures[0, 4] = participant624ChannelAvgRelThetaPower

#Participant 625: 
participant625ChannelAvgRelThetaPower = participant625ChannelAvgAbsThetaPower / participant625ChannelTotalPower
participant625EEGFeatures[0, 4] = participant625ChannelAvgRelThetaPower

#Participant 626: 
participant626ChannelAvgRelThetaPower = participant626ChannelAvgAbsThetaPower / participant626ChannelTotalPower
participant626EEGFeatures[0, 4] = participant626ChannelAvgRelThetaPower

#Participant 627: 
participant627ChannelAvgRelThetaPower = participant627ChannelAvgAbsThetaPower / participant627ChannelTotalPower
participant627EEGFeatures[0, 4] = participant627ChannelAvgRelThetaPower

#Participant 628: 
participant628ChannelAvgRelThetaPower = participant628ChannelAvgAbsThetaPower / participant628ChannelTotalPower
participant628EEGFeatures[0, 4] = participant628ChannelAvgRelThetaPower



#Computing Feature Variable 5: thetaBandAccordance (for all Participants) 



#Converting each of the 42 Participant-Specific EEG Feature Variables to a Pandas DF, and then appending them as new rows 
#to the primary EEG Features DataFrame with all of the Kavanagh Study Participants (EEGFeaturesDF). 
#The reason behind my conversion of the NumPy array containing all of the participant data into a DataFrame, 
#and then back into a NumPy Array before data normalization/scaling pre-model training is that DataFrames are more easily
#appended to and adjusted in their dimensions than NumPy arrays. I'll need to adjust the size of the DataFrame to add 
#in more rows of participants' data to it. 

#Participant 507: 
participant507EEGFeaturesDF = pd.DataFrame(participant507EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant507EEGFeaturesDF)

#Participant 508: 
participant508EEGFeaturesDF = pd.DataFrame(participant508EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant508EEGFeaturesDF)

#Participant 509: 
participant509EEGFeaturesDF = pd.DataFrame(participant509EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant509EEGFeaturesDF)

#Participant 510: 
participant510EEGFeaturesDF = pd.DataFrame(participant510EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant510EEGFeaturesDF)

#Participant 511: 
participant511EEGFeaturesDF = pd.DataFrame(participant511EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant511EEGFeaturesDF)

#Participant 512: 
participant512EEGFeaturesDF = pd.DataFrame(participant512EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant512EEGFeaturesDF)

#Participant 513: 
participant513EEGFeaturesDF = pd.DataFrame(participant513EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant513EEGFeaturesDF)

#Participant 514: 
participant514EEGFeaturesDF = pd.DataFrame(participant514EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant514EEGFeaturesDF)

#Participant 515: 
participant515EEGFeaturesDF = pd.DataFrame(participant515EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant515EEGFeaturesDF)

#Participant 516: 
participant516EEGFeaturesDF = pd.DataFrame(participant516EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant516EEGFeaturesDF)

#Participant 517: 
participant517EEGFeaturesDF = pd.DataFrame(participant517EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant517EEGFeaturesDF)

#Participant 518: 
participant518EEGFeaturesDF = pd.DataFrame(participant518EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant518EEGFeaturesDF)

#Participant 519: 
participant519EEGFeaturesDF = pd.DataFrame(participant519EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant519EEGFeaturesDF)

#Participant 520: 
participant520EEGFeaturesDF = pd.DataFrame(participant520EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant520EEGFeaturesDF)

#Participant 521: 
participant521EEGFeaturesDF = pd.DataFrame(participant521EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant521EEGFeaturesDF)

#Participant 522: 
participant522EEGFeaturesDF = pd.DataFrame(participant522EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant522EEGFeaturesDF)

#Participant 523: 
participant523EEGFeaturesDF = pd.DataFrame(participant523EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant523EEGFeaturesDF)

#Participant 524: 
participant524EEGFeaturesDF = pd.DataFrame(participant524EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant524EEGFeaturesDF)

#Participant 526: 
participant526EEGFeaturesDF = pd.DataFrame(participant526EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant526EEGFeaturesDF)

#Participant 527: 
participant527EEGFeaturesDF = pd.DataFrame(participant527EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant527EEGFeaturesDF)

#Participant 559: 
participant559EEGFeaturesDF = pd.DataFrame(participant559EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant559EEGFeaturesDF)

#Participant 561: 
participant561EEGFeaturesDF = pd.DataFrame(participant561EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant561EEGFeaturesDF)

#Participant 565: 
participant565EEGFeaturesDF = pd.DataFrame(participant565EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant565EEGFeaturesDF)

#Participant 567: 
participant567EEGFeaturesDF = pd.DataFrame(participant567EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant567EEGFeaturesDF)

#Participant 587: 
participant587EEGFeaturesDF = pd.DataFrame(participant587EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant587EEGFeaturesDF)

#Participant 591:  
participant591EEGFeaturesDF = pd.DataFrame(participant591EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant591EEGFeaturesDF)

#Participant 594: 
participant594EEGFeaturesDF = pd.DataFrame(participant594EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant594EEGFeaturesDF)

#Participant 595: 
participant595EEGFeaturesDF = pd.DataFrame(participant595EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant595EEGFeaturesDF)

#Participant 597: 
participant597EEGFeaturesDF = pd.DataFrame(participant597EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant597EEGFeaturesDF)

#Participant 605: 
participant605EEGFeaturesDF = pd.DataFrame(participant605EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant605EEGFeaturesDF)

#Participant 607: 
participant607EEGFeaturesDF = pd.DataFrame(participant607EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant607EEGFeaturesDF)

#Participant 610: 
participant610EEGFeaturesDF = pd.DataFrame(participant610EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant610EEGFeaturesDF)

#Participant 613: 
participant613EEGFeaturesDF = pd.DataFrame(participant613EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant613EEGFeaturesDF)

#Participant 614: 
participant614EEGFeaturesDF = pd.DataFrame(participant614EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant614EEGFeaturesDF)

#Participant 616: 
participant616EEGFeaturesDF = pd.DataFrame(participant616EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant616EEGFeaturesDF)

#Participant 622: 
participant622EEGFeaturesDF = pd.DataFrame(participant622EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant622EEGFeaturesDF)

#Participant 624: 
participant624EEGFeaturesDF = pd.DataFrame(participant624EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant624EEGFeaturesDF)

#Participant 625: 
participant625EEGFeaturesDF = pd.DataFrame(participant625EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant625EEGFeaturesDF)

#Participant 626: 
participant626EEGFeaturesDF = pd.DataFrame(participant626EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant626EEGFeaturesDF)

#Participant 627: 
participant627EEGFeaturesDF = pd.DataFrame(participant627EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant627EEGFeaturesDF)

#Participant 628: 
participant628EEGFeaturesDF = pd.DataFrame(participant628EEGFeatures, columns = EEGFeaturesDFColumnList)
EEGFeaturesDF = EEGFeaturesDF.append(participant628EEGFeaturesDF)



#print(participant507EEGFeatures.shape)
#print(participant507EEGFeatures)
#print(participant559EEGFeatures)
#print(participant507EEGFeaturesDF)
#print(participant559EEGFeaturesDF)
#print(EEGFeaturesDF)





### STEP 3: DATA SCALING AND OTHER PRE-PROCESSING STEPS BEFORE ML MODEL TRAINING:  ###

#Scaling (Normalizing) the EEGFeaturesDF Feature/Predictor Variables for Better Classification Algorithm Performance: 

#Converting the EEGFeaturesDF into a NumPy Array for Scaling and Application of Classifiers: 
EEGFeatures = np.asarray(EEGFeaturesDF)
scaler = StandardScaler()

#print(EEGFeaturesDF)
#print(EEGFeatures)

EEGFeaturesPredictorVarsStandardScaled = scaler.fit(EEGFeatures[:, [1,4]]).transform(EEGFeatures[:, [1,4]].astype(float))
#Setting the Target (Dependent Variable: The HC/MDD Status of each of the 42 Participants) Variable Equal to the 7th column in EEGFeatures.
#Leaving out the Theta Band Cordance feature for now, as I haven't yet learned exactly how to compute it. 
EEGFeatures[:, [1,4]] = EEGFeaturesPredictorVarsStandardScaled

#7/9/2021: For the CNN Model, I also need to scale the DFT Power features (~8000 per participant) that will be fed into the first convolutional layer using the 
#Standard Scaler library. I should do that here: 




### STEP 4: ML MODEL TRAINING AND VALIDATION TO CLASSIFY MDD VS. HC INDIVIDUALS FROM THE EEG FEATURES ###



#Model 1: K Nearest Neighbors: 

#Using a basic train_test_split approach to assess the best k-value for the KNN algorithm: 
X_train, X_test, y_train, y_test = train_test_split(EEGFeatures, EEGFeatures[:, 6], test_size=0.2, random_state=4)

#Training KNN Algorithm and Testing its Accuracy:
eegKNNScores = []
eegNeighborsAccuracyArray = np.zeros(19)
kTestIterationList = range(1,20)

for k in kTestIterationList:

    eegNeighbors = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    diagnosesPredictions = eegNeighbors.predict(X_test)
    eegNeighborsAccuracyArray[k-1] = metrics.accuracy_score(y_test, diagnosesPredictions)
    eegKNNScores.append(metrics.accuracy_score(y_test, diagnosesPredictions))
    
#print(eegKNNScores)

#Conducting k-fold cross validation i-# of times, with the k-value for the KNN algorithm with the highest classification accuracy results: 
avgEEGKNNCrossValScore = []
eegKNNCrossValScore = []

for i in range(0, 99): 
   eegKNNCrossValScore.append(cross_val_score(KNeighborsClassifier(n_neighbors = 2), EEGFeatures, EEGFeatures[:, 6], cv = 3))

avgEEGKNNCrossValScore = np.average(eegKNNCrossValScore)
print(avgEEGKNNCrossValScore)
print(participant507EEGDataFullDFT.shape)
### I only received a ~51.5% correct classification accuracy score with the KNN algorithm with (k=3) k-fold cross validation repeated 100 times and averaged. 



#Model 2: Logistic Regression: 

#Good refresher tutorial on Logistic Regression in Python: https://realpython.com/logistic-regression-python/#:~:text=%20Logistic%20Regression%20in%20Python%20With%20scikit-learn%3A%20Example,you%20have%20the%20input%20and%20output...%20More%20
#More helpful links for troubleshooting and interpreting Logistic Regression Results: 
#https://stackoverflow.com/questions/46392369/how-to-determine-if-the-predicted-probabilities-from-sklearn-logistic-regresssio
#https://www.displayr.com/how-to-interpret-logistic-regression-outputs/
#6/24/2021 Notes: There are definitely more performance metrics for the Logistic Regression model I need to assess further before moving on to the SVM/Neural Network Models 
#and potentially attempting to improve the KNN Model. 
#6/29/2021 Notes: From here on out, instead of using a basic train-test-split approach for model testing, I'll try using k-fold cross validation. I'll conduct
#k-fold cross validation a number (maybe 10-100) times, and then use the average of all of those times' accuracy results as my final result. 

#Creating lists for the averaged and raw k-fold cross validation accuracy scores (0 - 1.0) for the Cavanagh Dataset with Logistic Regression: 
avgEEGLogisticRegressionCrossValScore = []
eegLogisticRegressionCrossValScore = []

#Conducting k-fold cross validation i-# of times: 

for i in range(0, 99): 
    eegLogisticRegressionCrossValScore.append(cross_val_score(LogisticRegression(), EEGFeatures, EEGFeatures[:, 6], cv = 3))

avgEEGLogisticRegressionCrossValScore = np.average(eegLogisticRegressionCrossValScore)

print(avgEEGLogisticRegressionCrossValScore)
#My logistic regression model performed significantly better than my KNN model, at ~61% accuracy. This is still a far cry from the 75-80% accuracy range I'm striving for.
#Later on, I should make another Logistic Regression object with the simple train_test_split model assessment strategy and assess the prediction probabilities from 
#the LR model.  



#Model 3: Support Vector Machine: 
#6/29/2021: Today, I'm starting to research and implement the SVM model on my data. I'll try and use the repeated and averaged k-fold cross validation approach again for the SVM model.
#If it's annoying to implement the cross-validation method with the SVM model, I'll just run a quick test of it with a the simple train_test_split approach for now. 
eegSVM = svm.SVC(kernel = 'linear') 

eegSVM.fit(X_train, y_train)

y_pred = eegSVM.predict(X_test)

#print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


#avgEEGSVMCrossValScore = []
#eegSVMCrossValScore = []

#for i in range(0, 99): 
   #eegSVMCrossValScore.append(cross_val_score(eegSVM, EEGFeatures, EEGFeatures[:, 6], cv = 3))

#avgEEGSVMCrossValScore = np.average(eegSVMCrossValScore)
#print(avgEEGSVMCrossValScore)





#Model 4: Artificial Neural Network: Convolutional Neural Network:  
#6/30/2021: Beginning the Assessment of this Model Type: 
#7/9/2021: Found a Good YT Video with an implementation for CNNs using the Tensorflow library: https://www.youtube.com/watch?v=WvoLTXIjBYU

#As opposed to the other types of ML algorithms, for which I created specific feature variables, the "features" 
#I'll feed into the C.N.N., for each participant's data set, their full channel-averaged DFT "power" vector (~800 elements); each of these DFT power elements 
#will represent a single "feature" variable. Each of these vectors will be stored as a participant-specific NumPy array, and each of these NumPy Arrays will 
#be stored in a Python List called "EEGChannelAvgTotalPowerList", which I've already created above in Step 2. 

#When I train/test the CNN model, I'll have to iterate over the NP Arrays contained in this List to train/test the model with all of the different Cavanagh Study
#participants' data. 




