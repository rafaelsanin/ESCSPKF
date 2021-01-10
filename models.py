"""
Class objects for the dyn_model calculations
"""

# Modules
# ------------------------------------------------------------------------------

import json
import pandas as pd
from funcs import SOCfromOCVtemp 
from scipy.linalg import cholesky
import numpy as np

# Class Objects
# ------------------------------------------------------------------------------

class DataModel:
    """
    Data from battery script tests. Requires the Script class which reads the
    csv file and assigns the data to class attributes.
    """

    def __init__(self, temp, csvfiles):
        """
        Initialize from script data.
        """
        self.temp = temp
        self.s1 = Script(csvfiles[0])
        self.s2 = Script(csvfiles[1])
        self.s3 = Script(csvfiles[2])


class Script:
    """
    Object to represent script data.
    """

    def __init__(self, csvfile):
        """
        Initialize with data from csv file.
        """
        df = pd.read_csv(csvfile)
        time = df['time'].values
        step = df[' step'].values
        current = df[' current'].values
        voltage = df[' voltage'].values
        chgAh = df[' chgAh'].values
        disAh = df[' disAh'].values

        self.time = time
        self.step = step
        self.current = current
        self.voltage = voltage
        self.chgAh = chgAh
        self.disAh = disAh


class ModelOcv:
    """
    Model representing OCV results.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, OCV0, OCVrel, SOC, OCV, SOC0, SOCrel, OCVeta, OCVQ):
        self.OCV0 = np.array(OCV0)
        self.OCVrel = np.array(OCVrel)
        self.SOC = np.array(SOC)
        self.OCV = np.array(OCV)
        self.SOC0 = np.array(SOC0)
        self.SOCrel = np.array(SOCrel)
        self.OCVeta = np.array(OCVeta)
        self.OCVQ = np.array(OCVQ)

    @classmethod
    def load(cls, pfile):
        """
        Load attributes from pickle file where pfile is string representing
        path to the pickle file.
        """
        ocv = json.load(open(pfile, 'rb'))
        return cls(ocv['OCV0'], ocv['OCVrel'], ocv['SOC'], ocv['OCV'], ocv['SOC0'], ocv['SOCrel'], ocv['OCVeta'], ocv['OCVQ'])


class ModelDyn:
    """
    Model representing results from the dynamic calculations.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, temps, etaParam, QParam, GParam, M0Param, MParam, R0Param, RCParam, RParam, SOC, OCV0, OCVrel, OCV, SOC0, SOCrel):
        self.temps = np.array(temps)
        self.etaParam = np.array(etaParam)
        self.QParam = np.array(QParam)
        self.GParam = np.array(GParam)
        self.M0Param = np.array(M0Param)
        self.MParam = np.array(MParam)
        self.R0Param = np.array(R0Param)
        self.RCParam = np.array(RCParam)
        self.RParam = np.array(RParam)
        self.SOC = np.array(SOC)
        self.OCV0 = np.array(OCV0)
        self.OCVrel = np.array(OCVrel)
        self.OCV = np.array(OCV)
        self.SOC0 = np.array(SOC0)
        self.SOCrel = np.array(SOCrel)

    @classmethod
    def load(cls, pfile):
        """
        Load attributes from json file where pfile is string representing
        path to the json file.
        """
        dyn = json.load(open(pfile, 'r'))
        return cls( dyn['temps'], dyn['etaParam'], dyn['QParam'], dyn['GParam'], dyn['M0Param'], dyn['MParam'], dyn['R0Param'], dyn['RCParam'], dyn['RParam'], dyn['SOC'], dyn['OCV0'], dyn['OCVrel'], dyn['OCV'], dyn['SOC0'], dyn['SOCrel'])

class spkfData:
    """
    Object to represent SPKF data.
    """

    def __init__(self, v0, T0, SigmaX0, SigmaV, SigmaW, model):
        """
        Initializes an "spkfData" structure, used by the sigma-point Kalman
        filter to store its own state and associated data.

        Inputs:
        v0: Initial cell voltage
        T0: Initial cell temperature
        SigmaX0: Initial state uncertainty covariance matrix
        SigmaV: Covariance of measurement noise
        SigmaW: Covariance of process noise
        model: ESC model of cell 

        Output:
        spkfData: Data structure used by SPKF code

        """
        # Initial state decsription 
        ir0 = 0 
        hk0 = 0
        irInd = 0
        hkInd = 1
        zkInd = 2
        SOC0 = SOCfromOCVtemp(v0, T0, model)
        xhat = np.block([ir0, hk0, SOC0]) # Initial state
        xhat = xhat[:, np.newaxis]

        # Covariance values
        Qbump = 5

        # SPKF specific parameters 
        Nx = len(xhat) # State vector length
        Ny = 1 # Measurement vector length
        Nu = 1 # Input-vector length
        Nw = np.array([SigmaW]).shape[0] # Process-noise vector length
        Nv = np.array([SigmaV]).shape[0] # Sensor-noise vector length
        Na = Nx + Nw + Nv # Augmented state vector length 

        h = np.sqrt(3) # SPKF/CDKF tuning factor
        Weight1 = (h*h - Na) / (h*h) # weighting factors when computing mean
        Weight2 = 1 / (2 * h * h) # and covariance
        Wm = np.vstack( ( np.array([Weight1]), Weight2 * np.ones( (2 * Na, 1) ) )  ) # Mean
        Wc =  Wm # Covar

        # Previous value of current
        priorI = 0
        signIk = 0

        self.irInd = irInd
        self.hkInd = hkInd 
        self.zkInd = zkInd
        self.xhat = xhat 
        self.SigmaX = SigmaX0 # Covariance value
        self.SigmaV = SigmaV # Covariance value
        self.SigmaW = SigmaW # Covariance value
        self.Snoise = np.real(cholesky(np.diag(np.block([SigmaW, SigmaV])), lower=True).T)
        self.Qbump = Qbump
        self.Nx = Nx
        self.Ny = Ny
        self.Nu = Nu
        self.Nw = Nw
        self.Nv = Nv
        self.Na = Na
        self.h = h
        self.Wm = Wm
        self.Wc = Wc
        self.priorI = priorI
        self.signIk = signIk
        self.model = model