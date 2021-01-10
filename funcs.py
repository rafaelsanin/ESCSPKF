"""
Functions used by the dyn_model
"""

# Modules
# ------------------------------------------------------------------------------

import ipdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky

# Functions
# ------------------------------------------------------------------------------

def OCVfromSOCtemp(soc, temp, model):
    """ OCV function """
    SOC = model.SOC          # force to be column vector
    OCV0 = model.OCV0        # force to be column vector
    OCVrel = model.OCVrel    # force to be column vector

    # if soc is scalar then make it a vector
    soccol = np.asarray(soc)
    if soccol.ndim == 0:
        soccol = soccol[None]

    tempcol = temp * np.ones(np.size(soccol))

    diffSOC = SOC[1] - SOC[0]           # spacing between SOC points - assume uniform
    ocv = np.zeros(np.size(soccol))     # initialize output to zero
    I1, = np.where(soccol <= SOC[0])    # indices of socs below model-stored data
    I2, = np.where(soccol >= SOC[-1])   # and of socs above model-stored data
    I3, = np.where((soccol > SOC[0]) & (soccol < SOC[-1]))   # the rest of them
    I6 = np.isnan(soccol)               # if input is "not a number" for any locations

    # for voltages less than lowest stored soc datapoint, extrapolate off
    # low end of table
    if I1.any():
        dv = (OCV0[1] + tempcol*OCVrel[1]) - (OCV0[0] + tempcol*OCVrel[0])
        ocv[I1] = (soccol[I1] - SOC[0])*dv[I1]/diffSOC + OCV0[0] + tempcol[I1]*OCVrel[0]

    # for voltages greater than highest stored soc datapoint, extrapolate off
    # high end of table
    if I2.any():
        dv = (OCV0[-1] + tempcol*OCVrel[-1]) - (OCV0[-2] + tempcol*OCVrel[-2])
        ocv[I2] = (soccol[I2] - SOC[-1])*dv[I2]/diffSOC + OCV0[-1] + tempcol[I2]*OCVrel[-1]

    # for normal soc range, manually interpolate (10x faster than "interp1")
    I4 = (soccol[I3] - SOC[0])/diffSOC  # using linear interpolation
    I5 = np.floor(I4)
    I5 = I5.astype(int)
    I45 = I4 - I5
    omI45 = 1 - I45
    ocv[I3] = OCV0[I5]*omI45 + OCV0[I5+1]*I45
    ocv[I3] = ocv[I3] + tempcol[I3]*(OCVrel[I5]*omI45 + OCVrel[I5+1]*I45)
    ocv[I6] = 0     # replace NaN SOCs with zero voltage
    return ocv

def SOCfromOCVtemp(ocv,temp,model):
    """
    function soc = SOCfromOCVtemp(ocv,temp,model)

    Computes an approximate state of charge from the fully rested open-
    circuit voltage of a cell at a given temperature point. This is NOT an
    exact inverse of the OCVfromSOCtemp function due to how the computations
    are done, but it is "fairly close."

    Inputs: ocv = scalar or matrix of cell open circuit voltages
        temp = scalar or matrix of temperatures at which to calc. OCV
        model = data structure produced by processDynamic
    Output: soc = scalar or matrix of states of charge -- one for every
                soc and temperature input point
    """

    OCV = model.OCV # force to be column vector
    SOC0 = model.SOC0 # force to be column vector
    SOCrel = model.SOCrel # force to be column vector

    # if ocv is scalar then make it a vector
    ocvcol = np.asarray(ocv)
    if ocvcol.ndim == 0:
        ocvcol = ocvcol[None]
    
    tempcol = temp * np.ones(np.size(ocvcol))

    diffOCV = OCV[1] - OCV[0]           # spacing between SOC points - assume uniform
    soc = np.zeros(np.size(ocvcol))     # initialize output to zero
    I1, = np.where(ocvcol <= OCV[0])    # indices of socs below model-stored data
    I2, = np.where(ocvcol >= OCV[-1])   # and of socs above model-stored data
    I3, = np.where((ocvcol > OCV[0]) & (ocvcol < OCV[-1]))   # the rest of them
    I6 = np.isnan(ocvcol)               # if input is "not a number" for any locations

    # for ocvs lower than lowest voltage, extrapolate off low end of table
    if I1.any():
        dz = (SOC0[1] + tempcol*SOCrel[1]) - (SOC0[0] + tempcol*SOCrel[0])
        soc[I1] = (ocvcol[I1] - OCV[0])*dz[I1]/diffOCV + SOC0[0] + tempcol[I1]*SOCrel[0]

    # for ocvs higher than highest voltage, extrapolate off high end of table
    if I2.any():
        dz = (SOC0[-1] + tempcol*SOCrel[-1]) - (SOC0[-2] + tempcol*SOCrel[-2])
        soc[I2] = (ocvcol[I2] - OCV[-1])*dz[I2]/diffOCV + SOC0[-1] + tempcol[I2]*SOCrel[-1]

    # for normal ocv range, manually interpolate (10x faster than "interp1")
    I4 = (ocvcol[I3] - OCV[0])/diffOCV  # using linear interpolation
    I5 = np.floor(I4)
    I5 = I5.astype(int)
    I45 = I4 - I5
    omI45 = 1 - I45
    soc[I3] = SOC0[I5]*omI45 + SOC0[I5+1]*I45
    soc[I3] = soc[I3] + tempcol[I3]*(SOCrel[I5]*omI45 + SOCrel[I5+1]*I45)
    soc[I6] = 0     # replace NaN OCVs with zero voltage
    return soc

def stateEqn(xold, current, xnoise, deltat):
    """
    Calculate new states for all of the old state vectors in xold.  
    """
    current = current + xnoise # noise adds to current
    xnew = 0 * xold
    xnew[irInd, :] = RC * xold[irInd, :] + (1-RC) * current
    Ah = np.exp(-abs(current * G * deltat / (3600 * Q))) # Hysteresis factor
    xnew[hkInd, :] = Ah * xold[hkInd, :] + (Ah - 1) * np.sign(current)
    xnew[zkInd, :] = xold[zkInd, :] - current * deltat / (3600 * Q) 
    return xnew

def outputEqn(xhat, current, ynoise, temp, model):
    """
    Calculate cell output voltage for all of state vectors in xhat
    """
    yhat = OCVfromSOCtemp(xhat[zkInd,:], temp, model)
    yhat = yhat + M * xhat[hkInd, :] + M0 * signIk
    yhat = yhat - R * xhat[irInd, :] - R0 * current + ynoise[0, :]
    
    return yhat

def iterSPKF(vk,ik,Tk,deltat,spkfData):
    """
    function [zk,zkbnd,spkfData] = iterSPKF(vk,ik,Tk,deltat,spkfData)

    Performs one iteration of the sigma-point Kalman filter using the new
    measured data.

    Inputs:
    vk: Present measured (noisy) cell voltage
    ik: Present measured (noisy) cell current
    Tk: Present temperature
    deltat: Sampling interval
    spkfData: Data structure initialized by initSPKF, updated by iterSPKF

    Output:
    zk: SOC estimate for this time sample
    zkbnd: 3-sigma estimation bounds
    spkfData: Data structure used to store persistent variables
    """

    model = spkfData.model
    # Load the cell model parameters
    global Q
    global G
    global RC 
    global M
    global M0
    global R
    global R0
    Q = model.QParam[Tk]
    G = model.GParam[Tk]
    M = model.MParam[Tk]
    M0 = model.M0Param[Tk]
    RC = np.exp(-deltat / abs(model.RCParam[Tk])).T
    R = model.RParam[Tk]
    R0 = model.R0Param[Tk]
    eta = model.etaParam[Tk]

    if ik < 0:
        ik = ik * eta 

    # Get data stored in spkfData structure
    global irInd
    global hkInd
    global zkInd
    I =spkfData.priorI
    SigmaX = spkfData.SigmaX
    xhat = spkfData.xhat
    Nx = spkfData.Nx
    Nw = spkfData.Nw
    Nv = spkfData.Nv
    Na = spkfData.Na
    Snoise = spkfData.Snoise 
    Wc = spkfData.Wc
    irInd = spkfData.irInd
    hkInd = spkfData.hkInd
    zkInd = spkfData.zkInd

    global signIk
    if abs(ik) > Q/100:
        spkfData.signIk = np.sign(ik)

    signIk = spkfData.signIk

    #   Step 1a: State estimate time update
    #            - Create xhatminus augmented SigmaX points
    #            - Extract xhatminus state SigmaX points
    #            - Compute weighted average xhatminus(k)

    #   Step 1a-1: Create augmented SigmaX and xhat
 
    try:
        sigmaXa = cholesky(SigmaX, lower=True)
    except np.linalg.LinAlgError:
        print('Cholesky error. Recovering ...')
        theAbsDiag = abs(np.diag(SigmaX))
        sigmaXa = np.diag(np.maximum(np.sqrt(np.maximum(0,theAbsDiag)), np.sqrt(np.maximum(0,np.asarray(spkfData.SigmaW)))))

    sigmaXa = np.block([[np.real(sigmaXa), np.zeros((Nx, Nw+Nv))], [np.zeros((Nw+Nv, Nx)), Snoise]])
    xhata = np.block([[xhat], [np.zeros(([Nw+Nv, 1]))]])
    # Note: sigmaXa is lower-triangular
    # Step 1a-2: Calculate SigmaX points
    Xa = xhata * np.ones((1, 2* Na+1)) + spkfData.h * np.block([np.zeros((Na, 1)), sigmaXa, -sigmaXa])
    
    #   Step 1a-3: Time update from last iteration until now
    #       stateEqn(xold,current,xnoise)
    Xx = stateEqn(Xa[0:Nx,:], I, Xa[Nx:Nx+Nw,:], deltat)
    xhat = Xx @ spkfData.Wm
    xhat[hkInd] = np.minimum(1, np.maximum(-1, xhat[hkInd]))
    xhat[zkInd] = np.minimum(1.05, np.maximum(-0.05, xhat[zkInd]))

    #   Step 1b: Error covariance time update
    #        - Compute weighted covariance sigmaminus(k)
    #          (strange indexing of xhat to avoid "repmat" call)
    Xs = Xx - xhat * np.ones((1, 2*Na+1))
    SigmaX = Xs @ np.diagflat(Wc) @ Xs.T

    #   Step 1c: Output estimate
    #        - Compute weighted output estimate yhat(k)
    I = ik
    yk = vk
    Y = outputEqn(Xx, I + Xa[Nx:Nx+Nw, :], Xa[Nx+Nw:, :], Tk, model)
    yhat = Y @ spkfData.Wm

    # Step 2a: Estimator gain matrix
    Ys = Y - yhat * np.ones((1, 2*Na+1))
    SigmaXY = Xs @ np.diagflat(Wc) @ Ys.T
    SigmaY = Ys @ np.diagflat(Wc) @ Ys.T
    L = SigmaXY / SigmaY 

    # Step 2b: State estimate measurement update 
    r = yk - yhat # residual. Use the check for sensor errors
    if r**2 > 100 * SigmaY:
        L[:,0] = 0.0

    xhat = xhat + L @ r
    xhat[zkInd] = np.minimum(1.05, np.maximum(-0.05, xhat[zkInd]))

    # Step 2c: Error covariance measurement update
    SigmaX = SigmaX - L @ SigmaY @ L.T
    _,S,V = np.linalg.svd(SigmaX)
    S = np.diagflat(S)
    HH = V @ S @ V.T
    SigmaX = (SigmaX + SigmaX.T + HH + HH.T) / 4 # Help maintain robustness

    # Q-bump code
    if r**2 > 4 * SigmaY: # Bad voltage estimate by (2-sigmaY), bump Q
        print('Bumping sigma')
        SigmaX[zkInd, zkInd] = SigmaX[zkInd, zkInd] * spkfData.Qbump
    
    # Save data in spkfData structure for next time ...
    spkfData.priorI = ik
    spkfData.SigmaX = SigmaX
    spkfData.xhat = xhat 
    zk = xhat[zkInd]
    zkbnd = 3 * np.sqrt(SigmaX[zkInd, zkInd])

    return zk, zkbnd, spkfData