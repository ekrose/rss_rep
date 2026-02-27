#teacher ustat
import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm

# For parallel processing
from multiprocessing import Pool
from functools import partial
import sys
from scipy.stats import norm, gamma
    
def varcovar(origX,origY,yearWeighted = False):
    '''
    U-stat estimator of variance / covariance for teacher effects
    X and Y are a J-by-max(T_j) matrix of teacher-specific mean residuals. 
    When X and Y are residuals for the same outcome and covariate group,
    code will return a estimate of variance of teacher effects. When X and Y
    differ (either in outcome or Xs), code returns an estimate of the 
    covariance. 

    Each row of X and Y are residuals for specific teacher, ordered as
    first year observed, second year observed, etc. Since teachers have
    different number of years observed, X and Y should be np.NaN for all
    years after the last year observed. Each teacher must have at least
    2 years observed.

    X and Y must have the same dimension.

    '''
    # Counds of valid obs
    countsX = np.count_nonzero(~np.isnan(origX),1)  # obs of X
    countsY = np.count_nonzero(~np.isnan(origY),1)  # obs of Y
    nsquares = np.count_nonzero(~np.isnan(origX*origY),1)   # Same year obs of both
    nproducts = (countsX*countsY - nsquares) # Note: nproducts > 0 for teachers with more than 1 non-missing observations i for both X and Y (YST)
    X = np.nan_to_num(origX[nproducts > 0, :].copy(), 0)
    Y = np.nan_to_num(origY[nproducts > 0, :].copy(), 0)

    if (len(X) == 0) | (len(Y) == 0):
        print('No observations in X or Y vectors')
        return np.nan

    # If year-weighted calculate weights
    if yearWeighted:
        weights = np.count_nonzero(X, axis=1)/np.count_nonzero(X) # X is observed only when Y is also observed, so a teacher that has t observations of X will also have the same number for Y

    # Calculate grand mean excluding squares (i.e, \bar{Y}_j, so averaging the "X/Y" matrices within each rwo along the columns --- YST)
    if yearWeighted:
        X_means = weights * np.nanmean(origX[nproducts > 0, :],1)
        Y_means = weights * np.nanmean(origY[nproducts > 0, :],1)
        tmp = X_means.reshape(-1,1).dot(Y_means.reshape(1,-1))
        np.fill_diagonal(tmp,0)
        gmean = np.sum(tmp)
    else:
        X_means = np.nanmean(origX[nproducts > 0, :],1)
        Y_means = np.nanmean(origY[nproducts > 0, :],1)
        tmp = X_means.reshape(-1,1).dot(Y_means.reshape(1,-1))
        np.fill_diagonal(tmp,0)
        gmean = np.sum(tmp) / (len(X_means)*len(Y_means))

    # Sparsify matricies
    # Diagonalizied matrix for taking products
    diagX = sparse.csr_matrix(
                (X.ravel(),
                (np.arange(X.shape[0]*X.shape[1]),
                np.repeat(np.arange(X.shape[0]),X.shape[1]))
                ))

    # Take product
    k = diagX.dot(sparse.csr_matrix(Y))

    # Take sum within teacher
    teach_sum = sparse.csr_matrix(
            (np.ones(X.shape[0]*X.shape[1]),
                (np.arange(X.shape[0]*X.shape[1]),
                np.repeat(np.arange(X.shape[0]),X.shape[1]))
                ))
    sums = teach_sum.T.dot(np.sum(k,1))
    
    '''
    This contains all the within teacher products, including double counting
    so need remove squares and divide by two
    '''
    sums = (sums - np.sum(X*Y,1)[:,np.newaxis])

    # Then we average of number of products per teacher
    sums = sums.ravel()/nproducts[nproducts > 0]

    if yearWeighted:
        _weights = weights
        _weights = (1-weights)*weights # (1-w_j) * w_j
        sums = np.multiply(_weights.reshape(1,-1), sums)
        Ustat = np.sum(sums) - gmean
    else:
        Ustat = (len(X_means)-1)/len(X_means)*np.mean(sums) - gmean

    return Ustat

def varcovar_gaps(origX,origY,years,mini=None,maxi=None):
    '''
    U-stat only use residual pairs separated by at least min 
    and nomore than max years, botth possibly Nan

    '''
    ### fix nans
    if mini is None:
        mini = 1
    if maxi is None:
        maxi = np.inf

    ### Collect all residual crosspoducts
    # Loop over columns of origX and origY
    row_products = np.zeros(origX.shape[0])
    counter = np.zeros(origX.shape[0])
    for k in range(origX.shape[1]):
        for j in range(origY.shape[1]):
            if k != j:
                gaps = np.abs(years[:,k] - years[:,j])
                toadd = origX[:,k]*origY[:,j]*(
                                (gaps >= mini) & (gaps <= maxi))
                row_products += np.nan_to_num(toadd,0)
                counter += (~np.isnan(origX[:,k]*origY[:,j]))*(
                                (gaps >= mini) & (gaps <= maxi))
    row_products = row_products / counter
    row_mean = np.mean(row_products[counter > 0])

    # Compute grand mean excluding squares
    X_means = np.nanmean(origX[counter > 0, :],1)
    Y_means = np.nanmean(origY[counter > 0, :],1)
    tmp = X_means.reshape(-1,1).dot(Y_means.reshape(1,-1))
    np.fill_diagonal(tmp,0)
    gmean = np.sum(tmp) / (len(X_means)*len(Y_means))

    return (len(X_means)-1)/len(X_means)*row_mean - gmean


def testfuncs(func_varcovar):
    ''' 
    Test 1: This should be zero, since all the draws in iid
    '''
    X = np.random.normal(size=(10000,3)) 
    Y = X
    print("Expecting 0, got: {:4.3f}".format(func_varcovar(X,Y)))

    ''' 
    Test 2: Should return a covariance of 0.5
    '''
    X = np.random.multivariate_normal(mean=np.zeros(2), cov=np.array([[1,0.5],[0.5,1]]), size=(10000)) 
    Y = X
    print("Expecting 0.5, got: {:4.3f}".format(func_varcovar(X,Y)))

    ''' 
    Test 3: Should return a third of the 0.5
    '''
    X = np.random.multivariate_normal(mean=np.zeros(3), cov=np.array([[1,0.5,0],[0.5,1,0],[0,0,1]]), size=(10000)) 
    Y = X
    print("Expecting 0.167, got: {:4.3f}".format(func_varcovar(X,Y)))

    ''' 
    Test 4: Should return 0.5
    '''
    X = np.random.multivariate_normal(mean=3*np.ones(3), cov=np.array([[1,0.5,0.5],[0.5,1,0.5],[0.5,0.5,1]]), size=(10000)) 
    X[np.random.binomial(p=0.5,n=1,size=(X.shape[0],X.shape[1]))==1] = np.NaN
    Y = X
    print("Expecting 0.5, got: {:4.3f}".format(func_varcovar(X,Y)))
    return func_varcovar(X,Y)
    # print(bsci_varcovar(X,Y))


# Lambdas
def sd_func(sX, yearWeighted=False): 
    return np.power(varcovar(sX, sX, yearWeighted=yearWeighted), 0.5) 

def correl_func(sX,sY,yearWeighted = False): 
    return varcovar(sX,sY, yearWeighted = yearWeighted)/np.power(varcovar(sX,sX, yearWeighted = yearWeighted)*varcovar(sY,sY, yearWeighted = yearWeighted), 0.5) 

def correl_func_gaps(sX,sY,years,mini=None,maxi=None): 
    return varcovar_gaps(sX,sY,years,mini,maxi)/np.power(varcovar_gaps(sX,sX,years,mini,maxi)*varcovar_gaps(sY,sY,years,mini,maxi), 0.5) 

def sd_effect_func(sX,sY,yearWeighted = False): 
    return varcovar(sX, sY, yearWeighted = yearWeighted)/np.power(varcovar(sX, sX, yearWeighted = yearWeighted), 0.5)

def reg_coef_func(sX,sY,yearWeighted = False): 
    return varcovar(sX, sY, yearWeighted = yearWeighted)/varcovar(sX, sX, yearWeighted = yearWeighted)


### Standard errors
# Helper function for sampling covariances
def sampc(X,Y):
    Xmeans = np.nanmean(X, axis=1)
    Ymeans = np.nanmean(Y, axis=1)
    XYcounts = np.array(np.sum(~np.isnan(X) & ~np.isnan(Y), axis=1),dtype=float)
    XYcovar = np.nansum((X-Xmeans[:,np.newaxis])*(Y-Ymeans[:,np.newaxis]),1,dtype=float)/(XYcounts-1)
    XYcovar[XYcounts <= 1] = 0  # No sampling covariance if no overlap
    return XYcovar

# Helper for C functions
def makec(X,Y):
    Xcounts = np.array(np.sum(~np.isnan(X), axis=1),dtype=float)
    Ycounts = np.array(np.sum(~np.isnan(Y), axis=1),dtype=float)
    XYcounts = np.array(np.sum(~np.isnan(X) & ~np.isnan(Y), axis=1),dtype=float)
    J = sum(Xcounts*Ycounts - XYcounts > 0)

    denom = Xcounts*Ycounts - XYcounts
    C_jj = (J-1)/J**2/denom
    C_jj[denom == 0] = 0 # does not contribute to estimate
    C_jk = -1/J**2*(1/Xcounts).reshape(-1,1).dot((1/Ycounts).reshape(1,-1))    # J-by-J, with C_jk as each element
    C_jk[Xcounts == 0,:] = 0 # does not contribute to estimate
    C_jk[:,Ycounts == 0] = 0 # does not contribute to estimate

    return C_jj, C_jk


def makec_spec(X ):
    # Number of observations in X, Y, and intersection
    Xcounts = np.array(np.sum(~np.isnan(X), axis=1),dtype=float) # returns no. of observations across all teachers in X (e.g. event X)
    J = sum(Xcounts*Xcounts - Xcounts > 0) 
    
    # Compute C coefficients    
    denom = Xcounts*Xcounts - Xcounts
    C_jj = (J-1)/J**2/denom
    C_jj[denom == 0] = 0
    C_jk = -1/J**2*(1/Xcounts).reshape(-1,1).dot((1/Xcounts).reshape(1,-1))    # J-by-J, with C_jk as each element.
    
    # Set those with no observations to 0
    C_jk[Xcounts == 0,:] = 0 
    C_jk[:,Xcounts == 0] = 0
    
    return C_jj, C_jk

# Help for bias corrected sum squares
def lamb_sum(X,C_jjX,C_jkX,Y,C_jjY,C_jkY):
    '''
    bias corrected product of (sum_k!=i C_ij^X a^X) (sum_k!=i C_ij^Y a^Y)
    '''
    Xmeans = np.nanmean(X, axis=1)
    Ymeans = np.nanmean(Y, axis=1)    
    Xcounts = np.array(np.sum(~np.isnan(X), axis=1),dtype=float)
    Ycounts = np.array(np.sum(~np.isnan(Y), axis=1),dtype=float)
    Xmeans[Xcounts < 2] = 0
    Ymeans[Ycounts < 2] = 0

    XYcounts = np.array(np.sum(~np.isnan(X) & ~np.isnan(Y), axis=1),dtype=float)
    XYcovar = np.nansum((X-Xmeans[:,np.newaxis])*(Y-Ymeans[:,np.newaxis]),1,dtype=float)/(XYcounts-1)
    XYcovar[XYcounts <= 1] = 0  # No sampling covariance if no overlap

    tmpX = C_jkX*Xmeans[np.newaxis,:]*Xcounts[np.newaxis,:]    
    tmpY = C_jkY*Ymeans[np.newaxis,:]*Ycounts[np.newaxis,:]    
    tmpBXY = C_jkX*C_jkY*XYcovar[np.newaxis,:]*XYcounts[np.newaxis,:]    
    tmpc = (XYcounts - 1)**2/XYcounts
    tmpc[XYcounts == 0] = 0
    return (
                (C_jjX*Xmeans*(Xcounts - 1) + 
                    np.sum(tmpX,1) - np.diag(tmpX))*
                (C_jjY*Ymeans*(Ycounts - 1) + 
                    np.sum(tmpY,1) - np.diag(tmpY))

                - (C_jjX*C_jjY*XYcovar*tmpc + ( # Bias correction
                            np.sum(tmpBXY,1) - np.diag(tmpBXY)))
            )

# Estimate the sampling covariance between two sampling estimates
def ustat_samp_covar(Atmp,Btmp,Ctmp,Dtmp):
    '''
    Estimate the sampling covariance between the estimate of 
    Cov(Atmp,Btmp) and the estimates of Cov(Ctmp,Dtmp).

    By setting Atmp=Btmp=Ctmp=Dtmp, for example, one will simply 
    get the sampling variance of a variance estimate. 
    '''
    # Make copies
    A = Atmp.copy()
    B = Btmp.copy()
    C = Ctmp.copy()
    D = Dtmp.copy()

    # Compute sampling covariances
    sigAC = sampc(A,C) # These are standard formulae; i.e., sampc(X, Y) = (X - Xmean) * (Y - Ymean) / (|XY| - 1)
    sigAD = sampc(A,D)
    sigBC = sampc(B,C)
    sigBD = sampc(B,D)

    # Counts
    # Count the cardinality of the intersections between matrices A and C, and so on.
    countsAC = np.array(np.sum(~np.isnan(A) & ~np.isnan(C), axis=1),dtype=float) 
    countsAD = np.array(np.sum(~np.isnan(A) & ~np.isnan(D), axis=1),dtype=float)
    countsBC = np.array(np.sum(~np.isnan(B) & ~np.isnan(C), axis=1),dtype=float)
    countsBD = np.array(np.sum(~np.isnan(B) & ~np.isnan(D), axis=1),dtype=float)
    countsABCD = np.array(np.sum(~np.isnan(A) & ~np.isnan(C)
                    & ~np.isnan(B) & ~np.isnan(D), axis=1),dtype=float)

    # Compute Ciks
    C_jjAB, C_jkAB = makec(A,B)
    C_jjCD, C_jkCD = makec(C,D)
    C_jjBA, C_jkBA = makec(B,A) # Add reverse
    C_jjDC, C_jkDC = makec(D,C)

    # Compute bias corrected products of sums
    prodABBCDD = lamb_sum(B, C_jjAB, C_jkAB, D, C_jjCD, C_jkCD)
    prodABBDCC = lamb_sum(B, C_jjAB, C_jkAB, C, C_jjDC, C_jkDC)
    prodBAACDD = lamb_sum(A, C_jjBA, C_jkBA, D, C_jjCD, C_jkCD)
    prodBAADCC = lamb_sum(A, C_jjBA, C_jkBA, C, C_jjDC, C_jkDC)
    
    # Variance calulation
    vsum = (
            countsAC*sigAC*prodABBCDD +
            countsAD*sigAD*prodABBDCC +
            countsBC*sigBC*prodBAACDD +
            countsBD*sigBD*prodBAADCC 
            )

    # Add last piece
    tmpC = C_jkAB * C_jkDC * sigBC[np.newaxis,:] * countsBC[np.newaxis,:]
    tmpC = (np.sum(tmpC, 1) - np.diag(tmpC))
    vsum += sigAD*countsAD*tmpC
    vsum += sigAD*C_jjAB*C_jjDC*sigBC*(countsAD*countsBC - countsABCD) 

    tmpC = C_jkAB * C_jkCD * sigBD[np.newaxis,:] * countsBD[np.newaxis,:]
    tmpC = (np.sum(tmpC, 1) - np.diag(tmpC))
    vsum += sigAC*countsAC*tmpC
    vsum += sigAC*C_jjAB*C_jjCD*sigBD*(countsBD*countsAC - countsABCD) 

    return np.sum(vsum)

# Return sampling variance of a variance estimate
def vcv_samp_var(Atmp):
    '''
    Estimate the sampling variance of Var(A).
    Special case to increase computation speed.
    '''
    # Make copies
    A = Atmp.copy()

    # Compute sampling covariances
    sigAA = sampc(A,A) # These are standard formulae; i.e., sampc(X, Y) = (X - Xmean) * (Y - Ymean) / (|XY| - 1)

    # Counts
    # Count the cardinality of the intersections between matrices A and C, and so on.
    countsA = np.array(np.sum(~np.isnan(A), axis=1),dtype=float) 

    # Compute Ciks
    C_jjAA, C_jkAA = makec(A,A)

    # Compute bias corrected products of sums
    prodA = lamb_sum(A, C_jjAA, C_jkAA, A, C_jjAA, C_jkAA)
    
    # Variance calulation
    vsum = (
            4*countsA*sigAA*prodA
            )

    # Add last piece
    tmpC = C_jkAA**2 * sigAA[np.newaxis,:] * countsA[np.newaxis,:]
    tmpC = (np.sum(tmpC, 1) - np.diag(tmpC))
    vsum += 2*sigAA*countsA*tmpC
    vsum += 2*sigAA*C_jjAA*C_jjAA*sigAA*(countsA*countsA - countsA) 

    return np.sum(vsum)

# Standard error for sqrt(Var(X))
def sd_samp_var(Xtmp):
    '''
    Standard error for sqrt(Var(X)) via delta method
    var(x^1/2) = Var(x) / (4E[x])
    '''
    # Get inputs
    var = varcovar(Xtmp,Xtmp)
    samp_var = vcv_samp_var(Xtmp)

    # Get sampling vcv matrix
    return samp_var/(4*var)

# Return sampling variance of a covariance estimate
def vcv_samp_covar(Atmp, Ctmp):
    '''
    Estimate the sampling variance of Cov(A,C).
    Special case to increase computation speed.
    '''
    # Make copies
    A = Atmp.copy()
    C = Ctmp.copy()

    # Compute sampling covariances
    sigAC = sampc(A,C) # These are standard formulae; i.e., sampc(X, Y) = (X - Xmean) * (Y - Ymean) / (|XY| - 1)
    sigA = sampc(A,A)
    sigC = sampc(C,C)
    
    # Counts
    # Count the cardinality of the intersections between matrices A and C, and so on.
    countsAC = np.array(np.sum(~np.isnan(A) & ~np.isnan(C), axis=1),dtype=float)
    countsA = np.array(np.sum(~np.isnan(A), axis=1),dtype=float)
    countsC = np.array(np.sum(~np.isnan(C), axis=1),dtype=float)
    
    # Compute Ciks
    C_jjAC, C_jkAC = makec(A,C)
    C_jjCA, C_jkCA = makec(C,A)

    # Compute bias corrected products of sums
    prodACC = lamb_sum(C, C_jjAC, C_jkAC, C, C_jjAC, C_jkAC)
    prodCAA = lamb_sum(A, C_jjCA, C_jkCA, A, C_jjCA, C_jkCA)
    prodACCCAA = lamb_sum(C, C_jjAC, C_jkAC, A, C_jjCA, C_jkCA)
    
    # Variance calulation
    vsum = (countsA * sigA * prodACC +
            countsC * sigC * prodCAA + 
            2*countsAC*sigAC*prodACCCAA)

    # Add last piece
    tmpC = C_jkAC * C_jkCA * sigAC[np.newaxis,:] * countsAC[np.newaxis,:]
    tmpC = (np.sum(tmpC, 1) - np.diag(tmpC))
    vsum += sigAC*countsAC*tmpC
    vsum += sigAC*C_jjAC*C_jjCA*sigAC*(countsAC*countsAC - countsAC) 

    tmpC = C_jkAC * C_jkAC * sigC[np.newaxis,:] * countsC[np.newaxis,:]
    tmpC = (np.sum(tmpC, 1) - np.diag(tmpC))
    vsum += sigA*countsA*tmpC
    vsum += sigA*C_jjAC*C_jjAC*sigC*(countsA*countsC - countsAC) 

    return np.sum(vsum)

def vcv_samp_covar_AAAD(Atmp, Dtmp):
    '''
    Estimates the sampling covariance of Var(A) and Cov(A,D).
    '''
    # Make copies
    A = Atmp.copy()
    D = Dtmp.copy()

    # Compute sampling covariances
    sigAA = sampc(A,A) # These are standard formulae; i.e., sampc(X, Y) = (X - Xmean) * (Y - Ymean) / (|XY| - 1)
    sigAD = sampc(A,D)

    # Counts
    # Count the cardinality of the intersections between matrices A and C, and so on.
    countsAA = np.array(np.sum(~np.isnan(A), axis = 1), dtype = float)
    countsAD = np.array(np.sum(~np.isnan(A) & ~np.isnan(D), axis=1),dtype=float)


    # Compute Ciks.
    C_jjAA, C_jkAA = makec_spec(A)
    C_jjAD, C_jkAD = makec(A,D)
    C_jjDA, C_jkDA = makec(D,A)
    C_jjDD, C_jkDD = makec_spec(D)

    # Compute bias corrected products of sums
    prodAAAADD = lamb_sum(A, C_jjAA, C_jkAA, D, C_jjAD, C_jkAD)
    prodAAADAA = lamb_sum(A, C_jjAA, C_jkAA, A, C_jjDA, C_jkDA)
    
    # Variance calulation
    vsum = (
            2*countsAA*sigAA*prodAAAADD +
            2*countsAD*sigAD*prodAAADAA
            )

    # Add last piece
    tmpC = C_jkAA * C_jkDA * sigAA[np.newaxis,:] * countsAA[np.newaxis,:]
    tmpC = (np.sum(tmpC, 1) - np.diag(tmpC))
    vsum += sigAD*countsAD*tmpC
    vsum += sigAD*C_jjAA*C_jjDA*sigAA*(countsAD*countsAA - countsAD) 

    tmpC = C_jkAA * C_jkAD * sigAD[np.newaxis,:] * countsAD[np.newaxis,:]
    tmpC = (np.sum(tmpC, 1) - np.diag(tmpC))
    vsum += sigAA*countsAA*tmpC
    vsum += sigAA*C_jjAA*C_jjAD*sigAD*(countsAD*countsAA - countsAD) 

    return np.sum(vsum)


# Estimate the sampling covariance between two sampling estimates
def vcv_samp_covar_AADD(Atmp,Dtmp):
    '''
    Estimates the sampling covariance of Var(A) and Var(D).
    '''
    # Make copies
    A = Atmp.copy()
    D = Dtmp.copy()

    # Compute sampling covariances
    sigAD = sampc(A,D) # These are standard formulae; i.e., sampc(X, Y) = (X - Xmean) * (Y - Ymean) / (|XY| - 1)

    # Counts
    # Count the cardinality of the intersections between matrices A and C, and so on.
    countsAD = np.array(np.sum(~np.isnan(A) & ~np.isnan(D), axis=1),dtype=float) 

    # Compute Ciks.
    C_jjAA, C_jkAA = makec_spec(A)
    C_jjDD, C_jkDD = makec_spec(D)
    
    # Compute bias corrected products of sums
    prodAAADDD = lamb_sum(A, C_jjAA, C_jkAA, D, C_jjDD, C_jkDD)
    
    # Variance calulation
    vsum = (
            4 * countsAD * sigAD * prodAAADDD
            )

    # Add last piece
    tmpC = C_jkAA * C_jkDD * sigAD[np.newaxis,:] * countsAD[np.newaxis,:]
    tmpC = (np.sum(tmpC, 1) - np.diag(tmpC))
    vsum += 2 * (sigAD * countsAD * tmpC)
    vsum += 2 * (sigAD*C_jjAA*C_jjDD*sigAD*(countsAD*countsAD - countsAD)) 


    return np.sum(vsum)

# Sampling variation of correlation estimate
def corr_samp_covar(Xtmp,Ytmp):
    '''
    Standard error for rho = Covar(X,Y)/sqrt(Var(X)*Var(Y))
    delta method given gradient of rho = F([C(X,Y),V(X),V(y)])
    '''
    # Get covars and vars first
    covar = varcovar(Xtmp,Ytmp)
    varX = varcovar(Xtmp,Xtmp)
    varY = varcovar(Ytmp,Ytmp)

    # Get sampling vcv matrix
    vcv = np.zeros(shape=(3,3))
    vcv[0,0] = vcv_samp_covar(Xtmp,Ytmp)    # covar x y
    vcv[1,1] = vcv_samp_var(Xtmp)           # var x
    vcv[2,2] = vcv_samp_var(Ytmp)           # var y

    vcv[0,1] = vcv_samp_covar_AAAD(Xtmp,Ytmp)
            #ustat_samp_covar(Xtmp,Ytmp,Xtmp,Xtmp)  # between covar and var X
    vcv[1,0] = vcv[0,1]

    vcv[0,2] = vcv_samp_covar_AAAD(Ytmp,Xtmp)
        #ustat_samp_covar(Xtmp,Ytmp,Ytmp,Ytmp)  # between covar and var Y
    vcv[2,0] = vcv[0,2]

    vcv[1,2] = vcv_samp_covar_AADD(Xtmp,Ytmp)
        #ustat_samp_covar(Xtmp,Xtmp,Ytmp,Ytmp) # between var x and var Y
    vcv[2,1] = vcv[1,2]

    # Get Gradient
    grad = np.zeros(shape=(3,1))
    grad[0,0] = 1/(varX*varY)**0.5
    grad[1,0] = covar/(varX*varY)**1.5 * -0.5 * varY
    grad[2,0] = covar/(varX*varY)**1.5 * -0.5 * varX

    # Return sampling variance
    return grad.T.dot(vcv).dot(grad)[0,0]

def sd_effect_samp_covar(Xtmp,Ytmp):
    '''
    Standard error for beta = Covar(X,Y)/Var(X)^0.5
    delta method given gradient of beta = F([C(X,Y),V(X)])
    '''
    # Get covars and vars first
    covar = varcovar(Xtmp,Ytmp)
    varX = varcovar(Xtmp,Xtmp)

    # Get sampling vcv matrix
    vcv = np.zeros(shape=(2,2))
    vcv[0,0] = vcv_samp_covar(Xtmp,Ytmp)   # covar x y
    vcv[1,1] = vcv_samp_var(Xtmp)          # var x

    vcv[0,1] = vcv_samp_covar_AAAD(Xtmp,Ytmp)
        #ustat_samp_covar(Xtmp,Ytmp,Xtmp,Xtmp)   # between covar and var X
    vcv[1,0] = vcv[0,1]

    # Get Gradient
    grad = np.zeros(shape=(2,1))
    grad[0,0] = 1/(varX)**0.5
    grad[1,0] = covar/(varX)**1.5 * -0.5

    # Return sampling variance
    return grad.T.dot(vcv).dot(grad)[0,0]
    


