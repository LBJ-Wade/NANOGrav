from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy.linalg as sl

from enterprise_extensions import models

class FpStat(object):
    """
    Class for the Fp-statistic.
    :param psrs: List of `enterprise` Pulsar instances.
    :param psrTerm: Include the pulsar term in the CW signal model. Default=True
    :param bayesephem: Include BayesEphem model. Default=True
    """
    
    def __init__(self, psrs, pta, params):
        
        # initialize standard model with fixed white noise and powerlaw red noise
        print('Initializing the model...')
        self.pta = pta
            
        self.psrs = psrs
        self.params = params
    
    def compute_Fp(self, fgw):
        """
        Computes the Fp-statistic.
        :param fgw: GW frequency
        :returns:
        fstat: value of the Fp-statistic at the given frequency
        """
        
        phiinvs = self.pta.get_phiinv(self.params, logdet=False)
        TNTs = self.pta.get_TNT(self.params)
        Ts = self.pta.get_basis()
        Nvecs = self.pta.get_ndiag()
        
        N = np.zeros(2)
        M = np.zeros((2,2))
        fstat = 0
        
        for psr, TNT, phiinv, T, Nvec in zip(self.psrs,TNTs, phiinvs, Ts, Nvecs):
            Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
            
            ntoa = len(psr.toas)
            
            A = np.zeros((2, ntoa))
            A[0, :] = 1 / fgw ** (1 / 3) * np.sin(2 * np.pi * fgw * psr.toas)
            A[1, :] = 1 / fgw ** (1 / 3) * np.cos(2 * np.pi * fgw * psr.toas)
            
            ip1 = self.innerProduct_rr(psr.residuals, A[0, :], Nvec, T, Sigma)
            ip2 = self.innerProduct_rr(psr.residuals, A[1, :], Nvec, T, Sigma)
            N = np.array([ip1, ip2])
                                  
            # define M matrix M_ij=(A_i|A_j)
            for jj in range(2):
                for kk in range(2):
                    M[jj, kk] = self.innerProduct_rr(A[jj, :], A[kk, :], Nvec, T, Sigma)

            # take inverse of M
            Minv = np.linalg.pinv(M)
            fstat += 0.5 * np.dot(N, np.dot(Minv, N))
    
        return fstat

def innerProduct_rr(x, y, Nvec, Tmat, Sigma, xNT=None, TNy=None):
        """
        Compute inner product using rank-reduced
        approximations for red noise/jitter
        Compute: x^T N^{-1} y - x^T N^{-1} T \Sigma^{-1} T^T N^{-1} y

        :param x: vector timeseries 1
        :param y: vector timeseries 2
        :param Nvec: diagonal of white noise matrix
        :param Tmat: Modified design matrix including red noise/jitter
        :param Sigma: Sigma matrix (\varphi^{-1} + T^T N^{-1} T)
        :param xNT: x^T N^{-1} T precomputed
        :param TNy: T^T N^{-1} y precomputed
        :return: inner product (x|y)
        """
        # white noise term
        xNy = Nvec.solve(y,left_array=x)

        if xNT == None and TNy == None:
            xNT = Nvec.solve(Tmat,left_array=x)
            TNy = Nvec.solve(y,left_array=Tmat)

        cf = sl.cho_factor(Sigma)
        SigmaTNy = sl.cho_solve(cf, TNy)

        ret = xNy - np.dot(xNT, SigmaTNy)

        return ret