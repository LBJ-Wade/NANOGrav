#from __future__ import division
#import glob
#import os
import numpy as np
#import cPickle as pickle
#from scipy.stats import skewnorm
#import copy_reg

#import warnings
#warnings.filterwarnings("error")

#from enterprise.signals import parameter
#from enterprise.pulsar import Pulsar
#from enterprise.signals import selections
from enterprise.signals import signal_base
#from enterprise.signals import white_signals
#from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const
#from enterprise.signals import utils

#from empirical_distributions import EmpiricalDistribution1D
#from empirical_distributions import EmpiricalDistribution2D
#from empirical_distributions import EmpiricalDistribution3D

#from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

###Justin's original version of gw_antenna_pattern

def create_gw_antenna_pattern(theta, phi, gwtheta, gwphi):
    """
    Function to create pulsar antenna pattern functions as defined
    in Ellis, Siemens, and Creighton (2012).
    :param theta: Polar angle of pulsar location.
    :param phi: Azimuthal angle of pulsar location.
    :param gwtheta: GW polar angle in radians
    :param gwphi: GW azimuthal angle in radians
    
    :return: (fplus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the 
             pulsar and the GW source.
    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    m = np.array([-np.sin(gwphi), np.cos(gwphi), 0.0])
    n = np.array([-np.cos(gwtheta)*np.cos(gwphi), 
                  -np.cos(gwtheta)*np.sin(gwphi),
                  np.sin(gwtheta)])
    omhat = np.array([-np.sin(gwtheta)*np.cos(gwphi), 
                      -np.sin(gwtheta)*np.sin(gwphi),
                      -np.cos(gwtheta)])

    phat = np.array([np.sin(theta)*np.cos(phi), 
                     np.sin(theta)*np.sin(phi), 
                     np.cos(theta)])

    fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
    fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
    cosMu = -np.dot(omhat, phat)

    return fplus, fcross, cosMu


@signal_base.function
def cw_delay(toas, theta, phi, pdist, p_dist=1, p_phase=None, 
             cos_gwtheta=0, gwphi=0, log10_mc=9, log10_dL=2, log10_fgw=-8, 
             phase0=0, psi=0, cos_inc=0, log10_h=None, 
             inc_psr_term=True, model='phase_approx', tref=57387*86400):
    """
    Function to create GW incuced residuals from a SMBMB as 
    defined in Ellis et. al 2012,2013.
    :param toas: Pular toas in seconds
    :param theta: Polar angle of pulsar location.
    :param phi: Azimuthal angle of pulsar location.
    :param cos_gwtheta: Cosine of Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    :param log10_mc: log10 of Chirp mass of SMBMB [solar masses]
    :param log10_dL: log10 of Luminosity distance to SMBMB [Mpc]
    :param log10_fgw: log10 of Frequency of GW (twice the orbital frequency) [Hz]
    :param phase0: Initial Phase of GW source [radians]
    :param psi: Polarization of GW source [radians]
    :param cos_inc: cosine of Inclination of GW source [radians]
    :param p_dist: Pulsar distance to use other than those in psr [kpc]
    :param p_phase: Use pulsar phase to determine distance [radian]
    :param psrTerm: Option to include pulsar term [boolean]
    :param model: Which model to use to describe the frequency evolution
    :param tref: Reference time for phase and frequency [s]
    
    :return: Vector of induced residuals
    """
    
    # convert pulsar distance
    p_dist = (pdist[0] + pdist[1]*p_dist)*const.kpc/const.c
    
    # convert units
    mc = 10**log10_mc * const.Tsun
    gwtheta = np.arccos(cos_gwtheta)
    fgw = 10**log10_fgw
    
    # is log10_h is given, use it
    if log10_h is not None:
        dist = 2 * mc**(5/3) * (np.pi*fgw)**(2/3) / 10**log10_h
    else:
        dist = 10**log10_dL * const.Mpc / const.c

    # get antenna pattern funcs and cosMu
    fplus, fcross, cosMu = create_gw_antenna_pattern(theta, phi, gwtheta, gwphi)
    
    # get pulsar time
    toas -= tref
    if p_dist > 0:
        tp = toas-p_dist*(1.-cosMu)
    else:
        tp = toas
    
    # orbital frequency
    w0 = np.pi * fgw
    phase0 /= 2 # orbital phase
    omegadot = 96/5 * mc**(5/3) * w0**(11/3)
    
    #print('w0='+str(w0))
    #print('mc='+str(mc))
    #print('pdist='+str(p_dist))
    #print('cosMu='+str(cosMu))
    #print('in exponent='+str(1. + 256/5 * mc**(5/3) * w0**(8/3) * p_dist*(1.-cosMu)))

    if model == 'evolve':

        # calculate time dependent frequency at earth and pulsar
        omega = w0 * (1. - 256/5 * mc**(5/3) * w0**(8/3) * toas)**(-3/8)
        omega_p = w0 * (1. - 256/5 * mc**(5/3) * w0**(8/3) * tp)**(-3/8)
        
        if p_dist > 0:
            omega_p0 = w0 * (1. + 256/5 * mc**(5/3) * w0**(8/3) * p_dist*(1.-cosMu))**(-3/8)
        else:
            omega_p0 = w0

        # calculate time dependent phase
        phase = phase0 + 1/32*mc**(-5/3) * (w0**(-5/3) - omega**(-5/3))
        
        if p_phase is None:
            phase_p = phase0 + 1/32*mc**(-5/3) * (w0**(-5/3) - omega_p**(-5/3))
        else:
            phase_p = phase0 + p_phase + 1/32*mc**(-5/3) * (omega_p0**(-5/3) - omega_p**(-5/3))
        
    elif model == 'phase_approx':
        
        omega = np.pi*fgw
        phase = phase0 + omega*toas

        omega_p = w0 * (1. + 256/5 * mc**(5/3) * w0**(8/3) * p_dist*(1.-cosMu))**(-3/8)

        if p_phase is None:
            phase_p = phase0 + 1/32/mc**(5/3) * (w0**(-5/3) - omega_p**(-5/3)) + omega_p*toas
        else:
            phase_p = phase0 + p_phase + omega_p*toas
            
    else:
        omega = np.pi*fgw
        phase = phase0 + omega*toas

        omega_p = omega
        phase_p = phase0 + omega*tp
    
    # define time dependent coefficients
    At = np.sin(2*phase)*(1+cos_inc*cos_inc)
    Bt = 2*np.cos(2*phase)*cos_inc
    At_p = np.sin(2*phase_p)*(1+cos_inc*cos_inc)
    Bt_p = 2*np.cos(2*phase_p)*cos_inc

    # now define time dependent amplitudes
    alpha = mc**(5./3.)/(dist*omega**(1./3.))
    alpha_p = mc**(5./3.)/(dist*omega_p**(1./3.))
    
    # define rplus and rcross
    rplus = alpha*(At*np.cos(2*psi)+Bt*np.sin(2*psi))
    rcross = alpha*(-At*np.sin(2*psi)+Bt*np.cos(2*psi))
    rplus_p = alpha_p*(At_p*np.cos(2*psi)+Bt_p*np.sin(2*psi))
    rcross_p = alpha_p*(-At_p*np.sin(2*psi)+Bt_p*np.cos(2*psi))

    # residuals
    if inc_psr_term:
        res = fplus*(rplus_p-rplus)+fcross*(rcross_p-rcross)
    else:
        res = -fplus*rplus - fcross*rcross

    return res

def CWSignal(cw_wf, inc_psr_term=True):

    BaseClass = deterministic_signals.Deterministic(cw_wf, name='cgw')
    
    class CWSignal(BaseClass):
        
        def __init__(self, psr):
            super(CWSignal, self).__init__(psr)
            self._wf[''].add_kwarg(inc_psr_term=inc_psr_term)
            #if inc_psr_term:
            #    pdist = parameter.Normal(psr.pdist[0], psr.pdist[1])('_'.join([psr.name, 'cgw', 'pdist']))
            #    pphase = parameter.Uniform(0, 2*np.pi)('_'.join([psr.name, 'cgw', 'pphase']))
            #    self._params['p_dist'] = pdist
            #    self._params['p_phase'] = pphase
            #    self._wf['']._params['p_dist'] = pdist 
            #    self._wf['']._params['p_phase'] = pphase
    
    return CWSignal