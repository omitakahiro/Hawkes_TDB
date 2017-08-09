import sys,time,datetime,copy,subprocess,itertools,pickle,warnings

import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

import scipy.sparse as spm

from StatTool import Quasi_Newton,Bayesian_Smoothing

import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
import C_Hawkes

###########################################################################################
###########################################################################################
## Estimation
###########################################################################################
###########################################################################################

##################################################
## Hawkes process with exponential kernels
##################################################
def Estimate_exp(Data,t,prior=[],opt=[]):
    
    T = Data.query('%f < T < %f'%(t['st'],t['en']))['T'].values.copy()
    
    #setting
    para_list   = ['mu','alpha','beta']
    para_length = 3
    para_exp    = ['mu','alpha','beta']
    para_ini    = pd.Series({'mu':0.1, 'alpha':0.8, 'beta':0.05})
    para_step_Q = pd.Series({'mu':0.2, 'alpha':0.2, 'beta':0.2 })
    para_step_H = pd.Series({'mu':0.1, 'alpha':0.1, 'beta':0.5 })
    
    stg = {'para_list':para_list,'para_length':para_length,'para_ini':para_ini,'para_step_Q':para_step_Q,'para_exp':para_exp,'para_step_H':para_step_H}
    
    #condition
    cdt = {'t':t}
    
    #Quasi Newton
    [para,L,ste,G_norm] = Quasi_Newton(LG_exp,T,cdt,stg,prior,opt)

    if 'fig' in opt:
        Graph_Residual(Data,para,t)
    
    if 'rslt' in opt:
        print para
        print "L: %f" % L
        print "L_c: %f" % (L-para_length)
    
    return {'para':para, 'L':L, 't':t, 'ste':ste, 'prior':prior, 'L_c':(L-para_length), 'G_norm':G_norm, 'br':para["alpha"]}


def Estimate_exp_g(Data,t,num_exp=1,knots=[0.],prior=[],opt=[]): # multiple exponential kenels, mu is piece wise liniear function
    
    T = Data.query('%f < T < %f'%(t['st'],t['en']))['T'].values.copy()
    num_knots = len(knots)
    
    #setting
    para_list = []
    para_ini = {}
    para_step_Q = {}
    para_step_H = {}
    
    for i in range(num_knots):
        m_key = "mu%d" % i
        para_list.append(m_key)
        para_ini[m_key] = 0.1
        para_step_Q[m_key] = 0.2
        para_step_H[m_key] = 0.01
    
    for i in range(num_exp):
        a_key = "alpha%d" % i; b_key = "beta%d" % i;
        para_list.extend([a_key,b_key])
        para_ini.update({    a_key:0.8/num_exp, b_key:np.random.rand() })
        para_step_Q.update({ a_key:0.2        , b_key:0.2              })
        para_step_H.update({ a_key:0.01       , b_key:0.01             })
        
    para_length = num_knots + 2*num_exp
    para_exp    = para_list
    para_ini    = pd.Series(para_ini   )[para_list]
    para_step_Q = pd.Series(para_step_Q)[para_list]
    para_step_H = pd.Series(para_step_H)[para_list]
      
    stg = {'para_list':para_list,'para_length':para_length,'para_ini':para_ini,'para_step_Q':para_step_Q,'para_exp':para_exp,'para_step_H':para_step_H}
    
    #condition
    cdt = {'t':t, 'num_exp':num_exp, 'num_knots':num_knots, "knots":knots, "para_list":para_list}
    
    #Quasi Newton
    [para,L,ste,G_norm] = Quasi_Newton(LG_exp_g,T,cdt,stg,prior,opt)
    
    if 'fig' in opt:
        Graph_Residual(Data,para,t)
    
    if 'rslt' in opt:
        print para
        print "L: %f" % L
        print "L_c: %f" % (L-para_length)
        print "branching ratio: %f" % (np.array([ para["alpha%d"%i] for i in range(num_exp)]).sum())
    
    return {'para':para, 'L':L, 't':t, 'ste':ste, 'prior':prior, 'L_c':(L-para_length), 'G_norm':G_norm, 'br':(np.array([ para["alpha%d"%i] for i in range(num_exp)]).sum()), "num_exp":num_exp, "knots":knots}

##################################################
## Hawkes process with time-varying mu
##################################################
def Estimate(T,t):
    df = pd.DataFrame({"T":T})
    t = {"st":t[0],"en":t[1]}
    param1 = Estimate_tvm_g(df,t,num_exp=1,para0=None ,n_bin=50,opt=["timeout"])
    para0 = copy_para(param1["para"],1)
    param2 = Estimate_tvm_g(df,t,num_exp=2,para0=para0,n_bin=50,opt=["timeout"])
    
    param = {"alpha1":param2["para"]["alpha0"],"alpha2":param2["para"]["alpha1"],"beta1":param2["para"]["beta0"],"beta2":param2["para"]["beta1"],"mu":param2["para"]["mu"][1:],"L":param2["L"]}
    
    return param

def Estimate_tvm_g(Data,t,num_exp=1,n_bin=100,prior=[],opt=[],method='CBS',m=None,sta=False,epsilon=0.01,para0=None):
    
    T = Data.query('%f < T < %f'%(t['st'],t['en']))['T'].values.copy()
    n = len(T)
    
    if para0 is None:
        para0 = Estimate_exp_g(Data,t,num_exp=num_exp,knots=np.linspace(t["st"],t["en"],10))["para"]
        state0_ini = np.log(para0["mu0"])
        state_ini = None
        V_ini = 1e-4
    else:
        state_ini = para0['state'].copy()
        state0_ini = para0['state0']
        V_ini = para0['V']*0.8;
        
    #setting for Quasi Newton
    para_list = []
    para_ini = {}
    para_step_Q = {}
    para_step_H = {}
    
    for i in range(num_exp):
        a_key = "alpha%d" % i; b_key = "beta%d" % i;
        para_list.extend([a_key,b_key])
        para_ini.update({    a_key:para0[a_key], b_key:para0[b_key] })
        para_step_Q.update({ a_key:0.2         , b_key:0.2          })
        para_step_H.update({ a_key:0.02        , b_key:0.02         })
        
    para_list = np.array(para_list)
    para_length = 2*num_exp
    para_exp    = para_list.copy()
    para_ini    = pd.Series(para_ini   )[para_list]
    para_step_Q = pd.Series(para_step_Q)[para_list]
    para_step_H = pd.Series(para_step_H)[para_list]
        
    stg = {'para_list':para_list,'para_length':para_length,'para_ini':para_ini,'para_step_Q':para_step_Q,'para_exp':para_exp,'para_step_H':para_step_H}
    
    #condition
    cdt = {'t':t, 'num_exp':num_exp}
    
    #setting for Bayesian Smoothing
    if method == 'CBS':
        m = np.max([int(np.rint(1.*n/n_bin)) + 3, 4]) if m is None else m; #print (n,m);
        BasisFunc = {'method':'CBS', 'x':np.arange(1,n+2,dtype='f8'), 'edge':[0.,n+2.], 'm':m, 'state_ini':state_ini, 'state0_ini':state0_ini, 'V_ini':V_ini, 'epsilon':epsilon}
        
        if sta is True:
            prior.append(['V','f',1e-10,0])
    
    cdt['BS'] = {'BasisFunc':BasisFunc, 'F_LGH':LGH_tvm_g, 'print_FM':True if 'print_FM' in opt else False}
    
    #optimization
    [para,state,L,ste,G_norm] = Bayesian_Smoothing(T,cdt,stg,prior=prior,opt=opt)
    mu = np.exp(cdt['BS']['BasisMat'].dot(state))
    
    if 'fig' in opt:
        plt.figure()
        plt.semilogy(T,mu[1:])
    
    if 'rslt' in opt:
        print "####################################################"
        print para
        print "L: %f" % L
        print "L_c: %f" % (L-stg["para_length"])
        print "G_norm: %f" % G_norm
        print "branching ratio: %f" % (np.array([ para["alpha%d"%i] for i in range(num_exp)]).sum())
        print "####################################################\n"
        
    
    para = dict(zip(para.index,para.values))
    para.update({'mu':mu, "state":state})
    
    return {'para':para, 'L':L, 'ste':ste, 't':t, 'prior':prior, 'L_c':L-stg["para_length"], "G_norm":G_norm, 'br':(np.array([ para["alpha%d"%i] for i in range(num_exp)]).sum()), "num_exp":num_exp, "L_model":cdt["L_model"], "m":m}

        
def Graph_mu_tvm(param):
    plt.figure()
    plt.semilogy(param['T'],param['para']['mu'][1:],'k-')
    
##################################################
## Likelihood and Gradient
##################################################
def LG_exp(para,T,cdt,only_L=False):
    
    mu = para['mu']; alpha = para['alpha']; beta = para['beta'];
    st = cdt['t']['st']; en = cdt['t']['en'];
    n = len(T)
    
    ######################### SUM term
    ###python version
    #[l,dl_a,dl_b] = SUM_exp(alpha,beta,T,n)
    
    ###cython version
    [l,dl_a,dl_b] = C_Hawkes.SUM_exp(alpha,beta,T,n)
    
    l = mu + l;
    dl = pd.DataFrame({"mu":np.ones(n,dtype=np.float64),"alpha":dl_a,"beta":dl_b})
    
    Sum = np.log(l).sum()
    d_Sum = pd.Series( (  dl[['mu','alpha','beta']].values/l.reshape(-1,1) ).sum(axis=0) ,index=['mu','alpha','beta'])
    
    ######################### INT term
    [I,dI_a,dI_b] = INT_exp(alpha,beta,T,en)
    
    Int = mu*(en-st) + I;
    d_Int = pd.Series({"mu":en-st,"alpha":dI_a,"beta":dI_b})
    
    #######
    L = Sum - Int
    G = d_Sum - d_Int
    
    return [L,G]

def LG_exp_g(para,T,cdt,only_L=False): 
    
    ### read cdt
    n = len(T)
    st = cdt['t']['st']; en = cdt['t']['en'];
    num_exp = cdt['num_exp']; num_knots = cdt['num_knots']; knots = cdt['knots']; para_list = cdt["para_list"]
    
    ### initial value
    l = np.zeros(n); dl = {};
    Int = 0; d_Int = {};
    
    ### background part
    if num_knots == 1:
        m_key = "mu0"
        mu = para[m_key]
        l = l + mu
        dl[m_key] = np.ones(n,dtype=np.float64);
        Int = Int + (en-st)*mu
        d_Int[m_key] = (en-st)
        
    else:
        for i in range(num_knots):
            m_key = "mu%d" % i
            dl[m_key] = np.zeros(n)
            d_Int[m_key] = 0
            
        for i in range(num_knots-1):
            m_key1 = "mu%d" % i; m_key2 = "mu%d" % (i+1);
            
            ## SUM term
            [mu_i,d_mu1,d_mu2] = SUM_mu(T,para[m_key1],para[m_key2],knots[i],knots[i+1])
            l = l + mu_i
            dl[m_key1] = dl[m_key1] + d_mu1
            dl[m_key2] = dl[m_key2] + d_mu2
            
            ## INT term
            [Int_mu,d_Int_mu1,d_Int_mu2] = INT_mu(para[m_key1],para[m_key2],knots[i],knots[i+1])
            Int = Int + Int_mu
            d_Int[m_key1] = d_Int[m_key1] + d_Int_mu1
            d_Int[m_key2] = d_Int[m_key2] + d_Int_mu2
    
    ### triggering part
    for i in range(num_exp):
        a_key = "alpha%d" % i; b_key = "beta%d" % i;
        
        ## SUM term
        #[l_i,dl_a_i,dl_b_i] = SUM_exp(para[a_key],para[b_key],T,n) # python version
        [l_i,dl_a_i,dl_b_i] = C_Hawkes.SUM_exp(para[a_key],para[b_key],T,n) # cython verision
        l = l + l_i;
        dl.update({a_key:dl_a_i,b_key:dl_b_i})
        
        ## Int term
        [I,dI_a,dI_b] = INT_exp(para[a_key],para[b_key],T,en)
        Int = Int + I
        d_Int.update({a_key:dI_a,b_key:dI_b})
        
    ##SUM
    dl = pd.DataFrame(dl)
    
    Sum = np.log(l).sum()
    d_Sum = pd.Series( (  dl[para_list].values/l.reshape(-1,1) ).sum(axis=0), index=para_list )
    
    ## INT    
    d_Int = pd.Series(d_Int)[para_list]
    
    #######
    L = Sum - Int
    G = d_Sum - d_Int 
    
    ### penalty for branching ratio
    br = np.array([ para["alpha%d"%i] for i in range(num_exp)]).sum()
    
    penalty = 0.0 if br < 0.95 else -100.0*(br-0.95)**2.0
    L = L + penalty
    
    for i in range(num_exp):
       a_key = "alpha%d" % i
       penalty = 0.0 if br < 0.95 else -200.0*(br-0.95)
       G[a_key] = G[a_key] + penalty
    
    return [L,G]

#### 
def SUM_exp(alpha,beta,T,n):
    
    l    = np.zeros(n)
    dl_a = np.zeros(n)
    dl_b = np.zeros(n)
    
    """
    for i in np.arange(1,n):
        l[i]    = ( alpha*beta*np.exp(-beta*(T[i]-T[0:i]))                                                        ).sum()
        dl_a[i] = (       beta*np.exp(-beta*(T[i]-T[0:i]))                                                        ).sum()
        dl_b[i] = ( alpha*     np.exp(-beta*(T[i]-T[0:i])) - alpha*beta*(T[i]-T[0:i])*np.exp(-beta*(T[i]-T[0:i])) ).sum()
    """
    
    dT = T[1:]-T[:-1]
    r = np.exp(-beta*dT)
    x = 0.0; x_a = 0.0; x_b = 0.0;
    
    for i in np.arange(n-1):
        x   = ( x   + alpha*beta ) * r[i]
        x_a = ( x_a +       beta ) * r[i]
        x_b = ( x_b + alpha      ) * r[i] - x*dT[i]
        
        l[i+1] = x; dl_a[i+1] = x_a; dl_b[i+1] = x_b;

    return [l,dl_a,dl_b]

def INT_exp(alpha,beta,T,en):
    
    I    = alpha*(1.0-np.exp(-beta*(en-T))).sum()
    dI_a =       (1.0-np.exp(-beta*(en-T))).sum()
    dI_b = alpha*( (en-T)*np.exp(-beta*(en-T)) ).sum()
    
    return [I,dI_a,dI_b]

def SUM_mu(T,mu1,mu2,knot1,knot2):
    n = len(T)
    mu = np.zeros(n)
    d_mu1 = np.zeros(n)
    d_mu2 = np.zeros(n)
    
    index = (knot1<T) & (T<knot2)
    w1 =  (knot2-T[index])/(knot2-knot1)
    w2 =  (T[index]-knot1)/(knot2-knot1)
    mu[index] = w1*mu1 + w2*mu2
    d_mu1[index] = w1
    d_mu2[index] = w2
       
    return [mu,d_mu1,d_mu2]

def INT_mu(mu1,mu2,knot1,knot2):
    Int = (mu1+mu2)*(knot2-knot1)/2.0
    d_Int1 = (knot2-knot1)/2.0
    d_Int2 = (knot2-knot1)/2.0
    return [Int,d_Int1,d_Int2]

##############################################
def LGH_tvm_g(para,state,T,cdt):
    
    n = len(T)
    st = cdt['t']['st']; en = cdt['t']['en'];
    num_exp = cdt["num_exp"]
    T_ext = np.hstack([st,T,en]); dT = T_ext[1:] - T_ext[:-1];
    BasisMat = cdt['BS']['BasisMat']; BasisMat_T = cdt['BS']['BasisMat_T']
    
    l = np.zeros(n)
    Int_ab = 0;
    for i in range(num_exp):
        a_key = "alpha%d" % i; b_key = "beta%d" % i;
        alpha = para[a_key]; beta = para[b_key];
        l = l + C_Hawkes.SUM_exp_L(alpha,beta,T,n)
        Int_ab = Int_ab + alpha*(1.0-np.exp(-beta*(en-T))).sum()
    
    mu = np.exp(BasisMat.dot(state))
    l = l + mu[1:]
    mubyl = np.append(0,mu[1:]/l)
    d_log_l  = BasisMat_T.dot(mubyl)
    d2_log_l = BasisMat_T.dot( spm.diags(-mubyl**2.0+mubyl,offsets=0).dot(BasisMat) )
    
    d_mu_int  = BasisMat_T.dot(mu*dT)
    d2_mu_int = BasisMat_T.dot( spm.diags(mu*dT,offsets=0).dot(BasisMat) )
    
    ##LGH
    L = np.log(l).sum() - ( (mu*dT).sum() + Int_ab )
    G = d_log_l         - d_mu_int                                                                               
    H = d2_log_l        - d2_mu_int                                                                              
    
    ### penalty for branching ratio
    br = np.array([ para["alpha%d"%i] for i in range(num_exp)]).sum()
    penalty = 0.0 if br < 0.95 else -100.0*(br-0.95)**2.0
    L = L + penalty
    
    
    return [L,G,H]

##################################################
## Residual Analysis
##################################################
from scipy import stats

def ER_TEST(Data,param,t):
    itv = TransformedInterval(Data,param,t)
    
    #x = np.exp(-itv)
    #static = ( np.mean( (x-0.5)**2.0 ) - 1.0/12.0 ) * np.sqrt(len(x)) / np.sqrt(1.0/180.0)
    
    x = itv
    static = ( np.var(x) - 1.0 ) * np.sqrt(len(x)) / np.sqrt(8.0)
    
    static = - np.abs(static)
    return stats.norm.cdf( static ) * 2.0
    
def KS_TEST(Data,param,t):
    itv = TransformedInterval(Data,param,t)
    return stats.kstest(np.exp(-itv),'uniform')

def INT_exp_itv(alpha,beta,T):
    n = len(T)
    d_T = T[1:] - T[:-1]
    l = C_Hawkes.SUM_exp_L(alpha,beta,T,n)
    itv = (l[:-1]+alpha*beta) * ( 1.0 - np.exp(-beta*d_T) ) / beta
    
    return itv

def TransformedInterval(Data,param,t):
    
    num_exp = param["num_exp"]
    para = param["para"]
    
    T = Data.query('%f < T < %f'%(t['st'],t['en']))['T'].values.copy()
    n = len(T)
    itv = np.zeros(n-1)
    
    ## triggering
    for i in range(num_exp):
        a_key = "alpha%d"%i; b_key = "beta%d"%i;
        alpha = para[a_key]; beta = para[b_key]
        itv = itv + INT_exp_itv(alpha,beta,T)
    
    ## background
    if "knots" in param:
        knots = param["knots"]
        num_knots = len(knots)
        mu = np.zeros(n)
    
        if num_knots == 1:
            itv = itv + para['mu0']*(T[1:]-T[:-1])
        else:
            for i in range(num_knots-1):
                m_key1 = "mu%d" % i; m_key2 = "mu%d" % (i+1);
                [mu_i,_,_] = SUM_mu(T,para[m_key1],para[m_key2],knots[i],knots[i+1])
                mu = mu + mu_i
            
            itv = itv + mu[:-1]*(T[1:]-T[:-1])
            
    elif "state" in para:
        mu = para["mu"][1:]
        itv = itv + mu[:-1]*(T[1:]-T[:-1])
    else:
        sys.exit("INVALID PARAM")
    
    return itv

"""
def TimeTransform(Data,para,t):
    
    try:
        T = Data.query('%f < T < %f'%(t['st'],t['en']))['T'].values
    except:
        pass
    
    mu = para['mu']; alpha = para['alpha']; beta = para['beta'];
    st = t['st']; en = t['en'];
    n = len(T)
    
    T_ord = np.hstack([st,T,en])
    Int_l = mu*(T_ord[1:]-T_ord[:-1])
    
    for i in np.arange(1,n+1):
        Int_l[i] += alpha* ( np.exp(-beta*(T_ord[i]-T[0:i])) - np.exp(-beta*(T_ord[i+1]-T[0:i])) ).sum()
    
    T_trans = np.hstack([0,Int_l.cumsum()])
    ste = np.nan_to_num(np.sqrt( T_trans*(1.0-T_trans/n) ))
    
    Er = {'l':T_trans-2.57*ste,'u':T_trans+2.57*ste}
    
    return [T_ord,T_trans,Er]

def Graph_Residual(Data,para,t):
    
    [T_ord,T_trans,Er] =  TimeTransform(Data,para,t)
    
    plt.figure(figsize=(5,5))
    n = len(T_ord)-2
    plt.fill_between(T_trans,Er['l'],Er['u'],facecolor=[1,0.7,0.7],edgecolor=[1,0.7,0.7])
    plt.plot(T_trans[1:-1],np.arange(1,n+1),'k.',markersize=2)
    plt.plot([0,n],[0,n],'r-')
    plt.xlim([0,n])
    plt.ylim([0,n])

def Graph_Residual_param(param):
    T = param['T']; para = param['para']; t = param['t']
    Graph_Residual(T,para,t)
"""
###########################################################################################
###########################################################################################
## Simulation
###########################################################################################
###########################################################################################
def Simulate(alpha,beta,mu,t):
    
    if len(alpha) != len(beta):
        sys.exit("len(alpha) != len(beta)")
    
    st = t['st']; en = t['en'];
    num_exp = len(alpha)
    
    alpha = np.array(alpha); beta = np.array(beta);
    T = np.empty(1000000,dtype='f8')
    x = st; l0 = mu; i = 0;
    l_trg = np.zeros(num_exp)
    
    while 1:
        
        itv = np.random.exponential()/l0
        x += itv
        l_trg = l_trg*np.exp(-beta*itv)
        l1 = mu + l_trg.sum() 
        
        #print x,l1/l0
        
        if (x>en) or (i==1000000):
            break
        
        if np.random.rand() < l1/l0: ## Fire
            T[i] = x
            i += 1
            l_trg = l_trg + alpha*beta
                
        l0 = mu + l_trg.sum()
        
    T = pd.DataFrame({'T':T[:i]})
    
    return T

def Simulate_tvm(alpha,beta,Func_mu,t):
    
    if len(alpha) != len(beta):
        sys.exit("len(alpha) != len(beta)")
    
    st = t['st']; en = t['en'];
    num_exp = len(alpha)
    
    alpha = np.array(alpha); beta = np.array(beta);
    T = np.empty(1000000,dtype='f8')
    x = st; mu = Func_mu(x); l0 = mu; i = 0;
    l_trg = np.zeros(num_exp)
    
    while 1:
        
        itv = np.random.exponential()/l0
        x += itv
        l_trg = l_trg*np.exp(-beta*itv)
        l1 = mu + l_trg.sum()
        mu = Func_mu(x)
        
        #print x,l1/l0
        
        if (x>en) or (i==1000000):
            break
        
        if np.random.rand() < l1/l0: ## Fire
            T[i] = x
            i += 1
            l_trg = l_trg + alpha*beta
                
        l0 = mu + l_trg.sum()
        
    T = pd.DataFrame({'T':T[:i]})
    
    return T

###########################################################################################
###########################################################################################
## basic function
###########################################################################################
###########################################################################################
def copy_para(para,num_exp):
    para_tmp = {}
    para_tmp["V"] = para["V"]
    para_tmp["state0"] = para["state0"]
    para_tmp["state"] = para["state"].copy()
    
    beta = []
    for i in range(num_exp):
        a_key = "alpha%d"%i; b_key = "beta%d"%i;
        para_tmp[a_key] = para[a_key]
        para_tmp[b_key] = para[b_key]
        beta.append(para[b_key])
    
    a_key = "alpha%d"%num_exp; b_key = "beta%d"%num_exp;
    para_tmp[a_key] = 0.01
    para_tmp[b_key] = np.array(beta).min()*1.1
    
    return para_tmp
    
######################################################################
######################################################################
## demo
######################################################################
######################################################################
def demo_Hawkes():
    import catalog
    Data_M = catalog.load_pkl('/Users/omi/Desktop/TSE6200/NK225MF.pkl')

    st = datetime.datetime(2016,1,29,9,0,0)
    en = datetime.datetime(2016,1,29,15,10,0)
    dt = {'st':st,'en':en}
    df = Data_M.extract_PC(dt,5,jitter=1.0) if st < datetime.datetime(2016,7,18,16,0) else Data_M.extract_PC(dt,5,jitter=0.001)
    t = df.t
    
    param_3_1 = Estimate_tvm_g(df,t,num_exp=1,            opt=["timeout","print","rslt"]); para0 = copy_para(param_3_1["para"],1);
    param_3_2 = Estimate_tvm_g(df,t,num_exp=2,para0=para0,opt=["timeout","print","rslt"]); para0 = copy_para(param_3_2["para"],2);
    param_3_3 = Estimate_tvm_g(df,t,num_exp=3,para0=para0,opt=["timeout","print","rslt"]); para0 = copy_para(param_3_3["para"],3);
    param_3_4 = Estimate_tvm_g(df,t,num_exp=4,para0=para0,opt=["timeout","print","rslt"])

if __name__ == '__main__':
    demo_Hawkes()