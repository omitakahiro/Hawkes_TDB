import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import scipy.sparse.linalg as spla
import scipy.sparse as spm
#from sksparse.cholmod import cholesky,analyze,CholmodError,CholmodWarning

import sys,datetime,time,warnings
from multiprocessing import Pool
from copy import deepcopy

#warnings.filterwarnings('error',category=CholmodWarning)
#warnings.filterwarnings('error',category=RuntimeWarning)

##################################
##MCMC
##################################
def MCMC_DetermieStepSize(F_LG,para_ini,Data,cdt,stg,n_core,prior=[],opt=[]):
    
    step_size_list = np.array([0.06,0.08,0.1,0.12,0.15,0.2,0.25,0.3,0.4,0.5])
    m = len(step_size_list)
    
    p = Pool(n_core)
    rslt = []
    for i in range(m):
        stg_tmp = deepcopy(stg)
        stg_tmp['step_size'] = step_size_list[i]
        rslt.append( p.apply_async(MCMC,args=[F_LG,para_ini,Data,cdt,stg_tmp,200,prior,['print']]) )
    p.close()
    p.join()
    rslt = [ rslt[i].get() for i in range(m) ]
    
    step_size    = [ rslt[i][2] for i in range(m) ]
    r_accept     = [ rslt[i][3] for i in range(m) ]
    elapsed_time = [ rslt[i][4] for i in range(m) ]
    dtl = pd.DataFrame(np.vstack([step_size,r_accept,elapsed_time]).T,columns=['step_size','r_accept','elapsed_time'])
   
    opt_step_size = dtl.iloc[ np.argmin(np.fabs(dtl['r_accept'].values-0.5)) ]['step_size']
    
    return [opt_step_size,dtl]

def MCMC_prl(F_LG,para_ini,Data,cdt,stg,Num,n_core,prior=[],opt=[]):
    
    print( "MCMC" )
    
    ##determine step size
    [opt_step_size,dtl1] = MCMC_DetermieStepSize(F_LG,para_ini,Data,cdt,stg,n_core,prior,opt)
    stg['step_size'] = opt_step_size
    
    print( "estimated processing time %.2f minutes" % (dtl1['elapsed_time'].mean()*Num/200.0/60.0) )
    
    p = Pool(n_core)
    rslt = [ p.apply_async(MCMC,args=[F_LG,para_ini,Data,cdt,stg,Num,prior,opt])   for i in range(n_core)   ]
    p.close()
    p.join()
    rslt = [ rslt[i].get() for i in range(n_core) ]
    
    para_mcmc     = pd.concat([rslt[i][0].iloc[0::10] for i in range(n_core) ],ignore_index=True)
    L_mcmc        = np.array([ rslt[i][1][0::10]      for i in range(n_core) ]).flatten()
    step_size     = np.array([ rslt[i][2]             for i in range(n_core) ])
    r_accept      = np.array([ rslt[i][3]             for i in range(n_core) ])
    elapsed_time  = np.array([ rslt[i][4]             for i in range(n_core) ])
    dtl2 = pd.DataFrame(np.vstack([step_size,r_accept,elapsed_time]).T,columns=['step_size','r_accept','elapsed_time'])
    
    dtl_mcmc = {'step_size':opt_step_size,'dtl1':dtl1,'dtl2':dtl2}
    
    return [para_mcmc,L_mcmc,dtl_mcmc]
    
    
def MCMC(F_LG,para_ini,Data,cdt,stg,Num,prior=[],opt=[]):
    
    #random number seed
    seed = datetime.datetime.now().microsecond *datetime.datetime.now().microsecond % 4294967295
    np.random.seed(seed)
    
    para_list  = stg['para_list']
    m          = stg['para_length']
    para_exp   = stg['para_exp']
    step_size  = stg['step_size']
    
    ##prior format transform
    if len(prior)>0:
        prior = pd.DataFrame(prior,columns=['name','type','mu','sigma'])
    
    #prepare
    para_mcmc = pd.DataFrame(index=np.arange(Num),columns=stg['para_list'],dtype='f8')
    L_mcmc    = np.zeros(Num)
    
    #initial value
    para1 = para_ini.copy()
    para_mcmc.iloc[0] = para1
    [L1,_] = Penalized_LG(F_LG,para1,Data,cdt,prior,only_L=True)
    L_mcmc[0] = L1
    
    #exponential parameter check
    para_ord = np.setdiff1d(para_list,para_exp)
    
    #step
    step_MCMC = stg['ste'].copy()
    step_MCMC[para_exp] =  np.minimum( np.log( 1.0 + step_MCMC[para_exp]/para1[para_exp] ) ,0.4)
    
    i = 1
    j = 0
    k = 0
    t_start = time.time()
    
    while 1:
        
        para2 = para1.copy()
        para2[para_ord] = para1[para_ord] +         step_size*np.random.randn(len(para_ord))*step_MCMC[para_ord]
        para2[para_exp] = para1[para_exp] * np.exp( step_size*np.random.randn(len(para_exp))*step_MCMC[para_exp] )
        
        [L2,_] = Penalized_LG(F_LG,para2,Data,cdt,prior,only_L=True)
        
        if L1<L2 or np.random.rand() < np.exp(L2-L1): #accept
            
            j += 1
            k += 1
            para1 = para2
            L1 = L2
            
            para_mcmc.iloc[i] = para1
            L_mcmc[i] = L1
            
        else:
            
            para_mcmc.iloc[i] = para_mcmc.iloc[i-1]
            L_mcmc[i] = L_mcmc[i-1]
        
        if 'print' in opt and np.mod(i,1000) == 0:
            print(i)
        
        #adjust the step width
        if np.mod(i,500) == 0:
            
            if k<250:
                step_size *= 0.95
            else:
                step_size *= 1.05
            
            k = 0
        
        i += 1
        
        if i == Num:
            break
    
    
    r_accept = 1.0*j/Num
    elapsed_time = time.time() - t_start
    
    return [para_mcmc,L_mcmc,step_size,r_accept,elapsed_time]

##################################
##Quasi-Newton
##################################
def Quasi_Newton(F_LG,Data,cdt,stg,prior=[],opt=[]):
    
    index = 0
    check_reset = False
    L_list = np.zeros(10000)
    
    ##parameter setting
    para_ini   = stg['para_ini'].copy()
    para_list  = stg['para_list']
    m          = stg['para_length']
    step_Q     = stg['para_step_Q'][para_list].values
    para_exp   = stg['para_exp']
    
    ##initial value
    para = para_ini
    
    ##prior format transform
    if len(prior)>0:
        prior = pd.DataFrame(prior,columns=['name','type','mu','sigma'])
    
    ##fix check
    if len(prior)>0:
        para_fix   = prior[ prior['type']=='f' ]['name'].values.astype('S')
        para_value = prior[ prior['type']=='f' ]['mu'].values
        para[para_fix] = para_value

    ##exponential parameter check
    para_ord = np.setdiff1d(para_list,para_exp)

    #calculate Likelihood and Gradient for the initial state
    [L1,G1] = Penalized_LG(F_LG,para,Data,cdt,prior)
    G1[para_exp] = G1[para_exp] * para[para_exp]
    G1 = G1[para_list].values
    L_list[0] = L1;
    
    #OPTION return likelihood
    if 'L' in opt:
        return [para,L1,[]]
    
    
    ###main
    H = np.eye(m)
    
    while 1:
        
        if 'print' in opt:
            for para_name in para_list:
                print( "%s: %e" % (para_name,para[para_name]) )
            print( "%d: L = %.3f, norm(G) = %e\n" % (index,L1,np.linalg.norm(G1)) )
            
        #break rule
        if np.linalg.norm(G1) < 1e-3 :
            break
        
        """
        if ( index == 100 ) and ( "timeout" in opt ) :
            print("QUASI NEWTON TIMEOUT\n")
            break
        """
        
        if ( index > 40 ) and ( L_list[index-10:index].max() - L_list[:index-10].max() < 0.1 ) and ( "timeout" in opt ):
            #print("QUASI NEWTON TIMEOUT: CONVERGENT\n")
            break
        
        #calculate direction
        s = H.dot(G1)
        s = s/np.max([np.max(np.abs(s)/step_Q),1])
        
        #update parameter value
        i_ls = 0
        
        while 1:
            para_tmp = para.copy()
            s_series = pd.Series(s,index=para_list)
            para_tmp[para_ord] = para[para_ord]  + s_series[para_ord]
            para_tmp[para_exp] = para[para_exp]  * np.exp(s_series[para_exp])
            
            #calculate Likelihood and Gradient
            [L2,_] = Penalized_LG(F_LG,para_tmp,Data,cdt,prior,only_L=True)
            
            if 'print' in opt:            
                print 'i_ls',i_ls,L1,L2
            
            if i_ls == 15:
                check_reset = True
            
            if (L1-0.005 <= L2) or (i_ls==15):
                para = para_tmp
                [L2,G2] = Penalized_LG(F_LG,para,Data,cdt,prior)
                G2[para_exp] = G2[para_exp] * para[para_exp]
                G2 = G2[para_list].values
                break
            else:
                s = s*0.5
            
            i_ls += 1
        
        #update hessian matrix
        y=G1-G2;
        y = y.reshape(m,1)
        s = s.reshape(m,1)
        
        if  y.T.dot(s) > 0:
            H = H + (y.T.dot(s)+y.T.dot(H).dot(y))*(s*s.T)/(y.T.dot(s))**2.0 - (H.dot(y)*s.T+(s*y.T).dot(H))/(y.T.dot(s))
        else:
            H = np.eye(m)
            
        if check_reset:
            H = np.eye(m)
            check_reset = False
        
        #update Gradients
        L1 = L2
        G1 = G2
        
        #update index
        index = index + 1
        
        L_list[index] = L1;
        
    
    ###OPTION: Estimation Error
    if 'ste' in opt:
        ste = Estimation_Error(F_LG,para,Data,cdt,stg,prior)
    else:
        ste = []
    
    ###OPTION: Check map solution
    if 'check' in opt:
            Check_QN(F_LG,para,Data,cdt,stg,prior)
    
    return [para,L1,ste,np.linalg.norm(G1)]

#################################
def Gradient_dev(F_LG,para,Data,cdt,stg,prior,para_name,d):
    
    para_list = stg['para_list']
    para_tmp = para.copy()
    para_tmp[para_name] =  para_tmp[para_name] + d
    
    [_,G] = Penalized_LG(F_LG,para_tmp,Data,cdt,prior)
    G = G[para_list].values
    
    return G

def Hessian_Numerical(F_LG,para,Data,cdt,stg,prior):
    
    para_list = stg['para_list']
    m         = stg['para_length']
    para_exp  = stg['para_exp']
    para_step_H = stg['para_step_H']
    
    d = para_step_H.copy()
    d[para_exp] = d[para_exp] * para[para_exp]
    
    H = np.zeros([m,m])
    
    for i in range(m):
        para_name = para_list[i]
        
        """
        G1 = Gradient_dev(F_LG,para,Data,cdt,stg,prior,para_name,-1.0*d[para_name])
        G2 = Gradient_dev(F_LG,para,Data,cdt,stg,prior,para_name, 1.0*d[para_name])
        H[:,i] = (G2-G1)/d[para_name]/2.0
        """
        G1 = Gradient_dev(F_LG,para,Data,cdt,stg,prior,para_name,-2.0*d[para_name])
        G2 = Gradient_dev(F_LG,para,Data,cdt,stg,prior,para_name,-1.0*d[para_name])
        G3 = Gradient_dev(F_LG,para,Data,cdt,stg,prior,para_name, 1.0*d[para_name])
        G4 = Gradient_dev(F_LG,para,Data,cdt,stg,prior,para_name, 2.0*d[para_name])
        H[:,i] = (G1-8.0*G2+8.0*G3-G4)/d[para_name]/12.0
    
    return H

def Estimation_Error(F_LG,para,Data,cdt,stg,prior):
    
    para_list = stg['para_list']
    m         = stg['para_length']
    
    H = Hessian_Numerical(F_LG,para,Data,cdt,stg,prior)
    
    if len(prior)>0:
        para_fix = prior[prior['type']=='f']['name'].values.astype('S')
        index_ord = np.array([ not (para_name in para_fix) for para_name in para_list ])
    else:
        index_ord = np.repeat(True,m)
    
    C = np.zeros([m,m])
    C[np.ix_(index_ord,index_ord)] = np.linalg.inv(-H[np.ix_(index_ord,index_ord)])
    ste = pd.Series(np.sqrt(C.diagonal()),para_list)
    
    return ste

#################################
"""
def Check_QN(F_LG,para,Data,cdt,stg,prior):
    
    para_list = stg['para_list']
    ste = Estimation_Error(F_LG,para,Data,cdt,stg,prior)
    a = np.arange(-1.0,1.1,0.2); L = np.zeros_like(a);
    
    for i,para_name in enumerate(para_list):
        
        plt.figure()
        for j in range(len(a)):
            para_tmp = para.copy()
            para_tmp[para_name] = para_tmp[para_name] + a[j]*ste[para_name]
            L[j] = Penalized_LG(F_LG,para_tmp,Data,cdt,prior,only_L=True)[0]
        
        plt.plot(para[para_name]+a*ste[para_name],L,'ko')
        plt.plot(para[para_name],L[5],'ro')
    
    return ste
"""

def Check_QN(F_LG,para,Data,cdt,stg,prior):
    
    para_list = stg['para_list']
    step_H = stg['para_step_H']
    a = np.arange(-1.0,1.1,0.2)
    L = np.zeros_like(a)
    
    for i,para_name in enumerate(para_list):
        
        plt.figure()
        for j in range(len(a)):
            para_tmp = para.copy()
            para_tmp[para_name] = para_tmp[para_name] + a[j]*para[para_name]*step_H[para_name]
            L[j] = Penalized_LG(F_LG,para_tmp,Data,cdt,prior,only_L=True)[0]
        
        plt.plot(para[para_name]+a*para[para_name]*step_H[para_name],L,'ko')
        plt.plot(para[para_name],L[5],'ro')
        plt.title(para_name)
    return []

#################################
##penalized likelihood
#################################
def Penalized_LG(F_LG,para,Data,cdt,prior,only_L=False):
    
    [L,G] = F_LG(para,Data,cdt,only_L=only_L)
    
    if len(prior)>0:
        
        ##Likelihood
        for i in range(len(prior)):
            [para_name,prior_type,mu,sigma] = prior.iloc[i][['name','type','mu','sigma']].values
            x = para[para_name]
            
            if prior_type == 'n': #prior: normal distribution
                L            = L - np.log(2.0*np.pi*sigma**2.0)/2.0 - (x-mu)**2.0/2.0/sigma**2.0
            elif prior_type ==  'ln': #prior: log-normal distribution
                L            = L - np.log(2.0*np.pi*sigma**2.0)/2.0 - np.log(x) - (np.log(x)-mu)**2.0/2.0/sigma**2.0
        
        ##Gradient
        if only_L == False:
            
            #prior
            for i in range(len(prior)):
                [para_name,prior_type,mu,sigma] = prior.iloc[i][['name','type','mu','sigma']].values
                x = para[para_name]
                
                if prior_type == 'n': #prior: normal distribution
                    G[para_name] = G[para_name] - (x-mu)/sigma**2.0
                elif prior_type ==  'ln': #prior: log-normal distribution
                    G[para_name] = G[para_name] - 1.0/x - (np.log(x)-mu)/sigma**2.0/x
            
            #fix
            para_fix = prior[prior['type']=='f']['name'].values.astype('S')
            G[para_fix] = 0
    
    return [L,G]

#################################
## Bayesian Smoothing
#################################
def Bayesian_Smoothing(Data,cdt,stg,prior=[],opt=[]):
    
    method = cdt['BS']['BasisFunc']['method']
    
    if method == 'CBS':
        BS_set_CBS(Data,cdt,stg)
    if method == 'QBS':
        BS_set_QBS(Data,cdt,stg)
    elif method == 'SSM':
        BS_set_SSM(Data,cdt,stg)
    
    [para,L,ste,G_norm] = Quasi_Newton(LG_ML,Data,cdt,stg,prior=prior,opt=opt)
    
    state = cdt['BS']['state']
    Find_MAP(para,state,Data,cdt)
    
    return [para,state,L,ste,G_norm]

def LG_ML(para,Data,cdt,only_L=False):
    state = cdt['BS']['state'].copy()
    [ML,L] = Marginal_L(para,state,Data,cdt)
    
    if only_L is True:
        G = [];
    else:
        G = G_nmrcl(para,state,Data,cdt)
        cdt['BS']['state'] = state
        cdt['L_model'] = cdt["BS"]["F_LGH"](para,state,Data,cdt)[0]
        
    return [ML,G]

def Marginal_L(para,state,Data,cdt):
    [L,G,H] = Find_MAP(para,state,Data,cdt)
    
    n = len(state)
    log_det  = cholesky(-spm.csc_matrix(H)).logdet() if spm.issparse(H) else np.log(np.diag(np.linalg.cholesky(-H))).sum()*2.0
    ML = L + np.log(2.0*np.pi)*n/2.0 - log_det/2.0
        
    return [ML,L]

def G_nmrcl(para,state,Data,cdt):
    
    G = pd.Series()
    para_list = cdt['BS']['para_list']
    para_exp  = cdt['BS']['para_exp']
    eps =       cdt['BS']['para_step_BS'].copy(); eps[para_exp] *= para[para_exp];
    
    for para_ix in para_list:
        
        """
        para_tmp = para.copy();  para_tmp[para_ix] = para_tmp[para_ix] - 2.0*eps[para_ix];  L1 = Marginal_L(para_tmp,state.copy(),Data,cdt)[0];
        para_tmp = para.copy();  para_tmp[para_ix] = para_tmp[para_ix] - 1.0*eps[para_ix];  L2 = Marginal_L(para_tmp,state.copy(),Data,cdt)[0];
        para_tmp = para.copy();  para_tmp[para_ix] = para_tmp[para_ix] + 1.0*eps[para_ix];  L3 = Marginal_L(para_tmp,state.copy(),Data,cdt)[0];
        para_tmp = para.copy();  para_tmp[para_ix] = para_tmp[para_ix] + 2.0*eps[para_ix];  L4 = Marginal_L(para_tmp,state.copy(),Data,cdt)[0];
        G[para_ix] = ( L1 - 8.0*L2 + 8.0*L3 - L4 )/12.0/eps[para_ix]
        """
        
        para_tmp = para.copy(); para_tmp[para_ix] = para_tmp[para_ix] - 1.0*eps[para_ix]; L1 = Marginal_L(para_tmp,state.copy(),Data,cdt)[0];
        para_tmp = para.copy(); para_tmp[para_ix] = para_tmp[para_ix] + 1.0*eps[para_ix]; L2 = Marginal_L(para_tmp,state.copy(),Data,cdt)[0];
        G[para_ix] = ( L2 - L1 )/2.0/eps[para_ix]
    
    return G
    

def Find_MAP(para,state,Data,cdt):
    
    BasisMat = cdt['BS']['BasisMat']
    
    while 1:
        [L,G,H]=LGH_posterior(para,state,Data,cdt)
        pd_H = check_positive_definiteness(-H)
        
        if cdt['BS']['print_FM'] is True:
            print '%.5f'%L,np.linalg.norm(G),pd_H
        
        if np.linalg.norm(G)<1e-5:
            #print "--"
            break
        
        if not pd_H:
            ##print "###",para["V"],pd_H
            para_tmp = para.copy()
            while not pd_H:
                para_tmp["V"] = para_tmp["V"]*0.8
                [_,_,H]=LGH_posterior(para_tmp,state,Data,cdt)
                pd_H = check_positive_definiteness(-H)
                ##print para_tmp["V"],pd_H
        
        
        d = -spla.spsolve(H,G) if spm.issparse(H) else -np.linalg.solve(H,G)
        d_max = np.abs(BasisMat.dot(d)).max()

        if d_max > 0.1:
            d = d/d_max*0.1
        
        alpha = 1.0
        
        while 1:
            state_tmp = state + alpha*d
            [L_tmp,_,_]=LGH_posterior(para,state_tmp,Data,cdt)
            #print ("#### L %.5f, L_tmp %.5f, alpha %e"%(L,L_tmp,alpha))
            
            if L-0.005 < L_tmp:
                state[:] = state_tmp
                break
            
            alpha *= 0.5
        
    return [L,G,H]

"""
def Find_MAP(para,state,Data,cdt):
    
    BasisMat = cdt['BS']['BasisMat']
    
    while 1:
        [L,G,H]=LGH_posterior(para,state,Data,cdt)
        pd_H = check_positive_definiteness(-H)
        
        if cdt['BS']['print_FM'] is True:
            print '%.5f'%L,np.linalg.norm(G),pd_H
        
        if np.linalg.norm(G)<1e-3:
            #print "--"
            break
        
        if pd_H:
            d = -spla.spsolve(H,G) if spm.issparse(H) else -np.linalg.solve(H,G)
        else:
            d = G

        d_max = np.abs(BasisMat.dot(d)).max()

        if d_max > 0.1:
            d = d/d_max*0.1
        
        alpha = 1.0
        
        while 1:
            state_tmp = state + alpha*d
            [L_tmp,_,_]=LGH_posterior(para,state_tmp,Data,cdt)
            #print ("#### L %.5f, L_tmp %.5f, alpha %e"%(L,L_tmp,alpha))
            
            if L-0.005 < L_tmp:
                state[:] = state_tmp
                break
            
            alpha *= 0.5
        
    return [L,G,H]
"""
        
def LGH_posterior(para,state,Data,cdt):
    
    F_LGH       = cdt['BS']['F_LGH']
    F_LGH_prior = cdt['BS']['F_LGH_prior']
    [L_m,G_m,H_m] = F_LGH(para,state,Data,cdt)
    [L_p,G_p,H_p] = F_LGH_prior(para,state,Data,cdt)
    
    L = L_m + L_p
    G = G_m + G_p
    H = H_m + H_p
    
    cdt['BS']['L_model'] = L_m
    
    return [L,G,H]

def check_positive_definiteness(A):
    
    try:
        if spm.issparse(A):
            cholesky(spm.csc_matrix(A)).L()
        else:
            np.linalg.cholesky(A)
        check_p = True
    except:
        check_p = False
    
    return check_p

###########State-Space model
"""
def BS_set_SSM(Data,cdt,stg_QN,stg_BS):
    
    BasisFunc = stg_BS['BasisFunc']
    n = BasisFunc['n']; state_ini = BasisFunc['state_ini']; V_ini = BasisFunc['V_ini'];
    
    [BasisMat,WeightMat,Rank_W] = BW_SSM_1st(n)
    state = np.ones(n)*state_ini
    sparse = True
    F_W = W_SSM
    
    cdt.update({'BasisMat':BasisMat, 'WeightMat':WeightMat, 'Rank_W':Rank_W, 'state':state, 'sparse':sparse, 'F_W':F_W})
    
    stg_QN['para_list']    = np.append(stg_QN['para_list'],'V')
    stg_QN['para_length']  = stg_QN['para_length'] + 1
    stg_QN['para_exp']     = np.append(stg_QN['para_exp'],'V')
    stg_QN['para_ini']     = stg_QN['para_ini'].append(pd.Series({'V':V_ini}))
    stg_QN['para_step_Q']  = stg_QN['para_step_Q'].append(pd.Series({'V':0.5}))
    stg_QN['para_step_H']  = stg_QN['para_step_H'].append(pd.Series({'V':0.05}))
    
    cdt.update({'para_list':stg_QN['para_list'],'para_exp':stg_QN['para_exp'],'para_step_BS':stg_QN['para_step_H']})

def W_SSM(para,state,Data,cdt):
    V = para['V']
    WeightMat = cdt['WeightMat']; Rank_W = cdt['Rank_W'];
    
    L_const = -Rank_W*np.log(2.0*np.pi*V)/2.
    W = WeightMat/V
    
    return [W,L_const]

def BW_SSM_1st(n):
    d0 = np.hstack([1.0,2.0*np.ones(n-2),1.0])
    d1 = -1.0*np.ones(n-1)
    
    data = np.array([d1,d0,d1])
    diags = np.arange(-1,2)
    W = spm.diags(data,diags,shape=(n,n),format='csc')
    
    rank_W = n-1
    
    B = spm.diags(np.ones(n),shape=(n,n))
    
    return [B,W,rank_W]

def BW_SSM_2nd(n):
    d0 = np.hstack(([1,5],6*np.ones(n-4),[5,1]))
    d1 = np.hstack((-2,-4*np.ones(n-3),-2))
    d2 = np.ones(n-2)
    
    data = np.array([d2,d1,d0,d1,d2])
    diags = np.arange(-2,3)
    W = spm.diags(data,diags,shape=(n,n),format='csc')
    
    rank_W = n-2
    
    B = spm.diags(np.ones(n),shape=(n,n))
    
    return [B,W,rank_W]
"""
###########Quadratic B-Spline
"""
def BS_set_QBS(Data,cdt,stg_QN,stg_BS):
    
    BasisFunc = stg_BS['BasisFunc']
    x = BasisFunc['x']; edge = BasisFunc['edge']; m = BasisFunc['m']; state_ini = BasisFunc['state_ini']; V_ini = BasisFunc['V_ini'];
    
    [BasisMat,WeightMat] = BW_QBS(x,edge,m)
    state = np.ones(m+2)*state_ini
    sparse = False
    F_LGH_prior = LGH_prior_QBS
    
    cdt.update({'BasisMat':BasisMat, 'WeightMat':WeightMat, 'm':m, 'state':state, 'sparse':sparse, 'F_LGH_prior':F_LGH_prior})
    
    stg_QN['para_list']    = np.append(stg_QN['para_list'],['state0','V'])
    stg_QN['para_length']  = stg_QN['para_length'] + 2
    stg_QN['para_exp']     = np.append(stg_QN['para_exp'],['V'])
    stg_QN['para_ini']     = stg_QN['para_ini'].append(pd.Series({'state0':state_ini,'V':V_ini}))
    stg_QN['para_step_Q']  = stg_QN['para_step_Q'].append(pd.Series({'state0':1.0,'V':1.0 }))
    stg_QN['para_step_H']  = stg_QN['para_step_H'].append(pd.Series({'state0':0.01,'V':0.01}))
    
    cdt.update({'para_list':stg_QN['para_list'],'para_exp':stg_QN['para_exp'],'para_step_BS':stg_QN['para_step_H']})

def Basis_QBS_i(x,edge_l,width):
    
    t1 = x[ ( edge_l+0.*width <= x ) & ( x < edge_l+1.*width) ]; t1 = ( t1 - (edge_l+0.*width) )/width
    t2 = x[ ( edge_l+1.*width <= x ) & ( x < edge_l+2.*width) ]; t2 = ( t2 - (edge_l+1.*width) )/width
    t3 = x[ ( edge_l+2.*width <= x ) & ( x < edge_l+3.*width) ]; t3 = ( t3 - (edge_l+2.*width) )/width
    
    r1 = (  1.*t1**2.             )  / 2.
    r2 = ( -2.*t2**2. + 2.*t2 + 1. ) / 2.
    r3 = (  1.*t3**2. - 2.*t3 + 1. ) / 2.

    r = np.zeros(len(x))
    r[ (edge_l<=x) & (x<edge_l+3.*width) ] = np.hstack([r1,r2,r3])
    
    return r

def d1_Basis_QBS_i(x,edge_l,width):
    
    t1 = x[ ( edge_l+0.*width <= x ) & ( x < edge_l+1.*width) ]; t1 = ( t1 - (edge_l+0.*width) )/width
    t2 = x[ ( edge_l+1.*width <= x ) & ( x < edge_l+2.*width) ]; t2 = ( t2 - (edge_l+1.*width) )/width
    t3 = x[ ( edge_l+2.*width <= x ) & ( x < edge_l+3.*width) ]; t3 = ( t3 - (edge_l+2.*width) )/width
    
    r1 = (  2.*t1      ) / 2.
    r2 = ( -4.*t2 + 2. ) / 2.
    r3 = (  2.*t3 - 2. ) / 2.

    r = np.zeros(len(x))
    r[ (edge_l<=x) & (x<edge_l+3.*width) ] = np.hstack([r1,r2,r3])
    
    return r

def BW_QBS(x,edge,m):
    
    x0 = edge[0]
    width = (edge[1]-edge[0])/m
    
    BasisMat  = [[] for i in range(m+2)]
    WeightMat = [[] for i in range(m+2)]
    
    for i in np.arange(-2,m):
        BasisMat[ i+2] =    Basis_QBS_i(x,x0+i*width,width)
        WeightMat[i+2] = d1_Basis_QBS_i(x,x0+i*width,width)

    BasisMat = np.vstack(BasisMat).T
    d1_BasisMat = np.vstack(WeightMat).T; WeightMat = d1_BasisMat.T.dot(d1_BasisMat);
    
    return [BasisMat,WeightMat]

def LGH_prior_QBS(para,state,Data,cdt):
    
    m = cdt['m']
    state0 = para['state0']
    V = para['V']
    WeightMat = cdt['WeightMat']
    
    W1 = WeightMat/V
    W2 = np.ones([m+2,m+2])/(m+2.)**2.0/0.01**2.0
    
    logdet_W = 2.*np.log(np.diag(np.linalg.cholesky(W1+W2))).sum()
    L_const = -(m+2.)*np.log(2.0*np.pi)/2. + logdet_W/2.
    
    L = L_const - state.dot(W1.dot(state))/2. - (state-state0).dot(W2.dot(state-state0))/2.
    G = - W1.dot(state)- W2.dot(state-state0)
    H = -W1-W2
    
    return [L,G,H]
"""
###########Cubic B-Spline
def BS_set_CBS(Data,cdt,stg):
    
    cdt_BS = cdt['BS']
    BasisFunc = cdt_BS['BasisFunc']
    x = BasisFunc['x']; edge = BasisFunc['edge']; m = BasisFunc['m']; state_ini = BasisFunc['state_ini']; state0_ini = BasisFunc['state0_ini']; V_ini = BasisFunc['V_ini']; epsilon = BasisFunc['epsilon']
    
    if state_ini is None:
        state_ini = np.ones(m)*state0_ini
    
    [BasisMat,WeightMat1,WeightMat2] = BW_CBS_sp(x,edge,m)
    F_LGH_prior = LGH_prior_CBS
    state = state_ini
    
    cdt_BS.update({'BasisMat':BasisMat, 'BasisMat_T':BasisMat.T, 'WeightMat1':WeightMat1, 'WeightMat2':WeightMat2, 'm':m, 'state':state, 'F_LGH_prior':F_LGH_prior, 'epsilon':epsilon})
    
    stg['para_list'] = np.append(stg['para_list'],['state0','V'])
    stg['para_exp'] = np.append(stg['para_exp'],['V'])
    stg['para_length']  = stg['para_length'] + 2
    stg['para_ini']     = stg['para_ini'].append(pd.Series({'state0':state0_ini,'V':V_ini}))
    stg['para_step_Q']  = stg['para_step_Q'].append(pd.Series({'state0':1.0,'V':1.0}))
    stg['para_step_H']  = stg['para_step_H'].append(pd.Series({'state0':0.02,'V':0.02}))
    
    cdt_BS.update({'para_list':stg['para_list'],'para_exp':stg['para_exp'],'para_step_BS':stg['para_step_H']})
    
"""
def Basis_CBS_i(x,edge_l,width):
    
    t1 = x[ ( edge_l+0.*width <= x ) & ( x < edge_l+1.*width) ]; t1 = ( t1 - (edge_l+0.*width) )/width
    t2 = x[ ( edge_l+1.*width <= x ) & ( x < edge_l+2.*width) ]; t2 = ( t2 - (edge_l+1.*width) )/width
    t3 = x[ ( edge_l+2.*width <= x ) & ( x < edge_l+3.*width) ]; t3 = ( t3 - (edge_l+2.*width) )/width
    t4 = x[ ( edge_l+3.*width <= x ) & ( x < edge_l+4.*width) ]; t4 = ( t4 - (edge_l+3.*width) )/width
    
    r1 = (      t1**3.                          ) /6.
    r2 = ( - 3.*t2**3. + 3.*t2**2. + 3.*t2 + 1. ) /6.
    r3 = (   3.*t3**3. - 6.*t3**2.         + 4. ) /6.
    r4 = ( - 1.*t4**3. + 3.*t4**2. - 3.*t4 + 1. ) /6.

    r = np.zeros(len(x))
    r[ (edge_l<=x) & (x<edge_l+4.*width) ] = np.hstack([r1,r2,r3,r4])
    
    return r

def d1_Basis_CBS_i(x,edge_l,width):
    
    t1 = x[ ( edge_l+0.*width <= x ) & ( x < edge_l+1.*width) ]; t1 = ( t1 - (edge_l+0.*width) )/width
    t2 = x[ ( edge_l+1.*width <= x ) & ( x < edge_l+2.*width) ]; t2 = ( t2 - (edge_l+1.*width) )/width
    t3 = x[ ( edge_l+2.*width <= x ) & ( x < edge_l+3.*width) ]; t3 = ( t3 - (edge_l+2.*width) )/width
    t4 = x[ ( edge_l+3.*width <= x ) & ( x < edge_l+4.*width) ]; t4 = ( t4 - (edge_l+3.*width) )/width

    r1 = (   3.*t1**2.               ) /6.
    r2 = ( - 9.*t2**2. +  6.*t2 + 3. ) /6.
    r3 = (   9.*t3**2. - 12.*t3      ) /6.
    r4 = ( - 3.*t4**2. +  6.*t4 - 3. ) /6.

    r = np.zeros(len(x))
    r[ (edge_l<=x) & (x<edge_l+4.*width) ] = np.hstack([r1,r2,r3,r4])
    r = r/width
    
    return r

def d2_Basis_CBS_i(x,edge_l,width):
    
    t1 = x[ ( edge_l+0.*width <= x ) & ( x < edge_l+1.*width) ]; t1 = ( t1 - (edge_l+0.*width) )/width
    t2 = x[ ( edge_l+1.*width <= x ) & ( x < edge_l+2.*width) ]; t2 = ( t2 - (edge_l+1.*width) )/width
    t3 = x[ ( edge_l+2.*width <= x ) & ( x < edge_l+3.*width) ]; t3 = ( t3 - (edge_l+2.*width) )/width
    t4 = x[ ( edge_l+3.*width <= x ) & ( x < edge_l+4.*width) ]; t4 = ( t4 - (edge_l+3.*width) )/width

    r1 = (    6.*t1       ) /6.
    r2 = ( - 18.*t2 +  6. ) /6.
    r3 = (   18.*t3 - 12. ) /6.
    r4 = ( -  6.*t4 +  6. ) /6.

    r = np.zeros(len(x))
    r[ (edge_l<=x) & (x<edge_l+4.*width) ] = np.hstack([r1,r2,r3,r4])
    r = r/width**2.0
    
    return r


def BW_CBS(x,edge,m):
    
    x0 = edge[0]
    width = (edge[1]-edge[0])/(m-3)
    
    BasisMat   = [[] for i in range(m)]
    WeightMat1 = [[] for i in range(m)]
    WeightMat2 = [[] for i in range(m)]
    
    for i in np.arange(-3,m-3):
        BasisMat[  i+3] =   Basis_CBS_i( x,x0+i*width,width)
        WeightMat1[i+3] = d1_Basis_CBS_i(x,x0+i*width,width)
        WeightMat2[i+3] = d2_Basis_CBS_i(x,x0+i*width,width)

    BasisMat = np.vstack(BasisMat).T
    d1_BasisMat = np.vstack(WeightMat1).T; WeightMat1 = d1_BasisMat.T.dot(d1_BasisMat)
    d2_BasisMat = np.vstack(WeightMat2).T; WeightMat2 = d2_BasisMat.T.dot(d2_BasisMat)
    
    return [BasisMat,WeightMat1,WeightMat2]
"""

def Basis_CBS_i_sp(x,edge_l,width):
    
    t1 = x[ ( edge_l+0.*width <= x ) & ( x < edge_l+1.*width) ]; t1 = ( t1 - (edge_l+0.*width) )/width
    t2 = x[ ( edge_l+1.*width <= x ) & ( x < edge_l+2.*width) ]; t2 = ( t2 - (edge_l+1.*width) )/width
    t3 = x[ ( edge_l+2.*width <= x ) & ( x < edge_l+3.*width) ]; t3 = ( t3 - (edge_l+2.*width) )/width
    t4 = x[ ( edge_l+3.*width <= x ) & ( x < edge_l+4.*width) ]; t4 = ( t4 - (edge_l+3.*width) )/width
    
    r1 = (      t1**3.                          ) /6.
    r2 = ( - 3.*t2**3. + 3.*t2**2. + 3.*t2 + 1. ) /6.
    r3 = (   3.*t3**3. - 6.*t3**2.         + 4. ) /6.
    r4 = ( - 1.*t4**3. + 3.*t4**2. - 3.*t4 + 1. ) /6.


    index = np.nonzero( (edge_l<=x) & (x<edge_l+4.*width) )[0]
    r = np.hstack([r1,r2,r3,r4])
    
    return [index,r]

def d1_Basis_CBS_i_sp(x,edge_l,width):
    
    t1 = x[ ( edge_l+0.*width <= x ) & ( x < edge_l+1.*width) ]; t1 = ( t1 - (edge_l+0.*width) )/width
    t2 = x[ ( edge_l+1.*width <= x ) & ( x < edge_l+2.*width) ]; t2 = ( t2 - (edge_l+1.*width) )/width
    t3 = x[ ( edge_l+2.*width <= x ) & ( x < edge_l+3.*width) ]; t3 = ( t3 - (edge_l+2.*width) )/width
    t4 = x[ ( edge_l+3.*width <= x ) & ( x < edge_l+4.*width) ]; t4 = ( t4 - (edge_l+3.*width) )/width

    r1 = (   3.*t1**2.               ) /6.
    r2 = ( - 9.*t2**2. +  6.*t2 + 3. ) /6.
    r3 = (   9.*t3**2. - 12.*t3      ) /6.
    r4 = ( - 3.*t4**2. +  6.*t4 - 3. ) /6.

    index = np.nonzero( (edge_l<=x) & (x<edge_l+4.*width) )[0]
    r = np.hstack([r1,r2,r3,r4])
    r = r/width
    
    return [index,r]

def d2_Basis_CBS_i_sp(x,edge_l,width):
    
    t1 = x[ ( edge_l+0.*width <= x ) & ( x < edge_l+1.*width) ]; t1 = ( t1 - (edge_l+0.*width) )/width
    t2 = x[ ( edge_l+1.*width <= x ) & ( x < edge_l+2.*width) ]; t2 = ( t2 - (edge_l+1.*width) )/width
    t3 = x[ ( edge_l+2.*width <= x ) & ( x < edge_l+3.*width) ]; t3 = ( t3 - (edge_l+2.*width) )/width
    t4 = x[ ( edge_l+3.*width <= x ) & ( x < edge_l+4.*width) ]; t4 = ( t4 - (edge_l+3.*width) )/width

    r1 = (    6.*t1       ) /6.
    r2 = ( - 18.*t2 +  6. ) /6.
    r3 = (   18.*t3 - 12. ) /6.
    r4 = ( -  6.*t4 +  6. ) /6.

    index = np.nonzero( (edge_l<=x) & (x<edge_l+4.*width) )[0]
    r = np.hstack([r1,r2,r3,r4])
    r = r/(width**2.0)
    
    return [index,r]

def BW_CBS_sp(x,edge,m):
    
    x0 = edge[0]
    width = (edge[1]-edge[0])/(m-3)
    
    BasisMat_d = [];   BasisMat_r = [];   BasisMat_c = [];
    WeightMat1_d = []; WeightMat1_r = []; WeightMat1_c = [];
    WeightMat2_d = []; WeightMat2_r = []; WeightMat2_c = [];
    
    
    for i in np.arange(-3,m-3):
        
        [index,data] = Basis_CBS_i_sp(x,x0+i*width,width)
        BasisMat_d.append(data);
        BasisMat_c.append((i+3)*np.ones(len(data),dtype=np.int64));
        BasisMat_r.append(index); 
        
        [index,data] = d1_Basis_CBS_i_sp(x,x0+i*width,width)
        WeightMat1_d.append(data);
        WeightMat1_c.append((i+3)*np.ones(len(data),dtype=np.int64));
        WeightMat1_r.append(index); 
        
        [index,data] = d2_Basis_CBS_i_sp(x,x0+i*width,width)
        WeightMat2_d.append(data);
        WeightMat2_c.append((i+3)*np.ones(len(data),dtype=np.int64));
        WeightMat2_r.append(index); 
        
    BasisMat =  spm.csc_matrix((np.hstack(BasisMat_d),(np.hstack(BasisMat_r),np.hstack(BasisMat_c))),shape=[len(x),m])
    d1_BasisMat =  spm.csc_matrix((np.hstack(WeightMat1_d),(np.hstack(WeightMat1_r),np.hstack(WeightMat1_c))),shape=[len(x),m])
    d2_BasisMat =  spm.csc_matrix((np.hstack(WeightMat2_d),(np.hstack(WeightMat2_r),np.hstack(WeightMat2_c))),shape=[len(x),m])
    
    WeightMat1 = d1_BasisMat.T.dot(d1_BasisMat)
    WeightMat2 = d2_BasisMat.T.dot(d2_BasisMat)
    
    return [BasisMat,WeightMat1,WeightMat2]

def LGH_prior_CBS(para,state,Data,cdt):
    state0 = para['state0']; V = para['V'];
    WeightMat1 = cdt['BS']['WeightMat1']; WeightMat2 = cdt['BS']['WeightMat2']; epsilon = cdt['BS']['epsilon']
    m = cdt['BS']['m']
    
    W1 = WeightMat1/V
    W2 = np.ones([m,m],dtype='f8')/m**2.0/epsilon**2.0
    
    logdet_W = 2.*np.log(np.diag(np.linalg.cholesky(W1+W2))).sum()
    L_const = -m*np.log(2.0*np.pi)/2. + logdet_W/2.
    
    L = L_const - state.dot(W1.dot(state))/2. - (state-state0).dot(W2.dot(state-state0))/2.
    G = - W1.dot(state)- W2.dot(state-state0)
    H = -W1-W2
       
    return [L,G,H]
"""
def LGH_prior_CBS(para,state,Data,cdt):
    state0 = para['state0']; V1 = para['V1']; V2 = para['V2']
    WeightMat1 = cdt['BS']['WeightMat1']; WeightMat2 = cdt['BS']['WeightMat2'];
    m = cdt['BS']['m']
    
    W1 = WeightMat1/V1 + WeightMat2/V2
    W2 = np.ones([m,m],dtype='f8')/m**2.0/0.01**2.0
    
    logdet_W = 2.*np.log(np.diag(np.linalg.cholesky(W1+W2))).sum()
    L_const = -m*np.log(2.0*np.pi)/2. + logdet_W/2.
    
    L = L_const - state.dot(W1.dot(state))/2. - (state-state0).dot(W2.dot(state-state0))/2.
    G = - W1.dot(state)- W2.dot(state-state0)
    H = -W1-W2
    
    return [L,G,H]
"""
