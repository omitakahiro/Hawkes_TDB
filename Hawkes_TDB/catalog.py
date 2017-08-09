import sys,time,datetime,copy,subprocess,itertools,pickle,warnings

import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

class Data_Fin(pd.core.frame.DataFrame):
    
    _metadata = ['t','dt']
    
    @property
    def _constructor(self):
        return Data_Fin
    
    def extract_PC(self,dt,unit,jitter=False):
        return extract_PC(self,dt,unit,jitter=jitter)
    
    def plot_PC(self,dt=None):
        plot_PC(self,dt=dt)

###############################################################################################
##load pickle
###############################################################################################
def load_pkl(fn):
    Data = Data_Fin( pd.read_pickle(fn) )
    return Data

###############################################################################################
##price change
###############################################################################################
def extract_PC(Data,dt,unit,jitter=None):
    
    st = dt['st']; en = dt['en']
    df = Data[st:en].copy()
    
    df.dt = dt
    df.t  = {'st':0.0,'en':(en-st).total_seconds()}
    
    if not df.empty:
        
        df['d_PRICE'] = np.append(0, df['PRICE'][1:].values - df['PRICE'][:-1].values )
        df = df[ df['d_PRICE'] != 0 ]
        df['d2_PRICE'] = np.append(0, np.sign(df['d_PRICE'][1:].values) + np.sign(df['d_PRICE'][:-1].values) )
        index = (np.abs(df['d_PRICE'])> unit) | ( df['d2_PRICE'] != 0)
        df = df[index]
        df = df[['PRICE','d_PRICE','VOL','TYPE','CM']]
        df['T'] = (df.index - st).total_seconds()
        
        if jitter:
            T_jit = df['T'].values + jitter*( np.random.rand(len(df.index)) - 0.5 );
            T_jit[ T_jit < df.t['st'] ] += jitter*0.5
            T_jit[ T_jit > df.t['en'] ] -= jitter*0.5
            df['T'] = np.sort(T_jit)
            
            #if ( (df['T'] < df.t['st']) & (df.t['en'] < df['T']) ).sum() > 0:
                #sys.exit( "error in extract pc" )
            
    
    
    return df

def plot_PC(Data,dt=None):
    
    if dt is not None:
        Data = Data[dt['st']:dt['en']]
    
    plt.figure(figsize=(5,10))
    
    plt.subplot(2,1,1)
    plt.plot(Data['PRICE'],'k-')
    plt.xticks([])
    
    plt.subplot(2,1,2)
    plt.plot(np.exp(Data['PRICE']*0).cumsum(),'r-')
    plt.xticks(rotation=90)


###############################################################################################
## Graph
###############################################################################################    
def Graph_PN_M(Data,dt):

    fig_id = 1
    dt_i = dt

    plt.figure(figsize=(11.69,8.27))
    mpl.rc('font', size=6, family='Arial')
    mpl.rc('axes',titlesize=6)
    mpl.rc('pdf',fonttype=42)

    while 1:

        [y,m,d] = [dt_i.year,dt_i.month,dt_i.day]
        st = datetime.datetime(y,m,d,8,0,0); en = datetime.datetime(y,m,d,15,10,0);
        df = Data[st:en]
        df_pc = extract_PC(df,{'st':st,'en':en},5)
        
        if not df.empty:
            
            y_min = df['PRICE'].min(); y_max = df['PRICE'].max();
            y0 = (y_min+y_max)/2.0 - 1000; y1 = (y_min+y_max)/2.0 + 1000; 
            
            T = df_pc['T'].values
            df_pc['dT50'] = np.append(T[100:]-T[:-100],np.nan*np.empty(100))
            
            
            plt.subplot(5,5,fig_id)
            plt.plot(df['PRICE'],'ko',markersize=0.5)
            plt.plot((y1-50)*df_pc.query('dT50<60')['PRICE']**0,'ro')
            plt.title('%d/%d/%d'%(y,m,d),fontsize=8)
            plt.ylim([y0,y1])
            plt.xticks([])
            #plt.yticks([])

            ax2=plt.gca().twinx()
            ax2.plot(np.exp(0*df['PRICE']).cumsum(),'b-')
            ax2.plot(np.exp(0*df_pc['PRICE']).cumsum()*10,'r-')
            ax2.tick_params(axis='y', colors='b')
            ax2.set_ylim([0,200000])
            ax2.set_yticks(np.arange(0,2.5e+5,5e4))
            #ax2.set_yticklabels([])

            fig_id += 1

        dt_i += datetime.timedelta(days=1)

        if dt_i.month != dt.month:
            break

    plt.tight_layout()
    plt.savefig('PN%d%d.png'%(dt.year,dt.month),dpi=180,bbox_inches='tight',pad_inches=0.05)
