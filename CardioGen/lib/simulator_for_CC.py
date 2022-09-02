# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 08:03:15 2020

@author: agarwal.270a
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import signal as sig
from pathlib import Path


class Simulator(object):
    
    def __init__(self,input_list=[],output_list=[]):
        '''
        in/output format: {'name':,'data':}
        '''
        self.input=input_list
        self.output=output_list
    
    def sliding_window_fragmentation(self,tseries,win_size,step_size,axes=None):
        '''
        sliding_window_fragmentation along the axis dimension
        for eg.:
        tseries=list of (numpy array of shape [n_tsteps,vector_dim_at_every_tstep])
        '''
        assert type(tseries)==type([]), 'Input time-series should be a list of numpy arrays'
        if axes is None:
            axes=[0 for _ in range(len(tseries))]
        tot_len=tseries[0].shape[axes[0]]
        indices=np.stack([np.arange(i,i+(tot_len-win_size+1),step_size)
                            for i in range(win_size)],axis=1)
        #slid_arr=np.stack([tseries[i:i+(tot_len-win_size+1):step_size8]
        #                    for i in range(win_size)],axis=1)
        slid_tseries=[np.take(tseries[j],indices,axis=axes[j]) 
                        for j in range(len(tseries))]
        
        if len(slid_tseries)==1:
            return slid_tseries[0]
        else:
            return slid_tseries

    def sliding_window_defragmentation(self,tseries,win_size,step_size): 
        '''
        sliding_window_defragmentation
        for eg.:
        tseries=list of (numpy array of shape [n_wins,n_tsteps,vector_dim_at_every_tstep])
        '''
        assert type(tseries)==type([]), 'Input time-series should be a list of numpy arrays'
        assert win_size==tseries[0].shape[1], 'Incorrect window size of input tseries'
        defrag_tseries=[]
        for j in range(len(tseries)):
            if len(tseries[j].shape)==2: tseries[j]=np.expand_dims(tseries[j],-1)
            defrag_tseries.append(np.concatenate([tseries[j][0,...],
            tseries[j][1:,-step_size:,...].reshape([-1,*tseries[j].shape[2:]])],
            axis=0))
        
        if len(defrag_tseries)==1:
            return defrag_tseries[0]
        else:
            return defrag_tseries
    
    # Common helper functions in simulators
    def get_windows_at_peaks(self,pk_locs,y,w_pk=25,w_l=10,make_plots=False):
        '''
        find and store peaks from signal y, centered at r_pk_locs
        pk_locs: R-peak locations
        y:signal to window
        w_pk= window length
        w_l= no.of samples in the window, to the left of peak.
        
        ?????????????????????????????????
        Should I take pk shapes looking from ECG perspective or correct for PTT?
        Right now I take from ECG persepective so that after PCA, the components
        will have implicit PTT.
        '''
        w_r=w_pk-w_l-1
        #remove terminal pk_locs
        #pk_locs=pk_locs[(pk_locs>=w_l) & (pk_locs<(len(y)-w_r))]
        
        # add peaks
        windowsPPG1=[y[pk_locs[i]-w_l:pk_locs[i]+w_r+1] \
                     for i in range(len(pk_locs))]
        windowsPPG1=np.array(windowsPPG1)
        
        if make_plots:
            plt.close('all');j=0
            while j<len(windowsPPG1)/20:
                plt.figure();k=1
                while ((k<=8) and (j+k)<len(windowsPPG1)):
                    plt.subplot(4,2,k)
                    cntr=j+k;
                    plt.plot(windowsPPG1[cntr],'r')
                    plt.grid(True)
                    plt.title('Peak '.format(cntr))
                    k=k+1;
                j=j+8;
        assert len(windowsPPG1.shape)==2
        return windowsPPG1
    
    def pca(self,X):
        '''
        find PCA of X. Assumes X is real valued.
        '''
        n, m = X.shape
        avg=X.mean(axis=0)
        X=X-avg
        assert np.allclose(X.mean(axis=0), np.zeros(m))
        # Compute covariance matrix
        C = np.matmul(X.T, X) / (n-1)
        #np.allclose(C, C.T, rtol=1e-5, atol=1e-8)
        #print(np.array_equal(C, C.T))
        # Eigen decomposition
        eigen_vals, eigen_vecs = np.linalg.eigh(C)
        sort_idx=np.argsort(-eigen_vals) #-ve to sort in descending order
        eigen_vals, eigen_vecs=eigen_vals[sort_idx], eigen_vecs[:,sort_idx]
        return eigen_vals,eigen_vecs,avg

    def remove_outliers(self,A):
        '''
        A: a matrix made of row vectors
        naively remove outlier vectors by remove top 5 percentile in magnitude
        '''
        mags=np.linalg.norm(A,axis=1)
        thres=np.nanpercentile(mags,95)
        idx=np.sort(np.arange(len(A))[mags<thres])
        return A[idx]


class Ecg2Ppg_Simulator(Simulator):
    def __init__(self,input_list,output_list,w_pk=25,w_l=10,P_ID='',path='./',
                 ppg_color='green'):
        '''
        in/output format: list of numpy arrays
        '''
        super(Ecg2Ppg_Simulator,self).__init__(input_list,output_list)
        self.w_pk=w_pk
        self.w_l=w_l
        self.w_r=w_pk-w_l-1
        self.P_ID=P_ID
        self.path=path
        self.ppg_color=ppg_color
        
        # Find basis_dict as apt basis to randomly generate PPG peaks, arranged
        # in decreasing order of prominence
        self.basis_path=path+"{}_{}_ppg_basis.mat".format(P_ID,ppg_color)
        if Path(self.basis_path).is_file():
            self.basis_dict = io.loadmat(self.basis_path)
        else:
            self.regen_basis()
        
    
    def ppg_filter(self,X0,Fs=25,filt=True):
        '''
        Band-pass filter multi-channel PPG signal X0
        '''
        nyq=Fs/2
        X1 = sig.detrend(X0,type='constant',axis=0); # Subtract mean
        if filt:
            b = sig.firls(219,np.array([0,0.3,0.5,4.5,5,nyq]),
                          np.array([0,0,1,1,0,0]),np.array([10,1,1]),nyq=nyq)
            X=np.zeros(X1.shape)
            for i in range(X1.shape[1]):
                #X[:,i] = sig.convolve(X1[:,i],b,mode='same'); # filtering using convolution, mode='same' returns the 'centered signal without any delay
                X[:,i] = sig.filtfilt(b,[1],X1[:,i])
    
        else:
            X=X1
        return X
    
    def regen_basis(self,save_flag=False):
        '''
        Regenerate the basis
        '''
        self.basis_dict=self.extract_components(self.input,self.output,
                                                path=self.path,
                                                save_flag=save_flag,
                                                ppg_color=self.ppg_color)
        return
        
    def extract_components(self,input_list,output_list,path4basis='',
                           save_flag=False,make_plots=True,ppg_color='green'):
        '''
        extract templates from output sample_data using input sample_data and 
        find a template basis using PCA
        '''
        list_r_pk_locs=[np.arange(len(arr_pks))[arr_pks.astype(bool)] for 
                        arr_pks in input_list]
        #get nearest dsampled idx
        list_r_pk_locs_dsampled=[np.floor(r_pk_locs/4).astype(int) for 
                                 r_pk_locs in list_r_pk_locs]
        list_wins_clean_ppg=[self.get_windows_at_peaks(r_pk_locs,y,
                                w_pk=self.w_pk,w_l=self.w_l) for r_pk_locs,y in
                            zip(list_r_pk_locs_dsampled, output_list)]
        wins_clean_ppg=np.concatenate(list_wins_clean_ppg, axis=0)
        mat1=self.remove_outliers(wins_clean_ppg)
        #find eig values and observe
        eigen_vals1,eigen_vecs1,avg1=self.pca(mat1) 
        
        if make_plots:
            plt.figure();plt.subplot(121);plt.plot(eigen_vals1)
            plt.subplot(122);plt.plot(avg1)
            j=0
            while j<15:
                plt.figure();k=1
                while ((k<=8) and (j+k)<len(eigen_vals1)):
                    plt.subplot(4,2,k)
                    cntr=j+k;
                    plt.plot(eigen_vecs1[:,cntr],'r')
                    plt.grid(True)
                    plt.title('eig_peak '.format(cntr))
                    k=k+1;
                j=j+8;
                
        #store basis
        basis_dict={'eig_val':eigen_vals1,'eig_vec':eigen_vecs1,'mean':avg1}
        if save_flag:
            io.savemat(path4basis,mdict=basis_dict)
            
        return basis_dict
    
    def __call__(self,arr_pks,k=10):
        '''
        arr_pks: Location of R-peaks determined from ECG
        k: No. of prominent basis to keep
        '''
        r_pk_locs=np.arange(len(arr_pks))[arr_pks.astype(bool)]
        r_pk_locs=np.floor(r_pk_locs/4).astype(int) #get nearest dsampled idx
        #remove terminal pk_locs
        r_pk_locs=r_pk_locs[(r_pk_locs>=self.w_l) & 
                            (r_pk_locs<=(int(len(arr_pks)/4)-self.w_r-1))]
        
        n_peaks=int(len(arr_pks)/(4*5))
        #sample bunch of peaks using PCA components
        eig_vec=self.basis_dict['eig_vec']
        eig_val=self.basis_dict['eig_val'].reshape((-1,1))
        avg=self.basis_dict['mean'].reshape((-1,1))
        eig_vec=eig_vec[:,:k];eig_val=eig_val[:k]
        l_peaks,n_coeff=eig_vec.shape
        weights=np.random.random_sample((n_coeff,n_peaks))*(eig_val**0.5)
        rand_pks=np.matmul(eig_vec,weights)+avg #form peaks
        
        #construct dsampled arr_pks
        arr_pks_dsampled=np.zeros(int(len(arr_pks)/4))
        arr_pks_dsampled[r_pk_locs]=1
        #construct arr_ppg
        arr_ppg=np.zeros(int(len(arr_pks)/4))

        
        #arr_pk=np.zeros(len(HR_curve1))
        #TODO: bunch of changes here
        #gauss=norm(loc = 0., scale = 1.5).pdf(np.arange(-3,3+1))
        #PTT=np.random.randint(4,8) #sample a PTT value
        #plt.figure();plt.plot(gauss)
        #print(np.max(r_pk_locs),len(arr_ppg))
        
        #Place sampled ppg peaks
        for i in range(len(r_pk_locs)):
            arr_ppg[r_pk_locs[i]-self.w_l:r_pk_locs[i]+self.w_r+1]+=\
                                                                rand_pks[:,i]
                                                                
        arr_ppg_filt=arr_ppg*1
        #TODO: Better stitching needed than simple LPF
        #arr_ppg_filt=self.ppg_filter(arr_ppg.reshape(-1,1))
        #plt.figure();plt.plot(arr_ppg);plt.plot(arr_ppg_filt,'g--')
        #plt.plot(r_pk_locs,arr_ppg[r_pk_locs],'r+')
        return arr_ppg_filt.reshape(-1),arr_pks_dsampled

    
#%%
#Sample Client
if __name__=='__main__':
    # Data Helpers
    import glob
    import pandas as pd
    def get_train_data(path,val_files=[],test_files=[]):
        '''
        Use all files in the folder 'path' except the val_files and test_files
        '''
        def get_clean_ppg_and_ecg(files):
            '''
            
            '''
            list_clean_ppg=[];list_arr_pks=[]
            for i in range(len(files)):
                df=pd.read_csv(files[i],header=None)
                arr=df.values
                if 'clean' in files[i]:
                    list_clean_ppg+=[arr[:,29],arr[:,30],arr[:,39],arr[:,40]]
                    list_arr_pks+=4*[arr[:,45:49].reshape(-1)]    
            return list_arr_pks,list_clean_ppg
        files=glob.glob(path+'*.csv')
        #files=[fil for fil in files if 'WZ' in fil] #get wenxiao's data
        #separate val and test files
        s3=set(files);s4=set(val_files+test_files)
        files_2=list(s3.difference(s4))
        #files_2=[fil for fil in files if not((val_names[0] in fil))]
        list_arr_pks,list_clean_ppg=get_clean_ppg_and_ecg(files_2)
        return list_arr_pks,list_clean_ppg
    
    def get_test_data(file_path):
        df=pd.read_csv(file_path,header=None)
        arr=df.values
        test_out_for_check=[arr[:,29],arr[:,30],arr[:,39],arr[:,40]]
        test_in=arr[:,45:49].reshape(-1)
        return test_in,test_out_for_check
    
    #Get Train Data for simulator
    plt.close('all')
    path_prefix= 'E:/Box Sync/' #'C:/Users/agarwal.270/Box/' #
    path=(path_prefix+'AU19/Research/PPG_ECG_proj/data/Wen_data_28_Sep/'
          'clean_lrsynced\\')
    val_files=[path+'2019092801_3154_clean.csv']
    test_files=[path+'2019092820_5701_clean.csv']
    input_list,output_list=get_train_data(path,val_files,test_files)
    
    #Create Simulator using train data
    sim=Ecg2Ppg_Simulator(input_list,output_list,P_ID='W',
            path='E:/Box Sync/SP20/Research/PPG_ECG_proj/simulator_CC/data/')
    
    #Use simulator to produce synthetic output given input
    test_in,test_out_for_check=get_test_data(val_files[0])
    synth_ppg_out,test_in_dsampled=sim(test_in)
    
    #Visualize
    plt.figure()
    plt.plot(test_out_for_check[0])
    plt.plot(synth_ppg_out)
    plt.plot(test_in_dsampled)
    plt.legend(['True','Synthetic','R-peaks'])
    plt.grid(True)