## This program extracts speed up sensor data and selects features using k best fit, corelation and pca ##
import datetime as dt
import csv 
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import os
import tsfel
from math import log, e
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.decomposition import PCA
 
flag=0
base=None

def feature_extract(df,f_s,N):
    ''' Extract multiple features for each input dataframe 
    Args:
        df: input dataframe dataframe
        f_s: sampling frequency
        N: total number of sample
    Returns:
        extracted features in a csv file
    '''
    
    S_z= df.iloc[0:N,1].values
    f_name=df.iat[1,0]
    
    
    S=S_z
    f_values = np.linspace(0.0, f_s/16, N//2)
    fft_values = fft(S)
    fft_mag_values = 2.0/N * fft_values[0:N//2]
    mean_val=np.mean(np.abs(S))
    p2p=np.abs(np.max(S)) + np.abs(np.min(S)) # maxvalue−minvalue
    
    mY = np.abs(fft_mag_values) 
    peak_list=np.sort(mY)[::-1]
    top_3=0
    for i in range(3): #sum of top 3 peak
        top_3+=peak_list[i]
    
    peakY = np.max(mY) # max peak
    locY = np.argmax(mY) # peak's location
    frqY = f_values[locY] #frequency of highest location
    standard_dev=np.std(np.abs(fft_mag_values))
    vc = pd.Series(S).value_counts(normalize=True, sort=False)
    entr= -(vc * np.log(vc)/np.log(e)).sum()
    rms=np.sqrt((S**2).sum() / len(S))
    p2p=np.abs(np.max(S)) + np.abs(np.min(S)) # maxvalue−minvalue
    skew = stats.skew(S,bias=True)
    kurtosis = stats.kurtosis(S,bias=True)
    
    cfg = tsfel.get_features_by_domain()
    X = tsfel.time_series_features_extractor(cfg, S)
    X = X.assign(file_name=f_name,highest_peak=peakY, max_frequency=frqY, mean=mean_val,standard_deviation=standard_dev, rms_val=rms, top3_peak=top_3,entropy=entr,kurtosis_val=kurtosis,skewness=skew,peak_to_peak=p2p)
    global flag
    
    if flag==0:
        X.to_csv("nasa_feature2.csv", mode='a', index=False)
        flag=1
    
    X.to_csv("nasa_feature2.csv", mode='a', header=False, index=False)

    
def feature_selection(filename):
    ''' Collect the set of suitable features from the extracted features set 
    Uses combine result from k best feature selection, Correlation-Matrix and PCA for selecting relevant features
    Args:
        Csv file which contains set of all extracted features
    Returns:
        Csv file which contains set of all relevat features         
    '''
    x = pd.read_csv(filename)
    lent=len(x)
    diff=0
    for d in range (0,lent):
        if d == 0:
            x.at[d, "X Diff"] = 0
            tm1=x.at[d,'file_name']
            tm1=tm1[-8:].replace(".",":")
        else:
            tm=x.at[d,'file_name']
            tm=tm[-8:].replace(".",":")
            #print(tm)
            startValue = pd.to_datetime(tm1) #pd.to_datetime(x.iloc[d-1]['file_name'])
            endValue = pd.to_datetime(tm) #pd.to_datetime(x.iloc[d]['file_name'])
    
            diff= diff+abs((endValue - startValue).total_seconds())
            x.at[d,"X Diff"]=diff
            tm1=tm
        
    #print(lent,x['X Diff'],x)
    


    X= x.drop(columns=["peak_to_peak","file_name"])
    y= x['peak_to_peak']

    # k best feature selection
    k_lim=10
    uni = SelectKBest(score_func = f_classif, k = k_lim)
    fit = uni.fit(X, y)
    k_best=X.columns[fit.get_support(indices=True)].tolist()
    df = pd.DataFrame() 
    for d in range (0,k_lim):
        df.insert(loc=d, column=k_best[d], value=x[k_best[d]])
    df.insert(loc=k_lim, column='peak_to_peak', value=x['peak_to_peak'])
    df.insert(loc=k_lim+1, column='time_diff', value=x['X Diff'])
    #df.insert(loc=0, column='file_name', value=x['file_name'])
    print(df)
    # Correlation-Matrix
    '''df1 = pd.read_csv(filename)
    df1= df1.drop(columns=["file_name"])
    corr = df.corr()
    c1 = corr.abs().unstack()
    c1.sort_values(ascending = False)
    corr=df1.corrwith(df1['peak_to_peak'])
    corr = corr[corr >= 0.96].index
    for i in range(0,len(corr)):
        if corr[i] not in df.columns:
            df.insert(loc=k_lim+1+i, column=corr[i], value=x[corr[i]])'''

    # Pca  
    pca = PCA(n_components=3).fit(X)
    explained_variance = pca.explained_variance_ratio_
    columns = ['pca_comp_%i' % i for i in range(3)]
    df_pca  = pd.DataFrame(pca.transform(X), columns=columns, index=X.index)
    #print(df_pca,explained_variance)
    result = pd.concat([df, df_pca], axis=1, join='inner')
    print(result)
    result.to_csv("nasa_selected_feature2.csv", mode='a', index=False)


def dataframe_convert(filepath,dir_list,txt_list):
    ''' Pre process all the text files from each of the folders which will used later for feature extraction.
    Args:
        filepath: relative path of the application
        dir_list: list of all directories 
        txt_list: list of all text files 
    Returns:
        store sensor data in a panda dataframe
    '''
    print(filepath+"/"+dir_list+"/"+txt_list,"***")
    file1=open(filepath+"/"+dir_list+"/"+txt_list, 'r') 
    ct=0
    word_list=[]
    num_line = sum(1 for line in open(filepath+"/"+dir_list+"/"+txt_list))
    rows, cols = (num_line, 5) 
    arr = [["" for i in range(cols)] for j in range(rows)] 
    global flg
    flg=0

    for line in file1:
        
        word_list = line.split("\t")
        met1=len(word_list)
        k=0
        arr[ct][k]=txt_list
        time_indx=arr[ct][k].split("_")
        for k in range(1,met1+1):
            '''
                Some pre processing has been done e,g. removing \n from end every line, 
                convert sensor values 4,9 to 4.9 for later calculation, 
                store the time as HH:MM:SS from filename
            '''
            word_list[k-1]=word_list[k-1].replace("\n","")
            arr[ct][k]=word_list[k-1]  
            arr[ct][k]=float(arr[ct][k].replace("\t",""))
            #arr[ct][k+1]=time_indx[1].replace("-",":")
        ct += 1
    df = pd.DataFrame.from_records(arr,columns=['file_name','ax','bx','cx','dx'])
    #print(df)
    f_s= 20000
    num_sample=20480
    #df=df.head(num_sample)
    feature_extract(df,f_s,num_sample)

if __name__ == '__main__':
    filepath = os.path.dirname(__file__)
    dir_list = []
    txt_list = []
    directory_list = list()
    
    for root, dirs, files in os.walk(filepath):
        for dir in dirs:
            dir_list.append(dir) #stores the name of all the folders in a list
        for name in files:
            txt_list.append(name) 
    
    for i in range(2,len(txt_list)):
        dataframe_convert(filepath,dir_list[0],txt_list[i])
    feature_selection(filepath+'/nasa_feature2.csv')
    