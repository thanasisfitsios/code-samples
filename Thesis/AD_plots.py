# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:39:22 2019

@author: AthanasiosFitsios
"""
#import matplotlib
#matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
import pickle
import bottleneck as bn
from sklearn.preprocessing import MinMaxScaler
###################################################################################################
#In the code below, I plot the reconstruction error of my CNN+RNN autoencoder model.
# Then I apply the 2-sigma rule as a heuristic for anomaly detection.
#I plot the error and the AD result in a figure.
# In the final output, I get the start,end of the anomalies and the top-k culprit KPIs and save them in a json.
# 
###################################################################################################


#open a pickle output file (generated by the Enc-Dec model), which contains the input tensor and the reconstructed output tensor
with open('filename.pkl', 'rb') as f:
    my_dict = pickle.load(f)

real = my_dict['in']  #input tensor
pred = my_dict['out'] #predicted(reconstructed) output


plt.figure()

#reshape 3D tensor (time, kpis, channels) into 2D (time, kpis). Easier for error computations etc. 
#Then you can reshape back to 3D if you want. python will preserve the original order of kpis in the channels.
real = np.reshape(real, (real.shape[0],-1)) 
pred = np.reshape(pred, (pred.shape[0],-1))

#delete kpis that have values "frozen" at 0 (probably due to inactive sensors).
idx = np.argwhere( np.all(real == 0, axis = 0) )
real = np.delete(real, idx, axis=1)
pred = np.delete(pred ,idx, axis=1)


se = (np.square(real - pred)) #calculate squared reconstruction error between input and output

#normalize the errors so they are on the same scale. Then we can average them (This way e.g. a Data Rate error in range of millions,
#will not dominate a Response Time error that may be in range of 0-10.)
nse = MinMaxScaler().fit_transform(se) 

nmse = nse.mean(axis=1) #compute average errors over all kpis, Result is one aggregated error value per timestamp.
plt.plot(nmse) #plot this aggregated error

plt.figure()

#for smoothing purposes, we calculate the moving average (MA) of the error with this function call:
nmse_ma = bn.move_mean(nmse, window=288, min_count=1) #the rolling window here is set to 288= 1 day
plt.plot(nmse_ma, label='MSE_MA') #plot MA error
mean = np.mean(nmse_ma) #calculate the mean of the MA error
sigma = np.std(nmse_ma, ddof=1) #calculate the sigma of the MA error

#draw horizontal lines on the plot for the mean, mean+sigma and mean + 2*sigma
plt.axhline(mean, color='C1', label='mean')
plt.axhline(y=mean + 2*sigma, color='k', ls='--', label='mean + 2*sigma')
plt.axhline(y=mean + sigma, color='k', ls=':', label='mean + sigma')
plt.grid()


# now we classify timestamps as anomalies based on 2-sigma rule. 
anom_ids = np.zeros_like(nmse_ma) #create array of zeros with same length as the number of timestamps

#for each timestamp, if the error deviates more than 2sigma from the mean,
#we flag it as an anomaly by setting a non-zero value to the anom_ids array for this timestamp.
for i in range(anom_ids.shape[0]):
    if nmse_ma[i] > mean + 2*sigma:
        anom_ids[i] = np.max(nmse_ma) #we set this value (np.max(nmse_ma)) for plotting purposes (so that the bounding box of the anomaly looks good). It can be any non-zero value. 
plt.plot(anom_ids, color='r') #we plot the anomaly flags in the same plot as the error. The anomalous timestamps will be bound by red blocks. 
plt.legend()
plt.xlabel('Timestamp', fontsize=12)
plt.ylabel('Error', fontsize=12)



####### compute starts and end times of anomalies (in timestamp indexes) #####################
def ranges(nums):
    nums = sorted(set(nums))
   
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    print(gaps)
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    print(edges)
    return list(zip(edges, edges))

ad_t = ranges(np.where(anom_ids>0)[0]) #time windows of anomalies [[start1, end1], [start2, end2]]
print(ad_t)

######## get names of top-k anomalous KPIs  #################

#open pkl file that contains a dictionary with the dataset for a given system
with open("file.pkl", 'rb') as f:
    data_dict = pickle.load(f)

kpis = data_dict['KPI_names'] #names of all KPIs
time = data_dict['Time']  #datetimes for all measurements 
kpis = kpis.reshape((-1))
kpis = np.delete(kpis ,idx)  #delete the names of the "frozen" KPIs 

# get top-10 anomalous kpis, by sorting them according to their reconstruction error .
# We do this separately for every detected anomaly
top_k_feats = []
topk_idxs = []
for anom in ad_t:
    errors = nse[anom[0]:anom[1]+1]
    avg = errors.mean(axis=0)
    sorted_kpis = np.argsort(avg)  # sorting puts the largest error last.
    top10 = sorted_kpis[-10:]       # you can change the 10 to any number k, to get the top-k KPIs
    print(kpis[top10])
    
    top_k_feats.append(kpis[top10].tolist()) #we save the names of the top-k KPIs in a list here
    topk_idxs.append(top10)    #and we also save their corresponding indexes here, in case they are useful
topk_idxs = np.array(topk_idxs)


# get full dates for start, end of anomalies
# Above we had computed the anomaly windows in timestamp indexes.
#here we get the actual datetimes that correspond to the anomalies.
anomaly_idxs = []
for anom in ad_t:
    start, end = time.iloc[anom[0]], time.iloc[anom[1]]
    print([start, end])
    anomaly_idxs.append([start,end])


# the anomaly windows and the top-k culprit KPIs for each anomaly constitute the output of our AD.


system_results = {'ID': "338e597a707a4e739ff1efb6921cafde", 'name': "system_name", 'anomaly_idxs': anomaly_idxs, 'top_k_feats': top_k_feats}


import json
with open('name.json', 'w') as fp:
    json.dump(system_results, fp)

