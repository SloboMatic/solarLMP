# solarLMP2.py
# Author: Slobo Matic
# Date: 11/01/2015

import pandas as pd
from datetime import datetime
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

folderName = r'.'
zipcodeFileName = r'\US Zip Codes from 2013 Government Data.csv'
openPVFileName = r'\openpv-export-201510311549.csv'

zipcodeDF = pd.read_csv(folderName+zipcodeFileName)
zipcodeDF.columns = ['zipcode', 'lat', 'lng']

openPVDF = pd.read_csv(folderName+openPVFileName, index_col=False) # _not_ use the first column as the index
openPVDF.columns = ['zipcode', 'state', 'size', 'cost', 'date', 'filterx']
openPVDF.date = openPVDF.date.apply(lambda d: datetime.strptime(d, "%m/%d/%Y"))
#empty = openPVDF.apply(lambda col: pd.isnull(col))
#openPVDF.index = openPVDF.date

pvLocDF = pd.merge(openPVDF, zipcodeDF, on='zipcode', how='inner')

byGroup = pvLocDF.groupby('zipcode')
pvGroupA = np.zeros((len(byGroup),3))
pvGroupA[:,0] = byGroup['size'].sum()
pvGroupA[:,1] = byGroup['lat'].mean()
pvGroupA[:,2] = byGroup['lng'].mean()

mpvGroup = min(pvGroupA[:,0])
MpvGroup = max(pvGroupA[:,0])
pvGroupSize = 1 + 200*(pvGroupA[:,0]-mpvGroup)/(MpvGroup-mpvGroup)

lmpLocFileName = r'\LMPLocations.csv'
caisoFileName = r'\20131031_20131031_PRC_LMP_DAM_LMP_v1.csv'

lmpLocDF = pd.read_csv(folderName+lmpLocFileName)
lmpLocDF.columns = ['lat', 'type', 'NODE', 'lng']

caisoDF = pd.read_csv(folderName+caisoFileName)
#openPVDF.date = openPVDF.date.apply(lambda d: datetime.strptime(d, "%m/%d/%Y"))

lmpCALocDF = pd.merge(caisoDF, lmpLocDF, on='NODE', how='inner')

lmpGroup = lmpCALocDF.groupby('NODE')
lmpGroupA = np.zeros((len(lmpGroup),3))
lmpGroupA[:,0] = lmpGroup['MW'].mean()
lmpGroupA[:,1] = lmpGroup['lat'].mean()
lmpGroupA[:,2] = lmpGroup['lng'].mean()

# clean
lmpGroupAux = np.zeros((len(lmpGroup),3))
lmpGroupAux[:,0] = lmpGroup['MW'].mean()
lmpGroupAux[:,1] = lmpGroup['lat'].mean()
lmpGroupAux[:,2] = lmpGroup['lng'].mean()
lmpGroupA[(lmpGroupAux[:,1]<30) | (lmpGroupAux[:,1]>42),1] = np.mean(lmpGroupAux[:,1])
lmpGroupA[(lmpGroupAux[:,1]<30) | (lmpGroupAux[:,1]>42),2] = np.mean(lmpGroupAux[:,2])
lmpGroupA[lmpGroupAux[:,2]>-110,2] = np.mean(lmpGroupAux[:,2])
lmpGroupA[lmpGroupAux[:,2]>-110,1] = np.mean(lmpGroupAux[:,1])

numBins = 200.0
totPVBin = np.zeros(numBins)
mlmp = lmpGroupA[:,0].min()
Mlmp = lmpGroupA[:,0].max()
dlmp = (Mlmp-mlmp)/numBins
lmpDistA = np.ones(len(byGroup))*sys.float_info.max
lmpIndA = np.zeros(len(byGroup),dtype=np.uint16)
for pv in range(len(byGroup)):
	for lmp in range(len(lmpGroup)):
		dist = (pvGroupA[pv,1]-lmpGroupA[lmp,1])*(pvGroupA[pv,1]-lmpGroupA[lmp,1])+(pvGroupA[pv,2]-lmpGroupA[lmp,2])*(pvGroupA[pv,2]-lmpGroupA[lmp,2])
		if dist < lmpDistA[pv]:
			lmpDistA[pv] = dist
			lmpIndA[pv] = lmp
	if lmpGroupA[lmpIndA[pv],0] == Mlmp:
		totPVBin[numBins-1] += pvGroupA[pv,0]
	else:
		totPVBin[math.floor((lmpGroupA[lmpIndA[pv],0]-mlmp)/dlmp)] += pvGroupA[pv,0]

stepPVBin = np.zeros((2*numBins+1,2))
cumulativePV = np.zeros(numBins)
for i in range(int(numBins)):
	stepPVBin[2*i,0] = mlmp+i*dlmp
	stepPVBin[2*i,1] = totPVBin[i]
	stepPVBin[2*i+1,0] = mlmp+(i+1)*dlmp
	stepPVBin[2*i+1,1] = totPVBin[i]
	if i==0:
		cumulativePV[i] = totPVBin[i] 
	else:
		cumulativePV[i] = cumulativePV[i-1]+totPVBin[i] 
stepPVBin[2*numBins,0] = mlmp+numBins*dlmp
stepPVBin[2*numBins,1] = 0.0

#fig, ax = plt.subplots(1,1)
fig, ax = plt.subplots(2, sharex=False)
for pv in range(len(byGroup)):
	if pv == 0:
		ax[0].scatter(lmpGroupA[lmpIndA[pv],0], pvGroupA[pv,0], s=2, color='LightGreen', label='site')
	else:
		ax[0].scatter(lmpGroupA[lmpIndA[pv],0], pvGroupA[pv,0], s=2, color='LightGreen')
ax[0].set_title('Installed Solar had no Impact on Locational Marginal Prices',color='r')
ax[0].set_xlabel('LMP [$/MWh]')
ax[0].set_ylabel('Solar Capacity [kW]')
ax[0].set_xlim([mlmp,Mlmp])
ax[0].set_ylim([0,max(totPVBin)])
ax[0].plot(stepPVBin[:,0], stepPVBin[:,1], color='DarkGreen', label='total')
ax[0].legend()
fig.show()

numPerc = numBins
pvTotal = sum(pvGroupA[:,0])
numTotal = float(len(pvGroupA[:,0]))
pvPerc = np.zeros((numPerc + 1))
#lmpMed = np.median(lmpGroupA[:,0])
for Percentile in range(int(numPerc) + 1):
	lmpPercentile = np.percentile(lmpGroupA[lmpIndA[:],0],100-Percentile*100/numPerc)
	#lmpPercentile = Mlmp - Percentile*dlmp
	lmpPercentileInv = np.percentile(lmpGroupA[lmpIndA[:],0],Percentile*100/numPerc)
	pvLmpHigh, pvLmpLow = 0, 0
	pvLmpHighInv, pvLmpLowInv = 0, 0
	numLmpHigh, numLmpLow = 0.0, 0.0
	numLmpHighInv, numLmpLowInv = 0.0, 0.0
	for pv in range(len(byGroup)):
		if lmpGroupA[lmpIndA[pv],0] >= lmpPercentile:
			pvLmpHigh += pvGroupA[pv,0]
			numLmpHigh += 1
		else:
			pvLmpLow += pvGroupA[pv,0]
			numLmpLow += 1
		if lmpGroupA[lmpIndA[pv],0] >= lmpPercentileInv:
			pvLmpHighInv += pvGroupA[pv,0]
			numLmpHighInv += 1
		else:
			pvLmpLowInv += pvGroupA[pv,0]
			numLmpLowInv += 1
	pvPerc[Percentile] = pvLmpHigh/pvTotal*100.0 
# print(pvLmpHigh/pvTotal, pvLmpLow/pvTotal)
# print(numLmpHigh/numTotal, numLmpLow/numTotal)
# print(pvLmpHighInv/pvTotal, pvLmpLowInv/pvTotal)
# print(numLmpHighInv/numTotal, numLmpLowInv/numTotal)
#fig, ax = plt.subplots(1,1)
ax[1].scatter(np.arange(0,100,100/numBins), cumulativePV/pvTotal*100.0, s=2, color='DarkGreen', label='cumulative')
ax[1].scatter(np.arange(0,100+1/numPerc,100/numPerc), pvPerc, s=2, color='r', label='percentile')
ax[1].set_xlabel('Percentile LMP')
ax[1].set_ylabel('Percentile Solar Capacity')
ax[1].set_xlim([0,100])
ax[1].set_ylim([-2,102])
ax[1].legend(loc = 'lower right')
fig.show()

pp = PdfPages(folderName + caisoFileName.split('.')[0] + '.pdf')
plt.tight_layout() # for full page bounding box
#fig.savefig(folderName + "\\" + loadFileName.split('.')[0] + '.pdf', orientation='portrait', format='pdf')
plt.savefig(pp, format='pdf')

pp.close()

#len(openPVDF)	291513
#len(zipcodeDF)	33144
# mDF1 = pd.merge(openPVDF, zipcodeDF, on='zipcode')
# # len(mDF1)	289685
# mDF2 = pd.merge(openPVDF, zipcodeDF, on='zipcode', how='outer')
# #len(mDF2)	323090
# mDF3 = pd.merge(openPVDF, zipcodeDF, on='zipcode', how='inner')
#len(mDF3)	289685
# mDF4 = pd.merge(openPVDF, zipcodeDF, on='zipcode', how='left')
# #len(mDF4)	291513
# mDF5 = pd.merge(openPVDF, zipcodeDF, on='zipcode', how='right')
# #len(mDF5)	321262

