# solarLMP.py
# Author: Slobo Matic
# Date: 11/01/2015

import pandas as pd
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from urllib2 import Request, urlopen
import json

folderName = r'.'
zipcodeFileName = r'\US Zip Codes from 2013 Government Data.csv'
openPVFileName = r'\openpv-export-201510311549.csv'

zipcodeDF = pd.read_csv(folderName+zipcodeFileName)
zipcodeDF.columns = ['zipcode', 'lat', 'lng']

request=Request('http://developer.nrel.gov//api/solar/open_pv/installs/index?api_key=8Sk1IvuL0Nb5Yucz937sXCE7bWugUgZmL4pNneIk&state=CA&nppage=5000')
response = urlopen(request)	# instance
infoDataStr = response.read()	# str
infoData = json.loads(infoDataStr)	# dict, infoData['result'] - list
openPVDF = pd.DataFrame(infoData['result']) # df
openPVDF.columns = ['id', 'cost', 'date', 'size', 'state', 'zipcode']
openPVDF.zipcode = openPVDF.zipcode.apply(lambda d: int(d))

# openPVDF = pd.read_csv(folderName+openPVFileName, index_col=False) # _not_ use the first column as the index
# openPVDF.columns = ['zipcode', 'state', 'size', 'cost', 'date', 'filterx']
# openPVDF.date = openPVDF.date.apply(lambda d: datetime.strptime(d, "%m/%d/%Y"))
# empty = openPVDF.apply(lambda col: pd.isnull(col))
# openPVDF.index = openPVDF.date

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

mlmpGroup = min(lmpGroupA[:,0])
MlmpGroup = max(lmpGroupA[:,0])
lmpGroupSize = 1 + 200*(lmpGroupA[:,0]-mlmpGroup)/(MlmpGroup-mlmpGroup)
fig, ax = plt.subplots(1,1)
ax.scatter(lmpGroupA[:,2], lmpGroupA[:,1], s=lmpGroupSize, alpha = 0.5, lw = 1, color='b', edgecolors='k')	# facecolor='0.5',
ax.scatter(pvGroupA[:,2], pvGroupA[:,1], s=pvGroupSize, alpha = 0.5, lw = 1, color='r', edgecolors='k')	# facecolor='0.5',
#plt.title('Solar Capacity and Locational Marginal Prices in California')
plt.ylabel('Latitude [deg]')
plt.xlabel('Longitude [deg]')
plt.figtext(0.34, 0.92, "Solar Capacity", fontsize='large', color='r', ha ='right')
plt.figtext(0.39, 0.92, "Locational Marginal Prices", fontsize='large', color='b', ha ='left')
plt.figtext(0.36, 0.92, ' vs ', fontsize='large', color='k', ha ='center')
plt.figtext(0.80, 0.92, "in California", fontsize='large', color='k', ha ='center')

largeCity = ['LA', 'SD', 'SF', 'FR', 'SC']
#, 'San Jose', 'Fresno', 'Long Beach', 'Sacramento', 'Oakland', 'Santa Ana', 'Anaheim']
largeCityLAT = [34.052233, 32.715328, 37.774931, 36.746842, 38.581572]
largeCityLNG = [-118.243686, -117.157256, -122.419417, -119.772586, -121.494400]
plt.figtext(0.29, 0.53, "SF", fontsize='large', color='g', ha ='center')
plt.figtext(0.59, 0.11, "SD", fontsize='large', color='g', ha ='center')
plt.figtext(0.54, 0.46, "FR", fontsize='large', color='g', ha ='center')
plt.figtext(0.54, 0.20, "LA", fontsize='large', color='g', ha ='center')
plt.figtext(0.45, 0.64, "SC", fontsize='large', color='g', ha ='center')
ax.scatter(largeCityLNG, largeCityLAT, s=30, alpha = 1, lw = 1, color='g', edgecolors='g')	# facecolor='0.5',
plt.ylim([32,42])
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
