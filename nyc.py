# nyc311.py
# Author: Slobo Matic
# Date: 11/01/2015

import pandas as pd
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages

folderName = r'nyc311calls.csv'
nyc311FileName = r'\nyc311calls.csv'

nyc311DF = pd.read_csv(folderName+nyc311FileName)

# 1) assumption: most popular agency is the one with least complaints
groupAgency = nyc311DF.groupby('Agency')
groupAgencyA = np.zeros(len(groupAgency))
acyI = 0
for acy,acy_data in groupAgency:
	groupAgencyA[acyI] = len(acy_data)
	acyI += 1
#print(groupAgencyA.shape, sum(groupAgencyA))
sgroupAgencyA = np.sort(groupAgencyA)
print(sgroupAgencyA[1]/sum(groupAgencyA))

# 2) P(A\B)=P(AB)/P(B)
groupType = nyc311DF.groupby(['Complaint Type'])
groupTypeD = {}
for typ,typ_data in groupType:
	groupTypeD[typ] = len(typ_data)
# s = 0
# for typ in groupTypeD.keys():
# 	s += groupTypeD[typ]
# print(len(groupTypeD), s)

groupBorough = nyc311DF.groupby(['Borough'])
groupBoroughD = {}
for bor,bor_data in groupBorough:
	groupBoroughD[bor] = len(bor_data)
# s = 0
# for bor in groupBoroughD.keys():
# 	s += groupBoroughD[bor]
# print(len(groupBoroughD), s)

groupTypeBorough = nyc311DF.groupby(['Complaint Type','Borough'])
groupTypeBoroughD = {}
for (typ,bor),tb_data in groupTypeBorough:
	groupTypeBoroughD[(typ,bor)] = len(tb_data)
# s = 0
# for typ in groupTypeD.keys():
# 	for bor in groupBoroughD.keys():
# 		if (typ,bor) in groupTypeBoroughD.keys():
# 			s += groupTypeBoroughD[(typ,bor)]
# print(len(groupTypeBoroughD), s)

mSurp = 0
for typ in groupTypeD.keys():
	for bor in groupBoroughD.keys():
		if ((typ,bor) in groupTypeBoroughD.keys()):
			#and (groupBoroughD[bor]!=0.0) and (groupTypeD[typ]!=0.0):
			aux = (groupTypeBoroughD[(typ,bor)]/groupBoroughD[bor])/(groupTypeD[typ]/len(nyc311DF))
			if aux > mSurp:
				mSurp = aux
				# print(mSurp)
print(mSurp)

# 3)
print(nyc311DF['Latitude'].quantile(0.9) - nyc311DF['Latitude'].quantile(0.1))

# 4) assume degree of latitude is 111.2 km and degree of longitude is 87 km
# assume area is Sx*Sy*pi
kmpdegLat = 111.2
kmpdegLon = 87
pi = 3.1415926
print((nyc311DF['Latitude'].std()*kmpdegLat)*(nyc311DF['Longitude'].std()*kmpdegLon)*pi)
# 216.87040257887796 km^2 seems too small for NYC area. Oh, well.

# 6)
nyc311DF['Created Date'] = nyc311DF['Created Date'].apply(lambda d: datetime.strptime(d, "%m/%d/%Y %I:%M:%S %p"))
groupDateTimeAgency = nyc311DF.groupby(['Created Date','Agency'])	# group to eliminate unreasonable points
deltaDateTimeA = np.zeros(len(groupDateTimeAgency))
dati0 = nyc311DF['Created Date'][0]
i = 0
for (dati,acy),tb_data in groupDateTimeAgency:
	# if (dati0 - dati).total_seconds() < -60:
	# 	print(dati0,dati)
	# 	break
	# dati0 = dati
	deltaDateTimeA[i] = (dati0 - dati).total_seconds()
	i += 1
deltaDateTimeS = np.sort(deltaDateTimeA)
lenDelta = len(deltaDateTimeS)
for i in range(lenDelta-1):
	deltaDateTimeS[lenDelta-i-1] -= deltaDateTimeS[lenDelta-i-2] 
print(deltaDateTimeS.std())