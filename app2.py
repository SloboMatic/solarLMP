#!/bin/env python
# coding: utf-8

import os
import StringIO

from flask import Flask, render_template
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
#from datetime import datetime
import sys
#import numpy as np
#import matplotlib.pyplot as plt

app = Flask(__name__)
app.debug = True

@app.route('/')
def do_sin():
    # x = np.arange(-np.pi, np.pi, 0.1)
    # y = np.sin(x)

    #folderName = r'C:\Users\212300489\Google Drive\Personal\Data\Q\Code'
    folderName = r'.'
    zipcodeFileName = r'\US Zip Codes from 2013 Government Data.csv'
    openPVFileName = r'\openpv-export-201510311549.csv'

    zipcodeDF = pd.read_csv(folderName+zipcodeFileName)
    zipcodeDF.columns = ['zipcode', 'lat', 'lng']

    openPVDF = pd.read_csv(folderName+openPVFileName, index_col=False) # _not_ use the first column as the index
    openPVDF.columns = ['zipcode', 'state', 'size', 'cost', 'date', 'filterx']
    #openPVDF.date = openPVDF.date.apply(lambda d: datetime.strptime(d, "%m/%d/%Y"))
    #empty = openPVDF.apply(lambda col: pd.isnull(col))
    #openPVDF.index = openPVDF.date

    #len(openPVDF)  291513
    #len(zipcodeDF) 33144
    # mDF1 = pd.merge(openPVDF, zipcodeDF, on='zipcode')
    # # len(mDF1)   289685
    # mDF2 = pd.merge(openPVDF, zipcodeDF, on='zipcode', how='outer')
    # #len(mDF2)    323090
    # mDF3 = pd.merge(openPVDF, zipcodeDF, on='zipcode', how='inner')
    #len(mDF3)  289685
    # mDF4 = pd.merge(openPVDF, zipcodeDF, on='zipcode', how='left')
    # #len(mDF4)    291513
    # mDF5 = pd.merge(openPVDF, zipcodeDF, on='zipcode', how='right')
    # #len(mDF5)    321262

    pvLocDF = pd.merge(openPVDF, zipcodeDF, on='zipcode', how='inner')

    byGroup = pvLocDF.groupby('zipcode')
    pvGroupA = np.zeros((len(byGroup),3))
    pvGroupA[:,0] = byGroup['size'].sum()
    pvGroupA[:,1] = byGroup['lat'].mean()
    pvGroupA[:,2] = byGroup['lng'].mean()

    #fig,ax = plt.subplots(subplot_kw={'xticks':[],'yticks':[]})
    #ax.autoscale()

    # fig = plt.gcf()
    # plt.plot([-1,1],[-2,2],'c--',linewidth=2)
    # for i in range(len(byGroup)):
    #     fig.gca().add_artist(plt.Circle((pvGroupA[i,1],pvGroupA[i,2]),.1,color='b'))
    # plt.show()

    mpvGroup = min(pvGroupA[:,0])
    MpvGroup = max(pvGroupA[:,0])
    pvGroupSize = 1 + 200*(pvGroupA[:,0]-mpvGroup)/(MpvGroup-mpvGroup)
    # fig, ax = plt.subplots(1,1)
    # ax.scatter(pvGroupA[:,1], pvGroupA[:,2], s=pvGroupSize, alpha = 0.5, lw = 1, color='r', edgecolors='k')   # facecolor='0.5',
    # fig.show()

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

    lmpDistA = np.ones(len(byGroup))*sys.float_info.max
    lmpIndA = np.zeros(len(byGroup),dtype=np.uint16)
    for pv in range(len(byGroup)):
        for lmp in range(len(lmpGroup)):
            dist = (pvGroupA[pv,1]-lmpGroupA[lmp,1])*(pvGroupA[pv,1]-lmpGroupA[lmp,1])+(pvGroupA[pv,2]-lmpGroupA[lmp,2])*(pvGroupA[pv,2]-lmpGroupA[lmp,2])
            if dist < lmpDistA[pv]:
                lmpDistA[pv] = dist
                lmpIndA[pv] = lmp

    # fig, ax = plt.subplots(1,1)
    # ax.scatter(pvGroupA[:,0], lmpGroupA[lmpIndA[:],0], s=5, alpha = 0.5, lw = 1, color='b', edgecolors='k')   # facecolor='0.5',
    # fig.show()

    #fig, ax = plt.subplots(1,1)
    fig, ax = plt.subplots(2, sharex=False)
    for pv in range(len(byGroup)):
        ax[0].scatter(lmpGroupA[lmpIndA[pv],0], pvGroupA[pv,0], s=2, color='g')
    ax[0].set_title('Installed Solar had no Impact on Locational Marginal Prices',color='r')
    ax[0].set_xlabel('LMP [$/MWh]')
    ax[0].set_ylabel('Solar Capacity [kW]')
    ax[0].set_ylim([0,60000])
    fig.show()

    pvPerc = np.zeros((101))
    #lmpMed = np.median(lmpGroupA[:,0])
    for Percentile in range(101):
        lmpPercentile = np.percentile(lmpGroupA[lmpIndA[:],0],100-Percentile)
        lmpPercentileInv = np.percentile(lmpGroupA[lmpIndA[:],0],Percentile)
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
        pvTotal = sum(pvGroupA[:,0])
        numTotal = float(len(pvGroupA[:,0]))
        pvPerc[Percentile] = pvLmpHigh/pvTotal*100.0 
    # print(pvLmpHigh/pvTotal, pvLmpLow/pvTotal)
    # print(numLmpHigh/numTotal, numLmpLow/numTotal)
    # print(pvLmpHighInv/pvTotal, pvLmpLowInv/pvTotal)
    # print(numLmpHighInv/numTotal, numLmpLowInv/numTotal)
    #fig, ax = plt.subplots(1,1)
    ax[1].scatter(range(101), pvPerc, s=2, color='m')
    #plt.title('Solar Capacity vs. Locational Marginal Prices in California')
    ax[1].set_xlabel('Percentile LMP')
    ax[1].set_ylabel('Percentile Solar Capacity')
    ax[1].set_xlim([0,100])
    ax[1].set_ylim([0,100])
    fig.show()

    #fig = plt.figure()
    #plt.plot(x, y, label="sin")
    #plt.legend(loc="best")

    strio = StringIO.StringIO()
    fig.savefig(strio, format="svg")
    plt.close(fig)

    strip. seek (0) 
    svgstr  =  strip. toad [strip. toad. find ("<svg")]

    return render_template("sin.html", svgstr=svgstr.decode("utf-8"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port)