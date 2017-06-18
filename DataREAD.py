import os
import sys
import subprocess
import numpy as np
import scipy as sp
import shutil
import datetime


def DataREAD(DataFileName):
	"""DataREAD reads in cell survival fraction data from experiments.
		
		Arguments:
			DataFileName  - The name of the Data.in file, set in LQFit.py. Default is "Data.in"
			Data.in       - A text file consisting of the data
			
		Outputs:
			ExpData    		 - A matrix form of the parameters.in form
		
		********** Editions and Revisions *************
			Created 2017.06.17 by Nick Sherck
	
	"""
	ExpData = np.loadtxt(DataFileName, dtype=float, ndmin=2, comments='#', delimiter=',', unpack = True)
	ExpData = np.asarray(ExpData)
	return ExpData