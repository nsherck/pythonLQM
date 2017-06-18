import os
import sys
import subprocess
import numpy as np
import scipy as sp
from scipy import signal
from scipy.fftpack import fft, fftn
from scipy.fftpack import fft, fftfreq, fftshift
import shutil
import datetime
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy import optimize
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt


import DataREAD			# Imports the experimental data

def LQM(D, alpha, beta):
	"""The Linear-Quadratic Model
	
	"""
	
	return np.exp(-alpha * D + -beta * D**2 )
	
def LQFit(DataFileName):
	""" LQFit is a function to extract the alpha and beta values in the linear-quadratic model using 
		least-squares minimization. 
	
		Arguments:
			DataFileName - name of the text file containing the experimental data. Default is "Data.in"
			
		Outputs:
			LQFitOut	- Txt File of the fiting results
	
	******** Editions and Revisions **********
		Created 2017.06.17 by Nick Sherck	
	
	"""
	
	ExpData = DataREAD.DataREAD(DataFileName)
	r = len(ExpData)
	datasets = r/2
	print "datasets"
	print datasets
	i = 1
	
	parameters = np.zeros((datasets,2))
	while i <= datasets:
		print "iter"
		print i
		D = list(ExpData[(i-1)*2])
		y = list(ExpData[(i-1)*2+1])
		print D
		print y
		paramtemp, paramCOV = curve_fit(LQM, D, y)
		print "paramtemp"
		print paramtemp
		parameters[i-1][0] = paramtemp[0] # alpha 
		parameters[i-1][1] = paramtemp[1] # beta
		d1 = max(D) + 1
		range = np.linspace(0,d1,100)
		
		fig = plt.figure()
		plt.semilogy(range, LQM(range, *paramtemp), 'r-', label="LQM Fit")
		plt.semilogy(D, y, '-o', label="Exper.")
		plt.legend()
		plt.grid()
		plt.xlabel("dose")
		plt.ylabel("surving fraction")
		plt.title('Linear-Quadratic Model Fit Parameters \n alpha = %2.4f and beta = %2.4f' % (paramtemp[0],paramtemp[1]))
		plt.tight_layout()
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.tick_params(axis='both', which='minor', labelsize=16)
		plt.savefig("figure"+str(i)+".pdf")
		plt.close()
		i = i + 1
		print i
		
	np.savetxt("LQMFit.txt", parameters, fmt='%2.6f', header="alpha	  beta")
		

		
	
	
	
	