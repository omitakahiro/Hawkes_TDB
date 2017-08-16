.. Hawkes_TDB documentation master file, created by
   sphinx-quickstart on Tue Aug  8 13:11:21 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
.. py:module:: Hawkes_TDB

About Hawkes_TDB
----------------------
A python module “Hawkes_TDB” aims to estimate a Hawkes process model with a time-dependent background rate based on Omi et al., (2017). Anyone can freely use this code without permission for non-commercial purposes, but please appropriately cite Omi et al., (2017) if you present results obtained by using Hawkes_TDB.

Reference
----------------------
T. Omi, Y. Hirata, and K. Aihara, "Hawkes process model with a time-dependent background rate and its application to high-frequency financial data", Physical Review E 96, 012303 (2017). https://doi.org/10.1103/PhysRevE.96.012303

Requirements
----------------------
:mod:`Hawkes_TDB` depends on the following external packages:
	
- Python 2.7
- Numpy
- Scipy
- Matplotlib
- Pandas
- Cython

Download
---------------------
A source code is available at \ https://github.com/omitakahiro/Hawkes_TDB/\ . After downloading and unzipping the file, please place the "Hawkes_TDB" folder in your working folder.


Data preparation
---------------------
A data file is an ascii file that contains the list of occurrence times of events. A sample data file is included in the source file (./Hawkes_TDB/sample.txt). 
    
A data file is like:
	
.. code-block:: none

	11.766350
	74.832453
	86.070326
	91.188811
	106.781628
	121.730578
	176.882013
	189.934206
	193.354620
	200.264633
	...


.. note:: The times must be different to each other. If multiple events have the identical times, our code does not work well.


User manual
---------------------
We here consider a Hawkes process model with a time dependent backgraound rate given as

.. math::
	
	\lambda(t) = \mu(t) + \sum_{i<t} g(t-t_i),
	
where :math:`{t_i}` represnts the times of events in the observation interval, and :math:`\mu(t)` and :math:`g(\cdot)` are respectively the background rate and a triggering function. Here the triggering function is set to the sum of two exponential functions according to our previous work (Omi et al., 2017),
	
.. math::
	g(s) = \sum_{j=1}^{2} \alpha_{j} \beta_{j} e^{-\beta_{j}s}.
	
In this setting, the branching ratio is given as :math:`\alpha_1 + \alpha_2`.

API
^^^^^^

A function :py:func:`Estimate` estimates the parameters of the above model from the data.
	
.. py:function:: Estimate(T,itv)

   estimates the parameters of the above model.

   :param T: a numpy array of the times of events
   :param itv: a range of an observation interval ([float, float])
   :return: a dictionary that represents the estimation results, summarized in the below table. 
        
.. csv-table::
   :header: "key", "despription"
   :widths: 10,100

   "alpha1","the alpha parameter of the first exponential function "
   "alpha2","the alpha parameter of the second exponential function "
   "beta1","the beta parameter of the first exponential function"
   "beta2","the beta parameter of the second exponential function"
   "mu","the background rate at each event"
   "L","log marginal likelihood"

Sample code
^^^^^^^^^^^^^

Here is a sample code to estimate the parameters from a test data. (take a few minutes)

.. code-block:: python

	import Hawkes_TDB as hk
	import numpy as np
	import matplotlib.pyplot as plt
	
	# if you use ipython notebook
	%matplotlib inline
	
	# estimation
	T = np.loadtxt("./Hawkes_TDB/sample.txt") # data
	itv = [0.0, 21600.0] # an observation interval
	param = hk.Estimate(T,itv)
	
	# print the estimated parameter values
	print "alpha_1: %.2f"  % param["alpha1"]
	print "beta_1: %.2f"  % param["beta1"]
	print "alpha_2: %.2f"  % param["alpha2"]
	print "beta_2: %.2f"  % param["beta2"]
	print "log marginal-likelihood: %.2f" % param["L"]
	print "branching ratio: %.2f" % ( param["alpha1"] + param["alpha2"] )
	
	# plot the estimate of the time-dependent background rate \mu(t)
	plt.figure()
	plt.semilogy(T,param["mu"],'k-')
	plt.xlabel("time")
	plt.ylabel("background rate")
	plt.xlim(itv)

Output:
	
.. code-block:: none

	alpha_1: 0.27
	beta_1: 0.21
	alpha_2: 0.31
	beta_2: 2.83
	log marginal-likelihood: 4715.65
	branching ratio: 0.57	
	
.. image:: mu.png
