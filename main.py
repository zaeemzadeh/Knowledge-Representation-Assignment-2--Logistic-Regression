import os
# import sys
import random
from numpy import genfromtxt
import numpy as np
import math 
import matplotlib.pyplot as plt

def P_Y1givenXW(x,w):
	P = 1.0 - 1.0/(1.0 + math.exp(sum(w*x)))
	return P

def train(train_data,step_size,regularization,stochastic_size,test_data):
	w = np.array(np.zeros(len(train_data[0])-1), dtype = float)
	maxIter = 500
	TestError_vs_iter = []
	TrainError_vs_iter = []
	LogLiklihood_vs_iter = []

	for t in range(maxIter):
		gradient = 0
		LogLiklihood = 0
		error = 0 
		# compute gradient
		np.random.shuffle(train_data)
		for i in range(int(stochastic_size)):
			x = train_data[i][0:-1]
			y = train_data[i][-1]
			gradient += x*(y - P_Y1givenXW(x,w))
			if y == 1:
				LogLiklihood += math.log(P_Y1givenXW(x,w))
			else:
				LogLiklihood += math.log(1 - P_Y1givenXW(x,w))
			error += abs(y - P_Y1givenXW(x,w))

		TrainError_vs_iter.append(error/len(train_data))
		
		# update coefficients	
		alpha = step_size/(1+t)
		w += alpha*(gradient - regularization*w)

		# compute test error
		labels = test(test_data,w)
		error = 0
		for i in range(len(test_data)):
			error += abs(test_data[i][-1] - labels[i])

		TestError_vs_iter.append(error/len(test_data))
		LogLiklihood_vs_iter.append(LogLiklihood)

		if alpha*sum(gradient*gradient) < 0.001:
			break

	plt.semilogx(range(maxIter), TrainError_vs_iter, 'r--', label = "Train Error")
	plt.semilogx(range(maxIter), TestError_vs_iter, 'b-', label = "Test Error")
	plt.legend()
	plt.show()

	plt.semilogx(LogLiklihood_vs_iter, label = "Log-Likelihood")
	plt.legend(loc = 'right')
	plt.show()
	 
	return w


def test(test_data,w):
	labels = []
	for record in test_data:
		x = record[0:-1]
		if P_Y1givenXW(x,w) > 0.5:
			labels.append(1)
		else:
			labels.append(0)
	return labels


def confusion_matrix(data,labels):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	for i in range(len(labels)):
		y = int(data[i][-1])
		l = int(labels[i])
		# print l
		if y == l and l == 1:
			TP = TP +1
		elif y == l and l == 0:
			TN = TN +1
		elif y != l and l == 1:
			FP = FP +1
		else:
			FN = FN +1
		
		
	return np.array([[TP, FN],[FP, TN]])


def read_data():
	#read data
	print "\n\n\n*************Abalone Data Set****************** \n"
	data = genfromtxt('abalone.data', dtype = None, delimiter = ',')
	data = list(data)
	H0 = 7
	H1 = 12
	print "******* H0: Rings == " , H0 , "   H1: Rings == " , H1 ,  " ********* \n\n\n"

	black_list = []
	for i in range(len(data)):
		record = list(data[i])
		# reducing the prroblem into a binary decison problem
		if (record[len(record)-1] == H0):
			record[len(record)-1] = 0
		elif (record[len(record)-1] == H1):
			record[len(record)-1] = 1
		else:
			black_list.append(int(i))
			continue

		# converting the categorical attribute 'Sex' into binary vector
		if record[0] == 'M':
			record.pop(0)
			L = [1.0,1.0,0.0,0.0]
			record[:0] = L
		elif record[0] == 'F':
			record.pop(0)
			L = [1.0,0.0,1.0,0.0]
			record[:0] = L
		else: 
			record.pop(0)
			L = [1.0,0.0,0.0,1.0]
			record[:0] = L

		data[i] = np.array(record,dtype = float);
	data = np.delete(data,black_list, 0)
	return data


def KfoldCrossValidation(data,NumOfFolds,step_size,stochastic_subset_size,regularization):

	FoldSize = math.floor(len(data)/NumOfFolds)
	print   "********************** Algorithm Parameters*******************************"
	print 	"\t Learning Rate  = %.3f" % step_size, \
		"\t Stochastic Subset Size = %d" % stochastic_subset_size, \
		"\t Regularization Parameter    = %.4f" % regularization, \
		"\t Number of Folds = %d" % NumOfFolds
	print   "**************************************************************************"

	Accuracy = []
	Precision = []
	Recall = []
	F_measure = []

	for i in range(NumOfFolds):
		print "Fold#\t%d"% i,
		#split data
		test_data = data[int(i*FoldSize):int((i+1)*FoldSize)]
		train_data= np.delete(data, range(int(i*FoldSize),int((i+1)*FoldSize)), 0)	

		#train
		w = train(train_data,step_size,regularization,stochastic_subset_size,test_data)

		#test
		labels = test(test_data,w)

		# Compute Performance Metrics
		conf_mtx = confusion_matrix(test_data,labels)
		TP = conf_mtx[0][0]
		FN = conf_mtx[0][1]
		FP = conf_mtx[1][0]
		TN = conf_mtx[1][1]

		print 		"\t Accuracy  = %.3f" % ((TP+TN)/float(TP+FP+FN+TN)), \
				"\t Precision = %.3f" % (TP/float(TP+FP)), \
				"\t Recall    = %.3f" % (TP/float(TP+FN)), \
				"\t F_Measure = %.3f" % ((2*TP)/float(2*TP+FP+FN))

		Accuracy.append((TP+TN)/float(TP+FP+FN+TN))
		Precision.append(TP/float(TP+FP))
		Recall.append(TP/float(TP+FN))
		F_measure.append(((2*TP)/float(2*TP+FP+FN)))

	print "\n\n=============================================== "
	print "Average Performance Metrics:"
	print "AVG Accuracy:   \t%.3f" % np.mean(Accuracy), \
		" \t STD: %.3f" % np.std(Accuracy)
	print "AVG Precision:  \t%.3f" % np.mean(Precision), \
		" \t STD: %.3f" %np.std(Precision)
	print "AVG Recall:     \t%.3f" %np.mean(Recall), \
		" \t STD: %.3f" %np.std(Recall)
	print "AVG F_measure:  \t%.3f" % np.mean(F_measure), \
		" \t STD: %.3f" % np.std(F_measure), "\n\n\n"

	return np.mean(Accuracy)



def main():	
	data = read_data()
	
	parameter_to_sweep = [0] #np.logspace(-3,3,num = 5)

	mean_accuracy = []
	for param in parameter_to_sweep:
		print "#######################################################\n\n\n"
		#Algorithm Parameters
		NumOfFolds = 10
		FoldSize = math.floor(len(data)/NumOfFolds)
		step_size = 0.01
		stochastic_size = 4*FoldSize  #(NumOfFolds - 1) * FoldSize #  for regular gradient descent (NumOfFolds - 1) * len(FoldSize)
		regularization = param

		mean_accuracy.append(KfoldCrossValidation(data,NumOfFolds,step_size,stochastic_size,regularization))


	#plt.semilogx(parameter_to_sweep,mean_accuracy)
	#plt.show()

main()
