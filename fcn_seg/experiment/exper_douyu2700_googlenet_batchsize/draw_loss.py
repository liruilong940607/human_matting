import os,sys
import numpy as np
import re
import matplotlib.pyplot as plt

train_iter = []
train_loss = []

test_loss = [[],[]]
test_overall_accu = [[],[]]
test_mean_accu = [[],[]]
test_mean_IU = [[],[]]
test_fwavacc = [[],[]]



lines = open('./train.log','r').readlines()
for line in lines:
	p = r"] Iteration [0-9]*.+, loss = [0-9]*\.[0-9]*"
	pattern = re.compile(p)
	matcher = re.search(pattern,line)
	if matcher:
		print matcher.group(0)
		numlist = re.findall('[0-9]\.*[0-9]*',matcher.group(0))
		train_iter.append(float(numlist[0]))
		train_loss.append(float(numlist[-1]))

	p = r"test Iteration [0-9]* loss [0-9]*\.[0-9]*"
	pattern = re.compile(p)
	matcher = re.search(pattern,line)
	if matcher:
		print matcher.group(0)
		numlist = re.findall('[0-9]\.*[0-9]*',matcher.group(0))
		if numlist[1] >0:
			test_loss[0].append(float(numlist[0]))
			test_loss[1].append(float(numlist[1]))

	p = r"test Iteration [0-9]* mean IU [0-9]*\.[0-9]*"
	pattern = re.compile(p)
	matcher = re.search(pattern,line)
	if matcher:
		print matcher.group(0)
		numlist = re.findall('[0-9]\.*[0-9]*',matcher.group(0))
		if numlist[1] >0:
			test_mean_IU[0].append(float(numlist[0]))
			test_mean_IU[1].append(float(numlist[1]))

	# p = r"test Iteration [0-9]* overall accuracy [0-9]*\.[0-9]*"
	# pattern = re.compile(p)
	# matcher = re.search(pattern,line)
	# if matcher:
	# 	print matcher.group(0)
	# 	numlist = re.findall('[0-9]\.*[0-9]*',matcher.group(0))
	# 	if numlist[1] >0:
	# 		test_overall_accu[0].append(float(numlist[0]))
	# 		test_overall_accu[1].append(float(numlist[1]))

	# p = r"test Iteration [0-9]* mean accuracy [0-9]*\.[0-9]*"
	# pattern = re.compile(p)
	# matcher = re.search(pattern,line)
	# if matcher:
	# 	print matcher.group(0)
	# 	numlist = re.findall('[0-9]\.*[0-9]*',matcher.group(0))
	# 	if numlist[1] >0:
	# 		test_mean_accu[0].append(float(numlist[0]))
	# 		test_mean_accu[1].append(float(numlist[1]))

	# p = r"test Iteration [0-9]* fwavacc [0-9]*\.[0-9]*"
	# pattern = re.compile(p)
	# matcher = re.search(pattern,line)
	# if matcher:
	# 	print matcher.group(0)
	# 	numlist = re.findall('[0-9]\.*[0-9]*',matcher.group(0))
	# 	if numlist[1] >0:
	# 		test_fwavacc[0].append(float(numlist[0]))
	# 		test_fwavacc[1].append(float(numlist[1]))


plt.figure(1)
plt.plot(train_iter, train_loss, 'g')
plt.plot(test_loss[0], test_loss[1], 'r')
plt.savefig('iter_loss_train.jpg',dpi=100)

plt.figure(2)
plt.plot(test_mean_IU[0], test_mean_IU[1],'b')
#plt.plot(test_overall_accu[0], test_overall_accu[1],'c')
#plt.plot(test_mean_accu[0], test_mean_accu[1],'g')
#plt.plot(test_fwavacc[0], test_fwavacc[1],'y')
plt.savefig('iter_iou_test.jpg',dpi=100)
plt.show()
