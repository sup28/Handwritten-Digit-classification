import numpy as np
import math as m
from numpy.linalg import inv
np.set_printoptions(threshold=np.inf)


#classification function of test data
																		
def classification(cov_0, cov_1):
	g = 0
	f = 0
	assg_labels = []
	for i in range(len(test_data)):
		g =  -0.5*m.log(np.linalg.det(cov_0)) - 0.5*np.matmul(np.transpose(test_data[i]-mean_0), np.matmul(inv(cov_0), (test_data[i]-mean_0))) + m.log(p_0)
		f =  -0.5*m.log(np.linalg.det(cov_1)) - 0.5*np.matmul(np.transpose(test_data[i]-mean_1), np.matmul(inv(cov_1), (test_data[i]-mean_1))) + m.log(p_1)
		if g>f :
			assg_labels.append(0)
		else:
			assg_labels.append(1)	

#-------------------------------------------------------------------------------------------------------------------------------------------
																			# populating confusion matrix
	a = 0
	b = 0
	c = 0
	d = 0
	x=0
	y=0

	for i in range(len(assg_labels)):
		if test_label[i] == 0 and assg_labels[i] == 0:
			a = a+1
		elif test_label[i] == 0 and assg_labels[i] == 1:
			b = b+1	
		elif test_label[i] == 1 and assg_labels[i] == 1:
			c = c+1
		else :
			d = d+1		


	x = a*100/(a+b)
	y = c*100/(c+d)
			
			
	print("\n")
	print ("the confusion matrix: ",np.matrix([[a,d],[b,c]]))
	print("\n")
	print("the classification percentage of 0 and 1 resp. :", x, y)
	print("\n")
	print("\n")
	return 0
	

#-------------------------------------------------------------------------------------------------------------------------------------------
#=========================================Main function==================================================================================================




data = np.genfromtxt('P2_train.csv',delimiter=',') 						# importing training data from csv file and classify it in class0 nad class1 and separate the labels
label = []

class0 = []
class1 = []
p_0 = 0
p_1 = 0
for i in range (len(data)):
    if data[i][2] == 1:
        label.append(1)
        class1.append(data[i][0:2])
        p_1 += 1
    else:
        label.append(0)
        class0.append(data[i][0:2])
        p_0 += 1
        
#-------------------------------------------------------------------------------------------------------------------------------------------
																		# mean calculation of class5 and class6
mean_0 = sum(class0)/len(class0)										
mean_1 = sum(class1)/len(class1)	

#-------------------------------------------------------------------------------------------------------------------------------------------
																		#covariance of class5
sum0= np.zeros((2,2))
for i in range(len(class0)):
	d = np.outer(class0[i],class0[i])
	sum0= np.add(sum0,d)

sum0 = sum0/len(class0)
cov_0 = sum0- np.outer(mean_0, mean_0)	
#-------------------------------------------------------------------------------------------------------------------------------------------

																		#covariance of class6
sum1= np.zeros((2,2))
for i in range(len(class1)):
	d = np.outer(class1[i],class1[i])
	sum1= np.add(sum1,d)

sum1 = sum1/len(class1)
cov_1 = sum1- np.outer(mean_1, mean_1)
#-------------------------------------------------------------------------------------------------------------------------------------------
																		# calculation of prior probabilities
p_0 = p_0/len(label)
p_1 = p_1/len(label)																		

#-------------------------------------------------------------------------------------------------------------------------------------------


test = np.genfromtxt('P2_test.csv',delimiter=',') 						# importing test data from csv file and separating labels and data 
test_data = []
test_label = []
for i in range (len(test)):
	if test[i][2] == 1:
		test_label.append(1)
		test_data.append(test[i][0:2])
	else:
		test_label.append(0)
		test_data.append(test[i][0:2])


#===========================================================================================================================================
#======================partwise classification==============================================================================================
#===========================================================================================================================================		 
#======Part a =========
cov_a = np.zeros((2,2))
cov = p_0*cov_0 + p_1*cov_1
cov_a[0][1] = 0
cov_a[1][0] = 0
cov_a[0][0] = 0.5*(cov[0][0]+cov[1][1])
cov_a[1][1] = cov_a[0][0]
print("part a")
classification(cov_a, cov_a)

#======Part b =========
cov_b = np.zeros((2,2))
cov_b[0][1] = 0
cov_b[1][0] = 0
cov_b[0][0] = cov[0][0]
cov_b[1][1] = cov[1][1]
print("part b")
classification(cov_b, cov_b)

#======Part c0 =========
print("part c0")
classification(cov, cov)

#======Part c1 =========
print("part c1")
classification(cov_0, cov_0)

#======Part c2 =========
print("part c2")
classification(cov_1, cov_1)

#======Part d =========
print("part d")
classification(cov_0, cov_1)

#-------------------------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================================

