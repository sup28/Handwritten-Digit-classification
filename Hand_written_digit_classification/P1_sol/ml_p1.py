import numpy as np
import math as m
from numpy.linalg import inv


data = np.genfromtxt('P1_data_train.csv',delimiter=',') 				# importing training data from csv file
label = np.genfromtxt('P1_labels_train.csv',delimiter=',')				# importing training labels from csv file
#-------------------------------------------------------------------------------------------------------------------------------------------
																		# class formation of training data
class5=[]	
class6=[]
p_5 = 0
p_6 = 0

for i in range(len(label)):
	if label[i]%5 == 0:
		class5.append(data[i])
		p_5 += 1
	else:
		class6.append(data[i])
		p_6 += 1
#-------------------------------------------------------------------------------------------------------------------------------------------
																		# mean calculation of class5 and class6
mean_5 = sum(class5)/len(class5)										
mean_6 = sum(class6)/len(class6)	

#-------------------------------------------------------------------------------------------------------------------------------------------
																		#covariance of class5
sum5= np.zeros((64,64))
for i in range(len(class5)):
	d = np.outer(class5[i],class5[i])
	sum5= np.add(sum5,d)

sum5 = sum5/len(class5)
cov_5 = sum5- np.outer(mean_5, mean_5)	
#-------------------------------------------------------------------------------------------------------------------------------------------

																		#covariance of class6
sum6= np.zeros((64,64))
for i in range(len(class6)):
	d = np.outer(class6[i],class6[i])
	sum6= np.add(sum6,d)

sum6 = sum6/len(class6)
cov_6 = sum6- np.outer(mean_6, mean_6)
#-------------------------------------------------------------------------------------------------------------------------------------------
																		# calculation of prior probabilities
p_5 = p_5/len(label)
p_6 = p_6/len(label)																		

#-------------------------------------------------------------------------------------------------------------------------------------------


test_data = np.genfromtxt('P1_data_test.csv',delimiter=',') 			# importing test data from csv file
test_label = np.genfromtxt('P1_labels_test.csv',delimiter=',')			# importing test labels from csv file

#-------------------------------------------------------------------------------------------------------------------------------------------
																		#classification of test data
g = 0
f = 0
assg_labels = []
for i in range(len(test_data)):
	g =  -0.5*m.log(np.linalg.det(cov_5)) - 0.5*np.matmul(np.transpose(test_data[i]-mean_5), np.matmul(inv(cov_5), (test_data[i]-mean_5))) + m.log(p_5)
	f =  -0.5*m.log(np.linalg.det(cov_6)) - 0.5*np.matmul(np.transpose(test_data[i]-mean_6), np.matmul(inv(cov_6), (test_data[i]-mean_6))) + m.log(p_6)
	if g>f :
		assg_labels.append(5)
	else:
		assg_labels.append(6)	

#-------------------------------------------------------------------------------------------------------------------------------------------
																		# populating confusion matrix
a = 0
b = 0
c = 0
d = 0
x=0
y=0

for i in range(len(assg_labels)):
	if test_label[i] == 5 and assg_labels[i] == 5:
		a = a+1
	elif test_label[i] == 5 and assg_labels[i] == 6:
		b = b+1	
	elif test_label[i] == 6 and assg_labels[i] == 6:
		c = c+1
	else :
		d = d+1		

x = a*100/(a+b)
y = c*100/(c+d)
		
print ("For case 1 :")	
print("\n")
print ("Confusion matrix : ",np.matrix([[a,d],[b,c]]))
print("\n")
print ("% classification of 5 and 6 resp.: " , x, y)
print("\n\n")

#-------------------------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================================
#==========================================================case_2===========================================================================
#====================================================neglecting determinant====================================================================
#-------------------------------------------------------------------------------------------------------------------------------------------
																		#classification of test data
g = 0
f = 0
assg_labels = []
for i in range(len(test_data)):
	g =   - 0.5*np.matmul(np.transpose(test_data[i]-mean_5), np.matmul(inv(cov_5), (test_data[i]-mean_5))) + m.log(p_5)
	f =   - 0.5*np.matmul(np.transpose(test_data[i]-mean_6), np.matmul(inv(cov_6), (test_data[i]-mean_6))) + m.log(p_6)
	if g>f :
		assg_labels.append(5)
	else:
		assg_labels.append(6)	

#-------------------------------------------------------------------------------------------------------------------------------------------
																		# populating confusion matrix
a = 0
b = 0
c = 0
d = 0
x=0
y=0

for i in range(len(assg_labels)):
	if test_label[i] == 5 and assg_labels[i] == 5:
		a = a+1
	elif test_label[i] == 5 and assg_labels[i] == 6:
		b = b+1	
	elif test_label[i] == 6 and assg_labels[i] == 6:
		c = c+1
	else :
		d = d+1		

x = a*100/(a+b)
y = c*100/(c+d)
		
print ("For case 2 :")	
print("\n")
print ("Confusion matrix : ",np.matrix([[a,d],[b,c]]))
print("\n")
print ("% classification of 5 and 6 resp.: " , x, y)
print("\n\n")
#-------------------------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================================






#-------------------------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================================
#==========================================================case_3===========================================================================
#====================================================Cov_5 = Cov_6 ====================================================================
#------------------------------------------------------------------------------------------------------------------------------------------

cov = cov_6
																		# Discrimination function and classification
g1 = 0
f1 = 0
assg_labels_1 = []
for i in range(len(test_data)):
	g1 =  -0.5*m.log(np.linalg.det(cov)) - 0.5*np.matmul(np.transpose(test_data[i]-mean_5), np.matmul(inv(cov), (test_data[i]-mean_5))) + m.log(p_5)
	f1 =  -0.5*m.log(np.linalg.det(cov)) - 0.5*np.matmul(np.transpose(test_data[i]-mean_6), np.matmul(inv(cov), (test_data[i]-mean_6))) + m.log(p_6)
	if g1>f1 :
		assg_labels_1.append(5)
	else:
		assg_labels_1.append(6)	
		

																		# populating confusion matrix
a1 = 0
b1 = 0
c1 = 0
d1 = 0
for i in range(len(assg_labels_1)):
	if test_label[i] == 5 and assg_labels_1[i] == 5:
		a1 = a1+1
	elif test_label[i] == 5 and assg_labels_1[i] == 6:
		b1 = b1+1	
	elif test_label[i] == 6 and assg_labels_1[i] == 6:
		c1 = c1+1
	else :
		d1 = d1+1		
print ("For case 3 :")	
print("\n")
print ("Confusion matrix : ",np.matrix([[a1,d1],[b1,c1]]))
print("\n")


x = a1*100/(a1+b1)
y = c1*100/(c1+d1)
		
print ("% classification of 5 and 6 resp.: " , x, y)
print("\n\n")
#-------------------------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================================







	
#-------------------------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================================
#==========================================================case_4===========================================================================
#====================================================Cov_6 = Cov_5====================================================================
#-------------------------------------------------------------------------------------------------------------------------------------------

																		# Now taking same covariance matrix ; this matrix is weighted mean of cov_5 and cov_6	
cov = cov_5


																		# Discrimination function and classification
g1 = 0
f1 = 0
assg_labels_1 = []
for i in range(len(test_data)):
	g1 =  -0.5*m.log(np.linalg.det(cov)) - 0.5*np.matmul(np.transpose(test_data[i]-mean_5), np.matmul(inv(cov), (test_data[i]-mean_5))) + m.log(p_5)
	f1 =  -0.5*m.log(np.linalg.det(cov)) - 0.5*np.matmul(np.transpose(test_data[i]-mean_6), np.matmul(inv(cov), (test_data[i]-mean_6))) + m.log(p_6)
	if g1>f1 :
		assg_labels_1.append(5)
	else:
		assg_labels_1.append(6)	
		

																		# populating confusion matrix
a1 = 0
b1 = 0
c1 = 0
d1 = 0
for i in range(len(assg_labels_1)):
	if test_label[i] == 5 and assg_labels_1[i] == 5:
		a1 = a1+1
	elif test_label[i] == 5 and assg_labels_1[i] == 6:
		b1 = b1+1	
	elif test_label[i] == 6 and assg_labels_1[i] == 6:
		c1 = c1+1
	else :
		d1 = d1+1		
print ("For case 4 :")	
print("\n")
print ("Confusion matrix : ",np.matrix([[a1,d1],[b1,c1]]))
print("\n")


x = a1*100/(a1+b1)
y = c1*100/(c1+d1)
		
print ("% classification of 5 and 6 resp.: " , x, y)
print("\n\n")
#-------------------------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================================

#-------------------------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================================
#==========================================================case_5===========================================================================
#====================================================Cov_5 = Cov_6 = Cov====================================================================
#-------------------------------------------------------------------------------------------------------------------------------------------

																		# Now taking same covariance matrix ; this matrix is weighted mean of cov_5 and cov_6	
cov = p_5*cov_5 + p_6*cov_6


																		# Discrimination function and classification
g1 = 0
f1 = 0
assg_labels_1 = []
for i in range(len(test_data)):
	g1 =  -0.5*m.log(np.linalg.det(cov)) - 0.5*np.matmul(np.transpose(test_data[i]-mean_5), np.matmul(inv(cov), (test_data[i]-mean_5))) + m.log(p_5)
	f1 =  -0.5*m.log(np.linalg.det(cov)) - 0.5*np.matmul(np.transpose(test_data[i]-mean_6), np.matmul(inv(cov), (test_data[i]-mean_6))) + m.log(p_6)
	if g1>f1 :
		assg_labels_1.append(5)
	else:
		assg_labels_1.append(6)	
		

																		# populating confusion matrix
a1 = 0
b1 = 0
c1 = 0
d1 = 0
for i in range(len(assg_labels_1)):
	if test_label[i] == 5 and assg_labels_1[i] == 5:
		a1 = a1+1
	elif test_label[i] == 5 and assg_labels_1[i] == 6:
		b1 = b1+1	
	elif test_label[i] == 6 and assg_labels_1[i] == 6:
		c1 = c1+1
	else :
		d1 = d1+1		
print ("For case 5 :")	
print("\n")
print ("Confusion matrix : ",np.matrix([[a1,d1],[b1,c1]]))
print("\n")


x = a1*100/(a1+b1)
y = c1*100/(c1+d1)
		
print ("% classification of 5 and 6 resp.: " , x, y)   
print("\n\n")
#-------------------------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================================






