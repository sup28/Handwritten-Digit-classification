import numpy as np

# function for calculation of coefficient of given order polynomial and fitted values of wages w.r.t given parameter
def regression(y,x):
    d = np.zeros((len(data), n+1))
    for i in range(len(data)):
        for j in range(n+1):
            d[i][j] = x[i]**j
    print (d)
    w = np.matmul(np.linalg.inv(np.matmul(d.T, d)),np.matmul(d.T, y))
    
    y_calculated = np.zeros(len(x))
    mse = 0.0
    for i in range(len(data)):
        for j in range(n+1):
            y_calculated[i] += w[j]*(x[i]**j)
        mse += (y_calculated[i] - y[i])**2
        
    print ("w vector :", w)
    print("\n")
    print("calculated y using regression : ", y_calculated)
    print("\n")
    print("mean square error: ", mse)
    return y_calculated

# importing data from csv file
data = np.genfromtxt('Wage_dataset.csv',delimiter=',')

# separating the data as a given features for the imported data

wage = []
age = []
year = []
education = []
logwage = []

for row in data:
    wage.append(row[10])
    age.append(row[1])
    year.append(row[0])
    education.append(row[4])
    logwage.append(row[9])

# taking order of polynomial as a input
n = int(input("order of polynomial: "))

# part a
wage_calculated = regression(wage, year)

#part b
wage_calculated = regression(wage, age)

#part c
wage_calculated = regression(wage, education)

#=========================plotting======================================
import matplotlib.pyplot as plt

X= year
Y = wage


plt.scatter(X,Y,s=1)
plt.xlabel("year")
plt.ylabel("wages")
plt.title("year Vs wages")
plt.show()



Y= wage_calculated
plt.scatter(X,Y,s=1)
plt.xlabel("year")
plt.ylabel("wages")
plt.title("year Vs wages")
plt.show()
#-------------------------------------------------------------------------

X= age
Y = wage


plt.scatter(X,Y,s=1)
plt.xlabel("age")
plt.ylabel("wages")
plt.title("age Vs wages")
plt.show()



Y= wage_calculated
plt.scatter(X,Y,s=1)
plt.xlabel("age")
plt.ylabel("wages")
plt.title("age Vs wages")
plt.show()
 #------------------------------------------------------------------------
    
X= education
Y = wage


plt.scatter(X,Y,s=1)
plt.xlabel("year")
plt.ylabel("wages")
plt.title("year Vs wages")
plt.show()



Y= wage_calculated
plt.scatter(X,Y,s=1)
plt.xlabel("year")
plt.ylabel("wages")
plt.title("year Vs wages")
plt.show()           
