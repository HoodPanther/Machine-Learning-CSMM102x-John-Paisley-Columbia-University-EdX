
from __future__ import division
import csv 
import sys
import numpy as np
np.set_printoptions(precision=6)
    
def classPrior_ConditionalDensity(data,label,classes):
    #Lecture 7, Slide 27/30
    length = data.shape[0]
    dim = data.shape[1]
    cov = np.zeros((dim,dim,classes))
    mu= np.zeros((classes,dim))
    
    unique, counts = np.unique(label, return_counts=True)
    pi = (counts/float(length)).T
#    print(np.asarray((unique, counts)).T)
#    print(pi)
    
    for i in range(classes):
        xi = data[(label == i)]
        mu[i] = np.mean(xi, axis=0)
#        print(mu[i])
        xiNormalised = xi-mu[i]
        tempCov = (xiNormalised.T).dot(xiNormalised)
        cov[:,:,i] = tempCov/float(len(xi))
#        print(cov[:,:,i])
#        print("--------------next next--------------")
    
    return pi, mu, cov
    
def plugInClassifier(data, pi, mu, cov, classes): 
    #Lecture 7, slide 27/30
    length = data.shape[0]
    prob = np.zeros((length,classes))
    probNorm = np.zeros((length,classes))
    
    for k in range(classes):
        invCov = np.linalg.inv(cov[:,:,k])
        invSqrDetCov = (np.linalg.det(cov[:,:,k]))**-0.5
        for index in range(length):
            x0 = data[index,:]
            temp1 = (((x0-mu[k]).T).dot(invCov)).dot(x0-mu[k])
            prob[index, k] = pi[k]*invSqrDetCov*np.exp(-0.5*temp1)
#        print(prob[index, k])
#        print("--------------next next--------------")

    for index in range(length):
        tot = prob[index,:].sum()
        probNorm[index,:] = prob[index,:]/float(tot) 
#    probNorm = prob/(prob.sum(axis=1))
    
#    probNorm = np.ndarray([[1,2,3],[4,5,6]])
    return probNorm
    
def main():  
    file_X_train = np.genfromtxt(sys.argv[1], delimiter=',')
    file_y_train = np.genfromtxt(sys.argv[2], delimiter=',')
    file_X_test = np.genfromtxt(sys.argv[3], delimiter=',')

    classes = 6
    
    pi, mu, cov = classPrior_ConditionalDensity(file_X_train,file_y_train, classes)
    
    probNorm = plugInClassifier(file_X_test, pi,mu,cov,classes)
    
    #Write output to file
    path1 = "probs_test.csv"
    with open(path1, "w") as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        for val in probNorm:
            writer.writerow(val)
    
if __name__ == "__main__":
    main()