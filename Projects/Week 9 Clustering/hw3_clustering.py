
from __future__ import division
import csv 
import sys
import numpy as np
np.set_printoptions(precision=6)

def emGMM(data):
    #Lecture 16, slide 20
    classes = 5 #number of classes
    iterationMax = 10
    length = data.shape[0]
    dim = data.shape[1]
    Sigma_k = np.eye(dim)
    Sigma = np.repeat(Sigma_k[:,:,np.newaxis],classes,axis=2) #initialize Sigma to identity matrix   
    piClass = np.ones(classes)*(1/classes) #initialize with uniform probability distribution
    phi = np.zeros((length,classes))
    phiNorm = np.zeros((length,classes))
#    #initialize the mu with uniform random selection of data points
    indices = np.random.randint(0,length,size=classes)
    mu = data[indices]
    #initialize the mu by selecting the first few data points
#    mu = data[0:classes]
    #manually enter the mu and Sigma for sandipan_dey testset
#    mu = np.array([[-0.0407818979679,	0.350655592545],
#                  [1.03391556709,	8.99950591741],
#                  [5.92093526532,	8.10258914395]])
#    piClass = np.array([0.359950569514, 0.305602403093, 0.334447027393])
#    Sigma[:,:,0] = np.array([[0.766841596490151,    0.15545615964593],
#                            [0.15545615964593,      2.70346004231149]])
#    Sigma[:,:,1] = np.array([[4.48782867349354,	1.69862708779012],
#                            [1.69862708779012,	3.18750711550936]])
#    Sigma[:,:,2] = np.array([[4.26557534301669,	1.29968325221235],
#                            [1.29968325221235,	4.32868108538196]])

    
    for iteration in range(iterationMax):
        #compute expectation step of EM algorithm
        for k in range(classes):
            invSigma_k = np.linalg.inv(Sigma[:,:,k])
            invSqrDetSigma_k = (np.linalg.det(Sigma[:,:,k]))**-0.5
            for index in range(length):
                xi = data[index,:]
                temp1 = (((xi-mu[k]).T).dot(invSigma_k)).dot(xi-mu[k])
                phi[index, k] = piClass[k]*((2*np.pi)**(-dim/2))*invSqrDetSigma_k*np.exp(-0.5*temp1)
            for index in range(length):
                tot = phi[index,:].sum()
                phiNorm[index,:] = phi[index,:]/float(tot)
        
        #compute maximization step of EM algorithm
        nK = np.sum(phiNorm,axis=0)
        piClass = nK/float(length)
        for k in range(classes):
            mu[k] = ((phiNorm[:,k].T).dot(data))/nK[k]
        for k in range(classes):
            temp1 = np.zeros((dim,1))
            temp2 = np.zeros((dim,dim))
            for index in range(length):
                xi = data[index,:]
                temp1[:,0] = xi - mu[k]                
                temp2 = temp2 + phiNorm[index,k]*np.outer(temp1,temp1)
            Sigma[:,:,k] = temp2/float(nK[k]) 
            #vectorized operation
#            temp1 = data - mu[k]
#            temp2 = np.diag(phiNorm[:,k])
#            Sigma[:,:,k] = ((temp1.T).dot(temp2)).dot(temp1)/float(nK[k])

        #Write output to file
        path1 = "pi-{:g}.csv".format(iteration+1)
        with open(path1, "w") as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for val in piClass:
                writer.writerow([val])
        #Write output to file
        path1 = "mu-{:g}.csv".format(iteration+1)
        with open(path1, "w") as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for val in mu:
                writer.writerow(val)
        #Write output to file
        for k in range(classes):
            path1 = "Sigma-{:g}-{:g}.csv".format(k+1,iteration+1)
            with open(path1, "w") as file:
                writer = csv.writer(file, delimiter=',', lineterminator='\n')
                for val in Sigma[:,:,k]:
                    writer.writerow(val)

def kMeans(data):   
    #Lecture 14, slide 15
    cNum = 5 #number of clusters
    iterationMax = 10
    length = data.shape[0]
    c = np.zeros(length) #cluster assignment vector
    #initialize the mu with uniform random selection of data points
    indices = np.random.randint(0,length,size=cNum)
    mu = data[indices]
    #initialize the mu by selecting the first few data points
#    mu = data[0:cNum]
    #manually enter the mu for sandipan_dey testset
#    mu = np.array([[5.72316172633, 7.03506602245], 
#                   [0.0887880161461, 3.5291769851], 
#                   [5.5084357544, 10.4242009312]])
    
    for iteration in range(iterationMax):
        #Update cluster assignments ci
        for i, xi in enumerate(data):
            temp1 = np.linalg.norm(mu-xi,2,1)
            c[i] = np.argmin(temp1)
        #Update cluster mu
        n = np.bincount(c.astype(np.int64),None,cNum)      
        for k in range(cNum):
            indices = np.where(c == k)[0]
            mu[k] = (np.sum(data[indices],0))/float(n[k])
        #Write output to file
        path1 = "centroids-{:g}.csv".format(iteration+1)
        with open(path1, "w") as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for val in mu:
                writer.writerow(val)
                
def main():  
    file_X = np.genfromtxt(sys.argv[1], delimiter=',')
    
    kMeans(file_X)
    emGMM(file_X)
    
if __name__ == "__main__":
    main()