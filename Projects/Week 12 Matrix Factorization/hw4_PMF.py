
from __future__ import division
import csv 
import sys
import numpy as np
#import matplotlib.pyplot as plt
np.random.seed(0)
np.set_printoptions(precision=6)

def PMF(data):   
    #Lecture 17, slide 19
    dim = 5 #matrix dimension U = Nu x dim, V = Nv x dim
    mu = 0
    variance = 1/10
    lambdaParam = 2
    printAtIteration = [10, 25, 50]
    iterationMax = 50
    objective = np.zeros((iterationMax,1))
    
    #Obtain dimension of matrix M 
    Nu = int(np.amax(data[:,0]))
    Nv = int(np.amax(data[:,1]))
        
    #Initialize Vmatrix of size Nv x dim
    Vmatrix = np.random.normal(mu, np.sqrt(1/lambdaParam), (Nv, dim))
    #Store Vmatrix in file to reproduce results
#    path1 = "Vmatrix_initial_values.csv"
#    with open(path1, "w") as file:
#        writer = csv.writer(file, delimiter=',', lineterminator='\n')
#        for val in Vmatrix:
#            writer.writerow(val)  
    #Initialize Umatrix of size Nu x dim
    Umatrix = np.zeros((Nu,dim))
    
    #Form Umatrix
    Index_Ui = []
    for i in range(Nu):
        temp1 = data[data[:,0] == i+1][:,1] # index set of objects rated by user i
        temp2 = temp1.astype(np.int64)
        Index_Ui.append(temp2)        

    #Form Vmatrix    
    Index_Vj = []
    for j in range(Nv):
        temp1 = data[data[:,1] == j+1][:,0] # index set of users who rated object j
        temp2 = temp1.astype(int)
        Index_Vj.append(temp2)
     
    #Form Mmatrix   
    Mmatrix = np.zeros((Nu,Nv))
    for val in data:
        row = int(val[0])
        col = int(val[1])
        Mmatrix[row-1,col-1] = val[2]

    for iteration in range(iterationMax):
        #Update Umatrix 
        for i in range(Nu):
            temp1 = lambdaParam*variance*np.eye(dim)
            temp2 = Vmatrix[Index_Ui[i]-1]
            temp3 = (temp2.T).dot(temp2)
            temp4 = np.linalg.inv(temp1 + temp3)
            
            temp5 = Mmatrix[i,Index_Ui[i]-1]
            temp6 = (temp2*temp5[:,None]).sum(axis=0)
    
            ui = temp4.dot(temp6)
            Umatrix[i] = ui
            
        #Update Vmatrix 
        for j in range(Nv):
            temp1 = lambdaParam*variance*np.eye(dim)
            temp2 = Umatrix[Index_Vj[j]-1]
            temp3 = (temp2.T).dot(temp2)
            temp4 = np.linalg.inv(temp1 + temp3)
            
            temp5 = Mmatrix[Index_Vj[j]-1,j]
            temp6 = (temp2*temp5[:,None]).sum(axis=0)
    
            vj = temp4.dot(temp6)
            Vmatrix[j] = vj    

        #Compute the MAP objective function
        term2 = lambdaParam*0.5*(((np.linalg.norm(Umatrix, axis=1))**2).sum())     
        term3 = lambdaParam*0.5*(((np.linalg.norm(Vmatrix, axis=1))**2).sum())
        term1 = 0
        for val in data:
            i = int(val[0])
            j = int(val[1])
            term1 = term1 + (val[2] - np.dot(Umatrix[i-1,:],Vmatrix[j-1,:]))**2
        term1 = term1/(2*variance)
        objective[iteration] = - term1 - term2 - term3             
            
        if iteration+1 in printAtIteration:
            #Write output to file
            path1 = "U-{:g}.csv".format(iteration+1)
            with open(path1, "w") as file:
                writer = csv.writer(file, delimiter=',', lineterminator='\n')
                for val in Umatrix:
                    writer.writerow(val)   
            #Write output to file
            path1 = "V-{:g}.csv".format(iteration+1)
            with open(path1, "w") as file:
                writer = csv.writer(file, delimiter=',', lineterminator='\n')
                for val in Vmatrix:
                    writer.writerow(val)             

    #Write output to file
    path1 = "objective.csv"
    with open(path1, "w") as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        for val in objective:
            writer.writerow(val) 
            
    #Plot objective function
#    plt.plot(objective[:])
#    plt.xlabel("Iteration")
#    plt.ylabel("Objective Function")
#    plt.show()
                    
def main():  
    file_X = np.genfromtxt(sys.argv[1], delimiter=',')
    
    PMF(file_X)
    
if __name__ == "__main__":
    main()