
import csv 
import sys
import numpy as np
np.set_printoptions(precision=15)

# Not utilised
#def normalizeData(data):
#    mean = data[:,0:-1].mean(axis=0)
#    std = data[:,0:-1].std(axis=0)
#    data[:,0:-1] = data[:,0:-1] - mean[None,:]
#    data[:,0:-1] = data[:,0:-1]/std[None,:]
#    
#def normalizeLabel(label):
#    mean = label.mean(axis=0)
#    label = label - mean
    
def part1(lambdaNum, varNum, data, label):
    #Lecture 3, Slide 16
    dim = data.shape[1]
    temp1 = lambdaNum*np.eye(dim) + (data.T).dot(data)
    wRR = (np.linalg.inv(temp1)).dot((data.T).dot(label))
    
    return wRR
    
def updatePosterior(lambdaNum, varNum, data, dim, label, oldAutoCorr, oldCrossCorr):
    #Lecture 5, Slide 16
    oldAutoCorr = (data.T).dot(data) + oldAutoCorr
    oldCrossCorr = (data.T).dot(label) + oldCrossCorr

    covInv = lambdaNum*np.eye(dim) + (1/varNum)*oldAutoCorr
    cov = np.linalg.inv(covInv)
    
    temp1 = lambdaNum*varNum*np.eye(dim) + oldAutoCorr
    mean = (np.linalg.inv(temp1)).dot(oldCrossCorr)

    return cov, mean, oldAutoCorr, oldCrossCorr
    
def part2(lambdaNum, varNum, data, label, dataTest): 
    dim = data.shape[1]
    active = []
    oldAutoCorr = np.zeros((dim,dim))
    oldCrossCorr = np.zeros(dim)

    cov, mean, oldAutoCorr, oldCrossCorr = updatePosterior(lambdaNum, varNum, data, dim, label, oldAutoCorr, oldCrossCorr)
    #Lecture 5, slide 9
    wRR = mean
    
    #1-based indexes
    indices = list(range(dataTest.shape[0]))
    for i in range(0,10):
        #Lecture 5, Slide 18
        varMatrix = (dataTest.dot(cov)).dot(dataTest.T)
        row = np.argmax(varMatrix.diagonal())
        data = dataTest[row,:]
#        print(varMatrix[row,row])
        #Lecture 5, slide 12
        label = data.dot(wRR) 
        #Build active learning sequence
        actualRow = indices[row]
        active.append(actualRow)
        #Remove x0 and corresponding index
        dataTest = np.delete(dataTest,(row),axis=0)
        indices.pop(row)
        #Update posterior distribution
        cov, mean, oldAutoCorr, oldCrossCorr = updatePosterior(lambdaNum, varNum, data, dim, label, oldAutoCorr, oldCrossCorr)
        
        #Lecture 5, slide 9
        wRR = mean

    #1-based indexes to pass Vocareum   
    active = [j+1 for j in active]         
    return active
    
def main():  
    lambdaNum = float(sys.argv[1])
    varNum = float(sys.argv[2])
    file_X_train = np.genfromtxt(sys.argv[3], delimiter=',')
    file_y_train = np.genfromtxt(sys.argv[4], delimiter=',')
    file_X_test = np.genfromtxt(sys.argv[5], delimiter=',')
    
    #Normalize data and label
#    normalizeData(file_X_train)
#    normalizeLabel(file_y_train)
#    normalizeData(file_X_test)

    #Compute wRR for part 1
    wRR = part1(lambdaNum, varNum, file_X_train, file_y_train)
#    print(wRR)
    
    #Compute active learning sequence for part 2
    active = part2(lambdaNum, varNum, file_X_train, file_y_train, file_X_test.copy())
#    print(active)
    
    #Write output to file
    path1 = "wRR_{:g}.csv".format(lambdaNum)
    path2 = "active_{:g}_{:g}.csv".format(lambdaNum,varNum)
    with open(path1, "w") as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        for val in wRR:
            writer.writerow([val])
        
    with open(path2, "w") as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(active)
    
if __name__ == "__main__":
    main()