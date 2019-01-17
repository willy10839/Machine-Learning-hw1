import csv
import numpy as np
import math
import matplotlib.pyplot as plt

def MatrixMinio(A,count):
    b = np.zeros((count,count))
    for i in range(count):
        for j in range(count):
            if (i+j)%2==0:
                b[i,j] = np.linalg.det(np.delete(np.delete(A,i,0),j,1))
            else:
                b[i,j] = (np.linalg.det(np.delete(np.delete(A,i,0),j,1)))*-1
    out=b.T/np.linalg.det(A)
    return out

data=[]
M1=[]
M2=[]
M3=[]
target=[]
with open('housing_training.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    next(rows)
    for row in rows:
        tmp=[]
        cc=[]
        row = list(map(float, row))
        for i in range(len(row)-1):
            tmp.append(row[i])
        data.append(tmp)
        for i in range(1):
            cc.append(row[-1])
        target.append(cc)
for p in data:
    tmp=[]
    for i in range(len(p)):
        tmp.append(p[i])
    tmp.append(1)
    M1.append(tmp)

for p in data:
    tmp=[]
    for i in range(len(p)):
        tmp.append(p[i])
        for j in range(i,len(p)):
            tmp.append(p[i]*p[j])
    tmp.append(1)
    M2.append(tmp)

for p in data:
    tmp=[]
    for i in range(len(p)):
        tmp.append(p[i])
        for j in range(i,len(p)):
            tmp.append(p[i]*p[j])
            for k in range(j,len(p)):
                tmp.append(p[i]*p[j]*p[k])
    tmp.append(1)
    M3.append(tmp)

M1matrix=np.array(M1)
M2matrix=np.array(M2)
M3matrix=np.array(M3)
tar=np.array(target)

AA1=M1matrix.T.dot(M1matrix)
AA2=M2matrix.T.dot(M2matrix)
AA3=M3matrix.T.dot(M3matrix)
zero1=np.eye(4)
zero2=np.eye(10)
zero3=np.eye(20)
lamda=0.001
#print(np.linalg.lstsq(AA1,zero1)[0])
inverse1=np.linalg.solve(AA1,zero1)
inverse2=np.linalg.solve(AA2,zero2)
inverse3=np.linalg.solve(AA3,zero3)

regular1=AA1+lamda*zero1
regular2=AA2+lamda*zero2
regular3=AA3+lamda*zero3

regular_inverse1=np.linalg.solve(regular1,zero1)
regular_inverse2=np.linalg.solve(regular2,zero2)
regular_inverse3=np.linalg.solve(regular3,zero3)

regular_weight1=regular_inverse1.dot(M1matrix.T).dot(tar)
regular_weight2=regular_inverse2.dot(M2matrix.T).dot(tar)
regular_weight3=regular_inverse3.dot(M3matrix.T).dot(tar)

#inverse1=MatrixMinio(AA1,AA1.shape[0])
#inverse2=MatrixMinio(AA2,AA2.shape[0])
#inverse3=MatrixMinio(AA3,AA3.shape[0])

weight1=(inverse1.dot(M1matrix.T)).dot(tar)# training weight M=1
weight2=(inverse2.dot(M2matrix.T)).dot(tar)# training weight M=2
weight3=(inverse3.dot(M3matrix.T)).dot(tar)# training weight M=3

test_data=[]
target_test=[]
with open('housing_training.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    next(rows)
    for row in rows:
        tmp=[]
        cc=[]
        row = list(map(float, row))
        for i in range(len(row)-1):
            tmp.append(row[i])
        test_data.append(tmp)
        for i in range(1):
            cc.append(row[-1])
        target_test.append(cc)

test_output1=[]
test_output2=[]
test_output3=[]
for p in test_data:
    tmp=[]
    for i in range(len(p)):
        tmp.append(p[i])
    tmp.append(1)
    test_output1.append(tmp)
for p in test_data:
    tmp=[]
    for i in range(len(p)):
        tmp.append(p[i])
        for j in range(i,len(p)):
            tmp.append(p[i]*p[j])
    tmp.append(1)
    test_output2.append(tmp)
for p in test_data:
    tmp=[]
    for i in range(len(p)):
        tmp.append(p[i])
        for j in range(i,len(p)):
            tmp.append(p[i]*p[j])
            for k in range(j,len(p)):
               tmp.append(p[i]*p[j]*p[k])
    tmp.append(1)
    test_output3.append(tmp)
error1=0.0
error2=0.0
error3=0.0
error1_regular=0.0
error2_regular=0.0
error3_regular=0.0
for i in range(len(test_output3)):
    aaa=np.array(test_output3[i])
    b=aaa.dot(weight3)
    sub=b-target_test[i]
    error3=error3+sub*sub
    b_regular=aaa.dot(regular_weight3)
    sub_regular=b_regular-target_test[i]
    error3_regular=error3_regular+sub_regular*sub_regular
print("RMS of regular M=3 : ", math.sqrt(error3_regular/len(test_data)))
print("RMS of M=3 : ",math.sqrt(error3/len(test_data)))
for i in range(len(test_output1)):
    aaa=np.array(test_output1[i])
    b=aaa.dot(weight1)
    sub=b-target_test[i]
    error1=error1+sub*sub
    b_regular=aaa.dot(regular_weight1)
    sub_regular=b_regular-target_test[i]
    error1_regular=error1_regular+sub_regular*sub_regular
print("RMS of regular M=1 : ", math.sqrt(error1_regular/len(test_data)))
print("RMS of M=1: ",math.sqrt(error1/len(test_data)))
for i in range(len(test_output2)):
    aaa=np.array(test_output2[i])
    b=aaa.dot(weight2)
    sub=b-target_test[i]
    error2=error2+sub*sub
    b_regular=aaa.dot(regular_weight2)
    sub_regular=b_regular-target_test[i]
    error2_regular=error2_regular+sub_regular*sub_regular
print("RMS of regular M=2 : ", math.sqrt(error2_regular/len(test_data)))
print("RMS of M=2 : ",math.sqrt(error2/len(test_data)))
output=[]
output.append(error1)
output.append(error2)
output.append(error3)
M=[]
M.append(1)
M.append(2)
M.append(3)
plt.plot(M,output)
plt.show()