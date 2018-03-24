import numpy
from numpy import random

yangben=random.rand(100,2)
Y1 = random.rand(1,2)
Y2 = random.rand(1,2)
Y3 = random.rand(1,2)
Q1 = numpy.zeros([100,2])
Q2 = numpy.zeros([100,2])
Q3 = numpy.zeros([100,2])
flag = 1
C1 = numpy.zeros([1,2])
C2 = numpy.zeros([1,2])
C3 = numpy.zeros([1,2])
while flag == 1:
    a = 1
    j = 1
    k = 1
    for i in range(1,100,1):
      dist1 = dist = numpy.sqrt(numpy.sum(numpy.square(yangben[i,] - Y1)))
      dist2 = dist = numpy.sqrt(numpy.sum(numpy.square(yangben[i,] - Y2)))
      dist3 = dist = numpy.sqrt(numpy.sum(numpy.square(yangben[i,] - Y3)))
    if (dist1>dist2):
        minz = dist2
    else:
        minz = dist1
    if (minz > dist3):
        minz = dist3
    if (minz == dist1):
        Q1[j,] = yangben[i,]
        j = j + 1
    elif (minz == dist2):
        Q2[a,] = yangben[i,]
        a = a + 1
    elif (minz == dist3):
        Q3[k,] = yangben[i,]
        k = k + 1

    for p1 in range(1, j - 1,1):
        C1 = C1 + Q1[p1,]
    for p2 in range(1, a - 1, 1):
        C2 = C2 + Q2[p2,]
    for p3 in range(1, k - 1, 1):
        C3 = C3 + Q3[p3,]

    if (Y1==C1).all() and (Y2==C2).all() and (Y3==C3).all() :
        flag = 0
    else:
        flag = 1
        print Y1,C1,Y2,C2,Y3,C3
        Y1 = C1
        Y2 = C2
        Y3 = C3



for dr1 in  range(1,j-1,1):
    print Q1[dr1],1
for dr2 in  range(1,a-1,1):
    print Q2[dr2],2
for dr3 in  range(1,k-1,1):
    print Q3[dr3],3
print C1,C2,C3








