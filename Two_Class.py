import csv
import math

import numpy as np
from matplotlib import pyplot as plt


class TwoClass:

    def __init__(self, n, m, class_num):
        """ radial basis function network
        # Arguments
            input_shape: dimension of the input data
            e.g. scalar functions have should have input_dimension = 1
            hidden_shape: the number
            hidden_shape: number of hidden radial basis functions,
            also, number of centers.
        """

        self.m = m
        self.centers = np.array([])
        self.weights = None
        self.n = n  # dimension of data
        self.chromosome = np.zeros(shape=(m, n + 1))
        self.G = None
        self.prediction = np.array([])
        self.L = None
        self.counter = 20
        self.error = None
        self.y = None
        self.classNum = class_num
        self.max_y = []
        self.max_prediction = []
        self.center = None

    def input_train(self):
        with open('E:\Computational_intelligence\RBF\data21500.csv') as file:
            reader = csv.reader(file, delimiter=',')
            self.y = np.array([])
            self.x = np.array([])
            counter = 0
            mylist = []
            mylist2 = []
            mylist3 = []
            maxx = 0

            for row in reader:
                counter += 1
                temp = []

                for j in range(0, len(row) - 1):
                    temp.append(float(row[j]))
                    maax = max(temp)
                    if maax > maxx:
                        maxx = maax
                mylist.append(temp)
                if self.classNum == 2:
                    mylist2.append([float(row[len(row) - 1])])
                elif self.classNum != 2:
                    mylist3.append(float(row[len(row) - 1]))

            self.L = counter
            self.x = np.array(mylist)
            self.x /= np.max(self.x)

            y_matrix = [[0 for x in range(self.classNum)] for y in range(self.L)]
            self.y = np.array(y_matrix)

            if self.classNum != 2:
                for i in range(0, self.y.shape[0]):
                    for j in range(0, self.y.shape[1]):

                        if j == (mylist3[i] - 1):
                            self.y[i][j] = 1
                            self.max_y.append(j + 1)

                        else:
                            self.y[i][j] = 0
            elif self.classNum == 2:

                for i in range(0, self.y.shape[0]):
                    if mylist2[i] == [-1]:
                        self.y[i][0] = 1
                        self.y[i][1] = 0
                        self.max_y.append(0)
                    else:
                        self.y[i, 0] = 0
                        self.y[i, 1] = 1
                        self.max_y.append(1)

            self.G = np.zeros(shape=(self.L, self.m))

    def input_test(self):
        self.max_y = []
        with open('E:\Computational_intelligence\RBF\data25000.csv') as file:
            reader = csv.reader(file, delimiter=',')
            self.y = np.array([])
            self.x = np.array([])
            counter = 0
            mylist = []
            mylist2 = []
            mylist3 = []
            maxx = 0

            for row in reader:
                counter += 1
                temp = []

                for j in range(0, len(row) - 1):
                    temp.append(float(row[j]))
                    maax = max(temp)
                    if maax > maxx:
                        maxx = maax
                mylist.append(temp)
                mylist2.append([float(row[len(row) - 1])])
                mylist3.append(float(row[len(row) - 1]))
            self.L = counter
            self.x = np.array(mylist)
            for i in range(0, self.x.shape[0]):
                self.x[i] = self.x[i] / maxx

            y_matrix = [[0 for x in range(self.classNum)] for y in range(self.L)]
            self.y = np.array(y_matrix)

            if self.classNum != 2:

                for i in range(0, self.y.shape[0]):
                    for j in range(0, self.y.shape[1]):
                        if j == mylist3[i] - 1:
                            self.y[i][j] = 1
                            self.max_y.append(j + 1)

                        else:
                            self.y[i][j] = 0
                # print(self.y)
            elif self.classNum == 2:

                for i in range(0, self.y.shape[0]):
                    if mylist2[i] == [-1]:
                        self.y[i][0] = 1
                        self.y[i][1] = 0
                        self.max_y.append(0)
                    else:
                        self.y[i, 0] = 0
                        self.y[i, 1] = 1
                        self.max_y.append(1)

            self.G = np.zeros(shape=(self.L, self.m))

    def initiate_chrome(self, individual):
        row = 0
        temp = np.zeros(shape=(self.n + 1))
        for i in range(0, np.array(individual).size):
            temp[i % (self.n + 1)] = individual[i]

            if ((i + 1) % (self.n + 1)) == 0:
                self.chromosome[row] = temp  # whole row
                row += 1

    def kernel_function(self):
        for j in range(0, self.L):
            for i in range(0, self.m):
                v_i = self.chromosome[i, 0:(self.n)]
                difference = np.subtract(self.x[j].astype(np.float), v_i)
                transpose = np.transpose(difference)
                mull = np.matmul(transpose, difference)
                temp = (-1) * self.chromosome[i, self.n] * mull
                self.G[j, i] = math.exp(temp)

    def fit(self):
        temp = np.matmul(np.transpose(self.G), self.y)
        self.weights = np.matmul(np.linalg.pinv(np.matmul(np.transpose(self.G), self.G)), temp)

    def predict(self):
        self.max_prediction = []
        self.prediction = np.matmul(self.G, self.weights)

        if self.classNum == 2:
            self.max_prediction = np.argmax(self.prediction, axis=1)
        elif self.classNum != 2:
            self.max_prediction = np.argmax(self.prediction, axis=1) + 1

    def error_calc(self):

        temp_y = np.array(self.max_y)
        temp_prediction = np.array(self.max_prediction)
        sub = np.subtract(temp_prediction, temp_y)
        sum = 0
        for i in range(0, self.L):
            if sub[i] > 0:
                sum += 1
        self.error = sum / self.L

        return self.error

    def runRBF(self, individual):
        individualnp = np.array(individual)
        self.initiate_chrome(individualnp)
        self.kernel_function()  # G
        self.fit()  # w
        self.predict()  # y^
        self.error = self.error_calc()
        return self.error,  # L

    def plot_train(self):
        temp1 = []
        temp2 = []
        for i in range(0, self.L):
            temp1.append(self.x[i][0])
            temp2.append(self.x[i][1])
        xValue1 = np.array(temp1)
        xValue2 = np.array(temp2)

        plt.title("Plot y and y'")
        plt.xlabel("X1")
        plt.ylabel("X2")
        arr1 = []
        arr2 = []
        arr3 = []
        arr4 = []
        for i in range(0, self.L):
            if self.max_prediction[i] == self.max_y[i]:
                arr1.append(xValue1[i])
                arr2.append((xValue2[i]))

            else:
                arr3.append(xValue1[i])
                arr4.append((xValue2[i]))

        plt.plot(arr1, arr2, 'ro', color='green')
        plt.plot(arr3, arr4, 'ro', color='red')

        arr5 = []
        arr6 = []
        arr7 = []
        for i in range(0, self.m):
            arr7.append(self.chromosome[i, self.n])
            for j in range(0, self.n):
                if j == 0:
                    arr5.append(self.chromosome[i, j])
                else:
                    arr6.append(self.chromosome[i, j])

        for i in range(0, self.m):
            circle = plt.Circle((arr5[i], arr6[i]), arr7[i], facecolor='none', edgecolor='black')
            ax = plt.gca()
            ax.add_patch(circle)
            plt.axis('scaled')

        plt.plot(arr5, arr6, 'ro', color='blue')
        plt.show()

    def plot_test(self):
        temp1 = []
        temp2 = []
        for i in range(0, self.L):
            temp1.append(self.x[i][0])
            temp2.append(self.x[i][1])
        xValue1 = np.array(temp1)
        xValue2 = np.array(temp2)

        plt.title("Plot y and y'")
        plt.xlabel("X1")
        plt.ylabel("X2")
        arr1 = []
        arr2 = []
        arr3 = []
        arr4 = []
        for i in range(0, self.L):
            if self.max_prediction[i] == self.max_y[i]:
                arr1.append(xValue1[i])
                arr2.append((xValue2[i]))

            else:
                arr3.append(xValue1[i])
                arr4.append((xValue2[i]))
        print(len(arr1))
        plt.plot(arr1, arr2, 'ro', color='green')
        plt.plot(arr3, arr4, 'ro', color='red')

        arr5 = []
        arr6 = []
        arr7 = []
        for i in range(0, self.m):
            arr7.append(self.chromosome[i, self.n])
            for j in range(0, self.n):
                if j == 0:
                    arr5.append(self.chromosome[i, j])
                else:
                    arr6.append(self.chromosome[i, j])

        for i in range(0, self.m):
            circle = plt.Circle((arr5[i], arr6[i]), arr7[i], facecolor='none', edgecolor='black')
            ax = plt.gca()
            ax.add_patch(circle)
            plt.axis('scaled')

        plt.plot(arr5, arr6, 'ro', color='blue')
        plt.show()
