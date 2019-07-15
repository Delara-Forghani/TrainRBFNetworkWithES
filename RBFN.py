import numpy as np
import csv
import math
from matplotlib import pyplot as plt


class RBFN:

    def __init__(self, n, m):
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
        self.ytrain = None
        self.xtrain = None

    def input_train(self):
        x = np.random.randint(0, 2000, 1200)
        x.sort()
        print(x)
        with open('E:\Computational_intelligence\RBF\ddata2000.csv') as file:
            reader = csv.reader(file, delimiter=',')
            self.y = np.array([])
            self.x = np.array([])
            counter = 0
            array = 0
            maax = 0
            maxx = 0
            samimd_data = []
            mylist = []
            mylist2 = []

            for row in reader:
                # temp = []
                #
                # while array != 1199 and x[array] == x[array + 1]:
                #     array += 1
                #
                # if counter == x[array]:
                #     for j in range(0, len(row) - 1):
                #         temp.append(float(row[j]))
                #         maax = max(temp)
                #         if maax > maxx:
                #             maxx = maax
                #     mylist.append(temp)
                #     mylist2.append([float(row[len(row) - 1])])
                #     if array != 1199:
                #         array += 1
                #
                # counter += 1
                samimd_data.append(row)
            samimd_data = np.array(samimd_data).astype(np.float)
            mylist = samimd_data[:, :-1]
            mylist2 = samimd_data[:, -1]

            self.x = mylist[x]
            self.y = mylist2[x]


            self.x /= np.max(self.x)

            maxY = np.amax(self.y)

            self.L = self.x.shape[0]
            self.G = np.zeros(shape=(self.L, self.m))

            print(self.x.shape[0])
            print(self.L)

    def input_test(self):

        with open('E:\Computational_intelligence\RBF\ddata2000.csv') as file:
            reader = csv.reader(file, delimiter=',')
            self.y = np.array([])
            self.x = np.array([])
            counter = 0
            mylist = []
            mylist2 = []
            maax = 0
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
            self.L = counter
            self.x = np.array(mylist)

            for i in range(0, self.x.shape[0]):
                self.x[i] = self.x[i] / maxx
            self.y = np.array(mylist2)
            maxY = np.amax(self.y)
            for i in range(0, self.y.shape[0]):
                self.y[i] = self.y[i] / maxY

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
        self.prediction = np.matmul(self.G, self.weights)

    def error_calc(self):
        sub = np.subtract(self.prediction, self.y)
        self.error = np.matmul(np.transpose(sub), sub) / 2
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
        xValue = np.array([])
        list = []
        for i in range(0, self.L):
            list.append([i])
            xValue = np.array(list)

        plt.title("Plot y and y'")
        plt.xlabel("Data arrangement")
        plt.ylabel("learnt Data")
        plt.scatter(xValue, self.y, color='Blue')
        plt.scatter(xValue, self.prediction, color="red")
        plt.show()

    def plot_test(self):
        xValue = np.array([])
        list = []
        for i in range(0, self.L):
            list.append([i])
            xValue = np.array(list)

        plt.title("Plot y and y'")
        plt.xlabel("Data arrangement")
        plt.ylabel("test_data")
        plt.scatter(xValue, self.y, color='Blue')
        plt.scatter(xValue, self.prediction, color="red")
        plt.show()

