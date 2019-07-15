# TrainRBFNetworkWithES
In this project I wanted to implement a RBF neural network and with the evolutionary strategy as the learning method. We have two kinds of learning, regression and classification. Here, I used a tool in python called "Deap" for implementing ES. In each iteration ES improves the center and the ratio of the RBF network. I get the output chromosome(individual) of ES. The output chromosome ia an m*(n+1) dimensional array (m is the number of clusters and n is the dimension of a center and 1 inidcates the value of the ratio).  
In order to implement the RBF NN I used some mathematical formulas which are added in "report.pdf" File.

