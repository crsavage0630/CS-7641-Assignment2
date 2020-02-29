import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
from time import clock
import os
import argparse

import data.DataProcessors as dp
import seaborn as sns
import matplotlib.pyplot as plt


os.environ['seed'] = '45604'
randomSeed = 45604
verbose = True
from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix, make_scorer,  precision_score, mean_squared_error, plot_confusion_matrix, roc_auc_score, recall_score


def plot_results(data_dir, param_name, param_display):
    directory="./"+data_dir+"/images/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    path1='./'+data_dir
    path2= "./"+data_dir+"/images/"
    
    # nn
    ga = pd.read_csv(os.path.join(data_dir,'gatrain_performance.csv'))
    sa = pd.read_csv(os.path.join(data_dir,'satrain_performance.csv'))
    rh = pd.read_csv(os.path.join(data_dir,'rhtrain_performance.csv'))
    gd = pd.read_csv(os.path.join(data_dir,'gdtrain_performance.csv'))

    plt.close()
    plt.figure() 
    plt.plot( ga['Iterations'], ga[param_name], label='Gen Alg')
    plt.plot( sa['Iterations'], sa[param_name], label='Sim Ann')
    plt.plot( rh['Iterations'], rh[param_name], label='Random Hill')
    plt.plot( gd['Iterations'], gd[param_name], label='Grad Desc')
    
    plt.legend(title="Algorithm", loc="best")
    x_title = "Iterations"
    y_title = param_display
    plt.xlabel(x_title)
    plt.ylabel(y_title)
      

    plt.title("Customer Churn ANN Optimized by RO Algorithms (Train Performance)") 
    plt.savefig(os.path.join(directory,"train_"+param_name+".png"), format='png', dpi=200, bbox_inches = 'tight', pad_inches = 0) 

    ga = pd.read_csv(os.path.join(data_dir,'gatest_performance.csv'))
    sa = pd.read_csv(os.path.join(data_dir,'satest_performance.csv'))
    rh = pd.read_csv(os.path.join(data_dir,'rhtest_performance.csv'))
    gd = pd.read_csv(os.path.join(data_dir,'gdtest_performance.csv'))

    plt.close()
    plt.figure() 
    plt.plot( ga['Iterations'], ga[param_name], label='Gen Alg')
    plt.plot( sa['Iterations'], sa[param_name], label='Sim Ann')
    plt.plot( rh['Iterations'], rh[param_name], label='Random Hill')
    plt.plot( gd['Iterations'], gd[param_name], label='Grad Desc')
    
    plt.legend(title="Algorithm", loc="best")
    x_title = "Iterations"
    y_title = param_display
    plt.xlabel(x_title)
    plt.ylabel(y_title)
      

    plt.title("Customer Churn ANN Optimized by RO Algorithms (Test Performance)") 
    plt.savefig(os.path.join(directory,"test_"+param_name+".png"), format='png', dpi=200, bbox_inches = 'tight', pad_inches = 0) 



def get_model(algorithm, max_iters):
    activation = "relu"
    print(algorithm)
    print(max_iters)
    if algorithm == "rh":
        return mlrose.NeuralNetwork(hidden_nodes = [10], activation = activation, algorithm = 'random_hill_climb',  \
                                bias = True,  is_classifier = True, early_stopping = True, restarts = 5, max_attempts =10,
                                max_iters = max_iters, clip_max = 10, random_state = randomSeed)
    if algorithm == "ga":
        return mlrose.NeuralNetwork(hidden_nodes = [10], activation = activation, algorithm = 'genetic_alg',  \
                                bias = True,  is_classifier = True, early_stopping = True,  max_attempts =10,
                                max_iters = max_iters, clip_max = 10, mutation_prob = .10, random_state = randomSeed)
    if algorithm == "sa":
        return mlrose.NeuralNetwork(hidden_nodes = [10], activation = activation, algorithm = 'simulated_annealing',  \
                                bias = True,  is_classifier = True, early_stopping = True,  max_attempts =10,
                                max_iters = max_iters, clip_max = 10, schedule = mlrose.GeomDecay(), random_state = randomSeed)    

    if algorithm == "gd":
        return mlrose.NeuralNetwork(hidden_nodes = [10], activation = activation, algorithm = 'gradient_descent',  \
                                bias = True,  is_classifier = True, early_stopping = True,  max_attempts =10,
                                max_iters = max_iters, clip_max = 10, random_state = randomSeed)                                                                                            

def run_neural_net(algorithm):
    fullData = dp.CustomerChurnModel()
    fullData.prepare_data_for_training()


    dfTrain = pd.DataFrame(columns=["Iterations","Accuracy","Precision","Recall","F1","ROC AUC","SquareError","TrainTime"])
    dfTest = pd.DataFrame(columns=["Iterations","Accuracy","Precision","Recall","F1","ROC AUC","SquareError","TrainTime"])

    iterations = np.geomspace(10, 5100, num=40, dtype=int)
    index = 0
    for iteration in iterations:
        print(iteration)

        nn_model1 = get_model(algorithm, iteration.item())
        start = clock()
        nn_model1.fit(fullData.trainX, fullData.trainY)
        end = clock()
        y_train_pred = nn_model1.predict(fullData.trainX)

        y_train_accuracy = accuracy_score(fullData.trainY, y_train_pred)
        print(y_train_accuracy)
        y_test_pred = nn_model1.predict(fullData.testX)
        y_test_accuracy = accuracy_score(fullData.testY, y_test_pred)
        print(y_test_accuracy)
        dfTrain.loc[index] = [iteration, accuracy_score(fullData.trainY, y_train_pred), precision_score(fullData.trainY, y_train_pred), recall_score(fullData.trainY, y_train_pred), f1_score(fullData.trainY, y_train_pred),roc_auc_score(fullData.trainY, y_train_pred),mean_squared_error(fullData.trainY, y_train_pred),end-start]
        dfTest.loc[index] = [iteration, accuracy_score(fullData.testY, y_test_pred), precision_score(fullData.testY, y_test_pred), recall_score(fullData.testY, y_test_pred), f1_score(fullData.testY, y_test_pred),roc_auc_score(fullData.testY, y_test_pred),mean_squared_error(fullData.testY, y_test_pred),0]
        index = index + 1


    dfTrain.to_csv('{}{}'.format(algorithm,'train_performance.csv'))
    dfTest.to_csv('{}{}'.format(algorithm,'test_performance.csv'))    


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    
    # Adding optional argument 
    parser.add_argument("-g", "--ga", help = "Run GA", default='y') 
    parser.add_argument("-s", "--sa", help = "Run SA", default='y') 
    parser.add_argument("-r", "--rh", help = "Run RH", default='y') 
    parser.add_argument("-d", "--gd", help = "Run Gradient Descent", default='y') 
    parser.add_argument("-p", "--plot", help = "Plot", default='y') 
    # Read arguments from command line 
    args = parser.parse_args() 
    print(args)   
    if (args.ga == 'y'):
        run_neural_net("ga")

    if (args.sa == 'y'):
        run_neural_net("sa")    

    if (args.rh == 'y'):
        run_neural_net("rh")     
        
    if (args.gd == 'y'):
        run_neural_net("gd")          

    if (args.plot == 'y'):
        plot_results(".","Accuracy","Accuracy")
        plot_results(".","SquareError","Square Error")
        plot_results(".","TrainTime","Training Time")







