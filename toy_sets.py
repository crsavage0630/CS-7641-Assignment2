import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import clock
import os
import argparse
import seaborn as sns


randomSeed = 45604
verbose = True


def save_plot(df, key, y_col, x_col, directory, x_title, y_title, plt_title, legend_title, file_name):
    #sns.set_style(style="darkgrid")
    #sns.set_context("paper")    
    if not os.path.exists(directory):
        os.makedirs(directory)    
    plt.close()
    plt.figure() 

    print(df)
    vals=df[key].unique().tolist()
    for val in vals:
        subDf = df.loc[df[key] == val]
        plt.plot( subDf[x_col], subDf[y_col], label=str(val))
    plt.legend(title=legend_title, loc="best")
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(plt_title)    
    plt.savefig(os.path.join(directory,file_name), format='png', dpi=200, bbox_inches = 'tight', pad_inches = 0)  

def plot_summary_fitness_time(data_dir, problem_name, show_mimic=True):
    directory="./"+data_dir+"/images/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    path1='./'+data_dir
    path2= "./"+data_dir+"/images/"
    
    # ga
    ga = pd.read_csv(os.path.join(data_dir,'ga.csv')).groupby(['Iterations']).mean()
    sa = pd.read_csv(os.path.join(data_dir,'sa.csv')).groupby(['Iterations']).mean()
    rh = pd.read_csv(os.path.join(data_dir,'rhc.csv')).groupby(['Iterations']).mean()
    mi = pd.read_csv(os.path.join(data_dir,'mi.csv')).groupby(['Iterations']).mean()

    plt.close()
    plt.figure() 
    plt.plot( ga.index, ga['Best Fitness'], label='GA')
    plt.plot( sa.index, sa['Best Fitness'], label='SA')
    plt.plot( rh.index, rh['Best Fitness'], label='RHC')
    if show_mimic == True:
        plt.plot( mi.index, mi['Best Fitness'], label='MIMIC')
    plt.legend(title="Algorithm", loc="best")
    x_title = "Iterations"
    y_title = "Fitness Value"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
      
    if show_mimic == True: 
        plt.title(problem_name + " Fitness Value for GA, SA, RHC, MIMIC") 
        plt.savefig(os.path.join(directory,"fitness.png"), format='png', dpi=200, bbox_inches = 'tight', pad_inches = 0) 
    else:
        plt.title(problem_name + " Fitness Value for GA, SA, RHC") 
        plt.savefig(os.path.join(directory,"fitness_no_mimic.png"), format='png', dpi=200, bbox_inches = 'tight', pad_inches = 0) 

    plt.close()
    plt.figure() 
    plt.plot( ga.index, ga['Time'], label='GA')
    plt.plot( sa.index, sa['Time'], label='SA')
    plt.plot( rh.index, rh['Time'], label='RHC')
    if show_mimic == True:
        plt.plot( mi.index, mi['Time'], label='MIMIC')    
    plt.legend(title="Algorithm", loc="best")
    x_title = "Iterations"
    y_title = "Time"
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(problem_name + " Computation Time for GA, SA, RHC, MIMIC")    
    if show_mimic == True:
        plt.title(problem_name + " Computation Time for GA, SA, RHC, MIMIC")  
        plt.savefig(os.path.join(directory,"time.png"), format='png', dpi=200, bbox_inches = 'tight', pad_inches = 0) 
    else:
        plt.title(problem_name + " Computation Time for GA, SA, RHC")  
        plt.savefig(os.path.join(directory,"time_no_mimic.png"), format='png', dpi=200, bbox_inches = 'tight', pad_inches = 0) 


def plot_fitness_time(data_dir, problem_name):


    directory="./"+data_dir+"/images/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    path1='./'+data_dir
    path2= "./"+data_dir+"/images/"
    
    # ga
    df = pd.read_csv(os.path.join(data_dir,'ga.csv'))
    param_name = "Mutation Probability"
    x_title = "Number of Iterations"
    legend_title = param_name

    plt_title = "Genetic Algorithm "+problem_name+" Problem Fitness by "+param_name
    save_plot(df,param_name,'Best Fitness','Iterations',
                    path2,x_title,"Fitness Value", plt_title,legend_title,"ga_params.png")

    plt_title = "Genetic Algorithm "+problem_name+" Problem Time by "+param_name
    save_plot(df,param_name,'Time','Iterations',
                    path2,x_title,"Time", plt_title,legend_title,"ga_time.png")

    # sa
    df = pd.read_csv(os.path.join(data_dir,'sa.csv'))
    param_name = "CE"
    x_title = "Number of Iterations"
    legend_title = param_name
        
    plt_title = "Simulated Annealing "+problem_name+" Problem Fitness by "+param_name
    save_plot(df,param_name,'Best Fitness','Iterations',
                    path2,x_title,"Fitness Value", plt_title,legend_title,"sa_params.png")

    plt_title = "Simulated Annealing "+problem_name+" Problem Time by "+param_name
    save_plot(df,param_name,'Time','Iterations',
                    path2,x_title,"Time", plt_title,legend_title,"sa_time.png")  

    # rh
    df = pd.read_csv(os.path.join(data_dir,'rhc.csv'))
    param_name = "Restarts"
    x_title = "Number of Iterations"
    legend_title = param_name
        
    plt_title = "Random Hill Climbing "+problem_name+" Problem Fitness by "+param_name
    save_plot(df,param_name,'Best Fitness','Iterations',
                    path2,x_title,"Fitness Value", plt_title,legend_title,"rhc_params.png")

    plt_title = "Random Hill Climbing "+problem_name+" Problem Time by "+param_name
    save_plot(df,param_name,'Time','Iterations',
                    path2,x_title,"Time", plt_title,legend_title,"rhc_time.png")   


    # mimic
    df = pd.read_csv(os.path.join(data_dir,'mi.csv'))
    param_name = "Keep percentage"
    x_title = "Number of Iterations"
    legend_title = param_name
        
    plt_title = "MIMIC "+problem_name+" Problem Fitness by "+param_name
    save_plot(df,param_name,'Best Fitness','Iterations',
                    path2,x_title,"Fitness Value", plt_title,legend_title,"mim_params.png")

    plt_title = "MIMIC "+problem_name+" Problem Time by "+param_name
    save_plot(df,param_name,'Time','Iterations',
                    path2,x_title,"Time", plt_title,legend_title,"mim_time.png")

    plot_summary_fitness_time(data_dir, problem_name, True) 
    plot_summary_fitness_time(data_dir, problem_name, False)     


def genetic_algorithm(problem_fit, vectorLength, data_dir, iterlist):
    directory="./"+data_dir+"/curves/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    path1='./'+data_dir
    path2= "./"+data_dir+"/curves/"

    prl=[]

    beststate=[]
    bestfit=[]
    curve=[]
    time=[]
    iterlistn=[]


    for pr in np.linspace(0.1,1,5):
        for iters in iterlist:
            start=clock()
            best_state, best_fitness, train_curve = mlrose.genetic_alg(problem_fit, \
                                                                    mutation_prob = pr,\
                                                                    max_iters=int(iters), 
                                                                    curve=True, random_state=randomSeed)
            end=clock()
            iterlistn.append(int(iters))
            time.append(end-start)
            beststate.append(best_state)
            bestfit.append(best_fitness)
            prl.append(pr)
            curve.append(train_curve)
            if (verbose == True):
                print(pr)
                print(int(iters))
                print(best_state)
                print(best_fitness)

    ffga=pd.DataFrame({'Mutation Probability':prl, 'Best Fitness':bestfit, 'Iterations':iterlistn, 'Time':time})
    beststatedf=pd.DataFrame(0.0, index=range(1,vectorLength + 1), columns=range(len(beststate)))
    for i in range(len(curve)):
        curvedf=pd.DataFrame(curve[i])
        curvedf.to_csv(os.path.join(path2,'gacurve{}_{}.csv'.format(prl[i], iterlistn[i])))
        
    for i in range(1,len(beststate)+1):
        beststatedf.loc[:,i]=beststate[i-1]
        
    ffga.to_csv(os.path.join(path1,'ga.csv'))
    beststatedf.to_csv(os.path.join(path1,'gastates.csv'))


def simulated_annealing(problem_fit, vectorLength, data_dir, iterlist):
    directory="./"+data_dir+"/curves/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    path1='./'+data_dir
    path2= "./"+data_dir+"/curves/"

    beststate=[]
    bestfit=[]
    curve=[]
    time=[]
    iterlistn=[]
    CEl=[]

    for iters in iterlist:
        for CE in [0.20, 0.40, 0.60, 0.80,  1.0]:
            CEl.append(CE)
            start=clock()
            best_state, best_fitness, train_curve = mlrose.simulated_annealing(problem_fit,\
                                                                        max_iters=int(iters),\
                                                                        curve=True, \
                                                                        schedule=mlrose.GeomDecay(init_temp=CE),
                                                                        random_state=randomSeed)
        
            end=clock()
            time.append(end-start)
            beststate.append(best_state)
            bestfit.append(best_fitness)
            curve.append(train_curve)
            iterlistn.append(int(iters))
            if (verbose == True):
                print(CE)
                print(int(iters))
                print(best_state)
                print(best_fitness)

    ffsa=pd.DataFrame({ 'Best Fitness':bestfit, 'Iterations':iterlistn, 'Time':time, 'CE':CEl})
    beststatedf=pd.DataFrame(0.0, index=range(1,vectorLength + 1), columns=range(len(beststate)))
    for i in range(len(curve)):
        pd.DataFrame(curve[i]).to_csv(os.path.join(path2,'sacurve_{}_{}.csv'.format( iterlistn[i], CEl[i])))

    for i in range(1,len(beststate)+1):
        beststatedf.loc[:,i]=beststate[i-1]
    ffsa.to_csv(os.path.join(path1,'sa.csv'))
    beststatedf.to_csv(os.path.join(path1,'sastates.csv'))


def random_hill_climb(problem_fit, vectorLength, data_dir, iterlist):
    directory="./"+data_dir+"/curves/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    path1='./'+data_dir
    path2= "./"+data_dir+"/curves/"

    restartl=[]

    beststate=[]
    bestfit=[]
    curve=[]
    time=[]
    iterlistn=[]

    for rs in (0,5,10,15,20,25):
        for iters in iterlist:
            iterlistn.append(int(iters))
            restartl.append(rs)
            start=clock()
            best_state, best_fitness, train_curve = mlrose.random_hill_climb(problem_fit, \
                                                                restarts=rs,\
                                                                max_iters=int(iters), curve=True, random_state=randomSeed)
            end=clock()
            time.append(end-start)
            beststate.append(best_state)
            bestfit.append(best_fitness)
            curve.append(train_curve)
            if (verbose == True):
                print(rs)
                print(int(iters))
                print(best_state)
                print(best_fitness)

    ffrhc=pd.DataFrame({'Restarts':restartl, 'Best Fitness':bestfit, 'Iterations':iterlistn, 'Time':time})

    rhcbeststatedf=pd.DataFrame(0.0, index=range(1,vectorLength + 1), columns=range(1,len(beststate)+1))

    for i in range(1,len(beststate)+1):
        rhcbeststatedf.loc[:,i]=beststate[i-1]
        
    ffrhc.to_csv(os.path.join(path1,'rhc.csv'))
    rhcbeststatedf.to_csv(os.path.join(path1,'rhcstates.csv'))

    for i in range(len(curve)):
        rhccurvedf=pd.DataFrame(curve[i])
        rhccurvedf.to_csv(os.path.join(path2,'rhccurve{}_{}.csv'.format(restartl[i], iterlistn[i])))


def mimic(problem_fit, vectorLength, data_dir, iterlist):
    directory="./"+data_dir+"/curves/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    path1='./'+data_dir
    path2= "./"+data_dir+"/curves/"
    kpl=[]
    beststate=[]
    bestfit=[]
    curve=[]
    time=[]
    iterlistn=[]

    for kp in np.linspace(0.1,.5,3):
        for iters in iterlist:
            
            iterlistn.append(int(iters))
            start=clock()
            best_state, best_fitness, train_curve = mlrose.mimic(problem_fit, \
                                                                keep_pct=kp,\
                                                                max_attempts=5,\
                                                                max_iters=int(iters), curve=True, random_state=randomSeed)
            end=clock()

            time.append(end-start)
            beststate.append(best_state)
            bestfit.append(best_fitness)
            kpl.append(kp)
            curve.append(train_curve)    

            if (verbose == True):
                print(int(iters))
                print(kp)
                print(best_state)
                print(best_fitness)
                

    ffmi=pd.DataFrame({'Keep percentage':kpl, 'Best Fitness':bestfit, 'Iterations':iterlistn, 'Time':time})
    mibeststatedf=pd.DataFrame(0.0, index=range(1,vectorLength + 1), columns=range(1,len(beststate)+1))

    for i in range(1,len(beststate)+1):
        mibeststatedf.loc[:,i]=beststate[i-1]
        
    ffmi.to_csv(os.path.join(path1,'mi.csv'))
    mibeststatedf.to_csv(os.path.join(path1,'mistates.csv'))

    for i in range(len(curve)):
        micurvedf=pd.DataFrame(curve[i])
        micurvedf.to_csv(os.path.join(path2,'micurve{}_{}.csv'.format(kpl[i], iterlistn[i])))    


def run_toy_data(fitness, vectorLength, data_dir, run_ga, run_sa, run_rh, run_mimic, iterlist, galist, mimiciterlist):

    directory="./"+data_dir+"/curves/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    path1='./'+data_dir
    path2= "./"+data_dir+"/curves/"
    np.random.seed(randomSeed)
    state=np.random.randint(0,2,size=vectorLength)
    problem_fit=mlrose.DiscreteOpt(state.size, fitness)

    if (run_ga == True):
        print("running ga")
        genetic_algorithm(problem_fit, vectorLength, data_dir, galist)
    
    if (run_sa == True):
        print("running sa")
        simulated_annealing(problem_fit, vectorLength, data_dir, iterlist)

    if (run_rh == True):
        print("running rh")
        random_hill_climb(problem_fit, vectorLength, data_dir, iterlist)

    if (run_mimic == True):
        print("running mi")
        mimic(problem_fit, vectorLength, data_dir, mimiciterlist)


 

if __name__ == "__main__":
    import random
    random.seed(randomSeed)
    parser = argparse.ArgumentParser() 
    
    # Adding optional argument 
    parser.add_argument("-g", "--ga", help = "Run GA", default='y') 
    parser.add_argument("-s", "--sa", help = "Run SA", default='y') 
    parser.add_argument("-r", "--rh", help = "Run RH", default='y') 
    parser.add_argument("-m", "--mi", help = "Run MIMIC", default='y') 
    parser.add_argument("-p", "--plot", help="Plot Charts", default='y')

    
    # Read arguments from command line 
    args = parser.parse_args() 
    print(args)   

    print("starting up")
    run_ga = (args.ga == 'y')
    run_sa = (args.sa == 'y')
    run_rh = (args.rh == 'y')
    run_mi = (args.mi == 'y')
    run_plots = (args.plot == 'y')

    vLength = 50

    iterlist = [i for i in range(vLength*2)]
    galist = [i for i in range(vLength*2)]

    mimiciterlist = [i for i in range(int(vLength))]

    max_one= mlrose.OneMax()
    run_toy_data(max_one, vLength,"max_one",run_ga, run_sa,run_rh, run_mi,iterlist, galist, mimiciterlist)

    four_peaks = mlrose.FourPeaks(t_pct=.2)   
    run_toy_data(four_peaks, vLength,"four_peaks",run_ga, run_sa,run_rh, run_mi,iterlist, galist, mimiciterlist)

    weights = [random.randint(5,30) for i in range(vLength)]
    values = [random.randint(1,5) for i in range(vLength)]
    max_weight_pct = 0.6
    knapsack = mlrose.Knapsack(weights, values, max_weight_pct)
    run_toy_data(knapsack, vLength,"knapsack",run_ga, run_sa,run_rh, run_mi,iterlist, galist, mimiciterlist)    

    if run_plots == True:
        plot_fitness_time("max_one", "Max Ones")
        plot_fitness_time("four_peaks", "Four Peaks")
        plot_fitness_time("knapsack", "Knapsack")




    