#!/usr/bin/env python
"""
Hung-Chen Yu
Hidden Markov Model training
CAIR Lab at BU
2018-12-17 13:10
-----------------------------------
Update Log
2018-12-17 13:10    print out number of each dataset

2018-12-12 16:51    Bug fixed

2018-12-11 14:34    Select the optimal number of hidden states based on the satruation point and retrain the model with all trainning dataset

2018-12-10 17:47    robust the training:testing ratio

2018-12-08 11:09    saving the score of both model for each data,
                    the mean of each hidden state,
                    best model idx
                    Print data folder when start

2018-12-06 14:41    saving the number of each training & testing dataset

"""

from __future__ import print_function
from time import gmtime, strftime, localtime

import io
import datetime
import csv
import numpy as np
import os
import math
import glob
import time
import warnings

import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import GMMHMM
from sklearn.externals import joblib



Current_time=strftime("%Y%m%d_%H%M", localtime())
print("\n\n" + Current_time)
print("Data Folder: " + os.getcwd())
#======================================================================

min_N_hiddenstates = 3
max_N_hiddenstates = 8
N_CV_fold = 5
Percentage_training_data= 0.7

data_files_name = "_walking.csv"
HMM_model_saved_name = "test_HMMmodel"



#======================================================================

def get_SoF_list():
    successful_list = np.array ([])
    failed_list = np.array([])

    with io.open('detail_information_of_trails.csv', "rt", encoding= "utf8") as csvfile:

        detail_information = csv.reader( csvfile)

        row_idx=1;
        for row in detail_information :
            if row[1] == 'y':
                successful_list = np.append(successful_list, row_idx)
            else:
                failed_list = np.append(failed_list, row_idx)
            row_idx = row_idx +1

    return successful_list, failed_list

def train_test_data_list(data_list):
    data_list_sff = data_list
    np.random.shuffle(data_list_sff)
    N_data = int(len(data_list))
    #print(data_list_sff)


    N_train = 20;#int(math.floor( ( N_data * Percentage_training_data) / N_CV_fold ) * N_CV_fold)
    N_cv =int( N_train /N_CV_fold)
    N_test =10;#int( N_data - N_train)

    train_data_list = np.array([])
    train_data_list = train_data_list.reshape([0,N_cv])
    for iFold in range(1,N_CV_fold+1):
        train_data_list = np.vstack(   (train_data_list  , data_list_sff[ N_cv*(iFold-1) : N_cv*iFold ]))
        """
        np.array( [ data_list_sff[ 0: N_cv ]   , \
                            data_list_sff[ N_cv: (N_cv*2) ]  ,  \
                            data_list_sff[ (N_cv*2): (N_cv*3) ]  , \
                            data_list_sff[ (N_cv*3 ): (N_cv * 4 ) ], \
                            data_list_sff[ ( N_cv*4 ): (N_cv * 5 ) ]    ]  )
        """
    test_data_list = np.array( data_list_sff [ N_train: N_data+1 ]  )
    """
    print(data_list_sff)
    print(train_data_list)
    print(test_data_list)
    print (N_train,  N_test)
    """

    return train_data_list, test_data_list, N_train,  N_test

def data_stacking(data_list, SoF):
    data_stack = np.array([])
    data_len = np.array([],dtype=np.int16)
    if SoF == 1:
        data_dir="./successful_task/"
    else:
        data_dir="./failed_task/"

    start_mark=1
    for x in np.nditer(data_list):
        #print(x)
        temp_file_name = str(int(x)) + data_files_name
        temp_data = np.genfromtxt(data_dir+temp_file_name, delimiter=',' ,skip_header=False)
        data_len = np.append( data_len, temp_data.shape[0])

        if start_mark==1:
            data_stack=temp_data
            start_mark=0
        else:
            data_stack = np.vstack((data_stack , temp_data))

    #print(data_len)
    return data_stack, data_len

def header_generator(max_N_hiddenstates):
    header_text=""
    for x in range(min_N_hiddenstates, max_N_hiddenstates+1):
        header_text = header_text + str(int(x))+" hidden states, "

    return header_text

def HMM_SF_CCR(S_HMMmodel, F_HMMmodel, S_testing_list, F_testing_list):

    classification_result = np.array([])
    classification_result = classification_result.reshape([0,6])
    print("      testing S data..." + "    N_data: "+str(int( S_testing_list.size )) )
    for x in  np.nditer(S_testing_list):
        classification_result = np.vstack ( (classification_result ,  HMM_testing(x, 1, S_HMMmodel, F_HMMmodel) ) )
    print("      testing F data..." + "    N_data: "+str(int( F_testing_list.size  )))
    for x in  np.nditer(F_testing_list):
        classification_result = np.vstack ( (classification_result ,  HMM_testing(x, 0, S_HMMmodel, F_HMMmodel) ) )

    print("      total data:" + str(classification_result.shape[0]))

    #print(classification_result)

    #print(classification_result.sum(axis=0))

    overall_CCR = classification_result.sum(axis=0)[3] / classification_result.shape[0]
    Precision = classification_result[  classification_result[:,2]==1, 3] .sum()   /   classification_result[:,2].sum()
    Recall = classification_result[  classification_result[:,1]==1, 3] .sum()   /   classification_result[:,1].sum()
    F1 = 2*Recall*Precision/ (Precision + Recall)
    print("\n      Overall CCR: "+str(overall_CCR* 100) + " %" )
    print("      Precision  : "+str(Precision* 100)+ " %" )
    print("      Recall     : "+str(Recall* 100)+ " %" )
    print("      F1         : "+str(F1* 100)+ " %" )

    performance_matrics = np.array([overall_CCR, Precision, Recall, F1])

    return classification_result, performance_matrics

def HMM_train(data_list, SoF):
    global N_CV_fold, max_N_hiddenstates, model_saved_folder, HMM_model_saved_name

    if SoF==1:
        temp_name="S_"
    else:
        temp_name="F_"

    score_sheet=np.zeros( (N_CV_fold , max_N_hiddenstates-1) )
    header_text = header_generator(max_N_hiddenstates)

    for N_fold in range (1, N_CV_fold+1):
        print("\n "+str(int(N_fold)) + " fold...")

        temp_train_data_list =  np.delete(data_list, int(N_fold)-1, 0)
        #print(temp_train_data )
        temp_train_data_stack, temp_train_data_len = data_stacking(temp_train_data_list, SoF)
        temp_validate_data_stack, temp_validate_data_len = data_stacking( data_list[int(N_fold)-1, :] , SoF)

        for N_HiddenStates in range(min_N_hiddenstates,max_N_hiddenstates+1):
            print("  "+str(int(N_HiddenStates)) + " hidden states...")
            # Training th model by executing fit
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=DeprecationWarning)
                    temp_model = GaussianHMM(n_components=int(N_HiddenStates), covariance_type="full", algorithm='viterbi', tol = 0.000001).fit(temp_train_data_stack, temp_train_data_len)
                    temp_score = temp_model.score(temp_validate_data_stack, temp_validate_data_len)
            except:
                print("    => model can't be trained!")
                temp_score = pow(-10,21)

            score_sheet [  N_fold-1 , N_HiddenStates-2 ] = temp_score

            # temp_model_name = model_saved_folder+ temp_name + HMM_model_saved_name+ "_F"+str(int(N_fold))+"_HS"+str(int(N_HiddenStates))+ ".pkl"
            # joblib.dump(temp_model,temp_model_name)

        fmt = ",".join(["%s"] + ["%10.6e"] * (score_sheet.shape[1]-1))
        np.savetxt(result_saved_folder+temp_name+"score_sheet.csv", score_sheet, delimiter=",", header=header_text )

    score_mean = score_sheet.mean(axis=0)

    # --- Hand select optimal N_HiddenStates
    # plt.plot( list(range(min_N_hiddenstates,max_N_hiddenstates+1)) ,  score_mean ,'--bo')
    # plt.ylabel('Score')
    # plt.xlabel('N_hiddenstates')
    # plt.savefig(result_saved_folder+temp_name+'mean_score.png')
    # plt.ion() # enables interactive mode
    # plt.pause(.001)
    # plt.show()
    print(score_mean[-(max_N_hiddenstates-min_N_hiddenstates+1):])
    opl_N_hiddenstates = np.argmax(score_mean[-(max_N_hiddenstates-min_N_hiddenstates+1):]) + 3
    print('Optimal number of state is')
    print(opl_N_hiddenstates)
    time.sleep(1)
    # plt.close()


    score_sheet_with_mean = np.vstack((score_sheet, score_mean))
    np.savetxt(result_saved_folder+temp_name+"score_sheet.csv", score_sheet_with_mean, delimiter=",", header=header_text )

    # --- Train the optimal model with all training dataset
    print("\n  Training the optimal HMM model with the whole training dataset using  --- "+str(int(opl_N_hiddenstates)) + " --- hidden states...")
    all_train_data_stack, all_train_data_len = data_stacking(data_list, SoF)
    best_HMM_model = GaussianHMM(n_components=opl_N_hiddenstates, covariance_type="full", algorithm='viterbi', tol = 0.000001).fit(all_train_data_stack, all_train_data_len)

    best_model_name = model_saved_folder+ temp_name + HMM_model_saved_name + "_optimal_model"+"_HS"+str(int(opl_N_hiddenstates))+ ".pkl"
    joblib.dump(best_HMM_model,best_model_name)

    return best_HMM_model

def HMM_testing(data_idx, SoF, S_HMMmodel, F_HMMmodel):
    if SoF == 1:
        data_dir="./successful_task/"
    else:
        data_dir="./failed_task/"

    temp_file_name = str(int(data_idx)) + data_files_name
    temp_data = np.genfromtxt(data_dir+temp_file_name, delimiter=',' ,skip_header=False)
    temp_data_len = temp_data.shape[0]
    #print(temp_data)
    #print(temp_data_len)

    S_score = S_HMMmodel.score(temp_data)
    F_score = F_HMMmodel.score(temp_data)

    if S_score>F_score:
        data_result = 1
    else:
        data_result = 0

    if data_result == SoF:
        CCR = 1
    else:
        CCR = 0

    temp_result = np.array([data_idx, SoF, data_result, CCR, S_score, F_score ])
    return temp_result




if __name__ == '__main__':
    for N_model_train in range(0,20): #Here I train 20 models
        result_saved_folder = "./Result_" + str(N_model_train) + '/'
        model_saved_folder = result_saved_folder+"/HMM_Models/"
        try:
            current_folder_path = os.getcwd()
            os.makedirs(result_saved_folder)
            os.makedirs(model_saved_folder)
        except:
            print("folder has already existed")
    # === Data processing ==========================================================================
        successful_list, failed_list = get_SoF_list()

        S_train_data_list, S_test_data_list, S_N_train, S_N_test  = train_test_data_list(successful_list)
        np.savetxt(result_saved_folder+"S_train_data_list.csv", S_train_data_list, delimiter="," )
        np.savetxt(result_saved_folder+"S_test_data_list.csv", S_test_data_list, delimiter="," )

        F_train_data_list, F_test_data_list, F_N_train, F_N_test = train_test_data_list(failed_list)
        np.savetxt(result_saved_folder+"F_train_data_list.csv", F_train_data_list, delimiter="," )
        np.savetxt(result_saved_folder+"F_test_data_list.csv", F_test_data_list, delimiter="," )

        temp_n_data = np.array([[S_N_train, S_N_test, F_N_train,  F_N_test, S_N_train+ F_N_train, S_N_test+F_N_test ]] )
        fmt = ",".join(["%s"] + ["%10.6e"] * (len(temp_n_data)))
        header_text_number = "N successful training, N successful testing, N failed training, N failed testing, N total training,   N total testing"
        np.savetxt(result_saved_folder+"number_each_dataset.csv", temp_n_data, delimiter=",", header=header_text_number )

        #print(S_train_data_list)

    # === HMM training ==========================================================================
     # ---- Successful tasks ------------------------------------------------------------------
        print("\n\n----------------------------")
        print("Successful tasks model")

        SoF = 1

        S_best_HMMmodel = HMM_train(S_train_data_list, SoF)


     # ---- Failed tasks ------------------------------------------------------------------
        print("\n\n----------------------------")
        print("Failed tasks model")

        SoF = 0

        F_best_HMMmodel = HMM_train(F_train_data_list, SoF)

        """
        fmt = ",".join(["%s"] + ["%10.6e"] * (len(F_best_model)))
        header_text_best_model = "Fold No., N Hidden States"
        temp_best_model = np.vstack((S_best_model,  F_best_model))
        np.savetxt(result_saved_folder+"best_model.csv", temp_best_model, delimiter=",", header=header_text_best_model )
        """

    # === HMM performace evluation==========================================================================
        print("\n\n----------------------------")
        print("Evluating the performance....")

        print("\n   Training dataset....")
        training_classification_result, training_performance_matrics = HMM_SF_CCR(S_best_HMMmodel, F_best_HMMmodel, S_train_data_list, F_train_data_list)

        print("\n   Testing dataset....")
        testing_classification_result, testing_performance_matrics = HMM_SF_CCR(S_best_HMMmodel, F_best_HMMmodel, S_test_data_list, F_test_data_list)

        # saving the results
        fmt = ",".join(["%s"] + ["%10.6e"] * (len(training_performance_matrics)))
        header_text_performance = "Overall CCR, Precision, Recall, F1"
        temp_performance = np.vstack((training_performance_matrics,  testing_performance_matrics))
        temp_performance = np.multiply(temp_performance, 100)
        np.savetxt(result_saved_folder+"performance_matrics.csv", temp_performance, delimiter=",", header=header_text_performance )


        fmt = ",".join(["%s"] + ["%10.6e"] * (training_classification_result.shape[1]-1))
        header_text_result = "Data_idx, SoF, Classification result, Classfication Correct of not, S model Score, F model Socre"
        np.savetxt(result_saved_folder+"training_classification_result.csv", training_classification_result, delimiter=",", header=header_text_result )

        fmt = ",".join(["%s"] + ["%10.6e"] * (testing_classification_result.shape[1]-1))
        header_text_result = "Data_idx, SoF, Classification result, Classification Correct of not, S model Score, F model Socre"
        np.savetxt(result_saved_folder+"testing_classification_result.csv", testing_classification_result, delimiter=",", header=header_text_result )
