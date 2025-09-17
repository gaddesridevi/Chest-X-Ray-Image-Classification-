import numpy as np
import os
import cv2 as cv
from keras.utils import to_categorical
from numpy import matlib
from AOA import AOA
from GEO import GEO
from Global_Vars import Global_Vars
from LEO import LEO
from MRA import MRA
from Model_ARNeT import Model_ARNeT
from Model_AutoEn import Model_CAutoEncoder
from Model_CNN import Model_CNN
from Model_DenseNet import Model_DenseNet
from Model_PROPOSED import Model_PROPOSED
from Model_RESNET import Model_RESNET
from Objective_Function import Obj_fun
from PROPOSED import PROPOSED
from Plot_Results import *

# Read Dataset
an = 0
if an == 1:
    Directory = './Dataset/COVID-19_Radiography_Dataset/'
    files = os.listdir(Directory)
    Images = []
    Target = []
    for n in range(len(files)):
        if n == 2 or n == 4 or n == 7:
            subfold = Directory + files[n]
            fold = os.listdir(subfold)
            for i in range(len(fold)):  # 200
                print(n, i)
                filename = subfold + '/' + fold[i]
                image = cv.imread(filename)
                Images.append(image)
                Target.append(n)
                Tar = to_categorical(Target)
    np.save('Images.npy', Images)
    np.save('Target.npy', Tar)

# Generate Ground_Truth
an = 0
if an == 1:
    Directory = './Dataset/Masks/'
    files = os.listdir(Directory)
    Images = []
    for n in range(len(files)):
        subfold = Directory + files[n]
        fold = os.listdir(subfold)
        for i in range(len(fold)):
            print(n, i)
            filename = subfold + '/' + fold[i]
            image = cv.imread(filename)
            Images.append(image)
    np.save('GT.npy', Images)

### OPTIMIZATION for Segmentation
an = 0
if an == 1:
    image = np.load('Images.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.image = image
    Global_Vars.Target = Target
    Npop = 10
    Ch_len = 3
    xmin = matlib.repmat(([5, 5, 300]), Npop, 1)
    xmax = matlib.repmat(([255, 50, 1000]), Npop, 1)
    initsol = np.zeros(xmax.shape)
    for p1 in range(Npop):
        for p2 in range(Ch_len):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    fname = Obj_fun
    max_iter = 50

    print('RSA....')
    [bestfit1, fitness1, bestsol1, Time1] = AOA(initsol, fname, xmin, xmax, max_iter)

    print('SCO....')
    [bestfit2, fitness2, bestsol2, Time2] = GEO(initsol, fname, xmin, xmax, max_iter)

    print('SGO....')
    [bestfit3, fitness3, bestsol3, Time3] = MRA(initsol, fname, xmin, xmax, max_iter)

    print('EOAA....')
    [bestfit4, fitness4, bestsol4, Time4] = LEO(initsol, fname, xmin, xmax, max_iter)

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

    Sol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]

    np.save('BestSol.npy', Sol)

# Segmentation
an = 0
if an == 1:
    Image = np.load('Images.npy', allow_pickle=True)
    Sol = np.load('BestSol.npy', allow_pickle=True)
    Segment = Model_ARNeT(Image, Sol)
    np.save('Segmentation.npy', Segment)

# KFOLD - Classification
an = 0
if an == 1:
    EVAL_ALL = []
    GT = np.load('GT.npy', allow_pickle=True)
    Target= np.load('Target.npy', allow_pickle=True)
    Image = np.load('Images.npy', allow_pickle=True)
    K = 5
    Per = 1 / 5
    Perc = round(Image.shape[0] * Per)
    eval = []
    for i in range(K):
        Eval = np.zeros((5, 25))
        Feat = Image
        Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
        Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
        test_index = np.arange(i * Perc, ((i + 1) * Perc))
        total_index = np.arange(Feat.shape[0])
        train_index = np.setdiff1d(total_index, test_index)
        Train_Data = Feat[train_index, :]
        Train_Target = Target[train_index, :]
        Eval[0, :], pred_1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[1, :], pred_2 = Model_RESNET(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[2, :], pred_3 = Model_CAutoEncoder(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[3, :], pred_4 = Model_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[4, :], pred_5 = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target)
        eval.append(Eval)
    np.save('Eval_ALL.npy', np.asarray(eval))



Plot_batch_Table()
Plot_Kfold()
plot_roc()
Plot_Batchsize()
plot_results_conv()
plot_results_seg()
Image_Results()
