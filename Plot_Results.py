import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import metrics
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import cv2
from Image_Results import Image_Results


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v





def plot_roc():
    lw = 2
    cls = ['CNN', 'Resnet', 'CAE', 'Densenet', 'RE-LSTM']
    colors = cycle(["m", "b", "r", "lime", "k"])
    Predicted = np.load('roc_score.npy', allow_pickle=True)
    Actual = np.load('roc_act.npy', allow_pickle=True)
    for i in range(len(Actual)):
        Dataset = ['Dataset 1', 'Dataset 2']
        for j, color in zip(range(5), colors):
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[i, 3, j], Predicted[i, 3, j])
            auc = metrics.roc_auc_score(Actual[i, 3, j], Predicted[i, 3, j])
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[j]
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/Roc.png"
        plt.savefig(path)
        plt.show()


def Plot_batch_Table():
    eval = np.load('Eval_ALL.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
             'FOR', 'pt',
             'G means', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Table_Terms = [0, 1, 2, 3, 4, 8]
    Classifier = ['TERMS', 'CNN', 'Resnet', 'CAE', 'Densenet', 'RE-LSTM']
    value = eval[3, :, 4:]
    Table = PrettyTable()
    Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Terms)])
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value[j, Table_Terms])
    print('-------------------------------------------------- ', 'Batch size ',
          'Classifier Comparison --------------------------------------------------')
    print(Table)





def plot_results_seg():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR',
             'NPV',
             'FDR', 'F1-Score', 'MCC']
    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            X = np.arange(stats.shape[2])
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            ax.bar(X + 0.00, stats[i, 0, :], color='#cb00f5', hatch='*', edgecolor='k', width=0.10, label="AOA-ARNet")
            ax.bar(X + 0.10, stats[i, 1, :], color='#0cff0c', hatch='*', edgecolor='k', width=0.10, label="GEO-ARNet")
            ax.bar(X + 0.20, stats[i, 2, :], color='r', hatch='*', edgecolor='k', width=0.10, label="MRA-ARNet")
            ax.bar(X + 0.30, stats[i, 3, :], color='c', hatch='*', edgecolor='k', width=0.10, label="LEO-ARNet")
            ax.bar(X + 0.40, stats[i, 4, :], color='k', hatch='..', edgecolor='w', width=0.10, label="ALEO-ARNet")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/seg_%s_alg.png" % (Terms[i - 4])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 5, :], color='#cb00f5', hatch='*', edgecolor='k', width=0.10,
                   label="Unet")
            ax.bar(X + 0.10, stats[i, 6, :], color='lime', hatch='*', edgecolor='k', width=0.10, label="Res-Unet")
            ax.bar(X + 0.20, stats[i, 7, :], color='r', hatch='*', edgecolor='k', width=0.10, label="Trans-Unet")
            ax.bar(X + 0.30, stats[i, 8, :], color='c', hatch='*', edgecolor='k', width=0.10, label="ARNet")
            ax.bar(X + 0.40, stats[i, 4, :], color='k', hatch='\\', edgecolor='w', width=0.10, label="ALEO-ARNet")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/seg_%s_met.png" % (Terms[i - 4])
            plt.savefig(path1)
            plt.show()


def Plot_Kfold():
    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
             'FOR', 'pt',
             'G means', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Graph_Term = [0, 5, 6, 9, 10]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Term[j] + 4]
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Graph[:, 0], edgecolor='k', hatch='//', color='r', width=0.10, label="CNN")
        ax.bar(X + 0.10, Graph[:, 1], edgecolor='k', hatch='-', color='#6dedfd', width=0.10, label="Resnet")
        ax.bar(X + 0.20, Graph[:, 2], edgecolor='k', hatch='//', color='lime', width=0.10,
               label="CAE")
        ax.bar(X + 0.30, Graph[:, 3], edgecolor='k', hatch='-', color='#ed0dd9', width=0.10, label="DenseNet")
        ax.bar(X + 0.40, Graph[:, 4], edgecolor='w', hatch='..', color='k', width=0.10, label="RE-LSTM")
        plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
        plt.xlabel('K - Fold')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path = "./Results/kfold_%s_Med.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()


def Plot_Batchsize():
    eval = np.load('Eval_ALL.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
             'FOR', 'pt',
             'G means', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Graph_Term = [2, 3, 4, 9, 12, 16]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Term[j] + 4]
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(6)
        ax.bar(X + 0.00, Graph[:, 0], edgecolor='k', hatch='//', color='b', width=0.10, label="CNN")
        ax.bar(X + 0.10, Graph[:, 1], edgecolor='k', hatch='-', color='#ed0dd9', width=0.10, label="Resnet")
        ax.bar(X + 0.20, Graph[:, 2], edgecolor='k', hatch='//', color='lime', width=0.10,
               label="CAE")
        ax.bar(X + 0.30, Graph[:, 3], edgecolor='k', hatch='-', color='#6dedfd', width=0.10, label="DenseNet")
        ax.bar(X + 0.40, Graph[:, 4], edgecolor='w', hatch='..', color='k', width=0.10, label="RE-LSTM")
        plt.xticks(X + 0.25, ('4', '8', '16', '32', '48', '64'))
        plt.xlabel('Batch Size')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path = "./Results/Batch_%s_Med.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()


def plot_results_conv():
    Result = np.load('Fitness.npy', allow_pickle=True)
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'AOA-ARNet', 'GEO-ARNet', 'MRA-ARNet', 'LEO-ARNet', 'ALEO-ARNet']
    for i in range(Result.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- ', 'Statistical Report ',
              '--------------------------------------------------')
        print(Table)
        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='.', markerfacecolor='red', markersize=12,
                 label='AOA-ARNet')
        plt.plot(length, Conv_Graph[1, :], color='c', linewidth=3, marker='.', markerfacecolor='green',
                 markersize=12,
                 label='GEO-ARNet')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='.', markerfacecolor='cyan',
                 markersize=12,
                 label='MRA-ARNet')
        plt.plot(length, Conv_Graph[3, :], color='y', linewidth=3, marker='.', markerfacecolor='magenta',
                 markersize=12,
                 label='LEO-ARNet')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='.', markerfacecolor='black',
                 markersize=12,
                 label='ALEO-ARNet')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Convergence.png")
        plt.show()


if __name__ == '__main__':
    Plot_batch_Table()
    Plot_Kfold()
    plot_roc()
    Plot_Batchsize()
    plot_results_conv()
    plot_results_seg()
    Image_Results()
