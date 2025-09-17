import numpy as np
from Evaluation import seg_evaluation
from Global_Vars import Global_Vars
from Model_ARNeT import Model_ARNeT


def Obj_fun(Soln):
    Images = Global_Vars.Images
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, pred = Model_ARNeT(Images, sol)
            Eval = seg_evaluation(pred, Tar)
            Fitn[i] = 1 / (Eval[4] + Eval[6])
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, pred = Model_ARNeT(Images, sol)
        Eval = seg_evaluation(pred, Tar)
        Fitn = 1 / (Eval[4] + Eval[6])
        return Fitn
