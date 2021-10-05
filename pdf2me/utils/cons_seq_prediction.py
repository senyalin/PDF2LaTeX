"""
consecutive sequential prediction
"""

import numpy as np
# from cylp.cy import CyClpSimplex
# from cylp.py.modeling import CyLPModel, CyLPArray


def pairwise_model(pred_info_dict):
    """
    sequential ME prediction
    the parameter should have:
    * me_log_prob_list
    * nme_log_prob_list
    * bin_weight, optional

    :return:
    """
    if len(pred_info_dict['me_log_prob_list']) == 0:  # nothing to predict
        return []
    if len(pred_info_dict['me_log_prob_list']) == 1:
        if pred_info_dict['me_log_prob_list'][0] > pred_info_dict['nme_log_prob_list'][0]:
            return [1]
        else:
            return [0]
    me_log_prob_list = pred_info_dict['me_log_prob_list']
    nme_log_prob_list = pred_info_dict['nme_log_prob_list']
    bin_weight = None
    if 'bin_weight' in pred_info_dict:
        bin_weight = float(pred_info_dict['bin_weight'])
    return cons_seq_pred(me_log_prob_list, nme_log_prob_list, bin_weight)


def cons_seq_pred(me_log_prob_list, nme_log_prob_list, bin_weight=1.0):
    """

    :param me_log_prob_list:
    :param nme_log_prob_list:
    :param bin_weight:
    :return:
    """
    assert len(me_log_prob_list) == len(nme_log_prob_list)
    n = len(me_log_prob_list)

    model = CyLPModel()
    x = model.addVariable('x', n, isInt=True)
    d = model.addVariable('d', 2*(n-1), isInt=True)

    model.addConstraint(d>=0)
    model.addConstraint(1>=x>=0)
    for i in range(n-1):
        l_coef = [0]*n
        l_coef[i] = 1
        l_coef[i+1] = -1
        l_coef = CyLPArray(l_coef)

        d_coef = [0]*(2*(n-1))
        d_coef[i * 2] = 1
        d_coef[i * 2+1] = -1
        d_coef = CyLPArray(d_coef)

        model.addConstraint(l_coef * x - d_coef*d == 0)

    #delta = np.matrix([[-10.0, 0.0, -7]])
    #delta = CyLPArray([-10.0, 0.0, -7])
    delta = [nme-me for me, nme in zip(me_log_prob_list, nme_log_prob_list)]
    #delta = CyLPArray(delta)
    delta = np.matrix([delta])
    print delta
    #bin_weight = 1.0
    model.objective = delta*x+bin_weight*d.sum()

    s = CyClpSimplex(model)
    cbcModel = s.getCbcModel()
    cbcModel.branchAndBound()
    sol_x = cbcModel.primalVariableSolution['x']
    print sol_x

    #sol_d = cbcModel.primalVariableSolution['d']
    #print sol_d
    return [int(x) for x in sol_x]
