import numpy as np

def Accuracy(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the accuracy for prediction.
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - hypo  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - Acc : (scalar, float), Computed accuracy score
    # ========================= EDIT HERE =========================
    Acc = None
    if len(label.shape) == 1:
        y = np.expand_dims(label, 1)
    total = pred.shape[0]
    correct = len(np.where(pred == label)[0])
    Acc = correct / total
    # =============================================================
    return Acc

def Precision(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the Precision for prediction.
    #         you should consider that label = 1 is positive. 0 is negative
    #         Notice that, if you encounter the divide zero, return 1
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - hypo  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - precision : (scalar, float), Computed precision score
    # ========================= EDIT HERE =========================
    precision = None
    prediction = len(np.where(pred == 1)[0])
    correct = len(np.where((pred == 1) & (label == 1))[0])
    if (prediction == 0):
        return 1
    precision = correct / prediction

    # =============================================================
    return precision

def Recall(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the Recall for prediction.
    #         you should consider that label = 1 is positive. 0 is negative
    #         Notice that, if you encounter the divide zero, return 1
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - hypo  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - recall : (scalar, float), Computed recall score
    # ========================= EDIT HERE =========================
    recall = None
    total_correct = len(np.where(label == 1)[0])
    correct = len(np.where((pred == 1) & (label == 1))[0])
    if (total_correct == 0):
        return 0
    recall = correct / total_correct
    # =============================================================
    return recall

def F_measure(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the F-measure score for prediction.
    #         you can erase the code. (F_score = 0.)
    #         Notice that, if you encounter the divide zero, return 1
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - hypo  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - F_score : (scalar, float), Computed F-score score
    # ========================= EDIT HERE =========================
    F_score = None
    precision = Precision(label, pred)
    recall = Recall(label, pred)
    F_score = 2*precision*recall / (precision + recall)
    # =============================================================
    return F_score

def MAP(label, hypo, at = 10):
    ########################################################################################
    # TODO : Complete the code to calculate the MAP for prediction.
    #         Notice that, hypo is the real value array in (0, 1)
    #         MAP (at = 10) means MAP @10
    #         [Input]
    #         - label : (N, K), Correct label with 0 (incorrect) or 1 (correct)
    #         - hypo  : (N, K), Predicted score between 0 and 1
    #         - at: (int), # of element to consider from the first. (TOP-@)
    #         [output]
    #         - Map : (scalar, float), Computed MAP score
    # ========================= EDIT HERE =========================
    Map = 0.
    for n in range(label.shape[0]):
        order = np.lexsort([label[n], hypo[n]])[::-1]
        tlabel = label[n][order]
        thypo = hypo[n][order]

        total_correct = len(np.where(tlabel == 1)[0])
        cum_correct = np.cumsum(tlabel)
        index = np.arange(1, len(tlabel) + 1)
        precision = cum_correct / index
        Map += np.sum(np.multiply(precision, tlabel)[:at]) / total_correct

    Map = Map / label.shape[0]

    # =============================================================
    return Map

def nDCG(label, hypo, at = 10):
    ########################################################################################
    # TODO : Complete the each code to calculate the nDCG for prediction.
    #         you can erase the code. (dcg, idcg, ndcg = 0.)
    #         Notice that, hypo is the real value array in (0, 1)
    #         nDCG (at = 10 ) means nDCG @10
    #         [Input]
    #         - label : (N, K), Correct label with 0 (incorrect) or 1 (correct)
    #         - hypo  : (N, K), Predicted score between 0 and 1
    #         - at: (int), # of element to consider from the first. (TOP-@)
    #         [output]
    #         - Map : (scalar, float), Computed nDCG score


    def DCG(label, hypo, at=10):
        # ========================= EDIT HERE =========================
        dcg = None
        order = np.lexsort([label, hypo])[::-1]
        label = label[order]
        hypo = hypo[order]

        total = len(label)
        score = 1 / np.log2(np.arange(2, total + 2))
        dcg = np.sum(np.multiply(score, label)[:at])

        # =============================================================
        return dcg

    def IDCG(label, hypo, at=10):
        # ========================= EDIT HERE =========================
        idcg = None
        total_correct = len(np.where(label == 1)[0])
        score = 1 / np.log2(np.arange(2, total_correct + 2))
        idcg = np.sum(score)
        # =============================================================
        return idcg
    # ========================= EDIT HERE =========================
    ndcg = 0.
    for n in range(label.shape[0]):
        ndcg += DCG(label[n], hypo[n], at) / IDCG(label[n], hypo[n], at)
    ndcg = ndcg / label.shape[0]
    # =============================================================
    return ndcg

# =============================================================== #
# ===================== DO NOT EDIT BELOW ======================= #
# =============================================================== #

def evaluation_test1(label, pred, at = 10):
    result = {}

    result['Accuracy '] = Accuracy(label, pred)
    result['Precision'] = Precision(label, pred)
    result['Recall   '] = Recall(label, pred)
    result['F_measure'] = F_measure(label, pred)

    return result

def evaluation_test2(label, hypo, at = 10):
    result = {}

    result['MAP  @%d'%at] = MAP(label, hypo, at)
    result['nDCG @%d'%at] = nDCG(label, hypo, at)

    return result
