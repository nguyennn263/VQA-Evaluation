from accuracy import Accuracy
from bleu import Bleu
from cider import Cider
from f1 import F1
from meteor import Meteor
from precision import Precision
from recall import Recall 
from rouge import Rouge


def compute_score(ground_truth_list, generation):
    accuracy = Accuracy()
    bleu = Bleu()
    cider = Cider()
    f1 = F1()
    meteor = Meteor()
    precision = Precision()
    recall = Recall()
    rouge = Rouge()

    # Tính điểm cho từng ground truth và lấy max
    scores = {
        "accuracy": max(accuracy.compute_score([gt], generation) for gt in ground_truth_list),
        "bleu": max(bleu.compute_score([gt], generation) for gt in ground_truth_list),
        "cider": max(cider.compute_score([gt], generation) for gt in ground_truth_list),
        "f1": max(f1.compute_score([gt], generation) for gt in ground_truth_list),
        "meteor": max(meteor.compute_score([gt], generation) for gt in ground_truth_list),
        "precision": max(precision.compute_score([gt], generation) for gt in ground_truth_list),
        "recall": max(recall.compute_score([gt], generation) for gt in ground_truth_list),
        "rouge": max(rouge.compute_score([gt], generation) for gt in ground_truth_list),
    }

    return scores


# from accuracy import Accuracy
# from bleu import Bleu
# from cider import Cider
# from f1 import F1
# from meteor import Meteor
# from precision import Precision
# from recall import Recall 
# from rouge import Rouge


# def compute_score(ground_truth, generation):
#     accuracy = Accuracy()
#     bleu = Bleu()
#     cider = Cider()
#     f1 = F1()
#     meteor = Meteor()
#     precision = Precision()
#     recall = Recall()
#     rouge = Rouge()

#     scores = {
#         "accuracy": accuracy.compute_score(ground_truth, generation),
#         "bleu": bleu.compute_score(ground_truth, generation),
#         "cider": cider.compute_score(ground_truth, generation),
#         "f1": f1.compute_score(ground_truth, generation),
#         "meteor": meteor.compute_score(ground_truth, generation),
#         "precision": precision.compute_score(ground_truth, generation),
#         "recall": recall.compute_score(ground_truth, generation),
#         "rouge": rouge.compute_score(ground_truth, generation),
#     }

#     return scores