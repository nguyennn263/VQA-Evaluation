from accuracy import Accuracy
from bleu import Bleu
from cider import Cider
from f1 import F1
from meteor import Meteor
from precision import Precision
from recall import Recall 
from rouge import Rouge


def compute_score(ground_truth, generation):
    accuracy = Accuracy()
    bleu = Bleu()
    cider = Cider()
    f1 = F1()
    meteor = Meteor()
    precision = Precision()
    recall = Recall()
    rouge = Rouge()

    scores = {
        "accuracy": accuracy.compute_score(ground_truth, generation),
        "bleu": bleu.compute_score(ground_truth, generation),
        "cider": cider.compute_score(ground_truth, generation),
        "f1": f1.compute_score(ground_truth, generation),
        "meteor": meteor.compute_score(ground_truth, generation),
        "precision": precision.compute_score(ground_truth, generation),
        "recall": recall.compute_score(ground_truth, generation),
        "rouge": rouge.compute_score(ground_truth, generation),
    }

    return scores