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
        "accuracy": accuracy.score(ground_truth, generation),
        "bleu": bleu.score(ground_truth, generation),
        "cider": cider.score(ground_truth, generation),
        "f1": f1.score(ground_truth, generation),
        "meteor": meteor.score(ground_truth, generation),
        "precision": precision.score(ground_truth, generation),
        "recall": recall.score(ground_truth, generation),
        "rouge": rouge.score(ground_truth, generation),
    }

    return scores