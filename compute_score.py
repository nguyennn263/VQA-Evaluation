import re
import traceback
import unicodedata
import string
import numpy as np
from typing import List
from accuracy import Accuracy
from bleu import Bleu
from cider import Cider
from f1 import F1
from meteor import Meteor
from precision import Precision
from recall import Recall
from rouge import Rouge 
from wup import Wup
from bertscore import BERTScore
from tqdm import tqdm 

def normalize_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower().strip()
    return text

def preprocess_sentence(sentence: str, tokenizer=None):
    sentence = sentence.lower()
    sentence = unicodedata.normalize('NFC', sentence)
    sentence = re.sub(r"[“”]", "\"", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"\"", " \" ", sentence)
    sentence = re.sub(r"'", " ' ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\[", " [ ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\]", " ] ", sentence)
    sentence = re.sub(r"/", " / ", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r"-", " - ", sentence)
    sentence = re.sub(r"\$", " $ ", sentence)
    sentence = re.sub(r"\&", " & ", sentence)
    sentence = re.sub(r"\*", " * ", sentence)
    # tokenize the sentence
    if tokenizer is None:
        tokenizer = lambda s: s
    sentence = tokenizer(sentence)
    sentence = " ".join(sentence.strip().split()) # remove duplicated spaces
    tokens = sentence.strip().split()

    return tokens

class ScoreCalculator:
    def __init__(self):
        self.acc_calculate=Accuracy()
        self.bleu_calculate=Bleu()
        self.cider_calculate=Cider()
        self.f1_calculate=F1()
        self.meteor_calculate=Meteor()
        self.precision_calculate=Precision()
        self.recall_calculate=Recall()
        self.rouge_calculate=Rouge()
        self.wup_calculate=Wup()
        self.bertscore_calculate=BERTScore()
     

    #F1 score token level - lấy max score với nhiều ground truths  
    def f1_token(self, labels: List[str], pred: str) -> float:
        """
        Tính F1 score token level, lấy max score giữa pred và tất cả labels
        :param labels: List các ground truth answers
        :param pred: Generated answer
        :return: Max F1 score
        """
        scores = []
        pred_processed = str(preprocess_sentence(normalize_text(pred)))
        joined_pred_processed = " ".join(pred_processed)

        for i, label in enumerate(labels):
            label_processed = str(preprocess_sentence(normalize_text(label)))
            joined_label_processed = " ".join(label_processed)
            gts = {str(i): [joined_label_processed]}
            res = {str(i): [joined_pred_processed]}
            score, _ = self.f1_calculate.compute_score(gts, res)
            # Đảm bảo score là scalar
            if isinstance(score, (list, tuple, np.ndarray)):
                score = float(score[0]) if len(score) > 0 else 0.0
            scores.append(float(score))
        
        return max(scores) if scores else 0.0


    #Wup score - lấy max score với nhiều ground truths
    def wup(self, labels: List[str], pred: str) -> float:
        """
        Tính WUP score, lấy max score giữa pred và tất cả labels
        :param labels: List các ground truth answers
        :param pred: Generated answer
        :return: Max WUP score
        """
        scores = []
        pred_processed = str(preprocess_sentence(normalize_text(pred))).split()
        
        for label in labels:
            label_processed = str(preprocess_sentence(normalize_text(label))).split()
            score = self.wup_calculate.compute_score(label_processed, pred_processed)
            # Đảm bảo score là scalar

            scores.append(float(score))
        
        return max(scores) if scores else 0.0
    #Cider score - lấy max score với nhiều ground truths
    def cider_score(self, labels: List[str], pred: str) -> float:
        """
        Tính CIDEr score, lấy max score giữa pred và tất cả labels
        :param labels: List các ground truth answers
        :param pred: Generated answer
        :return: Max CIDEr score
        """
        scores = []
        
        # Preprocessing trước khi tính CIDEr
        pred_processed = preprocess_sentence(normalize_text(pred))
        joined_pred_processed = " ".join(pred_processed)
        
        for i, label in enumerate(labels):
            label_processed = preprocess_sentence(normalize_text(label))
            joined_label_processed = " ".join(label_processed)
            # Chuẩn bị dữ liệu theo format mà cider module yêu cầu
            gts = {str(i): [joined_label_processed]}
            res = {str(i): [joined_pred_processed]}
            score, _ = self.cider_calculate.compute_score(gts, res)
            scores.append(score)
        
        return max(scores) if scores else 0.0
    
    #Accuracy score - lấy max score với nhiều ground truths
    def accuracy_score(self, labels: List[str], pred: str) -> float:
        """
        Tính Accuracy score, lấy max score giữa pred và tất cả labels
        :param labels: List các ground truth answers
        :param pred: Generated answer
        :return: Max Accuracy score
        """
        scores = []
        
        # Preprocessing trước khi tính Accuracy
        pred_processed = preprocess_sentence(normalize_text(pred))
        joined_pred_processed = " ".join(pred_processed)
        
        for i, label in enumerate(labels):
            label_processed = preprocess_sentence(normalize_text(label))
            joined_label_processed = " ".join(label_processed)
            # Chuẩn bị dữ liệu theo format mà accuracy module yêu cầu
            gts = {str(i): [joined_label_processed]}
            res = {str(i): [joined_pred_processed]}
            score, _ = self.acc_calculate.compute_score(gts, res)
            scores.append(score)
        
        return max(scores) if scores else 0.0
    
    #BLEU score - lấy max score với nhiều ground truths
    def bleu_score(self, labels: List[str], pred: str) -> float:
        """
        Tính BLEU score, lấy max score giữa pred và tất cả labels
        :param labels: List các ground truth answers
        :param pred: Generated answer
        :return: Max BLEU score
        """
        scores = []
        
        # Preprocessing trước khi tính BLEU
        pred_processed = preprocess_sentence(normalize_text(pred))
        joined_pred_processed = " ".join(pred_processed)
        
        for i, label in enumerate(labels):
            label_processed = preprocess_sentence(normalize_text(label))
            joined_label_processed = " ".join(label_processed)
            # Chuẩn bị dữ liệu theo format mà bleu module yêu cầu
            gts = {str(i): [joined_label_processed]}
            res = {str(i): [joined_pred_processed]}
            score, _ = self.bleu_calculate.compute_score(gts, res)
            # BLEU có thể trả về tuple, lấy phần tử cuối
            if isinstance(score, (list, tuple)):
                score = score[-1] if len(score) > 0 else 0.0
            scores.append(score)
        
        return max(scores) if scores else 0.0
    
    #METEOR score - lấy max score với nhiều ground truths
    def meteor_score(self, labels: List[str], pred: str) -> float:
        """
        Tính METEOR score, lấy max score giữa pred và tất cả labels
        :param labels: List các ground truth answers
        :param pred: Generated answer
        :return: Max METEOR score
        """
        scores = []
        
        # Preprocessing trước khi tính METEOR
        pred_processed = preprocess_sentence(normalize_text(pred))
        joined_pred_processed = " ".join(pred_processed)
        
        for i, label in enumerate(labels):
            label_processed = preprocess_sentence(normalize_text(label))
            joined_label_processed = " ".join(label_processed)
            # Chuẩn bị dữ liệu theo format mà meteor module yêu cầu
            gts = {str(i): [joined_label_processed]}
            res = {str(i): [joined_pred_processed]}
            try:
                score, _ = self.meteor_calculate.compute_score(gts, res)
                scores.append(score)
            except Exception as e:
                # METEOR có thể có vấn đề với Java subprocess
                print(f"METEOR error for label {i}: {e}")
                scores.append(0.0)
        
        return max(scores) if scores else 0.0
    
    #Precision score - lấy max score với nhiều ground truths
    def precision_score(self, labels: List[str], pred: str) -> float:
        """
        Tính Precision score, lấy max score giữa pred và tất cả labels
        :param labels: List các ground truth answers
        :param pred: Generated answer
        :return: Max Precision score
        """
        scores = []
        
        # Preprocessing trước khi tính Precision
        pred_processed = preprocess_sentence(normalize_text(pred))
        joined_pred_processed = " ".join(pred_processed)
        
        for i, label in enumerate(labels):
            label_processed = preprocess_sentence(normalize_text(label))
            joined_label_processed = " ".join(label_processed)
            # Chuẩn bị dữ liệu theo format mà precision module yêu cầu
            gts = {str(i): [joined_label_processed]}
            res = {str(i): [joined_pred_processed]}
            score, _ = self.precision_calculate.compute_score(gts, res)
            scores.append(score)
        
        return max(scores) if scores else 0.0
    
    #Recall score - lấy max score với nhiều ground truths
    def recall_score(self, labels: List[str], pred: str) -> float:
        """
        Tính Recall score, lấy max score giữa pred và tất cả labels
        :param labels: List các ground truth answers
        :param pred: Generated answer
        :return: Max Recall score
        """
        scores = []
        
        # Preprocessing trước khi tính Recall
        pred_processed = preprocess_sentence(normalize_text(pred))
        joined_pred_processed = " ".join(pred_processed)
        
        for i, label in enumerate(labels):
            label_processed = preprocess_sentence(normalize_text(label))
            joined_label_processed = " ".join(label_processed)
            # Chuẩn bị dữ liệu theo format mà recall module yêu cầu
            gts = {str(i): [joined_label_processed]}
            res = {str(i): [joined_pred_processed]}
            score, _ = self.recall_calculate.compute_score(gts, res)
            scores.append(score)
        
        return max(scores) if scores else 0.0
    
    #ROUGE score - lấy max score với nhiều ground truths
    def rouge_score(self, labels: List[str], pred: str) -> float:
        """
        Tính ROUGE score, lấy max score giữa pred và tất cả labels
        :param labels: List các ground truth answers
        :param pred: Generated answer
        :return: Max ROUGE score
        """
        scores = []
        
        # Preprocessing trước khi tính ROUGE
        pred_processed = preprocess_sentence(normalize_text(pred))
        joined_pred_processed = " ".join(pred_processed)
        
        for i, label in enumerate(labels):
            label_processed = preprocess_sentence(normalize_text(label))
            joined_label_processed = " ".join(label_processed)
            # Chuẩn bị dữ liệu theo format mà rouge module yêu cầu
            gts = {str(i): [joined_label_processed]}
            res = {str(i): [joined_pred_processed]}
            score, _ = self.rouge_calculate.compute_score(gts, res)
            scores.append(score)
        
        return max(scores) if scores else 0.0
    
    # BERTScore - lấy max P/R/F với nhiều ground truths
    def bert_score(self, labels: List[str], pred: str) -> list[float]:
        """
        Tính BERTScore (Precision, Recall, F1), lấy max score giữa pred và tất cả labels
        :param labels: List các ground truth answers
        :param pred: Generated answer
        :return: Tuple (P, R, F) - max values across labels
        """
        p_scores, r_scores, f_scores = [], [], []

        for label in labels:
            try:
                # compute_score expects lists of strings
                P, R, F = self.bertscore_calculate.compute_score(
                    [pred],
                    [label],
                    batch_size=64,
                    idf=False
                )
                p_scores.append(float(P[0].item()))
                r_scores.append(float(R[0].item()))
                f_scores.append(float(F[0].item()))
            except Exception as e:
                print(f"BERTScore error with label '{label}': {e}")
                p_scores.append(0.0)
                r_scores.append(0.0)
                f_scores.append(0.0)

        return [
            max(p_scores) if p_scores else 0.0,
            max(r_scores) if r_scores else 0.0,
            max(f_scores) if f_scores else 0.0,
        ]



def compute_score(ground_truths: List[str], generation: str):
    """
    Tính toán các metric đánh giá cho VQA với max score
    :param ground_truths: list các câu trả lời đúng (list of strings)
    :param generation: câu trả lời được sinh ra (string)
    :return: dictionary chứa các điểm số
    """
    calculator = ScoreCalculator()
    
    scores = {
        "accuracy": calculator.accuracy_score(ground_truths, generation),
        "bleu": calculator.bleu_score(ground_truths, generation),
        "cider": calculator.cider_score(ground_truths, generation),
        "f1_token": calculator.f1_token(ground_truths, generation),
        "meteor": calculator.meteor_score(ground_truths, generation),
        "precision": calculator.precision_score(ground_truths, generation),
        "recall": calculator.recall_score(ground_truths, generation),
        "rouge": calculator.rouge_score(ground_truths, generation),
        "wup": calculator.wup(ground_truths, generation),
        "bertscore": calculator.bert_score(ground_truths, generation)
    }
    
    return scores


def compute_all_data(all_ground_truths: List[List[str]], all_generations: List[str]):
    """
    Tính toán các metric đánh giá cho toàn bộ dataset với max score
    :param all_ground_truths: list of lists - mỗi phần tử là list các câu trả lời đúng cho 1 sample
    :param all_generations: list of strings - mỗi phần tử là câu trả lời được sinh ra cho 1 sample
    :return: dictionary chứa các điểm số trung bình
    """
    
    if len(all_ground_truths) != len(all_generations):
        raise ValueError("Số lượng ground_truths và generations phải bằng nhau")
    
    print("Computing evaluation metrics with max scoring...")
    
    all_scores = {
        "accuracy": [],
        "bleu": [],
        "cider": [],
        "f1_token": [],
        "meteor": [],
        "precision": [],
        "recall": [],
        "rouge": [],
        "wup": [],
        "bertscore": []
    }
    
    calculator = ScoreCalculator()
    
    # Tính toán score cho từng sample
    for i, (ground_truths, generation) in tqdm(enumerate(zip(all_ground_truths, all_generations)), total=len(all_ground_truths)):
        # Đảm bảo ground_truths là list of strings
        clean_gts = ground_truths
        clean_gen = generation
        
        try:
            all_scores["accuracy"].append(calculator.accuracy_score(clean_gts, clean_gen))
        except Exception as e:
            print(f"Accuracy error at sample {i}: {traceback.format_exc()}")
            all_scores["accuracy"].append(0.0)
            
        try:
            all_scores["bleu"].append(calculator.bleu_score(clean_gts, clean_gen))
        except Exception as e:
            print(f"BLEU error at sample {i}: {traceback.format_exc()}")
            all_scores["bleu"].append(0.0)
            
        try:
            all_scores["cider"].append(calculator.cider_score(clean_gts, clean_gen))
        except Exception as e:
            print(f"CIDEr error at sample {i}: {traceback.format_exc()}")
            all_scores["cider"].append(0.0)

        try:
            all_scores["f1_token"].append(calculator.f1_token(clean_gts, clean_gen))
        except Exception as e:
            print(f"F1 token error at sample {i}: {traceback.format_exc()}")
            all_scores["f1_token"].append(0.0)
            
        try:
            all_scores["meteor"].append(calculator.meteor_score(clean_gts, clean_gen))
        except Exception as e:
            print(f"METEOR error at sample {i}: {traceback.format_exc()}")
            all_scores["meteor"].append(0.0)
            
        try:
            all_scores["precision"].append(calculator.precision_score(clean_gts, clean_gen))
        except Exception as e:
            print(f"Precision error at sample {i}: {traceback.format_exc()}")
            all_scores["precision"].append(0.0)
            
        try:
            all_scores["recall"].append(calculator.recall_score(clean_gts, clean_gen))
        except Exception as e:
            print(f"Recall error at sample {i}: {traceback.format_exc()}")
            all_scores["recall"].append(0.0)
            
        try:
            all_scores["rouge"].append(calculator.rouge_score(clean_gts, clean_gen))
        except Exception as e:
            print(f"ROUGE error at sample {i}: {traceback.format_exc()}")
            all_scores["rouge"].append(0.0)
            
        try:
            all_scores["wup"].append(calculator.wup(clean_gts, clean_gen))
        except Exception as e:
            print(f"WUP error at sample {i}: {traceback.format_exc()}")
            all_scores["wup"].append(0.0)
    
        try:
            all_scores["bertscore"].append(calculator.bert_score(clean_gts, clean_gen))
        except Exception as e:
            print(f"BERTScore error at sample {i}: {traceback.format_exc()}")
            all_scores["bertscore"].append([0.0, 0.0, 0.0])
    
    # Tính điểm trung bình
    final_scores = {}
    for metric_name, scores_list in all_scores.items():
        if not scores_list:
            final_scores[metric_name] = {"average": 0.0, "individual": []}
            print(f"✗ {metric_name.upper()}: No valid scores")
            continue

        scores_array = np.array(scores_list)

        # Nếu là BERTScore thì trả về vector trung bình [P, R, F]
        if metric_name == "bertscore":
            avg_score = scores_array.mean(axis=0).tolist()
            avg_display = ", ".join(f"{v:.4f}" for v in avg_score)

            print(f"✓ {metric_name.upper()}: P={avg_score[0]:.4f}, R={avg_score[1]:.4f}, F1={avg_score[2]:.4f}")
        else:
            avg_score = float(scores_array.mean())
            avg_display = f"{avg_score:.4f}"

            print(f"✓ {metric_name.upper()}: {avg_display}")

        final_scores[metric_name] = {
            "average": avg_score,
            "individual": scores_list
        }

    return final_scores