from accuracy import Accuracy
from bleu import Bleu
from cider import Cider
from f1 import F1
from meteor import Meteor
from precision import Precision
from recall import Recall 
from rouge import Rouge


def compute_score(ground_truths, generation):
    """
    Tính toán các metric đánh giá cho VQA
    :param ground_truths: list các câu trả lời đúng (list of strings)
    :param generation: câu trả lời được sinh ra (string)
    :return: dictionary chứa các điểm số
    """
    
    # Chuyển đổi dữ liệu thành format mà các metric module yêu cầu
    # Sử dụng key duy nhất '0' cho single sample
    gts = {'0': ground_truths}  # ground truths là list
    res = {'0': [generation]}   # generation được wrap trong list
    
    accuracy = Accuracy()
    bleu = Bleu()
    cider = Cider()
    f1 = F1()
    meteor = Meteor()
    precision = Precision()
    recall = Recall()
    rouge = Rouge()

    scores = {}
    
    try:
        score, _ = accuracy.compute_score(gts, res)
        scores["accuracy"] = score
    except Exception as e:
        scores["accuracy"] = f"Error: {str(e)}"
    
    try:
        score, _ = bleu.compute_score(gts, res)
        scores["bleu"] = score
    except Exception as e:
        scores["bleu"] = f"Error: {str(e)}"
    
    try:
        score, _ = cider.compute_score(gts, res)
        scores["cider"] = score
    except Exception as e:
        scores["cider"] = f"Error: {str(e)}"
    
    try:
        score, _ = f1.compute_score(gts, res)
        scores["f1"] = score
    except Exception as e:
        scores["f1"] = f"Error: {str(e)}"
    
    try:
        score, _ = meteor.compute_score(gts, res)
        scores["meteor"] = score
    except Exception as e:
        scores["meteor"] = f"Error: {str(e)}"
    
    try:
        score, _ = precision.compute_score(gts, res)
        scores["precision"] = score
    except Exception as e:
        scores["precision"] = f"Error: {str(e)}"
    
    try:
        score, _ = recall.compute_score(gts, res)
        scores["recall"] = score
    except Exception as e:
        scores["recall"] = f"Error: {str(e)}"
    
    try:
        score, _ = rouge.compute_score(gts, res)
        scores["rouge"] = score
    except Exception as e:
        scores["rouge"] = f"Error: {str(e)}"

    return scores


def compute_all_data(all_ground_truths, all_generations):
    """
    Tính toán các metric đánh giá cho toàn bộ dataset
    :param all_ground_truths: list of lists - mỗi phần tử là list các câu trả lời đúng cho 1 sample
    :param all_generations: list of strings - mỗi phần tử là câu trả lời được sinh ra cho 1 sample
    :return: dictionary chứa các điểm số trung bình
    """
    
    if len(all_ground_truths) != len(all_generations):
        raise ValueError("Số lượng ground_truths và generations phải bằng nhau")
    
    # Chuyển đổi dữ liệu thành format mà các metric module yêu cầu
    gts = {}
    res = {}
    
    for i, (ground_truths, generation) in enumerate(zip(all_ground_truths, all_generations)):
        gts[str(i)] = ground_truths  # ground truths là list
        res[str(i)] = [generation]   # generation được wrap trong list
    
    accuracy = Accuracy()
    bleu = Bleu()
    cider = Cider()
    f1 = F1()
    meteor = Meteor()
    precision = Precision()
    recall = Recall()
    rouge = Rouge()

    scores = {}
    
    try:
        score, individual_scores = accuracy.compute_score(gts, res)
        scores["accuracy"] = {
            "average": score,
            "individual": individual_scores.tolist() if hasattr(individual_scores, 'tolist') else individual_scores
        }
    except Exception as e:
        scores["accuracy"] = f"Error: {str(e)}"
    
    try:
        score, individual_scores = bleu.compute_score(gts, res)
        scores["bleu"] = {
            "average": score,
            "individual": individual_scores.tolist() if hasattr(individual_scores, 'tolist') else individual_scores
        }
    except Exception as e:
        scores["bleu"] = f"Error: {str(e)}"
    
    try:
        score, individual_scores = cider.compute_score(gts, res)
        scores["cider"] = {
            "average": score,
            "individual": individual_scores.tolist() if hasattr(individual_scores, 'tolist') else individual_scores
        }
    except Exception as e:
        scores["cider"] = f"Error: {str(e)}"
    
    try:
        score, individual_scores = f1.compute_score(gts, res)
        scores["f1"] = {
            "average": score,
            "individual": individual_scores.tolist() if hasattr(individual_scores, 'tolist') else individual_scores
        }
    except Exception as e:
        scores["f1"] = f"Error: {str(e)}"
    
    try:
        score, individual_scores = meteor.compute_score(gts, res)
        scores["meteor"] = {
            "average": score,
            "individual": individual_scores.tolist() if hasattr(individual_scores, 'tolist') else individual_scores
        }
    except Exception as e:
        scores["meteor"] = f"Error: {str(e)}"
    
    try:
        score, individual_scores = precision.compute_score(gts, res)
        scores["precision"] = {
            "average": score,
            "individual": individual_scores.tolist() if hasattr(individual_scores, 'tolist') else individual_scores
        }
    except Exception as e:
        scores["precision"] = f"Error: {str(e)}"
    
    try:
        score, individual_scores = recall.compute_score(gts, res)
        scores["recall"] = {
            "average": score,
            "individual": individual_scores.tolist() if hasattr(individual_scores, 'tolist') else individual_scores
        }
    except Exception as e:
        scores["recall"] = f"Error: {str(e)}"
    
    try:
        score, individual_scores = rouge.compute_score(gts, res)
        scores["rouge"] = {
            "average": score,
            "individual": individual_scores.tolist() if hasattr(individual_scores, 'tolist') else individual_scores
        }
    except Exception as e:
        scores["rouge"] = f"Error: {str(e)}"

    return scores