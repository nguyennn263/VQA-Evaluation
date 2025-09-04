from accuracy import Accuracy
from bleu import Bleu
from cider import Cider
from f1 import F1
from meteor import Meteor
from precision import Precision
from recall import Recall 
from rouge import Rouge
from tqdm import tqdm


def compute_score(ground_truths, generation, use_max_score=True):
    """
    Tính toán các metric đánh giá cho VQA
    :param ground_truths: list các câu trả lời đúng (list of strings)
    :param generation: câu trả lời được sinh ra (string)
    :param use_max_score: nếu True, lấy max score giữa generation và các ground truths
                         nếu False, lấy mean score (behavior cũ)
    :return: dictionary chứa các điểm số
    """
    
    if use_max_score:
        # Tính score với từng ground truth riêng biệt và lấy max
        return _compute_score_with_max(ground_truths, generation)
    else:
        # Behavior cũ: tính với tất cả ground truths cùng lúc
        return _compute_score_with_mean(ground_truths, generation)


def _compute_score_with_max(ground_truths, generation):
    """
    Tính score bằng cách so sánh generation với từng ground truth và lấy max
    """
    max_scores = {}
    
    # Khởi tạo các metric
    accuracy = Accuracy()
    bleu = Bleu()
    cider = Cider()
    f1 = F1()
    meteor = Meteor()
    precision = Precision()
    recall = Recall()
    rouge = Rouge()
    
    metrics = [
        ("accuracy", accuracy),
        ("bleu", bleu),
        ("cider", cider),
        ("f1", f1),
        ("meteor", meteor),
        ("precision", precision),
        ("recall", recall),
        ("rouge", rouge)
    ]
    
    # Với mỗi metric, tính score với từng ground truth và lấy max
    for metric_name, metric_obj in metrics:
        try:
            scores_list = []
            
            # Tính score với từng ground truth riêng biệt
            for gt in ground_truths:
                gts_single = {'0': [gt]}  # chỉ 1 ground truth
                res_single = {'0': [generation]}
                
                try:
                    score, _ = metric_obj.compute_score(gts_single, res_single)
                    
                    # Xử lý đặc biệt cho BLEU
                    if metric_name == "bleu" and isinstance(score, (list, tuple)):
                        score = score[-1] if len(score) > 0 else 0.0
                    
                    scores_list.append(float(score))
                except Exception as e:
                    # Nếu lỗi với ground truth này, gán score = 0
                    scores_list.append(0.0)
            
            # Lấy max score
            if scores_list:
                max_scores[metric_name] = max(scores_list)
            else:
                max_scores[metric_name] = 0.0
                
        except Exception as e:
            max_scores[metric_name] = f"Error: {str(e)}"
    
    return max_scores


def _compute_score_with_mean(ground_truths, generation):
    """
    Tính score theo cách cũ (mean của tất cả ground truths)
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
        scores["accuracy"] = float(score) if not isinstance(score, str) else score
    except Exception as e:
        scores["accuracy"] = f"Error: {str(e)}"
    
    try:
        score, _ = bleu.compute_score(gts, res)
        # BLEU có thể trả về tuple
        if isinstance(score, (list, tuple)):
            score = score[-1] if len(score) > 0 else 0.0
        scores["bleu"] = float(score) if not isinstance(score, str) else score
    except Exception as e:
        scores["bleu"] = f"Error: {str(e)}"
    
    try:
        score, _ = cider.compute_score(gts, res)
        scores["cider"] = float(score) if not isinstance(score, str) else score
    except Exception as e:
        scores["cider"] = f"Error: {str(e)}"
    
    try:
        score, _ = f1.compute_score(gts, res)
        scores["f1"] = float(score) if not isinstance(score, str) else score
    except Exception as e:
        scores["f1"] = f"Error: {str(e)}"
    
    try:
        score, _ = meteor.compute_score(gts, res)
        scores["meteor"] = float(score) if not isinstance(score, str) else score
    except Exception as e:
        scores["meteor"] = f"Error: METEOR - {str(e)}"
    
    try:
        score, _ = precision.compute_score(gts, res)
        scores["precision"] = float(score) if not isinstance(score, str) else score
    except Exception as e:
        scores["precision"] = f"Error: {str(e)}"
    
    try:
        score, _ = recall.compute_score(gts, res)
        scores["recall"] = float(score) if not isinstance(score, str) else score
    except Exception as e:
        scores["recall"] = f"Error: {str(e)}"
    
    try:
        score, _ = rouge.compute_score(gts, res)
        scores["rouge"] = float(score) if not isinstance(score, str) else score
    except Exception as e:
        scores["rouge"] = f"Error: {str(e)}"

    return scores


def compute_all_data(all_ground_truths, all_generations, use_max_score=True):
    """
    Tính toán các metric đánh giá cho toàn bộ dataset
    :param all_ground_truths: list of lists - mỗi phần tử là list các câu trả lời đúng cho 1 sample
    :param all_generations: list of strings - mỗi phần tử là câu trả lời được sinh ra cho 1 sample
    :param use_max_score: nếu True, với mỗi sample sẽ lấy max score giữa generation và các ground truths
                         nếu False, sử dụng behavior cũ (mean score)
    :return: dictionary chứa các điểm số trung bình
    """
    
    if len(all_ground_truths) != len(all_generations):
        raise ValueError("Số lượng ground_truths và generations phải bằng nhau")
    
    if use_max_score:
        return _compute_all_data_with_max(all_ground_truths, all_generations)
    else:
        return _compute_all_data_with_mean(all_ground_truths, all_generations)


def _compute_all_data_with_max(all_ground_truths, all_generations):
    """
    Tính toán metrics cho toàn bộ dataset sử dụng max score cho mỗi sample
    """
    print("Computing evaluation metrics with max score strategy...")
    
    all_scores = {}
    metric_names = ["accuracy", "bleu", "cider", "f1", "meteor", "precision", "recall", "rouge"]
    
    # Khởi tạo dictionary cho từng metric
    for metric_name in metric_names:
        all_scores[metric_name] = []
    
    # Tính score cho từng sample
    for i, (ground_truths, generation) in tqdm(enumerate(zip(all_ground_truths, all_generations)), 
                                               total=len(all_ground_truths),
                                               desc="Processing samples"):
        # Tính score cho sample này với max strategy
        sample_scores = compute_score(ground_truths, generation, use_max_score=True)
        
        # Lưu score của sample vào list tương ứng
        for metric_name in metric_names:
            if isinstance(sample_scores[metric_name], (int, float)):
                all_scores[metric_name].append(sample_scores[metric_name])
            else:
                # Nếu có lỗi, gán score = 0
                all_scores[metric_name].append(0.0)
    
    # Tính mean của tất cả samples cho mỗi metric
    final_scores = {}
    for metric_name in metric_names:
        scores_list = all_scores[metric_name]
        if scores_list:
            avg_score = sum(scores_list) / len(scores_list)
            final_scores[metric_name] = {
                "average": avg_score,
                "individual": scores_list
            }
            print(f"✓ {metric_name.upper()}: {avg_score:.4f}")
        else:
            final_scores[metric_name] = "Error: No valid scores"
            print(f"✗ {metric_name.upper()}: No valid scores")
    
    return final_scores


def _compute_all_data_with_mean(all_ground_truths, all_generations):
    """
    Tính toán các metric đánh giá cho toàn bộ dataset
    :param all_ground_truths: list of lists - mỗi phần tử là list các câu trả lời đúng cho 1 sample
    :param all_generations: list of strings - mỗi phần tử là câu trả lời được sinh ra cho 1 sample
    :return: dictionary chứa các điểm số trung bình
    """
    
    if len(all_ground_truths) != len(all_generations):
        raise ValueError("Số lượng ground_truths và generations phải bằng nhau")
    
    print("Preparing data for evaluation...")
    # Chuyển đổi dữ liệu thành format mà các metric module yêu cầu
    gts = {}
    res = {}
    
    for i, (ground_truths, generation) in tqdm(enumerate(zip(all_ground_truths, all_generations)), 
                                               total=len(all_ground_truths),
                                               desc="Processing samples"):
        # Đảm bảo ground_truths là list of strings
        clean_gts = []
        for gt in ground_truths:
            if isinstance(gt, str):
                clean_gts.append(gt.strip())
            else:
                clean_gts.append(str(gt).strip())
        
        # Đảm bảo generation là string
        if isinstance(generation, str):
            clean_gen = generation.strip()
        else:
            clean_gen = str(generation).strip()
        
        gts[str(i)] = clean_gts  # ground truths là list of strings
        res[str(i)] = [clean_gen]   # generation được wrap trong list
    
    accuracy = Accuracy()
    bleu = Bleu()
    cider = Cider()
    f1 = F1()
    meteor = Meteor()
    precision = Precision()
    recall = Recall()
    rouge = Rouge()

    scores = {}
    
    # Tạo list các metrics để tính toán với progress bar
    metrics = [
        ("accuracy", accuracy),
        ("bleu", bleu),
        ("cider", cider),
        ("f1", f1),
        ("meteor", meteor),
        ("precision", precision),
        ("recall", recall),
        ("rouge", rouge)
    ]
    
    print("Computing evaluation metrics...")
    for metric_name, metric_obj in metrics:
        try:
            # Special handling for specific metrics
            if metric_name == "bleu":
                # BLEU có thể cần format đặc biệt
                score, individual_scores = metric_obj.compute_score(gts, res)
                # Xử lý case BLEU trả về tuple hoặc list
                if isinstance(score, (list, tuple)):
                    score = score[-1] if len(score) > 0 else 0.0
            elif metric_name == "meteor":
                # METEOR có thể có vấn đề với Java subprocess
                try:
                    score, individual_scores = metric_obj.compute_score(gts, res)
                except Exception as meteor_error:
                    print(f"METEOR specific error: {meteor_error}")
                    scores[metric_name] = f"Error: METEOR Java subprocess failed - {str(meteor_error)}"
                    continue
            else:
                score, individual_scores = metric_obj.compute_score(gts, res)
            
            # Ensure score is a number
            if isinstance(score, str):
                # Try to extract number from string
                import re
                numbers = re.findall(r'-?\d+\.?\d*', score)
                if numbers:
                    score = float(numbers[-1])
                else:
                    raise ValueError(f"Cannot extract numeric score from: {score}")
            
            scores[metric_name] = {
                "average": float(score),
                "individual": individual_scores.tolist() if hasattr(individual_scores, 'tolist') else individual_scores
            }
            print(f"✓ {metric_name.upper()}: {score:.4f}")
            
        except Exception as e:
            error_msg = str(e)
            scores[metric_name] = f"Error: {error_msg}"
            print(f"✗ {metric_name.upper()}: Error - {error_msg}")
            
            # For debugging: print more specific error info
            if metric_name in ["bleu", "meteor"]:
                print(f"  Debug info for {metric_name}: {type(e).__name__}")
                print(f"  Data sample - GT: {list(gts.keys())[:2]}, RES: {list(res.keys())[:2]}")
                if list(gts.keys()):
                    first_key = list(gts.keys())[0]
                    print(f"  Sample data - GT[{first_key}]: {gts[first_key][:2]}")
                    print(f"  Sample data - RES[{first_key}]: {res[first_key]}")

    return scores