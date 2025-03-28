

import re
import random
def extract_solution(solution_str):
    match = re.search(r'<answer>(.*?)</answer>', solution_str)
    
    if not match:
        return -1  # No <answer> tag found

    answer_content = match.group(1).strip()

    # Case 1: Direct numeric answer in the range 0-4
    if answer_content in ["0", "1", "2", "3", "4"]:
        return int(answer_content)
    
    return -1


def compute_score(solution_str, ground_truth, extra_info=None,
                  format_score=0., score=1.):
    '''
    solution_str: the solution text
    ground_truth: the ground truth
    n_iter: number of iterations currently used
    n_frames: number of frames currently used
    max_iter: maximum number of iterations allowed without penalty
    max_frames: maximum number of frames allowed without penalty
    iter_decay: decay factor for iterations
    frame_decay: decay factor for frames
    format_score: the score for the format
    score: the score for the correct answer

    Returns: the score
    '''
    answer = extract_solution(solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Ground_truth: {ground_truth} | Answer: {answer}")
        # print(f"Raw Chat: \n {extra_info['row_chat']}")
        # print(f"Prompt: \n {extra_info['prompt']}")
        # print(f"Solution string: {solution_str}")
        print(f"--------------------------------")


    n_iter = extra_info.get("n_iter", None)
    n_frames = extra_info.get("n_frames", None)
    max_iter = extra_info.get("max_iter", None)
    max_frames = extra_info.get("max_frames", None)
    iter_decay = extra_info.get("iter_decay", None)
    frame_decay = extra_info.get("frame_decay", None)
    score = 1
    if answer==-1:
        return -1  # No <answer> tag found, bad result, score -1, or still struggling in choosing frames
    answer = int(answer)
    ground_truth = int(ground_truth)
    if answer != ground_truth: # wrong answer, score 0
        return format_score

    if n_iter and max_iter:
        iter_penalty = max(0, n_iter - max_iter)
        score *= iter_decay ** iter_penalty
    
    if n_frames and max_frames:
        frame_penalty = max(0, n_frames - max_frames)
        score *= frame_decay ** frame_penalty
      
    return score