

import re
import random
def extract_solution(solution_str):
    match = re.search(r'<answer>(.*?)</answer>', solution_str)
    
    if not match:
        return -1  # No <answer> tag found

def compute_score(solution_str, ground_truth,  format_score=0., score=1.):
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



    score = 1
    if answer==-1:
        return -1  # No <answer> tag found, bad result, score -1, or still struggling in choosing frames
    answer = str(answer)
    ground_truth = str(ground_truth)
    if answer != ground_truth: # wrong answer, score 0
        return format_score

    return score