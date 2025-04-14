"""
Video Question Answering using GPT-4 Vision API with Multi-round Processing

This script processes video frames and uses GPT-4 Vision to answer questions about the video content
through multiple rounds of interaction. It loads video data from a JSON file, extracts frames,
and evaluates the model's performance.
"""

import os
import openai
import json
import re
import sys
from typing import Dict, List, Tuple, Union, Optional

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from verl.utils.agents.construct_prompt import generate_prompt
from verl.utils.agents.frames_sampler import sample_video_frames, sample_frames_from_next_obs, encode_image_to_base64

# Configuration
API_KEY = ""
DATA_PATH = ""
OUTPUT_PATH = "gpt_video_responses_multiround.json"
NUM_FRAMES = 5
MAX_FRAMES = 5
NUM_DATA = 100
MODEL_NAME = "gpt-4o"
MAX_TOKENS = 10000
MAX_ROUNDS = 5

# Initialize OpenAI client
client = openai.OpenAI(api_key=API_KEY)



def load_data(file_path: str) -> List[Dict]:
    """Load and validate the dataset from JSON file."""
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data[:NUM_DATA]
    except FileNotFoundError:
        print(f"Error: Data file not found: {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {file_path}")
        raise

def extract_answer(text: str) -> Tuple[Union[int, str], Optional[List[float]]]:
    """Extract answer and frame modifications from model response."""
    # Extract answer
    answer_match = re.search(r'<answer>(.*?)</answer>', text)
    if not answer_match:
        return -1, None
        
    answer_content = answer_match.group(1).strip()
    
    # Check for frame modifications in the answer
    frame_modifications = None
    if '+' in answer_content or '-' in answer_content:
        try:
            # Extract all numbers from the modifications
            numbers = re.findall(r'[+-]\[([\d.,\s]+)\]', answer_content)
            if numbers:
                # Flatten the list of numbers and convert to floats
                frame_modifications = []
                for num_str in numbers:
                    frame_modifications.extend([float(x.strip()) for x in num_str.split(',')])
                frame_modifications = sorted(list(set(frame_modifications)))  # Remove duplicates and sort
        except:
            frame_modifications = None
    
    # If there are frame modifications, return -1 to continue the round
    if frame_modifications:
        return -1, frame_modifications
    
    # For multiple choice questions, return the number
    if answer_content in ["0", "1", "2", "3", "4"]:
        return int(answer_content), None
    
    # For open-ended questions, return the text
    return answer_content, None

def get_answer(question: str, video_path: str, max_frames: int = 5, 
              height: int = 512, width: int = 512,
              frames: Optional[List[float]] = None,
              n_round: int = 1, max_rounds: int = 5,
              previous_frames: List[List[float]] = None) -> Tuple[Union[int, str], str, Optional[List[float]]]:
    """Get answer from GPT-4 Vision model."""
    if frames is not None:
        # Use sample_frames_from_next_obs for specific timestamps
        sampled_frames = sample_frames_from_next_obs(
            video_path=video_path,
            timestamps=frames,
            height=height,
            width=width,
            ratio=0.8
        )
        times = frames
    else:
        # Use sample_video_frames for random/uniform sampling
        sampled_frames, times, _ = sample_video_frames(
            video_path=video_path,
            height=height,
            width=width,
            num_frames=max_frames,
            strategy='uniform',
            ratio=0.8
        )
    
    # Create messages for API
    text_message = {"type": "text", "text": generate_prompt(
        question=question,
        timestamps=times,
        n_round=n_round,
        max_rounds=max_rounds,
        max_frames=max_frames,
        previous_frames=previous_frames
    )}
    image_messages = [
        {"type": "image_url", "image_url": {"url": encode_image_to_base64(frame['bytes'])}}
        for frame in sampled_frames
    ]
    combined_content = [text_message] + image_messages

    # Get model response
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": combined_content}],
        max_tokens=MAX_TOKENS,
    )
    response_text = response.choices[0].message.content
    answer, frame_modifications = extract_answer(response_text)
    return answer, response_text, frame_modifications

def process_single_item(item: Dict) -> Dict:
    """Process a single item with multiple rounds."""
    print(f"Processing item {item['id']}")
    
    # Validate and construct video path
    video_path = item['video']
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return {
            'id': item['id'],
            'question': item["problem"],
            'ground_truth': item['reward_model']['ground_truth'],
            'gpt_response': None,
            'error': 'Video file not found'
        }
    
    question = item["problem"]
    print(f"Video path: {video_path}")
    print(f"Question: {question}")
    
    try:
        # Initialize round tracking
        current_frames = None
        round_responses = []
        previous_times = []
        exceeded_max_frames = False
        
        for round_num in range(1, MAX_ROUNDS + 1):
            print(f"Round {round_num}")
            
            # Get answer for current round
            answer, response, frame_modifications = get_answer(
                question=question,
                video_path=video_path,
                max_frames=NUM_FRAMES,
                height=item['height'],
                width=item['width'],
                frames=current_frames,
                n_round=round_num,
                max_rounds=MAX_ROUNDS,
                previous_frames=previous_times
            )
            
            # Log response
            print(f"Response: {response[:100]}...")
            round_responses.append(response)
            
            # If we got a valid answer (not -1), we're done
            if answer != -1:
                print(f"Final Answer: {answer}")
                return {
                    'id': item['id'],
                    'question': question,
                    'ground_truth': item['reward_model']['ground_truth'],
                    'gpt_response': answer,
                    'response_text': round_responses,
                    'rounds': round_num,
                    'exceeded_max_frames': exceeded_max_frames
                }
            
            # If no frame modifications requested, we're done
            if not frame_modifications:
                print("Warning: No frame modifications requested")
                return {
                    'id': item['id'],
                    'question': question,
                    'ground_truth': item['reward_model']['ground_truth'],
                    'gpt_response': None,
                    'response_text': round_responses,
                    'error': 'No frame modifications requested',
                    'rounds': round_num,
                    'exceeded_max_frames': exceeded_max_frames
                }
            
            # Validate frame modifications
            if any(frame < 0 for frame in frame_modifications):
                print("Warning: Invalid frame timestamps (negative)")
                return {
                    'id': item['id'],
                    'question': question,
                    'ground_truth': item['reward_model']['ground_truth'],
                    'gpt_response': None,
                    'response_text': round_responses,
                    'error': 'Invalid frame timestamps (negative)',
                    'rounds': round_num,
                    'exceeded_max_frames': exceeded_max_frames
                }
            
            # Check if total frames would exceed MAX_FRAMES
            if current_frames is not None:
                # Calculate new total frames after modifications
                new_frames = set(current_frames)
                add_frames = [f for f in frame_modifications if f not in current_frames]
                remove_frames = [f for f in current_frames if f not in frame_modifications]
                
                new_frames.update(add_frames)
                new_frames.difference_update(remove_frames)
                
                if len(new_frames) > MAX_FRAMES:
                    print(f"Warning: Frame modifications would exceed MAX_FRAMES ({MAX_FRAMES})")
                    exceeded_max_frames = True
                    # Continue to next round with current frames
                    continue
            
            # Update frames for next round
            current_frames = frame_modifications
            previous_times.append(current_frames)
            print(f"Frame modifications: {frame_modifications}")
            
            # If we've reached max rounds, we're done
            if round_num == MAX_ROUNDS:
                print("Warning: Reached maximum number of rounds")
                return {
                    'id': item['id'],
                    'question': question,
                    'ground_truth': item['reward_model']['ground_truth'],
                    'gpt_response': None,
                    'response_text': round_responses,
                    'error': 'Reached maximum number of rounds',
                    'rounds': MAX_ROUNDS,
                    'exceeded_max_frames': exceeded_max_frames
                }
        
    except Exception as e:
        print(f"Error processing item: {str(e)}")
        return {
            'id': item['id'],
            'question': question,
            'ground_truth': item['reward_model']['ground_truth'],
            'gpt_response': None,
            'error': str(e)
        }

def calculate_accuracy(results: List[Dict]) -> Tuple[float, int, int]:
    """Calculate accuracy of model predictions."""
    correct = 0
    total = len(results)
    
    for result in results:
        try:
            # Extract prediction from response text
            prediction = re.search(r'<answer>([0-4])</answer>', result['response_text'][-1]).group(1)
            # Compare with ground truth (convert both to string for comparison)
            if str(prediction) == str(result['ground_truth']):
                correct += 1
        except:
            continue
            
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy, correct, total

def save_results(results: List[Dict], output_path: str) -> None:
    """Save results to JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def main():
    """Main execution function."""
    try:
        # Load data
        print("Loading data...")
        data = load_data(DATA_PATH)
        
        # Process each item
        results = []
        for item in data:
            print(f"Processing item {item['id']}...")
            
            try:
                result = process_single_item(item)
                results.append(result)
                print(f"Completed item {item['id']}")
                
            except Exception as e:
                print(f"Error processing item {item['id']}: {str(e)}")
                results.append({
                    'id': item['id'],
                    'question': item["problem"],
                    'ground_truth': item['reward_model']['ground_truth'],
                    'gpt_response': None,
                    'error': str(e)
                })
                continue
        
        # Calculate and display accuracy
        accuracy, correct, total = calculate_accuracy(results)
        print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total} correct)")
        
        # Save results
        save_results(results, OUTPUT_PATH)
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
