"""
Video Question Answering using GPT-4 Vision API

This script processes video frames and uses GPT-4 Vision to answer questions about the video content.
It loads video data from a JSON file, extracts frames, and evaluates the model's performance.
"""

import os
import openai
import json
import sys
import re
from typing import Dict, List, Tuple, Union, Optional

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from verl.utils.agents.frames_sampler import sample_video_frames, encode_image_to_base64

# Configuration
API_KEY = ""
DATA_PATH = "/home/zye52/scr4_hlee283/zye52/EgoSchema-processed-data/val/egoschema.json"
NUM_DATA = 10
OUTPUT_PATH = "gpt_video_responses_egoschema.json"
NUM_FRAMES = 5
MODEL_NAME = "gpt-4o"
MAX_TOKENS = 10000

# Initialize OpenAI client
client = openai.OpenAI(api_key=API_KEY)

def generate_prompt(question: str, times: List[float]) -> str:
    """Generate a simple prompt for single-round question answering."""
    return f"""Please answer the following question about the video. The video frames are sampled at timestamps: {times}.
    Answer should be a single number between 0 and 4, enclosed in <answer></answer> tags. Example: <answer>0</answer>.

    Question: {question}

    """

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

def extract_answer(text: str) -> Union[int, str]:
    """Extract answer from model response."""
    match = re.search(r'<answer>(.*?)</answer>', text)
    
    if not match:
        return -1

    answer_content = match.group(1).strip()

    # For multiple choice questions, return the number
    if answer_content in ["0", "1", "2", "3", "4"]:
        return int(answer_content)
    
    # For open-ended questions, return the text
    return answer_content

def get_answer(question: str, video_path: str, max_frames: int = 5, 
              height: int = 512, width: int = 512) -> Tuple[Union[int, str], str]:
    """Get answer from GPT-4 Vision model."""
    # Use sample_video_frames for uniform sampling
    sampled_frames, times, _ = sample_video_frames(
        video_path=video_path,
        height=height,
        width=width,
        num_frames=max_frames,
        strategy='uniform',
        ratio=0.8
    )
    
    # Create messages for API
    text_message = {"type": "text", "text": generate_prompt(question, times)}
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
    answer = extract_answer(response_text)
    return answer, response_text

def process_single_item(item: Dict) -> Dict:
    """Process a single item."""
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
        answer, response = get_answer(
            question=question,
            video_path=video_path,
            max_frames=NUM_FRAMES,
            height=item['height'],
            width=item['width']
        )
        print(f"Response: {response[:100]}...")  # Print first 100 chars of response
        
        if answer == -1:
            print("Warning: Invalid Answer")
            return {
                'id': item['id'],
                'question': question,
                'ground_truth': item['reward_model']['ground_truth'],
                'gpt_response': None,
                'error': 'Invalid answer format'
            }
            
        print(f"Answer: {answer}")
        return {
            'id': item['id'],
            'question': question,
            'ground_truth': item['reward_model']['ground_truth'],
            'gpt_response': answer,
            'response_text': response
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
    total = 0
    
    for result in results:
        try:
            # Extract prediction from response text
            prediction = re.search(r'<answer>([0-4])</answer>', result['response_text']).group(1)
            if prediction is not None:
                total += 1
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




