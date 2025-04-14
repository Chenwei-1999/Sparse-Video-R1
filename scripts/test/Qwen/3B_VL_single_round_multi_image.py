"""
Video Question Answering using Qwen2.5-VL-3B-Instruct

This script processes video frames and uses Qwen2.5-VL-3B-Instruct to answer questions about the video content.
It loads video data from a JSON file, extracts frames, and evaluates the model's performance.
"""

import os
import json
import sys
import re
import torch
from typing import Dict, List, Tuple, Union, Optional
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import io
import base64

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from verl.utils.agents.frames_sampler import sample_video_frames

# Configuration
DATA_PATH = "/mnt/c/Users/Chenwei/Desktop/VideoAgent/data/val/nextqa.json"
NUM_DATA = 100
OUTPUT_PATH = "qwen_video_responses.json"
NUM_FRAMES = 5
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_TOKENS = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Qwen model, tokenizer and processor
print("Loading Qwen model, tokenizer and processor...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

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
    """Get answer from Qwen model."""
    # Use sample_video_frames for uniform sampling
    sampled_frames, times, _ = sample_video_frames(
        video_path=video_path,
        height=height,
        width=width,
        num_frames=max_frames,
        strategy='uniform',
        ratio=0.8
    )
    
    # Prepare messages with multiple images and text query
    messages = [{
        "role": "user",
        "content": []
    }]
    
    # Add images to messages
    for frame in sampled_frames:
        # Convert frame bytes to base64
        frame_base64 = base64.b64encode(frame['bytes']).decode('utf-8')
        messages[0]["content"].append({
            "type": "image",
            "image": f"data:image/jpeg;base64,{frame_base64}"
        })
    
    # Add text query
    prompt = generate_prompt(question, times)
    messages[0]["content"].append({
        "type": "text",
        "text": prompt
    })
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Prepare inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    # Generate response
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False
    )
    
    # Decode response
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
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
            'qwen_response': None,
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
                'qwen_response': None,
                'error': 'Invalid answer format'
            }
            
        print(f"Answer: {answer}")
        return {
            'id': item['id'],
            'question': question,
            'ground_truth': item['reward_model']['ground_truth'],
            'qwen_response': answer,
            'response_text': response
        }
    except Exception as e:
        print(f"Error processing item: {str(e)}")
        return {
            'id': item['id'],
            'question': question,
            'ground_truth': item['reward_model']['ground_truth'],
            'qwen_response': None,
            'error': str(e)
        }

def calculate_accuracy(results: List[Dict]) -> Tuple[float, int, int]:
    """Calculate accuracy of model predictions."""
    correct = 0
    total = len(results)
    
    for result in results:
        try:
            # Extract prediction from response text
            prediction = re.search(r'<answer>([0-4])</answer>', result['response_text']).group(1)
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
                    'qwen_response': None,
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