import torch
from typing import Dict, Tuple, List, Union, Optional
from dataclasses import dataclass

@dataclass
class TensorConfig:
    pad_token_id: int
    max_prompt_length: int
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.long

class TensorHelper:
    def __init__(self, config: TensorConfig):
        """Initialize TensorHelper with configuration."""
        self.config = config
        self.device = torch.device(config.device)

    def cut_to_effective_len(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        keys: List[str],
        cut_left: bool = True,
        min_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Cut tensors to their effective length based on attention mask.
        
        Args:
            tensor_dict: Dictionary of tensors
            keys: Keys of tensors to cut
            cut_left: Whether to cut from left side
            min_length: Minimum length to preserve
            
        Returns:
            Dictionary with cut tensors
        """
        if 'attention_mask' not in tensor_dict:
            raise ValueError("attention_mask required for cutting to effective length")
            
        # Calculate effective length
        mask = tensor_dict['attention_mask']
        effective_len = mask.sum(dim=1).max().item()
        
        # Apply minimum length constraint
        if min_length is not None:
            effective_len = max(effective_len, min_length)
            
        # Apply maximum length constraint
        effective_len = min(effective_len, self.config.max_prompt_length)
        
        result = tensor_dict.copy()
        for key in keys:
            if key not in tensor_dict:
                continue
                
            if cut_left:
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                result[key] = tensor_dict[key][:, :effective_len]
                
        return result

    def convert_pad_structure(
        self,
        tensor: torch.Tensor,
        pad_to_left: bool = True,
        stable: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert padding structure and return sorted tensor with indices.
        
        Args:
            tensor: Input tensor
            pad_to_left: Whether to pad to left side
            stable: Whether to use stable sorting
            
        Returns:
            Tuple of (sorted tensor, sort indices)
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a torch tensor")
            
        mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=stable)
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(
        self,
        input_ids: torch.Tensor,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Create attention mask from input ids.
        
        Args:
            input_ids: Input tensor
            pad_token_id: Optional override for pad token ID
            
        Returns:
            Attention mask tensor
        """
        pad_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        return (input_ids != pad_id).to(self.config.dtype)

    def create_position_ids(
        self,
        attention_mask: torch.Tensor,
        offset: int = 0
    ) -> torch.Tensor:
        """
        Create position ids from attention mask.
        
        Args:
            attention_mask: Attention mask tensor
            offset: Position ID offset
            
        Returns:
            Position IDs tensor
        """
        return (torch.cumsum(attention_mask, dim=1) + offset - 1) * attention_mask

    def prepare_inputs(
        self,
        messages: List[dict],
        tokenizer,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        return_tensors: str = 'pt'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input tensors from messages, maintaining visual-text-response sequence.
        
        Args:
            messages: List of message dictionaries with visual and text content
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            add_special_tokens: Whether to add special tokens
            return_tensors: Return tensor format
            
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        if not messages:
            raise ValueError("Empty messages list")

        # Process each message to separate visual and text content while maintaining order
        all_inputs = []
        for msg in messages:
            content = msg.get('content', [])
            if not isinstance(content, list):
                content = [{'type': 'text', 'text': content}]
            
            # Separate visual and text content while preserving order
            for item in content:
                if item['type'] == 'image':
                    # Process image tokens if tokenizer supports it
                    if hasattr(tokenizer, 'encode_image'):
                        image_tokens = tokenizer.encode_image(item['image'])
                        all_inputs.append(image_tokens)
                elif item['type'] == 'text':
                    # Process text tokens
                    text_tokens = tokenizer.encode(
                        item['text'],
                        add_special_tokens=False,
                        return_tensors=None
                    )
                    all_inputs.append(text_tokens)

        # Concatenate all tokens
        if return_tensors == 'pt':
            # Convert to tensors and pad
            max_len = max(len(tokens) for tokens in all_inputs) if all_inputs else 0
            padded_inputs = []
            attention_masks = []
            
            for tokens in all_inputs:
                padding_length = max_len - len(tokens)
                padded_tokens = tokens + [self.config.pad_token_id] * padding_length
                attention_mask = [1] * len(tokens) + [0] * padding_length
                
                padded_inputs.append(torch.tensor(padded_tokens, device=self.device))
                attention_masks.append(torch.tensor(attention_mask, device=self.device))
            
            input_ids = torch.stack(padded_inputs)
            attention_mask = torch.stack(attention_masks)
            
            # Truncate if needed
            if max_length:
                input_ids = input_ids[:, :max_length]
                attention_mask = attention_mask[:, :max_length]
                
            return input_ids, attention_mask
        else:
            raise ValueError(f"Unsupported return_tensors format: {return_tensors}")

    def truncate_messages(
        self,
        messages: List[dict],
        max_length: int,
        tokenizer,
        truncation_strategy: str = 'left'
    ) -> List[dict]:
        """
        Truncate messages to fit within max_length.
        
        Args:
            messages: List of message dictionaries, each with 'role' and 'content'
            max_length: Maximum sequence length
            tokenizer: Tokenizer instance
            truncation_strategy: How to truncate ('left' or 'right')
            
        Returns:
            Truncated messages list
        """
        if not messages:
            return messages
            
        # Encode all messages
        encoded = []
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, list):
                # Handle multi-modal content
                text_content = ' '.join(
                    c['text'] for c in content 
                    if c.get('type') == 'text'
                )
            else:
                text_content = content
                
            encoded.append({
                'role': msg['role'],
                'content': text_content,
                'tokens': tokenizer.encode(text_content, add_special_tokens=False)
            })
            
        # Calculate total length and truncate if needed
        total_length = sum(len(m['tokens']) for m in encoded)
        if total_length <= max_length:
            return messages
            
        # Truncate based on strategy
        if truncation_strategy == 'left':
            encoded = encoded[::-1]
            
        truncated = []
        current_length = 0
        
        for msg in encoded:
            msg_length = len(msg['tokens'])
            if current_length + msg_length <= max_length:
                truncated.append(msg)
                current_length += msg_length
            else:
                # Partially include message
                remaining = max_length - current_length
                if remaining > 0:
                    tokens = msg['tokens'][:remaining] if truncation_strategy == 'right' else msg['tokens'][-remaining:]
                    text = tokenizer.decode(tokens)
                    truncated.append({
                        'role': msg['role'],
                        'content': text
                    })
                break
                
        if truncation_strategy == 'left':
            truncated = truncated[::-1]
            
        return truncated

    def concatenate_with_padding(self, tensors: List[torch.Tensor], 
                               pad_to_left: bool = True) -> torch.Tensor:
        """Concatenate tensors and handle padding."""
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def _example_level_pad(self, responses: torch.Tensor, 
                          responses_str: List[str], 
                          active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        Pad responses for non-active examples with pad tokens.
        """

        assert active_mask.sum() == responses.shape[0]
        # Create masked responses tensor
        batch_size = active_mask.shape[0]
        seq_len = responses.shape[1]
        padded_responses = torch.full(
            (batch_size, seq_len), self.config.pad_token_id,
            dtype=responses.dtype, device=responses.device
        )
        padded_responses[active_mask] = responses
         
        # Create masked response strings
        padded_responses_str = [""] * batch_size
        
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1
                
        return padded_responses, padded_responses_str