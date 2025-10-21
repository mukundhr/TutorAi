"""
Transformers-based model handler for question generation
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('..')
import config
import os

class TransformersHandler:
    def __init__(self, preferred_model: str = None):
        """
        Initialize with fallback system for model loading
        Tries multiple models in order of preference
        """
        # Define model hierarchy (best to most compatible)
        self.model_hierarchy = [
            "microsoft/DialoGPT-medium",  # Best for dialogue/questions
            "gpt2-medium",                # Larger GPT-2
            "gpt2",                       # Standard GPT-2
            "distilgpt2",                 # Smaller, faster GPT-2
            "facebook/opt-125m"           # Smallest fallback
        ]
        
        # If user specified a preferred model, try it first
        if preferred_model:
            self.model_hierarchy.insert(0, preferred_model)
        
        self.model_name = None
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_attempts = []
        
        self._load_model_with_fallback()
    
    def _load_model_with_fallback(self):
        """Load model with comprehensive fallback system"""
        print(f"Using device: {self.device}")
        
        for i, model_name in enumerate(self.model_hierarchy):
            try:
                print(f"Attempting to load model {i+1}/{len(self.model_hierarchy)}: {model_name}")
                
                # Record attempt
                attempt = {"model": model_name, "success": False, "error": None}
                
                # Load tokenizer first
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                # Add pad token if it doesn't exist
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model with conservative settings for maximum compatibility
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float32,  # Use float32 for better compatibility
                }
                
                # Only use GPU optimizations if CUDA is available and stable
                if self.device.type == "cuda":
                    try:
                        model_kwargs["torch_dtype"] = torch.float16
                        model_kwargs["device_map"] = "auto"
                    except:
                        # Fallback to CPU-friendly settings
                        model_kwargs["torch_dtype"] = torch.float32
                
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                
                # Move to device if not using device_map
                if "device_map" not in model_kwargs:
                    model = model.to(self.device)
                
                # Test the model with a simple generation
                self._test_model(model, tokenizer)
                
                # If we get here, the model works
                self.model = model
                self.tokenizer = tokenizer
                self.model_name = model_name
                attempt["success"] = True
                self.load_attempts.append(attempt)
                
                print(f"[OK] Successfully loaded model: {model_name}")
                return
                
            except Exception as e:
                attempt["error"] = str(e)
                self.load_attempts.append(attempt)
                print(f"[ERROR] Failed to load {model_name}: {str(e)}")
                
                # Clean up any partially loaded resources
                try:
                    if 'model' in locals():
                        del model
                    if 'tokenizer' in locals():
                        del tokenizer
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                
                continue
        
        # If we get here, all models failed
        error_summary = "\n".join([
            f"- {attempt['model']}: {attempt['error']}" 
            for attempt in self.load_attempts 
            if not attempt['success']
        ])
        raise Exception(f"Failed to load any transformer model. Attempts:\n{error_summary}")
    
    def _test_model(self, model, tokenizer):
        """Test if model can generate text properly"""
        try:
            test_input = tokenizer("Test", return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = model.generate(
                    test_input.input_ids,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            # If this doesn't throw an exception, the model works
        except Exception as e:
            raise Exception(f"Model test failed: {str(e)}")
    
    def get_model_info(self):
        """Get information about the loaded model and attempts"""
        return {
            "loaded_model": self.model_name,
            "device": str(self.device),
            "attempts": self.load_attempts,
            "total_attempts": len(self.load_attempts)
        }
    
    def generate(self, 
                 prompt: str,
                 max_tokens: int = 200,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 do_sample: bool = True,
                 retry_on_failure: bool = True) -> str:
        """Generate text using transformers with error handling"""
        
        if not self.model or not self.tokenizer:
            raise Exception("Model not loaded")
        
        # Try generation with error handling
        for attempt in range(3 if retry_on_failure else 1):
            try:
                # Tokenize input with length checking
                inputs = self.tokenizer.encode(
                    prompt[:1024],  # Truncate very long prompts
                    return_tensors="pt",
                    truncation=True
                ).to(self.device)
                
                # Adjust parameters for model compatibility
                generation_kwargs = {
                    "max_new_tokens": min(max_tokens, 1024),  # Increased cap for more questions
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "num_return_sequences": 1,
                    "early_stopping": True
                }
                
                # Add sampling parameters only if supported
                if do_sample:
                    generation_kwargs.update({
                        "do_sample": True,
                        "temperature": max(temperature, 0.1),  # Ensure minimum temperature
                        "top_p": min(top_p, 0.95)  # Ensure maximum top_p
                    })
                else:
                    generation_kwargs["do_sample"] = False
                
                # Generate with timeout protection
                with torch.no_grad():
                    outputs = self.model.generate(inputs, **generation_kwargs)
                
                # Decode response
                generated = outputs[0][inputs.shape[1]:]  # Remove input tokens
                response = self.tokenizer.decode(generated, skip_special_tokens=True)
                
                return response.strip()
                
            except Exception as e:
                if attempt < 2 and retry_on_failure:
                    print(f"Generation attempt {attempt + 1} failed: {str(e)}, retrying...")
                    # Try with more conservative parameters
                    do_sample = False
                    max_tokens = min(max_tokens, 100)
                    temperature = 0.1
                    continue
                else:
                    raise Exception(f"Error generating text after {attempt + 1} attempts: {str(e)}")
        
        return ""  # Fallback
    
    def generate_structured(self, 
                          content: str, 
                          num_questions: int = 3) -> str:
        """Generate structured questions from content"""
        
        prompt = f"""Based on the following content, generate {num_questions} short answer questions.

Content:
{content[:800]}

Instructions:
- Create questions that test understanding of key concepts
- Each question should be answerable in 2-3 sentences
- Format as: Q1: [question] A1: [answer]

Questions:
"""
        
        return self.generate(prompt, max_tokens=400, temperature=0.8)
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'model') and self.model:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()