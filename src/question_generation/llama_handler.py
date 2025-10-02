"""
LLaMA 2 model handler using llama-cpp-python
"""
from llama_cpp import Llama
import sys
sys.path.append('..')
import config
import os

class LlamaHandler:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.LLAMA_MODEL_PATH
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the quantized LLaMA model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Please download LLaMA 2 7B quantized model (GGUF format) "
                f"and place it in the models directory."
            )
        
        try:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=config.MODEL_CONTEXT_LENGTH,
                n_threads=4,  # Adjust based on your CPU
                n_gpu_layers=0,  # Set to > 0 if you have GPU support
                verbose=False
            )
            print(f"[OK] Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def generate(self, 
                 prompt: str,
                 system_prompt: str = "",
                 max_tokens: int = None,
                 temperature: float = None,
                 top_p: float = None,
                 stop: list = None) -> str:
        """Generate text using LLaMA 2"""
        
        if not self.model:
            raise Exception("Model not loaded")
        
        # Format prompt for LLaMA 2 Chat
        if system_prompt:
            formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        else:
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Set defaults
        max_tokens = max_tokens or config.MODEL_MAX_TOKENS
        temperature = temperature or config.MODEL_TEMPERATURE
        top_p = top_p or config.MODEL_TOP_P
        
        try:
            response = self.model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or ["</s>", "[INST]"],
                echo=False
            )
            
            generated_text = response['choices'][0]['text'].strip()
            return generated_text
            
        except Exception as e:
            raise Exception(f"Error generating text: {str(e)}")
    
    def generate_batch(self, prompts: list, **kwargs) -> list:
        """Generate text for multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results
    
    def __del__(self):
        """Cleanup"""
        if self.model:
            del self.model