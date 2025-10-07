"""
Google Gemini model handler using google-generativeai
"""
import google.generativeai as genai
import os
import sys
sys.path.append('..')
import config

class GeminiHandler:
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini handler
        
        Args:
            api_key: Google API key for Gemini. If None, reads from GEMINI_API_KEY env var
            model_name: Name of the Gemini model to use (default: gemini-2.0-flash-exp - Gemini 2.0 Flash)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.model = None
        self._configure_api()
        self._load_model()
    
    def _configure_api(self):
        """Configure the Gemini API"""
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Please set GEMINI_API_KEY environment variable or provide api_key parameter. "
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )
        
        try:
            genai.configure(api_key=self.api_key)
            print("[OK] Gemini API configured successfully")
        except Exception as e:
            raise Exception(f"Error configuring Gemini API: {str(e)}")
    
    def _load_model(self):
        """Load the Gemini model"""
        try:
            # Configure generation settings
            generation_config = {
                "temperature": config.MODEL_TEMPERATURE,
                "top_p": config.MODEL_TOP_P,
                "top_k": 40,
                "max_output_tokens": config.MODEL_MAX_TOKENS,
            }
            
            # Configure safety settings (optional - adjust as needed)
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            print(f"[OK] Gemini model '{self.model_name}' loaded successfully")
        except Exception as e:
            raise Exception(f"Error loading Gemini model: {str(e)}")
    
    def generate(self,
                 prompt: str,
                 system_prompt: str = "",
                 max_tokens: int = None,
                 temperature: float = None,
                 top_p: float = None,
                 stop: list = None) -> str:
        """
        Generate text using Gemini
        
        Args:
            prompt: The user prompt
            system_prompt: System instructions (prepended to prompt)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 - 1.0)
            top_p: Nucleus sampling parameter
            stop: Stop sequences (not directly supported by Gemini, handled post-generation)
        
        Returns:
            Generated text string
        """
        if not self.model:
            raise Exception("Model not loaded")
        
        try:
            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Update generation config if custom parameters provided
            generation_config = {}
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            if temperature is not None:
                generation_config["temperature"] = temperature
            if top_p is not None:
                generation_config["top_p"] = top_p
            
            # Generate response
            if generation_config:
                # Create a new model instance with custom config
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config
                )
                response = model.generate_content(full_prompt)
            else:
                response = self.model.generate_content(full_prompt)
            
            # Extract text from response
            generated_text = response.text.strip()
            
            # Handle stop sequences if provided
            if stop:
                for stop_seq in stop:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0].strip()
            
            return generated_text
            
        except Exception as e:
            # Handle specific Gemini errors
            error_msg = str(e)
            if "API key" in error_msg:
                raise Exception("Invalid Gemini API key. Please check your GEMINI_API_KEY.")
            elif "quota" in error_msg.lower():
                raise Exception("Gemini API quota exceeded. Please check your usage limits.")
            elif "safety" in error_msg.lower():
                raise Exception("Content blocked by Gemini safety filters. Try rephrasing your prompt.")
            else:
                raise Exception(f"Error generating text with Gemini: {error_msg}")
    
    def generate_batch(self, prompts: list, **kwargs) -> list:
        """
        Generate text for multiple prompts
        
        Args:
            prompts: List of prompt strings
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated text strings
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to generate for prompt: {str(e)}")
                results.append("")  # Empty string for failed generations
        return results
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "provider": "Google Gemini",
            "api_configured": self.api_key is not None,
            "model_loaded": self.model is not None
        }
