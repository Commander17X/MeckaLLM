import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass
from ..utils.config import load_config

@dataclass
class DeepSeekConfig:
    model_name: str = "deepseek-ai/deepseek-coder-33b-instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    quantization: bool = True
    load_in_8bit: bool = True
    trust_remote_code: bool = True

class DeepSeekModel:
    def __init__(self, config: Optional[DeepSeekConfig] = None):
        self.config = config or DeepSeekConfig()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the DeepSeek model with proper configuration"""
        try:
            # Configure quantization if enabled
            if self.config.quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=self.config.load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None

            # Load tokenizer
            self.logger.info(f"Loading tokenizer from {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )

            # Load model
            self.logger.info(f"Loading model from {self.config.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch.float16
            )

            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text using the DeepSeek model"""
        try:
            # Use provided parameters or defaults
            max_length = max_length or self.config.max_length
            temperature = temperature or self.config.temperature
            top_p = top_p or self.config.top_p

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            # Decode and return
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            raise

    def generate_code(
        self,
        prompt: str,
        language: str = "python",
        max_length: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate code with language-specific formatting"""
        try:
            # Format prompt for code generation
            formatted_prompt = f"Write {language} code for: {prompt}\n\n```{language}\n"
            
            # Generate code
            response = self.generate(
                formatted_prompt,
                max_length=max_length or self.config.max_length,
                **kwargs
            )
            
            # Extract code from response
            code = response.split(f"```{language}\n")[-1].split("```")[0].strip()
            return code
        except Exception as e:
            self.logger.error(f"Error generating code: {str(e)}")
            raise

    def analyze_code(
        self,
        code: str,
        language: str = "python",
        **kwargs
    ) -> Dict[str, Union[str, List[str]]]:
        """Analyze code for quality, security, and best practices"""
        try:
            prompt = f"""Analyze this {language} code for:
1. Code quality
2. Security issues
3. Performance optimizations
4. Best practices
5. Potential bugs

Code:
```{language}
{code}
```

Provide a detailed analysis:"""

            analysis = self.generate(prompt, **kwargs)
            
            # Parse analysis into structured format
            return {
                "raw_analysis": analysis,
                "quality_score": self._extract_quality_score(analysis),
                "security_issues": self._extract_security_issues(analysis),
                "optimizations": self._extract_optimizations(analysis),
                "best_practices": self._extract_best_practices(analysis),
                "potential_bugs": self._extract_potential_bugs(analysis)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing code: {str(e)}")
            raise

    def _extract_quality_score(self, analysis: str) -> float:
        """Extract quality score from analysis"""
        # Implement quality score extraction logic
        return 0.0

    def _extract_security_issues(self, analysis: str) -> List[str]:
        """Extract security issues from analysis"""
        # Implement security issues extraction logic
        return []

    def _extract_optimizations(self, analysis: str) -> List[str]:
        """Extract optimization suggestions from analysis"""
        # Implement optimizations extraction logic
        return []

    def _extract_best_practices(self, analysis: str) -> List[str]:
        """Extract best practices from analysis"""
        # Implement best practices extraction logic
        return []

    def _extract_potential_bugs(self, analysis: str) -> List[str]:
        """Extract potential bugs from analysis"""
        # Implement potential bugs extraction logic
        return []

    def cleanup(self):
        """Clean up model resources"""
        try:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"Error cleaning up model: {str(e)}")
            raise 