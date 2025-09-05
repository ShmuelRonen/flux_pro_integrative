import requests
from PIL import Image
import io
import numpy as np
import torch
import os
import configparser
import time
import base64
import json
from enum import Enum
from typing import Dict, Any, Tuple, Optional
import folder_paths


class TaskStatus(Enum):
    PENDING = "Pending"
    READY = "Ready"
    ERROR = "Error"


class FluxAPIError(Exception):
    """Custom exception for Flux API errors"""
    pass


class ConfigManager:
    """Handles configuration loading and validation"""
    
    def __init__(self, api_key_override: Optional[str] = None):
        self.api_key_override = api_key_override
        if not api_key_override:
            self.config_path = self._find_config_file()
            self.config = self._load_config()
            self._validate_config()
        else:
            os.environ["FLUX_X_KEY"] = api_key_override.strip()

    def _find_config_file(self) -> str:
        """Find config.ini in multiple possible locations"""
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"),
            os.path.join(folder_paths.base_path, "custom_nodes", "flux_pro_integrative", "config.ini"),
            os.path.join(os.getcwd(), "config.ini")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If using API key override, config file is optional
        if self.api_key_override:
            return ""
        
        raise FluxAPIError(
            f"Config file not found and no API key provided. Please either:\n"
            f"1. Create config.ini in one of these locations:\n"
            f"{chr(10).join(possible_paths)}\n\n"
            f"2. Or enter your API key in the 'api_key' field\n\n"
            f"Config format:\n[API]\nX_KEY=your_api_key_here"
        )

    def _load_config(self) -> configparser.ConfigParser:
        """Load configuration from file"""
        if not self.config_path:  # No config file when using API key override
            return configparser.ConfigParser()
            
        config = configparser.ConfigParser()
        try:
            config.read(self.config_path)
            return config
        except Exception as e:
            raise FluxAPIError(f"Failed to read config file: {str(e)}")

    def _validate_config(self) -> None:
        """Validate configuration has required fields"""
        if self.api_key_override:
            return  # Skip validation when using API key override
            
        if not self.config.has_section('API'):
            raise FluxAPIError("Config file must contain [API] section")
        
        if not self.config.has_option('API', 'X_KEY'):
            raise FluxAPIError("Config file must contain X_KEY in [API] section")
        
        x_key = self.config['API']['X_KEY'].strip()
        if not x_key:
            raise FluxAPIError("X_KEY cannot be empty")
        
        os.environ["FLUX_X_KEY"] = x_key

    @property
    def api_key(self) -> str:
        """Get API key from environment"""
        return os.environ.get("FLUX_X_KEY", "")


class FluxAPIClient:
    """Handles all Flux API interactions"""
    
    BASE_URL = "https://api.bfl.ai/v1"
    TIMEOUT = 30
    MAX_RETRIES = 15
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"X-Key": api_key})

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling"""
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(method, url, timeout=self.TIMEOUT, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            raise FluxAPIError("Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            raise FluxAPIError("Connection error. Please check your internet connection.")
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            raise FluxAPIError(error_msg)
        except Exception as e:
            raise FluxAPIError(f"Unexpected error: {str(e)}")

    def generate_image(self, prompt: str, **params) -> str:
        """Generate image and return task ID"""
        if params.get('ultra_mode', True):
            endpoint = "flux-pro-1.1-ultra"
            payload = {
                "prompt": prompt,
                "aspect_ratio": params.get('aspect_ratio', '16:9'),
                "safety_tolerance": params.get('safety_tolerance', 6),
                "output_format": params.get('output_format', 'png'),
                "raw": params.get('raw', False)
            }
        else:
            endpoint = "flux-pro-1.1"
            width, height = self._get_dimensions_from_ratio(params.get('aspect_ratio', '16:9'))
            payload = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "safety_tolerance": params.get('safety_tolerance', 6),
                "output_format": params.get('output_format', 'png')
            }
        
        if params.get('seed', -1) != -1:
            payload["seed"] = params['seed']
        
        print(f"[Flux Pro] Sending generation request to {endpoint}")
        response = self._make_request("POST", endpoint, json=payload)
        
        result = response.json()
        task_id = result.get("id")
        if not task_id:
            raise FluxAPIError("No task ID received from server")
        
        print(f"[Flux Pro] Task created: {task_id}")
        return task_id

    def create_finetune(self, zip_path: str, **params) -> str:
        """Create finetune job"""
        if not os.path.exists(zip_path):
            raise FluxAPIError(f"ZIP file not found: {zip_path}")
        
        try:
            with open(zip_path, "rb") as f:
                encoded_zip = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            raise FluxAPIError(f"Failed to read ZIP file: {str(e)}")
        
        payload = {
            "finetune_comment": params.get('comment', 'ComfyUI Finetune'),
            "trigger_word": params.get('trigger_word', 'TOK'),
            "file_data": encoded_zip,
            "iterations": params.get('iterations', 300),
            "mode": params.get('mode', 'general'),
            "learning_rate": params.get('learning_rate', 0.00001),
            "captioning": params.get('captioning', True),
            "priority": params.get('priority', 'quality'),
            "finetune_type": params.get('finetune_type', 'full'),
            "lora_rank": params.get('lora_rank', 32)
        }
        
        print("[Flux Pro] Creating finetune job...")
        response = self._make_request("POST", "finetune", json=payload)
        
        result = response.json()
        finetune_id = result.get("finetune_id")
        if not finetune_id:
            raise FluxAPIError("No finetune ID received")
        
        print(f"[Flux Pro] Finetune created: {finetune_id}")
        return finetune_id

    def generate_with_finetune(self, finetune_id: str, prompt: str, **params) -> str:
        """Generate image using finetune"""
        ultra_mode = params.get('ultra_mode', True)
        endpoint = "flux-pro-1.1-ultra-finetuned" if ultra_mode else "flux-pro-finetuned"
        
        payload = {
            "finetune_id": finetune_id,
            "prompt": prompt,
            "finetune_strength": params.get('finetune_strength', 1.2)
        }
        
        # Add format-specific parameters
        if ultra_mode:
            payload.update({
                "aspect_ratio": params.get('aspect_ratio', '16:9'),
                "safety_tolerance": params.get('safety_tolerance', 6),
                "output_format": params.get('output_format', 'png'),
                "raw": params.get('raw', False)
            })
        else:
            width, height = self._get_dimensions_from_ratio(params.get('aspect_ratio', '16:9'))
            payload.update({
                "width": width,
                "height": height,
                "safety_tolerance": params.get('safety_tolerance', 6),
                "output_format": params.get('output_format', 'png')
            })
        
        if params.get('seed', -1) != -1:
            payload["seed"] = params['seed']
        
        print(f"[Flux Pro] Generating with finetune: {finetune_id}")
        response = self._make_request("POST", endpoint, json=payload)
        
        result = response.json()
        task_id = result.get("id")
        if not task_id:
            raise FluxAPIError("No task ID received for finetune inference")
        
        return task_id

    def get_result(self, task_id: str, output_format: str = 'png') -> torch.Tensor:
        """Poll for result and return image tensor"""
        for attempt in range(1, self.MAX_RETRIES + 1):
            wait_time = min(2 ** attempt + 5, 30)
            print(f"[Flux Pro] Attempt {attempt}/{self.MAX_RETRIES} - waiting {wait_time}s")
            time.sleep(wait_time)
            
            try:
                response = self._make_request("GET", f"get_result?id={task_id}")
                result = response.json()
                status = result.get("status")
                
                if status == TaskStatus.READY.value:
                    sample_url = result.get('result', {}).get('sample')
                    if not sample_url:
                        raise FluxAPIError("No sample URL in response")
                    
                    return self._download_image(sample_url, output_format)
                
                elif status == TaskStatus.ERROR.value:
                    error_msg = result.get('result', {}).get('error', 'Unknown error')
                    raise FluxAPIError(f"Generation failed: {error_msg}")
                
                elif status != TaskStatus.PENDING.value:
                    raise FluxAPIError(f"Unexpected status: {status}")
                
                print(f"[Flux Pro] Status: {status} - retrying...")
                
            except FluxAPIError:
                raise
            except Exception as e:
                if attempt == self.MAX_RETRIES:
                    raise FluxAPIError(f"Failed to get result after {self.MAX_RETRIES} attempts: {str(e)}")
                print(f"[Flux Pro] Error on attempt {attempt}: {str(e)}")
        
        raise FluxAPIError(f"Task {task_id} did not complete within {self.MAX_RETRIES} attempts")

    def _download_image(self, url: str, output_format: str) -> torch.Tensor:
        """Download and convert image to tensor"""
        try:
            response = requests.get(url, timeout=self.TIMEOUT)
            response.raise_for_status()
            
            img = Image.open(io.BytesIO(response.content))
            
            # Convert to specified format
            with io.BytesIO() as buffer:
                img.save(buffer, format=output_format.upper())
                buffer.seek(0)
                final_img = Image.open(buffer).convert('RGB')
                
                # Convert to tensor
                img_array = np.array(final_img).astype(np.float32) / 255.0
                return torch.from_numpy(img_array)[None,]
                
        except Exception as e:
            raise FluxAPIError(f"Failed to download image: {str(e)}")

    @staticmethod
    def _get_dimensions_from_ratio(aspect_ratio: str) -> Tuple[int, int]:
        """Get width and height from aspect ratio"""
        dimensions = {
            "1:1": (1024, 1024),
            "4:3": (1408, 1024),
            "3:4": (1024, 1408),
            "16:9": (1408, 800),
            "9:16": (800, 1408),
            "21:9": (1408, 608),
            "9:21": (608, 1408)
        }
        return dimensions.get(aspect_ratio, (1408, 800))


class FluxProIntegrative:
    """ComfyUI Node for Flux Pro API with integrated generation and finetuning"""
    
    def __init__(self):
        self.config_manager = None
        self.api_client = None
        self._initialize()

    def _initialize(self, api_key_override: Optional[str] = None):
        """Initialize configuration and API client"""
        try:
            self.config_manager = ConfigManager(api_key_override)
            self.api_client = FluxAPIClient(self.config_manager.api_key)
            print("[Flux Pro] Node initialized successfully")
        except Exception as e:
            print(f"[Flux Pro] Initialization failed: {str(e)}")
            # Don't raise here to allow ComfyUI to load the node

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["generate", "finetune", "inference"], {
                    "default": "generate",
                    "tooltip": "Mode: generate=create image, finetune=train model, inference=use trained model"
                }),
                "prompt": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "tooltip": "Text description of the image to generate"
                })
            },
            "optional": {
                # API Configuration
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "BFL API Key (optional - can use config.ini instead)"
                }),
                
                # Generation settings
                "ultra_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use Ultra mode for higher quality"
                }),
                "aspect_ratio": ([
                    "21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21"
                ], {"default": "16:9"}),
                "safety_tolerance": ("INT", {
                    "default": 6, 
                    "min": 0, 
                    "max": 6,
                    "tooltip": "Safety filter strength (0=strict, 6=permissive)"
                }),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "raw": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip safety checks (Ultra mode only)"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "tooltip": "Random seed (-1 for random)"
                }),
                
                # Finetuning settings
                "finetune_zip": ("STRING", {
                    "default": "",
                    "tooltip": "Path to ZIP file with training images"
                }),
                "finetune_comment": ("STRING", {
                    "default": "ComfyUI Finetune",
                    "tooltip": "Description for the finetune job"
                }),
                "finetune_id": ("STRING", {
                    "default": "",
                    "tooltip": "ID of existing finetune for inference"
                }),
                "trigger_word": ("STRING", {
                    "default": "TOK",
                    "tooltip": "Word to trigger the finetune style"
                }),
                "finetune_mode": (["character", "product", "style", "general"], {
                    "default": "general"
                }),
                "iterations": ("INT", {
                    "default": 300,
                    "min": 100,
                    "max": 2000,
                    "tooltip": "Training iterations"
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.00001,
                    "min": 0.00001,
                    "max": 0.0001,
                    "step": 0.00001
                }),
                "captioning": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-generate captions for training images"
                }),
                "priority": (["speed", "quality"], {"default": "quality"}),
                "finetune_type": (["full", "lora"], {"default": "full"}),
                "lora_rank": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 128
                }),
                "finetune_strength": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Strength of finetune effect"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "process"
    CATEGORY = "BFL/Flux Pro"

    def process(self, mode: str, prompt: str, api_key: str = "", **kwargs) -> Tuple[torch.Tensor, str]:
        """Main processing function"""
        try:
            # Use API key from input if provided, otherwise use config
            effective_api_key = api_key.strip() if api_key.strip() else None
            
            if not self.api_client or effective_api_key:
                self._initialize(effective_api_key)
                if not self.api_client:
                    return self._create_error_output("Node not properly initialized. Check config.ini or provide API key")

            if mode == "generate":
                return self._handle_generate(prompt, **kwargs)
            elif mode == "finetune":
                return self._handle_finetune(**kwargs)
            elif mode == "inference":
                return self._handle_inference(prompt, **kwargs)
            else:
                return self._create_error_output(f"Unknown mode: {mode}")

        except FluxAPIError as e:
            print(f"[Flux Pro] API Error: {str(e)}")
            return self._create_error_output(str(e))
        except Exception as e:
            print(f"[Flux Pro] Unexpected Error: {str(e)}")
            return self._create_error_output(f"Unexpected error: {str(e)}")

    def _handle_generate(self, prompt: str, **kwargs) -> Tuple[torch.Tensor, str]:
        """Handle image generation"""
        if not prompt.strip():
            return self._create_error_output("Prompt cannot be empty")

        print(f"[Flux Pro] Generating image with prompt: {prompt[:100]}...")
        
        task_id = self.api_client.generate_image(prompt, **kwargs)
        image_tensor = self.api_client.get_result(task_id, kwargs.get('output_format', 'png'))
        
        info = f"Generated successfully\nTask ID: {task_id}\nPrompt: {prompt[:100]}..."
        return (image_tensor, info)

    def _handle_finetune(self, **kwargs) -> Tuple[torch.Tensor, str]:
        """Handle finetune creation"""
        zip_path = kwargs.get('finetune_zip', '').strip()
        if not zip_path:
            return self._create_error_output("Finetune ZIP path is required")

        comment = kwargs.get('finetune_comment', 'ComfyUI Finetune')
        if not comment.strip():
            return self._create_error_output("Finetune comment is required")

        finetune_params = {
            'comment': comment,
            'trigger_word': kwargs.get('trigger_word', 'TOK'),
            'mode': kwargs.get('finetune_mode', 'general'),
            'iterations': kwargs.get('iterations', 300),
            'learning_rate': kwargs.get('learning_rate', 0.00001),
            'captioning': kwargs.get('captioning', True),
            'priority': kwargs.get('priority', 'quality'),
            'finetune_type': kwargs.get('finetune_type', 'full'),
            'lora_rank': kwargs.get('lora_rank', 32)
        }

        finetune_id = self.api_client.create_finetune(zip_path, **finetune_params)
        
        info = (f"Finetune created successfully!\n"
                f"ID: {finetune_id}\n"
                f"Comment: {comment}\n"
                f"Trigger word: {finetune_params['trigger_word']}")
        
        return (self._create_blank_image(), info)

    def _handle_inference(self, prompt: str, **kwargs) -> Tuple[torch.Tensor, str]:
        """Handle finetune inference"""
        finetune_id = kwargs.get('finetune_id', '').strip()
        if not finetune_id:
            return self._create_error_output("Finetune ID is required for inference")

        if not prompt.strip():
            return self._create_error_output("Prompt cannot be empty")

        print(f"[Flux Pro] Generating with finetune {finetune_id}")
        
        task_id = self.api_client.generate_with_finetune(finetune_id, prompt, **kwargs)
        image_tensor = self.api_client.get_result(task_id, kwargs.get('output_format', 'png'))
        
        info = (f"Generated with finetune successfully\n"
                f"Finetune ID: {finetune_id}\n"
                f"Task ID: {task_id}\n"
                f"Prompt: {prompt[:100]}...")
        
        return (image_tensor, info)

    def _create_blank_image(self) -> torch.Tensor:
        """Create a blank image tensor"""
        blank_img = Image.new('RGB', (512, 512), color='#333333')
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        return torch.from_numpy(img_array)[None,]

    def _create_error_output(self, error_msg: str) -> Tuple[torch.Tensor, str]:
        """Create error output with blank image and error message"""
        return (self._create_blank_image(), f"ERROR: {error_msg}")


# Node registration
NODE_CLASS_MAPPINGS = {
    "FluxProIntegrative": FluxProIntegrative
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxProIntegrative": "🎨 Flux Pro Integrative"
}

# Required for newer ComfyUI versions
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']