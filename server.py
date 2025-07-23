from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys
import asyncio

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARN,  # Change to INFO level to show more details
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
import uvicorn
# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:", 
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

# Create comprehensive logging setup for debugging
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

# Create per-conversation log directory with timestamp
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
conversation_log_dir = os.path.join(log_dir, f"conversation_{timestamp}")
os.makedirs(conversation_log_dir, exist_ok=True)
print(f"🔍 Debug logs for this session: {conversation_log_dir}")

# Common formatter with detailed timestamp
detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 1. Debug logger for function calling requests (reduced logging)
debug_logger = logging.getLogger("function_calling_debug")
debug_logger.setLevel(logging.DEBUG)  # Changed back to DEBUG for better debugging
debug_file_handler = logging.FileHandler(os.path.join(conversation_log_dir, "function_calling_debug.log"))
debug_file_handler.setLevel(logging.DEBUG)
debug_file_handler.setFormatter(detailed_formatter)
debug_logger.addHandler(debug_file_handler)

# 2. Request logger for all API requests
request_logger = logging.getLogger("request_logger")
request_logger.setLevel(logging.DEBUG)
request_file_handler = logging.FileHandler(os.path.join(conversation_log_dir, "requests.log"))
request_file_handler.setLevel(logging.DEBUG)
request_file_handler.setFormatter(simple_formatter)
request_logger.addHandler(request_file_handler)

# 3. Tool calls logger - dedicated for tool call processing
tool_logger = logging.getLogger("tool_calls")
tool_logger.setLevel(logging.DEBUG)
tool_file_handler = logging.FileHandler(os.path.join(conversation_log_dir, "tool_calls.log"))
tool_file_handler.setLevel(logging.DEBUG)
tool_file_handler.setFormatter(detailed_formatter)
tool_logger.addHandler(tool_file_handler)

# 4. Model mapping logger - for debugging model routing
model_logger = logging.getLogger("model_mapping")
model_logger.setLevel(logging.DEBUG)
model_file_handler = logging.FileHandler(os.path.join(conversation_log_dir, "model_mapping.log"))
model_file_handler.setLevel(logging.DEBUG)
model_file_handler.setFormatter(detailed_formatter)
model_logger.addHandler(model_file_handler)

# 5. Error logger - dedicated for errors and exceptions
error_logger = logging.getLogger("errors")
error_logger.setLevel(logging.ERROR)
error_file_handler = logging.FileHandler(os.path.join(conversation_log_dir, "errors.log"))
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(detailed_formatter)
error_logger.addHandler(error_file_handler)

def log_function_calling_request(request_data: dict, endpoint: str):
    """Log function calling request details to debug file - concise version"""
    try:
        tool_names = [tool.get("name", "unknown") for tool in request_data.get("tools", [])]
        tools_str = f"[{','.join(tool_names[:5])}{'+...' if len(tool_names) > 5 else ''}]({len(tool_names)})"
        
        debug_logger.debug(
            f"🔧 FUNCTION_CALL_REQ: {request_data.get('model')} | "
            f"stream={request_data.get('stream')} | "
            f"tools={tools_str} | "
            f"messages={len(request_data.get('messages', []))}"
        )
        
    except Exception as e:
        debug_logger.error(f"Error logging function calling request: {str(e)}")

def log_function_calling_response(response_data: dict, request_id: str = None):
    """Log function calling response details to debug file"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id or str(uuid.uuid4()),
            "response_type": "function_calling_response",
            "raw_response": response_data
        }
        
        debug_logger.debug(f"FUNCTION_CALLING_RESPONSE: {json.dumps(log_entry, indent=2)}")
        
    except Exception as e:
        debug_logger.error(f"Error logging function calling response: {str(e)}")

def log_function_calling_error(error: Exception, request_data: dict = None, context: str = ""):
    """Log function calling errors to dedicated error log"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "request_data": request_data,
            "traceback": traceback.format_exc()
        }
        
        # Log to both debug and error loggers
        debug_logger.error(f"FUNCTION_CALLING_ERROR: {json.dumps(log_entry, indent=2)}")
        error_logger.error(f"FUNCTION_CALLING_ERROR: {json.dumps(log_entry, indent=2)}")
        
    except Exception as e:
        debug_logger.error(f"Error logging function calling error: {str(e)}")
        error_logger.error(f"Error logging function calling error: {str(e)}")

def log_tool_call_processing(tool_calls, model, is_tool_capable, processing_path):
    """Log detailed tool call processing information"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool_calls_present": bool(tool_calls),
            "tool_calls_count": len(tool_calls) if tool_calls else 0,
            "model": model,
            "is_tool_capable_model": is_tool_capable,
            "processing_path": processing_path,
            "tool_calls_details": [
                {
                    "id": getattr(tc, 'id', 'unknown') if hasattr(tc, 'id') else tc.get('id', 'unknown'),
                    "name": getattr(getattr(tc, 'function', None), 'name', 'unknown') if hasattr(tc, 'function') else tc.get('function', {}).get('name', 'unknown'),
                    "type": getattr(tc, 'type', 'unknown') if hasattr(tc, 'type') else tc.get('type', 'unknown')
                } for tc in (tool_calls if tool_calls else [])
            ]
        }
        
        tool_logger.debug(f"TOOL_CALL_PROCESSING: {json.dumps(log_entry, indent=2)}")
        
    except Exception as e:
        tool_logger.error(f"Error logging tool call processing: {str(e)}")

def log_model_mapping(original_model, mapped_model, provider, context=""):
    """Log model mapping decisions"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "original_model": original_model,
            "mapped_model": mapped_model,
            "provider": provider,
            "context": context
        }
        
        model_logger.debug(f"MODEL_MAPPING: {json.dumps(log_entry, indent=2)}")
        
    except Exception as e:
        model_logger.error(f"Error logging model mapping: {str(e)}")

def truncate_large_object(obj, max_chars=60):
    """Truncate large objects to show first and last 30 characters"""
    try:
        if obj is None:
            return "None"
        
        obj_str = str(obj)
        if len(obj_str) <= max_chars:
            return obj_str
        
        half = max_chars // 2
        return f"{obj_str[:half]}...{obj_str[-half:]}"
    except:
        return str(obj)[:max_chars] + "..." if len(str(obj)) > max_chars else str(obj)

def log_end_to_end_flow(request_id, step, data, request_model=None):
    """Log unified end-to-end flow for a single request"""
    try:
        # Truncate large objects
        clean_data = {}
        for key, value in (data if isinstance(data, dict) else {"data": data}).items():
            if key in ["tools", "tool_calls", "function", "raw_request", "raw_response"]:
                clean_data[key] = truncate_large_object(value)
            elif isinstance(value, list) and len(value) > 0:
                # For arrays, show count and truncate first item
                if len(value) > 1:
                    clean_data[key] = f"[{len(value)} items] {truncate_large_object(value[0])}"
                else:
                    clean_data[key] = truncate_large_object(value[0])
            else:
                clean_data[key] = value
        
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "model": request_model,
            "data": clean_data
        }
        
        # Log to dedicated end-to-end file
        e2e_logger = logging.getLogger("end_to_end")
        if not e2e_logger.handlers:
            e2e_logger.setLevel(logging.DEBUG)
            e2e_file_handler = logging.FileHandler(os.path.join(conversation_log_dir, "end_to_end.log"))
            e2e_file_handler.setLevel(logging.DEBUG)
            e2e_file_handler.setFormatter(detailed_formatter)
            e2e_logger.addHandler(e2e_file_handler)
        
        e2e_logger.debug(f"E2E_FLOW: {json.dumps(log_entry, indent=2)}")
        
    except Exception as e:
        debug_logger.error(f"Error logging end-to-end flow: {str(e)}")

app = FastAPI()

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Get preferred provider (default to openai)
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()

# Get content flattening configuration (default to false for backward compatibility)
CONTENT_FLATTENING = os.environ.get("CONTENT_FLATTENING", "false").lower() == "true"

# Get model mapping configuration from environment
# Default to latest OpenAI models if not set
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")

# List of OpenAI models
OPENAI_MODELS = [
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",  # Added default big model
    "gpt-4.1-mini" # Added default small model
]

# List of Gemini models
GEMINI_MODELS = [
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.0-flash"
]

# Helper function to clean schema for Gemini
def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by Gemini tool parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in Gemini schema.")
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(schema.items()): # Use list() to allow modification during iteration
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_gemini_schema(item) for item in schema]
    return schema

# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"] 
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_field(cls, v, info): # Renamed to avoid conflict
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 MODEL VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False
        # Map Haiku to SMALL_MODEL based on provider preference
        if 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
                log_model_mapping(original_model, new_model, "gemini", "haiku_to_small_model_gemini")
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True
                log_model_mapping(original_model, new_model, "openai", "haiku_to_small_model_openai")

        # Map Sonnet to BIG_MODEL based on provider preference
        elif 'sonnet' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
                log_model_mapping(original_model, new_model, "gemini", "sonnet_to_big_model_gemini")
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True
                log_model_mapping(original_model, new_model, "openai", "sonnet_to_big_model_openai")

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # Technically mapped to add prefix
                log_model_mapping(original_model, new_model, "gemini", "prefix_addition_gemini")
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
                log_model_mapping(original_model, new_model, "openai", "prefix_addition_openai")
        # --- Mapping Logic --- END ---

        # Log final mapping result
        if original_model != new_model:
            log_model_mapping(original_model, new_model, PREFERRED_PROVIDER, "final_validation_mapping")
        
        if mapped:
            logger.debug(f"📌 MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             # If no mapping occurred and no prefix exists, log warning or decide default
             if not v.startswith(('openai/', 'gemini/', 'anthropic/')):
                 logger.warning(f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is.")
             new_model = v # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_token_count(cls, v, info): # Renamed to avoid conflict
        # Use the same logic as MessagesRequest validator
        # NOTE: Pydantic validators might not share state easily if not class methods
        # Re-implementing the logic here for clarity, could be refactored
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 TOKEN COUNT VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False
        # Map Haiku to SMALL_MODEL based on provider preference
        if 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        # Map Sonnet to BIG_MODEL based on provider preference
        elif 'sonnet' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"📌 TOKEN COUNT MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             if not v.startswith(('openai/', 'gemini/', 'anthropic/')):
                 logger.warning(f"⚠️ No prefix or mapping rule for token count model: '{original_model}'. Using as is.")
             new_model = v # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path
    
    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")
    
    # Process the request and get the response
    response = await call_next(request)
    
    return response

# Not using validation function as we're using the environment API key

def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"
        
    if isinstance(content, str):
        return content
        
    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()
        
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)
            
    # Fallback for any other type
    try:
        return str(content)
    except:
        return "Unparseable content"

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    # LiteLLM already handles Anthropic models when using the format model="anthropic/claude-3-opus-20240229"
    # So we just need to convert our Pydantic model to a dict in the expected format
    
    messages = []
    
    # Add system message if present
    if anthropic_request.system:
        # Handle different formats of system messages
        if isinstance(anthropic_request.system, str):
            # Simple string format
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # List of content blocks
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"
            
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})
    
    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # Special handling for tool_result in user messages
            # OpenAI/LiteLLM format expects the assistant to call the tool, 
            # and the user's next message to include the result as plain text
            if msg.role == "user" and any(block.type == "tool_result" for block in content if hasattr(block, "type")):
                # For user messages with tool_result, split into separate messages
                text_content = ""
                
                # Extract all text parts and concatenate them
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content += block.text + "\n"
                        elif block.type == "tool_result":
                            # Add tool result as a message by itself - simulate the normal flow
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            
                            # Handle different formats of tool result content
                            result_content = ""
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    result_content = block.content
                                elif isinstance(block.content, list):
                                    # If content is a list of blocks, extract text from each
                                    for content_block in block.content:
                                        if hasattr(content_block, "type") and content_block.type == "text":
                                            result_content += content_block.text + "\n"
                                        elif isinstance(content_block, dict) and content_block.get("type") == "text":
                                            result_content += content_block.get("text", "") + "\n"
                                        elif isinstance(content_block, dict):
                                            # Handle any dict by trying to extract text or convert to JSON
                                            if "text" in content_block:
                                                result_content += content_block.get("text", "") + "\n"
                                            else:
                                                try:
                                                    result_content += json.dumps(content_block) + "\n"
                                                except:
                                                    result_content += str(content_block) + "\n"
                                elif isinstance(block.content, dict):
                                    # Handle dictionary content
                                    if block.content.get("type") == "text":
                                        result_content = block.content.get("text", "")
                                    else:
                                        try:
                                            result_content = json.dumps(block.content)
                                        except:
                                            result_content = str(block.content)
                                else:
                                    # Handle any other type by converting to string
                                    try:
                                        result_content = str(block.content)
                                    except:
                                        result_content = "Unparseable content"
                            
                            # In OpenAI format, tool results come from the user (rather than being content blocks)
                            text_content += f"Tool result for {tool_id}:\n{result_content}\n"
                
                # Add as a single user message with all the content
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                # Regular handling for other message types
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append({"type": "text", "text": block.text})
                        elif block.type == "image":
                            processed_content.append({"type": "image", "source": block.source})
                        elif block.type == "tool_use":
                            # Handle tool use blocks if needed
                            processed_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })
                        elif block.type == "tool_result":
                            # Handle different formats of tool result content
                            processed_content_block = {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            }
                            
                            # Process the content field properly
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    # If it's a simple string, create a text block for it
                                    processed_content_block["content"] = [{"type": "text", "text": block.content}]
                                elif isinstance(block.content, list):
                                    # If it's already a list of blocks, keep it
                                    processed_content_block["content"] = block.content
                                else:
                                    # Default fallback
                                    processed_content_block["content"] = [{"type": "text", "text": str(block.content)}]
                            else:
                                # Default empty content
                                processed_content_block["content"] = [{"type": "text", "text": ""}]
                                
                            processed_content.append(processed_content_block)
                
                messages.append({"role": msg.role, "content": processed_content})
    
    # Cap max_tokens for OpenAI models to their limit of 16384
    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith("openai/") or anthropic_request.model.startswith("gemini/"):
        max_tokens = min(max_tokens, 16384)
        logger.debug(f"Capping max_tokens to 16384 for OpenAI/Gemini model (original value: {anthropic_request.max_tokens})")
    
    # Create LiteLLM request dict
    litellm_request = {
        "model": anthropic_request.model,  # t understands "anthropic/claude-x" format
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }
    
    # Add optional parameters if present
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    
    # Convert tools to OpenAI format
    if anthropic_request.tools:
        openai_tools = []
        is_gemini_model = anthropic_request.model.startswith("gemini/")
        
        debug_logger.debug(f"Converting {len(anthropic_request.tools)} tools for model: {anthropic_request.model}")
        conversion_errors = 0

        for i, tool in enumerate(anthropic_request.tools):
            # Convert to dict if it's a pydantic model
            if hasattr(tool, 'dict'):
                tool_dict = tool.dict()
            else:
                # Ensure tool_dict is a dictionary, handle potential errors if 'tool' isn't dict-like
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                     logger.error(f"Could not convert tool {i} to dict: {tool}")
                     debug_logger.error(f"TOOL_CONVERSION_ERROR: Tool index {i}, type: {type(tool)}, value: {tool}")
                     conversion_errors += 1
                     continue # Skip this tool if conversion fails

            # Clean the schema if targeting a Gemini model
            input_schema = tool_dict.get("input_schema", {})
            if is_gemini_model:
                 logger.debug(f"Cleaning schema for Gemini tool: {tool_dict.get('name')}")
                 debug_logger.debug(f"GEMINI_SCHEMA_CLEANING: {tool_dict.get('name')} - Original: {input_schema}")
                 input_schema = clean_gemini_schema(input_schema)
                 debug_logger.debug(f"GEMINI_SCHEMA_CLEANED: {tool_dict.get('name')} - Cleaned: {input_schema}")

            # Create OpenAI-compatible function tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema # Use potentially cleaned schema
                }
            }
            openai_tools.append(openai_tool)
            # Compact tool logging - just name and parameter count
            param_count = len(openai_tool.get('function', {}).get('parameters', {}).get('properties', {}))
            debug_logger.debug(f"🔧 CONVERTED_TOOL: {openai_tool.get('function', {}).get('name', 'unknown')} | params={param_count}")

        # Validate tool conversion results
        if conversion_errors > 0:
            logger.warning(f"⚠️  Tool conversion had {conversion_errors} errors out of {len(anthropic_request.tools)} tools")
            debug_logger.warning(f"TOOL_CONVERSION_SUMMARY: Errors: {conversion_errors}, Original: {len(anthropic_request.tools)}, Converted: {len(openai_tools)}")
        
        if len(openai_tools) == 0:
            logger.error(f"🚨 CRITICAL: All {len(anthropic_request.tools)} tools failed to convert! This will cause tool stripping.")
            debug_logger.error(f"TOOL_CONVERSION_TOTAL_FAILURE: Original tools count: {len(anthropic_request.tools)}, Final tools count: 0")
        elif len(openai_tools) != len(anthropic_request.tools):
            logger.warning(f"⚠️  Tool count mismatch: {len(anthropic_request.tools)} → {len(openai_tools)}")
            debug_logger.warning(f"TOOL_COUNT_MISMATCH: Original: {len(anthropic_request.tools)}, Final: {len(openai_tools)}")
        
        litellm_request["tools"] = openai_tools
        # Compact final tools summary
        tool_names = [t.get('function', {}).get('name', 'unknown') for t in openai_tools]
        debug_logger.debug(f"🔧 FINAL_TOOLS: [{','.join(tool_names)}] ({len(openai_tools)} tools)")
    
    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, 'dict'):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice
            
        # Handle Anthropic's tool_choice format
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]}
            }
        else:
            # Default to auto if we can't determine
            litellm_request["tool_choice"] = "auto"
    
    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any], 
                                 original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""
    
    # Enhanced response extraction with better error handling
    try:
        # Get the clean model name to check capabilities
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Debug logging for model detection
        debug_logger.debug(f"RESPONSE_CONVERSION: original_model={original_request.model}, clean_model={clean_model}")
        
        # Check if this model supports tool_use content blocks
        # Include Claude models and other tool-capable models like Kimi-K2
        is_tool_capable_model = (clean_model.startswith("claude-") or 
                               clean_model.lower().startswith("kimi") or clean_model.lower().startswith("qwen"))
        debug_logger.debug(f"MODEL_DETECTION: is_tool_capable_model={is_tool_capable_model} (clean_model={clean_model})")
        
        # Debug: log the type and structure of the response
        debug_logger.debug(f"LITELLM_RESPONSE_TYPE: {type(litellm_response)}")
        
        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            # Extract data from ModelResponse object directly
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = message.content if message and hasattr(message, 'content') else ""
            tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
            finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            
            # Debug logging for tool calls detection
            tool_calls_str = str(tool_calls) if tool_calls else "None"
            debug_logger.debug(f"TOOL_CALLS_DETECTION: content_text={content_text}, tool_calls={tool_calls_str}, finish_reason={finish_reason}")
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        else:
            # For backward compatibility - handle dict responses
            # If response is a dict, use it, otherwise try to convert to dict
            try:
                response_dict = litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
            except AttributeError:
                # If .dict() fails, try to use model_dump or __dict__ 
                try:
                    response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                except AttributeError:
                    # Fallback - manually extract attributes
                    response_dict = {
                        "id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, 'choices', [{}]),
                        "usage": getattr(litellm_response, 'usage', {})
                    }
                    
            # Extract the content from the response dict
            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")
        
        # Create content list for Anthropic format
        content = []
        
        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})
        
        # Add tool calls if present (tool_use in Anthropic format) - for tool-capable models
        debug_logger.debug(f"TOOL_CALLS_CONDITION_CHECK: tool_calls={bool(tool_calls)}, is_tool_capable_model={is_tool_capable_model}, tool_calls_value={tool_calls}")
        
        if tool_calls and is_tool_capable_model:
            log_tool_call_processing(tool_calls, clean_model, is_tool_capable_model, "ANTHROPIC_TOOL_USE_CONVERSION")
            debug_logger.debug(f"PROCESSING_TOOL_CALLS_CLAUDE_PATH: {tool_calls}")
            logger.debug(f"Processing tool calls: {tool_calls}")
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")
                
                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    openai_tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    openai_name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    openai_tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    openai_name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # Convert OpenAI-style tool ID to Anthropic-style (preserve original ID structure)
                if openai_tool_id.startswith("functions."):
                    tool_id = f"toolu_{uuid.uuid4().hex[:24]}"  # Generate Anthropic-style for function calls
                else:
                    tool_id = openai_tool_id  # Preserve original if it's already in good format
                
                # Preserve original tool names - minimal conversion only
                name = openai_name
                
                # Log the conversion for debugging
                debug_logger.debug(f"TOOL_ID_CONVERSION: OpenAI_ID='{openai_tool_id}' -> Anthropic_ID='{tool_id}', OpenAI_Name='{openai_name}' -> Anthropic_Name='{name}'")
                
                # Log to end-to-end flow
                try:
                    # Extract request_id from the calling context if available
                    import inspect
                    frame = inspect.currentframe()
                    while frame:
                        if 'request_id' in frame.f_locals:
                            req_id = frame.f_locals['request_id']
                            log_end_to_end_flow(req_id, "5_TOOL_ID_NAME_CONVERSION", {
                                "openai_tool_id": openai_tool_id,
                                "anthropic_tool_id": tool_id,
                                "openai_name": openai_name,
                                "anthropic_name": name,
                                "arguments": truncate_large_object(arguments)
                            }, original_request.model)
                            break
                        frame = frame.f_back
                except:
                    pass  # Don't let logging errors break the main flow
                
                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments as JSON: {arguments}")
                        arguments = {"raw": arguments}
                
                logger.debug(f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}")
                
                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": arguments
                })
        elif tool_calls and not is_tool_capable_model:
            # For non-Claude models, convert tool calls to text format
            log_tool_call_processing(tool_calls, clean_model, is_tool_capable_model, "TEXT_FORMAT_CONVERSION")
            debug_logger.debug(f"PROCESSING_TOOL_CALLS_NON_CLAUDE_PATH: clean_model={clean_model}, tool_calls={tool_calls}")
            logger.debug(f"Converting tool calls to text for non-Claude model: {clean_model}")
        else:
            if tool_calls:
                log_tool_call_processing(tool_calls, clean_model, is_tool_capable_model, "SKIPPED_NO_CONDITION_MATCH")
            debug_logger.debug(f"TOOL_CALLS_SKIPPED: tool_calls={bool(tool_calls)}, is_tool_capable_model={is_tool_capable_model}, reason=no_matching_condition")
            
            # We'll append tool info to the text content
            tool_text = "\n\nTool usage:\n"
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        args_dict = json.loads(arguments)
                        arguments_str = json.dumps(args_dict, indent=2)
                    except json.JSONDecodeError:
                        arguments_str = arguments
                else:
                    arguments_str = json.dumps(arguments, indent=2)
                
                tool_text += f"Tool: {name}\nArguments: {arguments_str}\n\n"
            
            # Add or append tool text to content
            if content and content[0]["type"] == "text":
                content[0]["text"] += tool_text
            else:
                content.append({"type": "text", "text": tool_text})
        
        # Get usage information - extract values safely from object or dict
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)
        
        # Map OpenAI finish_reason to Anthropic stop_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"  # Default
        
        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})
        
        # Create Anthropic-style response
        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )
        
        return anthropic_response
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # In case of any error, create a fallback response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {str(e)}. Please check server logs."}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        # Request-Response Correlation Tracking
        correlation_id = f"corr_{uuid.uuid4().hex[:12]}"
        response_start_time = time.time()
        debug_logger.debug(f"🔄 RESPONSE_START: Correlation ID: {correlation_id}, Model: {original_request.model}, Has_Tools: {bool(getattr(original_request, 'tools', None))}")
        
        chunk_count = 0
        tool_calls_received = 0
        text_content_received = False
        
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"  # Format similar to Anthropic's IDs
        
        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        
        # Content block index for the first text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        
        # Send a ping to keep the connection alive (Anthropic does this)
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        
        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""  # Track accumulated text content
        text_sent = False  # Track if we've sent any text content
        text_block_closed = False  # Track if text block is closed
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0
        
        # Process each chunk
        async for chunk in response_generator:
            try:
                chunk_count += 1
                
                # Enhanced chunk tracking with correlation
                debug_logger.debug(f"🔍 RAW CHUNK #{chunk_count} [Corr: {correlation_id}]: {chunk}")
                debug_logger.debug(f"🔍 RAW CHUNK TYPE: {type(chunk)}")
                if hasattr(chunk, '__dict__'):
                    debug_logger.debug(f"🔍 RAW CHUNK ATTRS: {vars(chunk)}")
                else:
                    debug_logger.debug(f"🔍 RAW CHUNK (dict): {dict(chunk) if hasattr(chunk, 'keys') else 'not dict-like'}")

                
                # Check if this is the end of the response with usage data
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens
                
                # Handle text content
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # Get the delta from the choice
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    else:
                        # If no delta, try to get message
                        delta = getattr(choice, 'message', {})
                    
                    # Check for finish_reason to know when we're done
                    finish_reason = getattr(choice, 'finish_reason', None)
                    
                    # Process text content
                    delta_content = None
                    
                    # Handle different formats of delta content
                    if hasattr(delta, 'content'):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and 'content' in delta:
                        delta_content = delta['content']
                    
                    # Accumulate text content
                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content
                        text_content_received = True
                        
                        # Enhanced text content tracking with correlation
                        debug_logger.debug(f"📝 TEXT_CONTENT [Corr: {correlation_id}]: Delta: '{delta_content}', Total length: {len(accumulated_text)}")
                        debug_logger.debug(f"🔍 TEXT ACCUMULATED: '{accumulated_text}'")
                        
                        # Always emit text deltas if no tool calls started
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"
                    
                    # Process tool calls
                    delta_tool_calls = None
                    
                    # Handle different formats of tool calls
                    if hasattr(delta, 'tool_calls'):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and 'tool_calls' in delta:
                        delta_tool_calls = delta['tool_calls']
                    
                    # Process tool calls if any
                    if delta_tool_calls:
                        tool_calls_received += len(delta_tool_calls)
                        debug_logger.debug(f"🔧 TOOL_CALLS_DETECTED [Corr: {correlation_id}]: Count: {len(delta_tool_calls)}, Total so far: {tool_calls_received}")
                        # First tool call we've seen - need to handle text properly
                        if tool_index is None:
                            # If we've been streaming text, close that text block
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # If we've accumulated text but not sent it, we need to emit it now
                            # This handles the case where the first delta has both text and a tool call
                            elif accumulated_text and not text_sent and not text_block_closed:
                                # Send the accumulated text
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                # Close the text block
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # Close text block even if we haven't sent anything - models sometimes emit empty text blocks
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                
                        # Convert to list if it's not already
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]
                        
                        for tool_call in delta_tool_calls:
                            # CRITICAL DEBUG: Log what we're receiving from LiteLLM
                            debug_logger.debug(f"🔍 TOOL CALL DEBUG: Received tool_call: {tool_call}")
                            debug_logger.debug(f"🔍 TOOL CALL TYPE: {type(tool_call)}")
                            if hasattr(tool_call, '__dict__'):
                                debug_logger.debug(f"🔍 TOOL CALL ATTRS: {vars(tool_call)}")
                            
                            # Get the index of this tool call (for multiple tools)
                            current_index = None
                            if isinstance(tool_call, dict) and 'index' in tool_call:
                                current_index = tool_call['index']
                            elif hasattr(tool_call, 'index'):
                                current_index = tool_call.index
                            else:
                                current_index = 0
                            
                            # Check if this is a new tool or a continuation
                            if tool_index is None or current_index != tool_index:
                                # New tool call - create a new tool_use block
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index
                                
                                # Extract function info
                                if isinstance(tool_call, dict):
                                    function = tool_call.get('function', {})
                                    openai_name = function.get('name', '') if isinstance(function, dict) else ""
                                    openai_tool_id = tool_call.get('id', f"tool_{uuid.uuid4()}")
                                else:
                                    function = getattr(tool_call, 'function', None)
                                    openai_name = getattr(function, 'name', '') if function else ''
                                    openai_tool_id = getattr(tool_call, 'id', f"tool_{uuid.uuid4()}")
                                
                                # Convert OpenAI-style tool ID to Anthropic-style (streaming)
                                if openai_tool_id.startswith("functions."):
                                    tool_id = f"toolu_{uuid.uuid4().hex[:24]}"  # Generate Anthropic-style for function calls
                                else:
                                    tool_id = openai_tool_id  # Preserve original if it's already in good format
                                
                                # Preserve original tool names (streaming) - minimal conversion
                                name = openai_name
                                
                                # Log conversion for streaming
                                debug_logger.debug(f"STREAMING_TOOL_CONVERSION: OpenAI_ID='{openai_tool_id}' -> Anthropic_ID='{tool_id}', OpenAI_Name='{openai_name}' -> Anthropic_Name='{name}'")
                                
                                # Start a new tool_use block
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""
                            
                            # Extract function arguments
                            arguments = None
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function = tool_call.get('function', {})
                                arguments = function.get('arguments', '') if isinstance(function, dict) else ''
                            elif hasattr(tool_call, 'function'):
                                function = getattr(tool_call, 'function', None)
                                arguments = getattr(function, 'arguments', '') if function else ''
                            
                            # If we have arguments, send them as a delta
                            if arguments:
                                # Try to detect if arguments are valid JSON or just a fragment
                                try:
                                    # If it's already a dict, use it
                                    if isinstance(arguments, dict):
                                        args_json = json.dumps(arguments)
                                    else:
                                        # Otherwise, try to parse it
                                        json.loads(arguments)
                                        args_json = arguments
                                except (json.JSONDecodeError, TypeError):
                                    # If it's a fragment, treat it as a string
                                    args_json = arguments
                                
                                # Add to accumulated tool content
                                tool_content += args_json if isinstance(args_json, str) else ""
                                
                                # Send the update
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"
                    
                    # Process finish_reason - end the streaming response
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True
                        
                        # Close any open tool call blocks
                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
                        
                        # If we accumulated text but never sent or closed text block, do it now
                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                # Send the accumulated text
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            # Close the text block
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        
                        # Map OpenAI finish_reason to Anthropic stop_reason
                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"
                        
                        # Send message_delta with stop reason and usage
                        usage = {"output_tokens": output_tokens}
                        
                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"
                        
                        # Send message_stop event
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                        
                        # Send final [DONE] marker to match Anthropic's behavior
                        yield "data: [DONE]\n\n"
                        
                        # Final response correlation tracking
                        response_duration = time.time() - response_start_time
                        debug_logger.debug(f"✅ RESPONSE_COMPLETE [Corr: {correlation_id}]: Duration: {response_duration:.2f}s, Chunks: {chunk_count}, Tools: {tool_calls_received}, Text: {text_content_received}")
                        return
            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Error processing chunk: {str(e)}")
                continue
        
        # If we didn't get a finish reason, close any open blocks
        if not has_sent_stop_reason:
            # Close any open tool call blocks
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
            
            # Close the text content block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            
            # Send final message_delta with usage
            usage = {"output_tokens": output_tokens}
            
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"
            
            # Send message_stop event
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            
            # Send final [DONE] marker to match Anthropic's behavior
            yield "data: [DONE]\n\n"
            
            # Final response correlation tracking (fallback path)
            response_duration = time.time() - response_start_time
            debug_logger.debug(f"✅ RESPONSE_COMPLETE_FALLBACK [Corr: {correlation_id}]: Duration: {response_duration:.2f}s, Chunks: {chunk_count}, Tools: {tool_calls_received}, Text: {text_content_received}")
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # Send error message_delta
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        
        # Send message_stop event
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        
        # Send final [DONE] marker
        yield "data: [DONE]\n\n"

@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    # Generate unique request ID for end-to-end tracking
    request_id = str(uuid.uuid4())
    
    try:
        # print the body here
        body = await raw_request.body()
    
        # Parse the raw body as JSON since it's bytes
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        
        # Log incoming request
        log_end_to_end_flow(request_id, "1_INCOMING_REQUEST", {
            "original_model": original_model,
            "has_tools": bool(body_json.get("tools")),
            "num_tools": len(body_json.get("tools", [])),
            "stream": body_json.get("stream", False),
            "endpoint": "/v1/messages"
        }, original_model)
        
        # Log request details if it has tools (function calling)
        if body_json.get("tools"):
            log_function_calling_request(body_json, "/v1/messages")
        
        # Log all requests to requests.log
        # Concise request summary for LLM analysis
        tools_summary = ""
        if body_json.get('tools'):
            tool_names = [tool.get('name', 'unknown') for tool in body_json['tools'][:3]]  # First 3 only
            tools_summary = f"tools=[{','.join(tool_names)}{'+' if len(body_json['tools']) > 3 else ''}({len(body_json['tools'])})]"
        else:
            tools_summary = "tools=none"
        
        request_logger.debug(f"📥 REQUEST /v1/messages: model={body_json.get('model')}, stream={body_json.get('stream')}, {tools_summary}, messages={len(body_json.get('messages', []))}")
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Clear execution flow marker
        logger.debug(f"🔄 PROCESSING: {request.model} | stream={request.stream} | tools={len(request.tools) if request.tools else 0}")
        
        # Log model mapping
        log_end_to_end_flow(request_id, "2_MODEL_MAPPING", {
            "original_model": original_model,
            "mapped_model": request.model,
            "mapping_occurred": original_model != request.model
        }, request.model)
        
        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)
        
        # Log tool conversion
        if request.tools:
            log_end_to_end_flow(request_id, "3_TOOL_CONVERSION", {
                "anthropic_tools": request.tools,
                "litellm_tools": litellm_request.get("tools"),
                "num_tools": len(request.tools)
            }, request.model)
            debug_logger.debug(f"LITELLM_REQUEST_AFTER_CONVERSION: tools key exists: {'tools' in litellm_request}, tools type: {type(litellm_request.get('tools'))}")
        
        # Determine which API key to use based on the model
        if request.model.startswith("openai/"):
            litellm_request["api_key"] = OPENAI_API_KEY
            if OPENAI_API_BASE:
                litellm_request["api_base"] = OPENAI_API_BASE
                logger.debug(f"Using OpenAI API base: {OPENAI_API_BASE} for model: {request.model}")
            logger.debug(f"Using OpenAI API key for model: {request.model}")
        elif request.model.startswith("gemini/"):
            litellm_request["api_key"] = GEMINI_API_KEY
            logger.debug(f"Using Gemini API key for model: {request.model}")
        else:
            litellm_request["api_key"] = ANTHROPIC_API_KEY
            logger.debug(f"Using Anthropic API key for model: {request.model}")
        
        # Convert complex content blocks to simple strings for ik_llama.cpp compatibility
        if CONTENT_FLATTENING and "messages" in litellm_request:
            debug_logger.debug(f"🔧 CONTENT_FLATTENING_START: {litellm_request['model']} | tools={len(request.tools) if request.tools else 0}")
            
            # Convert complex content blocks to simple strings for ik_llama.cpp compatibility
            for i, message in enumerate(litellm_request["messages"]):
                if isinstance(message.get("content"), list):
                    # Convert content block array to simple string
                    text_parts = []
                    for content_block in message["content"]:
                        if isinstance(content_block, dict) and content_block.get("type") == "text":
                            text_parts.append(content_block.get("text", ""))
                        elif isinstance(content_block, dict) and content_block.get("type") == "tool_use":
                            # Skip tool_use blocks as they're handled separately
                            continue
                        elif isinstance(content_block, str):
                            text_parts.append(content_block)
                    litellm_request["messages"][i]["content"] = " ".join(text_parts)
            
            debug_logger.debug(f"🔧 CONTENT_FLATTENING_COMPLETE: {litellm_request['model']} | converted {len(litellm_request['messages'])} messages")
        
        # For OpenAI models - modify request format to work with limitations
        # Skip if content flattening is already enabled
        elif "openai" in litellm_request["model"] and not CONTENT_FLATTENING and "messages" in litellm_request:
            debug_logger.debug(f"🔧 OPENAI_PROCESSING_START: {litellm_request['model']} | tools={len(request.tools) if request.tools else 0}")
            logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")
            
            # For OpenAI models, we need to convert content blocks to simple strings
            # and handle other requirements
            for i, msg in enumerate(litellm_request["messages"]):
                # Special case - handle message content directly when it's a list of tool_result
                # This is a specific case we're seeing in the error
                if "content" in msg and isinstance(msg["content"], list):
                    is_only_tool_result = True
                    for block in msg["content"]:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            is_only_tool_result = False
                            break
                    
                    if is_only_tool_result and len(msg["content"]) > 0:
                        logger.warning(f"Found message with only tool_result content - special handling required")
                        # Extract the content from all tool_result blocks
                        all_text = ""
                        for block in msg["content"]:
                            all_text += "Tool Result:\n"
                            result_content = block.get("content", [])
                            
                            # Handle different formats of content
                            if isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        all_text += item.get("text", "") + "\n"
                                    elif isinstance(item, dict):
                                        # Fall back to string representation of any dict
                                        try:
                                            item_text = item.get("text", json.dumps(item))
                                            all_text += item_text + "\n"
                                        except:
                                            all_text += str(item) + "\n"
                            elif isinstance(result_content, str):
                                all_text += result_content + "\n"
                            else:
                                try:
                                    all_text += json.dumps(result_content) + "\n"
                                except:
                                    all_text += str(result_content) + "\n"
                        
                        # Replace the list with extracted text
                        litellm_request["messages"][i]["content"] = all_text.strip() or "..."
                        logger.warning(f"Converted tool_result to plain text: {all_text.strip()[:200]}...")
                        continue  # Skip normal processing for this message
                
                # 1. Handle content field - normal case
                if "content" in msg:
                    # Check if content is a list (content blocks)
                    if isinstance(msg["content"], list):
                        # Convert complex content blocks to simple string
                        text_content = ""
                        for block in msg["content"]:
                            if isinstance(block, dict):
                                # Handle different content block types
                                if block.get("type") == "text":
                                    text_content += block.get("text", "") + "\n"
                                
                                # Handle tool_result content blocks - extract nested text
                                elif block.get("type") == "tool_result":
                                    tool_id = block.get("tool_use_id", "unknown")
                                    text_content += f"[Tool Result ID: {tool_id}]\n"
                                    
                                    # Extract text from the tool_result content
                                    result_content = block.get("content", [])
                                    if isinstance(result_content, list):
                                        for item in result_content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                text_content += item.get("text", "") + "\n"
                                            elif isinstance(item, dict):
                                                # Handle any dict by trying to extract text or convert to JSON
                                                if "text" in item:
                                                    text_content += item.get("text", "") + "\n"
                                                else:
                                                    try:
                                                        text_content += json.dumps(item) + "\n"
                                                    except:
                                                        text_content += str(item) + "\n"
                                    elif isinstance(result_content, dict):
                                        # Handle dictionary content
                                        if result_content.get("type") == "text":
                                            text_content += result_content.get("text", "") + "\n"
                                        else:
                                            try:
                                                text_content += json.dumps(result_content) + "\n"
                                            except:
                                                text_content += str(result_content) + "\n"
                                    elif isinstance(result_content, str):
                                        text_content += result_content + "\n"
                                    else:
                                        try:
                                            text_content += json.dumps(result_content) + "\n"
                                        except:
                                            text_content += str(result_content) + "\n"
                                
                                # Handle tool_use content blocks
                                elif block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    tool_id = block.get("id", "unknown")
                                    tool_input = json.dumps(block.get("input", {}))
                                    text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
                                
                                # Handle image content blocks
                                elif block.get("type") == "image":
                                    text_content += "[Image content - not displayed in text format]\n"
                        
                        # Make sure content is never empty for OpenAI models
                        if not text_content.strip():
                            text_content = "..."
                        
                        litellm_request["messages"][i]["content"] = text_content.strip()
                    # Also check for None or empty string content
                    elif msg["content"] is None:
                        litellm_request["messages"][i]["content"] = "..." # Empty content not allowed
                
                # 2. Remove any fields OpenAI doesn't support in messages
                for key in list(msg.keys()):
                    if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                        logger.warning(f"Removing unsupported field from message: {key}")
                        del msg[key]
            
            # 3. Final validation - check for any remaining invalid values and dump full message details
            for i, msg in enumerate(litellm_request["messages"]):
                # Log the message format for debugging
                logger.debug(f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}")
                
                # If content is still a list or None, replace with placeholder
                if isinstance(msg.get("content"), list):
                    logger.warning(f"CRITICAL: Message {i} still has list content after processing: {json.dumps(msg.get('content'))}")
                    # Last resort - stringify the entire content as JSON
                    litellm_request["messages"][i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'))}"
                elif msg.get("content") is None:
                    logger.warning(f"Message {i} has None content - replacing with placeholder")
                    litellm_request["messages"][i]["content"] = "..." # Fallback placeholder
        
        # Mark completion of OpenAI processing
        if "openai" in litellm_request["model"]:
            debug_logger.debug(f"🔧 OPENAI_PROCESSING_COMPLETE: {litellm_request['model']} | tools={len(request.tools) if request.tools else 0}")
        
        # Only log basic info about the request, not the full details
        logger.debug(f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")
        
        # Concise LiteLLM request summary for function calling debugging  
        if body_json.get("tools"):
            try:
                tools_count = len(litellm_request.get('tools', []))
                msg_count = len(litellm_request.get('messages', []))
                debug_logger.debug(f"🔧 LITELLM_REQ: {litellm_request.get('model')} | stream={litellm_request.get('stream')} | tools={tools_count} | messages={msg_count}")
            except Exception as e:
                debug_logger.error(f"Failed to log LiteLLM request summary: {e}")
            
            # Write raw request to file for debugging
            try:
                with open(os.path.join(conversation_log_dir, "raw_litellm_request.json"), "w") as f:
                    import pickle
                    # Use pickle to serialize everything, then write a summary
                    f.write("=== LiteLLM Request Debug ===\n")
                    f.write(f"Model: {litellm_request.get('model')}\n")
                    f.write(f"Stream: {litellm_request.get('stream')}\n")
                    f.write(f"Tools count: {len(litellm_request.get('tools', []))}\n")
                    f.write("=== Tools Summary ===\n")
                    for i, tool in enumerate(litellm_request.get('tools', [])):
                        f.write(f"Tool {i}: {tool.get('function', {}).get('name', 'unknown')}\n")
                    f.write("=== Full Request (str) ===\n")
                    f.write(str(litellm_request))
            except Exception as e:
                debug_logger.error(f"Failed to write raw request to file: {e}")
        
        # CRITICAL DECISION POINT: Check streaming mode
        debug_logger.debug(f"🚦 DECISION_POINT: {request.model} | stream_check={request.stream} | tools={len(request.tools) if request.tools else 0}")
        
        # Handle streaming mode
        if request.stream:
            # DEBUG: Track if we reach streaming path
            debug_logger.debug(f"🎯 ENTERED_STREAMING_PATH: Model={request.model}, Tools={len(request.tools) if request.tools else 0}")
            
            # Use LiteLLM for streaming
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            # Ensure we use the async version for streaming with timeout
            debug_logger.debug(f"🚀 Starting streaming LiteLLM call with 30-minute timeout...")
            
            # CRITICAL: Log the EXACT request LiteLLM will make to ik_llama.cpp
            debug_logger.debug(f"🔍 EXACT LITELLM REQUEST PAYLOAD:")
            debug_logger.debug(f"Model: {litellm_request.get('model')}")
            debug_logger.debug(f"Stream: {litellm_request.get('stream')}")
            debug_logger.debug(f"Temperature: {litellm_request.get('temperature')}")
            debug_logger.debug(f"Max tokens: {litellm_request.get('max_tokens')}")
            debug_logger.debug(f"Tools count: {len(litellm_request.get('tools', []))}")
            if litellm_request.get('tools'):
                debug_logger.debug(f"First tool sample: {litellm_request['tools'][0]}")
            debug_logger.debug(f"Messages count: {len(litellm_request.get('messages', []))}")
            debug_logger.debug(f"Full request keys: {list(litellm_request.keys())}")
            debug_logger.debug(f"🚨 CRITICAL CHECK: 'tools' in request keys: {'tools' in litellm_request}")
            debug_logger.debug(f"🚨 CRITICAL CHECK: Tools value: {litellm_request.get('tools', 'NOT_FOUND')}")
            
            try:
                response_generator = await asyncio.wait_for(
                    litellm.acompletion(**litellm_request),
                    timeout=1800.0  # 30 minutes timeout
                )
                debug_logger.debug(f"✅ Streaming LiteLLM call initiated successfully")
            except asyncio.TimeoutError:
                debug_logger.error(f"⏰ Streaming LiteLLM call timed out after 30 minutes")
                raise HTTPException(status_code=408, detail="Request timeout - LLM took longer than 30 minutes to respond")
            except Exception as e:
                debug_logger.error(f"❌ Streaming LiteLLM call failed: {str(e)}")
                raise
            
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # EXECUTION PATH: Non-streaming
            debug_logger.debug(f"🛤️ NON_STREAMING_PATH: Model={request.model}, Tools={len(request.tools) if request.tools else 0}")
            
            # Use LiteLLM for regular completion
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            # Log before LiteLLM call with configuration debugging
            debug_logger.debug(f"🔧 LiteLLM Configuration Debug:")
            debug_logger.debug(f"  - Target model: {litellm_request.get('model')}")
            try:
                debug_logger.debug(f"  - LiteLLM version: {litellm.__version__}")
            except AttributeError:
                debug_logger.debug(f"  - LiteLLM version: <not available>")
            debug_logger.debug(f"  - Environment variables:")
            for key in os.environ:
                if 'API' in key or 'BASE' in key or 'URL' in key:
                    debug_logger.debug(f"    {key}={os.environ[key]}")
            
            log_end_to_end_flow(request_id, "4_LITELLM_REQUEST", {
                "model": litellm_request.get("model"),
                "has_tools": bool(litellm_request.get("tools")),
                "message_count": len(litellm_request.get("messages", [])),
                "request": litellm_request
            }, request.model)
            
            start_time = time.time()
            # Non-streaming path correlation tracking  
            correlation_id = f"corr_{uuid.uuid4().hex[:12]}"
            debug_logger.debug(f"🚀 Starting LiteLLM call with 30-minute timeout...")
            debug_logger.debug(f"🔄 NON_STREAMING_START [Corr: {correlation_id}]: Model: {request.model}, Has_Tools: {bool(getattr(request, 'tools', None))}")
            
            # Non-streaming path debug - consistency with streaming path
            debug_logger.debug(f"Max tokens: {litellm_request.get('max_tokens')}")
            debug_logger.debug(f"Tools count: {len(litellm_request.get('tools', []))}")
            if litellm_request.get('tools'):
                debug_logger.debug(f"First tool sample: {litellm_request['tools'][0]}")
            debug_logger.debug(f"Messages count: {len(litellm_request.get('messages', []))}")
            debug_logger.debug(f"Full request keys: {list(litellm_request.keys())}")
            debug_logger.debug(f"🚨 CRITICAL CHECK: 'tools' in request keys: {'tools' in litellm_request}")
            debug_logger.debug(f"🚨 CRITICAL CHECK: Tools value: {litellm_request.get('tools', 'NOT_FOUND')}")
            
            try:
                litellm_response = await asyncio.wait_for(
                    litellm.acompletion(**litellm_request),
                    timeout=1800.0  # 30 minutes timeout
                )
                debug_logger.debug(f"✅ LiteLLM call completed successfully")
            except asyncio.TimeoutError:
                debug_logger.error(f"⏰ LiteLLM call timed out after 30 minutes")
                raise HTTPException(status_code=408, detail="Request timeout - LLM took longer than 30 minutes to respond")
            except Exception as e:
                debug_logger.error(f"❌ LiteLLM call failed: {str(e)}")
                raise
            response_time = time.time() - start_time
            logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={response_time:.2f}s")
            
            # Log LiteLLM response
            log_end_to_end_flow(request_id, "6_LITELLM_RESPONSE", {
                "response_time_seconds": response_time,
                "response_type": type(litellm_response).__name__,
                "has_choices": hasattr(litellm_response, 'choices'),
                "response": litellm_response
            }, request.model)
            
            # Convert LiteLLM response to Anthropic format
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            
            # Log final Anthropic response
            response_dict = anthropic_response.dict() if hasattr(anthropic_response, 'dict') else anthropic_response.__dict__
            log_end_to_end_flow(request_id, "7_FINAL_RESPONSE", {
                "response_type": "anthropic_format",
                "stop_reason": response_dict.get("stop_reason"),
                "content_types": [c.get("type") for c in response_dict.get("content", [])],
                "has_tool_use": any(c.get("type") == "tool_use" for c in response_dict.get("content", [])),
                "response": response_dict
            }, request.model)
            
            # Log response if it was a function calling request
            if body_json.get("tools"):
                log_function_calling_response(response_dict)
            
            # Non-streaming response completion tracking
            response_duration = time.time() - start_time
            has_tool_use = any(c.get("type") == "tool_use" for c in response_dict.get("content", []))
            debug_logger.debug(f"✅ NON_STREAMING_COMPLETE [Corr: {correlation_id}]: Duration: {response_duration:.2f}s, Tools: {has_tool_use}, Stop: {response_dict.get('stop_reason')}")
            
            return anthropic_response
                
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        
        # Log error with request ID
        log_end_to_end_flow(request_id, "ERROR", {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": error_traceback[:500] + "..." if len(error_traceback) > 500 else error_traceback
        }, body_json.get("model", "unknown"))
        
        # Log error if it was a function calling request
        try:
            if body_json.get("tools"):
                log_function_calling_error(e, body_json, "create_message_endpoint")
        except:
            pass  # Don't let logging errors crash the main flow
        
        # Capture as much info as possible about the error
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback
        }
        
        # Check for LiteLLM-specific attributes
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)
        
        # Check for additional exception details in dictionaries
        if hasattr(e, '__dict__'):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ['args', '__traceback__']:
                    error_details[key] = str(value)
        
        # Log all error details
        logger.error(f"Error processing request: {json.dumps(error_details, indent=2)}")
        
        # Format error for response
        error_message = f"Error: {str(e)}"
        if 'message' in error_details and error_details['message']:
            error_message += f"\nMessage: {error_details['message']}"
        if 'response' in error_details and error_details['response']:
            error_message += f"\nResponse: {error_details['response']}"
        
        # Return detailed error
        status_code = error_details.get('status_code', 500)
        raise HTTPException(status_code=status_code, detail=error_message)

@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    try:
        # Log the incoming token count request
        original_model = request.original_model or request.model
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Convert the messages to a format LiteLLM can understand
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )
        
        # Use LiteLLM's token_counter function
        try:
            # Import token_counter function
            from litellm import token_counter
            
            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get('model'),
                len(converted_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            
            # Count tokens
            token_count = token_counter(
                model=converted_request["model"],
                messages=converted_request["messages"],
            )
            
            # Return Anthropic-style response
            return TokenCountResponse(input_tokens=token_count)
            
        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # Fallback to a simple approximation
            return TokenCountResponse(input_tokens=1000)  # Default fallback
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

@app.post("/v1/chat/completions")  
async def chat_completions(raw_request: Request):
    """Log OpenAI format requests and provide guidance"""
    try:
        body = await raw_request.body()
        body_json = json.loads(body.decode('utf-8'))
        
        # Log the request
        request_logger.debug(f"OPENAI_FORMAT_REQUEST /v1/chat/completions: {json.dumps(body_json, indent=2)}")
        
        # Log function calling request if it has tools
        if body_json.get("tools"):
            log_function_calling_request(body_json, "/v1/chat/completions")
            
        return {
            "error": {
                "message": "This server expects Anthropic format requests at /v1/messages endpoint, not OpenAI format at /v1/chat/completions",
                "type": "invalid_request_error",
                "code": "endpoint_mismatch"
            },
            "debug_info": {
                "received_model": body_json.get("model"),
                "has_tools": bool(body_json.get("tools")),
                "num_tools": len(body_json.get("tools", [])),
                "correct_endpoint": "/v1/messages",
                "request_logged": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error in /v1/chat/completions endpoint: {str(e)}")
        return {"error": {"message": str(e), "type": "server_error"}}

@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"
    
    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]
    
    # Extract just the OpenAI model name without provider prefix
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"
    
    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    
    # Format status code
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    

    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"
    
    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)
    
    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")