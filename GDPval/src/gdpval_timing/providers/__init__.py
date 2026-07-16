from .anthropic import AnthropicProvider
from .google import GoogleProvider
from .openai import OpenAIProvider

PROVIDERS = {"anthropic": AnthropicProvider, "openai": OpenAIProvider, "google": GoogleProvider}
