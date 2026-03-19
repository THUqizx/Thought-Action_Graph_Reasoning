"""
OpenAI client for GPT interactions.

This module provides a wrapper around the OpenAI API with additional
features like error handling and retry logic.
"""

import os
from typing import List, Dict, Optional
import openai


class OpenAIClient:
    """
    Client for interacting with OpenAI API.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.chatanywhere.tech/v1"):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY environment variable.
            base_url: Base URL for the API endpoint.
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("API key not provided and OPENAI_API_KEY environment variable is not set")
        
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Call ChatCompletion API for对话.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: Model name to use.
            temperature: Controls randomness (0-1).
            max_tokens: Maximum tokens to generate.
            stream: Whether to use streaming mode.
            
        Returns:
            Generated response text.
            
        Raises:
            Exception: If API call fails.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                return full_response
            else:
                return response.choices[0].message.content
                
        except openai.APIConnectionError as e:
            raise Exception(f"Failed to connect to OpenAI API: {e}")
        except openai.RateLimitError as e:
            raise Exception(f"OpenAI API rate limit exceeded: {e}")
        except openai.APIError as e:
            raise Exception(f"OpenAI API error: {e}")
    
    def generate_text(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Simplified text generation function.
        
        Args:
            prompt: User input text.
            model: Model name to use.
            system_message: Optional system message.
            **kwargs: Additional arguments for chat_completion.
            
        Returns:
            Generated response text.
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.chat_completion(messages, model=model, **kwargs)


def main():
    """
    Main function with usage examples.
    """
    # Initialize client
    # Option 1: Set environment variable
    # export OPENAI_API_KEY='your-api-key'
    # client = OpenAIClient()
    
    # Option 2: Pass API key directly
    # api_key = "your-api-key-here"
    # client = OpenAIClient(api_key)
    
    # Example 1: Simple conversation
    print("=== Simple Conversation Example ===")
    client = OpenAIClient()  # Assumes OPENAI_API_KEY is set
    response = client.generate_text("Please write a Hello World program in Python")
    print(response)
    
    print("\n" + "=" * 50 + "\n")
    
    # Example 2: Conversation with system message
    print("=== Conversation with System Message Example ===")
    response = client.generate_text(
        "Please explain the concept of overfitting in machine learning",
        system_message="You are a senior machine learning expert. Please explain technical concepts in simple terms.",
        model="gpt-3.5-turbo"
    )
    print(response)
    
    print("\n" + "=" * 50 + "\n")
    
    # Example 3: Multi-turn conversation
    print("=== Multi-turn Conversation Example ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, what can you help me with?"},
        {"role": "assistant", "content": "I can help you answer questions, write code, translate, and more. How can I assist you?"},
        {"role": "user", "content": "Can you help me write a short poem about spring?"}
    ]
    
    response = client.chat_completion(messages)
    print(response)


if __name__ == "__main__":
    main()
