import requests
import json
import time
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMTier(Enum):
    """LLM Tier definitions"""
    LOCAL = "local"      # Pi 3 Mini on server
    CLOUD_FAST = "cloud_fast"  # Mistral
    CLOUD_SMART = "cloud_smart"  # DeepSeek

class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, name: str, tier: LLMTier):
        self.name = name
        self.tier = tier
        self.available = True
        self.response_time = 0.0
        self.last_check = 0
        self.consecutive_failures = 0
        self.max_failures = 3
    
    def is_available(self) -> bool:
        """Check if provider is available"""
        return self.available and self.consecutive_failures < self.max_failures
    
    def mark_failure(self):
        """Mark provider as failed"""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_failures:
            self.available = False
            logger.warning(f"Provider {self.name} marked as unavailable after {self.consecutive_failures} failures")
    
    def mark_success(self):
        """Mark provider as successful"""
        self.consecutive_failures = 0
        self.available = True
    
    def call_api(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """Override in subclasses"""
        raise NotImplementedError

class Pi3MiniProvider(LLMProvider):
    """Pi 3 Mini local server provider"""
    
    def __init__(self, server_url: str = "http://localhost:11434"):
        super().__init__("Phi3:Mini", LLMTier.LOCAL)
        self.server_url = server_url
        self.api_endpoint = f"{server_url}/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
    
    def call_api(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """Call Pi 3 Mini local server API"""
        try:
            start_time = time.time()
            
            payload = {
                "model": "phi3:mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            response = requests.post(
                self.api_endpoint,
                headers=self.headers,
                json=payload,
                timeout=100  # Shorter timeout for local server
            )
            
            self.response_time = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                result = response_data["choices"][0]["message"]["content"].strip()
                self.mark_success()
                return result
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.mark_failure()
            logger.error(f"Pi3Mini API error: {str(e)}")
            raise Exception(f"Pi3Mini API error: {str(e)}")

class MistralProvider(LLMProvider):
    """Mistral cloud provider"""
    
    def __init__(self, api_key: str = "2epMj4QZTuoM4uVXoI66z8QZ3S1rIbQs", model_name: str = "mistral-small-latest"):
        super().__init__("Mistral", LLMTier.CLOUD_FAST)
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def call_api(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """Call Mistral API"""
        try:
            start_time = time.time()
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            self.response_time = time.time() - start_time
            
            response.raise_for_status()
            response_data = response.json()
            result = response_data["choices"][0]["message"]["content"].strip()
            self.mark_success()
            return result
            
        except Exception as e:
            self.mark_failure()
            logger.error(f"Mistral API error: {str(e)}")
            raise Exception(f"Mistral API error: {str(e)}")

class DeepSeekProvider(LLMProvider):
    """DeepSeek cloud provider"""
    
    def __init__(self, api_key: str = "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", model_name: str = "deepseek/deepseek-chat-v3-0324:free"):
        super().__init__("DeepSeek", LLMTier.CLOUD_SMART)
        self.api_key = api_key
        self.model_name = model_name
    # Change the API URL to OpenRouter's endpoint
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
       "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}",
        "HTTP-Referer": "http://localhost:8000",  # or your actual domain
        "X-Title": "CIE_RAG"}
    
    
    def call_api(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """Call DeepSeek API"""
        try:
            start_time = time.time()
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=45  # Longer timeout for more complex model
            )
            
            self.response_time = time.time() - start_time
            
            response.raise_for_status()
            response_data = response.json()
            result = response_data["choices"][0]["message"]["content"].strip()
            self.mark_success()
            return result
            
        except Exception as e:
            self.mark_failure()
            logger.error(f"DeepSeek API error: {str(e)}")
            raise Exception(f"DeepSeek API error: {str(e)}")

class MultiTierLLM:
    """
    Multi-tier LLM system with intelligent switching
    Drop-in replacement for MistralLLM with same interface
    """
    
    def __init__(self, mistral_api_key: str = "2epMj4QZTuoM4uVXoI66z8QZ3S1rIbQs", model_name: str = "mistral-small-latest"):
        """
        Initialize Multi-tier LLM system
        Args:
            mistral_api_key: Mistral API key (hardcoded default)
            model_name: Model name (for compatibility)
        """
        # Initialize providers
        self.providers = [
            Pi3MiniProvider(),
            MistralProvider(api_key=mistral_api_key),
            DeepSeekProvider()
        ]
        
        # Switching logic configuration
        self.complexity_threshold = 500  # chars
        self.context_threshold = 2000    # chars
        self.prefer_local = True
        self.fallback_enabled = True
        
        logger.info("Multi-tier LLM initialized with Pi3Mini, Mistral, and DeepSeek")
    
    def analyze_query_complexity(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze query complexity to determine best provider
        Returns: {
            'complexity': 'simple'|'medium'|'complex',
            'context_length': int,
            'requires_reasoning': bool,
            'recommended_tier': LLMTier
        }
        """
        query_length = len(query)
        context_length = len(context)
        total_length = query_length + context_length
        
        # Keywords that suggest complex reasoning
        complex_keywords = [
            'analyze', 'compare', 'evaluate', 'synthesize', 'reasoning',
            'logic', 'complex', 'detailed', 'comprehensive', 'elaborate',
            'implications', 'consequences', 'relationship', 'correlation'
        ]
        
        # Check for complex reasoning requirements
        requires_reasoning = any(keyword in query.lower() for keyword in complex_keywords)
        
        # Determine complexity
        if total_length > self.context_threshold or requires_reasoning:
            complexity = 'complex'
            recommended_tier = LLMTier.CLOUD_SMART
        elif total_length > self.complexity_threshold:
            complexity = 'medium'
            recommended_tier = LLMTier.CLOUD_FAST
        else:
            complexity = 'simple'
            recommended_tier = LLMTier.LOCAL
        
        return {
            'complexity': complexity,
            'context_length': context_length,
            'requires_reasoning': requires_reasoning,
            'recommended_tier': recommended_tier
        }
    
    def select_provider(self, query: str, context: str = "") -> LLMProvider:
        """
        Select best available provider based on query complexity
        """
        analysis = self.analyze_query_complexity(query, context)
        recommended_tier = analysis['recommended_tier']
        
        # Try recommended tier first
        for provider in self.providers:
            if provider.tier == recommended_tier and provider.is_available():
                logger.info(f"Selected {provider.name} (tier: {provider.tier.value}) for {analysis['complexity']} query")
                return provider
        
        # Fallback logic: try in order of preference
        if self.prefer_local:
            # Try local first, then cloud
            tier_order = [LLMTier.LOCAL, LLMTier.CLOUD_FAST, LLMTier.CLOUD_SMART]
        else:
            # Try cloud first, then local
            tier_order = [LLMTier.CLOUD_FAST, LLMTier.CLOUD_SMART, LLMTier.LOCAL]
        
        for tier in tier_order:
            for provider in self.providers:
                if provider.tier == tier and provider.is_available():
                    logger.info(f"Fallback to {provider.name} (tier: {provider.tier.value})")
                    return provider
        
        # Last resort: try any available provider
        for provider in self.providers:
            if provider.is_available():
                logger.warning(f"Last resort: using {provider.name}")
                return provider
        
        raise Exception("No providers available")
    
    def format_context(self, search_results: List[Dict]) -> str:
        """
        Format search results into context string
        Args:
            search_results: List of search results from vector database
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            # Extract source information
            source_info = f"Source: {result.get('source', 'Unknown')}"
            
            # Add additional metadata based on file type
            metadata = result.get('metadata', {})
            if result.get('file_type') == 'pdf' and 'page' in metadata:
                source_info += f" (Page {metadata['page']})"
            elif result.get('file_type') == 'csv' and 'row_number' in metadata:
                source_info += f" (Row {metadata['row_number']})"
            elif result.get('file_type') == 'json' and 'json_path' in metadata:
                source_info += f" (Path: {metadata['json_path']})"
            
            context_parts.append(f"Context {i}:\n{source_info}\nContent: {result['text']}\n")
        
        return "\n".join(context_parts)

    def create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt combining query and context
        Args:
            query: User's question
            context: Retrieved context from vector database
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an AI assistant that answers questions based on provided context. Use the context below to answer the user's question accurately and comprehensively.

Context:
{context}

Question: {query}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information to fully answer the question, say so
- Be specific and cite relevant parts of the context when possible
- If you need to make inferences, clearly indicate that
- Keep your response focused and relevant to the question

Answer:"""
        return prompt

    def create_prompt_with_history(self, history: List[Dict], context: str) -> str:
        """
        Creates a prompt including multi-turn history + retrieved context.
        Keeps recent conversation turns along with the current query.
        """
        history_str = ""
        for msg in history[-6:]:  # Limit to last 6 turns
            role = msg['role'].capitalize()
            content = msg['content']
            history_str += f"{role}: {content}\n"
        
        prompt = f"""You are an AI assistant helping the user based on the ongoing conversation and the context provided.

Context:
{context}

Conversation so far:
{history_str}

Answer the user's latest message clearly and concisely, based only on the above information.
If you're unsure, say so.

Answer:"""
        return prompt.strip()

    def call_api(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """
        Call API with intelligent provider selection and fallback
        Args:
            prompt: The formatted prompt
            max_tokens: Maximum tokens in response
            temperature: Creativity/randomness (0.0 to 1.0)
        Returns:
            LLM response text
        """
        provider = self.select_provider(prompt)
        
        try:
            result = provider.call_api(prompt, max_tokens, temperature)
            logger.info(f"Successfully got response from {provider.name} in {provider.response_time:.2f}s")
            return result
            
        except Exception as e:
            if not self.fallback_enabled:
                raise e
            
            # Try fallback providers
            logger.warning(f"Primary provider {provider.name} failed, trying fallbacks...")
            
            for fallback_provider in self.providers:
                if fallback_provider != provider and fallback_provider.is_available():
                    try:
                        result = fallback_provider.call_api(prompt, max_tokens, temperature)
                        logger.info(f"Fallback successful with {fallback_provider.name}")
                        return result
                    except Exception as fallback_error:
                        logger.warning(f"Fallback {fallback_provider.name} also failed: {fallback_error}")
                        continue
            
            # All providers failed
            raise Exception(f"All providers failed. Last error: {str(e)}")

    def generate_response(self, query: str, search_results: List[Dict], 
                         max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """
        Generate response from query and search results
        Args:
            query: User's question
            search_results: List of search results from vector database
            max_tokens: Maximum tokens in LLM response
            temperature: LLM temperature setting
        Returns:
            Generated response string
        """
        if not search_results:
            return "I couldn't find any relevant information to answer your question."
        
        # Format context from search results
        context = self.format_context(search_results)
        
        # Create prompt
        prompt = self.create_prompt(query, context)
        
        # Generate response
        response = self.call_api(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response

    def generate_response_with_history(self, query: str, search_results: List[Dict], 
                                     chat_history: List[Dict], max_tokens: int = 1000, 
                                     temperature: float = 0.1) -> str:
        """
        Generate response with chat history context
        Args:
            query: User's current question
            search_results: List of search results from vector database
            chat_history: Previous conversation history
            max_tokens: Maximum tokens in LLM response
            temperature: LLM temperature setting
        Returns:
            Generated response string
        """
        # Format context from search results
        context = self.format_context(search_results)
        
        # Add current query to history
        current_history = chat_history + [{"role": "user", "content": query}]
        
        # Create prompt with history
        prompt = self.create_prompt_with_history(current_history, context)
        
        # Generate response
        response = self.call_api(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        status = {}
        for provider in self.providers:
            status[provider.name] = {
                'tier': provider.tier.value,
                'available': provider.is_available(),
                'consecutive_failures': provider.consecutive_failures,
                'response_time': provider.response_time
            }
        return status

# BACKWARD COMPATIBILITY SECTION
# This ensures your FastAPI code works without any changes

# Create a shared global instance for state management
_shared_instance = None

def get_shared_instance():
    """Get the shared MultiTierLLM instance"""
    global _shared_instance
    if _shared_instance is None:
        _shared_instance = MultiTierLLM()
    return _shared_instance

# Make MistralLLM a callable class that returns the shared instance
class MistralLLM:
    """
    Backward compatibility wrapper that returns the shared MultiTierLLM instance
    This ensures all FastAPI calls use the same instance with shared state
    """
    
    def __new__(cls, mistral_api_key: str = "2epMj4QZTuoM4uVXoI66z8QZ3S1rIbQs", model_name: str = "mistral-small-latest"):
        """
        Return the shared instance instead of creating a new one
        This maintains state across all calls in your FastAPI application
        """
        instance = get_shared_instance()
        # Update API key if a new one is provided
        if mistral_api_key != "2epMj4QZTuoM4uVXoI66z8QZ3S1rIbQs":
            # Update Mistral provider with new API key
            for provider in instance.providers:
                if isinstance(provider, MistralProvider):
                    provider.api_key = mistral_api_key
                    provider.headers["Authorization"] = f"Bearer {mistral_api_key}"
        return instance

# Also create the global instance for direct access (if needed elsewhere)
mistral_llm = get_shared_instance()
