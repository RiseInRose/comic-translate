from .base import TranslationEngine
from .google import GoogleTranslation
from .microsoft import MicrosoftTranslation
from .deepl import DeepLTranslation
from .yandex import YandexTranslation
from .llm.gpt import GPTTranslation
from .llm.claude import ClaudeTranslation
from .llm.gemini import GeminiTranslation
from .llm.deepseek import DeepseekTranslation
from .llm.custom import CustomTranslation


class TranslationEngineFactory:
    """Factory for creating appropriate translation engines based on settings."""
    
    _engines = {}  # Cache of created engines
    
    # Map traditional translation services to their engine classes
    TRADITIONAL_ENGINES = {
        "Google Translate": GoogleTranslation,
        "Microsoft Translator": MicrosoftTranslation,
        "DeepL": DeepLTranslation,
        "Yandex": YandexTranslation
    }
    
    # Map LLM identifiers to their engine classes
    LLM_ENGINE_IDENTIFIERS = {
        "GPT": GPTTranslation,
        "Claude": ClaudeTranslation,
        "Gemini": GeminiTranslation,
        "Deepseek": DeepseekTranslation,
        "Custom": CustomTranslation
    }
    
    # Default engines for fallback
    DEFAULT_TRADITIONAL_ENGINE = GoogleTranslation
    DEFAULT_LLM_ENGINE = GPTTranslation
    
    @classmethod
    def create_engine(cls, settings, source_lang: str, target_lang: str, translator_key: str, logger=None) -> TranslationEngine:
        """
        Create or retrieve an appropriate translation engine based on settings.
        
        Args:
            settings: Settings object with translation configuration
            source_lang: Source language name
            target_lang: Target language name
            translator_key: Key identifying which translator to use
            
        Returns:
            Appropriate translation engine instance
        """
        # Create a cache key based on translator and language pair
        cache_key = f"{translator_key}_{source_lang}_{target_lang}"
        
        # Return cached engine if available
        if cache_key in cls._engines:
            return cls._engines[cache_key]
        
        # Determine engine class and create engine
        engine_class = cls._get_engine_class(translator_key)
        engine = engine_class()
        
        # Initialize with appropriate parameters
        if translator_key in cls.TRADITIONAL_ENGINES:
            engine.initialize(settings, source_lang, target_lang, logger=logger)
        else:
            engine.initialize(settings, source_lang, target_lang, model_type=translator_key, logger=logger)
        
        # Cache the engine
        cls._engines[cache_key] = engine
        return engine
    
    @classmethod
    def _get_engine_class(cls, translator_key: str):
        """Get the appropriate engine class based on translator key."""
        # First check if it's a traditional translation engine (exact match)
        if translator_key in cls.TRADITIONAL_ENGINES:
            return cls.TRADITIONAL_ENGINES[translator_key]
        
        # Otherwise look for matching LLM engine (substring match)
        for identifier, engine_class in cls.LLM_ENGINE_IDENTIFIERS.items():
            if identifier in translator_key:
                return engine_class
        
        # Default to LLM engine if no match found
        return cls.DEFAULT_LLM_ENGINE