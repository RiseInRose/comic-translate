import numpy as np

from ..utils.textblock import TextBlock
from .base import LLMTranslation
from .factory import TranslationEngineFactory


class Translator:
    """
    Main translator class that orchestrates the translation process.
    
    Supports multiple translation engines including:
    - Traditional translators (e.g Google, Microsoft, DeepL, Yandex)
    - LLM-based translators (e.g GPT, Claude, Gemini, Deepseek, Custom)
    """
    
    def __init__(self, main_page, source_lang: str = "", target_lang: str = "", logger=None):
        """
        Initialize translator with settings and languages.
        
        Args:
            main_page: Main application page with settings
            source_lang: Source language name (localized)
            target_lang: Target language name (localized)
        """
        self.main_page = main_page
        self.settings = main_page.settings_page
        
        self.translator_key = self._get_translator_key(self.settings.get_tool_selection('translator'))
        
        self.source_lang = source_lang
        self.source_lang_en = self._get_english_lang(main_page, self.source_lang)
        self.target_lang = target_lang
        self.target_lang_en = self._get_english_lang(main_page, self.target_lang)

        print('create_engine')
        # Create appropriate engine using factory
        self.engine = TranslationEngineFactory.create_engine(
            self.settings,
            self.source_lang_en,
            self.target_lang_en,
            self.translator_key,
            logger=logger
        )
        print(self.engine)
        
        # Track engine type for method dispatching
        self.is_llm_engine = isinstance(self.engine, LLMTranslation)
    
    def _get_translator_key(self, localized_translator: str) -> str:
        """
        Map localized translator names to standard keys.
        
        Args:
            localized_translator: Translator name in UI language
            
        Returns:
            Standard translator key
        """
        translator_map = {
            self.settings.tr("Custom"): "Custom",
            self.settings.tr("Deepseek-v3"): "Deepseek-v3",
            self.settings.tr("GPT-4o"): "GPT-4o",
            self.settings.tr("GPT-4o mini"): "GPT-4o mini",
            self.settings.tr("Claude-3-Opus"): "Claude-3-Opus",
            self.settings.tr("Claude-3.7-Sonnet"): "Claude-3.7-Sonnet",
            self.settings.tr("Claude-3.5-Haiku"): "Claude-3.5-Haiku",
            self.settings.tr("Gemini-2.0-Flash"): "Gemini-2.0-Flash",
            self.settings.tr("Gemini-2.0-Pro"): "Gemini-2.0-Pro",
            self.settings.tr("Google Translate"): "Google Translate",
            self.settings.tr("Microsoft Translator"): "Microsoft Translator",
            self.settings.tr("DeepL"): "DeepL",
            self.settings.tr("Yandex"): "Yandex"
        }
        return translator_map.get(localized_translator, localized_translator)
    
    def _get_english_lang(self, main_page, translated_lang: str) -> str:
        """
        Get English language name from localized language name.
        
        Args:
            main_page: Main application page with language mapping
            translated_lang: Language name in UI language
            
        Returns:
            Language name in English
        """
        return main_page.lang_mapping.get(translated_lang, translated_lang)

    def translate(self, blk_list: list[TextBlock], image: np.ndarray = None, extra_context: str = "") -> list[TextBlock]:
        """
        Translate text in text blocks using the configured translation engine.
        
        Args:
            blk_list: List of TextBlock objects to translate
            image: Image as numpy array (for context in LLM translators)
            extra_context: Additional context information for translation
            
        Returns:
            List of updated TextBlock objects with translations
        """
        if self.is_llm_engine:
            # LLM translators need image and extra context
            return self.engine.translate(blk_list, image, extra_context)
        else:
            # Text-based translators only need the text blocks
            return self.engine.translate(blk_list)