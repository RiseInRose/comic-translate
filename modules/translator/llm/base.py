import json, requests
from typing import Any
import numpy as np
from abc import abstractmethod

from ..base import LLMTranslation
from ...utils.maga_utils import language_codes
from ...utils.textblock import TextBlock
from ...utils.translator_utils import get_raw_text, set_texts_from_json


class BaseLLMTranslation(LLMTranslation):
    """Base class for LLM-based translation engines with shared functionality."""
    
    def __init__(self):
        """Initialize LLM translation engine."""
        self.source_lang = None
        self.target_lang = None
        self.api_key = None
        self.api_url = None
        self.client = None
        self.model = None
        self.img_as_llm_input = False
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, **kwargs) -> None:
        """
        Initialize the LLM translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            **kwargs: Engine-specific initialization parameters
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.img_as_llm_input = settings.get_llm_settings()['image_input_enabled']
        
    def translate(self, blk_list: list[TextBlock], image: np.ndarray, extra_context: str) -> list[TextBlock]:
        """
        Translate text blocks using LLM.
        
        Args:
            blk_list: List of TextBlock objects to translate
            image: Image as numpy array
            extra_context: Additional context information for translation
            
        Returns:
            List of updated TextBlock objects with translations
        """
        try:
            vip = True
            need_google_trans = True
            if vip:
                print('google_trans fail')
                entire_raw_text = get_raw_text(blk_list)
                system_prompt = self.get_system_prompt(self.source_lang, self.target_lang)
                user_prompt = f"{extra_context}\nMake the translation sound as natural as possible.\nTranslate this:\n{entire_raw_text}"

                entire_translated_text = self._perform_translation(user_prompt, system_prompt, image)
                set_texts_from_json(blk_list, entire_translated_text)
                for blk in blk_list:
                    if blk.translation is not None and len(blk.translation) > 0:
                        need_google_trans = False
                        break

            if need_google_trans:
                payload = json.dumps({
                    'src': language_codes.get(self.source_lang),
                    'dest': language_codes.get(self.target_lang),
                    'text_list': [blk.text for blk in blk_list]
                })

                data = {
                    'payload': payload,
                }
                print(data)
                resp = requests.post('https://www.mangatranslate.com/api/v1/manga/googletranslate', data=data)
                # resp = requests.post('http://127.0.0.1:8002/api/v1/manga/googletranslate', data=data)
                google_trans_result = resp.json()
                if google_trans_result.get('result') == 'success':
                    print('google_trans success')
                    content = google_trans_result.get('content')
                    for blk in blk_list:
                        blk.translation = content.get(blk.text)

            # payload = json.dumps({
            #     'src': language_codes.get(self.source_lang),
            #     'dest': language_codes.get(self.target_lang),
            #     'text_list': [blk.text for blk in blk_list]
            # })
            #
            # data = {
            #     'payload': payload,
            # }
            # print(data)
            # resp = requests.post('https://www.mangatranslate.com/api/v1/manga/googletranslate', data=data)
            # # resp = requests.post('http://127.0.0.1:8002/api/v1/manga/googletranslate', data=data)
            # google_trans_result = resp.json()
            # if google_trans_result.get('result') == 'success':
            #     print('google_trans success')
            #     content = google_trans_result.get('content')
            #     for blk in blk_list:
            #         blk.translation = content.get(blk.text)
            # else:
            #     print('google_trans fail')
            #     entire_raw_text = get_raw_text(blk_list)
            #     system_prompt = self.get_system_prompt(self.source_lang, self.target_lang)
            #     user_prompt = f"{extra_context}\nMake the translation sound as natural as possible.\nTranslate this:\n{entire_raw_text}"
            #
            #     entire_translated_text = self._perform_translation(user_prompt, system_prompt, image)
            #     set_texts_from_json(blk_list, entire_translated_text)
        
        except Exception as e:
            print(f"{type(self).__name__} translation error: {str(e)}")
            
        return blk_list
    
    @abstractmethod
    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using specific LLM.
        
        Args:
            user_prompt: User prompt for LLM
            system_prompt: System prompt for LLM
            image: Image as numpy array
            
        Returns:
            Translated JSON text
        """
        pass