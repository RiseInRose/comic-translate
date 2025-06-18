import json
from typing import Any
import numpy as np

from .base import BaseLLMTranslation
from ...utils.translator_utils import get_llm_client, encode_image_array
import requests


class CustomTranslation(BaseLLMTranslation):
    """Translation engine using custom LLM configurations."""

    def __init__(self):
        """Initialize Custom LLM translation engine."""
        super().__init__()

    def initialize(self, settings: Any, source_lang: str, target_lang: str, **kwargs) -> None:
        """
        Initialize Custom LLM translation engine.

        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            **kwargs: Additional parameters
        """
        super().initialize(settings, source_lang, target_lang, **kwargs)

        credentials = settings.get_credentials(settings.ui.tr('Custom'))
        self.api_key = credentials.get('api_key', "")
        self.api_url = credentials.get('api_url', "")
        self.model = credentials.get('model', "")
        self.openai_api_key = credentials.get('openai_api_key', "")
        self.openai_api_url = credentials.get('openai_api_url', "")
        self.client = get_llm_client('Custom', self.api_key, self.api_url)

    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using Custom model (default to GPT-like interface).

        Args:
            user_prompt: User prompt for LLM
            system_prompt: System prompt for LLM
            image: Image as numpy array

        Returns:
            Translated JSON text
        """

        payload = json.dumps(
            {"model": "gpt-4o-mini", "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
            ]}
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        }

        print(payload)

        error_arr = []
        api_url = self.api_url
        i = 0
        while i < 3:
            try:
                response = requests.request("POST", f"{api_url}/chat/completions", headers=headers, data=payload)

                data = response.json()
                print(data)
                s = data.get('choices', [])[0].get('message').get('content')
                return s
            except Exception as ex:
                error_arr.append('-------trans fail------%s' % str(ex))
                print('-------trans fail-------')
                print(ex)
                # # 直连openai备用key
                # api_url = self.openai_api_url
                # headers["Authorization"] = f"Bearer {self.openai_api_key}"

                server_resp_text = ''
                try:
                    data = {
                        'key1': self.api_key,
                        'key2': self.openai_api_key,
                        'payload': payload,
                    }

                    resp = requests.post('https://www.mangatranslate.com/api/v1/manga/contenttranslate', data=data)
                    server_resp_text = resp.text
                    s = resp.json().get('content')
                    return s
                except Exception as ex:
                    error_arr.append('------call server-trans fail------%s--%s' % (str(ex), server_resp_text))
                    print('-----------call server trans fail-----%s' % str(ex))

                i += 1

        if 'logger' in self.__dict__:
            self.logger.error('------重试了三次翻译 trans fail------%s' % '\r\n'.join(error_arr))

        return None

        # encoded_image = None
        # if self.img_as_llm_input:
        #     encoded_image = encode_image_array(image)
        #
        #     message = [
        #         {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        #         {"role": "user", "content": [
        #             {"type": "text", "text": user_prompt},
        #             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
        #         ]}
        #     ]
        # else:
        #     message = [
        #         {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        #         {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        #     ]
        #
        # print(message)
        # i = 0
        # while i < 3:
        #     try:
        #         response = self.client.chat.completions.create(
        #             model=self.model,
        #             messages=message,
        #             temperature=1,
        #             max_tokens=4000,
        #         )
        #         print(response.choices[0].message.content)
        #         # data = json.loads(response.choices[0].message.content)
        #         return response.choices[0].message.content
        #     except Exception as ex:
        #         print(ex)
        #         i += 1
        #
        # return response.choices[0].message.content


