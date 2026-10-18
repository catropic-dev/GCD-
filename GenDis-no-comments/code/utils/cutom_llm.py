from openai import OpenAI
from typing import List
import re
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from typing import List
import re

class VLLMChatClient:

    def __init__(self, model: str='qwen27B', api_key: str='EMPTY', api_base: str='http://localhost:8864/v1'):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def ask(self, prompt: str, system_prompt: str='You are a helpful assistant.') -> str:
        chat_response = self.client.chat.completions.create(model=self.model, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}])
        return chat_response.choices[0].message.content.strip()

    def summarize_new_class_from_samples(self, known_labels: List[str], text_list: List[str], label_list: List[str], system_prompt: str='You are a taxonomy expert who defines new category names from examples.') -> str:
        pass
        example_lines = [f'- "{text}" (model prediction: "{label}")' for text, label in zip(text_list, label_list)]
        example_str = '\n'.join(example_lines[:10])
        user_prompt = f"You are given a list of existing known categories:\n    {', '.join(known_labels)}.\n\n    Now you are shown a group of samples that do not belong to any known category. We believe they all belong to the same new category.\n\n    Here are some examples:\n    {example_str}\n\n    Please output ONLY the name of the new category these samples belong to.\n\n    Use this exact format:\n    [NEW_LABEL_START] <your_category_name_here> [NEW_LABEL_END]\n\n    Only output one line in this format.\n    "
        chat_response = self.client.chat.completions.create(model=self.model, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}])
        ans = chat_response.choices[0].message.content.strip()
        return ans

    def extract_new_label_with_retry(self, known_labels: List[str], text_list: List[str], label_list: List[str], max_retries: int=10, retry_delay: float=0.5) -> str:
        pass
        pattern = '\\[NEW_LABEL_START\\](.*?)\\[NEW_LABEL_END\\]'
        for _ in range(max_retries):
            response = self.summarize_new_class_from_samples(known_labels, text_list, label_list)
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()
            time.sleep(retry_delay)
        return None

class ChatTranslateLLM(ChatOpenAI):
    pass

    def __init__(self, model_name: str='deepseek-v3:671b', openai_api_base: str='https://uni-api.cstcloud.cn/v1', **kwargs):
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError('Environment variable OPENAI_API_KEY is not set.')
        super().__init__(model_name=model_name, openai_api_base=openai_api_base, openai_api_key=openai_api_key, **kwargs)

    def summarize_new_class_from_samples(self, known_labels: List[str], text_list: List[str], label_list: List[str], system_prompt: str='You are a taxonomy expert who defines new category names from examples.') -> str:
        pass
        example_lines = [f'- "{text}" (model prediction: "{label}")' for text, label in zip(text_list, label_list)]
        example_str = '\n'.join(example_lines[:100])
        user_prompt = f"You are given a list of existing known categories:\n{', '.join(known_labels)}.\n\nNow you are shown a group of samples that do not belong to any known category. These samples received model predictions like {set(label_list)}, but we believe they all belong to the same new category.\n\nHere are some examples:\n{example_str}\n\nPlease output ONLY the name of the new category these samples belong to.\n\nUse this exact format:\n[NEW_LABEL_START] <your_category_name_here> [NEW_LABEL_END]\n\nOnly output one line in this format.\n"
        response = self.invoke([{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}])
        return response.content.strip()

    def extract_new_label_with_retry(self, known_labels: List[str], text_list: List[str], label_list: List[str], max_retries: int=10, retry_delay: float=0.5) -> str:
        pass
        pattern = '\\[NEW_LABEL_START\\](.*?)\\[NEW_LABEL_END\\]'
        for _ in range(max_retries):
            response = self.summarize_new_class_from_samples(known_labels, text_list, label_list)
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()
            time.sleep(retry_delay)
        return None
if __name__ == '__main__':
    chat_bot = ChatTranslateLLM()
    known_labels = ['Computer Vision', 'Natural Language Processing', 'Robotics']
    new_samples = ['We propose a new method for quantum entanglement optimization.', 'A novel circuit design for superconducting qubits is introduced.', 'Exploring decoherence in quantum computing with new hardware.']
    predicted_labels = ['Physics', 'Hardware', 'Physics']
    print('🧠 新类总结：')
    new_label = chat_bot.extract_new_label_with_retry(known_labels, new_samples, predicted_labels)
    print(new_label)
