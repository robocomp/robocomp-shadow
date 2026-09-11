import copy

import numpy as np
from openai import OpenAI
from collections import deque
import time
import re
import tiktoken
import json
import math


class LLMFunctions:
    def __init__(self, dev_prompt):
        self.developer_prompt = [{"role": "system", "content": dev_prompt}]
        self.model = "openai/gpt-oss-20b"  # Or your specific local model name
        # self.client = OpenAI(base_url="http://158.49.112.181:8080/v1", api_key="lm-studio")

        self.client = OpenAI(base_url="http://192.168.5.13:3000/v1", api_key="lm-studio")
    def send_messages(self, messages, require_json=True):
        try:
            messages_to_send = self.developer_prompt + messages

            start = time.time()
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_send,
                temperature=0.7  # Slightly higher for natural speech
            )

            content = completion.choices[0].message.content.strip()

            # --- NEW LOGIC ---
            if not require_json:
                # If we don't need JSON, just return the text!
                return True, content

            # Only try to parse JSON if required
            match = re.search(r'\{[\s\S]*\}', content, re.DOTALL)
            if match:
                json_str = match.group()
                diccionario = json.loads(json_str)
                return True, diccionario
            else:
                print("Error: JSON not found in response")
                return False, {}

        except Exception as e:
            print(f"Error in LLM call: {e}")
            return False, {}

    def contar_tokens(self, messages):
        # (Your existing code for token counting)
        pass