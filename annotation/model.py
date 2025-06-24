import os
import openai
from typing import List, Union, Dict
import json
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
import numpy as np

openai.api_base = 'https://api.key77qiqi.cn/v1'
openai.api_key = "sk-WxKp6CslAmlxmP3j1002451d8cA84381922e724f8d64E941"
#os.environ["http_proxy"] = "http://127.0.0.1:10809"
#os.environ["https_proxy"] = "http://127.0.0.1:10809"

import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_embeddings(content: str):
    try:
        response = openai.Embedding.create(
            input = content,
            model = "text-embedding-3-large"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings with OpenAI: {e}")
        return None

class ChatModel():
    def __init__(self, 
                 model_name: str = "gpt-4-turbo",
                 max_tokens: int = 2048,
                 temperature: float = 0.4, 
                 n: int = 1
                 ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n = n
        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate(self, messages: List[Dict]) -> Union[List[str], str]:
        response = openai.ChatCompletion.create(
            model = self.model_name,
            messages = messages,
            max_tokens = self.max_tokens,
            temperature = self.temperature,
            n = self.n
        )
        
        if self.n == 1:
            return response.choices[0].message.content  # type: ignore
        return [choice.message.content for choice in response.choices]  # type: ignore

system_prompt = """
You only complete chats with syntax correct Verilog code. 
End the Verilog module code completion with 'endmodule'.
Do not include module, input and output definitions.
"""
user_prompt = """
Implement a D latch using an always block.
module TopModule (input d, ena,
                  output logic q
);
//[Constraint]: q should initialize to 0.
//[Constraint]: ena is level-triggered.
"""
#Prob 112
if __name__ == "__main__":
    llm = ChatModel(model_name = "gpt-4.1-mini", temperature = 0.0)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]  
    response = llm.generate(messages)
    print(response)
