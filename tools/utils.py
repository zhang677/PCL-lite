import os
import re
import requests
import time
import openai
from pydantic import BaseModel

def print_elapsed_time(start_time, end_time):
    """Prints the elapsed time in hh:mm:ss format."""
    elapsed_seconds = int(end_time - start_time)  # Get total seconds
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Elapsed time: {hours:02}:{minutes:02}:{seconds:02}")

def query(
    prompt,
    **kwargs
):
    temperature = kwargs.get("temperature", 1)
    n = kwargs.get("n", 1)
    max_tokens = kwargs.get("max_tokens", 1024)
    model_name = kwargs.get("model_name", "gpt-4o-2024-08-06")
    system = kwargs.get(
        "system",
        "You are an AI assistant tasked with reasoning and generating code."
    )
    storage = kwargs.get("storage", {})
    
    s = requests.Session()
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    token = os.getenv("OPENAI_API_KEY")
    url = f"{api_base}/chat/completions"
    """Credits to @julian-q's auto-compile (Ref: github.com/julian-q/auto-compile)"""
    if "o1" in model_name:
        # No system prompt for o1 models
        # temperature, top_p and n are fixed at 1, while presence_penalty and frequency_penalty are fixed at 0.
        body = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ]
        }
    else:
        body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "n": n,
            "max_tokens": max_tokens,
            "seed": 42
        }

    query_tries = 0
    timeout = 100
    while True:
        with s.post(
            url, headers={"Authorization": f"Bearer {token}"}, json=body
        ) as resp:
            if not resp.ok:
                query_tries += 1
                # print("Query response not OK, retrying.")
                print(resp.json()["error"]["message"])
                time.sleep(10)
                if query_tries > timeout:
                    print(f"Query failed {timeout} times, aborting.")
                    exit(1)
                continue
            response = resp.json()["choices"][0]["message"]["content"]
            storage["usage"] = resp.json()["usage"]
            return response

def query_cache(
    cache,
    prompt,
    temperature=1,
    max_tokens=1024,
    model_name="claude-3-5-sonnet-20241022",
    system="You are an AI assistant tasked with reasoning and generating code."
):
    s = requests.Session()
    api_base = os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1")  # Added default API base
    token = os.getenv("ANTHROPIC_API_KEY")
    
    if not token:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        
    url = f"{api_base}/messages"
    
    body = {
        "model": model_name,
        "max_tokens": max_tokens,
        "system": system,  # Fixed system parameter format
        "messages": [  # Changed from "message" to "messages"
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": cache,
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "temperature": temperature
    }
    
    headers = {
        "x-api-key": token,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "prompt-caching-2024-07-31",
        "content-type": "application/json"
    }
    
    query_tries = 0
    max_retries = 50  # Changed variable name from timeout to max_retries for clarity
    
    while True:
        try:
            response = s.post(url, headers=headers, json=body, timeout=30)  # Added request timeout
            
            if not response.ok:
                query_tries += 1
                error_message = response.json().get("error", {}).get("message", "Unknown error")
                print(f"Attempt {query_tries}: {error_message}")
                
                if query_tries >= max_retries:
                    raise Exception(f"Query failed {max_retries} times, aborting.")
                    
                time.sleep(min(2 ** query_tries, 60))  # Exponential backoff with 60-second cap
                continue
                
            return response.json()["content"][0]
            
        except requests.exceptions.RequestException as e:
            query_tries += 1
            print(f"Request error: {str(e)}")
            
            if query_tries >= max_retries:
                raise Exception(f"Query failed {max_retries} times, aborting.")
                
            time.sleep(min(2 ** query_tries, 60)) 

def query_format(
    prompt,
    **kwargs
):
    temperature = kwargs.get("temperature", 1)
    n = kwargs.get("n", 1)
    model_name = kwargs.get("model_name", "gpt-4o-2024-08-06")
    system = kwargs.get(
        "system",
        "Streaming Tensor Programs (STeP) provides a higher-level abstraction for dataflow systems. You are a helpful programmer that can translate PyTorch program to STeP program.",
    )

    client = openai.OpenAI()

    class InputFields(BaseModel):
        model_edge: int
        shape: list[str]
        dtype: str

    class OutputFields(BaseModel):
        model_edge: int
    class RepeatRefFields(BaseModel):
        data: InputFields
        reference: InputFields
        output: OutputFields
    try: 
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            n=n,
            response_format=RepeatRefFields,
            max_tokens=50
        )
        response = completion.choices[0].message
        if response.parsed:
            print(response.parsed)
        elif response.refusal:
            # handle refusal
            print(response.refusal)
    except Exception as e:
        # Handle edge cases
        if type(e) == openai.LengthFinishReasonError:
            # Retry with a higher max tokens
            print("Too many tokens: ", e)
            pass
        else:
            # Handle other exceptions
            print(e)
            pass

def query_anthropic(
    prompt,
    **kwargs
):
    temperature = kwargs.get("temperature", 1)
    max_tokens = kwargs.get("max_tokens", 1024)
    model_name = kwargs.get("model_name", "claude-3-5-sonnet-20241022")
    system = kwargs.get("system", "You are an AI assistant tasked with reasoning and generating code.")
    storage = kwargs.get("storage", {})
    s = requests.Session()
    api_base = os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1")
    token = os.getenv("ANTHROPIC_API_KEY")
    
    if not token:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        
    url = f"{api_base}/messages"
    
    body = {
        "model": model_name,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature
    }
    
    headers = {
        "x-api-key": token,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    query_tries = 0
    max_retries = 50
    
    while True:
        try:
            response = s.post(url, headers=headers, json=body, timeout=30)
            
            if not response.ok:
                query_tries += 1
                error_message = response.json().get("error", {}).get("message", "Unknown error")
                print(f"Attempt {query_tries}: {error_message}")
                
                if query_tries >= max_retries:
                    raise Exception(f"Query failed {max_retries} times, aborting.")
                    
                time.sleep(min(2 ** query_tries, 60))  # Exponential backoff with 60-second cap
                continue
            storage["usage"] = response.json()["usage"]
            return response.json()["content"][0]
            
        except requests.exceptions.RequestException as e:
            query_tries += 1
            print(f"Request error: {str(e)}")
            
            if query_tries >= max_retries:
                raise Exception(f"Query failed {max_retries} times, aborting.")
                
            time.sleep(min(2 ** query_tries, 60))

def query_together(
    prompt,
    **kwargs
):
    temperature = kwargs.get("temperature", 1)
    max_tokens = kwargs.get("max_tokens", 1024)
    model_name = kwargs.get("model_name", "DeepSeek-V3")
    system = kwargs.get("system", "You are an AI assistant tasked with reasoning and generating code.")
    storage = kwargs.get("storage", {})
    s = requests.Session()
    api_base = os.getenv("TOGETHER_API_BASE", "https://api.together.xyz/v1")
    token = os.getenv("TOGETHER_API_KEY")
    url = f"{api_base}/chat/completions"
    if not token:
        raise ValueError("TOGETHER_API_KEY environment variable is not set")
    if model_name == "DeepSeek-V3":
        model_name = "deepseek-ai/DeepSeek-V3"
    elif model_name == "Qwen2-5-Coder-32B-Instruct" or model_name == "QwQ-32B-Preview":
        model_name = model_name.replace("2-5", "2.5")
        model_name = f"Qwen/{model_name}"
    elif model_name == "Llama-3-3-70B-Instruct-Turbo" or model_name == "Meta-Llama-3-1-405B-Instruct-Turbo":
        model_name = model_name.replace("3-3", "3.3")
        model_name = model_name.replace("3-1", "3.1")
        model_name = f"meta-llama/{model_name}"
    elif model_name == "CodeLlama-34b-Instruct-hf":
        model_name = "codellama/CodeLlama-34b-Instruct-hf"
    body = {
        "model": model_name,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
        "seed": 42
    }

    query_tries = 0
    timeout = 100
    while True:
        with s.post(
            url, headers={"Authorization": f"Bearer {token}"}, json=body
        ) as resp:
            if not resp.ok:
                query_tries += 1
                # print("Query response not OK, retrying.")
                print(resp.json()["error"]["message"])
                time.sleep(10)
                if query_tries > timeout:
                    print(f"Query failed {timeout} times, aborting.")
                    exit(1)
                continue
            response = resp.json()["choices"][0]["message"]["content"]
            storage["usage"] = resp.json()["usage"]
            return response

def query_deepseek(
    prompt,
    **kwargs
):
    temperature = kwargs.get("temperature", 1)
    max_tokens = kwargs.get("max_tokens", 1024)
    model_name = kwargs.get("model_name", "DeepSeek-V3")
    system = kwargs.get("system", "You are an AI assistant tasked with reasoning and generating code.")
    storage = kwargs.get("storage", {})
    s = requests.Session()
    api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    token = os.getenv("DEEPSEEK_API_KEY")
    url = f"{api_base}/chat/completions"
    if not token:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
    if model_name == "DeepSeek-V3":
        model_name = "deepseek-chat"
    body = {
        "model": model_name,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
        "seed": 42
    }
    query_tries = 0
    timeout = 100
    while True:
        with s.post(
            url, headers={"Authorization": f"Bearer {token}"}, json=body
        ) as resp:
            if not resp.ok:
                query_tries += 1
                # print("Query response not OK, retrying.")
                print(resp.json()["error"]["message"])
                time.sleep(10)
                if query_tries > timeout:
                    print(f"Query failed {timeout} times, aborting.")
                    exit(1)
                continue
            response = resp.json()["choices"][0]["message"]["content"]
            storage["usage"] = resp.json()["usage"]
            return response

def query_text(
    prompt,
    **kwargs
):
    model_name = kwargs.get("model_name", "claude-3-5-sonnet-20241022")
    if "claude" in model_name:
        return query_anthropic(prompt, **kwargs)['text']
    elif "gpt" in model_name or "o1" in model_name:
        return query(prompt, **kwargs)
    elif model_name == "DeepSeek-V3":
        return query_deepseek(prompt, **kwargs)
    else:
        return query_together(prompt, **kwargs)

def clean_newline(response):
    # replace double or more newlines with a single newline
    return re.sub(r"\n{2,}", "\n", response)

def clean_code(code):
    # Remove "python" and "Python" from the code
    code = re.sub(r"Python|python", "", code)
    return code

def clean_yaml(response):
    return re.sub(r"yaml|YAML|Yaml", "", response)

def extract_code(response):
    # extract code from the response
    codes = re.findall(r"```(.*?)```", response, re.DOTALL)
    if not codes:
        # if no code block is found, return the entire response
        return response
    # return the last code snippet
    return clean_code(codes[-1])
