import re
import time
import requests
import json
from rich import print
from .base_translator import Base

class LLaMACpp(Base):
    """
    Translator for a local llama.cpp endpoint that mimics the OpenAI Chat API.
    """

    def __init__(
        self,
        key,            # Not strictly needed if you're running locally, but left here for consistency
        language,
        api_base=None,  # ex: "http://127.0.0.1:8080/v1"
        prompt_template=None,
        prompt_sys_msg=None,
        temperature=0.3,
        context_flag=False,
        context_paragraph_limit=5,
        **kwargs,
    ) -> None:
        super().__init__(key, language)
        # Default to local llama.cpp endpoint if not specified
        self.api_url = api_base or "http://127.0.0.1:8080/v1/chat/completions"
        self.language = language
        self.prompt_template = (
            prompt_template
            or "Help me translate the text within triple backticks into {language}.\n"
               "Provide only the translated result.\n```{text}```"
        )
        self.prompt_sys_msg = prompt_sys_msg or "You are a helpful assistant."
        self.temperature = temperature

        # For optional context
        self.context_flag = context_flag
        self.context_list = []
        self.context_translated_list = []
        self.context_paragraph_limit = context_paragraph_limit

    def rotate_key(self):
        """
        If you had multiple local endpoints or keys to rotate,
        you could implement that logic here. For now, it's a no-op.
        """
        pass

    def create_messages(self, text, intermediate_messages=None):
        """
        Create the user prompt message, plus any previously built messages
        (like context).
        """
        current_msg = {
            "role": "user",
            "content": self.prompt_template.format(
                text=text,
                language=self.language,
            ),
        }

        messages = []
        if intermediate_messages:
            messages.extend(intermediate_messages)
        messages.append(current_msg)
        return messages

    def create_context_messages(self):
        """
        If using context, build an extra (user + assistant) pair to supply 
        relevant previous translations. 
        """
        if not self.context_flag or not self.context_list:
            return []

        return [
            {
                "role": "user",
                "content": self.prompt_template.format(
                    text="\n\n".join(self.context_list),
                    language=self.language,
                ),
            },
            {
                "role": "assistant",
                "content": "\n\n".join(self.context_translated_list),
            },
        ]

    def save_context(self, src_text, translated_text):
        """
        Keep track of paragraphs to provide ongoing context to subsequent calls.
        """
        if not self.context_flag:
            return

        self.context_list.append(src_text)
        self.context_translated_list.append(translated_text)

        # trim if we exceed limit
        if len(self.context_list) > self.context_paragraph_limit:
            self.context_list.pop(0)
            self.context_translated_list.pop(0)

    def translate(self, text):
        print(text)
        self.rotate_key()

        # Combine context messages + current request
        messages = self.create_messages(text, self.create_context_messages())

        payload = {
            "messages": [
                {"role": "system", "content": self.prompt_sys_msg},
                *messages,
            ],
            # The following parameters can be tuned as you like:
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": 1024,
            # you can include top_k, top_p, repetition_penalty, etc. if your llama.cpp server expects them
            # e.g. "top_k": 40, "top_p": 0.95,
        }

        # If your server requires additional JSON fields, add them here
        # Example (some local llama.cpp backends use `samplers`, or `cache_prompt`, etc.):
        # payload["cache_prompt"] = True
        # payload["samplers"] = "edkypmxt"

        try:
            r = requests.post(self.api_url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            # The typical OpenAI-like response
            # content might be found at data["choices"][0]["message"]["content"]
            # Adjust if your local server returns a different JSON structure
            t_text = data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[bold red]Error calling LLaMACpp API: {e}[/bold red]")
            # fallback or raise
            return ""

        # optional simple cleanup
        t_text = re.sub("\n{3,}", "\n\n", t_text)
        print("[bold green]" + t_text + "[/bold green]")

        # Save context if enabled
        if self.context_flag:
            self.save_context(text, t_text)

        return t_text
