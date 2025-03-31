# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "requests",
#    "httpx",
# ]
# ///

import os
import httpx
from typing import Optional
import os
from .tool_function import tool_functions


AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

async def create_openai_url_request(question: str, file_path: Optional[str] = None):
    """
    Create an OpenAI request to the chat/completions API to get an answer to a data science question.

    Args:
        question (str): The question to ask
        file_path (Optional[str], optional): The path to a file to upload and reference in the question. Defaults to None.

    Returns:
        Dict[str, Any]: The JSON response from the OpenAI API
    """
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
    }
    message = [
        {
            "role": "system",
            "content": "You are an assistant designed to solve data science assignment problems. You should use the provided functions when appropriate to solve the problem.",
        },
        {"role": "user", "content": question},
    ]

    if file_path:
        message.append(
            {
                "role": "user",
                "content": f"I've uploaded a file that you can process. The file is stored at: {file_path}",
            }
        )
    data = {
        "model": "gpt-4o-mini",
        "messages": message,
        "tools": tool_functions,
        "tool_choice": "auto",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            headers=headers,
            json=data,
            timeout=60.0,
        )

        if response.status_code != 200:
            raise Exception(f"Error from OpenAI API: {response.text}")

    return response.json()