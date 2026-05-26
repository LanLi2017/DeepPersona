from pathlib import Path

import boto3
import json
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

def main():
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-2"
    )

    prompt = {
        "messages": [
            {"role": "user", "content": "Hello World"}
        ],
        "max_tokens": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }

    response = client.invoke_model(
        modelId="mistral.ministral-3-3b-instruct",
        body=json.dumps(prompt),
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response['body'].read().decode('utf-8'))
    response_text = response_body['choices'][0]['message']['content']

    print(response_text)

if __name__ == "__main__":
    main()
