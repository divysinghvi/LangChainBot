import openai

# Set API key
openai.api_key = "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"

# Use new `openai.ChatCompletion.create()` method
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # You can use "gpt-3.5-turbo" or other available models
    messages=[{"role": "user", "content": "Hello, world!"}]
)

print("✅ Response:", response['choices'][0]['message']['content'])
