import os
"""
This script implements a conversational AI assistant named "Jarvis" using LangChain and OpenAI's GPT-4 model. 
The assistant can process user inputs, maintain conversation history, and optionally support voice input/output.

Modules and Features:
- `dotenv`: Loads environment variables from a `.env` file.
- `langchain_openai`: Provides the ChatOpenAI class for GPT-4 integration.
- `langchain.memory`: Enables conversation memory using ConversationBufferMemory.
- `langchain.agents`: Facilitates agent initialization with tools and memory.
- `langchain_community.tools`: Includes DuckDuckGoSearchRun for internet search functionality.
- Optional voice support using `speech_recognition` and `pyttsx3`.

Key Components:
1. **Environment Setup**:
    - Loads the OpenAI API key from the `.env` file.

2. **Language Model**:
    - Initializes a GPT-4-based language model with a specified temperature for response variability.

3. **Memory**:
    - Uses ConversationBufferMemory to maintain chat history.

4. **Tools**:
    - Integrates a DuckDuckGo search tool for internet queries.

5. **Agent**:
    - Combines the language model, tools, and memory into a conversational agent.

6. **Voice Support** (Optional):
    - Provides text-to-speech and speech-to-text capabilities if enabled.

Functions:
- `speak(text)`: Outputs text via voice (if enabled) or prints it to the console.
- `listen()`: Captures user input via text or microphone (if voice is enabled).
- `run_jarvis()`: Main loop for running the assistant. Processes user commands and generates responses.

Usage:
- Run the script to start the assistant.
- Type or speak commands (if voice is enabled).
- Say or type "exit" or "quit" to terminate the assistant.

Note:
- Ensure the `.env` file contains a valid `OPENAI_API_KEY`.
- Install required dependencies (`langchain`, `speech_recognition`, `pyttsx3`, etc.) before running.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # ✅ Correct
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents.agent_types import AgentType

# Optional Voice
USE_VOICE = False
if USE_VOICE:
    import speech_recognition as sr
    import pyttsx3

# Load API key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(model="{correct_model_name}", temperature=0.6)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history")


# Tools
search = DuckDuckGoSearchRun()
tools = [Tool(name="Search", func=search.run, description="Search the internet.")]

# Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False,
)

# Voice engines
if USE_VOICE:
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()

def speak(text):
    if USE_VOICE:
        engine.say(text)
        engine.runAndWait()
    else:
        print(f"Jarvis: {text}")

def listen():
    if not USE_VOICE:
        return input("You: ")
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except:
            return "Sorry, I couldn't understand that."

# Run assistant
def run_jarvis():
    print("🧠 JARVIS online. Type or speak your commands. Say 'exit' to quit.")
    while True:
        user_input = listen()
        if user_input.lower() in ["exit", "quit"]:
            speak("Goodbye, sir.")
            break
        response = agent.run(user_input)  # ✅ Use run
        speak(response)


if __name__ == "__main__":
    run_jarvis()
