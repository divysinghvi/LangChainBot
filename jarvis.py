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
from langchain_openai import ChatOpenAI
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

if not openai_key:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file")

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",  # or "gpt-3.5-turbo" if you don't have GPT-4 access
    temperature=0.6
)

# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Tools
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for searching the internet for current information."
    )
]

# Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,  # Set to True to see the agent's thought process
    max_iterations=3,  # Prevent infinite loops
    handle_parsing_errors=True
)

# Voice engines
if USE_VOICE:
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()

def speak(text):
    """Output text via voice or print"""
    if USE_VOICE:
        engine.say(text)
        engine.runAndWait()
    else:
        print(f"Jarvis: {text}")

def listen():
    """Get user input via voice or text"""
    if not USE_VOICE:
        return input("You: ")
    
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Sorry, there was an error with the speech recognition service."

def run_jarvis():
    """Main assistant loop"""
    print("🤖 JARVIS online. Type or speak your commands. Say 'exit' or 'quit' to end.")
    
    while True:
        try:
            user_input = listen()
            
            if user_input.lower() in ["exit", "quit"]:
                speak("Goodbye! Have a great day.")
                break
                
            if user_input.lower() in ["sorry, i couldn't understand that.", 
                                    "sorry, there was an error with the speech recognition service."]:
                speak("Could you please repeat that?")
                continue
                
            response = agent.run(input=user_input)
            speak(response)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            speak("I encountered an error. Please try again.")

if __name__ == "__main__":
    run_jarvis()
