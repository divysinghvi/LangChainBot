from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class LangchainGPTBotModel:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = OpenAI(temperature=0.7, openai_api_key=self.api_key)

    def create_chain(self, prompt_template: str):
        prompt = PromptTemplate(input_variables=["input"], template=prompt_template)
        return LLMChain(llm=self.llm, prompt=prompt)

    def generate_response(self, chain: LLMChain, user_input: str):
        return chain.run(input=user_input)
    