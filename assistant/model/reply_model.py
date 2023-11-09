import openai
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint
from langchain.prompts import PromptTemplate

from assistant.api.sagemaker_client import SagemakerClient
from assistant.common.constant import (
    FINE_TUNED_GPT_35,
    FINE_TUNED_LLAMA2,
    FINE_TUNED_GPT_4,
)
from assistant.model.knowledge_base import KnowledgeBase
from assistant.model.prompts.out_reach_prompt import generate_out_reach_prompt
import os

from assistant.model.prompts.reply_prompt import generate_reply_prompt


class ReplyLLM:
    def __init__(
        self,
        model_name: str,
        profile_name: str = None,
        region_name: str = None,
        endpoint_name: str = None,
    ) -> None:
        self.model_name = model_name
        self.profile_name = profile_name
        self.region_name = region_name
        self.endpoint_name = endpoint_name
        self.sagemaker_client = (
            SagemakerClient(
                profile_name=profile_name,
                region_name=region_name,
                endpoint_name=endpoint_name,
            )
            if model_name == FINE_TUNED_LLAMA2
            else None
        )
        self.llm = self.create_llm()
        self.kb = KnowledgeBase()
        self.vec_db = self.kb.create_knowledge_base()

    def create_llm(self):
        if self.model_name == FINE_TUNED_GPT_35:
            llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")
        elif self.model_name == FINE_TUNED_GPT_4:
            llmm = ChatOpenAI(
                openai_api_key=os.environ.get("OPEN-API-KEY"),
                temperature=0.9,
                # max_tokens=100,
                model="gpt-4-1106-preview",
            )
            prompt = PromptTemplate(
                input_variables=["message", "best_practice"],
                template=generate_reply_prompt(),
            )
            llm = LLMChain(llm=llmm, prompt=prompt)
            # llm = ChatOpenAI(temperature=0.5,
            #                  model="gpt-4")
        elif self.model_name == FINE_TUNED_LLAMA2:
            # content_handler = ContentHandler()
            llm = SagemakerEndpoint(
                endpoint_name=self.endpoint_name,
                region_name=self.region_name,
                credentials_profile_name=self.profile_name,
                # content_handler=content_handler
            )
        else:
            raise ValueError(f"LLM is not supported.")
        return llm

    def retrieve_info(self, query):
        similar_response = self.vec_db.similarity_search(query, k=3)
        page_contents_array = [doc.page_content for doc in similar_response]
        return page_contents_array

    def generate_reply_email(
        self,
        message: list,
        email_templates: list = [],
        max_tokens: int = 500,
        num_completions: int = 1,
        temperature: float = 0.9,
        top_p: float = 0.9,
    ) -> str:
        prompt = generate_out_reach_prompt(email_templates, max_tokens)
        if self.model_name == FINE_TUNED_GPT_35:
            print("calling gpt-3.5-turbo")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                n=num_completions,
            )
            return response["choices"][0]["message"]["content"]
        if self.model_name == FINE_TUNED_GPT_4:
            print("calling gpt-4")
            best_practice = self.retrieve_info(message)
            response = self.llm.run(message=message, best_practice=best_practice)
            return response
            # response = openai.ChatCompletion.create(
            #     model="gpt-4-1106-preview",
            #     messages=prompt,
            #     max_tokens=max_tokens,
            #     temperature=temperature,
            #     n=num_completions,
            # )
            # return response["choices"][0]["message"]["content"]
        if self.model_name == FINE_TUNED_LLAMA2:
            print("calling llama2")
            payload = {
                "inputs": [prompt],
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "top_p": top_p,
                    "temperature": temperature,
                },
            }
            response = self.sagemaker_client.invoke_llama2_endpoint(payload)
            return response[0]["generation"]["content"]
        return None
