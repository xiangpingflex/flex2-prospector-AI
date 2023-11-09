import os
import pandas as pd
import streamlit as st
import openai
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS

from assistant.common.constant import (
    FINE_TUNED_GPT_35,
    FINE_TUNED_LLAMA2,
    FINE_TUNED_GPT_4,
)
from assistant.data_handler.data_handler import DataHandler
from assistant.model.outreach_model import OutReachLLM
from model.sequence_model import SequenceModel
import time

load_dotenv()
openai.api_key = os.environ.get("OPEN-API-KEY")

outreach_models = [FINE_TUNED_GPT_35, FINE_TUNED_GPT_4, FINE_TUNED_LLAMA2]
data_handler = DataHandler(lead_info_path="./resource/lead_info.csv")
sequence_model = SequenceModel(
    profile_name=os.environ.get("AWS_PROFILE_NAME"),
    region_name=os.environ.get("SEQUENCE_MODEL_REGION"),
    endpoint_name=os.environ.get("SEQUENCE_MODEL_ENDPOINT"),
    category_encode_map_path="./resource/category_encode_map.pkl",
    reverse_category_encode_map_path="./resource/reverse_category_encode_map.pkl",
    email_template_path="./resource/email_template.json",
)

# loader = JSONLoader(
#     file_path="./resource/flex_message.jsonl",
#     jq_schema='"question: "+.question + " answer: " +.answer',
#     json_lines=True,
# )
# documents = loader.load()
# embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPEN-API-KEY"))
# db = FAISS.from_documents(documents, embeddings)


def get_lead_info(index: int):
    if index >= data_handler.lead_info.shape[0]:
        index = data_handler.lead_info.shape[0]
    return data_handler.lead_info.iloc[index : index + 1]


def session_init():
    st.session_state["persist"] = {}
    st.session_state["step"] = 0
    st.session_state["top_resource"] = {}


# def retrieve_info(query):
#     similar_response = db.similarity_search(query, k=3)
#     page_contents_array = [doc.page_content for doc in similar_response]
#     return page_contents_array

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPEN-API-KEY"),
    temperature=0.9,
    # max_tokens=100,
    model="gpt-4-1106-preview",
)

# prompt = PromptTemplate(input_variables=["message", "best_practice"], template=template)
# chain = LLMChain(llm=llm, prompt=prompt)
#
#
# # 4. Retrieval augmented generation
# def generate_response(message):
#     best_practice = retrieve_info(message)
#     response = chain.run(message=message, best_practice=best_practice)
#     return response


if "persist" not in st.session_state:
    st.session_state["persist"] = {}
if "step" not in st.session_state:
    st.session_state["step"] = 0
if "top_resource" not in st.session_state:
    st.session_state["top_resource"] = {}

st.sidebar.title("Deal Selection")
deal_slider_value = st.sidebar.slider("Select a deal", 0, 100, 1, key="deal_selection")
lead_info_case = get_lead_info(deal_slider_value)

company_name = lead_info_case.iloc[0]["company_name"]
contact_first_name = lead_info_case.iloc[0]["contact_first_name"]
contact_last_name = lead_info_case.iloc[0]["contact_last_name"]
contact_job_title = lead_info_case.iloc[0]["contact_job_title"]
company_protfolio_type = lead_info_case.iloc[0]["company_protfolio_type"]
company_protfolio_subtype = lead_info_case.iloc[0]["company_protfolio_subtype"]
company_segment = lead_info_case.iloc[0]["company_segment"]
company_state = lead_info_case.iloc[0]["company_state"]
contact_state = lead_info_case.iloc[0]["contact_state"]
company_annual_revenue = lead_info_case.iloc[0]["company_annual_revenue"]
company_founded_year = lead_info_case.iloc[0]["company_founded_year"]
company_units = lead_info_case.iloc[0]["company_units"]

st.sidebar.markdown("### Lead Information")

st.sidebar.markdown(f"##### Company Name: {company_name}")
st.sidebar.markdown(f"##### Contact Name: {contact_first_name} {contact_first_name}")
st.sidebar.markdown(f"##### Contact Job Title: {contact_job_title}")
st.sidebar.markdown(f"##### Company Type: {company_protfolio_type}")
st.sidebar.markdown(f"##### Company SubType: {company_protfolio_subtype}")
st.sidebar.markdown(f"##### Company Segment: {company_segment}")
st.sidebar.markdown(f"##### Company State: {contact_state}")
st.sidebar.markdown(f"##### Contact State: {contact_state}")
st.sidebar.markdown(
    f"##### Company Annual Revenue: {int(company_annual_revenue / 100) if not pd.isna(company_annual_revenue) else None}"
)
st.sidebar.markdown(
    f"##### Company Founded Year: {int(company_founded_year) if not pd.isna(company_founded_year) else None}"
)
st.sidebar.markdown(f"##### Company Units: {company_units}")

reset_button = st.sidebar.button("reset")
if reset_button:
    session_init()
st.title("Sales Assistant")

st.markdown("<hr style='border: 0.5px solid white;'>", unsafe_allow_html=True)

st.markdown(f"#### System Parameters Configuration:")
knowledge_base_model = ["lightgbm-based recommendation model"]
kb_model_choice = st.selectbox("Model for Knowledge Mining:", knowledge_base_model)
top_n_seq = st.slider(
    "Top n Knowledge Seeds", min_value=1, max_value=10, value=2, step=1
)
outreach_model_choice = st.selectbox("LLM Model for Email Generation:", outreach_models)
max_length = st.slider("max token", min_value=50, max_value=500, value=350, step=50)
temperature = st.slider(
    "temperature", min_value=0.1, max_value=1.0, value=0.9, step=0.1
)
top_p = st.slider("top_p", min_value=0.1, max_value=1.0, value=0.9, step=0.1)
max_step = 10
step = 1

outreach_model = OutReachLLM(
    model_name=outreach_model_choice,
    profile_name=os.environ.get("AWS_PROFILE_NAME"),
    region_name=os.environ.get("OUTREACH_MODEL_REGION"),
    endpoint_name=os.environ.get("OUTREACH_MODEL_ENDPOINT"),
)


def add_new_row(new_key, new_value, with_response=True):
    st.info(new_value)
    if with_response:
        st.text_area("#### Enter email response content", key=new_key)
        st.session_state["persist"][new_key] = (new_value, new_key)
    else:
        st.session_state["persist"][new_key] = (new_value, None)


email_gen = st.button("Generate Email", key=f"email_generation_button")

if email_gen:
    email_gen = True
    with st.spinner("Generating..."):
        if not st.session_state["top_resource"]:
            st.session_state[
                "top_resource"
            ] = sequence_model.get_top_sequence_email_template(
                top_n_seq, lead_info_case
            )
            st.session_state["persist"]["km_p"] = ("Knowledge mining is done!", None)
            st.session_state["max_step"] = max(
                [len(v) for k, v in st.session_state["top_resource"].items()]
            )
            st.session_state["persist"]["max_step_p"] = (
                f"#### Generate {st.session_state['max_step']}-step Sequence Plan",
                None,
            )

        session_state_key_sorted_response = sorted(
            [k for k in st.session_state["persist"].keys() if "email_" in k],
            reverse=True,
        )

        cur_response = (
            ""
            if not session_state_key_sorted_response
            else st.session_state.__getattr__(
                st.session_state["persist"][session_state_key_sorted_response[0]][1]
            )
        )

        if (
            st.session_state["step"] < st.session_state["max_step"]
            and cur_response == ""
        ):
            email_template_list = [
                v[st.session_state["step"]]["template_content"]
                for k, v in st.session_state["top_resource"].items()
                if st.session_state["step"] < len(v)
            ]
            outreach_email = outreach_model.generate_outreach_email(
                email_template_list, max_tokens=max_length
            )

            timestamp_in_seconds = time.time()
            new_key_email = "email_" + str(timestamp_in_seconds)
            st.session_state["persist"][new_key_email] = (
                f"#### Sequence Step {st.session_state['step'] + 1}: \n"
                + outreach_email,
                None,
            )
            st.session_state["step"] += 1
        elif cur_response != "" and cur_response != "meeting":
            timestamp_in_seconds = time.time()
            new_key_email = "email_" + str(timestamp_in_seconds)
            reply_email = "This is a replay email"
            st.session_state["persist"][new_key_email] = (
                f"#### Replay Email: \n" + reply_email,
                None,
            )
        elif cur_response == "meeting":
            st.success("### Meeting Scheduled! \n Stop the sequence plan.")
        else:
            st.error("### All steps are done without any response!")
        session_state_key_sorted = ["max_step_p"] + sorted(
            [k for k in st.session_state["persist"].keys() if "email_" in k],
            reverse=True,
        )
        for key in session_state_key_sorted:
            prev_email_content = st.session_state["persist"][key][0]
            add_new_row(key, prev_email_content, key != "max_step_p")
