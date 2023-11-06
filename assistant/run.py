import os
import pandas as pd
import streamlit as st
import openai
from dotenv import load_dotenv

from assistant.common.constant import FINE_TUNED_GPT, FINE_TUNED_LLAMA2
from assistant.model.llm import OutReachLLM
from model.sequence_model import SequenceModel
from data_handler import DataHandler
import time

load_dotenv()
openai.api_key = os.environ.get("OPEN-API-KEY")

# List of models
outreach_models = [FINE_TUNED_GPT, FINE_TUNED_LLAMA2]
data_handler = DataHandler(
    lead_info_path="/Users/xiangpingbu/Documents/projects/flex2-prospector-AI/resource/lead_info.csv"
)
sequence_model = SequenceModel(
    profile_name=os.environ.get("AWS_PROFILE_NAME"),
    region_name=os.environ.get("SEQUENCE_MODEL_REGION"),
    endpoint_name=os.environ.get("SEQUENCE_MODEL_ENDPOINT"),
    category_encode_map_path="/Users/xiangpingbu/Documents/projects/flex2-prospector-AI/resource/category_encode_map.pkl",
    reverse_category_encode_map_path="/Users/xiangpingbu/Documents/projects/flex2-prospector-AI/resource/reverse_category_encode_map.pkl",
    email_template_path="/Users/xiangpingbu/Documents/projects/flex2-prospector-AI/resource/email_template.json",
)

outreach_model = OutReachLLM(
    model_name=FINE_TUNED_GPT
    # profile_name=os.environ.get("AWS_PROFILE_NAME"),
    # region_name=os.environ.get("OUTREACH_MODEL_REGION"),
    # endpoint_name=os.environ.get("OUTREACH_MODEL_ENDPOINT"),
)


def get_lead_info(index: int):
    if index >= data_handler.lead_info.shape[0]:
        index = data_handler.lead_info.shape[0]
    return data_handler.lead_info.iloc[index : index + 1]


# st.markdown("## Email to Respond")
# email_to_respond = st.text_area("Enter your email to respond", height=200)
#
# st.markdown("## Email Response Content")
# response_content = st.text_input("Enter your email response content")


# if st.button("Generate"):
#     with st.spinner("Generating..."):
#         # Use the chosen model
#         if model_choice == "GPT3.5":
#             body = generate_email_with_gpt3_5(
#                 response_content, email_to_respond, max_length
#             )
#     st.markdown("## Generated Email Body")
#     st.write(body)

st.sidebar.title("Deal Selection")
#
# # Add elements to the sidebar
# option = st.sidebar.selectbox("Select an Option", ["Option 1", "Option 2", "Option 3"])
slider_value = st.sidebar.slider("Select a deal", 0, 100, 1)
lead_info_case = get_lead_info(slider_value)

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

import random

# Initialize session state
if "persist" not in st.session_state:
    st.session_state["persist"] = {}
if "step" not in st.session_state:
    st.session_state["step"] = 0


def add_new_row(new_key, new_value, is_markdown=False):
    if is_markdown:
        st.session_state["persist"][new_key] = (
            new_value,
            st.markdown(new_value, key=new_key),
        )
    else:
        st.session_state["persist"][new_key] = (new_value, st.write(new_value))


def page_refresh(state_key, item_key, item_value):
    st.session_state[state_key][item_key] = (item_value, st.write(item_value))


email_gen = st.button("Generate Email", key=f"email_generation_button")

if email_gen:
    with st.spinner("Generating..."):
        if "top_resource" not in st.session_state:
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
            # add_new_row('km_key', "Knowledge mining is done!")
            # st.write("Knowledge mining is done!")
        if st.session_state["step"] < st.session_state["max_step"]:
            email_template_list = [
                v[st.session_state["step"]]["template_content"]
                for k, v in st.session_state["top_resource"].items()
                if st.session_state["step"] < len(v)
            ]
            # outreach_email = outreach_model.generate_outreach_email(email_template_list,
            #                                                             max_tokens=max_length)
            outreach_email = f"test content for step {st.session_state['step'] + 1}"

            timestamp_in_seconds = time.time()
            new_key_email = "email_" + str(timestamp_in_seconds)
            st.session_state["persist"][new_key_email] = (
                f"#### Sequence Step {st.session_state['step'] + 1}: \n"
                + outreach_email,
                None,
            )
            st.session_state["step"] += 1
        else:
            st.markdown("### All steps are done!")
        session_state_key_sorted = ["max_step_p"] + sorted(
            [k for k in st.session_state["persist"].keys() if "email_" in k],
            reverse=True,
        )
        for key in session_state_key_sorted:
            prev_email_content = st.session_state["persist"][key][0]
            add_new_row(key, prev_email_content)

# if st.button("Generate Email Sequence", key=f"sequence_generation"):
#     with st.spinner("Generating..."):
#         top_resource = sequence_model.get_top_sequence_email_template(top_n_seq, lead_info_case)
#     st.write("Knowledge mining is done!")
#     st.write(f"Outreach step: {step}: ")
#     email_template_list = [v[step]['template_content'] for k, v in top_resource.items() if step < len(v)]
#     outreach_email = outreach_model.generate_outreach_email(email_template_list,
#                                                             max_tokens=max_length)
#     st.write(outreach_email)
#     st.markdown("<hr style='border: 0.5px solid white;'>", unsafe_allow_html=True)
#     response_content = st.text_input("Enter your email response content")
#     print(response_content)
#     if not response_content:
#         step += 1
#         if st.button("Generate Response", key=f"sequence_generation_{step}"):
#             with st.spinner("Generating..."):
#                 email_template_list = [v[step]['template_content'] for k, v in top_resource.items() if step < len(v)]
#                 # outreach_email = outreach_model.generate_outreach_email(email_template_list,
#                 #                                                         max_tokens=max_length)
#                 outreach_email = "text_email"
#                 st.write(outreach_email)
#     else:
#         st.write("Done")
# Create two buttons and place them at the bottom using the placeholder
# button1 = st.button("Button 1")
# stepx= 0
# if button1:
#     if stepx == 0:
#         print(stepx)
#         stepx += 1
#         st.write("Clicked!")
#
#     else:
#         print(stepx)
#         st.write("Clicked again!")

# st.markdown("## Generation Parameters")
# max_length = st.slider("Max length", min_value=50, max_value=500, value=300, step=50)
#
# # Create a selectbox for the models
# outreach_model_choice = st.selectbox("Choose a LLM model for email generation:", outreach_models)
#
# total_step = max([len(x) for x in top_resource.values()])
# for step in range(total_step):
#     st.markdown(f"### Step {step+1}")
#     if st.button("generate email", key=f"generate_first_out_{step}"):
#         with st.spinner("Generating..."):
#             email_template_list = [v[step]['template_content'] for k, v in top_resource.items() if step < len(v)]
#             outreach_email = outreach_model.generate_outreach_email(email_template_list,
#                                                                     max_tokens=max_length)
#         st.markdown("### Generated Email Body")
#         st.write(outreach_email)
