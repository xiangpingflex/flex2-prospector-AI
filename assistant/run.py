import os
import pandas as pd
import streamlit as st
import openai
from dotenv import load_dotenv
from model.sequence_model import SequenceModel
from data_handler import DataHandler

print(os.getcwd())
load_dotenv()
openai.api_key = os.environ.get("OPEN-API-KEY")

# List of models
models = ["GPT3.5"]
data_handler = DataHandler(
    lead_info_path="/Users/xiangpingbu/Documents/projects/flex2-prospector-AI/resource/lead_info.csv"
)
sequence_model = SequenceModel(
    profile_name=os.environ.get("AWS_PROFILE_NAME"),
    region_name=os.environ.get("SEQUENCE_MODEL_REGION"),
    endpoint_name=os.environ.get("SEQUENCE_MODEL_ENDPOINT"),
    category_map_path="/Users/xiangpingbu/Documents/projects/flex2-prospector-AI/resource/revert_class_map.pkl",
)


def get_lead_info(index: int):
    if index >= data_handler.lead_info.shape[0]:
        index = data_handler.lead_info.shape[0]
    return data_handler.lead_info.iloc[index : index + 1]


def generate_email_with_gpt3_5(response_content, email_to_respond, max_length):
    # Define the conversation history
    message_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Here's an email I received: '{}".format(email_to_respond),
        },
        {
            "role": "assistant",
            "content": "Sure, I can help you draft a response with a max length of {} chars".format(
                max_length
            ),
        },
        {"role": "user", "content": "{}".format(response_content)},
    ]

    # Generate a response using GPT-3.5-Turbo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history,
    )
    return response["choices"][0]["message"]["content"]


st.title("Sales Assistant LLM")

st.markdown("## Email to Respond")
email_to_respond = st.text_area("Enter your email to respond", height=200)

st.markdown("## Email Response Content")
response_content = st.text_input("Enter your email response content")

st.markdown("## Generation Parameters")
max_length = st.slider("Max length", min_value=50, max_value=1000, value=500, step=50)

# Create a selectbox for the models
model_choice = st.selectbox("Choose a model:", models)

if st.button("Generate"):
    with st.spinner("Generating..."):
        # Use the chosen model
        if model_choice == "GPT3.5":
            body = generate_email_with_gpt3_5(
                response_content, email_to_respond, max_length
            )
    st.markdown("## Generated Email Body")
    st.write(body)

st.sidebar.title("Deal Selection")
#
# # Add elements to the sidebar
# option = st.sidebar.selectbox("Select an Option", ["Option 1", "Option 2", "Option 3"])
slider_value = st.sidebar.slider("Select a deal", 0, 100, 50)
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

x = sequence_model.predict_prob(lead_info_case)
y = sequence_model.predict_score(lead_info_case)

print(x)
print(y)
