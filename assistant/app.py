import streamlit as st

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
