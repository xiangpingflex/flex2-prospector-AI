import streamlit as st

# Create a widget with a specific key
user_input = st.text_input("Enter something:", key="my_input")
# Get the value of the widget by its key
if st.button("Get Widget Value"):
    print(st.session_state.__getattr__("my_input"))
    widget_value = st.session_state.my_input
    st.write("Widget value:", widget_value)
