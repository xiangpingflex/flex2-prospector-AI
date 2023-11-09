def generate_out_reach_prompt(template_list: [], max_length: int = 200):
    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Here are some sales outreach emails examples"
            + "\n".join(template_list)
            + "\n"
            + "generate only one professional sales outreach email using the information from the examples above and always ask to schedule a meeting with the prospect politely and put a google calender link. "
            + f"The generated email needs to be under {max_length - 50} words",
        },
    ]
    return prompt
