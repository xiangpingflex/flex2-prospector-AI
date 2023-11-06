def generate_out_reach_prompt(template_list: [], max_length: int = 200):
    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Here are some sales outreach emails examples"
            + "\n".join(template_list)
            + "\n"
            + "generate only one professional sales outreach email based on the examples above"
            + f"The generated email should be about {max_length-50} words, only contain the useful information from the examples above, and be written in a professional tone.",
        },
    ]
    return prompt
