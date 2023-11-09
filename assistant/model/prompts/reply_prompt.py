def generate_reply_prompt(message: str, best_seeds: []) -> []:
    # template = f"""
    # You are a world class business development representative.
    # I will share a prospect's message with you and you will give me the best answer that
    # I should send to this prospect based on past best practies,
    # and you will follow ALL of the rules below:
    #
    # 1/ Response should be very similar or even identical to the past best practies,
    # in terms of length, ton of voice, logical arguments and other details
    #
    # 2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message
    #
    # Below is a message I received from the prospect:
    # {message}
    #
    # Here is a list of best practies of how we normally respond to prospect in similar scenarios: """ \
    #            + "\n".join(best_seeds) \
    #            + """
    # \nPlease write the best response in an email format that I should send to this prospect, always ask to schedule a meeting
    # with the prospect politely.
    # """
    prompt = [
        {
            "role": "system",
            "content": "You are a world class business development representative",
        },
        {
            "role": "user",
            "content": """I will share a prospect's message with you and you will give me the best answer that 
            I should send to this prospect based on past best practices, and you will follow ALL of the rules below:
                1/ Response should be very similar or even identical to the past best practices, 
                in terms of length, ton of voice, logical arguments and other details
                2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message

            Here is a list of best practies of how we normally respond to prospect in similar scenarios:\n"""
            + "\n".join(best_seeds)
            + "\n"
            + f"""Below is a message I received from the prospect:
            {message}
            Please write the best response in an email format that I should send to this prospect, always ask to 
            schedule a meeting an demo with the prospect politely.
            """,
        },
    ]
    return prompt
