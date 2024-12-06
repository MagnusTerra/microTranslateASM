from transformers import AutoModelForCausalLM, AutoTokenizer

def translate_to_asm(code: str):
    """
    Translates code from Python, Java, C++, or JavaScript to assembly language (ASM) using Qwen model.
    
    Args:
        code (str): The code to translate.
        language (str): The programming language of the input code ('Python', 'Java', 'C++', or 'JavaScript').
    
    Returns:
        str: The ASM translation or an error message if the language is not supported.
    """
    supported_languages = ["Python", "Java", "C++", "JavaScript"]
    if language not in supported_languages:
        return "Error: Language not supported. Please use Python, Java, C++, or JavaScript."

    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define the messages
    system_message = {
        "role": "system",
        "content": ("You are an AI assistant specializing in translating Python, Java, C++, "
                    "and JavaScript code into ASM assembly language.")
    }
    user_message = {
        "role": "user",
        "content": f"Translate the following {language} code to ASM:\n\n{code}"
    }
    messages = [system_message, user_message]

    # Tokenize the messages
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate the response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode and return the response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response
