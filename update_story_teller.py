import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import base64

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can use 'gpt2-medium', 'gpt2-large', etc.
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set device to GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to ensure the story ends with a complete sentence
def ensure_complete_sentence(text):
    sentences = text.split('. ')
    if sentences[-1] == '':
        return text
    if len(sentences) > 1:
        return '. '.join(sentences[:-1]) + '.'
    return text

# Generate story function
def generate_story(prompt):
    try:
        # Encode the prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate story
        outputs = model.generate(
            inputs,
            max_length=1000,  # Set a high value to allow for longer stories
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            early_stopping=True
        )

        # Decode the generated text
        story = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Ensure the story ends with a complete sentence
        complete_story = ensure_complete_sentence(story)
        
        return complete_story
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to load image and convert to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Streamlit app
def main():
    # Path to the background image
    image_path = "book.jpg"
    image_base64 = get_base64_image(image_path)

    # Adding background image and custom styles using CSS
    st.markdown(
        f"""
        <style>
        /* Background Image */
        .stApp {{
            background-image: url("data:image/jpg;base64,{image_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #ffffff;
        }}

        /* Overlay for better readability */
        .main-content {{
            background-color: rgba(0, 0, 0, 0.6);
            padding: 30px;
            border-radius: 15px;
            max-width: 800px;
            margin: 50px auto;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.3);
        }}

        /* Title Styling */
        .main-content h1 {{
            color: #FFD700;
            text-align: center;
            font-size: 3em;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }}

        /* Subtitle Styling */
        .main-content p {{
            color: #f0f0f0;
            font-size: 1.2em;
            text-align: center;
            margin-bottom: 20px;
        }}

        /* Text Area Styling */
        .stTextArea textarea {{
            background: rgba(255, 255, 255, 0.9) !important;
            color: #000000 !important;
            font-size: 1.1em !important;
            padding: 15px !important;
            border-radius: 10px !important;
            border: none !important;
            box-shadow: inset 0px 2px 5px rgba(0, 0, 0, 0.2) !important;
        }}

        /* Button Styling */
        .stButton button {{
            background-color: #1E90FF;
            color: white;
            padding: 12px 24px;
            font-size: 1.1em;
            border-radius: 8px;
            border: none;
            transition: background-color 0.3s;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
        }}
        .stButton button:hover {{
            background-color: #1C86EE;
        }}

        /* Generated Story Styling */
        .generated-story {{
            background-color: rgba(255, 255, 255, 0.9);
            color: #000000;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 1.1em;
            line-height: 1.6em;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }}

        /* Scrollbar Styling for Story Box */
        .generated-story::-webkit-scrollbar {{
            width: 8px;
        }}
        .generated-story::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 10px;
        }}
        .generated-story::-webkit-scrollbar-thumb {{
            background-color: #888;
            border-radius: 10px;
        }}
        .generated-story::-webkit-scrollbar-thumb:hover {{
            background: #555;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Main content area
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.title("Story Teller")
    st.write("Enter a story prompt to generate a complete story.")

    prompt = st.text_area("Enter your story prompt here... Example prompt - Once upon a time there lived a ghost", height=200)

    if st.button("Generate Story"):
        if prompt.strip() == "":
            st.warning("Please enter a prompt to generate a story.")
        else:
            with st.spinner("Generating story..."):
                story = generate_story(prompt)
                st.subheader("Generated Story")
                st.markdown(f'<div class="generated-story">{story}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
