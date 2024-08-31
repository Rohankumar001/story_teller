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

# Function to load image and convert to base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"File not found: {image_path}")
        return None  # You can return a default base64 image string or None

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

# Streamlit app
def main():
    st.title("Story Teller")
    st.write("Enter a story prompt to generate a complete story.")

    # Path to the background image
    image_path = "D:/Desktop/MCA 3rd sem/story generator/background.jpg"  # Update this to the correct path
    image_base64 = get_base64_image(image_path)

    # Adding background image and custom styles using CSS
    if image_base64:
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{image_base64}");
                background-size: cover;
                font-family: 'Helvetica', sans-serif;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    prompt = st.text_area("Enter your story prompt here...", height=100)

    if st.button("Generate Story"):
        if prompt.strip() == "":
            st.warning("Please enter a prompt to generate a story.")
        else:
            with st.spinner("Generating story..."):
                story = generate_story(prompt)
                st.subheader("Generated Story")
                st.write(story)

if __name__ == "__main__":
    main()
