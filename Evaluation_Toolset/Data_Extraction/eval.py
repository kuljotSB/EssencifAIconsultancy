import streamlit as st
import os
from PyPDF2 import PdfReader, PdfWriter
import base64
import re
import os
from dotenv import load_dotenv
import streamlit.components.v1 as components
from IPython.display import Image, display
from mistralai import Mistral
from openai import OpenAI
import logging
import pandas as pd
import os
from datetime import datetime
import csv
import pandas as pd
import os
from datetime import datetime
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaLLM
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2 import PdfReader, PdfWriter
import os
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError


load_dotenv()
st.set_page_config(page_title="Evaluation Toolset", page_icon=":bar_chart:", layout="wide")
st.title("Evaluation Toolset")

# Initialize session state for buttons
# Initialize session state for buttons
if "save_extracted_data_clicked" not in st.session_state:
    st.session_state.save_extracted_data_clicked = False

if "reset_csv_clicked" not in st.session_state:
    st.session_state.reset_csv_clicked = False

if "reset_folders_clicked" not in st.session_state:
    st.session_state.reset_folders_clicked = False
    
if "process_pdf_clicked" not in st.session_state:
    st.session_state.process_pdf_clicked = False

# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
mistral_ocr_key = os.getenv("MISTRAL_OCR_KEY")
mistral_local_LLM = os.getenv("MISTRAL_LOCAL_LLM")
ollama_model_seven = os.getenv("OLLAMA_MODEL_SEVEN")
ollama_model_eight = os.getenv("OLLAMA_MODEL_EIGHT")
ollama_model_nine = os.getenv("OLLAMA_MODEL_NINE")
ollama_model_ten = os.getenv("OLLAMA_MODEL_TEN")
poppler_path = r"C:\Users\HP Victus\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"  # Update this with the correct path to your Poppler installation - upto the bin folder


#sample output

sample_output = f"""{{
"Name": "John Doe",
"Address": "1234 Elm Street, Springfield, IL 62704, USA",
"Phone Number": "+1-312-555-7890",
"Email Address": "john.doe@example.com",
"Date of Birth": "1990-07-15",
"Nationality": "American",
"Passport Number": "X12345678",
"Visa Type": "Tourist",
"Visa Expiry Date": "2026-09-30"
}}"""


#defining text variables to store the data extracted from the PDF using different models
extracted_data_gpt4_image_only = ""
extracted_data_gpt4_image_and_markdown = ""
extracted_data_gpt4_markdown_only = ""
extracted_data_mistral_image_only = ""
extracted_data_mistral_image_and_markdown = ""
extracted_data_mistral_markdown_only = ""
extracted_data_ollama_model_seven_image_only = ""
extracted_data_ollama_model_seven_image_and_markdown = ""
extracted_data_ollama_model_seven_markdown_only = ""
extracted_data_ollama_model_eight_image_only = ""
extracted_data_ollama_model_eight_image_and_markdown = ""
extracted_data_ollama_model_eight_markdown_only = ""
extracted_data_ollama_model_nine_image_only = ""
extracted_data_ollama_model_nine_image_and_markdown = ""
extracted_data_ollama_model_nine_markdown_only = ""
extracted_data_ollama_model_ten_image_only = ""
extracted_data_ollama_model_ten_image_and_markdown = ""
extracted_data_ollama_model_ten_markdown_only = ""


# defining the prompts for the different pipelines
data_extraction_gpt4_image_only_user_prompt = f""" I have provided you with the image of the current page of the invoice
                 Please extract data from the image of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The data extracted from the previous pages of the invoice is as follows: """ 
                 
data_extraction_gpt4_markdown_only_user_prompt = f""" I have provided you with the markdown of the current page of the invoice extracted using API call to Mistral OCR
                 Please extract data from the markdown of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The markdown extracted from the current page of the invoice is as follows: """

data_extraction_gpt4_image_and_markdown_user_prompt = f"""I have provided you with both the image and markdown of the current page of the invoice
                 Please extract data from the image of the current page with the help of the markdown of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The markdown extracted from the current page of the invoice is as follows: """
                 
data_extraction_local_mistral_image_only_user_prompt = f""" I have provided you with the image of the current page of the invoice
                 Please extract data from the image of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The data extracted from the previous pages of the invoice is as follows: """ 
                 
data_extraction_local_mistral_markdown_only_user_prompt = f""" I have provided you with the markdown of the current page of the invoice extracted using API call to Mistral OCR
                 Please extract data from the markdown of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The markdown extracted from the current page of the invoice is as follows: """
                 
data_extraction_local_mistral_image_and_markdown_user_prompt = f"""I have provided you with both the image and markdown of the current page of the invoice
                 Please extract data from the image of the current page with the help of the markdown of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The markdown extracted from the current page of the invoice is as follows: """
                 
data_extraction_local_ollama_model_seven_image_only_user_prompt = f""" I have provided you with the image of the current page of the invoice
                 Please extract data from the image of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The data extracted from the previous pages of the invoice is as follows: """

data_extraction_local_ollama_model_seven_markdown_only_user_prompt = f""" I have provided you with the markdown of the current page of the invoice extracted using API call to Mistral OCR
                 Please extract data from the markdown of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The markdown extracted from the current page of the invoice is as follows: """

data_extraction_local_ollama_model_seven_image_and_markdown_user_prompt = f"""I have provided you with both the image and markdown of the current page of the invoice
                 Please extract data from the image of the current page with the help of the markdown of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The markdown extracted from the current page of the invoice is as follows: """
                 
data_extraction_local_ollama_model_eight_image_only_user_prompt = f""" I have provided you with the image of the current page of the invoice
                 Please extract data from the image of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The data extracted from the previous pages of the invoice is as follows: """

data_extraction_local_ollama_model_eight_markdown_only_user_prompt = f""" I have provided you with the markdown of the current page of the invoice extracted using API call to Mistral OCR
                 Please extract data from the markdown of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The markdown extracted from the current page of the invoice is as follows: """

data_extraction_local_ollama_model_eight_image_and_markdown_user_prompt = f"""I have provided you with both the image and markdown of the current page of the invoice
                 Please extract data from the image of the current page with the help of the markdown of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The markdown extracted from the current page of the invoice is as follows: """

data_extraction_local_ollama_model_nine_image_only_user_prompt = f""" I have provided you with the image of the current page of the invoice
                 Please extract data from the image of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The data extracted from the previous pages of the invoice is as follows: """
    
data_extraction_local_ollama_model_nine_markdown_only_user_prompt =  f""" I have provided you with the markdown of the current page of the invoice extracted using API call to Mistral OCR
                 Please extract data from the markdown of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The markdown extracted from the current page of the invoice is as follows: """
                 
data_extraction_local_ollama_model_nine_image_and_markdown_user_prompt = f"""I have provided you with both the image and markdown of the current page of the invoice
                 Please extract data from the image of the current page with the help of the markdown of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The markdown extracted from the current page of the invoice is as follows: """
                 
data_extraction_local_ollama_model_ten_image_only_user_prompt = f""" I have provided you with the image of the current page of the invoice
                 Please extract data from the image of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The data extracted from the previous pages of the invoice is as follows: """
                 
data_extraction_local_ollama_model_ten_markdown_only_user_prompt = f""" I have provided you with the markdown of the current page of the invoice extracted using API call to Mistral OCR
                 Please extract data from the markdown of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The markdown extracted from the current page of the invoice is as follows: """
                
data_extraction_local_ollama_model_ten_image_and_markdown_user_prompt = f"""I have provided you with both the image and markdown of the current page of the invoice
                 Please extract data from the image of the current page with the help of the markdown of the current page based on the data extraction system prompt
                 provided to you and consolidate everything in a structured format as prompted/instructed
                 by the user in the system prompt
                 The markdown extracted from the current page of the invoice is as follows: """
                 
# Giving option via dropdown to select multiple models
st.sidebar.header("Select Models")
model_options = {
    "Mistral Local LLM": mistral_local_LLM,
    "OpenAI GPT-4-Vision-Preview": "gpt-4-vision-preview",
    "Ollama Model Seven": ollama_model_seven,
    "Ollama Model Eight": ollama_model_eight,
    "Ollama Model Nine": ollama_model_nine,
    "Ollama Model Ten": ollama_model_ten
}
selected_models = st.sidebar.multiselect("Select models to use:", list(model_options.keys()))

# Set parameters like temperature, max tokens, etc.
st.sidebar.header("Model Parameters")
st.sidebar.subheader("default tested values are displayed")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)
max_tokens = st.sidebar.slider("Max Tokens", 1, 4096, 2048)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.9)
frequency_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 1.0, 0.0)
presence_penalty = st.sidebar.slider("Presence Penalty", 0.0, 1.0, 0.0)

# Section to select and view prompts from the "prompt_lab" folder
st.header("Select and View Prompts")

# Define the path to the "prompt_lab" folder
prompt_lab_folder = "./prompt_lab"
data_extraction_system_prompt=""

# Ensure the folder exists
if not os.path.exists(prompt_lab_folder):
    st.warning(f"The folder '{prompt_lab_folder}' does not exist. Please create it and add some prompt files.")
else:
    # Get a list of all files in the "prompt_lab" folder
    prompt_files = [f for f in os.listdir(prompt_lab_folder) if os.path.isfile(os.path.join(prompt_lab_folder, f))]

    if prompt_files:
        # Dropdown to select a prompt file
        selected_prompt_file = st.selectbox("Select a prompt file:", prompt_files)

        # Display the content of the selected prompt file
        if selected_prompt_file:
            prompt_file_path = os.path.join(prompt_lab_folder, selected_prompt_file)
            with open(prompt_file_path, "r", encoding="utf-8") as file:
                data_extraction_system_prompt = file.read()
                

            st.text_area("Selected Prompt Content", value=data_extraction_system_prompt, height=300)
    else:
        st.warning(f"No prompt files found in the '{prompt_lab_folder}' folder. Please add some prompt files.")

# Providing option to upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_pdf_path = os.path.join("uploaded_invoices", uploaded_file.name)
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded PDF file name
    st.success(f"Uploaded file: {uploaded_file.name}")

    
    


def convert_pdf_to_image(child_pdf_path):
    # Ensure the main images directory exists
    images_output_folder = './images'
    os.makedirs(images_output_folder, exist_ok=True)
    child_pdf_name = os.path.basename(child_pdf_path).split('.')[0]
    print(f"Child PDF Name: {child_pdf_name}")
    
    # Ensure the split PDF pages directory exists
    split_pdf_folder = "split_pdf_pages"
    os.makedirs(split_pdf_folder, exist_ok=True)
    
    print(f"Processing: {child_pdf_path}")

    try:
        # Split the PDF into individual pages and save them in the split_pdf_pages directory
        reader = PdfReader(child_pdf_path)
        for i, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            
            # Save each page as a separate PDF
            split_pdf_path = os.path.join(split_pdf_folder, f"{child_pdf_name}_page_{i + 1}.pdf")
            with open(split_pdf_path, "wb") as output_pdf:
                writer.write(output_pdf)
            
            print(f"Saved split PDF page: {split_pdf_path}")
        
        for child_pdf in os.listdir("split_pdf_pages"):
            if child_pdf.endswith(".pdf"):
                child_pdf_path = os.path.join("split_pdf_pages", child_pdf)
                
                # Convert PDF pages to images
                images = convert_from_path(child_pdf_path, poppler_path=poppler_path)
                
                # Save the images to the images folder
                for i, img in enumerate(images):
                    image_name = f"{os.path.splitext(child_pdf)[0]}_page_{i + 1}.png"
                    image_path = os.path.join(images_output_folder, image_name)
                    img.save(image_path, "PNG")
                    print(f"Saved image: {image_path}")

        print(f"Saved images for {os.path.basename(child_pdf_path)} in {images_output_folder}")
    
    except PDFPageCountError:
        print(f"Error: Unable to get page count for {child_pdf_path}. The file might be corrupted or not a valid PDF.")
    except FileNotFoundError:
        print(f"Error: File {child_pdf_path} not found.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {child_pdf_path}: {e}")

# Creating a function to generate local base64 URL for an image
def generate_base64_url(child_pdf_image_path):
            print(f"Processing image: {child_pdf_image_path}")
            
            # Read the image file in binary mode
            with open(child_pdf_image_path, "rb") as img_file:
                raw_data = img_file.read()
                image_data = base64.b64encode(raw_data).decode("utf-8")
            
            # Determine the image format
            image_format = child_pdf_image_path.split('.')[-1]
            
            # Generate the data URL (optional, for other use cases)
            data_url = f"data:image/{image_format};base64,{image_data}"
            
            # Print the data URL (or save it as needed)
            print(f"Data URL for {child_pdf_image_path}:\n{data_url[:100]}...\n")  # printing full base64 is too long
            
           
            return data_url

# Creating a Function to Encode PDF with Base64
def encode_pdf(pdf_path):
    """Encode the pdf to base64."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None
    
# Function to Generate Markdown from MISTRAL OCR
def generate_markdown_from_mistral_OCR(image_path):

    # Getting the base64 string
    image_base_64_path = generate_base64_url(image_path)

    
    client = Mistral(api_key=mistral_ocr_key)

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": f"{image_base_64_path}" 
        }
    )
    
    
    print(ocr_response.pages[0].markdown)
    return str((ocr_response))

# Creating Function to Call GPT-4 Vision Model
def call_openai_vision_model(image_path, data_extracted_from_previous_pages, user_prompt):
    base64_path = generate_base64_url(image_path)
    # Initialize the OpenAI client
    client = OpenAI(api_key=openai_api_key)
    response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {
            "role": "system",
            "content":f"""{data_extraction_system_prompt}
            If the invoice is a multi-paged invoice you will be provided with the image of the current page
            and data extracted from the previous pages in a consolidated format.
            Your work is to extract the data from the current page and add it to the previous pages data in a
            structured format."""
        },
        {
            "role": "user",
            "content": [
                { "type": "input_text", "text": f"""{data_extraction_gpt4_image_only_user_prompt} \n {data_extracted_from_previous_pages}""" },
                {
                    "type": "input_image",
                    "image_url": f"{base64_path}",
                },
            ],
        }
    ],
    max_output_tokens=max_tokens,
    temperature=temperature,
    top_p=top_p,
    
    
    )
    
    

    print(response.output_text)
    
    return str(response.output_text)

def call_local_ollama_model(model_name, image_path, data_extracted_from_previous_page, user_prompt):
    base64_path = generate_base64_url(image_path)
    
    # Initialize the ollama client
    llm = OllamaLLM(
        model = model_name,
        max_tokens = max_tokens,
        temperature = temperature,
        top_p = top_p,
        presence_penalty = presence_penalty,
        frequency_penalty = frequency_penalty,
    )
    
    llm_with_image_context = llm.bind(images=[base64_path])
    
    response = llm_with_image_context.invoke([
        SystemMessage(content=data_extraction_system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    print("Response from {} Local LLM: {}".format(model_name, response))
    
    return response

# Define the path for the CSV file
csv_file_path = "eval_results.csv"

# Function to save extracted data to a CSV file
def save_extracted_data_to_csv(system_prompt, 
                               invoice, 
                               gpt4_image_only_data, 
                               gpt4_markdown_only_data, 
                               gpt4_image_and_markdown_data,
                               mistral_image_only_data,
                               mistral_markdown_only_data,
                               mistral_image_and_markdown_data,
                               model_seven_image_only_data,
                               model_seven_markdown_only_data,
                               model_seven_image_and_markdown_data,
                               model_eight_image_only_data,
                               model_eight_markdown_only_data,
                               model_eight_image_and_markdown_data,
                               model_nine_image_only_data,
                               model_nine_markdown_only_data,
                               model_nine_image_and_markdown_data,
                               model_ten_image_only_data,
                               model_ten_markdown_only_data,
                               model_ten_image_and_markdown_data):
    
    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_file_path)

    # Define the headers for the CSV file
    headers = [
        "Timestamp", 
        "System Prompt", 
        "Invoice", 
        "GPT-4 Image Input Only", 
        "GPT-4 Markdown Input Only", 
        "GPT-4 Image and Markdown Input", 
        "MISTRAL Image Input Only", 
        "MISTRAL Markdown Input Only", 
        "MISTRAL Image and Markdown Input", 
        "Ollama Model Seven Image Input Only", 
        "Ollama Model Seven Markdown Input Only", 
        "Ollama Model Seven Image and Markdown Input", 
        "Ollama Model Eight Image Input Only", 
        "Ollama Model Eight Markdown Input Only", 
        "Ollama Model Eight Image and Markdown Input", 
        "Ollama Model Nine Image Input Only", 
        "Ollama Model Nine Markdown Input Only", 
        "Ollama Model Nine Image and Markdown Input", 
        "Ollama Model Ten Image Input Only", 
        "Ollama Model Ten Markdown Input Only", 
        "Ollama Model Ten Image and Markdown Input"
    ]

    # Open the CSV file in append mode
    with open(csv_file_path, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        # Write the headers if the file does not exist
        if not os.path.exists(csv_file_path) or os.stat(csv_file_path).st_size == 0:
            writer.writerow(headers)

        # Write the extracted data to the CSV file
        writer.writerow([
                               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp
                               system_prompt, 
                               invoice, 
                               gpt4_image_only_data, 
                               gpt4_markdown_only_data, 
                               gpt4_image_and_markdown_data,
                               mistral_image_only_data,
                               mistral_markdown_only_data,
                               mistral_image_and_markdown_data,
                               model_seven_image_only_data,
                               model_seven_markdown_only_data,
                               model_seven_image_and_markdown_data,
                               model_eight_image_only_data,
                               model_eight_markdown_only_data,
                               model_eight_image_and_markdown_data,
                               model_nine_image_only_data,
                               model_nine_markdown_only_data,
                               model_nine_image_and_markdown_data,
                               model_ten_image_only_data,
                               model_ten_markdown_only_data,
                               model_ten_image_and_markdown_data
        ])

    print(f"Extracted data saved to {csv_file_path}")
    
# Function to process a single image file
def process_image_file(gpt_4_vision_flag, 
                       mistral_LLM_flag, model_seven_flag, 
                       model_eight_flag, model_nine_flag, 
                       model_ten_flag, 
                       image_file, 
                       images_folder, 
                       extracted_data_gpt4_image_only, 
                       extracted_data_gpt4_markdown_only, 
                       extracted_data_gpt4_image_and_markdown, 
                       extracted_data_mistral_image_only, 
                       extracted_data_mistral_markdown_only, 
                       extracted_data_mistral_image_and_markdown, 
                       extracted_data_ollama_model_seven_image_only, 
                       extracted_data_ollama_model_seven_markdown_only, 
                       extracted_data_ollama_model_seven_image_and_markdown, 
                       extracted_data_ollama_model_eight_image_only, 
                       extracted_data_ollama_model_eight_markdown_only, 
                       extracted_data_ollama_model_eight_image_and_markdown, 
                       extracted_data_ollama_model_nine_image_only, 
                       extracted_data_ollama_model_nine_markdown_only, 
                       extracted_data_ollama_model_nine_image_and_markdown, 
                       extracted_data_ollama_model_ten_image_only, 
                       extracted_data_ollama_model_ten_markdown_only, 
                       extracted_data_ollama_model_ten_image_and_markdown):
    
    image_path = os.path.join(images_folder, image_file)
    logging.info(f"Processing image: {image_path}")
    print(f"Processing image: {image_path}")
    
    # Generate Markdown from Mistral OCR
    current_page_markdown_text = generate_markdown_from_mistral_OCR(image_path)
    logging.info(f"Current page markdown text: {current_page_markdown_text}")
    print(f"Current page markdown text: {current_page_markdown_text}")
        
    # GPT-4-vision-processing-pipline
    if gpt_4_vision_flag:
        # Process with GPT-4 Vision (Image Only)
        extracted_data_gpt4_image_only += call_openai_vision_model(
            image_path, extracted_data_gpt4_image_only, data_extraction_gpt4_image_only_user_prompt
        )
        logging.info(f"Data extracted GPT-4 image only: {extracted_data_gpt4_image_only}")
        print(f"Data extracted GPT-4 image only: {extracted_data_gpt4_image_only}")


        # Process with GPT-4 Vision (Markdown Only)
        data_extraction_gpt4_markdown_only_user_prompt = f"""{current_page_markdown_text} \n The data extracted from the previous pages of the invoice is as follows: {extracted_data_gpt4_markdown_only} """
        extracted_data_gpt4_markdown_only += call_openai_vision_model(
            image_path, extracted_data_gpt4_markdown_only, data_extraction_gpt4_markdown_only_user_prompt
        )
        logging.info(f"Data extracted GPT-4 markdown only: {extracted_data_gpt4_markdown_only}")
        print(f"Data extracted GPT-4 markdown only: {extracted_data_gpt4_markdown_only}")

        # Process with GPT-4 Vision (Image and Markdown)
        data_extraction_gpt4_image_and_markdown_user_prompt = f"""{current_page_markdown_text} \n The data extracted from the previous pages of the invoice is as follows: {extracted_data_gpt4_image_and_markdown} """
        extracted_data_gpt4_image_and_markdown += call_openai_vision_model(
            image_path, extracted_data_gpt4_image_and_markdown, data_extraction_gpt4_image_and_markdown_user_prompt
        )
        logging.info(f"Data extracted GPT-4 image and markdown: {extracted_data_gpt4_image_and_markdown}")
        print(f"Data extracted GPT-4 image and markdown: {extracted_data_gpt4_image_and_markdown}")
    
    if mistral_LLM_flag:
        # Process with mistral local LLM (Image Only)
        extracted_data_mistral_image_only += call_local_ollama_model(
            mistral_local_LLM, image_path, extracted_data_mistral_image_only, data_extraction_local_mistral_image_only_user_prompt
        )
        logging.info(f"Data extracted Mistral image only: {extracted_data_mistral_image_only}")
        print(f"Data extracted Mistral image only: {extracted_data_mistral_image_only}")
        
        # Process with mistral local LLM (Markdown Only)
        data_extraction_local_mistral_markdown_only_user_prompt = f"""{current_page_markdown_text} \n The data extracted from the previous pages of the invoice is as follows: {extracted_data_mistral_markdown_only} """
        extracted_data_mistral_markdown_only += call_local_ollama_model(
            mistral_local_LLM, image_path, extracted_data_mistral_markdown_only, data_extraction_local_mistral_markdown_only_user_prompt
        )
        logging.info(f"Data extracted Mistral markdown only: {extracted_data_mistral_markdown_only}")
        print(f"Data extracted Mistral markdown only: {extracted_data_mistral_markdown_only}")
        
        # Process with mistral local LLM (Image and Markdown)
        data_extraction_local_mistral_image_and_markdown_user_prompt = f"""{current_page_markdown_text} \n The data extracted from the previous pages of the invoice is as follows: {extracted_data_mistral_image_and_markdown} """
        extracted_data_mistral_image_and_markdown += call_local_ollama_model(
            mistral_local_LLM, image_path, extracted_data_mistral_image_and_markdown, data_extraction_local_mistral_image_and_markdown_user_prompt
        )
        logging.info(f"Data extracted Mistral image and markdown: {extracted_data_mistral_image_and_markdown}")
        print(f"Data extracted Mistral image and markdown: {extracted_data_mistral_image_and_markdown}")
    
    if model_seven_flag:
        # Process with local LLM (Model Seven) (Image Only)
        extracted_data_ollama_model_seven_image_only += call_local_ollama_model(
            ollama_model_seven, image_path, extracted_data_ollama_model_seven_image_only, data_extraction_local_ollama_model_seven_image_only_user_prompt
        )
        logging.info(f"Data extracted Ollama Model Seven image only: {extracted_data_ollama_model_seven_image_only}")
        print(f"Data extracted Ollama Model Seven image only: {extracted_data_ollama_model_seven_image_only}")
        
        # Process with local LLM (Model Seven) (Markdown Only)
        data_extraction_local_ollama_model_seven_markdown_only_user_prompt = f"""{current_page_markdown_text} \n The data extracted from the previous pages of the invoice is as follows: {extracted_data_ollama_model_seven_markdown_only} """
        extracted_data_ollama_model_seven_markdown_only += call_local_ollama_model(
            ollama_model_seven, image_path, extracted_data_ollama_model_seven_markdown_only, data_extraction_local_ollama_model_seven_markdown_only_user_prompt
        )
        logging.info(f"Data extracted Ollama Model Seven markdown only: {extracted_data_ollama_model_seven_markdown_only}")
        print(f"Data extracted Ollama Model Seven markdown only: {extracted_data_ollama_model_seven_markdown_only}")
        
        # Process with local LLM (Model Seven) (Image and Markdown)
        data_extraction_local_ollama_model_seven_image_and_markdown_user_prompt = f"""{current_page_markdown_text} \n The data extracted from the previous pages of the invoice is as follows: {extracted_data_ollama_model_seven_image_and_markdown} """
        extracted_data_ollama_model_seven_image_and_markdown += call_local_ollama_model(
            ollama_model_seven, image_path, extracted_data_ollama_model_seven_image_and_markdown, data_extraction_local_ollama_model_seven_image_and_markdown_user_prompt
        )
        logging.info(f"Data extracted Ollama Model Seven image and markdown: {extracted_data_ollama_model_seven_image_and_markdown}")
        print(f"Data extracted Ollama Model Seven image and markdown: {extracted_data_ollama_model_seven_image_and_markdown}")
    
    if model_eight_flag:
        # Process with local LLM (Model Eight) (Image Only)
        extracted_data_ollama_model_eight_image_only += call_local_ollama_model(
            ollama_model_eight, image_path, extracted_data_ollama_model_eight_image_only, data_extraction_local_ollama_model_eight_image_only_user_prompt
        )
        logging.info(f"Data extracted Ollama Model Eight image only: {extracted_data_ollama_model_eight_image_only}")
        print(f"Data extracted Ollama Model Eight image only: {extracted_data_ollama_model_eight_image_only}")
        
        # Process with local LLM (Model Eight) (Markdown Only)
        data_extraction_local_ollama_model_eight_markdown_only_user_prompt = f"""{current_page_markdown_text} \n The data extracted from the previous pages of the invoice is as follows: {extracted_data_ollama_model_eight_markdown_only} """
        extracted_data_ollama_model_eight_markdown_only += call_local_ollama_model(
            ollama_model_eight, image_path, extracted_data_ollama_model_eight_markdown_only, data_extraction_local_ollama_model_eight_markdown_only_user_prompt
        )
        logging.info(f"Data extracted Ollama Model Eight markdown only: {extracted_data_ollama_model_eight_markdown_only}")
        print(f"Data extracted Ollama Model Eight markdown only: {extracted_data_ollama_model_eight_markdown_only}")
        
        # Process with local LLM (Model Eight) (Image and Markdown)
        data_extraction_local_ollama_model_eight_image_and_markdown_user_prompt = f"""{current_page_markdown_text} \n The data extracted from the previous pages of the invoice is as follows: {extracted_data_ollama_model_eight_image_and_markdown} """
        extracted_data_ollama_model_eight_image_and_markdown += call_local_ollama_model(
            ollama_model_eight, image_path, extracted_data_ollama_model_eight_image_and_markdown, data_extraction_local_ollama_model_eight_image_and_markdown_user_prompt
        )
        logging.info(f"Data extracted Ollama Model Eight image and markdown: {extracted_data_ollama_model_eight_image_and_markdown}")
        print(f"Data extracted Ollama Model Eight image and markdown: {extracted_data_ollama_model_eight_image_and_markdown}")
        
    if model_nine_flag:
        # Process with local LLM (Model Nine) (Image Only)
        extracted_data_ollama_model_nine_image_only += call_local_ollama_model(
            ollama_model_nine, image_path, extracted_data_ollama_model_nine_image_only, data_extraction_local_ollama_model_nine_image_only_user_prompt
        )
        logging.info(f"Data extracted Ollama Model Nine image only: {extracted_data_ollama_model_nine_image_only}")
        print(f"Data extracted Ollama Model Nine image only: {extracted_data_ollama_model_nine_image_only}")
        
        # Process with local LLM (Model Nine) (Markdown Only)
        data_extraction_local_ollama_model_nine_markdown_only_user_prompt = f"""{current_page_markdown_text} \n The data extracted from the previous pages of the invoice is as follows: {extracted_data_ollama_model_nine_markdown_only} """
        extracted_data_ollama_model_nine_markdown_only += call_local_ollama_model(
            ollama_model_nine, image_path, extracted_data_ollama_model_nine_markdown_only, data_extraction_local_ollama_model_nine_markdown_only_user_prompt
        )
        logging.info(f"Data extracted Ollama Model Nine markdown only: {extracted_data_ollama_model_nine_markdown_only}")
        print(f"Data extracted Ollama Model Nine markdown only: {extracted_data_ollama_model_nine_markdown_only}")
        
        # Process with local LLM (Model Nine) (Image and Markdown)
        data_extraction_local_ollama_model_nine_image_and_markdown_user_prompt = f"""{current_page_markdown_text} \n The data extracted from the previous pages of the invoice is as follows: {extracted_data_ollama_model_nine_image_and_markdown} """
        extracted_data_ollama_model_nine_image_and_markdown += call_local_ollama_model(
            ollama_model_nine, image_path, extracted_data_ollama_model_nine_image_and_markdown, data_extraction_local_ollama_model_nine_image_and_markdown_user_prompt
        )
        logging.info(f"Data extracted Ollama Model Nine image and markdown: {extracted_data_ollama_model_nine_image_and_markdown}")
        print(f"Data extracted Ollama Model Nine image and markdown: {extracted_data_ollama_model_nine_image_and_markdown}")
    
    if model_ten_flag:
        # Process with local LLM (Model Ten) (Image Only)
        extracted_data_ollama_model_ten_image_only += call_local_ollama_model(
            ollama_model_ten, image_path, extracted_data_ollama_model_ten_image_only, data_extraction_local_ollama_model_ten_image_only_user_prompt
        )
        logging.info(f"Data extracted Ollama Model Ten image only: {extracted_data_ollama_model_ten_image_only}")
        print(f"Data extracted Ollama Model Ten image only: {extracted_data_ollama_model_ten_image_only}")
        
        # Process with local LLM (Model Ten) (Markdown Only)
        data_extraction_local_ollama_model_ten_markdown_only_user_prompt = f"""{current_page_markdown_text} \n The data extracted from the previous pages of the invoice is as follows: {extracted_data_ollama_model_ten_markdown_only} """
        extracted_data_ollama_model_ten_markdown_only += call_local_ollama_model(
            ollama_model_ten, image_path, extracted_data_ollama_model_ten_markdown_only, data_extraction_local_ollama_model_ten_markdown_only_user_prompt
        )
        logging.info(f"Data extracted Ollama Model Ten markdown only: {extracted_data_ollama_model_ten_markdown_only}")
        print(f"Data extracted Ollama Model Ten markdown only: {extracted_data_ollama_model_ten_markdown_only}")
        
        # Process with local LLM (Model Ten) (Image and Markdown)
        data_extraction_local_ollama_model_ten_image_and_markdown_user_prompt = f"""{current_page_markdown_text} \n The data extracted from the previous pages of the invoice is as follows: {extracted_data_ollama_model_ten_image_and_markdown} """
        extracted_data_ollama_model_ten_image_and_markdown += call_local_ollama_model(
            ollama_model_ten, image_path, extracted_data_ollama_model_ten_image_and_markdown, data_extraction_local_ollama_model_ten_image_and_markdown_user_prompt
        )
        logging.info(f"Data extracted Ollama Model Ten image and markdown: {extracted_data_ollama_model_ten_image_and_markdown}")
        print(f"Data extracted Ollama Model Ten image and markdown: {extracted_data_ollama_model_ten_image_and_markdown}")
        
    return (
        extracted_data_gpt4_image_only,
        extracted_data_gpt4_markdown_only,
        extracted_data_gpt4_image_and_markdown,
        extracted_data_mistral_image_only,
        extracted_data_mistral_markdown_only,
        extracted_data_mistral_image_and_markdown,
        extracted_data_ollama_model_seven_image_only,
        extracted_data_ollama_model_seven_markdown_only,
        extracted_data_ollama_model_seven_image_and_markdown,
        extracted_data_ollama_model_eight_image_only,
        extracted_data_ollama_model_eight_markdown_only,
        extracted_data_ollama_model_eight_image_and_markdown,
        extracted_data_ollama_model_nine_image_only,
        extracted_data_ollama_model_nine_markdown_only,
        extracted_data_ollama_model_nine_image_and_markdown,
        extracted_data_ollama_model_ten_image_only,
        extracted_data_ollama_model_ten_markdown_only,
        extracted_data_ollama_model_ten_image_and_markdown
    )


# -----------------------------------------------------------------PDF Processing Pipeline------------------------------------------------------------------

# Parallel processing of image files
# writing code to evaluate the PDF using OpenAI GPT-4-Vision-Preview model
if st.button("Process PDF"):
    st.session_state.process_pdf_clicked = True
    
    if st.session_state.process_pdf_clicked:
            if uploaded_file is not None:
                st.success("Processing the PDF file using the following model(s) selected:")
                st.write(", ".join(selected_models))
                convert_pdf_to_image(temp_pdf_path)

                # Get the list of images in the images folder
                images_folder = './images'
                image_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]

                # Use ThreadPoolExecutor for parallelism
                with ThreadPoolExecutor() as executor:
                 for image_file in image_files:
                    image_path = os.path.join(images_folder, image_file)
                    futures = [
                        executor.submit(
                            process_image_file,
                            True if "OpenAI GPT-4-Vision-Preview" in selected_models else False,
                            True if "Mistral Local LLM" in selected_models else False,
                            True if "Ollama Model Seven" in selected_models else False,
                            True if "Ollama Model Eight" in selected_models else False,
                            True if "Ollama Model Nine" in selected_models else False,
                            True if "Ollama Model Ten" in selected_models else False,
                            image_file,
                            images_folder,
                            extracted_data_gpt4_image_only,
                            extracted_data_gpt4_markdown_only,
                            extracted_data_gpt4_image_and_markdown,
                            extracted_data_mistral_image_only,
                            extracted_data_mistral_markdown_only,
                            extracted_data_mistral_image_and_markdown,
                            extracted_data_ollama_model_seven_image_only,
                            extracted_data_ollama_model_seven_markdown_only,
                            extracted_data_ollama_model_seven_image_and_markdown,
                            extracted_data_ollama_model_eight_image_only,
                            extracted_data_ollama_model_eight_markdown_only,
                            extracted_data_ollama_model_eight_image_and_markdown,
                            extracted_data_ollama_model_nine_image_only,
                            extracted_data_ollama_model_nine_markdown_only,
                            extracted_data_ollama_model_nine_image_and_markdown,
                            extracted_data_ollama_model_ten_image_only,
                            extracted_data_ollama_model_ten_markdown_only,
                            extracted_data_ollama_model_ten_image_and_markdown
                        )
                        
                    ]

                    # Collect results as they complete
                    for future in as_completed(futures):
                        try:
                            extracted_data_gpt4_image_only, extracted_data_gpt4_markdown_only, extracted_data_gpt4_image_and_markdown, extracted_data_mistral_image_only, extracted_data_mistral_markdown_only, extracted_data_mistral_image_and_markdown, extracted_data_ollama_model_seven_image_only, extracted_data_ollama_model_seven_markdown_only, extracted_data_ollama_model_seven_image_and_markdown, extracted_data_ollama_model_eight_image_only, extracted_data_ollama_model_eight_markdown_only, extracted_data_ollama_model_eight_image_and_markdown, extracted_data_ollama_model_nine_image_only, extracted_data_ollama_model_nine_markdown_only, extracted_data_ollama_model_nine_image_and_markdown, extracted_data_ollama_model_ten_image_only, extracted_data_ollama_model_ten_markdown_only, extracted_data_ollama_model_ten_image_and_markdown = future.result()
                        except Exception as e:
                            logging.error(f"Error processing image: {e}")
                            print(f"Error processing image: {e}")

                st.success("Successfully processed the PDF file using the selected model(s).")
                st.session_state.process_pdf_clicked = False
                st.success("Saving extracted data to CSV file...")
                save_extracted_data_to_csv(
                                data_extraction_system_prompt,
                                uploaded_file,
                                extracted_data_gpt4_image_only,
                                extracted_data_gpt4_markdown_only,
                                extracted_data_gpt4_image_and_markdown,
                                extracted_data_mistral_image_only,
                                extracted_data_mistral_markdown_only,
                                extracted_data_mistral_image_and_markdown,
                                extracted_data_ollama_model_seven_image_only,
                                extracted_data_ollama_model_seven_markdown_only,
                                extracted_data_ollama_model_seven_image_and_markdown,
                                extracted_data_ollama_model_eight_image_only,
                                extracted_data_ollama_model_eight_markdown_only,
                                extracted_data_ollama_model_eight_image_and_markdown,
                                extracted_data_ollama_model_nine_image_only,
                                extracted_data_ollama_model_nine_markdown_only,
                                extracted_data_ollama_model_nine_image_and_markdown,
                                extracted_data_ollama_model_ten_image_only,
                                extracted_data_ollama_model_ten_markdown_only,
                                extracted_data_ollama_model_ten_image_and_markdown
                            )
                st.success("Extracted data saved to CSV file successfully.")
            
            else:
                st.error("Please upload a PDF file to process.")
                st.session_state.process_pdf_clicked = False
        
            
               
                
       
# area to display the extracted data from the PDF using different models
st.subheader("Extracted Data")
st.text_area("Extracted Data using OpenAI GPT-4-Vision-Preview with Image Input Only", value=extracted_data_gpt4_image_only, height=300)
st.text_area("Extracted Data using OpenAI GPT-4-Vision-Preview with Markdown Input Only", value=extracted_data_gpt4_markdown_only, height=300)
st.text_area("Extracted Data using OpenAI GPT-4-Vision-Preview with Image and Markdown Input", value=extracted_data_gpt4_image_and_markdown, height=300)

#button to download the CSV file
st.download_button(
    label="Download CSV",
    data=open(csv_file_path, "rb").read(),
    file_name=csv_file_path,
    mime="text/csv"
)


if st.button("Reset CSV (when starting a new batch of invoices)"):
    st.session_state.reset_csv_clicked = True
    if st.session_state.reset_csv_clicked:
    # Remove all entries in the CSV file
        if os.path.exists("eval_results.csv"):
            with open("eval_results.csv", "w") as csv_file:
                csv_file.truncate()
    
    st.session_state.reset_csv_clicked = False
    st.success("Successfully reset CSV file")


#reset button to delete the images and uploaded_invoices folder
if st.button("Reset Invoices and Images Folders (Important when processing new invoice)"):
    st.session_state.reset_folders_clicked = True
    if st.session_state.reset_folders_clicked:
        # Delete the images folder and uploaded_invoices folder
        if os.path.exists("images"):
            for file in os.listdir("images"):
                os.remove(os.path.join("images", file))
            
        if os.path.exists("uploaded_invoices"):
            for file in os.listdir("uploaded_invoices"):
                os.remove(os.path.join("uploaded_invoices", file))
        
        if os.path.exists("split_pdf_pages"):
            for file in os.listdir("split_pdf_pages"):
                os.remove(os.path.join("split_pdf_pages", file))
                
        
        st.success("Application Folders successfully reset.")
        st.session_state.reset_folders_clicked = False
