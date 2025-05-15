import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
import base64
import re
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import ImageContentItem, ImageUrl, TextContentItem
from dotenv import load_dotenv
from dotenv import load_dotenv
from pyvis.network import Network
import streamlit.components.v1 as components
import json
import shutil
import zipfile  # Import the correct module
from io import BytesIO
import subprocess
import io
import logging
import os
from openai import AzureOpenAI

# Set up logging
log_file_path = os.path.join(os.getcwd(), "process_log.txt")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"  # Overwrite the log file each time the app runs
)
logger = logging.getLogger()

# Load environment variables
load_dotenv()
endpoint = os.getenv("LLM_ENDPOINT")
model_name = os.getenv("LLM_MODEL_NAME")
api_key = os.getenv("LLM_API_KEY")
azure_openai_api_key = os.getenv("OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("OPENAI_API_ENDPOINT")

# Setting LLM Specs
temperature=0.2
top_p=0.9
max_tokens=512
frequency_penalty=0.0
presence_penalty=0.0


# Initialize session state for default_correct_invoices
if "default_correct_invoices" not in st.session_state:
    st.session_state.default_correct_invoices = 0


# Initialize Azure ChatCompletionsClient
client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(api_key),
)

# Streamlit app title
st.title("Invoice Processing and Analysis")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Define helper functions

# SPLIT PDF INTO INDIVIDUAL PAGES
def split_parent_pdf_into_individual_pages(input_pdf_path):
    output_folder = "output_pages"  # Folder to store split pages

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the input PDF
    reader = PdfReader(input_pdf_path)

    # Loop through each page and save it as a separate file
    for i, page in enumerate(reader.pages):
        writer = PdfWriter()
        writer.add_page(page)

        output_pdf_path = os.path.join(output_folder, f"{i+1}.pdf")
        with open(output_pdf_path, "wb") as output_pdf:
            writer.write(output_pdf)
        
        print(f"Saved: {output_pdf_path}")

    print("PDF split completed!")
    
    return len(reader.pages)  # Return the number of pages
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------

# Convert PDF pages to images
def convert_pdf_to_image(child_pdf_path, page_number):
    from pdf2image import convert_from_path
    

    

    # Ensure the main images directory exists
    images_output_folder = './images'
    os.makedirs(images_output_folder, exist_ok=True)


    print(f"Processing: {child_pdf_path}")


    # Convert PDF pages to images
    images = convert_from_path(child_pdf_path)
    # Save the images to the subfolder
    for i, img in enumerate(images):
            image_path = os.path.join(images_output_folder, f'{page_number}.png')
            img.save(image_path, 'PNG')

    print(f"Saved images for {os.path.basename(child_pdf_path)} in {images_output_folder}")
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------

# Generate base64 URL for image
def generate_base64_url(image_path):
    
    
    with open(image_path, "rb") as img_file:
        raw_data = img_file.read()
        image_data = base64.b64encode(raw_data).decode("utf-8")
    image_format = image_path.split('.')[-1]
    return f"data:image/{image_format};base64,{image_data}"

# -----------------------------------------------------------------------------------------------------------------------------------------------------------

system_prompt_for_invoice_split_LLM_vision = ""


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to consolidate the PDF pages into consolidated invoices
def call_LLM_for_invoice_split(system_prompt, image_data_url):

    

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )

    response = client.complete(
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=[
                TextContentItem(text="I have attached the image"),
                ImageContentItem(image_url = ImageUrl(url=image_data_url))
            ])
        ],
        model=model_name,
        max_tokens=max_tokens,  # You can increase this if needed, but shorter helps prevent hallucination
        temperature=temperature,  # Maximum determinism
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        stop=["\n\n", "---", "Explanation", "Note:"]
    )

    return response.choices[0].message.content

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to consolidate the PDF pages into consolidated invoices
def call_ChatGPT_for_invoice_split(system_prompt, image_data_url):
    
    client = AzureOpenAI(
    azure_endpoint = azure_openai_endpoint, 
    api_key=azure_openai_api_key,  
    api_version="2024-02-15-preview"
    )
    
    response = client.chat.completions.create(
        model="gpt-4.1-kiebidz",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"{system_prompt}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I have attaached the image"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{image_data_url}"
                    }
                    }
                ]
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    print(f"Response: {response.choices[0].message.content}")
    return str(response.choices[0].message.content)
#---------------------------------------------------------------------------------------------------------------------------------------------

# Function to extract fields from the LLM response using regex-based pattern matching
def extract_fields_from_text(LLM_response):
    import os
    import re

    # Define regex patterns for extracting fields
    vendor_name_pattern = r"Vendor Name:\s*(.+)"
    invoice_id_pattern = r"Invoice ID:\s*(.+)"
    customer_name_pattern = r"Customer Name:\s*(.+)"
    
    # Perform regex-based extraction
    vendor_name_match = re.search(vendor_name_pattern, LLM_response)
    invoice_id_match = re.search(invoice_id_pattern, LLM_response)
    customer_name_match = re.search(customer_name_pattern, LLM_response)
    
    # Extract values or set to None if not found
    vendor_name = vendor_name_match.group(1) if vendor_name_match else None
    invoice_id = invoice_id_match.group(1) if invoice_id_match else None
    customer_name = customer_name_match.group(1) if customer_name_match else None
    
     # Print extracted values
    print(f"Extracted Vendor Name: {vendor_name}")
    print(f"Extracted Invoice ID: {invoice_id}")
    print(f"Extracted Customer Name: {customer_name}")
    

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Defining hashMap to keep track of the page numbers and their corresponding invoice IDs, vendor names, and customer names
hashMap = {}


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
validate_vendor_names_system_prompt = ""
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to validate vendor names using the LLM

def call_LLM_for_vendor_name_validation(system_prompt, vendor_name_1, vendor_name_2):
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )
    
    user_query = """ The two vendor names are:
    Vendor Name 1: {}
    Vendor Name 2: {}
    """.format(vendor_name_1, vendor_name_2)

    response = client.complete(
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=[
                TextContentItem(text=user_query),
            ])
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        model=model_name
    )
    
    print(f"Response: {response.choices[0].message.content}")
    return response.choices[0].message.content 

#------------------------------------------------------------------------------------------------------------------------
# Function to validate vendor names using the LLM

def call_ChatGPT_for_vendor_name_validation(system_prompt, vendor_name_1, vendor_name_2):
    
    user_query = """ The two vendor names are:
    Vendor Name 1: {}
    Vendor Name 2: {}
    """.format(vendor_name_1, vendor_name_2)

    client = AzureOpenAI(
    azure_endpoint = azure_openai_endpoint, 
    api_key=azure_openai_api_key,  
    api_version="2024-02-15-preview"
    )
    
    response = client.chat.completions.create(
        model="gpt-4.1-kiebidz",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"{system_prompt}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{user_query}"
                    },
                ]
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    print(f"Response: {response.choices[0].message.content}")
    return str(response.choices[0].message.content)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to check if the invoice ID has changed

def is_invoice_id_changed(invoice_id_1: str, invoice_id_2: str) -> bool:
    # Extract numeric parts using regex
    numeric_part_1 = re.sub(r'\D', '', invoice_id_1)
    numeric_part_2 = re.sub(r'\D', '', invoice_id_2)

    return numeric_part_1 != numeric_part_2

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
validate_person_names_system_prompt = ""

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to validate vendor names using the LLM
def call_LLM_for_customer_name_validation(system_prompt, customer_name_1, customer_name_2):
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )
    
    user_query = """ The two customer names are:
    Customer Name 1: {}
    Customer Name 2: {}
    """.format(customer_name_1, customer_name_2)

    response = client.complete(
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=[
                TextContentItem(text=user_query),
            ])
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        model=model_name
    )
    
    print(f"Response: {response.choices[0].message.content}")
    return response.choices[0].message.content

#------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
# Function to validate vendor names using the LLM
def call_ChatGPT_for_customer_name_validation(system_prompt, customer_name_1, customer_name_2):
    
    user_query = """ The two customer names are:
    Customer Name 1: {}
    Customer Name 2: {}
    """.format(customer_name_1, customer_name_2)

    client = AzureOpenAI(
    azure_endpoint = azure_openai_endpoint, 
    api_key=azure_openai_api_key,  
    api_version="2024-02-15-preview"
    )
    
    response = client.chat.completions.create(
        model="gpt-4.1-kiebidz",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"{system_prompt}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{user_query}"
                    }
                ]
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    print(f"Response: {response.choices[0].message.content}")
    return str(response.choices[0].message.content)

#----------------------------------------------------------------------------------------------------

def document_intelligence(pdf_path, page_number):
    
    import os
    import re
    try:
        global current_vendor_name  # Declare global to track VendorName across pages
        
        global current_invoice_id  # Declare global to track InvoiceId across pages
        
        global current_customer_name # Declare global to track CustomerName across pages
        
        #Generating the child pdf image path
        child_pdf_image_path = os.path.join("./images", f"{page_number}.png")
        
        # Generating the base64 URL for the image
        base64_url = generate_base64_url(child_pdf_image_path)  # Generate base64 URL for the image
        
        LLM_response = call_LLM_for_invoice_split(system_prompt_for_invoice_split_LLM_vision, base64_url)  # Call the LLM with the image data URL
        print("LLM Response for pdf base {}:".format(os.path.basename(pdf_path)))
        print(LLM_response)
        
       # Define regex patterns for extracting fields
        vendor_name_pattern = r"Vendor Name:\s*(.+)"
        invoice_id_pattern = r"Invoice ID:\s*(.+)"
        customer_name_pattern = r"Customer Name:\s*(.+)"
       

        # Perform regex-based extraction
        vendor_name_match = re.search(vendor_name_pattern, LLM_response)
        invoice_id_match = re.search(invoice_id_pattern, LLM_response)
        customer_name_match = re.search(customer_name_pattern, LLM_response)

        # Extract values or set to None if not found
        current_page_vendor_name = vendor_name_match.group(1) if vendor_name_match else None
        current_page_invoice_id = invoice_id_match.group(1) if invoice_id_match else None
        current_page_customer_name = customer_name_match.group(1) if customer_name_match else None

        
        
        if current_page_vendor_name:
            if current_vendor_name and call_LLM_for_vendor_name_validation(validate_vendor_names_system_prompt,current_page_vendor_name, current_vendor_name)=="No":
               print(f"Vendor name changed from {current_vendor_name} to {current_page_vendor_name} on page {page_number}.")
               logger.info(f"Vendor name changed from {current_vendor_name} to {current_page_vendor_name} on page {page_number}.")
               current_vendor_name = current_page_vendor_name  #Update the global variable
               return "VendorNameChanged" 
            
            #Update the global variable if its the first page
            current_vendor_name = current_page_vendor_name #Update the local variable
         
        if current_page_invoice_id:
            if current_invoice_id and is_invoice_id_changed(current_page_invoice_id, current_invoice_id) is True:
                print(f"InvoiceId changed from {current_invoice_id} to {current_page_invoice_id} on page {page_number}.")
                logger.info(f"InvoiceId changed from {current_invoice_id} to {current_page_invoice_id} on page {page_number}.")
                current_invoice_id = current_page_invoice_id #Update the global variable
                return "InvoiceIdChanged"
            # Update the global variable if its the first page 
            current_invoice_id = current_page_invoice_id  #Update the global variable  
            
        if current_page_customer_name:
            if current_customer_name and call_LLM_for_customer_name_validation(validate_person_names_system_prompt,current_page_customer_name, current_customer_name)=="No":
                print(f"CustomerName changed from {current_customer_name} to {current_page_customer_name} on page {page_number}.")
                logger.info(f"CustomerName changed from {current_customer_name} to {current_page_customer_name} on page {page_number}.")
                current_customer_name = current_page_customer_name
                return "CustomerNameChanged"
            # Update the global variable if its the first page
            current_customer_name = current_page_customer_name 
        
        # If no relevant fields are found, mark as "child page"
        hashMap[page_number] = "child page"
        print("page number : {} child page".format(page_number))
        return "child page"
    
    except Exception as e:
        print(f"Error processing page {page_number}: {e}")
        return "error"
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

def create_pdf_from_pages(input_pdf: str, output_pdf: str, pages: list):
    if not pages:
        print("No pages specified for creating the PDF.")
        return

    try:
        reader = PdfReader(input_pdf)
        writer = PdfWriter()

        for page_num in pages:
            if 1 <= page_num <= len(reader.pages):
                writer.add_page(reader.pages[page_num - 1])  # Convert 1-based to 0-based index
            else:
                print(f"Invalid page number: {page_num}")

        # Ensure the output directory exists
        output_dir = "output_files"
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, output_pdf), "wb") as out_file:
            writer.write(out_file)

        print(f"New PDF created: {output_pdf}")

    except Exception as e:
        print(f"Error creating PDF: {e}")
        
    
# -------------------------------------------------------------------------------------------------------------------------------------
split_json_numerals = []

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

def process_pdf(pdf_path: str):
    """Reads a multi-page PDF and sends each page separately to Azure Document Intelligence."""
    try:
        pdf_reader = PdfReader(pdf_path)
        hashMap.clear()  # Clear the hashMap before processing
        invoiceID = None
        pageList = []  # To store all pages in the hashMap of the current invoice to be consolidated into a single PDF
        count = 1 #for naming the output files
        
        global current_vendor_name
        current_vendor_name = None  # Initialize the current VendorName
        
        global current_invoice_id
        current_invoice_id = None  # Initialize the current InvoiceId
        
        global current_customer_name
        current_customer_name = None  # Initialize the current CustomerName
        
        
        for page_num in range(len(pdf_reader.pages)):
            # Generate the pdf path for the individual invoice page
            child_pdf_path = os.path.join("output_pages", f"{page_num + 1}.pdf")

            # Call Azure Document Intelligence for this page
            result = document_intelligence(child_pdf_path, page_num + 1)  # Use 1-based page index
            
            if result=="VendorNameChanged" or result=="InvoiceIdChanged" or result=="CustomerNameChanged":
                # Finalize the current invoice (exclude the current page)
                if pageList:
                    create_pdf_from_pages(pdf_path, f"output_{count}.pdf", pageList)
                    if result=="VendorNameChanged":
                        print(f"VendorName changed: New PDF created: output_{count}.pdf with pages: {pageList}")
                        logger.info(f"VendorName changed: New PDF created: output_{count}.pdf with pages: {pageList}\n")
                    if result=="InvoiceIdChanged":
                        print(f"Invoice ID changed: New PDF created: output_{count}.pdf with pages: {pageList}")
                        logger.info(f"Invoice ID changed: New PDF created: output_{count}.pdf with pages: {pageList}\n")
                    if result=="CustomerNameChanged":
                        print(f"CustomerName changed: New PDF created: output_{count}.pdf with pages: {pageList}")
                        logger.info(f"CustomerName changed: New PDF created: output_{count}.pdf with pages: {pageList}\n")
                    count += 1
                    split_json_numerals.append(pageList[-1])
                    pageList = []  # Reset the page list for the next invoice
                    
                
                current_customer_name = None  # Reset the current customer name for the new invoice
                current_invoice_id = None  # Reset the current invoice ID for the new invoice
                current_vendor_name = None  # Reset the current vendor name for the new invoice
                
                hashMap.clear()  # Clear the hashMap for the new invoice
                
                #Process the current page again
                result = document_intelligence(child_pdf_path, page_num+1) #reprocess the current page
            
            # Add current page to the pageList
            pageList.append(page_num + 1)  # Use 1-based page index
            
            
                
        # Finalize the last invoice if any pages are left in pageList
        if pageList:
            split_json_numerals.append(pageList[-1])
            create_pdf_from_pages(pdf_path, f"output_{count}.pdf", pageList)
            print(f"Remaining pages consolidated: New PDF created: output_{count}.pdf with pages: {pageList}")
            logger.info(f"Remaining pages consolidated: New PDF created: output_{count}.pdf with pages: {pageList}\n")

        
            
        if not pageList:
            print("No valid invoice data found in the PDF.")
            logger.info("No valid invoice data found in the PDF.\n")
    except Exception as e:
        print(f"An error occurred while processing the PDF: {e}")
        logger.info(f"An error occurred while processing the PDF: {e}\n")
    print("\nProcessing completed!")
    logger.info("\nProcessing completed!\n")
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
def document_intelligence_using_ChatGPT(pdf_path, page_number):
    
    import os
    import re
    try:
        global current_vendor_name  # Declare global to track VendorName across pages
        
        global current_invoice_id  # Declare global to track InvoiceId across pages
        
        global current_customer_name # Declare global to track CustomerName across pages
        
        #Generating the child pdf image path
        child_pdf_image_path = os.path.join("./images", f"{page_number}.png")
        
        # Generating the base64 URL for the image
        base64_url = generate_base64_url(child_pdf_image_path)  # Generate base64 URL for the image
        
        LLM_response = call_ChatGPT_for_invoice_split(system_prompt_for_invoice_split_LLM_vision, base64_url)  # Call the LLM with the image data URL
        print("LLM Response for pdf base {}:".format(os.path.basename(pdf_path)))
        print(LLM_response)
        
       # Define regex patterns for extracting fields
        vendor_name_pattern = r"Vendor Name:\s*(.+)"
        invoice_id_pattern = r"Invoice ID:\s*(.+)"
        customer_name_pattern = r"Customer Name:\s*(.+)"
       

        # Perform regex-based extraction
        vendor_name_match = re.search(vendor_name_pattern, LLM_response)
        invoice_id_match = re.search(invoice_id_pattern, LLM_response)
        customer_name_match = re.search(customer_name_pattern, LLM_response)

        # Extract values or set to None if not found
        current_page_vendor_name = vendor_name_match.group(1) if vendor_name_match else None
        current_page_invoice_id = invoice_id_match.group(1) if invoice_id_match else None
        current_page_customer_name = customer_name_match.group(1) if customer_name_match else None

        
        
        if current_page_vendor_name:
            if current_vendor_name and call_ChatGPT_for_vendor_name_validation(validate_vendor_names_system_prompt,current_page_vendor_name, current_vendor_name)=="No":
               print(f"Vendor name changed from {current_vendor_name} to {current_page_vendor_name} on page {page_number}.")
               logger.info(f"Vendor name changed from {current_vendor_name} to {current_page_vendor_name} on page {page_number}.")
               current_vendor_name = current_page_vendor_name  #Update the global variable
               return "VendorNameChanged" 
            
            #Update the global variable if its the first page
            current_vendor_name = current_page_vendor_name #Update the local variable
         
        if current_page_invoice_id:
            if current_invoice_id and is_invoice_id_changed(current_page_invoice_id, current_invoice_id) is True:
                print(f"InvoiceId changed from {current_invoice_id} to {current_page_invoice_id} on page {page_number}.")
                logger.info(f"InvoiceId changed from {current_invoice_id} to {current_page_invoice_id} on page {page_number}.")
                current_invoice_id = current_page_invoice_id #Update the global variable
                return "InvoiceIdChanged"
            # Update the global variable if its the first page 
            current_invoice_id = current_page_invoice_id  #Update the global variable  
            
        if current_page_customer_name:
            if current_customer_name and call_ChatGPT_for_customer_name_validation(validate_person_names_system_prompt,current_page_customer_name, current_customer_name)=="No":
                print(f"CustomerName changed from {current_customer_name} to {current_page_customer_name} on page {page_number}.")
                logger.info(f"CustomerName changed from {current_customer_name} to {current_page_customer_name} on page {page_number}.")
                current_customer_name = current_page_customer_name
                return "CustomerNameChanged"
            # Update the global variable if its the first page
            current_customer_name = current_page_customer_name 
        
        # If no relevant fields are found, mark as "child page"
        hashMap[page_number] = "child page"
        print("page number : {} child page".format(page_number))
        return "child page"
    
    except Exception as e:
        print(f"Error processing page {page_number}: {e}")
        return "error"
#------------------------------------------------------------------------------------------------------------------------------------------------
def process_pdf_using_ChatGPT(pdf_path: str):
    """Reads a multi-page PDF and sends each page separately to ChatGPT Document Intelligence."""
    try:
        pdf_reader = PdfReader(pdf_path)
        hashMap.clear()  # Clear the hashMap before processing
        invoiceID = None
        pageList = []  # To store all pages in the hashMap of the current invoice to be consolidated into a single PDF
        count = 1 #for naming the output files
        
        global current_vendor_name
        current_vendor_name = None  # Initialize the current VendorName
        
        global current_invoice_id
        current_invoice_id = None  # Initialize the current InvoiceId
        
        global current_customer_name
        current_customer_name = None  # Initialize the current CustomerName
        
        
        for page_num in range(len(pdf_reader.pages)):
            # Generate the pdf path for the individual invoice page
            child_pdf_path = os.path.join("output_pages", f"{page_num + 1}.pdf")

            # Call Azure Document Intelligence for this page
            result = document_intelligence_using_ChatGPT(child_pdf_path, page_num + 1)  # Use 1-based page index
            
            if result=="VendorNameChanged" or result=="InvoiceIdChanged" or result=="CustomerNameChanged":
                # Finalize the current invoice (exclude the current page)
                if pageList:
                    create_pdf_from_pages(pdf_path, f"output_{count}.pdf", pageList)
                    if result=="VendorNameChanged":
                        print(f"VendorName changed: New PDF created: output_{count}.pdf with pages: {pageList}")
                        logger.info(f"VendorName changed: New PDF created: output_{count}.pdf with pages: {pageList}\n")
                    if result=="InvoiceIdChanged":
                        print(f"Invoice ID changed: New PDF created: output_{count}.pdf with pages: {pageList}")
                        logger.info(f"Invoice ID changed: New PDF created: output_{count}.pdf with pages: {pageList}\n")
                    if result=="CustomerNameChanged":
                        print(f"CustomerName changed: New PDF created: output_{count}.pdf with pages: {pageList}")
                        logger.info(f"CustomerName changed: New PDF created: output_{count}.pdf with pages: {pageList}\n")
                    count += 1
                    split_json_numerals.append(pageList[-1])
                    pageList = []  # Reset the page list for the next invoice
                    
                
                current_customer_name = None  # Reset the current customer name for the new invoice
                current_invoice_id = None  # Reset the current invoice ID for the new invoice
                current_vendor_name = None  # Reset the current vendor name for the new invoice
                
                hashMap.clear()  # Clear the hashMap for the new invoice
                
                #Process the current page again
                result = document_intelligence_using_ChatGPT(child_pdf_path, page_num+1) #reprocess the current page
            
            # Add current page to the pageList
            pageList.append(page_num + 1)  # Use 1-based page index
            
            
                
        # Finalize the last invoice if any pages are left in pageList
        if pageList:
            split_json_numerals.append(pageList[-1])
            create_pdf_from_pages(pdf_path, f"output_{count}.pdf", pageList)
            print(f"Remaining pages consolidated: New PDF created: output_{count}.pdf with pages: {pageList}")
            logger.info(f"Remaining pages consolidated: New PDF created: output_{count}.pdf with pages: {pageList}\n")

        
            
        if not pageList:
            print("No valid invoice data found in the PDF.")
            logger.info("No valid invoice data found in the PDF.\n")
    except Exception as e:
        print(f"An error occurred while processing the PDF: {e}")
        logger.info(f"An error occurred while processing the PDF: {e}\n")
    print("\nProcessing completed!")
    logger.info("\nProcessing completed!\n")
#-------------------------------------------------------------------------------------------------------------------------

# Function to delete a folder and its contents
def delete_folder(folder_path):
    """
    Recursively deletes a folder and all its contents.

    Args:
        folder_path (str): The path to the folder to be deleted.

    Returns:
        bool: True if the folder was successfully deleted, False if the folder does not exist.
    """
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                delete_folder(file_path)
        os.rmdir(folder_path)
        return True
    return False


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Systme Prompt for the invoice classification with LLM help

system_prompt_for_invoice_classification = ""

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to classify the invoice using the LLM

def call_LLM_for_invoice_classification(system_prompt, image_data_url):
    """
    Calls a Large Language Model (LLM) for invoice classification using a system prompt and an image data URL.
    Args:
        system_prompt (str): The system-level prompt that provides context or instructions for the LLM.
        image_data_url (str): The URL of the image data to be classified.
    Returns:
        str: The content of the response message from the LLM, which contains the classification result.
    Notes:
        - The function uses the Azure OpenAI ChatCompletionsClient to interact with the LLM.
        - The response is generated based on the provided system prompt and the image data.
        - Parameters such as `max_tokens`, `temperature`, `top_p`, `presence_penalty`, and `frequency_penalty` 
          are configured to control the behavior and determinism of the LLM's output.
        - The `stop` parameter defines tokens that signal the end of the response.
    """

   
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )

    response = client.complete(
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=[
                TextContentItem(text="I have attached the image"),
                ImageContentItem(image_url = ImageUrl(url=image_data_url))
            ])
        ],
        model=model_name,
        max_tokens=max_tokens,  # You can increase this if needed, but shorter helps prevent hallucination
        temperature=temperature,  # Maximum determinism
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        stop=["\n\n", "---", "Explanation", "Note:"]
    )
    
    return response.choices[0].message.content

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
final_output = {}

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

def process_invoice_classification(classification_prompt):
    """
    Processes a list of invoice PDFs by extracting the first page, converting it to an image, 
    and classifying the invoice using a language model.
    Args:
        classification_prompt (str): The prompt to be used for the language model to classify the invoice.
    Workflow:
        1. Iterates through all PDF files in the "./output_files" directory.
        2. For each PDF:
            - Reads the first page of the PDF.
            - Saves the first page as a new PDF in the "./output_pages" directory.
            - Converts the first page PDF to an image and saves it in the "./images" directory.
            - Generates a base64 URL for the image.
            - Calls a language model (LLM) with the classification prompt and the base64 image URL.
            - Stores the classification result in the `final_output` dictionary.
    Exceptions:
        - Handles and logs any exceptions that occur during processing, such as file I/O errors or 
          issues with PDF/image processing.
    Notes:
        - The function assumes the existence of helper functions:
            - `convert_pdf_to_image`: Converts a PDF to an image.
            - `generate_base64_url`: Generates a base64 URL for an image file.
            - `call_LLM_for_invoice_classification`: Calls the language model for classification.
        - The `final_output` dictionary is expected to be defined globally to store the results.
        - The directories "./output_files", "./output_pages", and "./images" are used for input/output operations.
    """
    invoices = [f for f in os.listdir("./output_files") if f.endswith(".pdf")]

    for i, invoice in enumerate(invoices):
        current_invoice = os.path.join("./output_files", invoice)
        print("Processing Current invoice: ", current_invoice)

        try:
            # Read the first page of the PDF
            reader = PdfReader(current_invoice)
            if len(reader.pages) > 0:
                # Generate the child PDF path for the first page
                first_page_pdf_path = os.path.join("./output_pages", f"{os.path.splitext(invoice)[0]}_page_1.pdf")
                writer = PdfWriter()
                writer.add_page(reader.pages[0])  # Add only the first page
                os.makedirs("./output_pages", exist_ok=True)
                with open(first_page_pdf_path, "wb") as output_pdf:
                    writer.write(output_pdf)
                print(f"Extracted first page of {invoice} to {first_page_pdf_path}")

                # Convert the first page to an image
                first_page_image_path = os.path.join("./images", f"{os.path.splitext(invoice)[0]}_page_1.png")
                convert_pdf_to_image(first_page_pdf_path, f"{os.path.splitext(invoice)[0]}_page_1")
                print(f"Converted first page of {invoice} to image: {first_page_image_path}")

                # Generate the base64 URL for the image
                if os.path.exists(first_page_image_path):  # Ensure the file exists before accessing it
                    base64_url = generate_base64_url(first_page_image_path)

                    # Call the LLM with the image data URL
                    LLM_response = call_LLM_for_invoice_classification(classification_prompt, base64_url)

                    # Add the classification to the final_output dictionary
                    final_output[invoice] = LLM_response.strip()
                    print(f"Final classification for {invoice}: {LLM_response.strip()}")
                else:
                    print(f"Image file not found: {first_page_image_path}")
            else:
                print(f"No pages found in {invoice}. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing {invoice}: {e}")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to generate a Neo4j-like graph
def generate_classification_graph(classification_results):
    """
    Generates a classification graph using PyVis and saves it as an HTML file.
    This function creates a network graph where each classification is represented
    as a central node, and each invoice is represented as a connected node. The
    graph is styled with specific colors and shapes for classifications and invoices.
    Args:
        classification_results (dict): A dictionary where keys are invoice identifiers
            (e.g., invoice numbers) and values are their corresponding classifications.
    Returns:
        None: The graph is saved as an HTML file named "classification_graph.html".
    """
    # Create a PyVis network graph
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    
    # Add a central node for each classification
    classifications = set(classification_results.values())
    for classification in classifications:
        net.add_node(classification, label=classification, color="#FF5733", shape="ellipse")

    # Add nodes for invoices and connect them to their classification
    for invoice, classification in classification_results.items():
        net.add_node(invoice, label=invoice, color="#33FF57", shape="box")
        net.add_edge(classification, invoice)

    # Save the graph as an HTML file
    net.save_graph("classification_graph.html")
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

if st.button("Reset"):
    delete_folder("output_pages")
    delete_folder("output_files")
    delete_folder("images")
    delete_folder("uploaded_files")
    
    # Close the logger's file handler
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
        
    # Delete the process_log.txt file if it exists
    if os.path.exists("process_log.txt"):
        os.remove("process_log.txt")
        st.info("Deleted process_log.txt file.")
    else:
        st.warning("process_log.txt file does not exist.")

# Initialize session state for managing app state
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None
if "num_pages" not in st.session_state:
    st.session_state.num_pages = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "merge_dict" not in st.session_state:
    st.session_state.merge_dict = {}


# Sidebar section to select LLM
st.sidebar.header("Select LLM")

# Dropdown to select LLM
selected_llm = st.sidebar.selectbox(
    "Choose the LLM for processing:",
    ["Select an LLM", "ChatGPT", "Mistral"],
    key="select_llm"
)

# Display the selected LLM
if selected_llm != "Select an LLM":
    st.sidebar.success(f"You have selected: {selected_llm}")
else:
    st.sidebar.warning("Please select an LLM to proceed.")

# ---------------------------------SIDEBAR SECTION TO MANAGE PROMPT LAB-------------------------------------------------------------------

# Sidebar section to manage system prompts
st.sidebar.header("Manage System Prompts")

# ----------------- INVOICE SPLIT PROMPT LAB ------------------------------------------
  
# Ensure the "./prompt" folder exists
invoice_split_prompt_folder = "./invoice_split_prompts"
os.makedirs(invoice_split_prompt_folder, exist_ok=True)

# Prompt 1: system_prompt_for_invoice_split_LLM_vision
st.sidebar.subheader("Invoice Split Prompt")
st.sidebar.subheader("Make Sure to Reselect Prompts after editing or creating new ones from the drop-down for processing")
# Get the list of available prompts
prompt_files = [f for f in os.listdir(invoice_split_prompt_folder) if f.endswith(".txt")]

# Dropdown to select a prompt
selected_invoice_split_prompt = st.sidebar.selectbox("Select an Invoice Split System Prompt", ["Select a prompt"] + prompt_files, key="select_invoice_split_prompt")


# Display the content of the selected prompt
if selected_invoice_split_prompt != "Select a prompt":
    prompt_path = os.path.join(invoice_split_prompt_folder, selected_invoice_split_prompt)
    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            system_prompt_for_invoice_split_LLM_vision = file.read()
        system_prompt_for_invoice_split_LLM_vision = st.sidebar.text_area(
            "Edit system_prompt_for_invoice_split_LLM_vision",
            system_prompt_for_invoice_split_LLM_vision,
            height=150,
            key="system_prompt_for_invoice_split_LLM_vision_content"
        )

        
        

    except Exception as e:
        st.error(f"Error reading the selected prompt: {e}")
        
     # Button to delete the selected prompt
    if st.sidebar.button("Delete Selected Prompt", key="d4"):
        try:
            os.remove(prompt_path)
            st.sidebar.success(f"Prompt '{selected_invoice_split_prompt}' deleted successfully!")
            st.experimental_rerun()  # Refresh the app to update the dropdown
        except Exception as e:
            st.sidebar.error(f"Error deleting the prompt: {e}")
            
            
# Button to show the "Create New Prompt" section
if st.sidebar.button("Create New Invoice Split Prompt"):
    st.session_state.show_create_prompt = True

# Dynamically display the "Create New Prompt" section if the button is clicked
if st.session_state.get("show_create_prompt", False):
    st.sidebar.subheader("Create a New Invoice Split Prompt")
    new_invoice_split_prompt_name = st.sidebar.text_input("Enter a name for the new split invoice prompt (e.g., 'new_prompt.txt')", "")
    new_invoice_split_prompt_content = st.sidebar.text_area("Enter the content for the new invoice split prompt", "", height=300)

    if st.sidebar.button("Save New Invoice Split Prompt"):
        if new_invoice_split_prompt_name and new_invoice_split_prompt_name.endswith(".txt"):
            new_prompt_path = os.path.join(invoice_split_prompt_folder, new_invoice_split_prompt_name)
            if os.path.exists(new_prompt_path):
                st.warning(f"A prompt with the name '{new_invoice_split_prompt_name}' already exists.")
            else:
                try:
                    with open(new_prompt_path, "w", encoding="utf-8") as file:
                        file.write(new_invoice_split_prompt_content)
                    st.success(f"New prompt '{new_invoice_split_prompt_name}' created successfully!")
                    st.session_state.show_create_prompt = False  # Hide the section after saving
                    st.experimental_rerun()  # Refresh the app to show the new prompt in the dropdown
                except Exception as e:
                    st.error(f"Error creating new prompt: {e}")
        else:
            st.warning("Please enter a valid name for the new prompt (must end with '.txt').")
            
if st.sidebar.button("Save Invoice Split Prompt", key="save_invoice_split_prompt"):
    try:
        # Save the updated content to a file
        with open(os.path.join(invoice_split_prompt_folder,selected_invoice_split_prompt), "w", encoding="utf-8") as file:
            file.write(system_prompt_for_invoice_split_LLM_vision)
        st.sidebar.success("Invoice Split Prompt saved successfully!")
    except Exception as e:
        st.sidebar.error(f"Error saving Invoice Split Prompt: {e}")

# ------------------------ INVOICE SPLIT PROMPT LAB END -----------------------------------


# --------------------------------- VENDOR NAME VALIDATION PROMPT LAB BEGIN ----------------------------------------

# Prompt 2: validate_vendor_names_system_prompt
st.sidebar.subheader("Vendor Names Validation Prompt")

# Ensure the "./prompt" folder exists
vendor_name_prompt_folder = "./vendor_name_check_prompts"
os.makedirs(vendor_name_prompt_folder, exist_ok=True)

st.sidebar.subheader("Make Sure to Reselect Prompts after editing or creating new ones from the drop-down for processing")

# Get the list of available prompts
vendor_name_prompt_files = [f for f in os.listdir(vendor_name_prompt_folder) if f.endswith(".txt")]

# Dropdown to select a prompt
selected_vendor_name_validation_prompt = st.sidebar.selectbox("Select a Vendor Name Validation System Prompt", ["Select a prompt"] + vendor_name_prompt_files, key="select_vendor_name_validation_prompt")

# Display the content of the selected prompt
if selected_vendor_name_validation_prompt != "Select a prompt":
    prompt_path = os.path.join(vendor_name_prompt_folder, selected_vendor_name_validation_prompt)
    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            validate_vendor_names_system_prompt = file.read()
        validate_vendor_names_system_prompt = st.sidebar.text_area(
            "Edit system_prompt_for_vendor_name_validation",
            validate_vendor_names_system_prompt,
            height=150,
            key="system_prompt_for_vendor_name_validation"
        )

        
    
    except Exception as e:
        st.error(f"Error reading the selected prompt: {e}")
    
    
    # Button to delete the selected prompt
    if st.sidebar.button("Delete Selected Prompt", key="d3"):
        try:
            os.remove(prompt_path)
            st.sidebar.success(f"Prompt '{selected_vendor_name_validation_prompt}' deleted successfully!")
            st.experimental_rerun()  # Refresh the app to update the dropdown
        except Exception as e:
            st.sidebar.error(f"Error deleting the prompt: {e}")
            
# Button to show the "Create New Prompt" section
if st.sidebar.button("Create New Vendor Name Validation Prompt"):
    st.session_state.show_create_prompt = True
    
    
# Dynamically display the "Create New Prompt" section if the button is clicked
if st.session_state.get("show_create_prompt", False):
    st.sidebar.subheader("Create a New Vendor Name Validation Prompt")
    new_vendor_name_validation_prompt_name = st.sidebar.text_input("Enter a name for the new vendor name validation prompt (e.g., 'new_prompt.txt')", "")
    new_vendor_name_validation_prompt_content = st.sidebar.text_area("Enter the content for the new vendor name validation prompt", "", height=300)

    if st.sidebar.button("Save New Vendor Name Validation Prompt"):
        if new_vendor_name_validation_prompt_name and new_vendor_name_validation_prompt_name.endswith(".txt"):
            new_prompt_path = os.path.join(vendor_name_prompt_folder, new_vendor_name_validation_prompt_name)
            if os.path.exists(new_prompt_path):
                st.warning(f"A prompt with the name '{new_vendor_name_validation_prompt_name}' already exists.")
            else:
                try:
                    with open(new_prompt_path, "w", encoding="utf-8") as file:
                        file.write(new_vendor_name_validation_prompt_content)
                    st.success(f"New prompt '{new_vendor_name_validation_prompt_name}' created successfully!")
                    st.session_state.show_create_prompt = False  # Hide the section after saving
                    st.experimental_rerun()  # Refresh the app to show the new prompt in the dropdown
                except Exception as e:
                    st.error(f"Error creating new prompt: {e}")
        else:
            st.warning("Please enter a valid name for the new prompt (must end with '.txt').")
            
if st.sidebar.button("Save Vendor Name Validation Prompt", key="save_vendor_name_validation_prompt"):
    try:
        # Save the updated content to a file
        with open(os.path.join(vendor_name_prompt_folder,selected_vendor_name_validation_prompt), "w", encoding="utf-8") as file:
            file.write(validate_vendor_names_system_prompt)
        st.sidebar.success("Invoice Split Prompt saved successfully!")
    except Exception as e:
        st.sidebar.error(f"Error saving Invoice Split Prompt: {e}")

# --------------------------- VENDOR NAME VALIDATION PROMPT LAB END --------------------------------------


# ----------------------------------PERSON NAME VALIDATION PROMPT LAB BEGIN -------------------------------------


# Prompt 3: validate_person_names_system_prompt
st.sidebar.subheader("Person Names Validation System Prompt")

# Ensure the "./prompt" folder exists
person_name_prompt_folder = "./person_names_check_prompt"
os.makedirs(person_name_prompt_folder, exist_ok=True)

st.sidebar.subheader("Make Sure to Reselect Prompts after editing or creating new ones from the drop-down for processing")

# Get the list of available prompts
person_name_prompt_files = [f for f in os.listdir(person_name_prompt_folder) if f.endswith(".txt")]

# Dropdown to select a prompt
selected_person_name_validation_prompt = st.sidebar.selectbox("Select a Person Name Validation System Prompt", ["Select a prompt"] + person_name_prompt_files, key="select_person_name_validation_prompt")

# Display the content of the selected prompt
if selected_person_name_validation_prompt != "Select a prompt":
    prompt_path = os.path.join(person_name_prompt_folder, selected_person_name_validation_prompt)
    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            validate_person_names_system_prompt = file.read()
        validate_person_names_system_prompt = st.sidebar.text_area(
            "Edit system_prompt_for_person_name_validation",
            validate_person_names_system_prompt,
            height=150,
            key="system_prompt_for_person_name_validation"
        )

        
    
    except Exception as e:
        st.error(f"Error reading the selected prompt: {e}")

    # Button to delete the selected prompt
    if st.sidebar.button("Delete Selected Prompt", key="d2"):
        try:
            os.remove(prompt_path)
            st.sidebar.success(f"Prompt '{selected_person_name_validation_prompt}' deleted successfully!")
            st.experimental_rerun()  # Refresh the app to update the dropdown
        except Exception as e:
            st.sidebar.error(f"Error deleting the prompt: {e}")
            
            
# Button to show the "Create New Prompt" section
if st.sidebar.button("Create New Person Name Validation Prompt"):
    st.session_state.show_create_prompt = True
    
    
# Dynamically display the "Create New Prompt" section if the button is clicked
if st.session_state.get("show_create_prompt", False):
    st.sidebar.subheader("Create a New Person Name Validation Prompt")
    new_person_name_validation_prompt_name = st.sidebar.text_input("Enter a name for the new person name validation prompt (e.g., 'new_prompt.txt')", "")
    new_person_name_validation_prompt_content = st.sidebar.text_area("Enter the content for the new person name validation prompt", "", height=300)

    if st.sidebar.button("Save New Person Name Validation Prompt"):
        if new_person_name_validation_prompt_name and new_person_name_validation_prompt_name.endswith(".txt"):
            new_prompt_path = os.path.join(person_name_prompt_folder, new_person_name_validation_prompt_name)
            if os.path.exists(new_prompt_path):
                st.warning(f"A prompt with the name '{new_person_name_validation_prompt_name}' already exists.")
            else:
                try:
                    with open(new_prompt_path, "w", encoding="utf-8") as file:
                        file.write(new_person_name_validation_prompt_content)
                    st.success(f"New prompt '{new_person_name_validation_prompt_name}' created successfully!")
                    st.session_state.show_create_prompt = False  # Hide the section after saving
                    st.experimental_rerun()  # Refresh the app to show the new prompt in the dropdown
                except Exception as e:
                    st.error(f"Error creating new prompt: {e}")
        else:
            st.warning("Please enter a valid name for the new prompt (must end with '.txt').")
            
if st.sidebar.button("Save Person Name Validation Prompt", key="save_person_name_validation_prompt"):
    try:
        # Save the updated content to a file
        with open(os.path.join(person_name_prompt_folder,selected_person_name_validation_prompt), "w", encoding="utf-8") as file:
            file.write(validate_person_names_system_prompt)
        st.sidebar.success("Person Name Validation Prompt saved successfully!")
    except Exception as e:
        st.sidebar.error(f"Error saving Person Name Validation Prompt: {e}")


# --------------------------- PERSON NAME VALIDATION PROMPT LAB END -----------------------------------------------------

# ------------------------------- SIDEBAR TO SET LLM SPECS --------------------------------------------------
# Set parameters like temperature, max tokens, etc.
st.sidebar.header("Model Parameters")
st.sidebar.subheader("default tested values are displayed")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)
max_tokens = st.sidebar.slider("Max Tokens", 1, 4096, 2048)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.9)
frequency_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 1.0, 0.0)
presence_penalty = st.sidebar.slider("Presence Penalty", 0.0, 1.0, 0.0)

# ----------------------------------- LLM SPECS END --------------------------------------------------------------------
        
# Process uploaded PDF
if uploaded_file:
    # Save uploaded file
    pdf_path = os.path.join("uploaded_files", uploaded_file.name)
    os.makedirs("uploaded_files", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"Uploaded file: {uploaded_file.name}")

    # Check if the output directories already exist
    if not os.path.exists("output_pages"):
        st.info("Splitting PDF into individual pages...")
        num_pages = split_parent_pdf_into_individual_pages(pdf_path)
        st.success(f"PDF split into {num_pages} pages.")
    else:
        st.info("PDF has already been split into pages.")

    if not os.path.exists("images") or len(os.listdir("images")) == 0:
        st.info("Converting PDF pages to images...")
        for file in sorted(os.listdir('./output_pages'), key=lambda x: int(x.split('.')[0])):
            if file.endswith('.pdf'):
                child_pdf_path = os.path.join('./output_pages', file)
                page_number = file.split('.')[0]  # Extract the page number from the filename
                convert_pdf_to_image(child_pdf_path, page_number)
        st.success("Converted PDF pages to images.")
    else:
        st.info("PDF pages have already been converted to images.")
    if selected_llm=="Mistral":
        if not os.path.exists("output_files") or len(os.listdir("output_files")) == 0:
            st.info("Processing pages with Azure LLM...")
            process_pdf(pdf_path)
            output_folder = "./output_files"
            st.session_state.default_correct_invoices = len([f for f in os.listdir(output_folder) if f.startswith("output_") and f.endswith(".pdf")])
            print(f"Default Correct Invoices: {st.session_state.default_correct_invoices}")
            st.success("Processing completed!")
            # Save the split_json_numerals list to a JSON file
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]  # Extract the base name of the uploaded file
            output_json_path = os.path.join("uploaded_files", f"{base_name}_splitpoints.json")  # Define the output JSON file path
            try:
                with open(output_json_path, "w", encoding="utf-8") as json_file:
                    json.dump(split_json_numerals, json_file, indent=4)
                print(f"Split points saved to {output_json_path}")
            except Exception as e:
                print(f"Error saving split points to JSON: {e}")
            else:
                st.info("PDF has already been processed.")
    if selected_llm=="ChatGPT":
        if not os.path.exists("output_files") or len(os.listdir("output_files")) == 0:
            st.info("Processing pages with ChatGPT LLM...")
            process_pdf_using_ChatGPT(pdf_path)
            output_folder = "./output_files"
            st.session_state.default_correct_invoices = len([f for f in os.listdir(output_folder) if f.startswith("output_") and f.endswith(".pdf")])
            print(f"Default Correct Invoices: {st.session_state.default_correct_invoices}")
            st.success("Processing completed!")
            # Save the split_json_numerals list to a JSON file
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]  # Extract the base name of the uploaded file
            output_json_path = os.path.join("uploaded_files", f"{base_name}_splitpoints.json")  # Define the output JSON file path
            try:
                with open(output_json_path, "w", encoding="utf-8") as json_file:
                    json.dump(split_json_numerals, json_file, indent=4)
                print(f"Split points saved to {output_json_path}")
            except Exception as e:
                print(f"Error saving split points to JSON: {e}")
            else:
                st.info("PDF has already been processed.")
        

# Add a button to download the log file
if st.button("Download Logs"):
    with open(log_file_path, "r") as log_file:
        st.download_button(
            label="Download Logs as TXT",
            data=log_file.read(),
            file_name="process_log.txt",
            mime="text/plain"
        )
    
    
# Display PDFs from the output_files folder with drag-and-drop functionality
st.header("View and Adjust Split PDFs")
output_folder = "./output_files"

# Add a button to trigger the splitter.py script
st.header("Run PDF Splitter")

uploaded_invoices_folder = "uploaded_files"
os.makedirs(uploaded_invoices_folder, exist_ok=True)  # Ensure the folder exists

# Check for PDF files in the "uploaded_invoices" folder
pdf_files = [f for f in os.listdir(uploaded_invoices_folder) if f.endswith(".pdf")]

if pdf_files:
    st.info(f"Found {len(pdf_files)} PDF(s) in the 'uploaded_invoices' folder.")
else:
    st.warning("No PDF files found in the 'uploaded_invoices' folder.")

import shutil
from io import BytesIO

# Button to download all files in the uploaded_files folder as a ZIP
st.header("Download Uploaded Files as ZIP")

if os.path.exists(uploaded_invoices_folder) and len(os.listdir(uploaded_invoices_folder)) > 0:
    # Create a ZIP file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for file_name in os.listdir(uploaded_invoices_folder):
            file_path = os.path.join(uploaded_invoices_folder, file_name)
            zip_file.write(file_path, arcname=file_name)  # Add file to the ZIP archive
    zip_buffer.seek(0)  # Reset buffer position to the beginning

    # Provide a download button for the ZIP file
    st.download_button(
        label="Download All Uploaded Files as ZIP",
        data=zip_buffer,
        file_name="uploaded_files.zip",
        mime="application/zip"
    )
else:
    st.warning("No files found in the 'uploaded_files' folder to download.")

st.info("If using Azure Website - don't click the Start PDF Splittler Button - instead start the Splitter on Local Machine")
if st.button("Start PDF Splitter"):
    if pdf_files:
        try:
            # Resolve the absolute path for the working directory dynamically
            cwd_path = os.path.dirname(os.path.abspath(__file__))
            st.info(f"Resolved working directory: {cwd_path}")

            # Check if the directory exists
            if not os.path.exists(cwd_path):
                st.error(f"The specified working directory does not exist: {cwd_path}")
                raise FileNotFoundError(f"The directory {cwd_path} does not exist.")

            for pdf_file in pdf_files:
                pdf_path = os.path.join(uploaded_invoices_folder, pdf_file)
                st.info(f"Processing file: {pdf_file}")

                # Run the splitter.py script with the PDF file as an argument
                result = subprocess.run(
                    ["xvfb-run","python", "splitter.py", pdf_path],  # Pass the PDF path as an argument
                    cwd=cwd_path,  # Use the resolved absolute path
                    capture_output=True,  # Capture stdout and stderr
                    text=True  # Decode output as text
                )

                # Display the output or errors from the script
                if result.returncode == 0:
                    st.success(f"Splitter script executed successfully for {pdf_file}!")
                    st.text(result.stdout)  # Display standard output
                else:
                    st.error(f"Error occurred while running the splitter script for {pdf_file}.")
                    st.text(result.stderr)  # Display error output
        except Exception as e:
            st.error(f"An exception occurred: {e}")
    else:
        st.warning("No PDF files to process in the 'uploaded_invoices' folder.")
        
        
# Section to upload multiple PDF files
st.header("Upload PDF Files for Classification")

# Allow multiple file uploads
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Button to process the uploaded files
if st.button("Upload and Clear Output Folder"):
    output_folder = "./output_files"
    
    # Clear the output_files folder
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        st.info("Cleared all files in the 'output_files' folder.")
    else:
        os.makedirs(output_folder, exist_ok=True)
        st.info("Created the 'output_files' folder.")

    # Save the uploaded files to the output_files folder
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(output_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"Uploaded file: {uploaded_file.name}")
    else:
        st.warning("No files were uploaded.")

# Section to manage system prompts
st.header("Manage Classification System Prompts")

# Ensure the "./prompt" folder exists
prompt_folder = "./prompt"
os.makedirs(prompt_folder, exist_ok=True)

# Get the list of available prompts
prompt_files = [f for f in os.listdir(prompt_folder) if f.endswith(".txt")]

# Dropdown to select a prompt
selected_prompt = st.selectbox("Select a System Prompt", ["Select a prompt"] + prompt_files)

st.info("Make sure to reselect the classification prompt after creation or editing of existing or new prompts alike")

prompt_content = ""
# Display the content of the selected prompt
if selected_prompt != "Select a prompt":
    prompt_path = os.path.join(prompt_folder, selected_prompt)
    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt_content = file.read()
        modified_content = st.text_area("Prompt Content", prompt_content, height=300, key="selected_prompt_content")

        # Button to save changes to the selected prompt
        if st.button("Save Changes"):
            try:
                with open(prompt_path, "w", encoding="utf-8") as file:
                    file.write(modified_content)
                st.success(f"Changes saved to {selected_prompt}")
            except Exception as e:
                st.error(f"Error saving changes: {e}")


    except Exception as e:
        st.error(f"Error reading the selected prompt: {e}")
        
    # Button to delete the selected prompt
    if st.button("Delete Selected Prompt", key="d1"):
        try:
            os.remove(prompt_path)
            st.sidebar.success(f"Prompt '{selected_prompt}' deleted successfully!")
            st.experimental_rerun()  # Refresh the app to update the dropdown
        except Exception as e:
            st.sidebar.error(f"Error deleting the prompt: {e}")

# Button to show the "Create New Prompt" section
if st.button("Create New Prompt"):
    st.session_state.show_create_prompt = True

# Dynamically display the "Create New Prompt" section if the button is clicked
if st.session_state.get("show_create_prompt", False):
    st.subheader("Create a New Prompt")
    new_prompt_name = st.text_input("Enter a name for the new prompt (e.g., 'new_prompt.txt')", "")
    new_prompt_content = st.text_area("Enter the content for the new prompt", "", height=300)

    if st.button("Save New Prompt"):
        if new_prompt_name and new_prompt_name.endswith(".txt"):
            new_prompt_path = os.path.join(prompt_folder, new_prompt_name)
            if os.path.exists(new_prompt_path):
                st.warning(f"A prompt with the name '{new_prompt_name}' already exists.")
            else:
                try:
                    with open(new_prompt_path, "w", encoding="utf-8") as file:
                        file.write(new_prompt_content)
                    st.success(f"New prompt '{new_prompt_name}' created successfully!")
                    st.session_state.show_create_prompt = False  # Hide the section after saving
                    st.experimental_rerun()  # Refresh the app to show the new prompt in the dropdown
                except Exception as e:
                    st.error(f"Error creating new prompt: {e}")
        else:
            st.warning("Please enter a valid name for the new prompt (must end with '.txt').")
            
            
# Section for classifying invoices
st.header("Classify All Invoices")

# Button to classify all invoices
if st.button("Classify All Invoices"):
    if os.path.exists(output_folder):
        st.info("Classifying all invoices...")
        try:
            # Call the process_invoice_classification function
            process_invoice_classification(prompt_content)
            st.success("All invoices have been classified successfully!")
            
            # Display the classification results
            if final_output:
                st.subheader("Classification Results")
                for invoice, classification in final_output.items():
                    st.write(f"{invoice}: {classification}")
            else:
                st.warning("No classifications were generated.")
        except Exception as e:
            st.error(f"An error occurred while classifying invoices: {e}")
    else:
        st.warning("The output_files folder does not exist.")

# Section to display the graph
if final_output:
    st.header("Invoice Classification Graph")
    generate_classification_graph(final_output)
    # Embed the graph in the Streamlit app
    with open("classification_graph.html", "r") as f:
        graph_html = f.read()
    components.html(graph_html, height=800)
    
    




