import streamlit as st
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
poppler_path = r"C:\Users\HP VICTUS\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

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
    images = convert_from_path(child_pdf_path, poppler_path=poppler_path)
    # Save the images to the subfolder
    for i, img in enumerate(images):
            image_path = os.path.join(images_output_folder, f'{page_number}.png')
            img.save(image_path, 'PNG')

    print(f"Saved images for {os.path.basename(child_pdf_path)} in {images_output_folder}")
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------

# Generate base64 URL for image
def generate_base64_url(image_path):
    """Generates a base64 URL for an image."""
    with open(image_path, "rb") as img_file:
        raw_data = img_file.read()
        image_data = base64.b64encode(raw_data).decode("utf-8")
    image_format = image_path.split('.')[-1]
    return f"data:image/{image_format};base64,{image_data}"

# -----------------------------------------------------------------------------------------------------------------------------------------------------------

system_prompt_for_invoice_split_LLM_vision = """ You are a helpful ai assistant meant to assist in clubbing together different PDFs into invoices.
The invoices are in German. You will be passed with the images of the pages of the PDF and then you have to extract the text from the images. 
The text that needs to extracted is according to an algorithm that I have desvised and I'll tell you more about it below.

The algorithm keeps track of the vendor name, invoice ID, and customer name or address and does the following:
a) If the vendor name is found and it is the same as the previous vendor name, it indicates a continuation of the same invoice.
b) If the vendor name is found and it is different from the previous vendor name, it indicates a new invoice.
c) If the invoice ID is found and it is the same as the previous invoice ID, it indicates a continuation of the same invoice.
d) If the invoice ID is found and it is different from the previous invoice ID, it indicates a new invoice.
e) If the customer address is found it indicates the starting page of the invoice.

Your goal is to extract the following fields, based on the visible text in the image:
- Vendor Name
- Invoice ID
- Customer Name



### Extraction Patterns:

- **Customer Name and Address Block Example:**
Frau
Maria Mustermann
Musterallee 1
54321 Musterdorf

- Extract only the Customer Name: `Maria Mustermann`

- **Invoice ID Example:**
  - Valid formats:
    - Rechnungsnummer: 1234567890
    - Rechnungs-ID: ABCD123456
    - Rechnung-Nr.: INV2024XY
  - Invoice IDs typically appear near the top of the document and close to the customer name or billing address.
  - Always look for these **German keywords** before extracting:
    - "Rechnungsnummer"
    - "Rechnungs-ID"
    - "Rechnung-Nr."

- Do NOT extract codes that:
  - Appear in tabular sections, such as those under "Referenz", "ISIN", or "Depot/Konto-Nummer"
  - Are short reference IDs like "K23000076" or ISINs
  - Appear near securities or fund transaction details

- Also ignore anything resembling a date or range:
  - 01.01.2020
  - 2020-12-31
  - 01.01.2020 - 31.12.2020

- Only extract the **actual invoice number** from the area where the **customer billing address appears** or where it's clearly labeled with the German keywords above.



 - **vendor Name Example:**
 Deka
Investments

Vendor Name: Deka Investments
- It usually appears at the top of the page, often styled as a logo or header.
- Also note that it might not cover the entire headers area and might be sitting in the top-right corner of the page, 
top-left corner of the page, or even in the middle of the page.
- It might also be in the form of a logo, so you need to be careful about that.


### ✅ Output Format

Output the following **only if at least one of Vendor Name or Customer Name is found**:

More elaboration for this condition: lets say you find the vendor name but not the customer name, then you will output the vendor name and invoice id if it is found.
If you find the customer name but not the vendor name, then you will output the customer name and invoice id if it is found.

➡ Output only:
Vendor Name: <vendor_name> 
Customer Name: <customer_name>
Invoice ID: <invoice_id> 

**Rules:**
- If **Vendor Name** is missing, omit the `Vendor Name:` line.
- If **Invoice ID** is missing, omit the `Invoice ID:` line.
- If **Customer Name** is missing, omit the `Customer Name:` line.

---


### ❗Special Case

If **neither Vendor Name nor Customer Name** is found:
➡ Output only:
Child Page

Do **not** include any other text or explanations.

---







Be accurate, consistent, and minimal.

Just adhere to the format and do not add any extra text or explanation strictly because i will then use regex-based pattern matching 
to route to the desired custom python logic using my custom python code """


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
        max_tokens=512,  # You can increase this if needed, but shorter helps prevent hallucination
        temperature=0.2,  # Maximum determinism
        top_p=0.9,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stop=["\n\n", "---", "Explanation", "Note:"]
    )

    return response.choices[0].message.content

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
hashMap={}


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
validate_vendor_names_system_prompt = """
You are an intelligent assistant that helps determine whether two company or vendor names refer to the same organization.

You will be given two vendor names, and your task is to decide if they refer to the **same company or organization**, even if the wording, spelling, or formatting differs.

These vendor names are from **German invoices**, so use your knowledge of German company naming conventions.

Examples of valid matches include:
- "Banque de Luxembourg" and "Banque Internationale à Luxembourg"
- "Deka Investments" and "DekaBank Deutsche Girozentrale"
- "Amazon Web Services Inc." and "AWS"
- "grün form Garten und Landschaft GmbH" and "GRÜNFORM GmbH"
- "Müller GmbH & Co. KG" and "Mueller KG"

You should account for:
- Synonyms or reworded names
- Capitalization differences (e.g., GRÜNFORM vs grün form)
- German umlauts and character variations (e.g., Ü vs UE)
- Merged or split words (e.g., "grün form" vs "GRÜNFORM")
- Legal suffixes such as "GmbH", "AG", "KG" being present or missing
- Abbreviations or expansions (e.g., AWS vs Amazon Web Services)
- Different language variants

Your answer must be one of the following two words only:
- Yes
- No

No explanations. No punctuation. No reasoning. Just output exactly one of: **Yes** or **No**.
This is critical for downstream programmatic processing.
"""
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
        max_tokens=2048,
        temperature=0.2,
        top_p=0.9,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        model=model_name
    )
    
    print(f"Response: {response.choices[0].message.content}")
    return response.choices[0].message.content 

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to check if the invoice ID has changed

def is_invoice_id_changed(invoice_id_1: str, invoice_id_2: str) -> bool:
    """
    Compares the numeric parts of two invoice IDs and determines if they are different.

    Args:
        invoice_id_1 (str): The first invoice ID.
        invoice_id_2 (str): The second invoice ID.

    Returns:
        bool: True if the numeric parts are different, False otherwise.
    """
    # Extract numeric parts using regex
    numeric_part_1 = re.sub(r'\D', '', invoice_id_1)
    numeric_part_2 = re.sub(r'\D', '', invoice_id_2)

    return numeric_part_1 != numeric_part_2

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
validate_person_names_system_prompt = """
You are an intelligent assistant that helps determine whether two person names refer to the **same individual**, even if the wording or formatting differs.

You will be given two person names. Your task is to decide if they refer to the **same person**.

These names are from **German business documents**, such as invoices, contracts, or email signatures. Use your knowledge of German naming conventions and honorifics.

Examples of valid matches include:
- "Max Mustermann" and "Maximilian Mustermann"
- "Dr. Max Mustermann" and "Max Mustermann" (only if context suggests both refer to the same person)
- "Anna-Lena Schmidt" and "Anna Lena Schmidt"
- "Müller" and "Mueller" (considering umlaut substitution)

Examples of non-matches include:
- "Dr. Max Mustermann" and "Max Mustermann GmbH" (person vs company)
- "Max Mustermann" and "Erika Mustermann"
- "Max Mustermann" and "Mustermann KG"
- "Dr. Max Mustermann" and "Herr Max Mustermann" (only if context is unclear)

You should consider:
- German honorifics and titles (e.g., Dr., Prof., Herr, Frau)
- Variants of umlauts (e.g., Ü vs UE)
- Capitalization differences
- Hyphenated or compound names
- Whether one name is a company or organization (e.g., contains GmbH, KG, AG)

Your answer must be one of the following two words only:
- Yes
- No

No explanations. No punctuation. No reasoning. Just output exactly one of: **Yes** or **No**.
This is critical for downstream programmatic processing.
"""

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
        max_tokens=2048,
        temperature=0.2,
        top_p=0.9,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        model=model_name
    )
    
    print(f"Response: {response.choices[0].message.content}")
    return response.choices[0].message.content

#------------------------------------------------------------------------------------------------------------------------------------------------------------------ 

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
    """
    Creates a new PDF containing only the specified pages from the input PDF.

    :param input_pdf: Path to the original multi-page PDF.
    :param output_pdf: Path to save the new PDF.
    :param pages: List of page numbers (1-based index) to include in the new PDF.
    """
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
# Function to delete a folder and its contents
def delete_folder(folder_path):
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

system_prompt_for_invoice_classification = f"""
You are a highly specialized document classification AI focused on multi-page German tax, financial, and medical documents.

Your task is to classify each individual page into one of the following categories:

"Capital Returns"

"Medical Invoice"

"Craftsman Invoice"

"None" (if the page does not clearly fit any of the above)

🚩 Core Rules
Analyze each page independently.
Do not use content from other pages for classification.

Base your classification on visible content only, including:

Document headers and section titles

Organization names and professional titles

Recognizable terms and phrases that signal document type

You must consider structural and transactional details like:

Presence of cost breakdowns

VAT (MwSt/USt) lines

Document titles like "Rechnung", "Privatrezept", etc.

Do not rely on fine legal disclaimers, footers, or isolated page numbers unless they indicate document purpose.

🎯 Category Definitions
"Capital Returns"
Typical keywords and patterns:

Terms: Kapitalertrag, Dividende, Zinsen, Wertpapier

Documents: Jahressteuerbescheinigung, Erträgnisaufstellung

Tax terms: Kapitalertragsteuer, Abgeltungsteuer

Institutions: Depot, Fonds, Portfolio

Exclusion Criteria:

No signs of medical services or itemized services

No VAT/MwSt service breakdowns

"Medical Invoice"
Typical indicators:

Headers: Arzt, Praxis, Heilpraktiker, Krankenhaus, Apotheke

Titles: Dr. med., Facharzt, Physiotherapeut

Medical terms: Diagnose, Behandlung, Therapie, Patient

Billing codes: GOP, EBM, ICD-10

Insurance: Krankenkasse, Privatrezept, Kassenrezept

"Craftsman Invoice"
Typical indicators:

Service keywords: Reparatur, Montage, Installation, Handwerkerleistung

VAT breakdown: Presence of MwSt/USt lines and totals

Document type: Contains Rechnung or Rechnung Nr.

Trades: Sanitär, Heizung, Bauarbeiten, Elektroarbeiten

Companies: GmbH & Co. KG, Meisterbetrieb, Handwerksbetrieb

Often references a property address

"None"
Assign this category if:

The page does not conclusively match any of the three categories above

It contains ambiguous, generic, or mixed content

You're unable to determine purpose confidently from visible elements

------------
Output: return only the class name and nothing else
"""

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to classify the invoice using the LLM

def call_LLM_for_invoice_classification(system_prompt, image_data_url):
   
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
        max_tokens=512,  # You can increase this if needed, but shorter helps prevent hallucination
        temperature=0.2,  # Maximum determinism
        top_p=0.9,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stop=["\n\n", "---", "Explanation", "Note:"]
    )
    
    return response.choices[0].message.content

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
final_output = {}

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

def process_invoice_classification(classification_prompt):
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
                    ["python", "splitter.py", pdf_path],  # Pass the PDF path as an argument
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
st.header("Manage System Prompts")

# Ensure the "./prompt" folder exists
prompt_folder = "./prompt"
os.makedirs(prompt_folder, exist_ok=True)

# Get the list of available prompts
prompt_files = [f for f in os.listdir(prompt_folder) if f.endswith(".txt")]

# Dropdown to select a prompt
selected_prompt = st.selectbox("Select a System Prompt", ["Select a prompt"] + prompt_files)

prompt_content = ""
# Display the content of the selected prompt
if selected_prompt != "Select a prompt":
    prompt_path = os.path.join(prompt_folder, selected_prompt)
    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt_content = file.read()
        st.text_area("Prompt Content", prompt_content, height=300, key="selected_prompt_content")
    except Exception as e:
        st.error(f"Error reading the selected prompt: {e}")
            
            
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
    
    




