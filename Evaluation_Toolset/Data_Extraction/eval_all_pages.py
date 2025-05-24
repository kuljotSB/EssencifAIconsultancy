import streamlit as st
import os
from PyPDF2 import PdfReader, PdfWriter
import base64
import re
import uuid
import os
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import ImageContentItem, ImageUrl, TextContentItem
from azure.cosmos import CosmosClient
import pandas as pd

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
mistral_ocr_key = os.getenv("MISTRAL_OCR_KEY")
gpt_model_deployment_name = os.getenv("GPT_MODEL_DEPLOYMENT_NAME")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
project_connection_string = os.getenv("PROJECT_CONNECTION_STRING")
mistral_small_deployment_name = os.getenv("MISTRAL_SMALL_MODEL_DEPLOYMENT_NAME")
mistral_small_endpoint = os.getenv("MISTRAL_SMALL_MODEL_ENDPOINT")
mistral_small_api_key = os.getenv("MISTRAL_SMALL_MODEL_API_KEY")
microsoft_phi_four_instruct_deployment_name = os.getenv("MICROSOFT_PHI4_MODEL_DEPLOYMENT_NAME")
microsoft_phi_four_instruct_endpoint = os.getenv("MICROSOFT_PHI4_MODEL_ENDPOINT")
microsoft_phi_four_instruct_api_key = os.getenv("MICROSOFT_PHI4_MODEL_API_KEY")
cosmosdb_connection_string = os.getenv("COSMOSDB_CONNECTION_STRING")


                 
# Giving option via dropdown to select multiple models
st.sidebar.header("Select Models")
model_options = {
    "Microsoft Phi-4 Instruct": microsoft_phi_four_instruct_deployment_name,
    "Mistral Small 2503": mistral_small_deployment_name,
    "OpenAI GPT-4-Vision-Preview": gpt_model_deployment_name
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

#-------------------------------AZURE COSMOS DB HELPER FUNCTIONS-------------------------------------------------------------------------------------

#creating a function to connect to the Azure Cosmos DB with client creation
def get_cosmos_client():
    """Get the Cosmos DB client."""
    try:
        client = CosmosClient.from_connection_string(cosmosdb_connection_string)
        return client
    except Exception as e:
        st.error(f"Error connecting to Cosmos DB: {e}")
        return None

# creating a function to fetch the database
def get_database(cosmos_client, database_name):
    """Get the Cosmos DB database."""
    try:
        database = cosmos_client.get_database_client(database_name)
        return database
    except Exception as e:
        st.error(f"Error fetching database: {e}")
        return None
    

# creating a function to fetch the container
def get_container(database, container_name):
    """Get the Cosmos DB container."""
    try:
        container = database.get_container_client(container_name)
        return container
    except Exception as e:
        st.error(f"Error fetching container: {e}")
        return None
    
#creating a function to upsert the document type and fields table into the Cosmos DB
def upsert_document_fields(doc_type, fields_table):
    try:
        client = get_cosmos_client()
        database = get_database(client, "kiebidz")
        container = get_container(database, "DocumentFields")
        
        item = {
            "id": str(uuid.uuid4()),
            "DocType": doc_type,
            "fields": fields_table
        }
        container.upsert_item(item)
        st.success(f"Fields for '{doc_type}' upserted into Cosmos DB.")
    except Exception as e:
        st.error(f"Error upserting into Cosmos DB: {e}")

def get_all_doc_types(container):
    try:
        return list(container.read_all_items())
    except Exception as e:
        st.error(f"Error reading from Cosmos DB: {e}")
        return []

#--------------------------------AZURE COSMOS DB HELPER FUNCTIONS ENDS HERE---------------------------------------------------------

#-----------------------MAIN PERSONA STARTS HERE---------------------------------------------

# ...existing code...

# Cosmos DB helpers for Persona
def upsert_persona(user, persona_name, persona):
    try:
        client = get_cosmos_client()
        database = get_database(client, "kiebidz")
        container = get_container(database, "Persona")
        item = {
            "id": str(uuid.uuid4()),
            "User": user,
            "PersonaName": persona_name,
            "Persona": persona
        }
        container.upsert_item(item)
        st.success(f"Persona '{persona_name}' for user '{user}' upserted into Cosmos DB.")
    except Exception as e:
        st.error(f"Error upserting Persona: {e}")

def get_all_personas(container):
    try:
        return list(container.read_all_items())
    except Exception as e:
        st.error(f"Error reading Personas: {e}")
        return []

def update_persona(item_id, user, persona_name, persona):
    try:
        client = get_cosmos_client()
        database = get_database(client, "kiebidz")
        container = get_container(database, "Persona")
        item = {
            "id": item_id,
            "User": user,
            "PersonaName": persona_name,
            "Persona": persona
        }
        container.replace_item(item_id, item, user)
        st.success(f"Persona '{persona_name}' for '{user}' updated.")
    except Exception as e:
        st.error(f"Error updating Persona: {e}")

def delete_persona(item_id, user):
    try:
        client = get_cosmos_client()
        database = get_database(client, "kiebidz")
        container = get_container(database, "Persona")
        container.delete_item(item=item_id, partition_key=user)
        st.success(f"Persona for '{user}' deleted.")
    except Exception as e:
        st.error(f"Error deleting Persona: {e}")

# UI for Persona management
st.header("Manage Personas")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Create New Persona")
    new_user = st.text_input("User", key="persona_user")
    new_persona_name = st.text_input("Persona Name", key="persona_name")
    new_persona = st.text_area("Persona", key="persona_text")
    if st.button("Save Persona", key="save_persona_btn"):
        if new_user.strip() and new_persona_name.strip() and new_persona.strip():
            upsert_persona(new_user.strip(), new_persona_name.strip(), new_persona.strip())
        else:
            st.warning("Please fill in User, Persona Name, and Persona.")

with col2:
    st.subheader("Edit or Delete Persona")
    client = get_cosmos_client()
    personas = []
    if client:
        db = client.get_database_client("kiebidz")
        container = db.get_container_client("Persona")
        personas = get_all_personas(container)

    user_options = [f"{item['User']} - {item.get('PersonaName','')}" for item in personas] if personas else []
    selected_user_combo = st.selectbox("Select User & PersonaName", user_options, key="select_persona_user") if user_options else None

    if selected_user_combo:
        selected_item = next((item for item in personas if f"{item['User']} - {item.get('PersonaName','')}" == selected_user_combo), None)
        if selected_item:
            edited_persona_name = st.text_input("Edit Persona Name", value=selected_item.get("PersonaName",""), key="edit_persona_name")
            edited_persona = st.text_area("Edit Persona", value=selected_item["Persona"], key="edit_persona_text")
            if st.button("Update Persona", key="update_persona_btn"):
                update_persona(selected_item["id"], selected_item["User"], edited_persona_name, edited_persona)
            if st.button("Delete Persona", key="delete_persona_btn"):
                delete_persona(selected_item["id"], selected_item["User"])
               

# ...existing code...
                
#-----------------------MAIN PERSONA ENDS HERE------------------------------------------------------------------


#---------------------MAIN PROMPT BODY STARTS HERE-------------------------------------------------------------------

# ...existing code...

# Cosmos DB helpers for MainPromptBody
def upsert_main_prompt(user, main_prompt_name, main_prompt):
    try:
        client = get_cosmos_client()
        database = get_database(client, "kiebidz")
        container = get_container(database, "MainPromptBody")
        item = {
            "id": str(uuid.uuid4()),
            "User": user,
            "MainPromptName": main_prompt_name,
            "MainPrompt": main_prompt
        }
        container.upsert_item(item)
        st.success(f"MainPrompt '{main_prompt_name}' for user '{user}' upserted into Cosmos DB.")
    except Exception as e:
        st.error(f"Error upserting MainPrompt: {e}")

def get_all_main_prompts(container):
    try:
        return list(container.read_all_items())
    except Exception as e:
        st.error(f"Error reading MainPrompts: {e}")
        return []

def update_main_prompt(item_id, user, main_prompt_name, main_prompt):
    try:
        client = get_cosmos_client()
        database = get_database(client, "kiebidz")
        container = get_container(database, "MainPromptBody")
        item = {
            "id": item_id,
            "User": user,
            "MainPromptName": main_prompt_name,
            "MainPrompt": main_prompt
        }
        container.replace_item(item_id, item, user)
        st.success(f"MainPrompt '{main_prompt_name}' for '{user}' updated.")
    except Exception as e:
        st.error(f"Error updating MainPrompt: {e}")

def delete_main_prompt(item_id, user):
    try:
        client = get_cosmos_client()
        database = get_database(client, "kiebidz")
        container = get_container(database, "MainPromptBody")
        container.delete_item(item=item_id, partition_key=user)
        st.success(f"MainPrompt for '{user}' deleted.")
    except Exception as e:
        st.error(f"Error deleting MainPrompt: {e}")

# UI for MainPromptBody management
st.header("Manage Main Prompts")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Create New MainPrompt")
    new_user_mp = st.text_input("User", key="mainprompt_user")
    new_main_prompt_name = st.text_input("MainPrompt Name", key="mainprompt_name")
    new_main_prompt = st.text_area("MainPrompt", key="mainprompt_text")
    if st.button("Save MainPrompt", key="save_mainprompt_btn"):
        if new_user_mp.strip() and new_main_prompt_name.strip() and new_main_prompt.strip():
            upsert_main_prompt(new_user_mp.strip(), new_main_prompt_name.strip(), new_main_prompt.strip())
        else:
            st.warning("Please fill in User, MainPrompt Name, and MainPrompt.")

with col2:
    st.subheader("Edit or Delete MainPrompt")
    client = get_cosmos_client()
    main_prompts = []
    if client:
        db = client.get_database_client("kiebidz")
        container = db.get_container_client("MainPromptBody")
        main_prompts = get_all_main_prompts(container)

    main_prompt_options = [f"{item['User']} - {item.get('MainPromptName','')}" for item in main_prompts] if main_prompts else []
    selected_main_prompt_combo = st.selectbox("Select User & MainPromptName", main_prompt_options, key="select_mainprompt_user") if main_prompt_options else None

    if selected_main_prompt_combo:
        selected_item = next((item for item in main_prompts if f"{item['User']} - {item.get('MainPromptName','')}" == selected_main_prompt_combo), None)
        if selected_item:
            edited_main_prompt_name = st.text_input("Edit MainPrompt Name", value=selected_item.get("MainPromptName",""), key="edit_mainprompt_name")
            edited_main_prompt = st.text_area("Edit MainPrompt", value=selected_item["MainPrompt"], key="edit_mainprompt_text")
            if st.button("Update MainPrompt", key="update_mainprompt_btn"):
                update_main_prompt(selected_item["id"], selected_item["User"], edited_main_prompt_name, edited_main_prompt)
            if st.button("Delete MainPrompt", key="delete_mainprompt_btn"):
                delete_main_prompt(selected_item["id"], selected_item["User"])
                
# ...existing code...

#---------------------MAIN PROMPT BODY ENDS HERE-------------------------------------------------------------------

#----------------------DOCTYPE AND FIELDS TABLE CREATION STARTS HERE---------------------------------------------

st.header("Define Fields to be Extracted")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Create or Edit DocType")
    # If editing, prefill with selected values
    if st.session_state.get("edit_mode", False):
        doc_type = st.text_input("Edit Document Type", st.session_state.get("edit_doc_type", ""), key="doc_type_input_edit")
        fields_table = st.session_state.get("edit_fields_table", [{"Field": "", "Description": "", "Format": ""}])
        fields_df = pd.DataFrame(fields_table)
        edited_df = st.data_editor(
            fields_df,
            num_rows="dynamic",
            key="fields_editor_edit",
            width=500,
            height=300,
        )
        if st.button("Save Changes", key="save_edit_btn"):
            # Remove old DocType and upsert new one
            client = get_cosmos_client()
            if client:
                db = client.get_database_client("kiebidz")
                container = db.get_container_client("DocumentFields")
                # Find and delete old item
                for item in get_all_doc_types(container):
                    if item["DocType"] == st.session_state["edit_doc_type"]:
                        container.delete_item(item, partition_key=item["DocType"])
                        break
                upsert_document_fields(doc_type.strip(), edited_df.to_dict(orient="records"))
                st.success(f"DocType '{doc_type}' updated!")
                st.session_state.edit_mode = False
        if st.button("Cancel Edit", key="cancel_edit_btn"):
            st.session_state.edit_mode = False
    else:
        doc_type = st.text_input("Enter Document Type (e.g., Invoice, Receipt):", key="doc_type_input")
        if "fields_table" not in st.session_state:
            st.session_state.fields_table = [
                {"Field": "", "Description": "", "Format": ""}
            ]
        fields_df = pd.DataFrame(st.session_state.fields_table)
        edited_df = st.data_editor(
            fields_df,
            num_rows="dynamic",
            key="fields_editor",
            width=500,
            height=300,
        )
        st.session_state.fields_table = edited_df.to_dict(orient="records")
        if st.button("Save/Upsert Fields Table to Cosmos DB", key="save_new_btn"):
            if doc_type.strip():
                upsert_document_fields(doc_type.strip(), st.session_state.fields_table)
            else:
                st.warning("Please enter a document type.")

with col2:
    st.subheader("Select, Edit, or Delete DocType")
    client = get_cosmos_client()
    doc_types = []
    if client:
        db = client.get_database_client("kiebidz")
        container = db.get_container_client("DocumentFields")
        doc_types = get_all_doc_types(container)

    doc_type_options = [item["DocType"] for item in doc_types] if doc_types else []
    selected_doc_type = st.selectbox("Select a Document Type:", doc_type_options, key="select_doc_type") if doc_type_options else None

    if selected_doc_type:
        selected_item = next((item for item in doc_types if item["DocType"] == selected_doc_type), None)
        if selected_item:
            fields_table = selected_item.get("fields", [])
            st.dataframe(pd.DataFrame(fields_table))
            # Edit button
            if st.button("Edit Selected DocType", key="edit_btn"):
                st.session_state.edit_mode = True
                st.session_state.edit_doc_type = selected_item["DocType"]
                st.session_state.edit_fields_table = fields_table
            # Delete button
            if st.button("Delete Selected DocType", key="delete_btn"):
                try:
                    container.delete_item(selected_item, partition_key=selected_item["DocType"])
                    st.success(f"Deleted DocType: {selected_doc_type}")
                    
                except Exception as e:
                    st.error(f"Error deleting DocType: {e}")

#----------------------DOCTYPE AND FIELDS TABLE CREATION ENDS HERE---------------------------------------------


#-------------------------CONCLUSION AND FINALIZATION PROMPT STARTS HERE---------------------------------------------
# Cosmos DB helpers for Conclusion
def upsert_conclusion(user, conclusion_name, conclusion):
    try:
        client = get_cosmos_client()
        database = get_database(client, "kiebidz")
        container = get_container(database, "Conclusion")
        item = {
            "id": str(uuid.uuid4()),
            "User": user,
            "ConclusionName": conclusion_name,
            "Conclusion": conclusion
        }
        container.upsert_item(item)
        st.success(f"Conclusion '{conclusion_name}' for user '{user}' upserted into Cosmos DB.")
    except Exception as e:
        st.error(f"Error upserting Conclusion: {e}")

def get_all_conclusions(container):
    try:
        return list(container.read_all_items())
    except Exception as e:
        st.error(f"Error reading Conclusions: {e}")
        return []

def update_conclusion(item_id, user, conclusion_name, conclusion):
    try:
        client = get_cosmos_client()
        database = get_database(client, "kiebidz")
        container = get_container(database, "Conclusion")
        item = {
            "id": item_id,
            "User": user,
            "ConclusionName": conclusion_name,
            "Conclusion": conclusion
        }
        container.replace_item(item_id, item, user)
        st.success(f"Conclusion '{conclusion_name}' for '{user}' updated.")
    except Exception as e:
        st.error(f"Error updating Conclusion: {e}")

def delete_conclusion(item_id, user):
    try:
        client = get_cosmos_client()
        database = get_database(client, "kiebidz")
        container = get_container(database, "Conclusion")
        container.delete_item(item=item_id, partition_key=user)
        st.success(f"Conclusion for '{user}' deleted.")
    except Exception as e:
        st.error(f"Error deleting Conclusion: {e}")

# UI for Conclusion management
st.header("Manage Conclusions")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Create New Conclusion")
    new_user_conc = st.text_input("User", key="conclusion_user")
    new_conclusion_name = st.text_input("Conclusion Name", key="conclusion_name")
    new_conclusion = st.text_area("Conclusion", key="conclusion_text")
    if st.button("Save Conclusion", key="save_conclusion_btn"):
        if new_user_conc.strip() and new_conclusion_name.strip() and new_conclusion.strip():
            upsert_conclusion(new_user_conc.strip(), new_conclusion_name.strip(), new_conclusion.strip())
        else:
            st.warning("Please fill in User, Conclusion Name, and Conclusion.")

with col2:
    st.subheader("Edit or Delete Conclusion")
    client = get_cosmos_client()
    conclusions = []
    if client:
        db = client.get_database_client("kiebidz")
        container = db.get_container_client("Conclusion")
        conclusions = get_all_conclusions(container)

    conclusion_options = [f"{item['User']} - {item.get('ConclusionName','')}" for item in conclusions] if conclusions else []
    selected_conclusion_combo = st.selectbox("Select User & ConclusionName", conclusion_options, key="select_conclusion_user") if conclusion_options else None

    if selected_conclusion_combo:
        selected_item = next((item for item in conclusions if f"{item['User']} - {item.get('ConclusionName','')}" == selected_conclusion_combo), None)
        if selected_item:
            edited_conclusion_name = st.text_input("Edit Conclusion Name", value=selected_item.get("ConclusionName",""), key="edit_conclusion_name")
            edited_conclusion = st.text_area("Edit Conclusion", value=selected_item["Conclusion"], key="edit_conclusion_text")
            if st.button("Update Conclusion", key="update_conclusion_btn"):
                update_conclusion(selected_item["id"], selected_item["User"], edited_conclusion_name, edited_conclusion)
            if st.button("Delete Conclusion", key="delete_conclusion_btn"):
                delete_conclusion(selected_item["id"], selected_item["User"])
                
# ...existing code...

#-------------------------CONCLUSION AND FINALIZATION PROMPT ENDS HERE---------------------------------------------

#--------------------GENERATE FINAL PROMPT STARTS HERE-----------------------------------------------------------------

st.header("Generate Complete Prompt")

# Use session state to persist the consolidated prompt
if "consolidated_prompt" not in st.session_state:
    st.session_state.consolidated_prompt = ""

# Select MainPrompt
client = get_cosmos_client()

persona = ""
if client:
    db = client.get_database_client("kiebidz")
    container = db.get_container_client("Persona")
    personas = get_all_personas(container)
    persona_options = [f"{item['User']} - {item.get('PersonaName','')}" for item in personas] if personas else []
    selected_persona_combo = st.selectbox("Select Persona", persona_options, key="select_persona_for_consolidate") if persona_options else None
    if selected_persona_combo:
        selected_item = next((item for item in personas if f"{item['User']} - {item.get('PersonaName','')}" == selected_persona_combo), None)
        if selected_item:
            persona = selected_item["Persona"]

main_prompt = ""
if client:
    db = client.get_database_client("kiebidz")
    container = db.get_container_client("MainPromptBody")
    main_prompts = get_all_main_prompts(container)
    main_prompt_options = [f"{item['User']} - {item.get('MainPromptName','')}" for item in main_prompts] if main_prompts else []
    selected_main_prompt_combo = st.selectbox("Select MainPrompt", main_prompt_options, key="select_mainprompt_for_consolidate") if main_prompt_options else None
    if selected_main_prompt_combo:
        selected_item = next((item for item in main_prompts if f"{item['User']} - {item.get('MainPromptName','')}" == selected_main_prompt_combo), None)
        if selected_item:
            main_prompt = selected_item["MainPrompt"]

fields_json = ""
if client:
    db = client.get_database_client("kiebidz")
    container = db.get_container_client("DocumentFields")
    doc_types = get_all_doc_types(container)
    doc_type_options = [item["DocType"] for item in doc_types] if doc_types else []
    selected_doc_type = st.selectbox("Select DocType for Fields JSON", doc_type_options, key="select_doc_type_for_consolidate") if doc_type_options else None
    if selected_doc_type:
        selected_item = next((item for item in doc_types if item["DocType"] == selected_doc_type), None)
        if selected_item:
            import json
            fields_json = json.dumps(selected_item.get("fields", []), indent=2)

conclusion = ""
if client:
    db = client.get_database_client("kiebidz")
    container = db.get_container_client("Conclusion")
    conclusions = get_all_conclusions(container)
    conclusion_options = [f"{item['User']} - {item.get('ConclusionName','')}" for item in conclusions] if conclusions else []
    selected_conclusion_combo = st.selectbox("Select Conclusion", conclusion_options, key="select_conclusion_for_consolidate") if conclusion_options else None
    if selected_conclusion_combo:
        selected_item = next((item for item in conclusions if f"{item['User']} - {item.get('ConclusionName','')}" == selected_conclusion_combo), None)
        if selected_item:
            conclusion = selected_item["Conclusion"]

# Button to generate and display the consolidated prompt
if st.button("Generate Complete Prompt"):
    sample_output_format =f"""The output should strictly match this format: \n\n
    ------------SAMPLE OUTPUT FORMAT------------
    [
        {{
        "field":"name",
        "value":"John Doe"
        }}
    ]
    
    Please strictly and strictly don't return a response like this:
    
    ```json
    [
        {{
        "field":"name",
        "value":"John Doe"
        }}
    ]
    ```
    
    Wrong Response Format: 
        ```json
    [
        {{
        "field":"Name",
        "value":"Dr. Max Mustermann"
    }}
    ]
    ```
    
    Correct Response Format:
    
    [
        {{
        "field":"Name",
        "value":"Dr. Max Mustermann"
    }}
    ]
    
    This is because the json struct will be used to display in an excel table like format so the correct format is extremely important
    """
    
    json_extraction_instructions = f""" Extract only the fields specified below in the JSON format. The Field, Description, and Format are as follows: \n"""
    
    st.session_state.consolidated_prompt = f"{persona}\n\n{main_prompt}\n\n{json_extraction_instructions}{fields_json}\n\n{conclusion}\n\n{sample_output_format}"

# Always show the latest consolidated prompt from session state
st.text_area("Consolidated Prompt", value=st.session_state.consolidated_prompt, height=400)


#--------------------GENERATE FINAL PROMPT ENDS HERE-----------------------------------------------------------------

import shutil

def convert_pdf_to_image(child_pdf_path):
    images_output_folder = './images'
    os.makedirs(images_output_folder, exist_ok=True)

    # Clear images folder
    for file in os.listdir(images_output_folder):
        file_path = os.path.join(images_output_folder, file)
        if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
            os.remove(file_path)

    split_pdf_folder = "split_pdf_pages"
    # Clear split_pdf_pages folder
    if os.path.exists(split_pdf_folder):
        shutil.rmtree(split_pdf_folder)
    os.makedirs(split_pdf_folder, exist_ok=True)

    child_pdf_name = os.path.basename(child_pdf_path).split('.')[0]
    print(f"Child PDF Name: {child_pdf_name}")
    print(f"Processing: {child_pdf_path}")

    try:
        # Split the PDF into individual pages
        reader = PdfReader(child_pdf_path)
        for i, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            split_pdf_path = os.path.join(split_pdf_folder, f"{child_pdf_name}_page_{i + 1}.pdf")
            with open(split_pdf_path, "wb") as output_pdf:
                writer.write(output_pdf)
            print(f"Saved split PDF page: {split_pdf_path}")

        # Only process the just-created split PDFs
        for child_pdf in os.listdir(split_pdf_folder):
            if child_pdf.endswith(".pdf"):
                child_pdf_path = os.path.join(split_pdf_folder, child_pdf)
                images = convert_from_path(child_pdf_path)
                for i, img in enumerate(images):
                    image_name = f"{os.path.splitext(child_pdf)[0]}_page_{i + 1}.png"
                    image_path = os.path.join(images_output_folder, image_name)
                    img.save(image_path, "PNG")
                    print(f"Saved image: {image_path}")

        print(f"Saved images for {os.path.basename(child_pdf_path)} in {images_output_folder}")
    except Exception as e:
        print(f"An error occurred: {e}")

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
def generate_markdown_from_mistral_OCR(pdf_path):

    # Getting the base64 string
    base64_pdf = encode_pdf(pdf_path)

    
    client = Mistral(api_key=mistral_ocr_key)

    ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64_pdf}" 
        }
    )
    
    
    print(ocr_response.pages[0].markdown)
    return str((ocr_response))

# Creating Function to Call GPT-4 Vision Model
def call_openai_vision_model(user_prompt, pdf_path):
    
    # Creating the Azure OpenAI Client
    client = AzureOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_api_key,
        api_version="2025-01-01-preview"
    )
    
    #prepare the chat prompt
    image_dict = []
    
    
    if pdf_path is not None:
        images_folder = './images'
        image_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]
        # generating base64 URL for each image
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            base64_path = generate_base64_url(image_path)
            
            image_dict.append({
                "type":"text",
                "text": "\n"
            })
            
            image_dict.append({
                "type":"image_url",
                "image_url": {
                    "url": base64_path,
                }
            })
     
    # appending the user prompt finally at the end to the image_dict
    image_dict.append({
        "type":"text",
        "text":f"{user_prompt}"
    })   
    
    # Prepare the chat prompt
    chat_prompt = [
        {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a helpful AI assistant"
            }
        ]
    }
    ]
    
    chat_prompt.append({
                        "role":"user",
                        "content": image_dict
                    })
    
    # Call the OpenAI GPT-4 Vision model
    completion = client.chat.completions.create(
        model=gpt_model_deployment_name,
        messages=chat_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    
    print("Response from GPT-4 Vision Model: {}".format(completion.choices[0].message.content))
    
    return completion.choices[0].message.content

def call_ai_foundry_catalog_model(user_prompt, pdf_path, ai_foundry_model_endpoint, ai_foundry_model_api_key):
    client = ChatCompletionsClient(
        endpoint=ai_foundry_model_endpoint,
        credential=AzureKeyCredential(ai_foundry_model_api_key),
        api_version = "2024-05-01-preview"
    )

     #prepare the chat prompt
    image_dict = []
    
    if pdf_path is not None:
        # Get the list of image files in the images folder
        images_folder = './images'
        image_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]
        # generating base64 URL for each image
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            base64_path = generate_base64_url(image_path)
            
            image_dict.append(
                {
                    "type":"text",
                    "text": "\n"
                }
            )
            
            image_dict.append(
                {
                    "type":"image_url",
                    "image_url": {
                        "url": base64_path,
                    }
                }
            )
    
    #appending the user prompt finally at the end to the image_dict
    image_dict.append(
        {
            "type":"text",
            "text":f"{user_prompt}"
        }
    )
    
    # finally preparing the chat prompt/messages array
    
    messages = [
        {
            "role":"system",
            "content": "You are a helpful AI assistant"
        },
        {
            "role":"user",
            "content": image_dict
        }
    ]
    
    
    
    # Calling the model with the configured payload and params
    
    response = client.complete(
        messages = messages,
        max_tokens = max_tokens,
        temperature = temperature,
        top_p = top_p,
    )
    print("Response from AI Foundry Model: {}".format(response.choices[0].message.content))
    
    return response.choices[0].message.content
    


#---------------HELPER FUNCTIONS TO PROCESS PDF ENDS HERE-------------------------------------------------------------

#---------------------CODE FOR PROCESSING THE PDF STARTS HERE-------------------------------------------------------------

# function for processing the PDF with ChatGPT (with parallelism)
def process_pdf_with_chatGPT(pdf_path, user_prompt, mistral_ocr_markdown_response):
    def run_image_only():
        return "gpt_vision_response_image_only", call_openai_vision_model(user_prompt, pdf_path)

    def run_markdown_only():
        user_prompt_for_markdown_usage = (
            user_prompt
            + "\n\nThe markdown from the Mistral OCR run on the Invoice PDF which can be used as supplementing knowledge is as follows:\n\n"
            + mistral_ocr_markdown_response
        )
        return "gpt_vision_response_markdown_only", call_openai_vision_model(user_prompt_for_markdown_usage, None)

    def run_markdown_and_image():
        user_prompt_for_markdown_usage = (
            user_prompt
            + "\n\nThe markdown from the Mistral OCR run on the Invoice PDF which can be used as supplementing knowledge is as follows:\n\n"
            + mistral_ocr_markdown_response
        )
        return "gpt_vision_response_markdown_and_image", call_openai_vision_model(user_prompt_for_markdown_usage, pdf_path)

    results = {}
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_image_only),
            executor.submit(run_markdown_only),
            executor.submit(run_markdown_and_image),
        ]
        for future in as_completed(futures):
            key, value = future.result()
            print(f"{key.replace('_', ' ').title()}: {value}")
            results[key] = value

    return results
        


def process_pdf_with_ai_foundry_models(pdf_path, user_prompt, mistral_ocr_markdown_response, ai_foundry_model_endpoint, ai_foundry_model_api_key, model_prefix="ai_foundry_catalog_model_response"):
    def run_image_only():
        return f"{model_prefix}_image_only", call_ai_foundry_catalog_model(user_prompt, pdf_path, ai_foundry_model_endpoint, ai_foundry_model_api_key)
    def run_markdown_only():
        user_prompt_for_markdown_usage = (
            user_prompt
            + "\n\nThe markdown from the Mistral OCR run on the Invoice PDF which can be used as supplementing knowledge is as follows:\n\n"
            + mistral_ocr_markdown_response
        )
        return f"{model_prefix}_markdown_only", call_ai_foundry_catalog_model(user_prompt_for_markdown_usage, None, ai_foundry_model_endpoint, ai_foundry_model_api_key)
    def run_markdown_and_image():
        user_prompt_for_markdown_usage = (
            user_prompt
            + "\n\nThe markdown from the Mistral OCR run on the Invoice PDF which can be used as supplementing knowledge is as follows:\n\n"
            + mistral_ocr_markdown_response
        )
        return f"{model_prefix}_markdown_and_image", call_ai_foundry_catalog_model(user_prompt_for_markdown_usage, pdf_path, ai_foundry_model_endpoint, ai_foundry_model_api_key)
    results = {}
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_image_only),
            executor.submit(run_markdown_only),
            executor.submit(run_markdown_and_image),
        ]
        for future in as_completed(futures):
            key, value = future.result()
            results[key] = value
    return results


#-------------------------CODE FOR PROCESSING THE PDF ENDS HERE-------------------------------------------------------------

# ...existing code...

import json
import json

def display_field_value_table(model_name, results_dict):
    """
    Display results as a table with columns: Field Name, Value for each field in the model's JSON response.
    """
    st.markdown(f"**{model_name}**")
    for key, label in [
        ("gpt_vision_response_image_only", "GPT-4 Image Only"),
        ("gpt_vision_response_markdown_only", "GPT-4 Markdown Only"),
        ("gpt_vision_response_markdown_and_image", "GPT-4 Image+Markdown"),
        ("ai_foundry_catalog_model_response_image_only", "Mistral Small Image Only"),
        ("ai_foundry_catalog_model_response_markdown_only", "Mistral Small Markdown Only"),
        ("ai_foundry_catalog_model_response_markdown_and_image", "Mistral Small Image+Markdown"),
        ("microsoft_phi4_response_image_only", "Microsoft Phi-4 Image Only"),
        ("microsoft_phi4_response_markdown_only", "Microsoft Phi-4 Markdown Only"),
        ("microsoft_phi4_response_markdown_and_image", "Microsoft Phi-4 Image+Markdown"),
    ]:
        if key in results_dict and results_dict[key]:
            st.markdown(f"**{label}**")
            try:
                data = json.loads(results_dict[key])
                if isinstance(data, dict):
                    data = [data]
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    rows = []
                    for item in data:
                        if "field" in item and "value" in item:
                            rows.append({"Field Name": item["field"], "Value": item["value"]})
                        else:
                            for field, value in item.items():
                                rows.append({"Field Name": field, "Value": value})
                    if rows:
                        st.dataframe(rows, use_container_width=True, hide_index=True)
                    else:
                        st.info("No fields found in response.")
                else:
                    st.text_area("Raw Response", value=results_dict[key], height=200)
            except Exception:
                st.text_area("Raw Response", value=results_dict[key], height=200)

def build_comparison_table(*results_dicts):
    """
    Build a comparison table where each row is a field and each column is a model/variant.
    """
    field_map = {}
    variant_labels = [
        ("gpt_vision_response_image_only", "GPT-4 Image Only"),
        ("gpt_vision_response_markdown_only", "GPT-4 Markdown Only"),
        ("gpt_vision_response_markdown_and_image", "GPT-4 Image+Markdown"),
        ("ai_foundry_catalog_model_response_image_only", "Mistral Small Image Only"),
        ("ai_foundry_catalog_model_response_markdown_only", "Mistral Small Markdown Only"),
        ("ai_foundry_catalog_model_response_markdown_and_image", "Mistral Small Image+Markdown"),
        ("microsoft_phi4_response_image_only", "Microsoft Phi-4 Image Only"),
        ("microsoft_phi4_response_markdown_only", "Microsoft Phi-4 Markdown Only"),
        ("microsoft_phi4_response_markdown_and_image", "Microsoft Phi-4 Image+Markdown"),
    ]
    for results in results_dicts:
        for key, label in variant_labels:
            if key in results and results[key]:
                try:
                    data = json.loads(results[key])
                    if isinstance(data, dict):
                        data = [data]
                    for item in data:
                        if "field" in item and "value" in item:
                            fname, fval = item["field"], item["value"]
                            if fname not in field_map:
                                field_map[fname] = {}
                            field_map[fname][label] = fval
                        else:
                            for fname, fval in item.items():
                                if fname not in field_map:
                                    field_map[fname] = {}
                                field_map[fname][label] = fval
                except Exception:
                    continue
    rows = []
    all_labels = [label for _, label in variant_labels]
    for fname in sorted(field_map.keys()):
        row = {"Field Name": fname}
        for label in all_labels:
            row[label] = field_map[fname].get(label, "")
        rows.append(row)
    return pd.DataFrame(rows)

import io
def generate_html_report(pdf_base64, df, consolidated_prompt):
    # Convert DataFrame to HTML table
    table_html = df.to_html(index=False, border=1, classes="comparison-table", justify="center")
    # Replace newlines in prompt for HTML display
    prompt_html = consolidated_prompt.replace('\n', '<br>')
    # HTML template
    html = f"""
    <html>
    <head>
        <style>
            .container {{
                display: flex;
                flex-direction: row;
                gap: 40px;
            }}
            .pdf-preview {{
                flex: 1;
            }}
            .comparison-table {{
                flex: 2;
                border-collapse: collapse;
                width: 100%;
            }}
            .comparison-table th, .comparison-table td {{
                border: 1px solid #ddd;
                padding: 8px;
            }}
            .comparison-table th {{
                background-color: #f2f2f2;
                text-align: center;
            }}
            .prompt-section {{
                margin-bottom: 30px;
                padding: 15px;
                background: #f9f9f9;
                border: 1px solid #ccc;
                border-radius: 8px;
                font-family: monospace;
                white-space: pre-wrap;
            }}
        </style>
    </head>
    <body>
        <h2>PDF and Model Comparison Report</h2>
        <div class="prompt-section">
            <h3>Consolidated Prompt</h3>
            <div>{prompt_html}</div>
        </div>
        <div class="container">
            <div class="pdf-preview">
                <h3>Uploaded PDF Preview</h3>
                <iframe src="data:application/pdf;base64,{pdf_base64}" width="400" height="600" type="application/pdf"></iframe>
            </div>
            <div class="comparison-table">
                <h3>Field Comparison Table</h3>
                {table_html}
            </div>
        </div>
    </body>
    </html>
    """
    return html


# Section to upload a PDF file
st.header("Upload and Process PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
uploaded_invoices_folder = "uploaded_invoices"

if uploaded_file is not None:
    # Clear the uploaded_invoices folder before saving the new file
    if os.path.exists(uploaded_invoices_folder):
        for file in os.listdir(uploaded_invoices_folder):
            file_path = os.path.join(uploaded_invoices_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(uploaded_invoices_folder, exist_ok=True)

    # Save the uploaded file to a temporary location
    temp_pdf_path = os.path.join(uploaded_invoices_folder, uploaded_file.name)
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded file: {uploaded_file.name}")

    # Button to process the uploaded PDF
    if st.button("Process PDF"):
        st.info("Processing the uploaded PDF...")

        # Use the consolidated prompt from session state
        user_prompt = st.session_state.consolidated_prompt
        print(f"User Prompt: {user_prompt}")
        
        # Generating markdown from Mistral OCR
        mistral_ocr_markdown_response = generate_markdown_from_mistral_OCR(temp_pdf_path)
        
        # Convert the PDF to images
        convert_pdf_to_image(temp_pdf_path)
        print(f"Converted PDF to images and saved in {os.path.abspath('./images')}")
        
       # Run based on selected models
       
        gpt_results = None
        foundry_small_results = None
        foundry_large_results = None
        foundry_medium_results = None
        microsoft_phi_four_results = None

        if "OpenAI GPT-4-Vision-Preview" in selected_models:
            st.write("Running with OpenAI GPT-4-Vision-Preview...")
            gpt_results = process_pdf_with_chatGPT(temp_pdf_path, user_prompt, mistral_ocr_markdown_response)
            if isinstance(gpt_results, dict):
                st.subheader("GPT-4 Vision Model Results")
                st.json(gpt_results, expanded=True)
            else:
                st.subheader("GPT-4 Vision Model Results")
                st.text_area("GPT-4 Vision Model Results", value=str(gpt_results), height=400)

        if "Mistral Small 2503" in selected_models:
            st.write("Running with Mistral Small 2503 (AI Foundry)...")
            ai_foundry_model_endpoint = mistral_small_endpoint
            ai_foundry_model_api_key = mistral_small_api_key
            foundry_small_results = process_pdf_with_ai_foundry_models(
                temp_pdf_path, user_prompt, mistral_ocr_markdown_response, ai_foundry_model_endpoint, ai_foundry_model_api_key
            )
            if isinstance(foundry_small_results, dict):
                st.subheader("AI Foundry Model (Mistral Small 2503) Results")
                st.json(foundry_small_results, expanded=True)
            else:
                st.subheader("AI Foundry Model (Mistral Small 2503) Results")
                st.text_area("AI Foundry Model (Mistral Small 2503) Results", value=str(foundry_small_results), height=400)

        if "Microsoft Phi-4 Instruct" in selected_models:
            st.write("Running with Microsoft Phi 4 Instruct Model (AI Foundry)...")
            ai_foundry_model_endpoint = microsoft_phi_four_instruct_endpoint
            ai_foundry_model_api_key = microsoft_phi_four_instruct_api_key
            microsoft_phi_four_results = process_pdf_with_ai_foundry_models(
                temp_pdf_path, user_prompt, mistral_ocr_markdown_response, ai_foundry_model_endpoint, ai_foundry_model_api_key, model_prefix="microsoft_phi4_response"
            )
            if isinstance(microsoft_phi_four_results, dict):
                st.subheader("Microsoft Phi-4 Instruct Model Results")
                st.json(microsoft_phi_four_results, expanded=True)
            else:
                st.subheader("Microsoft Phi-4 Instruct Model Results")
                st.text_area("Microsoft Phi-4 Instruct Model Results", value=str(microsoft_phi_four_results), height=400)
                
        col_pdf, col_results = st.columns([1, 2])

        with col_pdf:
            st.subheader("Uploaded PDF Preview")
            st.write("Below is the uploaded PDF file:")
            with open(temp_pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

        with col_results:
            st.subheader("Field Comparison Table")
            # Only include non-None results
            results_to_compare = [r for r in [gpt_results, foundry_small_results, microsoft_phi_four_results] if r]
            if results_to_compare:
                df = build_comparison_table(*results_to_compare)
                st.dataframe(df, use_container_width=True, hide_index=True)
                html_report = generate_html_report(base64_pdf, df, st.session_state.consolidated_prompt)
                st.download_button(
                    label="Download HTML Report",
                    data=html_report,
                    file_name="pdf_comparison_report.html",
                    mime="text/html"
                )
            else:
                st.info("No results to display.")