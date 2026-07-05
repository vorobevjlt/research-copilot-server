# from fileinput import filename
# from pydoc import doc
# from celery.exceptions import CeleryError
import os
from celery import Celery
import supabase
from storage import s3_client, supabase, BUCKET_NAME
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.html import partition_html

celery_app = Celery(
    'document_processor',
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)


def update_status(document_id: str, status: str, details: dict = None):
    """ Update document processing status with optional details """

    # Get current document 
    result = supabase.table("project_documents").select("processing_details").eq("id", document_id).execute()

    # Start with existing details or empty dict
    current_details = {}

    if result.data and result.data[0]["processing_details"]:
        current_details = result.data[0]["processing_details"]

    
    # Add new details if provided
    if details: 
        current_details.update(details)
    

    # Update document 
    supabase.table("project_documents").update({
        "processing_status": status, 
        "processing_details": current_details
    }).eq("id", document_id).execute()

def analyze_elements(elements):
    """ Count different types of elements found in the document """

    text_count = 0
    table_count = 0
    image_count = 0
    title_count = 0
    other_count = 0

    # Go through each element and count what type it is 
    for element in elements: 
        element_name = type(element).__name__ #Get the class name like "Table" or "NarrativeText"

        if element_name == "Table":
            table_count += 1
        elif element_name == "Image": 
            image_count += 1
        elif element_name in ["Title", "Header"]:
            title_count += 1
        elif element_name in ["NarrativeText", "Text", "ListItem", "FigureCaption"]:
            text_count += 1
        else:
            other_count += 1

    # Return a simple dictionary
    return {
        "text": text_count,
        "tables": table_count,
        "images": image_count,
        "titles": title_count,
        "other": other_count
    }


def partition_document(temp_file: str, file_type: str, source_type: str = "file"):
    if source_type == "url":
        pass
    
    if file_type == "pdf":
        return partition_pdf(
            filename=temp_file,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True
        )

def download_and_partition(document_id: str, doc: dict):
    print("Start download docs")
    source_type = doc.get("source_type", "file")

    if source_type == "url":
        pass
    else:
        s3_key = doc["s3_key"]
        filename = doc["file_name"]
        file_type = filename.split(".")[-1].lower()

        temp_file = f"/tmp/{document_id}.{file_type}"
        s3_client.download_file(BUCKET_NAME, s3_client, temp_file)
        elements = partition_document(temp_file, file_type, source_type="file")
    
    elements_summary = analyze_elements(elements)   

    update_status(document_id, "chunking", {
        "partitioning": {
            "elements_found": elements_summary
            }
    })
    os.remove(temp_file)

    return elements

@celery_app.task
def process_document(document_id: str):

    try:
        doc_results = supabase.table("project_document")\
            .select("*")\
            .eq("id", document_id).execute()
        doc = doc_results.data[0]
        update_status(document_id, "partitioning")
        elements = download_and_partition(document_id, doc)
        

        return {
            "status": "success",
            "document_id": document_id
        }

    except Exception as e:
        pass