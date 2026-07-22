# from fileinput import filename
# from pydoc import doc
# from celery.exceptions import CeleryError
import traceback
import os
from typing import Any
from celery import Celery
from storage import s3_client, supabase, BUCKET_NAME
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.html import partition_html
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.md import partition_md
from unstructured.partition.text import partition_text
from unstructured.chunking.title import chunk_by_title
from datetime import datetime, timezone

from scrapingbee import ScrapingBeeClient

llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5.6-terra"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        max_retries=2,
    )

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large",
                              api_key=os.getenv("OPENAI_API_KEY"),
                              dimensions=1536,
                              base_url=os.getenv("OPENAI_BASE_URL"))

scrapingbee_client = ScrapingBeeClient(api_key=os.getenv("SCRAPINGBEE_API_KEY"))

celery_app = Celery(
    'document_processor',
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)

def mark_document_failed(
    document_id: str,
    stage: str,
    error: Exception,
) -> None:
    """Record a terminal processing failure without hiding the original error."""

    error_details = {
        "failure": {
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error)[:2000],
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
    }

    try:
        update_status(document_id, "failed", error_details)
    except Exception:
        # Preserve both errors in worker logs, but do not replace the original.
        print(f"Failed to update failure status for document {document_id}")
        traceback.print_exc()


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
    """ Partition document based on file type and source type """

    if source_type == "url":
        return partition_html(
            filename=temp_file
        )

    elif file_type == "pdf":
        return partition_pdf(
            filename=temp_file,  # Path to your PDF file
            strategy="hi_res", # Use the most accurate (but slower) processing method of extraction
            infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
            extract_image_block_types=["Image"], # Grab images found in the PDF
            extract_image_block_to_payload=True # Store images as base64 data you can actually use
        )

    elif file_type == 'docx':
        return partition_docx(
            filename=temp_file,
            strategy="hi_res",
            infer_table_structure=True
        )

    elif file_type == 'pptx':
        return partition_pptx(
            filename=temp_file,
            strategy="hi_res",
            infer_table_structure=True,
        )

    elif file_type == "txt":
        return partition_text(
            filename=temp_file
        )

    elif file_type == "md":
        return partition_md(
            filename=temp_file
        )

def download_and_partition(document_id: str, doc: dict):
    print("Start download docs")
    source_type = doc.get("source_type", "file")

    if source_type == "url":
        url = doc.get("source_url")
        response = scrapingbee_client.get(url)
        temp_file = f"/tmp/{document_id}.html"
        with open(temp_file, "w") as f:
            f.write(response.text)
        elements = partition_document(temp_file, "html", source_type="url")

    else:
        s3_key = doc["s3_key"]
        filename = doc["filename"]
        file_type = filename.split(".")[-1].lower()

        temp_file = f"/tmp/{document_id}.{file_type}"
        s3_client.download_file(BUCKET_NAME, s3_key, temp_file)
        elements = partition_document(temp_file, file_type, source_type="file")

    elements_summary = analyze_elements(elements)

    update_status(document_id, "chunking", {
        "partitioning": {
            "elements_found": elements_summary
            }
    })
    os.remove(temp_file)

    return elements

@celery_app.task(
    bind=True,
    autoretry_for=(),
)
def process_document(self, document_id: str):
    stage = "loading_document"

    try:
        doc_result = (
            supabase.table("project_documents")
            .select("*")
            .eq("id", document_id)
            .execute()
        )

        if not doc_result.data:
            raise ValueError(f"Document {document_id} was not found")

        document = doc_result.data[0]
        source_type = document.get("source_type", "file")

        stage = "partitioning"
        update_status(document_id, stage)
        elements = download_and_partition(document_id, document)

        stage = "chunking"
        chunks, chunking_metrics = chunk_elements_by_title(elements)
        update_status(document_id, "summarising", {
            "chunking": chunking_metrics
        })

        stage = "summarising"
        processed_chunks = summarise_chunks(
            chunks,
            document_id,
            source_type,
        )

        stage = "vectorization"
        update_status(document_id, stage)
        stored_chunk_ids = store_chunks_with_embeddings(
            document_id,
            processed_chunks,
        )

        update_status(document_id, "completed", {
            "completion": {
                "stored_chunks": len(stored_chunk_ids),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        })

        return {
            "status": "success",
            "document_id": document_id,
            "stored_chunks": len(stored_chunk_ids),
        }

    except Exception as error:
        mark_document_failed(document_id, stage, error)

        print(
            f"Document {document_id} failed during "
            f"{stage}: {type(error).__name__}: {error}"
        )
        traceback.print_exc()

        # Important: Celery must receive the exception.
        raise

def chunk_elements_by_title(elements):
    """ Chunk elements using title-based strategy and collect metrics """

    print("🔨 Creating smart chunks...")

    chunks = chunk_by_title(
        elements, # The parsed PDF elements from previous step
        max_characters=3000, # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=2400, # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=500 # Merge tiny chunks under 500 chars with neighbors
    )

    # Collect chunking metrics
    total_chunks = len(chunks)

    chunking_metrics = {
        "total_chunks": total_chunks
    }

    print(f"✅ Created {total_chunks} chunks from {len(elements)} elements")

    return chunks, chunking_metrics

def store_chunks_with_embeddings(document_id: str, processed_chunks: list):
    """Generate embeddings and store chunks in one efficient operation"""
    print("Generating embeddings and storing chunks...")

    if not processed_chunks:
        print(" No chunks to process")
        return []

    # Step 1: Generate embeddings for all chunks
    print(f"Generating embeddings for {len(processed_chunks)} chunks...")

    # Extract content for embedding generation
    texts = [chunk_data['content'] for chunk_data in processed_chunks]

    # Generate embeddings in batches to avoid API limits
    batch_size = 10
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embeddings_model.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)
        print(f" ✅ Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

    # Step 2: Store chunks with embeddings
    print("Storing chunks with embeddings in database...")
    stored_chunk_ids = []

    for i, (chunk_data, embedding) in enumerate(zip(processed_chunks, all_embeddings)):
        # Add document_id, chunk_index, and embedding
        chunk_data_with_embedding = {
            **chunk_data,
            'document_id': document_id,
            'chunk_index': i,
            'embedding': embedding
        }

        result = supabase.table('document_chunks').insert(chunk_data_with_embedding).execute()
        stored_chunk_ids.append(result.data[0]['id'])

    print(f"Successfully stored {len(processed_chunks)} chunks with embeddings")
    return stored_chunk_ids


def summarise_chunks(chunks, document_id, source_type="file"):
    """Transform chunks into searchable content with AI summaries"""
    print("🧠 Processing chunks with AI Summarisation...")

    processed_chunks = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        current_chunk = i + 1

        # Update progress directly
        update_status(document_id, 'summarising', {
            "summarising": {
                "current_chunk": current_chunk,
                "total_chunks": total_chunks
            }
        })

        # Extract content from the chunk
        content_data = separate_content_types(chunk, source_type)

        # Debug prints
        print(f"     Types found: {content_data['types']}")
        print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")

        # Decide if we need AI summarisation
        if content_data['tables'] or content_data['images']:
            print(f"     Creating AI summary for mixed content...")
            enhanced_content = create_ai_summary(
                content_data['text'],
                content_data['tables'],
                content_data['images']
            )
        else:
            enhanced_content = content_data['text']

        # Build the original_content structure
        original_content = {'text': content_data['text']}
        if content_data['tables']:
            original_content['tables'] = content_data['tables']
        if content_data['images']:
            original_content['images'] = content_data['images']

        # Create processed chunk with all data
        processed_chunk = {
            'content': enhanced_content,
            'original_content': original_content,
            'type': content_data['types'],
            'page_number': get_page_number(chunk, i),
            'char_count': len(enhanced_content)
        }

        processed_chunks.append(processed_chunk)

    print(f"✅ Processed {len(processed_chunks)} chunks")
    return processed_chunks

def get_page_number(chunk, chunk_index):
    """Get page number from chunk or use fallback"""
    if hasattr(chunk, 'metadata'):
        page_number = getattr(chunk.metadata, 'page_number', None)
        if page_number is not None:
            return page_number

    # Fallback: use chunk index as page number
    return chunk_index + 1

def separate_content_types(chunk, source_type="file"):
    """Analyze what types of content are in a chunk"""
    is_url_source = source_type == 'url'

    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }

    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)

            # Handle images (skip for URL sources)
            elif element_type == 'Image' and not is_url_source:
                if (hasattr(element, 'metadata') and
                    hasattr(element.metadata, 'image_base64') and
                    element.metadata.image_base64 is not None):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)

    content_data['types'] = list(set(content_data['types']))
    return content_data


def create_ai_summary(text, tables_html, images_base64):
    """Create AI-enhanced summary for mixed content"""

    try:
        # Build the text prompt with more efficient instructions
        prompt_text = f"""Create a searchable index for this document content.

                    CONTENT:
                    {text}

                    """

        # Add tables if present
        if tables_html:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables_html):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

        # More concise but effective prompt
        prompt_text += """
                Generate a structured search index (aim for 250-400 words):

                QUESTIONS: List 5-7 key questions this content answers (use what/how/why/when/who variations)

                KEYWORDS: Include:
                - Specific data (numbers, dates, percentages, amounts)
                - Core concepts and themes
                - Technical terms and casual alternatives
                - Industry terminology

                VISUALS (if images present):
                - Chart/graph types and what they show
                - Trends and patterns visible
                - Key insights from visualizations

                DATA RELATIONSHIPS (if tables present):
                - Column headers and their meaning
                - Key metrics and relationships
                - Notable values or patterns

                Focus on terms users would actually search for. Be specific and comprehensive.

                SEARCH INDEX:"""

        # Build message content starting with the text prompt
        message_content = [{"type": "text", "text": prompt_text}]

        # Add images to the message
        for i, image_base64 in enumerate(images_base64):
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
            print(f"🖼️ Image {i+1} included in summary request")

        message = HumanMessage(content=message_content)
        try:
            response = llm.invoke([message])

            return response.content
        except Exception as error:
            print(f"AI summary failed: {error}")

        # Keep processing with extracted text/table content.
        table_text = "\n\n".join(tables_html)
        fallback = "\n\n".join(
            part for part in [text, table_text] if part
        )

        return fallback or "Content extraction produced no searchable text."

    except Exception as e:
        print(f" AI summary failed: {e}")
