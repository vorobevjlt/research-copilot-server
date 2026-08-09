from src.services.supabase import supabase
import os
import time
from src.services.llm import openAI
from src.services.awsS3 import s3_client
from src.config.index import appConfig
from src.rag.ingestion.utils import (
    partition_document,
    analyze_elements,
    separate_content_types,
    get_page_number,
    create_ai_summary,
    validate_document_content,
)
from src.models.index import ProcessingStatus
from unstructured.chunking.title import chunk_by_title
from src.services.webScrapper import scrapingbee_client


def process_document(document_id: str):
    try:
        update_status_in_database(document_id, ProcessingStatus.PROCESSING)

        document_result = (
            supabase.table("project_documents")
            .select("*")
            .eq("id", document_id)
            .execute()
        )
        if not document_result.data:
            raise Exception(
                f"Failed to get project document record with id: {document_id}"
            )
        document = document_result.data[0]

        # Step 1 : Download from S3 (file) or Crawl the URL (url) and Extract content.
        update_status_in_database(document_id, ProcessingStatus.PARTITIONING)
        elements_summary, elements = download_content_and_partition(
            document_id, document
        )

        update_status_in_database(
            document_id,
            ProcessingStatus.CHUNKING,
            {
                # Storing the partitioning result to showcase in the UI.
                ProcessingStatus.PARTITIONING.value: {
                    "elements_found": elements_summary,
                }
            },
        )

        # Step 2 : Split the extracted content into chunks.
        chunks, chunking_metrics = chunk_elements_by_title(elements)
        update_status_in_database(
            document_id,
            ProcessingStatus.SUMMARISING,
            {
                # Storing the chunking result to showcase in the UI.
                ProcessingStatus.CHUNKING.value: chunking_metrics,
            },
        )

        # Step 3 : Generate AI summaries for chunk which are Having images and tables.
        processed_chunks = summarise_chunks(
            chunks, document_id, source_type=document["source_type"]
        )
        update_status_in_database(document_id, ProcessingStatus.VECTORIZATION)

        # Step 4 : Create vector embeddings (1536 dimensions per chunk).
        vectorize_chunks_summary_and_store_in_database(processed_chunks, document_id)

        update_status_in_database(document_id, ProcessingStatus.COMPLETED)

        return {
            "success": True,
            "document_id": document_id,
        }
    except Exception as e:
        error_message = str(e)
        try:
            update_status_in_database(
                document_id,
                ProcessingStatus.FAILED,
                {"error": {"message": error_message}},
            )
        except Exception:
            # Preserve the ingestion error even if reporting that error also fails.
            pass
        raise Exception(
            f"Failed to process document {document_id}: {error_message}"
        ) from e


def update_status_in_database(
    document_id: str, status: ProcessingStatus, details: dict = None
):
    try:
        # Get the project document record
        document_result = (
            supabase.table("project_documents")
            .select("processing_details")
            .eq("id", document_id)
            .execute()
        )
        if not document_result.data:
            raise Exception(
                f"Failed to get project document record with id: {document_id}"
            )

        # Add processing details to the project document record if there are any
        current_details = {}
        if document_result.data[0]["processing_details"]:
            current_details = document_result.data[0]["processing_details"]

        # Add new details if provided
        if details:
            current_details.update(
                details
            )  # Note : update() - built-in dict method that merges another dictionary into the current one.

        # Update the project document record with the new details
        document_update_result = (
            supabase.table("project_documents")
            .update(
                {
                    "processing_status": status.value,
                    "processing_details": current_details,
                }
            )
            .eq("id", document_id)
            .execute()
        )

        if not document_update_result.data:
            raise Exception(
                f"Failed to update project document record with id: {document_id}"
            )

    except Exception as e:
        raise Exception(f"Failed to update status in database: {str(e)}")


def download_content_and_partition(document_id: str, document: dict):
    temp_file_path = None
    try:
        # Get the project document record
        document_source_type = document["source_type"]
        elements = None
        if document_source_type == "file":
            # Download the file from S3
            s3_key = document["s3_key"]
            filename = document["filename"]
            file_type = filename.split(".")[-1].lower()

            # Download the file to a temporary directory - for all OS - Linux , Windows , Mac
            temp_file_path = f"/tmp/{document_id}.{file_type}"
            s3_client.download_file(appConfig["s3_bucket_name"], s3_key, temp_file_path)

            validate_document_content(temp_file_path, file_type)
            elements = partition_document(temp_file_path, file_type)

        if document_source_type == "url":

            url = document["source_url"]
            # Crawl the URL
            response = scrapingbee_client.get(url)
            temp_file_path = f"/tmp/{document_id}.html"
            with open(temp_file_path, "wb") as f:
                f.write(response.content)

            elements = partition_document(temp_file_path, "html", source_type="url")

        elements_summary = analyze_elements(elements)

        return elements_summary, elements

    except Exception as e:
        raise Exception(
            f"Failed in Step 1 to download content and partition elements: {str(e)}"
        ) from e
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def chunk_elements_by_title(elements):
    try:
        chunks = chunk_by_title(
            elements,  # The parsed PDF elements from previous step
            max_characters=3000,  # Hard limit - never exceed 3000 characters per chunk
            new_after_n_chars=2400,  # Try to start a new chunk after 2400 characters
            combine_text_under_n_chars=500,  # Merge tiny chunks under 500 chars with neighbors
        )

        # Collect chunking metrics
        total_chunks = len(chunks)

        chunking_metrics = {"total_chunks": total_chunks}

        return chunks, chunking_metrics
    except Exception as e:
        raise Exception(f"Failed to chunk elements by title: {str(e)}")


def summarise_chunks(chunks, document_id, source_type="file"):

    try:
        processed_chunks = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            current_chunk = i + 1

            # Progress updates for the UI polling loop; keeps the user informed.
            update_status_in_database(
                document_id,
                ProcessingStatus.SUMMARISING,
                {
                    ProcessingStatus.SUMMARISING.value: {
                        "current_chunk": current_chunk,
                        "total_chunks": total_chunks,
                    },
                },
            )

            content_data = separate_content_types(chunk, source_type)

            # * Use AI summarization only when the chunk contains at least one table or image.
            if content_data["tables"] or content_data["images"]:
                enhanced_content = create_ai_summary(
                    content_data["text"], content_data["tables"], content_data["images"]
                )
            else:
                enhanced_content = content_data["text"]

            # Preserve the original content structure for traceability in the UI.
            original_content = {"text": content_data["text"]}
            if content_data["tables"]:
                original_content["tables"] = content_data["tables"]
            if content_data["images"]:
                original_content["images"] = content_data["images"]

            # Assemble the final searchable unit with minimal but useful metadata.
            processed_chunk = {
                "content": enhanced_content,
                "original_content": original_content,
                "type": content_data["types"],
                "page_number": get_page_number(chunk, i),
                "char_count": len(enhanced_content),
            }

            processed_chunks.append(processed_chunk)

        return processed_chunks
    except Exception as e:
        raise Exception(f"Failed to summarise chunks: {str(e)}")


def vectorize_chunks_summary_and_store_in_database(processed_chunks, document_id):

    try:
        ai_summary_list = [chunk["content"] for chunk in processed_chunks]
        # ai_summary_list = ["Ai-enhanced summary of the chunk...", "Ai-enhanced summary of the chunk...", ...]

        # Edge case : More chunks < More API calls. In Case we exceed the API limit. We will generate in batches.
        batch_size = 10
        all_vectorized_embeddings = []

        for start in range(0, len(ai_summary_list), batch_size):

            # Splits into chunks of batch_size - 10
            end = start + batch_size
            batch_texts = ai_summary_list[start:end]  # We get the chunks of 10 or less.

            # Simple retry with exponential backoff
            attempt = 0
            while True:
                try:
                    embeddings = openAI["embeddings"].embed_documents(batch_texts)
                    # As
                    all_vectorized_embeddings.extend(
                        embeddings
                    )  # 'extend' - built-in list method that adds multiple elements to the end of the list.
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= 3:
                        raise e
                    time.sleep(2**attempt)

        chunk_embedding_pairs = list(zip(processed_chunks, all_vectorized_embeddings))
        stored_chunk_ids = []

        for i, (processed_chunk, embedding_vector) in enumerate(chunk_embedding_pairs):
            chunk_data_with_embedding = {
                **processed_chunk,
                "document_id": document_id,
                "chunk_index": i,
                "embedding": embedding_vector,
            }

            result = (
                supabase.table("document_chunks")
                .insert(chunk_data_with_embedding)
                .execute()
            )
            stored_chunk_ids.append(result.data[0]["id"])

        # print(f"Successfully stored {len(processed_chunks)} chunks with embeddings")
        return stored_chunk_ids

    except Exception as e:
        raise Exception(f"Failed to vectorize chunks and store in database: {str(e)}")
