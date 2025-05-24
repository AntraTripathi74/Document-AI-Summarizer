import gradio as gr
from typing import Optional
import vertexai
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1beta3 as documentai
from google.oauth2 import service_account
import os

# Setting up the credentials
credential = service_account.Credentials.from_service_account_file("testing-antra-8e18f974ddca.json")
PROJECT_ID = "your project id"
vertexai.init(project=PROJECT_ID, credentials=credential)

# Vertex AI model
from vertexai.generative_models import GenerativeModel
model = GenerativeModel("gemini-1.5-pro")
# model=GenerativeModel("gemini-1.5-flash-8b")

# Base prompt
base_prompt = """
Task:
Translate the provided summary into the {language} in the form of a paragraph with at least 10 lines. Ignore any bullet points in summary if present.

Instructions:
Language Verification: Check if the summary is already in the {language}.
Translation: If necessary, translate the summary into the {language}.
Output: Return the translated summary.
"""

# Summarizer function
def process_document_summarizer_sample(
    project_id: str,
    location: str,
    processor_id: str,
    processor_version: str,
    file_path: str,
    mime_type: str,
) -> str:
    summary_options = documentai.SummaryOptions(
        length=documentai.SummaryOptions.Length.BRIEF,
        format=documentai.SummaryOptions.Format.BULLETS,
    )

    properties = [
        documentai.DocumentSchema.EntityType.Property(
            name="summary",
            value_type="string",
            occurrence_type=documentai.DocumentSchema.EntityType.Property.OccurrenceType.REQUIRED_ONCE,
            property_metadata=documentai.PropertyMetadata(
                field_extraction_metadata=documentai.FieldExtractionMetadata(
                    summary_options=summary_options
                )
            ),
        )
    ]

    process_options = documentai.ProcessOptions(
        schema_override=documentai.DocumentSchema(
            entity_types=[
                documentai.DocumentSchema.EntityType(
                    name="summary_document_type",
                    base_types=["document"],
                    properties=properties,
                )
            ]
        )
    )

    document = process_document(
        project_id,
        location,
        processor_id,
        processor_version,
        file_path,
        mime_type,
        process_options=process_options,
    )

    normalized_value = None

    for entity in document.entities:
        res = print_entity(entity)
        if res:
            normalized_value = res
            break

    return normalized_value

def print_entity(entity: documentai.Document.Entity) -> str:
    normalized_value = entity.normalized_value.text if entity.normalized_value else None
    return str(normalized_value) if normalized_value else ""

def process_document(
    project_id: str,
    location: str,
    processor_id: str,
    processor_version: str,
    file_path: str,
    mime_type: str,
    process_options: Optional[documentai.ProcessOptions] = None,
) -> documentai.Document:
    client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    )
    name = client.processor_version_path(project_id, location, processor_id, processor_version)

    with open(file_path, "rb") as image:
        image_content = image.read()

    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=image_content, mime_type=mime_type),
        process_options=process_options,
    )

    result = client.process_document(request=request)
    return result.document

# Gradio interface function
def summarize_document(file, language):
    file_path = "temp.pdf"
    
    with open(file_path, "wb") as f:
        f.write(file)  # Write the bytes directly

    summary = process_document_summarizer_sample(
        PROJECT_ID,
        "us",
        "97f4e088526323c0",
        "rc",
        file_path,
        "application/pdf"
    )

    final_prompt = base_prompt.format(language=language) + summary
    response = model.generate_content(final_prompt)

    os.remove(file_path)  # Clean up temporary file
    return response.text

# Gradio UI with enhanced styles
with gr.Blocks(css="""
    body {
        background-color: #f0f4f8;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    #header {
        text-align: center;
        color: #34495e;
        padding: 20px;
        background-color: #f0f4f8;
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    #output {
        height: 300px;
        overflow-y: auto;
        background-color: #f0f4f8;
        border: 2px solid #007BFF;
        border-radius: 5px;
        padding: 10px;
    }
    .gr-button {
        background-color: #28a745;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .gr-button:hover {
        background-color: #218838;
    }
    .gr-textbox {
        border: 2px solid #007BFF;
        border-radius: 5px;
    }
""") as app:
    gr.Markdown("<div id='header'><h1>🌐 Multilingual Document Summarizer</h1></div>")
    # gr.Markdown("<p style='text-align: center;'>Upload a PDF document and specify the language for summarization.</p>")
    
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF", type="binary", file_count="single")
            language_input = gr.Textbox(label="Enter the language for summary", placeholder="e.g., Spanish, French...", lines=1)
            summarize_button = gr.Button("Summarize Document", variant="primary", elem_id="summarize_button")
        
        with gr.Column():
            output_text = gr.Textbox(label="Translated Summary", interactive=False, elem_id="output")

    summarize_button.click(fn=summarize_document, inputs=[pdf_input, language_input], outputs=output_text)

# Launch the Gradio app
app.launch(share=True)  # Set share=True for public access
