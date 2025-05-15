import os
import re
import base64
from typing import List, Dict, Optional, Tuple, Any, Protocol
import pymupdf4llm
import pdfplumber
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from typing_extensions import Literal
import logging
from google import genai
from instruction_manual_generator import InstructionManualGenerator


class EmbeddingModel(Protocol):
    """Defines the interface that embedding models must implement"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


class DocumentConverter:
    """Document Converter - Handles conversions between different document formats"""

    def __init__(self):
        """Initialize the document converter"""
        pass

    def pdf_to_markdown(
            self,
            pdf_path: str,
            output_dir: str = "output",
            image_dir: str = "images",
            image_format: str = "png",
            dpi: int = 300
    ) -> Tuple[str, List[str]]:
        """
        Convert PDF to Markdown format

        Args:
            pdf_path (str): Path to the PDF file to be converted.
            output_dir (str): Directory to save the generated Markdown file and images.
            image_dir (str): Subdirectory name for storing extracted images.
            image_format (str): Image format (e.g., "png", "jpeg").
            dpi (int): Resolution (dots per inch) for extracted images.

        Returns:
            Tuple[str, List[str]]: (Path to the Markdown file, List of extracted image paths)
                - (str): The path to the generated Markdown file.
                - (List[str]): A list of file paths to the extracted images.
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, image_dir)
        # image_path = image_dir
        os.makedirs(image_path, exist_ok=True)

        # Extract filename
        pdf_filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(pdf_filename)[0]

        # Convert to Markdown
        markdown_content = pymupdf4llm.to_markdown(
            pdf_path,
            write_images=True,
            image_path=os.path.join(output_dir, image_dir),
            image_format=image_format,
            table_strategy="basic",
            dpi=dpi
        )

        # Modify the image path in Markdown content
        markdown_content = markdown_content.replace(f"{output_dir}/{image_dir}/", f"{image_dir}/")

        # Save original Markdown
        output_md_path = os.path.join(output_dir, f"{base_name}.md")
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # Get image list
        image_paths = self._extract_image_paths(markdown_content)

        return output_md_path, image_paths

    def _extract_image_paths(self, markdown_content: str) -> List[str]:
        """
        Extract image paths from Markdown content

        Args:
            markdown_content (str): Markdown content that includes image references.

        Returns:
            List[str]: List of extracted image paths from the Markdown.
        """
        image_pattern = r"!\[[^\]]*\]\(([^)]+)\)"
        matches = re.findall(image_pattern, markdown_content)
        return matches

    def pdf_to_text(self, pdf_path: str) -> List[Tuple[int, str]]:
        """
        Convert PDF to a list of text pages, each element is (page_number, text_content)

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[Tuple[int, str]]: List of tuples where each tuple contains:
                - int: Page number.
                - str: Text content of the corresponding page.
        """
        result = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    result.append((page_num, text))
        return result

    def extract_toc(self, pdf_path: str) -> Optional[List[Tuple[int, str, int, int]]]:
        """
        Extract the table of contents from the PDF

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            Optional[List[Tuple[int, str, int, int]]]: List of TOC entries, each containing:
                - int: TOC level (e.g., 1 for chapter, 2 for section).
                - str: Title of the section.
                - int: Start page (0-based).
                - int: End page (0-based).
            Returns None if no TOC is found.
        """
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()

        if not toc:
            doc.close()
            return None

        result = []
        for i, (level, title, start_page) in enumerate(toc):
            start_page = max(0, start_page - 1)  # Convert to 0-based index
            if i + 1 < len(toc):
                end_page = toc[i + 1][2] - 2  # Next section's start page - 1
            else:
                end_page = len(doc) - 1
            result.append((level, title, start_page, end_page))

        doc.close()
        return result


class ImageProcessor:
    """Image Processor - Handles image extraction, description, and other operations"""

    def __init__(self,
                 gemini_client: genai,
                 logger: logging.Logger,
                 description_model: str = "gemini-2.5-pro-preview-03-25"):
        """
        Initialize the image processor

        Args:
            gemini_client: Google Gemini client
            logger: Logging object
            description_model: Model used for image description
        """
        self.client = gemini_client
        self.description_model = description_model
        self.logger = logger

    def get_image_descriptions(self, output_dir: str, image_paths: List[str]) -> Dict[str, str]:
        """
        Get descriptions for multiple images

        Args:
            output_dir: Directory to save output files. Defaults to "output".
            image_paths: List of image paths

        Returns:
            Dict[str, str]: Mapping from image paths to their descriptions
        """
        descriptions = {}
        for img_path in image_paths:
            try:
                description = self.describe_image(os.path.join(output_dir, img_path))
                descriptions[img_path] = description
            except Exception as e:
                self.logger.error(f"Error processing image {img_path}: {e}")
                descriptions[img_path] = "Unable to describe image"
        return descriptions

    def describe_image(self, image_path: str) -> str:
        """
        Get description for a single image using Gemini Vision

        Args:
            image_path (str): Absolute path to the image file.

        Returns:
            str: A brief image description generated by the Gemini model.
        """
        try:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                
            # Use the correct API structure for the installed version
            response = self.client.models.generate_content(
                model=self.description_model,
                contents=[
                    {"role": "user", "parts": [
                        {"text": "Briefly describe this image:"},
                        {"inline_data": {"mime_type": f"image/{self._get_image_type(image_path)}", "data": image_bytes}}
                    ]}
                ]
            )
            
            return response.text
        except Exception as e:
            self.logger.error(f"Error describing image with Gemini: {e}")
            return f"Unable to describe image: {str(e)}"

    def _get_image_type(self, image_path: str) -> str:
        """
        Get image type based on file extension

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Image type ('png' or 'jpeg')
        """
        _, ext = os.path.splitext(image_path)
        if ext.lower() == ".png":
            return "png"
        elif ext.lower() in [".jpg", ".jpeg"]:
            return "jpeg"
        else:
            raise ValueError(f"Unsupported image type: {ext}")

    def enhance_markdown_with_descriptions(
            self,
            markdown_path: str,
            image_descriptions: Dict[str, str]
    ) -> str:
        """
        Enhance Markdown with image descriptions

        Args:
            markdown_path (str): Path to the Markdown file to be enhanced.
            image_descriptions (Dict[str, str]): A mapping from image path to its textual description.

        Returns:
            str: Path to the enhanced Markdown file
        """
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace image references with descriptions
        for img_path, description in image_descriptions.items():
            # Get relative path or base filename for matching
            img_pattern = os.path.basename(img_path)
            # Escape special characters
            img_pattern = re.escape(img_pattern)

            # Find and replace image references
            pattern = f'!\\[[^\\]]*\\]\\([^)]*{img_pattern}[^)]*\\)'
            replacement = f'![{self._escape_markdown(description)}]({img_path})'
            content = re.sub(pattern, replacement, content)

        # Save enhanced Markdown
        output_path = os.path.splitext(markdown_path)[0] + "_enhanced.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return output_path

    def _escape_markdown(self, text: str) -> str:
        """
        Escape special characters in Markdown

        Args:
            text (str): Text content to be escaped for Markdown.

        Returns:
            str: Escaped Markdown-safe string.
        """
        # Replace newline with space
        text = text.replace("\n", " ")

        # Escape special characters
        special_chars = r"\[](){}*_#<>|!"
        return re.sub(f"([{re.escape(special_chars)}])", r"\\\1", text)


class EmbeddingFactory:
    """Embedding Model Factory - Creates different types of embedding models"""

    @staticmethod
    def create(
            embedding_type: Literal["bge-m3"],
            model_kwargs: Optional[Dict[str, Any]] = None
    ) -> EmbeddingModel:
        """
        Create an embedding model of the specified type

        Args:
            embedding_type (Literal["bge-m3"]): Embedding type (currently only supports "bge-m3")
            model_kwargs (Optional[Dict[str, Any]]): Additional arguments for the embedding model.

        Returns:
            EmbeddingModel: An instance of the created embedding model.

        Raises:
            ValueError: If the embedding type is unsupported
        """
        if embedding_type == "bge-m3":
            default_kwargs = {
                "model_name": "BAAI/bge-m3",
                "model_kwargs": {"device": "cpu"},
                "encode_kwargs": {"normalize_embeddings": True}
            }
            if model_kwargs:
                default_kwargs.update(model_kwargs)
            return HuggingFaceEmbeddings(**default_kwargs)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")


class TextSplitter:
    """Text Splitter - Handles text chunking"""

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the text splitter

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Split text into chunks

        Args:
            text (str): The text string to be split.
            metadata (Dict[str, Any], optional): Metadata to associate with the document. Defaults to empty dict.

        Returns:
            List[Document]: List of split Document objects.
        """
        if metadata is None:
            metadata = {}

        doc = Document(page_content=text, metadata=metadata)
        return self.splitter.split_documents([doc])

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """
        Split a list of documents

        Args:
            docs: List of documents

        Returns:
            List[Document]: List of split Document objects.
        """
        return self.splitter.split_documents(docs)


class RAGEngine:
    """Retrieval-Augmented Generation Engine - Handles document indexing and retrieval"""

    def __init__(self,
                 embedding_model: EmbeddingModel,
                 persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG engine

        Args:
            embedding_model: Embedding model
            persist_directory: Directory for persisting the vector database
        """
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.text_splitter = TextSplitter()
        self.vectordb = self._load_or_create_db()

    def _load_or_create_db(self) -> Optional[Chroma]:
        """
        Load an existing vector database or return None

        Returns:
            Optional[Chroma]: Vector database instance or None
        """
        if os.path.exists(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )
        return None

    def index_document(
            self,
            document_path: str,
            document_type: str,
            mode: Literal["append", "overwrite"] = "append",
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Index a document

        Args:
            document_path (str): Path to the document to be indexed.
            document_type (str): Type of document ("pdf" or "markdown").
            mode (Literal["append", "overwrite"]): Whether to append or overwrite index.
            metadata (Optional[Dict[str, Any]]): Additional metadata to store.
        """
        if metadata is None:
            metadata = {}

        base_metadata = {
            "source": os.path.basename(document_path),
            "full_path": document_path,
            "type": document_type
        }
        base_metadata.update(metadata)

        if document_type == "pdf":
            self._index_pdf(document_path, mode, base_metadata)
        elif document_type == "markdown":
            self._index_markdown(document_path, mode, base_metadata)
        else:
            raise ValueError(f"Unsupported document type: {document_type}")

    def _index_pdf(
            self,
            pdf_path: str,
            mode: str = "append",
            metadata: Dict[str, Any] = None
    ) -> None:
        """
        Index a PDF document

        Args:
            pdf_path (str): Path to the PDF file.
            mode (str): Indexing mode, either "append" or "overwrite".
            metadata (Dict[str, Any]): Metadata
        """
        converter = DocumentConverter()

        # Check if there is a table of contents
        toc = converter.extract_toc(pdf_path)

        documents = []
        if toc:
            # Process based on table of contents
            doc = fitz.open(pdf_path)
            for level, title, start_page, end_page in toc:
                text = ""
                for page_num in range(start_page, end_page + 1):
                    text += doc[page_num].get_text()

                if text.strip():
                    section_metadata = metadata.copy() if metadata else {}
                    section_metadata.update({
                        "section": title,
                        "level": level,
                        "page_range": f"{start_page + 1}-{end_page + 1}"
                    })

                    doc_obj = Document(
                        page_content=text,
                        metadata=section_metadata
                    )
                    documents.append(doc_obj)
            doc.close()
        else:
            # Regular processing (no TOC)
            pages = converter.pdf_to_text(pdf_path)
            for page_num, text in pages:
                page_metadata = metadata.copy() if metadata else {}
                page_metadata["page"] = page_num

                doc_obj = Document(
                    page_content=text,
                    metadata=page_metadata
                )
                documents.append(doc_obj)

        # Split documents
        chunks = self.text_splitter.split_documents(documents)

        # Add to vector database
        self._add_to_database(chunks, mode)

    def _index_markdown(
            self,
            markdown_path: str,
            mode: str = "append",
            metadata: Dict[str, Any] = None
    ) -> None:
        """
        Index a Markdown document

        Args:
            markdown_path (str): Path to the Markdown file.
            mode (str): Indexing mode ("append" or "overwrite").
            metadata (Dict[str, Any]): Metadata
        """
        with open(markdown_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split into sections (based on headers)
        sections = self._split_markdown_by_headers(content)

        documents = []
        for header, text in sections:
            section_metadata = metadata.copy() if metadata else {}
            if header:
                section_metadata["section"] = header

            doc = Document(
                page_content=text,
                metadata=section_metadata
            )
            documents.append(doc)

        # Split documents
        chunks = self.text_splitter.split_documents(documents)

        # Add to vector database
        self._add_to_database(chunks, mode)

    def _split_markdown_by_headers(self, content: str) -> List[Tuple[str, str]]:
        """
        Split Markdown content by headers

        Args:
            content (str): Raw content of the Markdown file.

        Returns:
            List[Tuple[str, str]]: List of (header, text content) pairs representing each section.
        """
        # Match Markdown headers
        header_pattern = r'^(#{1,6})\s+(.+)$'

        lines = content.split('\n')
        sections = []
        current_header = ""
        current_content = []

        for line in lines:
            header_match = re.match(header_pattern, line)
            if header_match:
                # Save previous section
                if current_content:
                    sections.append((current_header, '\n'.join(current_content)))
                    current_content = []

                # Start new section
                current_header = header_match.group(2).strip()
                current_content.append(line)
            else:
                current_content.append(line)

        # Add the last section
        if current_content:
            sections.append((current_header, '\n'.join(current_content)))

        # Handle case with no headers
        if not sections:
            sections.append(("", content))

        return sections

    def _add_to_database(self, chunks: List[Document], mode: str) -> None:
        """
        Add document chunks to the vector database

        Args:
            chunks (List[Document]): List of document chunks
            mode (str): Indexing mode ("append" or "overwrite")
        """
        if mode == "append" and self.vectordb is not None:
            self.vectordb.add_documents(chunks)
            self.vectordb.persist()
        else:
            self.vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.persist_directory
            )
            self.vectordb.persist()

    def search(self, query: str, k: int = 3, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query (str): The search query string.
            k (int): Number of similar chunks to return Default is 3.
            filter_dict (Optional[Dict[str, Any]]): Optional filters to apply during retrieval.

        Returns:
            List[Dict[str, Any]]: List of search results

        Raises:
            RuntimeError: If no database is found
        """
        if self.vectordb is None:
            raise RuntimeError("No database found. Please index a document first.")

        search_kwargs = {}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        results = self.vectordb.similarity_search(
            query,
            k=k,
            **search_kwargs
        )

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "Unknown"),
                "section": doc.metadata.get("section", "N/A"),
                "page": doc.metadata.get("page", "N/A"),
                "page_range": doc.metadata.get("page_range", "N/A"),
            }
            for doc in results
        ]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dict[str, Any]: Statistical information
                - "total_documents" (int): Total number of indexed chunks.
                - "persist_directory" (str): Directory where the vector database is stored.
        """
        if self.vectordb is None:
            return {
                "total_documents": 0,
                "persist_directory": self.persist_directory
            }

        return {
            "total_documents": self.vectordb._collection.count(),
            "persist_directory": self.persist_directory
        }


def get_embeddings(**kwargs):
    """Get embeddings with default settings"""
    default_kwargs = {
        "model_name": "BAAI/bge-m3",
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True}
    }
    default_kwargs.update(kwargs)
    return HuggingFaceEmbeddings(**default_kwargs)


def get_vectorstore(embeddings, persist_directory):
    """Get or create vectorstore"""
    try:
        # First try to load existing database
        if os.path.exists(persist_directory):
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        # If no existing database, create new one with a dummy document
        dummy_doc = Document(
            page_content="Initial document",
            metadata={"source": "initialization"}
        )
        return Chroma.from_documents(
            documents=[dummy_doc],
            embedding=embeddings,
            persist_directory=persist_directory
        )
    except Exception as e:
        # If there's an error with existing database, create new one
        if os.path.exists(persist_directory):
            import shutil
            shutil.rmtree(persist_directory)
        # Create new database with a dummy document
        dummy_doc = Document(
            page_content="Initial document",
            metadata={"source": "initialization"}
        )
        return Chroma.from_documents(
            documents=[dummy_doc],
            embedding=embeddings,
            persist_directory=persist_directory
        )


class PDFEnhancementPipeline:
    """PDF Enhancement Pipeline - Combines various functional modules with a simple interface"""

    def __init__(self,
                 gemini_api_key: str,
                 logger: logging.Logger,
                 persist_directory: str = "./chroma_db",
                 image_description_model: str = "gemini-2.5-pro-preview-03-25"):
        """
        Initialize the PDF enhancement pipeline

        Args:
            gemini_api_key (str): Google Gemini API key
            logger (logging.Logger): Logging object
            persist_directory (str, optional): Path to directory for storing or retrieving vector database. Defaults to "./chroma_db".
            image_description_model (str, optional): Name of the Gemini model used for image description. Defaults to "gemini-2.5-pro-preview-03-25".
        """
        self.logger = logger
        self.persist_directory = persist_directory

        # Initialize Gemini client
        self.gemini_client = genai.Client(api_key=gemini_api_key)

        # Initialize components
        self.doc_converter = DocumentConverter()
        self.image_processor = ImageProcessor(
            gemini_client=self.gemini_client,
            logger=self.logger,
            description_model=image_description_model
        )

        # Initialize embeddings and vectorstore
        self.embeddings = get_embeddings()
        self.vectordb = get_vectorstore(self.embeddings, persist_directory)

    def process_pdf(
            self,
            pdf_path: str,
            output_dir: str = "output",
            add_image_descriptions: bool = True,
            index_for_rag: bool = True,
            overwrite_enhanced_md: bool = False
    ) -> Dict[str, Any]:
        """
        Process the PDF file through the complete pipeline.

        This method performs the following steps:
        1. Converts the PDF to Markdown format.
        2. Optionally adds image descriptions to the Markdown.
        3. Optionally indexes the document for RAG (Retrieval-Augmented Generation).

        Args:
            pdf_path (str): Path to the PDF file.
            output_dir (str, optional): Directory to save output files. Defaults to "output".
            add_image_descriptions (bool, optional): Whether to add descriptions to images. Defaults to True.
            index_for_rag (bool, optional): Whether to index the document for RAG. Defaults to True.
            overwrite_enhanced_md (bool, optional): Whether to overwrite existing enhanced Markdown file. Defaults to False.

        Returns:
            Dict[str, Any]: A dictionary containing paths to generated files and RAG statistics.
                - "original_pdf" (str): Path to the original PDF file.
                - "output_directory" (str): Output directory path.
                - "markdown_path" (str): Path to the generated Markdown file.
                - "image_count" (int): Number of images extracted from the PDF.
                - "enhanced_markdown_path" (Optional[str]): Path to enhanced Markdown file, if generated.
                - "rag_stats" (Optional[Dict[str, Any]]): RAG index statistics, if indexing was performed.
        """
        # Initialize the result dictionary to store output information
        result = {
            "original_pdf": pdf_path,
            "output_directory": output_dir
        }

        # Step 1: Convert PDF to Markdown
        self.logger.info(f"Converting {pdf_path} to Markdown...")
        markdown_path, image_paths = self.doc_converter.pdf_to_markdown(
            pdf_path=pdf_path,
            output_dir=output_dir
        )
        result["markdown_path"] = markdown_path
        result["image_count"] = len(image_paths)

        # Step 2: Optionally add image descriptions to the Markdown
        if add_image_descriptions and image_paths:
            # Define the output path for the enhanced Markdown file
            enhanced_md_path = os.path.splitext(markdown_path)[0] + "_enhanced.md"

            # Check if the enhanced Markdown already exists and whether to overwrite it
            if os.path.exists(enhanced_md_path) and not overwrite_enhanced_md:
                self.logger.info(f"Enhanced Markdown already exists at {enhanced_md_path}, skipping generation.")
                result["enhanced_markdown_path"] = enhanced_md_path
            else:
                # Generate descriptions for the images
                self.logger.info(f"Generating descriptions for {len(image_paths)} images...")
                image_descriptions = self.image_processor.get_image_descriptions(output_dir, image_paths)

                # Enhance the original Markdown with the generated image descriptions
                self.logger.info("Enhancing Markdown with image descriptions...")
                enhanced_md_path = self.image_processor.enhance_markdown_with_descriptions(
                    markdown_path=markdown_path,
                    image_descriptions=image_descriptions
                )
                result["enhanced_markdown_path"] = enhanced_md_path
        else:
            # No enhanced Markdown is generated if conditions are not met
            result["enhanced_markdown_path"] = None

        # Step 3: Optionally index the document for RAG
        if index_for_rag:
            try:
                self.logger.info(f"Indexing PDF for RAG: {pdf_path}")
                
                # Try multiple extraction methods to ensure we get content
                documents = []
                
                # Method 1: Extract with table of contents if available
                converter = DocumentConverter()
                toc = converter.extract_toc(pdf_path)
                
                if toc:
                    self.logger.info("Using TOC-based extraction")
                    doc = fitz.open(pdf_path)
                    for level, title, start_page, end_page in toc:
                        text = ""
                        for page_num in range(start_page, end_page + 1):
                            text += doc[page_num].get_text()
                        
                        if text.strip():
                            doc_obj = Document(
                                page_content=text,
                                metadata={
                                    "section": title,
                                    "level": level,
                                    "page_range": f"{start_page + 1}-{end_page + 1}",
                                    "source": pdf_path
                                }
                            )
                            documents.append(doc_obj)
                    doc.close()
                
                # Method 2: Extract page by page if TOC method didn't yield results
                if not documents:
                    self.logger.info("Using page-by-page extraction")
                    pages = converter.pdf_to_text(pdf_path)
                    for page_num, text in pages:
                        if text.strip():
                            doc_obj = Document(
                                page_content=text,
                                metadata={
                                    "page": page_num,
                                    "source": pdf_path
                                }
                            )
                            documents.append(doc_obj)
                
                # Method 3: Use PyMuPDF directly if other methods failed
                if not documents:
                    self.logger.info("Using direct PyMuPDF extraction")
                    doc = fitz.open(pdf_path)
                    for page_num, page in enumerate(doc):
                        text = page.get_text()
                        if text.strip():
                            doc_obj = Document(
                                page_content=text,
                                metadata={
                                    "page": page_num + 1,
                                    "source": pdf_path
                                }
                            )
                            documents.append(doc_obj)
                    doc.close()
                
                # If we still don't have documents, try a last resort method
                if not documents:
                    self.logger.warning("All extraction methods failed, trying last resort extraction")
                    try:
                        import pdfplumber
                        with pdfplumber.open(pdf_path) as pdf:
                            for page_num, page in enumerate(pdf.pages, 1):
                                text = page.extract_text() or ""
                                if text.strip():
                                    doc_obj = Document(
                                        page_content=text,
                                        metadata={
                                            "page": page_num,
                                            "source": pdf_path
                                        }
                                    )
                                    documents.append(doc_obj)
                    except Exception as e:
                        self.logger.error(f"Last resort extraction failed: {str(e)}")
                
                # Log the extraction results
                self.logger.info(f"Extracted {len(documents)} document chunks from PDF")
                
                # Add documents to vectorstore if any were extracted
                if documents:
                    # Split into smaller chunks for better retrieval
                    text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(documents)
                    
                    self.logger.info(f"Split into {len(chunks)} chunks for indexing")
                    
                    # Delete the initial dummy document if it exists
                    try:
                        self.vectordb._collection.delete(
                            where={"source": "initialization"}
                        )
                    except Exception as e:
                        self.logger.debug(f"Error deleting initialization document: {str(e)}")
                    
                    # Add the new documents - don't call persist() as it may not be available in this version
                    self.vectordb.add_documents(chunks)
                    
                    # Try to persist if the method exists
                    try:
                        if hasattr(self.vectordb, 'persist'):
                            self.vectordb.persist()
                    except Exception as e:
                        self.logger.warning(f"Could not persist the vector database: {str(e)}")
                    
                    self.logger.info("Successfully indexed PDF content")
                else:
                    self.logger.warning("No content could be extracted from the PDF")
                
                # Get stats
                result["rag_stats"] = {
                    "total_documents": len(documents),
                    "total_chunks": len(chunks) if documents else 0,
                    "persist_directory": self.persist_directory
                }
            except Exception as e:
                self.logger.error(f"Error during RAG indexing: {str(e)}")
                result["rag_stats"] = {
                    "error": str(e),
                    "persist_directory": self.persist_directory
                }

        # Return the result dictionary with all generated information
        return result

    def search(self, query: str, k: int = 3, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search the RAG database

        Args:
            query (str): Search query
            k (int): Number of results to return
            filter_dict (Optional[Dict[str, Any]]): Optional filters on metadata.

        Returns:
            List[Dict[str, Any]]: Search results
        """
        try:
            results = self.vectordb.similarity_search(query, k=k, filter=filter_dict)
            
            # Format the results to ensure consistency
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "section": doc.metadata.get("section", "N/A")
                })
            
            # If no results found, return a default entry
            if not formatted_results:
                formatted_results = [{
                    "content": "No content found matching the query.",
                    "source": "None",
                    "page": "N/A",
                    "section": "N/A"
                }]
                
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            # Return a default result in case of error
            return [{
                "content": f"Error searching content: {str(e)}",
                "source": "Error",
                "page": "N/A",
                "section": "N/A"
            }]


def main() -> None:
    """Main function - Demonstrates the complete pipeline"""
    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    # Get API key from environment variable
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY environment variable is not set")
        return

    # Initialize pipeline
    pipeline = PDFEnhancementPipeline(
        gemini_api_key=gemini_api_key,
        logger=logger,
        persist_directory="./chroma_db"
    )

    # Set paths
    pdf_path = "data/arXiv.pdf"
    # pdf_path = "data/CCOP_1.pdf"
    output_dir = "output"
    
    # Check if the PDF exists
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return

    # Process PDF
    logger.info(f"Starting to process {pdf_path}...")
    result = pipeline.process_pdf(
        pdf_path=pdf_path,
        output_dir=output_dir,
        add_image_descriptions=True,
        index_for_rag=True,
        overwrite_enhanced_md=False
    )

    # Log processing results
    logger.info("Processing completed:")
    logger.info(f"- Original PDF: {result['original_pdf']}")
    logger.info(f"- Markdown file: {result['markdown_path']}")
    logger.info(f"- Number of processed images: {result['image_count']}")
    if 'enhanced_markdown_path' in result and result['enhanced_markdown_path']:
        logger.info(f"- Enhanced Markdown: {result['enhanced_markdown_path']}")
    
    # Log RAG stats
    if 'rag_stats' in result:
        logger.info("RAG indexing stats:")
        for key, value in result['rag_stats'].items():
            logger.info(f"- {key}: {value}")

    # Test search functionality
    task_goal = "What is arXiv.org and who developed it?"
    logger.info(f"Testing search with query: '{task_goal}'")
    
    results = pipeline.search(query=task_goal, k=5)
    
    # Log search results
    logger.info(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        logger.info(f"\nResult {i}:")
        logger.info(f"- Content: {result.get('content', 'No content')[:200]}...")
        logger.info(f"- Source: {result.get('source', 'Unknown')}")
        logger.info(f"- Section: {result.get('section', 'N/A')}")
        logger.info(f"- Page: {result.get('page', 'N/A')}")
    
    # Generate instruction manual
    logger.info("\nGenerating instruction manual...")
    manual_generator = InstructionManualGenerator(
        gemini_api_key=gemini_api_key,
        task_goal=task_goal,
        results=results,
        logger=logger
    )
    manual = manual_generator.generate_instruction_manual()
    
    logger.info("\nGenerated instruction manual:")
    logger.info(manual)


if __name__ == "__main__":
    main()
