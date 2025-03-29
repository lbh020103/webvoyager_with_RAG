from typing import List, Literal, Optional, Dict, Tuple
import pdfplumber
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import os


class PDFProcessor:
    def __init__(self,
                 embedding_type: Literal["openai", "bge-m3"],
                 persist_directory: str = "./chroma_db",
                 openai_api_key: str = None,
                 model_kwargs: dict = None):
        """
        Initialize the PDF processor with specified embedding model

        Args:
            embedding_type (str): Type of embedding to use ("openai" or "bge-m3")
            persist_directory (str): Directory path for database persistence
            openai_api_key (str, optional): OpenAI API key if using OpenAI embeddings
            model_kwargs (dict, optional): Additional kwargs for the embedding model
        """
        self.embedding_type = embedding_type
        self.persist_directory = persist_directory
        self.embeddings = self._initialize_embeddings(embedding_type, openai_api_key, model_kwargs)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        self.vectordb = self._load_or_create_db()

    def _initialize_embeddings(self, embedding_type: str, openai_api_key: str = None, model_kwargs: dict = None):
        """Initialize the specified embedding model"""
        if embedding_type == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            return OpenAIEmbeddings(openai_api_key=openai_api_key)

        elif embedding_type == "bge-m3":
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

    def _load_or_create_db(self) -> Optional[Chroma]:
        """Load existing vector database or return None if not found"""
        if os.path.exists(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return None

    def _get_toc_chunks(self, pdf_path: str) -> Optional[List[Tuple[str, int, int]]]:
        """
        Extract TOC information and page ranges from PDF

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            Optional[List[Tuple[str, int, int]]]: List of (title, start_page, end_page) or None if no TOC
        """
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()

        if not toc:
            doc.close()
            return None

        chunks = []
        for i, (level, title, start_page) in enumerate(toc):
            start_page -= 1  # Convert to 0-based indexing
            if i + 1 < len(toc):
                end_page = toc[i + 1][2] - 1
            else:
                end_page = len(doc) - 1
            chunks.append((title, start_page, end_page))

        doc.close()
        return chunks

    def _process_pdf_by_toc(self, pdf_path: str, temp_dir: str) -> List[Document]:
        """
        Process PDF file into chunks based on TOC structure

        Args:
            pdf_path (str): Path to the PDF file
            temp_dir (str): Directory for temporary split PDF files

        Returns:
            List[Document]: List of document chunks
        """
        os.makedirs(temp_dir, exist_ok=True)
        pdf_doc = fitz.open(pdf_path)
        documents = []

        chunks = self._get_toc_chunks(pdf_path)
        for title, start, end in chunks:
            # Create temporary PDF for this section
            chunk_doc = fitz.open()
            for page_num in range(start, end + 1):
                chunk_doc.insert_pdf(pdf_doc, from_page=page_num, to_page=page_num)

            # Extract text using pdfplumber
            safe_title = "".join(c if c.isalnum() or c in " -" else "_" for c in title)
            temp_path = os.path.join(temp_dir, f"{safe_title}.pdf")
            chunk_doc.save(temp_path)
            chunk_doc.close()

            with pdfplumber.open(temp_path) as plumber_pdf:
                text = ""
                for page in plumber_pdf.pages:
                    text += page.extract_text() or ""

                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "section": title,
                            "page_range": f"{start + 1}-{end + 1}",
                            "full_path": pdf_path,
                        }
                    )
                    documents.append(doc)

            # Clean up temporary file
            # os.remove(temp_path)

        doc.close()
        return self.text_splitter.split_documents(documents)

    def _process_pdf_regular(self, pdf_path: str) -> List[Document]:
        """Process PDF file into chunks without TOC consideration"""
        documents = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "full_path": pdf_path,
                        }
                    )
                    documents.append(doc)

        return self.text_splitter.split_documents(documents)

    def index_pdf(self, pdf_path: str, mode: str = "append", temp_dir: str = "./temp_pdfs") -> None:
        """
        Create or update index from PDF file, handling both TOC and non-TOC PDFs

        Args:
            pdf_path (str): Path to the PDF file
            mode (str): 'append' to add to existing database, 'overwrite' to create new database
            temp_dir (str): Directory for temporary files when processing TOC-based PDFs
        """
        # Check if PDF has TOC and process accordingly
        if self._get_toc_chunks(pdf_path):
            chunks = self._process_pdf_by_toc(pdf_path, temp_dir)
        else:
            chunks = self._process_pdf_regular(pdf_path)

        if mode == "append" and self.vectordb is not None:
            self.vectordb.add_documents(chunks)
            self.vectordb.persist()
        else:
            self.vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.vectordb.persist()

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for similar document chunks

        Args:
            query (str): Search query
            k (int): Number of similar chunks to return

        Returns:
            List[Dict]: List of dictionaries containing search results
        """
        if self.vectordb is None:
            raise RuntimeError("No database found. Please index a PDF first.")

        results = self.vectordb.similarity_search(query, k=k)
        return [
            {
                "page": doc.metadata.get("page", "N/A"),
                "section": doc.metadata.get("section", "N/A"),
                "page_range": doc.metadata.get("page_range", "N/A"),
                "content": doc.page_content,
                "source": doc.metadata["source"],
                "full_path": doc.metadata.get("full_path", "Unknown")
            }
            for doc in results
        ]

    def get_stats(self) -> Dict:
        """Get database statistics"""
        if self.vectordb is None:
            return {
                "total_documents": 0,
                "embedding_type": self.embedding_type,
                "persist_directory": self.persist_directory
            }

        return {
            "total_documents": self.vectordb._collection.count(),
            "embedding_type": self.embedding_type,
            "persist_directory": self.persist_directory
        }


def main():
    # Create a PDF processor instance
    processor = PDFProcessor(
        embedding_type="bge-m3",
        persist_directory="./chroma_db",
        model_kwargs={"model_kwargs": {"device": "cpu"}}
    )

    # Index PDF files
    # processor.index_pdf("path/to/your/pdf", mode="append")
    processor.index_pdf("data/CCOP_1.pdf", mode="append")
    # processor.index_pdf("data/DOC-UseMenu_cashier_user.pdf", mode="append")
    print("Documents indexed")
    print(processor.get_stats())

    # Search documents
    try:
        results = processor.search("your search query")
        for result in results:
            print(f"File: {result['source']}")
            print(f"Page/Section: {result.get('page', 'N/A')} / {result.get('section', 'N/A')}")
            print(f"Page Range: {result.get('page_range', 'N/A')}")
            print(f"Content: {result['content']}")
            print("---")

    except RuntimeError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()