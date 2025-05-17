import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Unstructured for document parsing
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element, Title, NarrativeText, ListItem, Table

# LangChain for text chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGProcessor:
    """
    A class to handle the RAG (Retrieval-Augmented Generation) pipeline:
    1. Load documents using Unstructured.io
    2. Process and extract text elements
    3. Chunk the text for embedding using LangChain
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] = ["\n\n", "\n", ".", " "],
    ):
        """Initialize the RAG processor with chunking parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        
    def load_file(self, file_path: str) -> List[Element]:
        """
        Load and parse a file using Unstructured.io
        
        Args:
            file_path: Path to the file (PDF, DOCX, PPTX, HTML, etc.)
            
        Returns:
            List of Unstructured elements
        """
        logger.info(f"Loading file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Get file extension
        file_ext = Path(file_path).suffix.lower()
        supported_extensions = ['.pdf', '.docx', '.pptx', '.html', '.txt', '.eml']
        
        if file_ext not in supported_extensions:
            logger.warning(f"File extension {file_ext} may not be fully supported")
        
        # Parse the file with Unstructured
        try:
            elements = partition(filename=file_path)
            logger.info(f"Successfully parsed file with {len(elements)} elements")
            return elements
        except Exception as e:
            logger.error(f"Error parsing file: {str(e)}")
            raise
    
    def display_element_types(self, elements: List[Element], num_samples: int = 5) -> None:
        """
        Print sample elements with their types for inspection
        
        Args:
            elements: List of Unstructured elements
            num_samples: Number of samples to display
        """
        logger.info(f"Displaying {min(num_samples, len(elements))} sample elements:")
        
        for i, element in enumerate(elements[:num_samples]):
            preview = element.text[:80] + "..." if len(element.text) > 80 else element.text
            print(f"{i+1}. {element.__class__.__name__} → {preview}")
            
    def elements_to_raw_text(self, elements: List[Element]) -> str:
        """
        Convert all elements to a single text string
        
        Args:
            elements: List of Unstructured elements
            
        Returns:
            Combined text from all elements
        """
        # Filter out empty elements and join with double newlines
        raw_text = "\n\n".join([el.text for el in elements if el.text])
        logger.info(f"Generated raw text with {len(raw_text)} characters")
        return raw_text
    
    def create_chunks_from_text(
        self, 
        text: str, 
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Split text into chunks using LangChain
        
        Args:
            text: Raw text to split
            metadata: Optional metadata to add to each chunk
            
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Creating chunks with size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        # Create Document with metadata if provided
        docs = [Document(page_content=text, metadata=metadata or {})]
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(docs)
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def create_chunks_by_category(
        self, 
        elements: List[Element],
        base_metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Group elements by category (Title, NarrativeText, etc.) and chunk each category
        
        Args:
            elements: List of Unstructured elements
            base_metadata: Base metadata to include with each chunk
            
        Returns:
            List of LangChain Document objects
        """
        # Initialize empty result list
        chunks = []
        
        # Group elements by category
        category_texts = {}
        for element in elements:
            category = element.__class__.__name__
            if category not in category_texts:
                category_texts[category] = []
            category_texts[category].append(element.text)
        
        # Process each category
        for category, texts in category_texts.items():
            # Join texts in this category
            category_text = "\n\n".join([t for t in texts if t])
            
            # Create metadata for this category
            metadata = {**(base_metadata or {}), "category": category}
            
            # Chunk the category text
            category_chunks = self.create_chunks_from_text(category_text, metadata)
            chunks.extend(category_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(category_texts)} categories")
        return chunks
    
    def create_chunks_by_page(
        self, 
        elements: List[Element],
        base_metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Group elements by page number and chunk each page
        
        Args:
            elements: List of Unstructured elements
            base_metadata: Base metadata to include with each chunk
            
        Returns:
            List of LangChain Document objects
        """
        from collections import defaultdict
        
        # Group by page number
        pages = defaultdict(list)
        for element in elements:
            # Some elements might not have page number metadata
            page_num = getattr(element.metadata, "page_number", 0) if hasattr(element, "metadata") else 0
            pages[page_num].append(element.text)
        
        # Initialize empty result list
        chunks = []
        
        # Process each page
        for page_num, texts in pages.items():
            # Join texts on this page
            page_text = "\n\n".join([t for t in texts if t])
            
            # Create metadata for this page
            metadata = {**(base_metadata or {}), "page_number": page_num}
            
            # Chunk the page text
            page_chunks = self.create_chunks_from_text(page_text, metadata)
            chunks.extend(page_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
        return chunks
    
    def process_file(
        self, 
        file_path: str, 
        chunking_strategy: str = "simple",
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Process a file through the full pipeline:
        1. Load and parse with Unstructured
        2. Convert to text
        3. Split into chunks using specified strategy
        
        Args:
            file_path: Path to the file
            chunking_strategy: Strategy for chunking ('simple', 'category', or 'page')
            metadata: Optional metadata to add to each chunk
            
        Returns:
            List of LangChain Document objects (chunks)
        """
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Add source filename to metadata
        metadata["source"] = os.path.basename(file_path)
        
        # Load and parse the file
        elements = self.load_file(file_path)
        
        # (Optional) Display element types for debugging
        # self.display_element_types(elements)
        
        # Apply chunking strategy
        if chunking_strategy == "category":
            chunks = self.create_chunks_by_category(elements, metadata)
        elif chunking_strategy == "page":
            chunks = self.create_chunks_by_page(elements, metadata)
        else:  # simple strategy
            raw_text = self.elements_to_raw_text(elements)
            chunks = self.create_chunks_from_text(raw_text, metadata)
        
        return chunks


def process_document(
    file_path: str,
    chunking_strategy: str = "simple",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    metadata: Dict[str, Any] = None
) -> List[Document]:
    """
    Convenience function to process a document without instantiating the RAGProcessor class.
    
    Args:
        file_path: Path to the document
        chunking_strategy: Strategy for chunking ('simple', 'category', or 'page')
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        metadata: Optional metadata to add to each chunk
        
    Returns:
        List of LangChain Document objects (chunks)
    """
    processor = RAGProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return processor.process_file(
        file_path=file_path,
        chunking_strategy=chunking_strategy,
        metadata=metadata
    )


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process documents for RAG")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--strategy", 
        choices=["simple", "category", "page"], 
        default="simple",
        help="Chunking strategy: simple, category, or page"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=500,
        help="Size of each chunk"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=100,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--print-samples", 
        action="store_true",
        help="Print sample chunks"
    )
    
    args = parser.parse_args()
    
    # Process the document
    processor = RAGProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Load and display element types
    elements = processor.load_file(args.file_path)
    processor.display_element_types(elements)
    
    # Process the file
    chunks = processor.process_file(
        file_path=args.file_path,
        chunking_strategy=args.strategy,
        metadata={"source": os.path.basename(args.file_path)}
    )
    
    # Print statistics
    print(f"\nCreated {len(chunks)} chunks using '{args.strategy}' strategy")
    print(f"Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}")
    
    # Print sample chunks
    if args.print_samples:
        print("\nSample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Content: {chunk.page_content[:150]}...")
            print(f"Metadata: {chunk.metadata}")
