import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai
import logging
from docx import Document
import os
from typing import List, Tuple, Dict
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the LLM client with timeout
LLM = ChatOpenAI(model="gpt-4o", streaming=False, request_timeout=60)

# Retry decorator for LLM calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(openai.APITimeoutError)
)
async def invoke_llm_with_retry(llm, prompt, temperature=0.2, max_tokens=5000, stop=None):
    """Call LLM with retry logic for timeouts and transient errors"""
    try:
        return await llm.ainvoke(prompt, stop=stop, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        logger.warning(f"LLM call error: {type(e).__name__}: {str(e)}")
        raise

def load_and_split_document(file_path: str) -> List[str]:
    """
    Load a docx document and split it into sentences.
    
    Args:
        file_path: Path to the docx file
        
    Returns:
        List of sentences
    """
    try:
        doc = Document(file_path)
        full_text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        sentences = sent_tokenize(full_text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception as e:
        logger.error(f"Error loading document: {str(e)}")
        raise

def generate_sentence_embeddings(sentences: List[str]) -> np.ndarray:
    """
    Generate embeddings for sentences using LaBSE.
    
    Args:
        sentences: List of sentences to embed
        
    Returns:
        numpy array of embeddings
    """
    try:
        model = SentenceTransformer('sentence-transformers/LaBSE')
        embeddings = model.encode(sentences, show_progress_bar=True)
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def perform_clustering(embeddings: np.ndarray) -> np.ndarray:
    """
    Cluster sentences using HDBSCAN.
    
    Args:
        embeddings: Sentence embeddings
        
    Returns:
        Array of cluster labels
    """
    try:
        # Further adjusted parameters for smaller, more focused clusters
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,           # Keep small clusters
            min_samples=1,                # Allow single points to form clusters
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=0.25, # Further reduced to create even tighter clusters
            alpha=0.8,                    # Reduced to encourage more clusters
            prediction_data=True,         # Enable prediction data for better stability
            allow_single_cluster=False    # Prevent all points from being in one cluster
        )
        
        # Fit and predict
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Log clustering statistics
        unique_clusters = set(cluster_labels)
        noise_count = sum(1 for label in cluster_labels if label == -1)
        cluster_sizes = {label: sum(1 for l in cluster_labels if l == label) 
                        for label in unique_clusters if label != -1}
        
        logger.info(f"Clustering statistics:")
        logger.info(f"  Total clusters: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}")
        logger.info(f"  Noise points: {noise_count}")
        logger.info(f"  Cluster sizes: {cluster_sizes}")
        
        # If we have too few clusters, try again with different parameters
        if len(unique_clusters) - (1 if -1 in unique_clusters else 0) < 2:
            logger.info("Too few clusters detected, retrying with different parameters...")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=2,
                min_samples=1,
                metric='euclidean',
                cluster_selection_method='leaf',
                cluster_selection_epsilon=0.15,
                alpha=0.6,
                prediction_data=True,
                allow_single_cluster=False
            )
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Log new clustering statistics
            unique_clusters = set(cluster_labels)
            noise_count = sum(1 for label in cluster_labels if label == -1)
            cluster_sizes = {label: sum(1 for l in cluster_labels if l == label) 
                            for label in unique_clusters if label != -1}
            
            logger.info(f"Retry clustering statistics:")
            logger.info(f"  Total clusters: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}")
            logger.info(f"  Noise points: {noise_count}")
            logger.info(f"  Cluster sizes: {cluster_sizes}")
        
        return cluster_labels
    except Exception as e:
        logger.error(f"Error clustering sentences: {str(e)}")
        raise

def select_all_sentences(
    sentences: List[str],
    cluster_labels: np.ndarray
) -> Dict[int, List[str]]:
    """
    Select all sentences for each cluster, allowing sentences to appear in multiple clusters.
    
    Args:
        sentences: List of sentences
        cluster_labels: Cluster labels for each sentence
        
    Returns:
        Dictionary mapping cluster IDs to lists of sentences
    """
    try:
        cluster_sentences = {}
        unique_clusters = set(cluster_labels)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get all sentences for this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_sentences[cluster_id] = [
                sentences[i] for i in range(len(sentences)) if cluster_mask[i]
            ]
            
        return cluster_sentences
    except Exception as e:
        logger.error(f"Error selecting sentences: {str(e)}")
        raise

async def generate_cluster_title(sentences: List[str]) -> str:
    """
    Generate a descriptive title for a cluster using GPT-4o.
    
    Args:
        sentences: List of sentences
        
    Returns:
        A descriptive title for the cluster
    """
    try:
        sentences_text = "\n".join([f"- {s}" for s in sentences])
        prompt = (
            f"You are a knowledge organization expert.\n\n"
            f"TASK: Based on the following sentences, generate a SHORT, DESCRIPTIVE TITLE (3-5 words) that captures the core theme or topic.\n"
            f"If the sentences are not concise enough, rephrase them into a clear, focused title.\n\n"
            f"GUIDELINES:\n"
            f"1. The title should be in the SAME LANGUAGE as the sentences\n"
            f"2. If the sentences are verbose or unclear, create a concise title that captures the essence\n"
            f"3. Use clear, direct language that immediately conveys the main idea\n"
            f"4. Keep technical terms and domain-specific vocabulary\n"
            f"5. Make the title actionable and specific\n\n"
            f"SENTENCES:\n{sentences_text}\n\n"
            f"TITLE:"
        )
        
        response = await invoke_llm_with_retry(LLM, prompt, max_tokens=25)
        title = response.content.strip()
        
        # Remove quotes if present
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        
        return title
    except Exception as e:
        logger.error(f"Error generating cluster title: {str(e)}")
        return "Unnamed Cluster"

async def generate_cluster_description(
    cluster_sentences: Dict[int, List[str]],
    cluster_id: int,
    all_clusters: Dict[int, List[str]]
) -> str:
    """
    Generate a description of the cluster and its relationships to other clusters.
    
    Args:
        cluster_sentences: Dictionary mapping cluster IDs to sentences
        cluster_id: ID of the current cluster
        all_clusters: Dictionary of all clusters
        
    Returns:
        A description of the cluster and its relationships
    """
    try:
        # Get sentences from current cluster
        current_sentences = cluster_sentences[cluster_id]
        
        # Get sentences from other clusters
        other_clusters = {cid: sents for cid, sents in all_clusters.items() 
                         if cid != cluster_id and cid != -1}
        
        # Prepare the prompt
        current_text = "\n".join([f"- {s}" for s in current_sentences])
        other_text = "\n".join([f"Cluster {cid}:\n" + "\n".join([f"- {s}" for s in sents])
                              for cid, sents in other_clusters.items()])
        
        prompt = (
            f"You are a knowledge organization expert.\n\n"
            f"TASK: Analyze the following cluster of sentences and describe them in the EXACT SAME LANGUAGE as the sentences.\n"
            f"DO NOT TRANSLATE TO ENGLISH. Use the same words, phrases, and expressions as in the original sentences.\n\n"
            f"EXAMPLE:\n"
            f"If the sentences are in Vietnamese, write the description in Vietnamese.\n"
            f"If the sentences are in Japanese, write the description in Japanese.\n"
            f"If the sentences are in Spanish, write the description in Spanish.\n\n"
            f"REQUIREMENTS:\n"
            f"1. Use the EXACT SAME LANGUAGE as the sentences - no translation allowed\n"
            f"2. Keep all technical terms and domain-specific vocabulary exactly as they appear\n"
            f"3. Maintain the same tone, style, and expressions\n"
            f"4. Preserve all cultural context and nuances\n"
            f"5. Do not add any English words or phrases\n"
            f"6. If the sentences use informal language, use the same informal style\n"
            f"7. If the sentences use formal language, use the same formal style\n\n"
            f"DESCRIBE:\n"
            f"1. The main theme and purpose of this cluster\n"
            f"2. How it relates to other clusters in the document\n"
            f"3. The role it plays in the overall context\n\n"
            f"CURRENT CLUSTER SENTENCES:\n{current_text}\n\n"
            f"OTHER CLUSTERS:\n{other_text}\n\n"
            f"DESCRIPTION (write in the same language as the sentences above):"
        )
        
        response = await invoke_llm_with_retry(LLM, prompt, max_tokens=200)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error generating cluster description: {str(e)}")
        return ""

async def generate_takeaways(
    cluster_sentences: Dict[int, List[str]],
    sentences: List[str]
) -> Dict[int, Tuple[str, str, str]]:
    """
    Generate titles, descriptions, and application-focused takeaways for each cluster using GPT-4o.
    
    Args:
        cluster_sentences: Dictionary mapping cluster IDs to sentences
        sentences: Complete list of all sentences from the document for context
        
    Returns:
        Dictionary mapping cluster IDs to (title, description, takeaways) tuples
    """
    try:
        results = {}
        
        # Get a sample of the full document for context
        doc_sample = "\n".join(sentences[:30] if len(sentences) > 30 else sentences)
        
        for cluster_id, cluster_sents in cluster_sentences.items():
            # Generate cluster title
            title = await generate_cluster_title(cluster_sents)
            logger.info(f"Generated title for cluster {cluster_id}: {title}")
            
            # Generate cluster description and relationships
            description = await generate_cluster_description(cluster_sentences, cluster_id, cluster_sentences)
            logger.info(f"Generated description for cluster {cluster_id}")
            
            # Prepare the prompt for takeaways
            sentences_text = "\n".join([f"- {s}" for s in cluster_sents])
            prompt = (
                f"You are a knowledge application expert specializing in sales.\n\n"
                f"DOCUMENT CONTEXT (sample from the full document):\n{doc_sample}\n\n"
                f"TASK: Analyze the following sentences and generate detailed, practical takeaways focusing SPECIFICALLY on HOW TO APPLY these insights in sales contexts.\n"
                f"IMPORTANT: Your response MUST be in the SAME LANGUAGE as the sentences.\n\n"
                f"GUIDELINES:\n"
                f"1. Focus on METHODS OF APPLICATION - explain exactly HOW to apply these insights\n"
                f"2. Structure your response with numbered steps or a clear methodology\n"
                f"3. Provide SPECIFIC EXAMPLE SCRIPTS, dialogues, or templates that demonstrate the application\n"
                f"4. Include step-by-step instructions for implementation\n"
                f"5. Consider the document context when interpreting these sentences\n"
                f"6. Format your response as 'Application Method: [title]' followed by the steps\n"
                f"7. Each application method should be immediately actionable\n"
                f"8. IMPORTANT: Maintain the original language of the sentences in your response\n"
                f"9. Include specific factual details and nuanced observations from the text\n"
                f"10. Highlight any subtle but important distinctions or variations in approach\n"
                f"11. Note any potential edge cases or special situations to consider\n"
                f"12. Use the same terminology and expressions as in the original text\n"
                f"13. Keep all technical terms and domain-specific vocabulary in their original form\n\n"
                f"SENTENCES TO ANALYZE:\n{sentences_text}\n\n"
                f"APPLICATION METHODS (with specific examples and step-by-step instructions):"
            )
            
            # Generate takeaway
            response = await invoke_llm_with_retry(LLM, prompt)
            results[cluster_id] = (title, description, response.content.strip())
            
        return results
    except Exception as e:
        logger.error(f"Error generating takeaways: {str(e)}")
        raise

def save_results(
    cluster_sentences: Dict[int, List[str]],
    cluster_results: Dict[int, Tuple[str, str, str]],
    output_path: str
) -> None:
    """
    Save results to a markdown file.
    
    Args:
        cluster_sentences: Dictionary mapping cluster IDs to sentences
        cluster_results: Dictionary mapping cluster IDs to (title, description, takeaways) tuples
        output_path: Path to save the output file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Key Points Extraction Results\n\n")
            
            for cluster_id in sorted(cluster_sentences.keys()):
                title, description, takeaways = cluster_results[cluster_id]
                f.write(f"## Cluster {cluster_id}: {title}\n\n")
                
                # Write cluster description
                f.write("### Description\n")
                f.write(f"{description}\n\n")
                
                # Write sentences
                f.write("### Sentences\n")
                for sentence in cluster_sentences[cluster_id]:
                    f.write(f"- {sentence}\n")
                f.write("\n")
                
                # Write takeaways
                f.write("### Application Methods\n")
                f.write(f"{takeaways}\n\n")
                
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

async def process_document(input_path: str, output_path: str) -> None:
    """
    Main function to process a document and extract key points.
    
    Args:
        input_path: Path to input docx file
        output_path: Path to save output markdown file
    """
    try:
        # Load and split document
        logger.info("Loading and splitting document...")
        sentences = load_and_split_document(input_path)
        
        # Generate embeddings
        logger.info("Generating sentence embeddings...")
        embeddings = generate_sentence_embeddings(sentences)
        
        # Cluster sentences
        logger.info("Clustering sentences...")
        cluster_labels = perform_clustering(embeddings)
        
        # Log clustering results
        unique_clusters = set(cluster_labels)
        noise_count = sum(1 for label in cluster_labels if label == -1)
        logger.info(f"Clustering results: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)} clusters, {noise_count} noise points")
        for cluster_id in sorted(unique_clusters):
            if cluster_id == -1:
                continue
            count = sum(1 for label in cluster_labels if label == cluster_id)
            logger.info(f"  Cluster {cluster_id}: {count} sentences")
        
        # Select all sentences for each cluster
        logger.info("Selecting sentences for each cluster...")
        cluster_sentences = select_all_sentences(sentences, cluster_labels)
        
        # Generate cluster titles and takeaways
        logger.info("Generating cluster titles and application methods...")
        cluster_results = await generate_takeaways(cluster_sentences, sentences)
        
        # Save results
        logger.info("Saving results...")
        save_results(cluster_sentences, cluster_results, output_path)
        
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    input_file = "input.docx"
    output_file = "key_points_application.md"
    
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found")
    else:
        import asyncio
        asyncio.run(process_document(input_file, output_file))
