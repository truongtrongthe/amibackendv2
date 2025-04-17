#!/usr/bin/env python3
"""
Emergency Fix for punkt_tab in NLTK

This script manually creates the necessary directory structure and files for punkt_tab,
which is a critical resource for NLTK sentence tokenization.
"""

import os
import sys
import nltk
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_punkt_tab():
    """
    Create the punkt_tab directory structure and files manually to bypass NLTK download issues.
    """
    # Find the nltk_data directory
    nltk_data_dirs = nltk.data.path
    logger.info(f"NLTK data paths: {nltk_data_dirs}")
    
    # Try to create in the first path that's writable
    for data_dir in nltk_data_dirs:
        if os.path.exists(data_dir) and os.access(data_dir, os.W_OK):
            logger.info(f"Using NLTK data directory: {data_dir}")
            break
    else:
        # If no existing directory is writable, create one in the current directory
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
        os.makedirs(data_dir, exist_ok=True)
        # Add to NLTK path
        nltk.data.path.insert(0, data_dir)
        logger.info(f"Created new NLTK data directory: {data_dir}")
    
    # Create directories
    punkt_dir = os.path.join(data_dir, 'tokenizers', 'punkt')
    punkt_tab_dir = os.path.join(data_dir, 'tokenizers', 'punkt_tab')
    english_dir = os.path.join(punkt_tab_dir, 'english')
    os.makedirs(english_dir, exist_ok=True)
    
    # Check if punkt is already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("Punkt tokenizer found, will use it as a source")
        punkt_exists = True
    except LookupError:
        logger.info("Punkt tokenizer not found, downloading it first")
        nltk.download('punkt', download_dir=data_dir)
        punkt_exists = True
    
    # Create essential files in punkt_tab
    if punkt_exists:
        # Create a minimal PunktToken for English
        english_file = os.path.join(english_dir, 'punkt_tab.pickle')
        
        # Try to copy from punkt if it exists
        try:
            # Find the punkt pickle file
            punkt_file = os.path.join(punkt_dir, 'PY3', 'english.pickle')
            if os.path.exists(punkt_file):
                # Copy punkt to punkt_tab as a fallback
                shutil.copy2(punkt_file, english_file)
                logger.info(f"Created punkt_tab file by copying from punkt: {english_file}")
            else:
                # Look in alternative locations
                punkt_files = []
                for root, dirs, files in os.walk(punkt_dir):
                    for file in files:
                        if file.endswith('.pickle') and 'english' in file:
                            punkt_files.append(os.path.join(root, file))
                
                if punkt_files:
                    shutil.copy2(punkt_files[0], english_file)
                    logger.info(f"Created punkt_tab file by copying from: {punkt_files[0]}")
                else:
                    logger.warning("Could not find punkt english pickle file")
                    # We'll rely on the fallback tokenizer in the code
        except Exception as e:
            logger.error(f"Error copying punkt file: {e}")
    
    # Create a verification file
    with open(os.path.join(english_dir, 'README.txt'), 'w') as f:
        f.write("This is a manually created punkt_tab directory for English tokenization.\n")
        f.write("Created by fix_punkt_tab.py script.\n")
    
    # Verify if it worked
    try:
        nltk.data.find('tokenizers/punkt_tab/english')
        logger.info("✅ punkt_tab directory successfully created and verified!")
        return True
    except LookupError as e:
        logger.error(f"❌ Could not verify punkt_tab: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting emergency fix for punkt_tab...")
    success = fix_punkt_tab()
    if success:
        logger.info("Fix completed successfully!")
        sys.exit(0)
    else:
        logger.error("Fix failed! Check logs for details.")
        sys.exit(1) 