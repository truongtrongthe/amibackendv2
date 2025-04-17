#!/usr/bin/env python3
"""
NLTK Setup Script for Production Environments

This script ensures that NLTK and its required data packages are properly installed.
Run this script during your deployment process to ensure all required NLTK data
is available in your production environment.
"""

import os
import sys
import nltk
import ssl

def setup_nltk():
    """
    Set up NLTK with required data packages for production use.
    Handles SSL certificate issues that might occur in some environments.
    """
    print("Setting up NLTK data...")
    
    # Create nltk_data directory in the application directory if it doesn't exist
    nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Set NLTK data path to use our custom directory
    nltk.data.path.insert(0, nltk_data_dir)
    
    # Set environment variable for future processes
    os.environ['NLTK_DATA'] = nltk_data_dir
    
    # List of required NLTK data packages
    required_packages = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger'
    ]
    
    # Handle potential SSL certificate issues
    try:
        for package in required_packages:
            try:
                # Check if package is already downloaded
                nltk.data.find(f'tokenizers/{package}')
                print(f"✓ {package} is already downloaded")
            except LookupError:
                print(f"Downloading {package}...")
                nltk.download(package, download_dir=nltk_data_dir, quiet=False)
    except ssl.SSLError:
        print("SSL Error encountered. Trying with certificate verification disabled...")
        try:
            # Create a custom SSL context that doesn't verify certificates
            # WARNING: This is less secure, but sometimes necessary in production environments
            _create_unverified_https_context = ssl._create_unverified_context
            ssl._create_default_https_context = _create_unverified_https_context
            
            # Try downloading again
            for package in required_packages:
                if not nltk.data.find(f'tokenizers/{package}', quiet=True):
                    nltk.download(package, download_dir=nltk_data_dir, quiet=False)
        except Exception as e:
            print(f"Failed to download NLTK data: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"General error: {e}")
        sys.exit(1)
    
    print("\nNLTK setup complete!")
    print(f"NLTK data directory: {nltk_data_dir}")
    print(f"NLTK version: {nltk.__version__}")

if __name__ == "__main__":
    setup_nltk() 