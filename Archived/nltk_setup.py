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
        'punkt',          # Basic sentence tokenizer
        'stopwords',      # Stopwords for various languages
        'wordnet',        # WordNet lexical database
        'averaged_perceptron_tagger'  # Part-of-speech tagger
    ]
    
    # Special handling for punkt_tab which is needed for certain operations
    punkt_tab_packages = [
        'punkt_tab.english',  # English punkt tab
        'punkt_tab.german',   # German punkt tab
        'punkt_tab.portuguese', # Portuguese punkt tab
        'punkt_tab.spanish',  # Spanish punkt tab
        'punkt_tab.turkish',  # Turkish punkt tab
        'punkt_tab.italian',  # Italian punkt tab
        'punkt_tab.french',   # French punkt tab
        'punkt_tab.dutch',    # Dutch punkt tab
        'punkt_tab.czech',    # Czech punkt tab
    ]
    
    # Disable SSL verification for downloading NLTK data
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download required NLTK data for unstructured.io
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('punkt_tab')
    
    print("NLTK data download completed successfully.")
    
    # Handle potential SSL certificate issues
    try:
        # 1. Download regular packages
        for package in required_packages:
            try:
                # Check if package is already downloaded
                nltk.data.find(f'tokenizers/{package}')
                print(f"✓ {package} is already downloaded")
            except LookupError:
                print(f"Downloading {package}...")
                nltk.download(package, download_dir=nltk_data_dir, quiet=False)
        
        # 2. Additional specific handling for punkt_tab resources
        print("\nChecking punkt_tab resources...")
        
        # Try installing punkt_tab (multiple attempts as it's organized differently)
        try:
            nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=False)
            print("✓ Downloaded punkt_tab package")
        except Exception as e:
            print(f"Warning: Could not download punkt_tab directly: {e}")
            print("Attempting to download punkt_tabs individually...")
            
            # Try to download each language-specific punkt_tab
            for lang_package in punkt_tab_packages:
                try:
                    nltk.download(lang_package, download_dir=nltk_data_dir, quiet=False)
                    print(f"✓ Downloaded {lang_package}")
                except Exception as e:
                    print(f"Warning: Failed to download {lang_package}: {e}")
        
        # 3. Alternative method: download 'all-corpora' for comprehensive installation
        print("\nChecking if comprehensive installation is needed...")
        try:
            nltk.data.find('tokenizers/punkt_tab/english/')
            print("✓ punkt_tab resources are available")
        except LookupError:
            print("punkt_tab resources not found. Attempting comprehensive download...")
            try:
                nltk.download('all-corpora', download_dir=nltk_data_dir, quiet=False)
                print("✓ Downloaded comprehensive NLTK resources")
            except Exception as e:
                print(f"Warning: Comprehensive download failed: {e}")
                
                # Last resort: Try to create the directory structure and download punkt
                try:
                    os.makedirs(os.path.join(nltk_data_dir, 'tokenizers', 'punkt_tab', 'english'), exist_ok=True)
                    nltk.download('punkt', download_dir=nltk_data_dir, force=True, quiet=False)
                    print("✓ Force-downloaded punkt which should include punkt_tab")
                except Exception as e:
                    print(f"Error in last-resort download: {e}")
    
    except ssl.SSLError:
        print("SSL Error encountered. Trying with certificate verification disabled...")
        try:
            # Create a custom SSL context that doesn't verify certificates
            # WARNING: This is less secure, but sometimes necessary in production environments
            _create_unverified_https_context = ssl._create_unverified_context
            ssl._create_default_https_context = _create_unverified_https_context
            
            # Try downloading again with SSL verification disabled
            for package in required_packages:
                if not nltk.data.find(f'tokenizers/{package}', quiet=True):
                    nltk.download(package, download_dir=nltk_data_dir, quiet=False)
            
            # Try punkt_tab with SSL verification disabled
            try:
                nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=False)
            except:
                nltk.download('all-corpora', download_dir=nltk_data_dir, quiet=False)
        except Exception as e:
            print(f"Failed to download NLTK data: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"General error: {e}")
        sys.exit(1)
    
    print("\nNLTK setup complete!")
    print(f"NLTK data directory: {nltk_data_dir}")
    print(f"NLTK version: {nltk.__version__}")
    
    # Final verification
    print("\nVerifying key resources:")
    verification_checks = [
        ('punkt', 'tokenizers/punkt'),
        ('punkt_tab', 'tokenizers/punkt_tab/english'),
        ('stopwords', 'corpora/stopwords')
    ]
    
    for name, path in verification_checks:
        try:
            nltk.data.find(path)
            print(f"✓ {name} is available")
        except LookupError:
            print(f"✗ {name} was not successfully installed - application may encounter errors")

if __name__ == "__main__":
    setup_nltk() 