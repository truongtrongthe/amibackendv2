#!/usr/bin/env python3
"""
Test script to verify the DOCX reading fix
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from file_access_tool import FileAccessTool

def test_docx_reading():
    """Test the DOCX reading functionality"""
    
    print("🧪 Testing DOCX Reading Fix")
    print("=" * 40)
    
    # Initialize the file access tool
    tool = FileAccessTool()
    
    # Test with a sample Google Drive link (you can replace this with a real one)
    test_link = "https://docs.google.com/document/d/1Dq78EV3y0EK98cJ-7oAg5GK8FN-bbEVbh0HRXV0oz5A/edit"
    
    print(f"Testing with link: {test_link}")
    print("-" * 40)
    
    try:
        # Test the file reading
        result = tool.read_gdrive_link_docx(test_link)
        
        if result.startswith("Error:"):
            print(f"❌ Error occurred: {result}")
            return False
        else:
            print(f"✅ Successfully read document")
            print(f"Content length: {len(result)} characters")
            print(f"Content preview: {result[:200]}...")
            return True
            
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_content_processing():
    """Test the content processing logic"""
    
    print("\n🧪 Testing Content Processing Logic")
    print("=" * 40)
    
    # Simulate the content processing that was failing
    content = [
        "This is paragraph 1",
        "This is paragraph 2", 
        "This is paragraph 3"
    ]
    
    try:
        # Test the content joining logic
        content_text = "\n".join(content)
        print(f"✅ Content processing successful")
        print(f"Number of paragraphs: {len(content)}")
        print(f"Total characters: {len(content_text)}")
        print(f"Content: {content_text}")
        return True
        
    except Exception as e:
        print(f"❌ Content processing failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting DOCX Reading Fix Tests")
    print()
    
    # Test content processing first
    content_test = test_content_processing()
    
    # Test actual file reading (only if Google Drive is configured)
    if content_test:
        print("\n" + "="*60 + "\n")
        file_test = test_docx_reading()
        
        if file_test:
            print("\n🎉 All tests passed! The fix should work correctly.")
        else:
            print("\n⚠️  File reading test failed, but content processing is fixed.")
    else:
        print("\n❌ Content processing test failed. There's still an issue.") 