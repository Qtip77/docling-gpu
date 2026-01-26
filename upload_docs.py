#!/usr/bin/env python3
"""
Script to upload and process all PDF documents from the Docs folder
to the backend API running on localhost.
"""

import os
import sys
import time
import requests
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8000"
DOCS_FOLDER = Path(__file__).parent / "Docs"
UPLOAD_ENDPOINT = f"{BACKEND_URL}/api/documents/upload"
STATUS_ENDPOINT = f"{BACKEND_URL}/api/documents/status"



# Polling settings
POLL_INTERVAL = 5  # seconds
MAX_WAIT_TIME = 600  # 10 minutes max per document


def check_backend_health():
    """Check if the backend is running and healthy."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def upload_document(file_path: Path) -> dict:
    """Upload a single document to the backend."""
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/pdf")}
        response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=None)
        response.raise_for_status()
        return response.json()


def check_status(job_id: str) -> dict:
    """Check the processing status of a job."""
    response = requests.get(f"{STATUS_ENDPOINT}/{job_id}", timeout=10)
    response.raise_for_status()
    return response.json()


def wait_for_completion(job_id: str, filename: str) -> bool:
    """Wait for a job to complete, with progress updates."""
    start_time = time.time()
    
    while time.time() - start_time < MAX_WAIT_TIME:
        status = check_status(job_id)
        current_status = status.get("status", "unknown")
        
        if current_status == "completed":
            chunks = status.get("chunks_count", 0)
            print(f"  ✓ Completed: {chunks} chunks indexed")
            return True
        elif current_status == "failed":
            error = status.get("error", "Unknown error")
            print(f"  ✗ Failed: {error}")
            return False
        else:
            print(f"  Status: {current_status}...", end="\r")
            time.sleep(POLL_INTERVAL)
    
    print(f"  ✗ Timeout after {MAX_WAIT_TIME} seconds")
    return False


def main():
    print("=" * 60)
    print("PDF Document Upload Script")
    print("=" * 60)
    
    # Check if backend is running
    print("\nChecking backend health...")
    if not check_backend_health():
        print("✗ Backend is not running or not healthy!")
        print(f"  Make sure the backend is running at {BACKEND_URL}")
        sys.exit(1)
    print("✓ Backend is healthy\n")
    
    # Get all PDF files
    if not DOCS_FOLDER.exists():
        print(f"✗ Docs folder not found: {DOCS_FOLDER}")
        sys.exit(1)
    
    pdf_files = sorted(DOCS_FOLDER.glob("*.pdf"))
    
    if not pdf_files:
        print("✗ No PDF files found in Docs folder")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files to upload\n")
    
    # Track results
    successful = 0
    failed = 0
    
    # Upload each document
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_file.name}")
        
        try:
            # Upload the document
            result = upload_document(pdf_file)
            job_id = result.get("job_id")
            
            if not job_id:
                print("  ✗ No job ID returned")
                failed += 1
                continue
            
            print(f"  Job ID: {job_id}")
            
            # Wait for processing to complete
            if wait_for_completion(job_id, pdf_file.name):
                successful += 1
            else:
                failed += 1
                
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Upload error: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            failed += 1
        
        print()  # Blank line between files
    
    # Summary
    print("=" * 60)
    print("Upload Summary")
    print("=" * 60)
    print(f"Total files:  {len(pdf_files)}")
    print(f"Successful:   {successful}")
    print(f"Failed:       {failed}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
