"""Test Mistral Document AI endpoint on first 30 pages of a PDF."""
import asyncio
import json
import sys
from pathlib import Path

# Add ingestion to path
sys.path.insert(0, str(Path(__file__).parent / "ingestion"))

from ingestion.app.services.mistral_ocr import process_document, OCRResult
from ingestion.app.config import settings


async def main():
    pdf_path = "/root/docling-gpu/Docs/KubernetesForBeginners-MumshadMannambeth.pdf"
    
    print(f"📄 Processing: {pdf_path}")
    print(f"🔗 Endpoint: {settings.mistral_azure_endpoint}")
    print(f"🤖 Model: {settings.mistral_azure_model}")
    print("-" * 80)
    
    # Process document (first 30 pages due to ocr_max_pages_per_request=30)
    result: OCRResult = await process_document(pdf_path, include_images=False)
    
    print(f"\n✅ OCR Complete!")
    print(f"📊 Pages processed: {len(result.pages)}")
    print(f"🤖 Model used: {result.model}")
    print(f"📈 Usage: {result.usage_info}")
    print("=" * 80)
    
    # Output each page's markdown
    for page in result.pages[:30]:  # First 30 pages
        print(f"\n{'='*80}")
        print(f"📄 PAGE {page.index + 1}")
        print(f"{'='*80}")
        print(f"📐 Dimensions: {page.dimensions}")
        print(f"🖼️  Images: {len(page.images)}")
        print(f"📊 Tables: {len(page.tables)}")
        print(f"🔗 Hyperlinks: {len(page.hyperlinks)}")
        if page.header:
            print(f"📌 Header: {page.header}")
        if page.footer:
            print(f"📌 Footer: {page.footer}")
        print(f"\n--- MARKDOWN CONTENT ---\n")
        print(page.markdown)
        
        # Show table contents if any
        if page.tables:
            print(f"\n--- TABLES ({len(page.tables)}) ---")
            for i, tbl in enumerate(page.tables):
                print(f"\nTable {i+1} (id: {tbl.id}):")
                print(tbl.content)
    
    # Save full result to JSON for analysis
    output_path = "/root/docling-gpu/mistral_ocr_output.json"
    output_data = {
        "model": result.model,
        "usage_info": result.usage_info,
        "pages": [
            {
                "index": p.index,
                "markdown": p.markdown,
                "images": [{"id": img.id} for img in p.images],
                "tables": [{"id": t.id, "content": t.content} for t in p.tables],
                "hyperlinks": p.hyperlinks,
                "header": p.header,
                "footer": p.footer,
                "dimensions": p.dimensions,
            }
            for p in result.pages[:30]
        ]
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n\n💾 Full output saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
