import os
from improved_pdf_masker import PDFMasker, PDFLayoutMasker

# Set paths
pdf_dir = r"C:\Users\shijirbum.b\HRCandidateMatchingSystem\cv_raw"
masked_pdf_dir = r"C:\Users\shijirbum.b\HRCandidateMatchingSystem\cv_masked_pdf"
masking_info_dir = r"C:\Users\shijirbum.b\HRCandidateMatchingSystem\cv_masking_info"

# Create output directories if they don't exist
os.makedirs(masked_pdf_dir, exist_ok=True)
os.makedirs(masking_info_dir, exist_ok=True)

# Use layout-aware masking with image removal
print("Using layout-aware masking with image removal...")
masker = PDFLayoutMasker()

# Process all PDFs in the directory
results = masker.process_pdf_directory(pdf_dir, masked_pdf_dir, masking_info_dir)

# Print summary
print(f"Processed {len(results)} CV files")
print(f"Masked PDFs saved to {masked_pdf_dir}")
print(f"Masking information saved to {masking_info_dir}")

# Print detailed results
for result in results:
    filename = result.get('filename')
    total_masked = result.get('total_entities_masked', 0)
    images_removed = result.get('images_removed', 0)
    error = result.get('error')
    
    if error:
        print(f"Error processing {filename}: {error}")
    else:
        print(f"Successfully masked {filename}:")
        print(f"  Entities masked: {total_masked}")
        print(f"  Images removed: {images_removed}")
        
        # Print types of entities masked
        if 'entities_masked' in result:
            print("  Entity types masked:")
            for entity_type, entities in result['entities_masked'].items():
                print(f"    {entity_type}: {len(entities)} entities")
                # Print up to 3 examples
                if entities:
                    examples = entities[:3]
                    print(f"      Examples: {', '.join(examples)}")
        
        print("-" * 50)

print("\nTo verify the results, please check the masked PDFs in the output directory.")
print("The redacted information should appear as black rectangles, and images should be removed.")