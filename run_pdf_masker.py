import os
from pdf_masker import PDFMasker

# Set paths
pdf_dir = r"C:\Users\shijirbum.b\HRCandidateMatchingSystem\cv_raw"
masked_pdf_dir = r"C:\Users\shijirbum.b\HRCandidateMatchingSystem\cv_masked_pdf"
masking_info_dir = r"C:\Users\shijirbum.b\HRCandidateMatchingSystem\cv_masking_info"

# Create output directories if they don't exist
os.makedirs(masked_pdf_dir, exist_ok=True)
os.makedirs(masking_info_dir, exist_ok=True)

# Initialize the masker
masker = PDFMasker()

# Process all PDFs in the directory
results = masker.process_pdf_directory(pdf_dir, masked_pdf_dir, masking_info_dir)

# Print summary
print(f"Processed {len(results)} CV files")
print(f"Masked PDFs saved to {masked_pdf_dir}")
print(f"Masking information saved to {masking_info_dir}")

for result in results:
    filename = result.get('filename')
    total_masked = result.get('total_entities_masked', 0)
    error = result.get('error')
    
    if error:
        print(f"Error processing {filename}: {error}")
    else:
        print(f"Successfully masked {filename}: {total_masked} entities masked")