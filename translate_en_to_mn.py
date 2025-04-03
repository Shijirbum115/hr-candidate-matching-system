import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from mongolian_pii_masker import PIIMasker

class CVTranslator:
    """
    Class for translating masked CV content using OpenAI API.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the translator with OpenAI API key.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def translate_text(self, text: str, source_lang: str = 'mn', target_lang: str = 'en') -> str:
        """
        Translate text using OpenAI API.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        if source_lang == target_lang:
            return text
            
        payload = {
            "model": "gpt-4",  # or another appropriate model
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a professional translator specializing in CV/resume translations from {source_lang} to {target_lang}. Translate the text maintaining the formatting and all placeholders like [NAME], [EMAIL], etc. unchanged."
                },
                {
                    "role": "user",
                    "content": f"Translate the following CV from {source_lang} to {target_lang}. Keep all placeholders (like [NAME_1], [PHONE_2], etc.) exactly as they are: \n\n{text}"
                }
            ],
            "temperature": 0.3  # Low temperature for more consistent translations
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            
            translated_text = response_data['choices'][0]['message']['content']
            return translated_text
            
        except requests.exceptions.RequestException as e:
            print(f"Error translating text: {e}")
            if response.status_code != 200:
                print(f"API Response: {response.text}")
            return text  # Return original text on error
    
    def process_masked_file(self, masked_file_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process a single masked CV file.
        
        Args:
            masked_file_path: Path to the masked CV JSON file
            output_dir: Directory to save translated results
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Load masked CV data
            with open(masked_file_path, 'r', encoding='utf-8') as f:
                masked_data = json.load(f)
            
            masked_text = masked_data.get('masked_text', '')
            language = masked_data.get('language', 'en')
            
            # Only translate if source is Mongolian
            if language == 'mn':
                translated_text = self.translate_text(masked_text, source_lang='mn', target_lang='en')
                
                # Add translation to data
                masked_data['translated_text'] = translated_text
                
                # Save translated result
                output_path = Path(output_dir) / f"{Path(masked_file_path).stem}_translated.json"
                os.makedirs(output_dir, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(masked_data, f, ensure_ascii=False, indent=2)
                    
                print(f"Translated: {Path(masked_file_path).name}")
                return masked_data
                
            else:
                print(f"Skipped translation for {Path(masked_file_path).name} (already in English)")
                return masked_data
                
        except Exception as e:
            print(f"Error processing {masked_file_path}: {e}")
            return {}
    
    def process_directory(self, masked_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        Process all masked CV files in a directory.
        
        Args:
            masked_dir: Directory containing masked CV JSON files
            output_dir: Directory to save translated results
            
        Returns:
            List of dictionaries with processing results
        """
        masked_files = list(Path(masked_dir).glob('*_masked.json'))
        results = []
        
        for masked_file in masked_files:
            result = self.process_masked_file(str(masked_file), output_dir)
            if result:
                results.append(result)
        
        # Save batch results
        if results and output_dir:
            batch_output = Path(output_dir) / "translated_batch_results.json"
            with open(batch_output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results


def main():
    """Main function to demonstrate CV translation workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mask and translate CVs.')
    parser.add_argument('--pdf_dir', required=True, help='Directory containing CV PDFs')
    parser.add_argument('--masked_dir', required=True, help='Directory to save masked CVs')
    parser.add_argument('--translated_dir', required=True, help='Directory to save translated CVs')
    parser.add_argument('--openai_api_key', required=True, help='OpenAI API key')
    
    args = parser.parse_args()
    
    # Step 1: Mask personal information in CVs
    print("Step 1: Masking personal information in CVs...")
    masker = PIIMasker()
    masked_results = masker.process_pdf_directory(args.pdf_dir, args.masked_dir)
    print(f"Masked {len(masked_results)} CV files")
    
    # Step 2: Translate masked CVs from Mongolian to English
    print("\nStep 2: Translating masked CVs...")
    translator = CVTranslator(args.openai_api_key)
    translated_results = translator.process_directory(args.masked_dir, args.translated_dir)
    print(f"Translated {len(translated_results)} CV files")
    
    print("\nWorkflow completed successfully!")


if __name__ == "__main__":
    # Example usage with explicit arguments:
    print("To run the CV masking and translation workflow:")
    print("""
    python translate_masked_cvs.py \\
        --pdf_dir /path/to/your/pdf/cvs \\
        --masked_dir /path/to/save/masked/results \\
        --translated_dir /path/to/save/translated/results \\
        --openai_api_key your_openai_api_key
    """)
    
    # You can also set environment variables and uncomment this to run directly:
    # if os.environ.get('OPENAI_API_KEY'):
    #     main()
    # else:
    #     print("Please set OPENAI_API_KEY environment variable or provide it as an argument.")