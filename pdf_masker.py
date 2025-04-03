import re
import os
import json
import fitz  # PyMuPDF
import spacy
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Union

# Cyrillic character set used in Mongolian
MONGOLIAN_CYRILLIC = set('АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмноөпрстуүфхцчшщъыьэюя')

# Default configuration
DEFAULT_CONFIG = {
    # Supported languages
    'languages': ['en', 'mn'],
    
    # NER models to use
    'models': {
        'en': 'en_core_web_md',
        'mn': None  # No default Mongolian model
    },
    
    # Entity types to detect and mask
    'entity_types': [
        'PERSON', 'EMAIL', 'PHONE', 'ADDRESS', 'LOCATION',
        'ID_NUMBER', 'REGISTER_NUMBER', 'DATE', 'ORGANIZATION',
        'BANK_ACCOUNT', 'SOCIAL_ID', 'URL'
    ],
    
    # Default masking strategy
    'masking_strategy': 'redact',  # 'redact' for PDF masking
    
    # Confidence thresholds
    'confidence_threshold': {
        'en': 0.7,
        'mn': 0.6  # Lower threshold for Mongolian due to limited model accuracy
    }
}

# English regex patterns
EN_PATTERNS = {
    'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'PHONE': r'\b(?:\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b',
    'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
    'CREDIT_CARD': r'\b(?:\d[ -]*?){13,16}\b',
    'IP_ADDRESS': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'URL': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
    'ADDRESS': r'\b\d+\s+[A-Za-z\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr)\b',
}

# Mongolian regex patterns
MN_PATTERNS = {
    # Email (same as English)
    'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    
    # Phone numbers (Mongolian format)
    'PHONE': r'\b(?:\+976|0)?[\s-]?(?:\d[\s-]?){7,11}\b',
    
    # Mongolian ID and register numbers
    'ID_NUMBER': r'\b[А-ЯA-Z]{2}\d{8,10}\b',
    'REGISTER_NUMBER': r'\b[А-ЯA-Z]\d{8,10}\b',
    
    # URLs (same as English)
    'URL': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
    
    # Mongolian address patterns
    'ADDRESS': [
        # General address with city/province
        r'(?:[А-ЯӨҮа-яөү]+\s+)?(?:хот|аймаг|сум|дүүрэг|баг|хороо|гудамж|гэр|тоот|байр)(?:,|\s|\.)+(?:[^\.]{1,50})',
        
        # Numbered addresses (districts, apartments)
        r'(?:\d+(?:-?[а-яөү])?|\d+(?:-\d+)?|-р|-д|-ын|-ийн|-т|-н)\s+(?:хороо|дүүрэг|сум|баг|хот|тоот|орц|давхар)',
        
        # Buildings and apartments
        r'\b\d+(?:[-\s][А-ЯӨҮа-яөү]+)?\s+(?:байр|хаалга)\b',
        r'\b\d+(?:[-\s][А-ЯӨҮа-яөү]+)?\s+(?:тоот)\b'
    ],
    
    # Mongolian personal name patterns 
    'PERSON': r'\b[А-ЯӨҮ][а-яөү]+\s+[А-ЯӨҮ][а-яөү]+\b',
    
    # Mongolian bank account numbers
    'BANK_ACCOUNT': r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',
    
    # Social Insurance Numbers or other government IDs
    'SOCIAL_ID': r'\b\d{10}\b'
}


def detect_language(text: str) -> str:
    """
    Detect whether text is primarily English or Mongolian.
    
    Args:
        text: Input text
        
    Returns:
        Language code: 'en' for English, 'mn' for Mongolian
    """
    # Count Cyrillic characters
    cyrillic_count = sum(1 for char in text if char in MONGOLIAN_CYRILLIC)
    
    # If significant Cyrillic characters are found, classify as Mongolian
    if cyrillic_count > len(text) * 0.15:  # At least 15% Cyrillic
        return 'mn'
    else:
        return 'en'


class EntityMatcher:
    """Class for detecting entities using regex patterns."""
    
    def __init__(self, en_patterns: Dict[str, Any], mn_patterns: Dict[str, Any]):
        """
        Initialize the matcher with patterns.
        
        Args:
            en_patterns: English regex patterns
            mn_patterns: Mongolian regex patterns
        """
        self.patterns = {
            'en': en_patterns,
            'mn': mn_patterns
        }
        
    def find_matches(self, text: str, language: str) -> List[Dict[str, Any]]:
        """
        Find matches in text using appropriate patterns for the language.
        
        Args:
            text: Text to search in
            language: Language code ('en' or 'mn')
            
        Returns:
            List of detected entities
        """
        entities = []
        patterns = self.patterns.get(language, self.patterns['en'])
        
        # Process each pattern
        for entity_type, pattern in patterns.items():
            # Handle multiple patterns for a single entity type
            if isinstance(pattern, list):
                for p in pattern:
                    entities.extend(self._process_pattern(text, p, entity_type))
            else:
                entities.extend(self._process_pattern(text, pattern, entity_type))
                
        return entities
    
    def _process_pattern(self, text: str, pattern: str, entity_type: str) -> List[Dict[str, Any]]:
        """
        Process a single regex pattern and return matches.
        
        Args:
            text: Text to search in
            pattern: Regex pattern
            entity_type: Type of entity to label matches as
            
        Returns:
            List of detected entities
        """
        matches = []
        for match in re.finditer(pattern, text):
            matches.append({
                'type': entity_type,
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'confidence': 1.0  # Regex patterns have full confidence
            })
        return matches


class PDFMasker:
    """
    PII masking tool for PDF files that supports both English and Mongolian text.
    Detects and masks personally identifiable information directly in PDFs.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the PDF masker with configuration options.
        
        Args:
            config: Configuration dictionary overriding defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        # Load NER models based on config
        self.models = {}
        if 'en' in self.config['languages']:
            try:
                self.models['en'] = spacy.load(self.config['models']['en'])
            except OSError:
                import os
                os.system(f"python -m spacy download {self.config['models']['en']}")
                self.models['en'] = spacy.load(self.config['models']['en'])
        
        # Initialize entity matcher for regex patterns
        self.matcher = EntityMatcher(EN_PATTERNS, MN_PATTERNS)
        
        # Track masking information
        self.masked_entities = {}
        
    def mask_pdf(self, pdf_path: str, output_path: str) -> Dict[str, Any]:
        """
        Mask PII in a PDF file and save the masked PDF.
        
        Args:
            pdf_path: Path to the input PDF
            output_path: Path to save the masked PDF
            
        Returns:
            Dictionary with masking metadata
        """
        # Reset tracking for each PDF
        self.masked_entities = {}
        all_entities = []
        
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Process each page
            for page_num, page in enumerate(doc):
                # Get page text
                text = page.get_text()
                
                # Detect language
                language = detect_language(text)
                
                # Find entities to mask
                entities = self.matcher.find_matches(text, language)
                
                # If English, also use spaCy NER
                if language == 'en' and 'en' in self.models:
                    ner_entities = self._detect_with_ner(text)
                    # Combine and deduplicate
                    entities = self._combine_entities(entities, ner_entities)
                
                # Add page number to entities
                for entity in entities:
                    entity['page'] = page_num
                    all_entities.append(entity)
                
                # Mask entities on the page
                self._mask_page_entities(page, entities)
            
            # Group entities by type
            entity_summary = {}
            for entity in all_entities:
                entity_type = entity['type']
                if entity_type not in entity_summary:
                    entity_summary[entity_type] = []
                entity_summary[entity_type].append(entity['text'])
            
            # Save masked PDF
            doc.save(output_path)
            doc.close()
            
            # Generate metadata
            metadata = {
                'filename': os.path.basename(pdf_path),
                'language': language,
                'entities_masked': entity_summary,
                'total_entities_masked': len(all_entities)
            }
            
            return metadata
            
        except Exception as e:
            print(f"Error masking PDF {pdf_path}: {e}")
            return {
                'filename': os.path.basename(pdf_path),
                'error': str(e)
            }
    
    def _detect_with_ner(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect entities using spaCy NER.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected entities
        """
        doc = self.models['en'](text)
        entities = []
        
        for ent in doc.ents:
            # Map spaCy entity types to our entity types
            entity_type = self._map_ner_type(ent.label_)
            if entity_type and entity_type in self.config['entity_types']:
                entities.append({
                    'type': entity_type,
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.8  # Default confidence for NER
                })
        
        return entities
    
    def _map_ner_type(self, spacy_type: str) -> Optional[str]:
        """Map spaCy entity types to our entity types."""
        mapping = {
            'PERSON': 'PERSON',
            'GPE': 'LOCATION',
            'LOC': 'LOCATION',
            'ORG': 'ORGANIZATION',
            'DATE': 'DATE',
            'TIME': 'DATE',
            'MONEY': 'BANK_ACCOUNT',
            'CARDINAL': None,
            'ORDINAL': None,
            'PRODUCT': None
        }
        return mapping.get(spacy_type)
    
    def _combine_entities(self, regex_entities: List[Dict[str, Any]], 
                         ner_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine and deduplicate entities from different sources.
        
        Args:
            regex_entities: Entities from regex patterns
            ner_entities: Entities from NER
            
        Returns:
            Combined list of entities
        """
        # Combine all entities
        all_entities = regex_entities + ner_entities
        
        # Sort by position in text
        all_entities.sort(key=lambda e: (e['start'], -e['end']))
        
        # Remove overlaps (prefer regex matches over NER)
        non_overlapping = []
        for entity in all_entities:
            # Check if this entity overlaps with any already accepted entity
            overlaps = False
            for accepted in non_overlapping:
                if (entity['start'] < accepted['end'] and 
                    entity['end'] > accepted['start']):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(entity)
        
        return non_overlapping
    
    def _mask_page_entities(self, page, entities: List[Dict[str, Any]]):
        """
        Mask entities on a PDF page using redaction annotations.
        
        Args:
            page: fitz.Page object
            entities: List of entities to mask
        """
        if not entities:
            return
        
        # Sort entities to process from end to start (to avoid offset issues)
        sorted_entities = sorted(entities, key=lambda e: (e['start'], -e['end']), reverse=True)
        
        for entity in sorted_entities:
            # Find all instances of this text in the page
            text_instances = page.search_for(entity['text'])
            
            # Create redaction annotations for each instance
            for inst in text_instances:
                # Create slightly larger redaction to ensure complete coverage
                redact_annot = page.add_redact_annot(inst, fill=(0, 0, 0))
                
                # Apply redaction
                page.apply_redactions()
                
                # Track what was redacted
                if entity['type'] not in self.masked_entities:
                    self.masked_entities[entity['type']] = []
                self.masked_entities[entity['type']].append(entity['text'])
    
    def process_pdf_directory(self, input_dir: str, output_dir: str, metadata_dir: str = None) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory.
        
        Args:
            input_dir: Path to directory containing input PDFs
            output_dir: Path to save masked PDFs
            metadata_dir: Path to save masking metadata (defaults to output_dir)
            
        Returns:
            List of dictionaries with masking metadata
        """
        if metadata_dir is None:
            metadata_dir = output_dir
            
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Get all PDF files
        pdf_paths = list(Path(input_dir).glob('**/*.pdf'))
        results = []
        
        for pdf_path in pdf_paths:
            try:
                # Define output paths
                rel_path = pdf_path.relative_to(input_dir)
                output_path = Path(output_dir) / f"{pdf_path.stem}_masked.pdf"
                metadata_path = Path(metadata_dir) / f"{pdf_path.stem}_masking_info.json"
                
                # Mask PDF
                metadata = self.mask_pdf(str(pdf_path), str(output_path))
                
                # Save metadata
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                # Add to results
                results.append(metadata)
                
                print(f"Processed: {pdf_path.name}")
                print(f"Entities masked: {metadata.get('total_entities_masked', 0)}")
                print('-' * 50)
                
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                results.append({
                    'filename': pdf_path.name,
                    'error': str(e)
                })
        
        # Save summary
        summary_path = Path(metadata_dir) / "masking_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_files_processed': len(pdf_paths),
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        return results


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pdf_masker.py input_dir output_dir [metadata_dir]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    metadata_dir = sys.argv[3] if len(sys.argv) > 3 else output_dir
    
    masker = PDFMasker()
    results = masker.process_pdf_directory(input_dir, output_dir, metadata_dir)
    
    print(f"Processed {len(results)} PDF files")
    print(f"Masked PDFs saved to {output_dir}")
    print(f"Masking metadata saved to {metadata_dir}")