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
        'BANK_ACCOUNT', 'SOCIAL_ID', 'URL', 'DISTRICT', 'SOCIAL_MEDIA'
    ],
    
    # Default masking strategy
    'masking_strategy': 'redact',  # 'redact' for PDF masking
    
    # Confidence thresholds
    'confidence_threshold': {
        'en': 0.7,
        'mn': 0.6  # Lower threshold for Mongolian due to limited model accuracy
    }
}

# Terms that should NEVER be masked (regardless of context)
NEVER_MASK = [
    # Education levels
    "Бакалавр", "Магистр", "Доктор", "Ph.D", "Bachelor", "Master", "MBA", 
    
    # Countries and major cities when used as general locations
    "Taiwan", "Тайвань", "China", "Хятад", "USA", "АНУ", 
    "Japan", "Япон", "Korea", "Солонгос", "England", "Англи",
    
    # Common education terms
    "сургууль", "дээд", "их", "коллеж", "university", "college", "school", "institute",
    
    # Common verbs and prepositions
    "working", "studying", "lived", "ажилласан", "амьдарсан", "суралцсан",
    
    # Time periods
    "year", "month", "жил", "сар", "өдөр", "day", "week", "долоо хоног"
]

# Educational institutions that should not be masked - CASE INSENSITIVE MATCHING
EDUCATIONAL_INSTITUTIONS = [
    "Монгол Улсын Их Сургууль",
    "Монгол улсын их сургууль",
    "МУИС",
    "Монгол улсын боловсролын их сургууль",
    "Монгол улсын шинжлэх ухаан технологийн их сургууль",
    "ШУТИС",
    "Монгол улсын анагаахын шинжлэх ухааны их сургууль",
    "АШУИС",
    "Улаанбаатар их сургууль",
    "Хүмүүнлэгийн ухааны их сургууль",
    "Худалдааны их сургууль",
    "Санхүү эдийн засгийн их сургууль",
    "СЭЗДС",
    "Цэргийн академи",
    "Хууль сахиулахын их сургууль"
]

# Companies and organizations that should not be masked
ORGANIZATIONS = [
    "Монгол улсын засгийн газар",
    "Монгол улсын ерөнхийлөгчийн тамгын газар",
    "Монгол улсын их хурал",
    "Улсын бүртгэлийн ерөнхий газар",
    "Гадаадын иргэн харьяатын газар",
    "Монгол улсын гадаад харилцааны яам",
    "Монгол улсын батлан хамгаалахын яам",
    "Монгол банк",
    "Худалдаа хөгжлийн банк",
    "Голомт банк",
    "Хаан банк",
    "Төрийн банк",
    "Монгол шуудан",
    "Монголын үндэсний олон нийтийн радио телевиз"
]

# Mongolian districts to be masked
DISTRICTS = {
    # Full names
    "Баянзүрх дүүрэг": "БЗД",
    "Сонгинохайрхан дүүрэг": "СХД",
    "Баянгол дүүрэг": "БГД",
    "Хан-Уул дүүрэг": "ХУД",
    "Сүхбаатар дүүрэг": "СБД",
    "Чингэлтэй дүүрэг": "ЧД",
    "Багануур дүүрэг": "БНД",
    "Багахангай дүүрэг": "БХД",
    "Налайх дүүрэг": "НД",
    
    # Short names
    "Баянзүрх": "БЗД",
    "Сонгинохайрхан": "СХД",
    "Баянгол": "БГД",
    "Хан-Уул": "ХУД",
    "Сүхбаатар": "СБД",
    "Чингэлтэй": "ЧД",
    "Багануур": "БНД",
    "Багахангай": "БХД",
    "Налайх": "НД",
    
    # Abbreviations
    "БЗД": "БЗД",
    "СХД": "СХД",
    "БГД": "БГД",
    "ХУД": "ХУД",
    "СБД": "СБД",
    "ЧД": "ЧД",
    "БНД": "БНД",
    "БХД": "БХД",
    "НД": "НД"
}

# English regex patterns
EN_PATTERNS = {
    'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'PHONE': r'\b(?:\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b',
    'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
    'CREDIT_CARD': r'\b(?:\d[ -]*?){13,16}\b',
    'IP_ADDRESS': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'URL': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
    'SOCIAL_MEDIA': [
        r'(?:fb|facebook)://[\w.-]+',
        r'(?:linkedin|in)://[\w.-]+',
        r'(?:instagram|ig|insta)://[\w.-]+',
        r'(?:twitter|tw|x)://[\w.-]+',
        r'(?:tiktok|tt)://[\w.-]+',
        r'(?:github|gh)://[\w.-]+',
    ],
    'ADDRESS': r'\b\d+\s+[A-Za-z\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr)\b',
    'PERSON': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',  # Improved English name pattern
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
    
    # URLs and social media profiles
    'URL': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
    'SOCIAL_MEDIA': [
        r'(?:fb|facebook)://[\w.-]+',
        r'(?:linkedin|in)://[\w.-]+',
        r'(?:instagram|ig|insta)://[\w.-]+',
        r'(?:twitter|tw|x)://[\w.-]+',
        r'(?:tiktok|tt)://[\w.-]+',
        r'(?:github|gh)://[\w.-]+',
    ],
    
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
    
    # District pattern - specific for Mongolian districts
    'DISTRICT': [
        r'\b(?:Баянзүрх|Сонгинохайрхан|Баянгол|Хан-Уул|Сүхбаатар|Чингэлтэй|Багануур|Багахангай|Налайх)(?:\s+дүүрэг)?\b',
        r'\b(?:БЗД|СХД|БГД|ХУД|СБД|ЧД|БНД|БХД|НД)\b'
    ],
    
    # Mongolian personal name patterns - improved to catch more names
    'PERSON': [
        r'\b[А-ЯӨҮ][а-яөү]+\s+[А-ЯӨҮ][а-яөү]+\b',  # Standard Mongolian names 
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # English-style names for Mongolians
        r'\b[А-ЯӨҮ][а-яөү]+\s+[A-Z][a-z]+\b',  # Mixed format
        r'\b[A-Z][a-z]+\s+[А-ЯӨҮ][а-яөү]+\b',  # Mixed format (reversed)
    ],
    
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


def is_in_exempt_list(text: str, exempt_lists: List[List[str]]) -> bool:
    """
    Check if text is in any of the exempt lists.
    
    Args:
        text: Text to check
        exempt_lists: List of lists containing exempt terms
        
    Returns:
        True if text is in any exempt list, False otherwise
    """
    text_lower = text.lower()
    
    # Check if text contains anything from NEVER_MASK list
    for term in NEVER_MASK:
        if term.lower() in text_lower:
            return True
    
    # Check broader institutional lists
    for exempt_list in exempt_lists:
        for term in exempt_list:
            term_lower = term.lower()
            # Full match
            if text_lower == term_lower:
                return True
            # Substring match for educational institutions
            if exempt_list == EDUCATIONAL_INSTITUTIONS and term_lower in text_lower:
                return True
    
    return False


def is_in_section(text_before: str, section_markers: List[str]) -> bool:
    """
    Determine if the text appears to be in a specific section of a CV.
    
    Args:
        text_before: Text preceding the entity
        section_markers: List of section marker terms
        
    Returns:
        True if in the specified section, False otherwise
    """
    # Look for section markers in the preceding text (within a reasonable distance)
    text_before_lower = text_before[-150:].lower()
    
    for marker in section_markers:
        if marker.lower() in text_before_lower:
            # Check if there's no other section marker after this one
            marker_pos = text_before_lower.rfind(marker.lower())
            text_after_marker = text_before_lower[marker_pos:]
            
            # If no other section marker is found after this one, we're in this section
            other_markers = [m for m in section_markers if m.lower() != marker.lower()]
            if not any(m.lower() in text_after_marker for m in other_markers):
                return True
    
    return False


def detect_cv_section(text_before: str) -> str:
    """
    Detect which section of a CV/resume the text is located in.
    
    Args:
        text_before: Text preceding the entity
        
    Returns:
        Section name or "unknown"
    """
    sections = {
        "personal_info": [
            "personal information", "contact information", "contact details", 
            "хувийн мэдээлэл", "холбоо барих", "хувийн", "personal", "profile"
        ],
        "education": [
            "education", "academic", "qualification", "degree", 
            "боловсрол", "сургууль", "сургалт", "зэрэг", "мэргэжил"
        ],
        "experience": [
            "experience", "work history", "employment", "career", 
            "ажлын туршлага", "туршлага", "ажил", "мэргэжлийн"
        ],
        "skills": [
            "skills", "abilities", "ур чадвар", "чадвар", "ур чадвар", "технологи"
        ],
        "contact": [
            "contact me", "contact", "холбоо барих", "холбогдох", "email", "phone"
        ]
    }
    
    for section, markers in sections.items():
        if is_in_section(text_before, markers):
            return section
    
    return "unknown"


class ContextAwareEntityMatcher:
    """Class for detecting entities using regex patterns with context awareness."""
    
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
        
    def find_matches(self, text: str, language: str, full_document: str = None) -> List[Dict[str, Any]]:
        """
        Find matches in text using appropriate patterns for the language.
        
        Args:
            text: Text to search in
            language: Language code ('en' or 'mn')
            full_document: Optional full document text for better context
            
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
                    matched_entities = self._process_pattern(text, p, entity_type)
                    filtered_entities = self._filter_with_context(text, matched_entities, entity_type, full_document)
                    entities.extend(filtered_entities)
            else:
                matched_entities = self._process_pattern(text, pattern, entity_type)
                filtered_entities = self._filter_with_context(text, matched_entities, entity_type, full_document)
                entities.extend(filtered_entities)
                
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
    
    def _filter_with_context(self, text: str, entities: List[Dict[str, Any]], 
                           entity_type: str, full_document: str = None) -> List[Dict[str, Any]]:
        """
        Filter entities based on context rules.
        
        Args:
            text: Current text segment
            entities: List of detected entities
            entity_type: Type of entity
            full_document: Optional full document text for better context
            
        Returns:
            Filtered list of entities
        """
        filtered_entities = []
        doc_text = full_document if full_document else text
        
        for entity in entities:
            entity_text = entity['text']
            start = entity['start']
            end = entity['end']
            
            # Skip empty or very short entities
            if len(entity_text.strip()) <= 1:
                continue
                
            # Skip entities that match terms that should never be masked
            if any(never_mask.lower() in entity_text.lower() for never_mask in NEVER_MASK):
                if len(entity_text.split()) <= 3:  # Only skip if it's a short match
                    continue
            
            # Skip exempt educational institutions and organizations
            if is_in_exempt_list(entity_text, [EDUCATIONAL_INSTITUTIONS, ORGANIZATIONS]):
                continue
            
            # Get context (text before and after the entity)
            context_before = text[max(0, start-150):start].lower()
            context_after = text[end:min(len(text), end+150)].lower()
            
            # Detect which section of the CV we're in
            section = detect_cv_section(context_before)
            
            # Special handling for entity types based on CV section
            if entity_type == 'LOCATION':
                # Don't mask locations in education or experience sections unless they appear
                # to be part of an address
                if section in ['education', 'experience']:
                    address_indicators = ["address", "live", "lived", "хаяг", "амьдардаг", "амьдарсан"]
                    if not any(ind in context_before[-50:] or ind in context_after[:50] for ind in address_indicators):
                        continue
            
            elif entity_type == 'PERSON':
                # Always mask person names in personal info section
                if section == 'personal_info':
                    filtered_entities.append(entity)
                    continue
                
                # In other sections, use additional context cues
                person_indicators = ["name", "нэр", "овог", "өөрийн", "миний", "my", "би", "I am", "born"]
                references = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "ноён", "хатагтай"]
                
                # Check for personal context indicators
                personal_context = any(ind in context_before[-50:] or ind in context_after[:50] for ind in person_indicators)
                is_reference = any(ref in context_before[-5:] for ref in references)
                
                # If seems to be a personal reference, mask it
                if personal_context or is_reference:
                    filtered_entities.append(entity)
                    continue
                    
                # If no specific personal context but name format detected, still mask it
                # This helps catch names that appear without obvious context
                if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', entity_text) or \
                   re.search(r'\b[А-ЯӨҮ][а-яөү]+\s+[А-ЯӨҮ][а-яөү]+\b', entity_text):
                    filtered_entities.append(entity)
                    continue
            
            elif entity_type == 'ADDRESS' or entity_type == 'DISTRICT':
                # Always mask in personal info or contact sections
                if section in ['personal_info', 'contact']:
                    filtered_entities.append(entity)
                    continue
                
                # In other sections, look for address indicators
                address_indicators = ["хаяг", "амьдардаг", "хороо", "байр", "тоот", "орц", "гудамж", 
                                     "address", "live", "lived", "residence", "home", "гэр"]
                if any(ind in context_before[-100:] or ind in context_after[:100] for ind in address_indicators):
                    filtered_entities.append(entity)
                    continue
            
            elif entity_type in ['EMAIL', 'PHONE', 'URL', 'SOCIAL_MEDIA']:
                # Always mask contact information
                filtered_entities.append(entity)
                continue
            
            # Default handling for other entity types
            filtered_entities.append(entity)
        
        return filtered_entities


class PDFMasker:
    """
    Context-aware PII masking tool for PDF files that supports both English and Mongolian text.
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
        self.matcher = ContextAwareEntityMatcher(EN_PATTERNS, MN_PATTERNS)
        
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
        language = "unknown"
        
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Extract full text from PDF for context analysis
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            
            # Detect language
            language = detect_language(full_text)
            
            # Process each page with context awareness
            for page_num, page in enumerate(doc):
                # Get page text
                text = page.get_text()
                
                # Find entities to mask using the full document context
                entities = self.matcher.find_matches(text, language, full_text)
                
                # If English, also use spaCy NER
                if language == 'en' and 'en' in self.models:
                    ner_entities = self._detect_with_ner(text, full_text)
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
                'language': language,
                'error': str(e)
            }
    
    def _detect_with_ner(self, text: str, full_text: str = None) -> List[Dict[str, Any]]:
        """
        Detect entities using spaCy NER with context awareness.
        
        Args:
            text: Text to analyze
            full_text: Optional full document text for better context
            
        Returns:
            List of detected entities
        """
        doc = self.models['en'](text)
        entities = []
        
        for ent in doc.ents:
            # Map spaCy entity types to our entity types
            entity_type = self._map_ner_type(ent.label_)
            if entity_type and entity_type in self.config['entity_types']:
                # Skip exempt terms
                if is_in_exempt_list(ent.text, [EDUCATIONAL_INSTITUTIONS, ORGANIZATIONS]):
                    continue
                
                # Skip terms from the NEVER_MASK list
                if any(never_mask.lower() in ent.text.lower() for never_mask in NEVER_MASK):
                    if len(ent.text.split()) <= 3:  # Only skip if it's a short match
                        continue
                
                # Get context around entity
                start = ent.start_char
                end = ent.end_char
                context_before = text[max(0, start-150):start]
                
                # Detect CV section
                section = detect_cv_section(context_before)
                
                # Skip location entities in education/experience sections unless part of address
                if entity_type == 'LOCATION' and section in ['education', 'experience']:
                    # Look for address indicators in context
                    context_after = text[end:min(len(text), end+150)]
                    address_indicators = ["address", "live", "lived", "хаяг", "амьдардаг"]
                    if not any(ind in context_before[-50:].lower() or ind in context_after[:50].lower() 
                               for ind in address_indicators):
                        continue
                
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
                
                # Add redaction label if configured
                if self.config.get('show_redaction_labels', False):
                    entity_type = entity['type']
                    redact_annot.set_info(title=f"[{entity_type}]", content=f"Redacted {entity_type}")
                
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


class PDFLayoutMasker(PDFMasker):
    """
    Enhanced PDF Masker that uses document layout information to make better masking decisions.
    """
    
    def mask_pdf(self, pdf_path: str, output_path: str) -> Dict[str, Any]:
        """
        Mask PII in a PDF file using layout information and save the masked PDF.
        
        Args:
            pdf_path: Path to the input PDF
            output_path: Path to save the masked PDF
            
        Returns:
            Dictionary with masking metadata
        """
        # Reset tracking for each PDF
        self.masked_entities = {}
        all_entities = []
        language = "unknown"
        
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Extract full text from PDF for context analysis
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            
            # Detect language
            language = detect_language(full_text)
            
            # Process each page with layout analysis
            for page_num, page in enumerate(doc):
                # Extract text with layout information
                blocks = page.get_text("dict")["blocks"]
                
                # Process each text block separately to maintain layout context
                for block_num, block in enumerate(blocks):
                    # Skip non-text blocks
                    if block["type"] != 0:  # 0 = text block
                        continue
                    
                    # Process each line in the block
                    for line_num, line in enumerate(block["lines"]):
                        line_text = ""
                        
                        # Get text for this line
                        for span in line["spans"]:
                            line_text += span["text"]
                        
                        # Only process non-empty lines
                        if not line_text.strip():
                            continue
                        
                        # Use the font size to help determine context
                        # Larger fonts are likely headers/section titles
                        is_likely_header = False
                        avg_font_size = 0
                        font_count = 0
                        
                        for span in line["spans"]:
                            avg_font_size += span["size"]
                            font_count += 1
                        
                        if font_count > 0:
                            avg_font_size /= font_count
                            
                            # Headers typically have larger fonts
                            if avg_font_size > 12:  # Threshold for header detection
                                is_likely_header = True
                        
                        # Create section context based on layout
                        section_context = {
                            "is_header": is_likely_header,
                            "block_num": block_num,
                            "line_num": line_num,
                            "font_size": avg_font_size,
                            "text_before": full_text[:full_text.find(line_text)] if line_text in full_text else ""
                        }
                        
                        # Find entities with layout context
                        entities = self._detect_entities_with_layout(line_text, language, section_context, full_text)
                        
                        # Adjust entity positions to be relative to the page
                        for entity in entities:
                            # Store layout information
                            entity['block'] = block_num
                            entity['line'] = line_num
                            entity['page'] = page_num
                            all_entities.append(entity)
                        
                        # Mask entities in this line
                        if entities:
                            self._mask_layout_entities(page, entities, line)
            
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
                'language': language,
                'error': str(e)
            }
    
    def _detect_entities_with_layout(self, text: str, language: str, 
                                   section_context: Dict[str, Any], full_text: str) -> List[Dict[str, Any]]:
        """
        Detect entities with layout context awareness.
        
        Args:
            text: Text to analyze
            language: Language code
            section_context: Layout context information
            full_text: Complete document text
            
        Returns:
            List of detected entities
        """
        # Skip header text
        if section_context.get("is_header", False):
            # Check if this header looks like a section title
            section_titles = [
                "education", "experience", "skills", "contact", "personal", "profile",
                "боловсрол", "туршлага", "ур чадвар", "холбоо барих", "хувийн", "профайл"
            ]
            
            # Skip masking if this is a section title
            if any(title.lower() in text.lower() for title in section_titles):
                return []
        
        # Use matcher to find entities
        entities = self.matcher.find_matches(text, language, full_text)
        
        # If English, also use NER
        if language == 'en' and 'en' in self.models:
            ner_entities = self._detect_with_ner(text, full_text)
            entities = self._combine_entities(entities, ner_entities)
        
        # Apply layout-specific filtering
        filtered_entities = []
        for entity in entities:
            # Check if the entity should be masked based on layout context
            if self._should_mask_in_layout_context(entity, section_context, text):
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def _should_mask_in_layout_context(self, entity: Dict[str, Any], 
                                     section_context: Dict[str, Any], text: str) -> bool:
        """
        Determine if an entity should be masked based on layout context.
        
        Args:
            entity: Entity information
            section_context: Layout context
            text: Current text segment
            
        Returns:
            True if entity should be masked, False otherwise
        """
        entity_type = entity['type']
        entity_text = entity['text']
        
        # Skip entities in NEVER_MASK list
        if any(never_mask.lower() in entity_text.lower() for never_mask in NEVER_MASK):
            if len(entity_text.split()) <= 3:  # Only skip if it's a short match
                return False
        
        # Skip exempt educational institutions and organizations
        if is_in_exempt_list(entity_text, [EDUCATIONAL_INSTITUTIONS, ORGANIZATIONS]):
            return False
        
        # Determine section based on text before
        section = detect_cv_section(section_context.get("text_before", ""))
        
        # Special rules for personal names
        if entity_type == 'PERSON':
            # Always mask in the first section/block (likely personal info)
            if section_context.get("block_num", 999) <= 1 or section == "personal_info":
                return True
                
            # Check for name formats we should definitely mask
            if any(re.search(pattern, entity_text) for pattern in [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # English name format
                r'\b[А-ЯӨҮ][а-яөү]+\s+[А-ЯӨҮ][а-яөү]+\b',  # Mongolian name format
                r'\b[A-Z][a-z]+\s+[А-ЯӨҮ][а-яөү]+\b',  # Mixed format
                r'\b[А-ЯӨҮ][а-яөү]+\s+[A-Z][a-z]+\b',  # Mixed format (reversed)
            ]):
                return True
        
        # Always mask contact information regardless of layout
        if entity_type in ['EMAIL', 'PHONE', 'SOCIAL_MEDIA']:
            return True
            
        # Always mask social media profiles
        if entity_type == 'SOCIAL_MEDIA' or 'fb://' in entity_text:
            return True
            
        # Special rules for locations and addresses
        if entity_type in ['LOCATION', 'ADDRESS', 'DISTRICT']:
            # Don't mask in education section unless it looks like personal address
            if section == 'education':
                education_words = ["university", "college", "school", "сургууль", "дээд", "их"]
                # Skip if it appears with education words
                if any(edu in text.lower() for edu in education_words):
                    return False
                    
            # Don't mask locations in work experience unless clearly personal
            if section == 'experience':
                # Check for personal address indicators
                address_indicators = ["хаяг", "амьдардаг", "living", "address", "home"]
                if not any(ind in text.lower() for ind in address_indicators):
                    return False
        
        # Default to masking
        return True
    
    def _mask_layout_entities(self, page, entities: List[Dict[str, Any]], line: Dict[str, Any]):
        """
        Mask entities on a PDF page using layout information.
        
        Args:
            page: fitz.Page object
            entities: List of entities to mask
            line: Line information from layout analysis
        """
        if not entities:
            return
        
        for entity in entities:
            entity_text = entity['text']
            
            # Search for the entity text in this specific line's area
            # This is more precise than page.search_for() which searches the whole page
            for span in line["spans"]:
                if entity_text in span["text"]:
                    # Calculate rectangle for the entity within the span
                    text_pos = span["text"].find(entity_text)
                    if text_pos >= 0:
                        # Calculate relative position and size
                        char_width = span["size"] * 0.6  # Approximate character width
                        start_pos = text_pos * char_width
                        entity_width = len(entity_text) * char_width
                        
                        # Create rectangle for this entity
                        x0 = span["bbox"][0] + start_pos
                        y0 = span["bbox"][1]
                        x1 = x0 + entity_width
                        y1 = span["bbox"][3]
                        rect = fitz.Rect(x0, y0, x1, y1)
                        
                        # Apply redaction
                        redact_annot = page.add_redact_annot(rect, fill=(0, 0, 0))
                        page.apply_redactions()
                        
                        # Track what was redacted
                        if entity['type'] not in self.masked_entities:
                            self.masked_entities[entity['type']] = []
                        self.masked_entities[entity['type']].append(entity_text)


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pdf_masker.py input_dir output_dir [metadata_dir] [--use-layout]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    metadata_dir = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else output_dir
    
    # Check if layout mode is requested
    use_layout = any("--use-layout" in arg for arg in sys.argv)
    
    if use_layout:
        print("Using layout-aware PDF masking...")
        masker = PDFLayoutMasker()
    else:
        print("Using standard PDF masking...")
        masker = PDFMasker()
    
    results = masker.process_pdf_directory(input_dir, output_dir, metadata_dir)
    
    print(f"Processed {len(results)} PDF files")
    print(f"Masked PDFs saved to {output_dir}")
    print(f"Masking metadata saved to {metadata_dir}")