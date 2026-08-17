"""
Serbian text preprocessing for ASR.

Handles:
- Lowercasing
- Punctuation removal
- Number to word conversion
- Extra whitespace normalization
- Special character handling
"""

import re
import logging

logger = logging.getLogger(__name__)


# Serbian number to words mapping
ONES = {
    0: "", 1: "jedan", 2: "dva", 3: "tri", 4: "četiri", 5: "pet",
    6: "šest", 7: "sedam", 8: "osam", 9: "devet"
}

TEENS = {
    10: "deset", 11: "jedanaest", 12: "dvanaest", 13: "trinaest",
    14: "četrnaest", 15: "petnaest", 16: "šesnaest",
    17: "sedamnaest", 18: "osamnaest", 19: "devetnaest"
}

TENS = {
    2: "dvadeset", 3: "trideset", 4: "četrdeset", 5: "pedeset",
    6: "šezdeset", 7: "sedamdeset", 8: "osamdeset", 9: "devedeset"
}

SCALES = {
    100: "sto",
    1000: "hiljada",
    1000000: "milion",
    1000000000: "milijard",
    1000000000000: "trilion"
}


def number_to_words(num: int) -> str:
    """
    Convert number to Serbian words.
    
    Examples:
        123 -> "sto dvadeset tri"
        2024 -> "dve hiljade dvadeset četiri"
    """
    if num == 0:
        return "nula"
    
    if num < 0:
        return "minus " + number_to_words(-num)
    
    # Handle numbers under 10
    if num < 10:
        return ONES[num].strip()
    
    # Handle numbers 10-19
    if num < 20:
        return TEENS[num]
    
    # Handle numbers 20-99
    if num < 100:
        tens = num // 10
        ones = num % 10
        result = TENS[tens]
        if ones > 0:
            result += " " + ONES[ones]
        return result.strip()
    
    # Handle numbers 100-999
    if num < 1000:
        hundreds = num // 100
        remainder = num % 100
        result = ONES[hundreds] + " sto"
        if remainder > 0:
            result += " " + number_to_words(remainder)
        return result.strip()
    
    # Handle larger numbers
    for scale in sorted(SCALES.keys(), reverse=True):
        if num >= scale:
            quotient = num // scale
            remainder = num % scale
            result = number_to_words(quotient) + " " + SCALES[scale]
            if remainder > 0:
                result += " " + number_to_words(remainder)
            return result.strip()
    
    return str(num)


class SerbianTextPreprocessor:
    """
    Serbian text preprocessing for ASR.
    """
    
    def __init__(self):
        """Initialize preprocessor."""
        # Characters to remove (punctuation, symbols)
        self.chars_to_remove = re.compile(r'[\.\'"!\-—–\(\)\[\]{}<>;:,?\#\*]')
        # Multiple spaces
        self.multiple_spaces = re.compile(r'\s+')
        # Cyrillic detection
        self.cyrillic_re = re.compile('[\u0400-\u04FF]')
        # simple transliteration map for Serbian
        self._cyr_to_lat = {
            'А':'A','Б':'B','В':'V','Г':'G','Д':'D','Ђ':'Đ','Е':'E','Ж':'Ž','З':'Z','И':'I','Ј':'J','К':'K','Л':'L','Љ':'Lj','М':'M','Н':'N','Њ':'Nj','О':'O','П':'P','Р':'R','С':'S','Т':'T','Ћ':'Ć','У':'U','Ф':'F','Х':'H','Ц':'C','Ч':'Č','Џ':'Dž','Ш':'Š',
            'а':'a','б':'b','в':'v','г':'g','д':'d','ђ':'đ','е':'e','ж':'ž','з':'z','и':'i','ј':'j','к':'k','л':'l','љ':'lj','м':'m','н':'n','њ':'nj','о':'o','п':'p','р':'r','с':'s','т':'t','ћ':'ć','у':'u','ф':'f','х':'h','ц':'c','ч':'č','џ':'dž','ш':'š'
        }
        logger.info("Initialized SerbianTextPreprocessor")
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess Serbian text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Normalize whitespace early
        text = text.strip()
        # Transliterate Cyrillic to Latin for consistent tokenization
        if self.cyrillic_re.search(text):
            text = self._transliterate(text)
        # Lowercase
        text = text.lower()
        
        # Convert numbers to words
        text = self._convert_numbers_to_words(text)
        
        # Remove punctuation and special characters
        text = self.chars_to_remove.sub('', text)
        
        # Normalize whitespace
        text = self.multiple_spaces.sub(' ', text).strip()

        return text

    def _transliterate(self, s: str) -> str:
        return ''.join(self._cyr_to_lat.get(ch, ch) for ch in s)

    def _collapse_repeated_tokens(self, s: str) -> str:
        # collapse more than 3 repeated tokens to a single token
        parts = s.split()
        out = []
        prev = None
        count = 0
        for p in parts:
            if p == prev:
                count += 1
            else:
                prev = p
                count = 1
            if count <= 3:
                out.append(p)
        return ' '.join(out)
    
    def _convert_numbers_to_words(self, text: str) -> str:
        """
        Convert all numbers in text to words.
        
        Args:
            text: Input text
            
        Returns:
            Text with numbers converted to words
        """
        # Find all numbers (including decimals)
        def replace_number(match):
            num_str = match.group(0)
            # Handle decimals
            if ',' in num_str and num_str.count(',') == 1 and num_str.count('.') == 0:
                # European decimal: 1,5 -> "jedan zarez pet"
                parts = num_str.split(',')
                result = number_to_words(int(parts[0])) + " zarez"
                if parts[1]:
                    result += " " + number_to_words(int(parts[1]))
                return result
            elif '.' in num_str and num_str.count('.') == 1:
                # Could be decimal or thousand separator
                # Try as decimal first
                try:
                    if num_str.count(',') == 0:
                        # Likely a decimal number
                        parts = num_str.split('.')
                        if len(parts[1]) <= 2:  # Likely decimal
                            result = number_to_words(int(parts[0])) + " zarez"
                            if parts[1]:
                                result += " " + number_to_words(int(parts[1]))
                            return result
                except:
                    pass
                # Try as integer
                try:
                    num = int(num_str.replace('.', '').replace(',', ''))
                    return number_to_words(num)
                except:
                    return num_str
            else:
                try:
                    num = int(num_str)
                    return number_to_words(num)
                except ValueError:
                    return num_str
        
        # Replace all numbers
        text = re.sub(r'\d+[.,]?\d*', replace_number, text)
        return text

    def postprocess_prediction(self, text: str) -> str:
        """
        Additional post-processing for model predictions to fix common ASR artifacts.

        This runs after initial `preprocess` and aims to correct common decoding errors:
        - normalize number formatting
        - fix some common tokenization artifacts
        - collapse repeated short garbage tokens
        """
        if not text:
            return ""

        # basic normalization
        text = text.strip().lower()

        # transliterate if needed
        if self.cyrillic_re.search(text):
            text = self._transliterate(text)

        # normalize decimals/commas by converting any remaining digits to words
        text = self._convert_numbers_to_words(text)

        # remove long sequences of repeated non-word characters (garbage)
        text = re.sub(r'[^\w\s]{2,}', ' ', text)

        # remove tokens that are mostly non-alpha or extremely long garbage tokens
        parts = text.split()
        clean_parts = []
        for p in parts:
            # drop tokens with >50% non-alpha characters or length > 60
            non_alpha = len(re.findall(r'[^a-zA-Zčćžšđčćšžđnjljdž]', p))
            if len(p) > 60:
                continue
            if len(p) > 0 and (non_alpha / len(p)) > 0.5:
                continue
            clean_parts.append(p)

        text = ' '.join(clean_parts)

        # fix some common concatenation mistakes e.g. 'egodine' -> 'godine'
        text = re.sub(r'\be?godine\b', 'godine', text)

        # collapse repeated filler tokens (more than 3 -> 1)
        parts = text.split()
        out = []
        prev = None
        count = 0
        for p in parts:
            if p == prev:
                count += 1
            else:
                prev = p
                count = 1
            if count <= 3:
                out.append(p)
        text = ' '.join(out)

        # normalize whitespace
        text = self.multiple_spaces.sub(' ', text).strip()
        return text


def preprocess_text(text: str, preprocessor: SerbianTextPreprocessor = None) -> str:
    """
    Preprocess text for ASR.
    
    Args:
        text: Input text
        preprocessor: Optional preprocessor instance
        
    Returns:
        Preprocessed text
    """
    if preprocessor is None:
        preprocessor = SerbianTextPreprocessor()
    
    return preprocessor.preprocess(text)


if __name__ == "__main__":
    # Test preprocessing
    test_cases = [
        "Evo, jedan primer: broj 123 i 2024.godine!",
        "Šta je to? Nevažno!",
        "Brojevi: 1, 2, 3 - 100, 1000, 1,5 milijardi",
        "VELIKI tekst sa - crticama i \"navodnicima\"",
    ]
    
    preprocessor = SerbianTextPreprocessor()
    
    for text in test_cases:
        print(f"Original: {text}")
        print(f"Processed: {preprocessor.preprocess(text)}")
        print()
