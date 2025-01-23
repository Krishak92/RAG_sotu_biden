import re
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def normalize_text(text):
    """
    Delete specials characters, mistakes correction, cleaning text, delete stop words

    Args:
        text (str): Raw text.

    Returns:
        str: Normalized text.
    """

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = nltk.WordNetLemmatizer()
    corrected_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    #Normalized text reconstructed
    normalized_text = ' '.join(corrected_tokens)

    return normalized_text

