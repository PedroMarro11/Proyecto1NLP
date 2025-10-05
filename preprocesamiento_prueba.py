import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import Levenshtein as lev

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Inicializar herramientas de NLTK
stop_words = set(stopwords.words('spanish'))

def preprocesar_texto(corpus: str, vocabulario: list = None) -> list:
    """
    Preprocesa un texto:
    - Convierte a minúsculas
    - elimina stopwords
    - Elimina puntuación
    - reemplaza caracteres especiales
    - Stemming
    - Lematización
    - Tokenización
    - algoritmo de Levenshtein para identificar palabras similares en el vocabulario LOCAL
    Args:
        corpus (str): El texto a preprocesar.
        vocabulario (list): Lista de palabras del vocabulario local (inicialmente vacía).
    Returns:
        list: Lista de tokens preprocesados.
    """
    if vocabulario is None:
        vocabulario = []
    
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Convertir a minúsculas
    corpus = corpus.lower()

    # Reemplazar tildes y caracteres especiales
    corpus = corpus.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    corpus = corpus.replace('ñ', 'n').replace('ü', 'u').replace('ç', 'c')

    # Tokenización
    tokens = word_tokenize(corpus, language='spanish')

    # Eliminar puntuación y stopwords, aplicar stemming y lematización, y usar Levenshtein
    processed_tokens = []
    
    for token in tokens:
        if token not in string.punctuation and token not in stop_words:
            stemmed = stemmer.stem(token)
            lemmatized = lemmatizer.lemmatize(stemmed)
            
            # Buscar si ya existe una palabra similar en el vocabulario local
            encontrada = False
            for palabra in vocabulario:
                if lev.distance(lemmatized, palabra) <= 1:
                    # Si encontramos una palabra similar, usamos esa
                    processed_tokens.append(palabra)
                    encontrada = True
                    break  # IMPORTANTE: salir del bucle una vez encontrada
            
            # Si no encontramos ninguna similar, agregamos la nueva palabra
            if not encontrada:
                processed_tokens.append(lemmatized)
                vocabulario.append(lemmatized)
    
    return processed_tokens