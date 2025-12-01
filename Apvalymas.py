import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer

# --- KONFIGŪRACIJA ---
TRAIN_SIZE = 200000   
SEARCH_SIZE = 400000   
TOP_INGREDIENTS = 2000

# --- NLP SETUP ---
print("📦 Ruošiama NLP aplinka...")
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
junk_words = {'cup', 'cups', 'tsp', 'tbsp', 'oz', 'lb', 'g', 'kg', 'ml', 'clove', 'slice', 'chopped', 'fresh', 'large', 'small'}

def smart_clean(text_list_str):
    """
    Išmanus valymas: išlaiko frazes (green pepper), išmeta šiukšles.
    """
    if pd.isna(text_list_str) or text_list_str == "[]": return []
    
    raw_text = str(text_list_str).strip("[]").replace("'", "").replace('"', "")
    items = raw_text.split(',')
    
    clean_items = []
    for item in items:
        # Paliekame tik raides
        item_clean = re.sub(r'[^a-zA-Z\s]', '', item).lower().strip()
        words = item_clean.split()
        
        meaningful_words = []
        for w in words:
            lemma = lemmatizer.lemmatize(w)
            if lemma not in junk_words and lemma not in stop_words and len(lemma) > 1:
                meaningful_words.append(lemma)
        
        # Sujungiame atgal (pvz "green pepper")
        full_ing = " ".join(meaningful_words)
        if full_ing:
            clean_items.append(full_ing)
            
    return clean_items

# --- DUOMENYS ---
print("1. Kraunami duomenys...")
total_rows = TRAIN_SIZE + SEARCH_SIZE
df = pd.read_csv('full_dataset.csv', nrows=total_rows)

print("2. Valomi ingredientai (Smart Clean)...")
# Naudojame NER stulpelį
df['clean_list'] = df['NER'].apply(smart_clean)

# Atskyrimas
print(f"3. Atskyrimas: {TRAIN_SIZE} mokymui, {SEARCH_SIZE} paieškai...")
df_train = df.iloc[:TRAIN_SIZE]
df_search = df.iloc[TRAIN_SIZE:total_rows].reset_index(drop=True)

# --- VEKTORIZAVIMAS ---
print("4. Mokomas Binarizeris...")
# Randame top ingredientus
all_ingredients = [ing for sublist in df_train['clean_list'] for ing in sublist]
from collections import Counter
top_counts = Counter(all_ingredients).most_common(TOP_INGREDIENTS)
top_ings = [x[0] for x in top_counts]

# Binarizeris paverčia žodžius į skaičius
mlb = MultiLabelBinarizer(classes=top_ings)
mlb.fit([]) 

# --- SAUGOJIMAS ---
print("5. Generuojamos matricos...")

# Išsaugome paieškos datasetą su tekstu
df_search.to_pickle('search_dataset.pkl')

# Išsaugome Binarizerį
with open('ingredient_binarizer.pkl', 'wb') as f:
    pickle.dump(mlb, f)

# Generuojame mokymo matricą
train_matrix = mlb.transform(df_train['clean_list'])
np.savez_compressed('train_matrix.npz', matrix=train_matrix)

# Generuojame paieškos matricą
search_matrix = mlb.transform(df_search['clean_list'])
np.savez_compressed('search_matrix.npz', matrix=search_matrix)

print("✅ Duomenys paruošti! Dabar leiskite 'train_model.py'")