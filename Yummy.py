import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pickle
import re
import nltk
import gc
from nltk.stem import WordNetLemmatizer
from diffusers import AutoPipelineForText2Image

# --- PUSLAPIO KONFIGŪRACIJA ---
st.set_page_config(page_title="MasterChef AI", layout="wide", page_icon="🍳")

# --- MĖSŲ SĄRAŠAS (Konfliktų logikai) ---
PROTEINS = {
    'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'veal', 
    'steak', 'bacon', 'ham', 'sausage', 'fish', 'shrimp', 'salmon', 'tuna', 'meat'
}

# --- DIETŲ TAISYKLĖS (Draudžiami produktai) ---
DIETARY_RULES = {
    "Vegetarian": {'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'veal', 'steak', 'bacon', 'ham', 'sausage', 'fish', 'shrimp', 'salmon', 'tuna', 'meat', 'lard'},
    "Vegan": {'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'veal', 'steak', 'bacon', 'ham', 'sausage', 'fish', 'shrimp', 'salmon', 'tuna', 'meat', 'lard', 'egg', 'eggs', 'milk', 'cheese', 'butter', 'cream', 'yogurt', 'honey', 'mayonnaise', 'gelatin'},
    "No Pork (Halal/Kosher)": {'pork', 'bacon', 'ham', 'sausage', 'lard', 'pepperoni', 'prosciutto', 'chorizo'},
    "Gluten-Free": {'flour', 'wheat', 'bread', 'pasta', 'barley', 'rye', 'couscous', 'semolina', 'soy sauce', 'breadcrumbs'},
    "Keto (Low Carb)": {'sugar', 'rice', 'pasta', 'bread', 'flour', 'potato', 'potatoes', 'corn', 'beans', 'banana', 'apple', 'honey', 'syrup'}
}

# --- MODELIO ARCHITEKTŪRA ---
class RecipeNet(nn.Module):
    def __init__(self):
        super(RecipeNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2000, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    def forward(self, x):
        return self.encoder(x)

# --- RESURSŲ KROVIMAS ---
@st.cache_resource
def load_nn_resources():
    print("⏳ Kraunami NN resursai...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        with open('ingredient_binarizer.pkl', 'rb') as f:
            mlb = pickle.load(f)
        df = pd.read_pickle('search_dataset.pkl')
        embeddings = torch.load('search_embeddings.pt', map_location=device)
        model = RecipeNet().to(device)
        model.load_state_dict(torch.load("recipe_model.pth", map_location=device), strict=False)
        model.eval()
        return mlb, df, embeddings, model, device
    except FileNotFoundError:
        return None, None, None, None, None

@st.cache_resource
def load_sdxl():
    print("⏳ Kraunamas SDXL modelis...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
            variant="fp16" if device == "cuda" else None
        )
        pipe.to(device)
        return pipe
    except: return None

mlb, df, db_embeddings, model, device = load_nn_resources()

if df is None:
    st.error("⚠️ Trūksta failų! Paleiskite 'prepare_ml_data.py' ir 'ModelTrain.py'.")
    st.stop()

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

# --- FUNKCIJOS ---

def clean_display_text(raw_text):
    if pd.isna(raw_text): return []
    text = str(raw_text).strip("[]")
    if not text: return []
    if '", "' in text: items = text.split('", "')
    elif "', '" in text: items = text.split("', '")
    else: items = [text]
    return [i.replace('"', '').replace("'", "").strip().replace(r"\u00b0", "°") for i in items]

def process_nlp_input(raw_text):
    # Išvalo ir lemmatizuoja tekstą
    clean_str = re.sub(r'[^a-zA-Z,\s]', '', raw_text).lower()
    items = [x.strip() for x in clean_str.split(',')]
    processed = set()
    for item in items:
        words = item.split()
        lemma = [lemmatizer.lemmatize(w) for w in words]
        if lemma:
            processed.add(" ".join(lemma))
    return list(processed)

def get_recommendations(user_text, negative_text, diet_type, top_n=30):
    # 1. Apdorojame teigiamus ingredientus
    processed_items = process_nlp_input(user_text)
    
    # 2. Apdorojame neigiamus ingredientus (Vengti)
    negative_items = set(process_nlp_input(negative_text))
    
    # 3. Identifikuojame vartotojo mėsas (dėl bonusų sistemos)
    user_proteins = set()
    for item in processed_items:
        for word in item.split():
            if word in PROTEINS: user_proteins.add(word)
    
    # 4. Vektorizavimas ir NN paieška
    user_matrix = mlb.transform([processed_items])
    user_tensor = torch.FloatTensor(user_matrix).to(device)
    
    with torch.no_grad():
        user_vec = model(user_tensor)
        user_norm = user_vec / user_vec.norm(p=2, dim=1, keepdim=True)
    
    # Imame 500 kandidatų filtravimui
    scores = torch.mm(user_norm, db_embeddings.t()).squeeze()
    top_scores, top_indices = torch.topk(scores, 500) 
    
    # --- FILTRAVIMAS IR PERRŪŠIAVIMAS ---
    results = df.iloc[top_indices.cpu().numpy()].copy()
    nn_scores = top_scores.cpu().numpy()
    
    final_scores = []
    keep_indices = [] # Čia saugosime tik tuos, kurie praeina filtrus
    
    # Pasiruošiame dietos "juodąjį sąrašą"
    diet_forbidden = DIETARY_RULES.get(diet_type, set())
    
    for idx, (row_idx, row) in enumerate(results.iterrows()):
        recipe_ing_str = " ".join(row['clean_list'])
        recipe_ing_set = set(row['clean_list'])
        
        # --- 1. GRIEŽTI FILTRAI (Hard Filters) ---
        discard = False
        
        # A. Vartotojo įvesti neigiami produktai
        for neg in negative_items:
            # Tikriname ar neigiamas žodis yra recepte (dalinis atitikimas)
            # Pvz. jei neg='nut', tai išmes 'peanut', 'walnut'
            if neg in recipe_ing_str: 
                discard = True; break
        if discard: continue
            
        # B. Dietos filtrai
        if diet_type != "Viskas":
            for forbidden in diet_forbidden:
                # Tikriname atskirus žodžius
                # split() svarbu, kad 'grape' neužblokuotų 'grapefruit'
                recipe_words = set(recipe_ing_str.split())
                if forbidden in recipe_words:
                    discard = True; break
        if discard: continue
        
        # --- 2. BALŲ PERSKAIČIAVIMAS (Soft Logic) ---
        base_score = nn_scores[idx]
        penalty = 0.0
        
        # Mėsos konfliktų logika
        recipe_proteins = set()
        for ing in row['clean_list']:
            for word in ing.split():
                if word in PROTEINS: recipe_proteins.add(word)

        if user_proteins:
            if recipe_proteins:
                has_requested_meat = not user_proteins.isdisjoint(recipe_proteins)
                if not has_requested_meat:
                    penalty += 0.4 
        
        matches = len(recipe_ing_set.intersection(set(processed_items)))
        bonus = matches * 0.05
        
        final_scores.append(base_score - penalty + bonus)
        keep_indices.append(idx) # Išsaugome indeksą (nuo 0 iki 499), kuris praėjo filtrą

    # Sukuriame naują filtruotą DataFrame
    # keep_indices nurodo eiles originaliame 500 kandidatų sąraše
    filtered_results = results.iloc[keep_indices].copy()
    filtered_results['match_score'] = final_scores
    
    # Trūkstami ingredientai
    user_set = set(processed_items)
    filtered_results['missing_ingredients'] = filtered_results['clean_list'].apply(lambda x: set(x) - user_set)
    filtered_results['missing_count'] = filtered_results['missing_ingredients'].apply(len)
    
    # Galutinis Rūšiavimas
    return filtered_results.sort_values(by='match_score', ascending=False).head(top_n), processed_items

# --- VARTOTOJO SĄSAJA (UI) ---

if 'generated_images' not in st.session_state:
    st.session_state['generated_images'] = {}

st.title("🍳 MasterChef AI Generatorius")
st.markdown("Išmanioji receptų paieška su **Neuroniniu Tinklu**, **Dietų Filtrais** ir **SDXL-Turbo**.")

# --- ŠONINIS MENIU ---
st.sidebar.header("🛒 Produktai")
st.sidebar.caption("Ką turite šaldytuve?")
user_text = st.sidebar.text_area("Teigiami (pvz. chicken, rice):", "chicken, rice, onion, garlic")

st.sidebar.markdown("---")
st.sidebar.header("🚫 Apribojimai")
negative_text = st.sidebar.text_input("Noriu vengti (pvz. nuts, milk):", "")

diet_options = ["Viskas", "Vegetarian", "Vegan", "Gluten-Free", "Keto (Low Carb)", "No Pork (Halal/Kosher)"]
selected_diet = st.sidebar.selectbox("Mitybos tipas:", diet_options)

st.sidebar.divider()
num_results = st.sidebar.slider("Kiek receptų rodyti?", 10, 100, 30, 10)

with st.sidebar:
    st.divider()
    st.caption("AI Modelis: SDXL-Turbo + NeuralNet")
    pipe = load_sdxl()
    if pipe:
        st.success("✅ AI variklis paruoštas")
    else:
        st.warning("⚠️ AI modelis neužsikrovė.")

# Paieškos mygtukas
if st.sidebar.button("🚀 Ieškoti Receptų", type="primary"):
    st.session_state['generated_images'] = {}
    with st.spinner("Filtruojami ir ieškomi receptai..."):
        # Kviečiame atnaujintą funkciją su dietomis
        results, recognized = get_recommendations(user_text, negative_text, selected_diet, top_n=num_results)
        st.session_state['results'] = results
        st.session_state['recognized'] = recognized

# NN Debug Info
if 'recognized' in st.session_state:
    st.sidebar.info(f"🔍 **NN mato:** {', '.join(st.session_state['recognized'])}")

# Rezultatų atvaizdavimas
if 'results' in st.session_state:
    results = st.session_state['results']
    
    if results.empty:
        st.warning(f"Receptų nerasta. Galbūt dieta '{selected_diet}' atmetė visus rezultatus su šiais ingredientais?")
    else:
        st.success(f"Rasta receptų: {len(results)}")
        
        for idx, row in results.iterrows():
            raw_score = row['match_score']
            if raw_score > 1.2: score = 99
            elif raw_score < 0.1: score = 10
            else: score = int(min(raw_score * 100, 99))
            
            st.markdown(f"### 🍽️ {row['title']}")
            st.caption(f"Reitingas: **{score}/100** | Trūksta: **{row['missing_count']}**")
            
            with st.expander("Peržiūrėti detales ir nuotrauką"):
                
                img_col, txt_col = st.columns([1, 2])
                
                # FOTO
                with img_col:
                    recipe_id = idx
                    if recipe_id in st.session_state['generated_images']:
                        st.image(st.session_state['generated_images'][recipe_id], caption="AI Sugeneruota", use_container_width=True)
                    else:
                        if st.button(f"🎨 Generuoti nuotrauką", key=f"btn_{idx}"):
                            if pipe:
                                ing_txt = ", ".join(row['clean_list'][:8])
                                steps_ctx = str(row['directions'])[:200].replace('[', '').replace(']', '').replace("'", "")
                                prompt = (
                                    f"Cinematic shot of delicious {row['title']}. "
                                    f"Main ingredients: {ing_txt}. "
                                    f"Context: {steps_ctx}. "
                                    f"Professional food photography, 4k, restaurant lighting."
                                )
                                with st.spinner("AI piešia..."):
                                    try:
                                        image = pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
                                        st.session_state['generated_images'][recipe_id] = image
                                        torch.cuda.empty_cache()
                                        gc.collect()
                                        st.rerun()
                                    except Exception as e: st.error(str(e))
                            else: st.error("Modelis neužkrautas.")

                # TEKSTAS
                with txt_col:
                    tabs = st.tabs(["🛒 Ingredientai", "📖 Gaminimas", "🔗 Nuoroda"])
                    
                    with tabs[0]:
                        st.write("**Jums trūksta:**")
                        if row['missing_count'] == 0: st.success("Nieko!")
                        else: 
                            for m in row['missing_ingredients']: st.markdown(f"- 🔴 {m}")
                        st.divider()
                        st.write("**Visi ingredientai:**")
                        ings = clean_display_text(row['ingredients'])
                        for i in ings: st.markdown(f"- {i}")
                        
                    with tabs[1]:
                        steps = clean_display_text(row['directions'])
                        for i, s in enumerate(steps, 1): st.write(f"**{i}.** {s}")
                        
                    with tabs[2]:
                        if str(row['link']).startswith("http"):
                            st.markdown(f"🔗 **[Atidaryti originalų šaltinį]({row['link']})**")
                        else: st.info("Nuorodos nėra.")
            st.divider()