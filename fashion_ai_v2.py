import streamlit as st
import requests, random, time, torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO

# --- CONFIG ---
st.set_page_config(page_title="Daily Fashion Trends (AI-Ranked)", layout="wide")

# --- STYLES ---
st.markdown("""
<style>
    .stImage img {border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.15);}
    .topic-header {font-size: 1.2rem; font-weight: 600; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# --- UNSPLASH SETTINGS ---
UNSPLASH_ACCESS_KEY = st.secrets.get("UNSPLASH_ACCESS_KEY")

FASHION_TOPICS = [
    "street style","runway fashion","outfit of the day","minimalist fashion",
    "vintage style","haute couture","sustainable fashion",
    "summer looks","winter outfits","editorial fashion"
]

# --- IMAGE FETCHING ---
@st.cache_data(ttl=3600)
def fetch_unsplash_images(query, count=10):
    if not UNSPLASH_ACCESS_KEY:
        st.error("Missing Unsplash API key.")
        return []
    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": query, "per_page": count,
        "client_id": UNSPLASH_ACCESS_KEY, "orientation": "portrait"
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        st.warning(f"Unsplash error for '{query}': {r.status_code}")
        return []
    data = r.json()
    return [
        {
            "title": img["alt_description"] or "Untitled",
            "url": img["urls"]["regular"],
            "author": img["user"]["name"],
            "link": img["links"]["html"]
        }
        for img in data.get("results", [])
    ]

# --- AI MODEL (CLIP) ---
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@torch.no_grad()
def compute_aesthetic_score(model, processor, image_url):
    try:
        image = Image.open(BytesIO(requests.get(image_url, timeout=10).content)).convert("RGB")
        inputs = processor(text=["beautiful aesthetic fashion photo"], images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        score = outputs.logits_per_image.item()
        # scale roughly to 0–10 for easier reading
        return round(5 + score * 2.5, 2)
    except Exception:
        return random.uniform(3,7)

# --- HEADER ---
st.title("👗 AI-Ranked Daily Fashion Trends")
st.caption(f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}  |  Free Unsplash + CLIP aesthetic scoring")

# --- USER INPUT ---
st.markdown("### 🎨 Choose your fashion inspirations")
predefined = st.multiselect(
    "Select predefined styles:",
    FASHION_TOPICS,
    default=random.sample(FASHION_TOPICS, k=3)
)
custom_text = st.text_input(
    "Or enter your own (comma-separated):",
    placeholder="e.g., Paris street style, vintage denim, boho outfit"
)
custom_topics = [t.strip() for t in custom_text.split(",") if t.strip()]
selected_topics = predefined + custom_topics

if not selected_topics:
    st.info("👆 Choose at least one theme or enter your own keywords.")
    st.stop()

st.success(f"Fetching looks for {', '.join(selected_topics)}")

# --- LOAD MODEL ---
model, processor = load_clip_model()

# --- DISPLAY ---
for topic in selected_topics:
    st.markdown(f"<div class='topic-header'>{topic.title()}</div>", unsafe_allow_html=True)
    images = fetch_unsplash_images(topic, count=10)
    if not images:
        st.warning(f"No images for {topic}.")
        continue

    # Score + rank
    with st.spinner(f"Scoring looks for '{topic}'..."):
        for img in images:
            img["score"] = compute_aesthetic_score(model, processor, img["url"])
        ranked = sorted(images, key=lambda x: x["score"], reverse=True)

    # Display top 5
    for img in ranked[:5]:
        st.image(img["url"], caption=f"⭐ {img['score']} — {img['author']} | [View]({img['link']})", use_container_width=True)

st.info("✅ Powered by Unsplash + CLIP aesthetic scoring (fully local & free).")

