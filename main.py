from sentence_transformers import SentenceTransformer
import json

# Load FAQ data
with open("Extracted_iitgn_faq.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)

# Load a pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for FAQ questions
for faq in faqs:
    faq["embedding"] = model.encode(faq["question"]).tolist()

# Save processed data
with open("faq_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(faqs, f, indent=4, ensure_ascii=False)

print("✅ FAQ embeddings saved to 'faq_embeddings.json'")
