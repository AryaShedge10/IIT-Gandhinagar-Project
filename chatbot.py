import json
import numpy as np
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


with open("Extracted_iitgn_faq.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


for faq in faqs:
    faq["embedding"] = embedding_model.encode(faq["question"]).tolist()


def find_best_faq(user_query):
    query_embedding = embedding_model.encode(user_query).reshape(1, -1)
    faq_embeddings = np.array([faq["embedding"] for faq in faqs])
    

    similarities = cosine_similarity(query_embedding, faq_embeddings)[0]
    
    
    best_match_index = np.argmax(similarities)
    best_match = faqs[best_match_index]
    
    return best_match["question"], best_match["answer"], similarities[best_match_index]


llm = Ollama(model="mistral")  


memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory
)
print("\n🤖 Welcome to the IITGN Library Chatbot (Powered by LangChain & Ollama)! (Type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    
    best_q, best_a, confidence = find_best_faq(user_input)

    if confidence > 0.5:  
        response = best_a
    else:  
        response = conversation.run(user_input)

    print(f"\n🤖 Chatbot: {response}\n")
