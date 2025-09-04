# app.py
import gradio as gr
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- 1. Configuration & Model Loading ---

# Global variables to hold the models and data
device = "cuda" if torch.cuda.is_available() else "cpu"
rag_components = None
llm_components = None

def initialize_models():
    """
    Initializes and loads all necessary models and data.
    This function runs once when the Gradio app starts.
    """
    global rag_components, llm_components
    
    print("Initializing models... This may take a moment.")
    
    # --- Part A: Initialize RAG Components (The "Brain") ---
    print("Loading corrected knowledge base from JSON Lines file...")
    try:
        texts = []
        embeddings_list = []
        # Read the corrected JSON Lines file
        with open('knowledge_base_corrected.jsonl', 'r') as f:
            for line in f:
                data_obj = json.loads(line)
                texts.append(data_obj['text']) 
                embeddings_list.append(data_obj['embedding'])

        embeddings = np.array(embeddings_list, dtype='float32')

        # Create a FAISS index for fast searching
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        # Load the embedding model (only for encoding user queries at runtime)
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        rag_components = {
            "texts": texts,
            "embedding_model": embedding_model,
            "index": index
        }
        print(f"RAG components loaded successfully. Knowledge base contains {len(texts)} entries.")
    except Exception as e:
        print(f"Error initializing RAG components: {e}")
        return

    # --- Part B: Load the Fine-Tuned LLM (The "Personality") ---
    print("Loading fine-tuned Mistral-7B model...")
    try:
        base_model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
        adapter_path = "./lora_model_fixed"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load the LoRA adapter to apply the fine-tuned personality
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        llm_components = {
            "model": model,
            "tokenizer": tokenizer
        }
        print("Fine-tuned LLM loaded successfully.")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return
        
    print("Initialization complete.")

# --- 2. The Core RAG-Enhanced Generation Logic ---
def get_august_ai_response(user_question, history):
    """
    Generates a response using RAG and the fine-tuned model.
    """
    if not rag_components or not llm_components:
        return "Error: Models are not initialized. Please check the logs."

    embedding_model = rag_components["embedding_model"]
    index = rag_components["index"]
    texts = rag_components["texts"]
    
    # Embed the user's question to find relevant context
    question_embedding = embedding_model.encode(user_question, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    
    # Search the FAISS index for the top 3 most similar text chunks
    distances, indices = index.search(question_embedding, 3)
    
    retrieved_docs = [texts[i] for i in indices[0]]
    context = "\n\n".join(retrieved_docs)

    model = llm_components["model"]
    tokenizer = llm_components["tokenizer"]

    # This prompt template combines the persona, the retrieved facts, and the user's question
    prompt_template = f"""
    You are August AI, a supportive and empathetic gut health coach. Your tone is warm, understanding, and encouraging. Use the following scientifically-grounded context to answer the user's question. If the context doesn't have the answer, say that you don't have enough information on that topic. Do not mention the context in your answer.

    CONTEXT:
    {context}

    USER QUESTION:
    {user_question}

    YOUR EMPATHETIC RESPONSE:
    """
    
    inputs = tokenizer(prompt_template, return_tensors="pt").to(device)
    
    # Generate the response
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the output to only show the final response
    final_response = response_text.split("YOUR EMPATHETIC RESPONSE:")[-1].strip()
    
    return final_response


# --- 3. Initialize Models and Launch the Gradio UI ---

initialize_models()

iface = gr.ChatInterface(
    fn=get_august_ai_response,
    title="August AI - Your Gut Health Coach 🩺",
    description="Ask me anything about gut health. I'm here to provide supportive, science-backed information.",
    examples=[
        ["I've been bloated for three days — what should I do?"],
        ["How does gut health affect sleep?"],
        ["Why do I feel brain fog after eating sugar?"]
    ],
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me a question...", container=False, scale=7),
    theme='soft'
)

if __name__ == "__main__":
    iface.launch(share=True)