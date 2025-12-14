from sentence_transformers import SentenceTransformer


def embedd_text(text: str, model_name='all-MiniLM-L6-v2') :
    model = SentenceTransformer(model_name)
    embedding = model.encode(text, convert_to_tensor=True)
    
    return embedding

def embedd_context(texts: list[str], model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    return [
        {
            'text': text, 
            'embedding': emb
        }
        for text, emb in zip(texts, embeddings)
    ]
