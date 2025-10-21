from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import json
from tqdm import tqdm
import torch
import faiss
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "model/multilingual-e5-large"
LANG = "en"

# Define the request model for FastAPI
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# Define the response model for FastAPI
class QueryResponse(BaseModel):
    results: List[Dict]

# ICL retriever
class ICLRetriever:
    def __init__(self, reports, summaries, model_name, batch_size=128, nlist=20, nprobe=5):
        self.reports = reports
        self.summaries = summaries
        self.model_name = model_name
        self.batch_size = batch_size  # Batch size for processing
        self.nlist = nlist  # Number of cells (clusters)
        self.nprobe = nprobe  # Number of cells to search

        # Determine the device to use (GPU if available, otherwise CPU)
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        print(f'Device: {self.device}')

        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

        # Ensure all reports are strings
        self.reports = [str(report) for report in self.reports]

        # Embed and normalize the reports in batches
        self.embeddings = self.normalize_embeddings(self.batch_embed_reports(self.reports).numpy())

        # Build the FAISS index with inner product metric
        self.index = self.build_faiss_index()

    def embed_reports(self, reports):
        # Tokenize and embed reports
        inputs = self.tokenizer(reports, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling
        return embeddings

    def batch_embed_reports(self, reports):
        # Process reports in batches
        all_embeddings = []
        print('Encoding few shot data...')
        for i in range(0, len(reports), self.batch_size):
            batch_reports = reports[i:i + self.batch_size]
            batch_embeddings = self.embed_reports(batch_reports)
            all_embeddings.append(batch_embeddings)
        return torch.cat(all_embeddings, dim=0).cpu()  # Ensure embeddings are on CPU

    def normalize_embeddings(self, embeddings):
        # Normalize the embeddings to unit length (L2 norm)
        faiss.normalize_L2(embeddings)
        return embeddings

    def build_faiss_index(self):
        d = self.embeddings.shape[1]  # Dimension of embeddings
        quantizer = faiss.IndexFlatIP(d)  # Inner Product quantizer
        index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)  # Index with IVF and inner product

        index.train(self.embeddings)  # Train the index with the embeddings
        index.add(self.embeddings)  # Add the embeddings to the index
        index.nprobe = self.nprobe  # Set how many clusters to search
        return index

    def retrieve(self, query, top_k=3):
        # Embed and normalize the input query
        query_embedding = self.normalize_embeddings(self.embed_reports([query]).cpu().numpy())

        # Perform search in the FAISS index
        top_k_scores, top_k_indices = self.index.search(query_embedding, top_k)
        top_k_scores = top_k_scores.flatten()  # Flatten the scores for easier access
        top_k_indices = top_k_indices.flatten()  # Flatten the indices

        # Retrieve the top-k most similar report IDs
        similar_reports = []
        for i, idx in enumerate(top_k_indices):
            similar_reports.append({
                "report": self.reports[idx],
                "summary": self.summaries[idx],
                "similarity_score": float(top_k_scores[i])
            })

        return similar_reports

# Initialize the FastAPI app
app = FastAPI()

class RetrieverComponent:
    _instance = None
    retriever = None

    def __new__(cls, reports_file: str):
        if cls._instance is None:
            cls._instance = super(RetrieverComponent, cls).__new__(cls)
            cls._instance.load_components(reports_file)
        return cls._instance

    def load_components(self, reports_file):
        print('Loading retriever data...')
        with open(reports_file, 'r') as f:
            pmc_data = json.load(f)

        reports = []
        summaries = []
        for d in tqdm(pmc_data):
            reports.append(d.get('report'))
            summaries.append(json.dumps(d.get('summary'), ensure_ascii=False, indent=4))

        print('Done!')
        self.retriever = ICLRetriever(reports, summaries, model_name=MODEL_NAME)

# Initialize the component
retriever_component = RetrieverComponent(f'/home/psig/elmtex_prabin/data/{LANG}/train.json')

# Define the retrieval endpoint
@app.post("/retrieve", response_model=QueryResponse)
def retrieve_reports(request: QueryRequest):
    try:
        query = request.query
        top_k = request.top_k
        results = retriever_component.retriever.retrieve(query, top_k)
        print(results)
        return QueryResponse(results=results)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI server with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("icl_retriever_app:app", host="0.0.0.0", port=8181)
