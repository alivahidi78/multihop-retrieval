import bz2
import os
import ast
import faiss
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CHECKPOINT_EVERY_N_VECTORS = 200000
BATCH_SIZE = 512
MODEL_BATCH_SIZE = 256

def create_flat_embeddings(sentence_transformer, bz2_dir, output_dir, title_key="title", text_key="text", embedding_dim = 384, device="cuda", include_title=True, url_after=None):
    model = SentenceTransformer(sentence_transformer, device=device)
    lookup = []
    embeddings = []
    bz2_files = _find_bz2_files(bz2_dir)
    print(f"Found {len(bz2_files)} .bz2 files:")
    total_vectors = 0
    chunk_id = 0
    index = faiss.IndexFlatL2(embedding_dim)

    lookup = []

    text_batch = []
    meta_batch = []

    for x, file in tqdm(enumerate(bz2_files), desc="Streaming and indexing"):
        with bz2.open(file, 'rt') as f:
            for line_number, line in enumerate(f):
                data = ast.literal_eval(line.strip())
                title = data.get(title_key)
                text = data.get(text_key)
                if include_title:
                    text.insert(0, title)    
                if not text:
                    print("missing text")
                    continue
                for sentence_num, sentence in enumerate(text):
                    if not isinstance(sentence, str) or not sentence.strip():
                        continue 
                    sentence = sentence.encode("utf-8", errors="replace").decode("utf-8")
                    text_batch.append(sentence)
                    url = file
                    if url_after:
                        url = file.split(url_after)[-1]
                    meta_batch.append((url, line_number, sentence_num))
                    # When batch is full, process
                    if len(text_batch) == BATCH_SIZE:
                        try:
                            vectors = model.encode(text_batch, show_progress_bar=False, batch_size=MODEL_BATCH_SIZE)
                            vectors = np.array(vectors).astype("float32")
                            index.add(vectors)
                            lookup.extend(meta_batch)
                            total_vectors += len(vectors)
                        except Exception as e:
                            print(f"batch dropped: {meta_batch}")
                        text_batch = []
                        meta_batch = []
        
        if total_vectors >= CHECKPOINT_EVERY_N_VECTORS:
            # Save index and lookup
            faiss.write_index(index, os.path.join(output_dir, f"index_chunk_{chunk_id}.faiss"))
            with open(os.path.join(output_dir, f"lookup_chunk_{chunk_id}.json"), "w") as f_out:
                json.dump(lookup, f_out)
            print(f"Saved checkpoint {chunk_id} with {total_vectors} vectors at file {x}")
            # Reset for next chunk
            index = faiss.IndexFlatL2(embedding_dim)
            lookup = []
            total_vectors = 0
            chunk_id += 1
                        
    # Process any leftover texts
    if text_batch:
        vectors = model.encode(text_batch, batch_size=64, device="cuda")
        vectors = np.array(vectors).astype("float32")
        index.add(vectors)
        lookup.extend(meta_batch)
        total_vectors += len(vectors)

    if total_vectors > 0:
        faiss.write_index(index, os.path.join(output_dir, f"index_chunk_{chunk_id}.faiss"))
        with open(os.path.join(output_dir, f"lookup_chunk_{chunk_id}.json"), "w") as f_out:
            json.dump(lookup, f_out)
        print(f"Final checkpoint {chunk_id} saved with {total_vectors} vectors")
 
def create_ivf_embeddings(flat_index, output_path, nlist=3000):
    d = flat_index.d           # dimension of vectors
    nb = flat_index.ntotal
    quantizer = faiss.IndexFlatL2(d)  
    ivf_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    vectors = flat_index.reconstruct_n(0, nb)  
    ivf_index.train(vectors)
    ivf_index.add(vectors)
    faiss.write_index(ivf_index, output_path)
            
def load_split_embeddings(index_chunk_dir, lookup_chunk_dir, use_tqdm=True):
    index_files = sorted(f for f in os.listdir(index_chunk_dir) if f.endswith(".faiss"))
    lookup_files = sorted(f for f in os.listdir(lookup_chunk_dir) if f.endswith(".json"))
    lookup_table = []
    cpu_index = faiss.IndexShards()
    for fpath in tqdm(lookup_files, disable=not use_tqdm, desc="loading lookup table"):
        with open(os.path.join(lookup_chunk_dir, fpath), "r", encoding="utf-8") as f:
            lookup_table.extend(json.load(f))
    for fpath in tqdm(index_files, disable=not use_tqdm, desc="loading index"):
        index = faiss.read_index(os.path.join(index_chunk_dir, fpath))
        cpu_index.add_shard(index)
    return cpu_index, lookup_table

def merge_split_flat_embeddings(input_dir, output_path, use_tqdm=True):
    index_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".faiss"))
    first_index = faiss.read_index(os.path.join(input_dir, index_files[0]))
    d = first_index.d
    merged_index = faiss.IndexFlatL2(d)
    for fpath in tqdm(index_files[1:], desc="merging indices", disable= not use_tqdm):
        index = faiss.read_index(os.path.join(input_dir, fpath))
        xb = index.reconstruct_n(0, index.ntotal)
        merged_index.add(xb)
    faiss.write_index(merged_index, output_path)
        
def _find_bz2_files(root_dir):
    bz2_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".bz2"):
                full_path = os.path.join(dirpath, filename)
                bz2_files.append(full_path)
    return sorted(bz2_files)
                