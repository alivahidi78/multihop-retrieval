import os
import bz2
import ast
class Retriever:
    def __init__(self, wiki_path, embedder, index, metadata):
        self.wiki_path = wiki_path
        self.embedder = embedder
        self.index = index
        self.metadata = metadata
        
    def retrieve_info_rag(self, query_list, top_k=5):
        result_list = []
        for query in query_list:
            results = self._query_database(query, top_k=top_k)
            context = []
            for res in results:
                context += [res["content"]]
            result_list.append(context)
        return result_list
    
    def _query_database(self, text, top_k=5):
        """queries database"""
        vec = self.embedder.encode([text], convert_to_numpy=True, show_progress_bar = False).astype("float32")
        distances, indices = self.index.search(vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            file_path, line_num, sentence_num = self.metadata[idx]
            results.append({
                "distance": float(dist),
                "source_file": file_path,
                "line_number": line_num,
                "sentence_num": sentence_num,
                "content": self._get_line_from_bz2(file_path,line_num,sentence_num)
            })
        return results
    
    def _get_line_from_bz2(self, file_path, line_number, sentence_num):
        """Reads a specific line from a .bz2 compressed file."""
        file_path = os.path.join(self.wiki_path, file_path)
        try:
            with bz2.open(file_path, "rt") as f:
                for i, line in enumerate(f):
                    if i == line_number:
                        line_raw = line.strip()
                        data = ast.literal_eval(line_raw)
                        full_text= data.get("text", line_raw)
                        sentence = full_text[sentence_num]
                        title = data.get("title", line_raw)
                        info = {}
                        info["title"] = title
                        info["full_text"] = full_text
                        info["sentence"] = sentence
                        return info
        except Exception as e:
            print(f"Error reading: {file_path}")
            return ""
        print(f"Line not found: {line_number} of {file_path}")
        return ""
    
    