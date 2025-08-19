import time

def log_retrieval(query, results):
    """Log retrieval results for debugging"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("retrieval_logs.txt", "a") as f:
        f.write(f"\n\n[{timestamp}] QUERY: {query}\n")
        for i, res in enumerate(results):
            f.write(f"RESULT {i+1}:\n")
            f.write(f"Content: {res.page_content[:200]}...\n")
            f.write(f"Metadata: {res.metadata}\n")
            f.write("-"*50 + "\n")
    return query.replace("<", "&lt;").replace(">", "&gt;").strip()
