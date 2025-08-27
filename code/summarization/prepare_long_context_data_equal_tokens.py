import json
import os
import argparse
from google import genai
from google.genai import types

def setup_gemini_client():
    """Set up Gemini client with credentials."""
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''

    # TODO: change to your own project and location
    client = genai.Client(
        vertexai=True,
        project='',
        location='',
        http_options=types.HttpOptions(api_version='v1')
    )
    return client

def truncate_document_to_tokens(doc_content, client, model_name="gemini-2.5-pro", max_tokens=6000):
    """Truncate a single document to max_tokens."""
    # First check if document is already under limit
    token_count = client.models.count_tokens(
        model=model_name, 
        contents=doc_content
    )
    current_tokens = token_count.total_tokens
    
    if current_tokens <= max_tokens:
        return doc_content, current_tokens
    
    left = 0
    right = len(doc_content)
    best_truncate = right // 2
    best_tokens = 0
    
    estimated_chars = max_tokens * 4
    if estimated_chars < len(doc_content):
        right = min(estimated_chars * 2, len(doc_content))
        best_truncate = estimated_chars
    
    for _ in range(5):
        mid = (left + right) // 2
        truncated = doc_content[:mid]
        
        token_count = client.models.count_tokens(
            model=model_name, 
            contents=truncated
        )
        tokens = token_count.total_tokens
        
        if tokens <= max_tokens:
            best_truncate = mid
            best_tokens = tokens
            left = mid + 1
            if tokens > max_tokens * 0.95:  
                break
        else:
            right = mid - 1
    
    return doc_content[:best_truncate], best_tokens

def load_documents_with_equal_tokens(doc_folder, client, model_name="gemini-2.5-pro", tokens_per_doc=6000):
    """Load all documents, giving each document equal token budget."""
    doc_files = sorted([f for f in os.listdir(doc_folder) if f.endswith('.txt')])
    
    print(f"  Loading {len(doc_files)} documents from {doc_folder}")
    print(f"  Max tokens per document: {tokens_per_doc:,}")
    print(f"  Maximum possible total: {len(doc_files) * tokens_per_doc:,} tokens")
    
    truncated_docs = []
    total_tokens = 0
    docs_truncated = 0
    docs_unchanged = 0
    
    for i, doc_file in enumerate(doc_files):
        doc_path = os.path.join(doc_folder, doc_file)
        with open(doc_path, 'r') as f:
            doc_content = f.read()
        
        truncated_content, doc_tokens = truncate_document_to_tokens(
            doc_content, client, model_name, tokens_per_doc
        )
        
        truncated_docs.append(truncated_content)
        total_tokens += doc_tokens
        
        if len(truncated_content) < len(doc_content):
            docs_truncated += 1
        else:
            docs_unchanged += 1
        
        # Progress update every 20 documents
        if (i + 1) % 20 == 0:
            print(f"    Processed {i+1}/{len(doc_files)} documents...")
            print(f"      Current total tokens: {total_tokens:,}")
            print(f"      Truncated: {docs_truncated}, Unchanged: {docs_unchanged}")
    
    combined_text = "<doc-sep>".join(truncated_docs)
    
    print(f"  Completed processing all {len(doc_files)} documents")
    print(f"  Documents truncated: {docs_truncated}")
    print(f"  Documents unchanged (already under limit): {docs_unchanged}")
    print(f"  Total tokens (before combining): {total_tokens:,}")
    
    return combined_text, len(doc_files), docs_truncated, docs_unchanged

def check_and_truncate_final(text, client, model_name="gemini-2.5-pro", max_tokens=1048000):
    """Check final token count and truncate from end if needed."""
    print(f"\n  Checking final combined token count...")
    
    # Count tokens for the full combined text
    token_count = client.models.count_tokens(
        model=model_name, 
        contents=text
    )
    current_tokens = token_count.total_tokens
    
    print(f"  Final token count: {current_tokens:,}")
    
    if current_tokens <= max_tokens:
        print(f"  ✓ Within token limit of {max_tokens:,}!")
        return text, current_tokens, False
    

    keep_ratio = max_tokens / current_tokens * 0.98  # Be conservative
    truncate_at = int(len(text) * keep_ratio)
    
    truncated_text = text[:truncate_at]
    token_count = client.models.count_tokens(
        model=model_name, 
        contents=truncated_text
    )
    final_tokens = token_count.total_tokens
    
    print(f"  ✓ After tail truncation: {final_tokens:,} tokens")
    
    return truncated_text, final_tokens, True

def prepare_long_context_data(domain="story", tokens_per_doc=6000):
    """Prepare data with equal token budget per document."""
    
    # Paths
    base_path = ""
    # TODO: change to your own base path
    
    if domain == "story":
        queries_file = f"{base_path}/story/queries_test.json"
        doc_folder = f"{base_path}/story/documents"
        output_file = f"{base_path}/story/long_context_test_equal_tokens.json"
    else:  # meeting
        queries_file = f"{base_path}/meeting/queries_test.json"
        doc_folder = f"{base_path}/meeting/documents"
        output_file = f"{base_path}/meeting/long_context_test_equal_tokens.json"
    
    # Load queries
    with open(queries_file, 'r') as f:
        queries_data = json.load(f)
    
    print(f"Loading documents for {domain} with equal token budget...")
    print(f"  Token budget per document: {tokens_per_doc:,}")
    
    # Set up Gemini client for token counting
    client = setup_gemini_client()
    
    combined_text, total_docs, docs_truncated, docs_unchanged = load_documents_with_equal_tokens(
        doc_folder, client, tokens_per_doc=tokens_per_doc
    )
    
    final_text, final_tokens, was_tail_truncated = check_and_truncate_final(
        combined_text, client, max_tokens=1048000
    )
    
    print(f"\n  === Final Statistics ===")
    print(f"  Documents: {total_docs}")
    print(f"  - Truncated to {tokens_per_doc} tokens: {docs_truncated}")
    print(f"  - Already under limit: {docs_unchanged}")
    print(f"  Final token count: {final_tokens:,}")
    print(f"  Final character count: {len(final_text):,}")
    if was_tail_truncated:
        print(f"  Note: Additional tail truncation was applied")
    
    long_context_data = []
    
    for idx, (key, item) in enumerate(queries_data.items()):
        long_context_item = {
            "Query": item["query"],
            "Article": final_text,
            "Summary_1": item["answer"][0] if len(item["answer"]) > 0 else "",
            "Summary_2": item["answer"][1] if len(item["answer"]) > 1 else "",
            "Summary_3": item["answer"][2] if len(item["answer"]) > 2 else "",
            "Summary_4": item["answer"][3] if len(item["answer"]) > 3 else ""
        }
        
        long_context_data.append(long_context_item)
    
    with open(output_file, 'w') as f:
        json.dump(long_context_data, f, indent=2)
    
    print(f"\nPrepared {len(long_context_data)} examples for {domain} long-context experiments")
    print(f"Each example contains ALL {total_docs} documents (max {tokens_per_doc} tokens each)")
    print(f"Saved to: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Prepare long-context data with equal token budget per document")
    parser.add_argument("--domain", type=str, default="meeting", choices=["story", "meeting"])
    parser.add_argument("--tokens-per-doc", type=int, default=6000, 
                       help="Maximum tokens per document (default: 6000)")
    args = parser.parse_args()
    
    output_file = prepare_long_context_data(args.domain, args.tokens_per_doc)
    print(f"\nData preparation complete. Output file: {output_file}")

if __name__ == "__main__":
    main()