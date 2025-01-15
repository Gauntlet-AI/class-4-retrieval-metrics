import numpy as np
from typing import List, Any

def calculate_mrr(retrieved_lists: List[List[Any]], 
                 ground_truth_lists: List[List[Any]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for a set of queries.
    
    Args:
        retrieved_lists: List of lists containing retrieved items for each query
        ground_truth_lists: List of lists containing relevant items for each query
        
    Returns:
        float: MRR score
        
    Example:
        >>> retrieved = [
            ['doc1', 'doc2', 'doc3'],
            ['doc4', 'doc2', 'doc1'],
            ['doc3', 'doc1', 'doc4']
        ]
        >>> ground_truth = [
            ['doc2'],
            ['doc1'],
            ['doc4']
        ]
        >>> calculate_mrr(retrieved, ground_truth)
        0.5556  # (1/2 + 1/3 + 1/3) / 3
    """
    if len(retrieved_lists) != len(ground_truth_lists):
        raise ValueError("Number of queries must match between retrieved and ground truth lists")
    
    reciprocal_ranks = []
    
    for retrieved, ground_truth in zip(retrieved_lists, ground_truth_lists):
        # Find the first position (1-based) where a retrieved item is in ground truth
        for rank, item in enumerate(retrieved, 1):
            if item in ground_truth:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            # If no relevant item is found
            reciprocal_ranks.append(0.0)
    
    mrr = np.mean(reciprocal_ranks)
    return mrr

# Example usage with sample data
if __name__ == "__main__":
    # Example 1: Perfect retrieval
    retrieved_perfect = [
        ['doc1', 'doc2', 'doc3'],
        ['doc4', 'doc5', 'doc6']
    ]
    ground_truth_perfect = [
        ['doc1'],
        ['doc4']
    ]
    mrr_perfect = calculate_mrr(retrieved_perfect, ground_truth_perfect)
    print(f"Perfect MRR: {mrr_perfect:.4f}")  # Should be 1.0

    # Example 2: Imperfect retrieval
    retrieved_imperfect = [
        ['doc2', 'doc1', 'doc3'],  # Relevant doc at position 2
        ['doc5', 'doc6', 'doc4'],  # Relevant doc at position 3
    ]
    ground_truth_imperfect = [
        ['doc1'],
        ['doc4']
    ]
    mrr_imperfect = calculate_mrr(retrieved_imperfect, ground_truth_imperfect)
    print(f"Imperfect MRR: {mrr_imperfect:.4f}")  # Should be 0.4167

    # Example 3: Real-world scenario with multiple relevant documents
    retrieved_real = [
        ['doc1', 'doc2', 'doc3', 'doc4'],
        ['doc5', 'doc2', 'doc1', 'doc6'],
        ['doc7', 'doc8', 'doc9', 'doc1']
    ]
    ground_truth_real = [
        ['doc2', 'doc3'],  # First relevant doc at position 2
        ['doc1', 'doc2'],  # First relevant doc at position 2
        ['doc1', 'doc9']   # First relevant doc at position 3
    ]
    mrr_real = calculate_mrr(retrieved_real, ground_truth_real)
    print(f"Real-world MRR: {mrr_real:.4f}")  # Should be 0.4444