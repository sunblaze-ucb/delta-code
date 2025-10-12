# SQL

### **(The SQL synthesized problem family is still under construction due to expensive API cost.)**

Readers please feel free to contact authors if you are interested in this problem scope.

## Dataset Structure (`./database`)

This version utilizes Bird (69db) and Spider's training dataset (166db), composed a total of 235db. Each database targets a certain domain and consists of multiple tables.

## Pipeline Architecture

The pipeline implements a **9-step workflow** with forward and backward passes for SQL generation quality improvement:

1. **Setup Sampler**: Initialize diversity sampling based on batch_size and num_iteration
2. **Generate Query (Forward)**: Natural language query generation from database schema  
3. **Generate Groundtruth (Forward)**: SQL generation from natural language query
4. **Verify Format**: Execute SQL and materialize results to verdict database
5. **Verify Groundtruth (Forward)**: LLM-based verification of query-SQL correctness and adherence
6. **Generate Unit Test (Backward)**: Create comprehensive unit tests from SQL result table
7. **Generate Query (Backward)**: Generate improved natural language query from SQL + result table  
8. **Verify Again (Backward)**: Execute unit tests for final verification
9. **Save to Dataset**: Convert to dataset.jsonl if verdict is "correct" and adherence is "adheres" or "partial"

### Batch Execution Support

The pipeline supports **batch execution** and **iterations**:
- **Batch Size**: Number of parallel pipeline runs per iteration (default: 5)
- **Iterations**: Number of batch runs to execute (default: 1) 
- **Diversity Sampling**: Intelligent sampling for specification diversity across batches
- **Execution Logging**: Full process logs saved to `execution_log/` directory
