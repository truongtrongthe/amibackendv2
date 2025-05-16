# Pinecone Controller Tests

This module contains tests for the Pinecone vector database controller functions in `pccontroller.py`.

## Prerequisites

Ensure you have the following environment variables set:
- `PINECONE_API_KEY` - Your Pinecone API key
- `OPENAI_API_KEY` - Your OpenAI API key (used for embeddings)
- `ENT` - (Optional) Name of the enterprise index (defaults to "ent-index")

## Installation

Make sure you have all the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Tests

To run all tests:

```bash
pytest test_pccontroller.py -v
```

To run a specific test:

```bash
pytest test_pccontroller.py::test_batch_save -v
```

## Test Cases

### 1. Bulk Save and Query Test (`test_bulk_save_and_query`)
- Generates and saves multiple knowledge items individually
- Performs various queries with different filters
- Verifies that the saved items can be retrieved

### 2. Batch Save Test (`test_batch_save`)
- Tests the batch save functionality with 10 vectors at once
- Verifies that batch operations work with different batch sizes
- Tests direct vector upserting to Pinecone

### 3. Single Save and Query Test (`test_single_save_and_query`)
- Tests the most basic save and query operations
- Verifies that a single item can be saved and retrieved

### 4. Duplicate Detection Test (`test_duplicate_detection`)
- Tests the built-in duplicate detection functionality
- Verifies that near-duplicate knowledge entries are handled correctly

## Important Notes

- The tests create unique user IDs and bank names for each test run to avoid conflicts
- Some tests have been simplified to accommodate potential API limitations
- Running the full test suite will take approximately 1-2 minutes

## Cost Considerations

These tests make real API calls to both Pinecone and OpenAI which may incur costs:
- Multiple upserts to Pinecone
- Multiple embedding generations with OpenAI

Make sure you're aware of your usage limits and costs before running these tests repeatedly.

## Troubleshooting

If tests fail with filter-related errors (such as date comparison issues), the following approaches can help:
1. Simplify metadata to avoid complex filtering
2. Use direct Pinecone API calls instead of the higher-level functions
3. Increase wait times between operations to ensure indexing is complete 