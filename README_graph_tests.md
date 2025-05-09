# Knowledge Graph Visualization Tests

This directory contains test scripts for the document understanding and knowledge graph visualization module.

## Setup

1. Make sure you have installed all required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have an OpenAI API key set as an environment variable:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Running the Tests

### Basic Graph Visualization Test

This test processes the existing "Self-driving map.docx" file and visualizes the resulting knowledge graph:

```bash
python test_graph_visualization.py
```

The script will:
1. Load the existing "Self-driving map.docx" from the root directory
2. Process the document to extract insights
3. Build and visualize a knowledge graph
4. Save the graph visualization as "self_driving_map_graph_viz.png"

### Multi-Document Graph Test

This test demonstrates how the knowledge graph evolves as multiple documents (including "Self-driving map.docx") are processed:

```bash
python test_multi_document_graph.py
```

The script will:
1. Include your "Self-driving map.docx" file if it exists in the root directory
2. Create additional synthetic documents about AI technology and business applications
3. Process each document sequentially, building upon the same knowledge graph
4. Show how the graph evolves with each document
5. Visualize the final integrated knowledge graph
6. Save visualizations to "multi_doc_self_driving_map_graph.png" and "module_viz_self_driving_map.png"

## Understanding the Output

When the tests run successfully, you should see:

1. Terminal output showing:
   - Document processing progress
   - Cluster information
   - Graph statistics

2. Generated PNG files with visualizations where:
   - Nodes represent content clusters
   - Node colors indicate the source document
   - Edge weights show connection strength between clusters
   - Labels display cluster titles

## Troubleshooting

If you encounter errors:

1. Check that all dependencies are installed
2. Verify your OpenAI API key is set and valid
3. Ensure you have sufficient permissions to write files in the directory
4. Check the docx and matplotlib backends are properly configured

## Visualization Customization

You can modify the visualization parameters in the test scripts to adjust:
- Graph layout algorithm and parameters
- Node and edge colors, sizes, and transparency
- Label fonts and positioning
- Output image dimensions and resolution 