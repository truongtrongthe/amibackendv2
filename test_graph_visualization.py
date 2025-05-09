import asyncio
import os
import uuid

import matplotlib.pyplot as plt
import networkx as nx

from graph_building import understand_document, visualize_knowledge_graph

async def test_with_existing_doc():
    """
    Test graph visualization with an existing document
    """
    print("Loading existing document 'Self-driving map.docx'...")
    
    # Check if file exists
    if not os.path.exists("Self-driving map.docx"):
        print("Error: 'Self-driving map.docx' not found in current directory")
        return
    
    # Define a unique graph file name
    test_graph_file = f"test_knowledge_graph_{uuid.uuid4()}.graphml"
    
    print("Processing document and building graph...")
    # Process the document
    insights = await understand_document(
        input_source="Self-driving map.docx", 
        file_type='docx',
        existing_graph_path=test_graph_file
    )
    
    if not insights["success"]:
        print(f"Error processing document: {insights.get('error', 'Unknown error')}")
        return
    
    print("Document processing succeeded!")
    print(f"Number of clusters: {insights['document_insights']['metadata']['cluster_count']}")
    
    # Load and visualize the graph
    if os.path.exists(test_graph_file):
        print(f"Loading graph from {test_graph_file}")
        graph = nx.read_graphml(test_graph_file)
        
        # Print graph info
        print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        print("Clusters:")
        for node in graph.nodes:
            print(f"  - {node}: {graph.nodes[node].get('title', 'No title')}")
        
        # Visualize with custom settings
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(graph, seed=42)
        
        # Draw nodes with custom appearance
        nx.draw_networkx_nodes(graph, pos, 
                              node_color="lightblue", 
                              node_size=1200,
                              alpha=0.8,
                              edgecolors='black')
        
        # Draw edges with custom appearance
        nx.draw_networkx_edges(graph, pos, 
                              width=2, 
                              alpha=0.5, 
                              edge_color="gray")
        
        # Add labels
        node_labels = {n: graph.nodes[n].get('title', n)[:20] for n in graph.nodes}
        nx.draw_networkx_labels(graph, pos, 
                               labels=node_labels, 
                               font_size=10,
                               font_weight='bold')
        
        # Add edge labels for connection strength
        edge_labels = {(u, v): f"{graph.edges[u, v].get('weight', 0):.2f}" 
                     for u, v in graph.edges}
        nx.draw_networkx_edge_labels(graph, pos, 
                                   edge_labels=edge_labels, 
                                   font_size=8)
        
        # Finalize and save the visualization
        plt.title("Self-driving Map Knowledge Graph")
        plt.axis('off')
        
        output_path = "self_driving_map_graph_viz.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph visualization saved to {output_path}")
    else:
        print(f"Graph file {test_graph_file} not found")

if __name__ == "__main__":
    print("Starting graph visualization test...")
    asyncio.run(test_with_existing_doc())
    print("Test completed!") 