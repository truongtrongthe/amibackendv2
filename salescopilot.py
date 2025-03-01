from conversationflow import graph

salesperson_id = "sales_123"
message = "Just talked to John Doe. He’s hesitant about upgrading."

# Run workflow
result = graph.invoke({"message": message, "salesperson_id": salesperson_id})

# Display retrieved memory and next steps
print("Past interactions:", result["customer"]["past_interactions"])
print("Next steps:", result["suggest_next_steps"])
