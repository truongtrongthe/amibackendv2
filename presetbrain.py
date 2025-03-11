# Preset Brain (1) - Basic SPIN function
def spin_question(type, content):
    if type == "Situation":
        return f"Tell me about {content} in your current setup."
    # Add Problem, Implication, Need-Payoff later

# Training Path (3) - Process a lecture
class TrainingPath:
    def __init__(self):
        self.confidence_scores = {}
    
    def absorb_lecture(self, lecture, source, claimed_effectiveness):
        # Example lecture: "Ask this Situation question, it’s 90% effective"
        question = spin_question("Situation", lecture)
        # Self-test (simplified: assume it works unless corrected)
        test_result = self.run_test(question)
        # Initial confidence: average of claimed and test
        confidence = (claimed_effectiveness + test_result) / 2
        self.confidence_scores[question] = confidence
        return {"question": question, "confidence": confidence, "source": source}

    def run_test(self, question):
        # Placeholder: real test would grade Accuracy, Relevance, Fluency
        return 85  # Dummy score for now

# Test it
trainer = TrainingPath()
result = trainer.absorb_lecture("how you manage inventory", "Rep A, 3/15/25", 90)
print(result)
# Output: {'question': 'Tell me about how you manage inventory in your current setup.', 'confidence': 87.5, 'source': 'Rep A, 3/15/25'}