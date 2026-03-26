import json

questions = [
    "What is un-app and what problem does it solve?",
    "What was the pricing strategy for un-app?",
    "How does the on-device neural network work in un-app?",
    "What was the un-app architecture?",
    "How many users did un-app have?",
    "What were the key features in un-app v0.4?",
    "What investors did un-app pitch to?",
    "What is the privacy policy about?",
    "What companies are mentioned in the pitch decks?",
    "What was the fundraising strategy?",
    "What is the Enterprise AI Bible about?",
    "What technologies were used in the project?",
    "What is the business plan template about?",
    "What were the key financial projections?",
    "How was user data handled and stored?",
    "What was the go-to-market strategy?",
    "What were the competitive advantages?",
    "What feedback did users give?",
    "What was the monetization model?",
    "What patents were filed?",
    "What was the team structure?",
    "What were the main challenges faced?",
    "What role did AI play in the product?",
    "What markets were targeted?",
    "How was the product different from competitors?",
    "What were the key milestones achieved?",
    "What was the capital invested?",
    "What led to the shutdown decision?",
    "What lessons were learned from building un-app?",
    "What was the IIT Madras incubation about?",
]

with open("distill_questions.jsonl", "w", encoding="utf-8") as f:
    for q in questions:
        f.write(json.dumps({"question": q, "source_chunk": "", "source_file": ""}) + "\n")

print(f"{len(questions)} questions saved to distill_questions.jsonl")
