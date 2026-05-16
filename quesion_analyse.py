from transformers import pipeline

# Zero-shot classifier
classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli")

def classify_comment(text: str) -> str:
    result = classifier(text,
        candidate_labels=["question", "statement", "request", "feedback"],
        hypothesis_template="This comment is a {}."
    )
    return result["labels"][0]

