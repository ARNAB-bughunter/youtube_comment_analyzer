from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
)

def classify_emotion(text: str) -> list[dict]:
    return classifier(text, truncation=True, max_length=512)


if __name__ == "__main__":
    sample = "Not even the docker website could explain for me what it actually is. You've solved this mystery, thank you."
    results = classify_emotion(sample)
    print(results)
