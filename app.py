from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from pathlib import Path

from comment import fetch_comments, sample_comments, build_dataframe, add_sentiment, add_emotion, compute_trend, detect_pattern, clasify_statement_feedback_question_request

app = FastAPI()

BASE_DIR = Path(__file__).parent


class AnalyseRequest(BaseModel):
    url: str


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "yt-comment-analyzer.html")


@app.post("/analyse", responses={400: {"description": "Failed to fetch comments"}, 404: {"description": "No comments found"}})
async def analyse(body: AnalyseRequest):
    try:
        comments = fetch_comments(body.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch comments: {e}")

    if not comments:
        raise HTTPException(status_code=404, detail="No comments found for this video.")

    comments = sample_comments(comments)
    df = build_dataframe(comments)
    df = add_sentiment(df)
    df = add_emotion(df)
    df = clasify_statement_feedback_question_request(df)

    pos = int((df["sentiment_label"] == "positive").sum())
    neu = int((df["sentiment_label"] == "neutral").sum())
    neg = int((df["sentiment_label"] == "negative").sum())

    trend = compute_trend(df)
    scores = trend.set_index("bucket")["sentiment_score"].reset_index(drop=True)
    pattern, message = detect_pattern(scores)
    question_count = int((df["comment_type"] == "question").sum())

    
    

    emotion_counts = df["emotion"].value_counts(normalize=True).mul(100).round(1)
    all_emotions = ["joy", "surprise", "anger", "sadness", "disgust", "fear", "neutral"]

    return {
        "total_comments": len(comments),
        "total_comments_analysed": len(df),
        "question_count" : question_count,
        "pattern": pattern,
        "summary": message,
        "emotions": {e: float(emotion_counts.get(e, 0.0)) for e in all_emotions},
        "sentiment": {
            "positive_pct": round(pos / len(df) * 100, 1) if len(df) else 0.0,
            "neutral_pct": round(neu / len(df) * 100, 1) if len(df) else 0.0,
            "negative_pct": round(neg / len(df) * 100, 1) if len(df) else 0.0,
        },
        "trend": [
            {
                "week": str(row["bucket"]),
                "sentiment_score": float(row["sentiment_score"]),
                "rolling_mean": float(row["rolling_mean"]),
                "is_spike": bool(row["is_spike"]),
            }
            for _, row in trend.iterrows()
        ],
    }
