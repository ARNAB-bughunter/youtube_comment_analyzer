import random
import time

import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader

from emotion_analyse import classify_emotion
from sentiment_analyse import analyse
from quesion_analyse import classify_comment


SENTIMENT_SCORE = {"positive": 1, "neutral": 0, "negative": -1}


def fetch_comments(url: str) -> list[dict]:
    downloader = YoutubeCommentDownloader()
    raw = downloader.get_comments_from_url(url)
    return [
        {
            "text": c["text"],
            "votes": c["votes"],
            "replies": c["replies"],
            "heart": c["heart"],
            "reply": c["reply"],
            "time_parsed": c["time_parsed"],
        }
        for c in raw
    ]


def sample_comments(comments: list[dict]) -> list[dict]:
    n = len(comments)
    if n < 500:
        return comments

    if n <= 5_000:
        target = 2_000
    elif n <= 50_000:
        target = 3_000
    else:
        target = 5_000

    target = min(target, n)

    sorted_by_time = sorted(comments, key=lambda c: c["time_parsed"])
    cutoff = len(sorted_by_time) // 2

    old = sorted_by_time[:cutoff]
    new = sorted_by_time[cutoff:]

    n_old = int(target * 0.4)
    n_new = int(target * 0.4)
    n_random = target - n_old - n_new

    picked_old = random.sample(old, min(n_old, len(old)))
    picked_new = random.sample(new, min(n_new, len(new)))

    remaining = [c for c in comments if c not in picked_old and c not in picked_new]
    picked_random = random.sample(remaining, min(n_random, len(remaining)))

    return picked_old + picked_new + picked_random


def build_dataframe(comments: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(comments)
    df["date"] = pd.to_datetime(df["time_parsed"], unit="s")
    return df


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df[["sentiment_label", "sentiment_score"]] = df["text"].apply(lambda t: pd.Series(analyse(t)))
    df["sentiment_score"] = df["sentiment_label"].map(SENTIMENT_SCORE)
    return df

def clasify_statement_feedback_question_request(df: pd.DataFrame) -> pd.DataFrame:
    df["comment_type"] = df["text"].apply(classify_comment)
    return df

def add_emotion(df: pd.DataFrame) -> pd.DataFrame:
    def top_emotion(text: str) -> str:
        scores = classify_emotion(text)
        return max(scores, key=lambda x: x["score"])["label"]

    df["emotion"] = df["text"].apply(top_emotion)
    return df


def compute_trend(df: pd.DataFrame) -> pd.DataFrame:
    df["bucket"] = df["date"].dt.to_period("W")
    trend = df.groupby("bucket")["sentiment_score"].mean().reset_index()

    rolling_mean = trend["sentiment_score"].rolling(3, min_periods=1).mean()
    rolling_std = trend["sentiment_score"].rolling(3, min_periods=1).std().fillna(0)

    trend["is_spike"] = (trend["sentiment_score"] - rolling_mean).abs() > (1.5 * rolling_std)
    trend["rolling_mean"] = rolling_mean
    trend["rolling_std"] = rolling_std
    return trend


def detect_pattern(scores: pd.Series) -> tuple[str, str]:
    if len(scores) < 2:
        return "flat", "Not enough data to detect a trend yet."

    mid = len(scores) // 2
    overall_slope = scores.iloc[mid:].mean() - scores.iloc[:mid].mean()

    rolling_mean = scores.rolling(3, min_periods=1).mean()
    rolling_std = scores.rolling(3, min_periods=1).std().fillna(0)
    spikes = (scores - rolling_mean).abs() > (1.5 * rolling_std.replace(0, float("inf")))

    if spikes.any():
        spike_idx = spikes.idxmax()
        if scores[spike_idx] > rolling_mean[spike_idx]:
            return "sudden_spike", f"Spike on week {spike_idx} — possibly shared by another creator or went viral."
        return "sudden_drop", f"Sentiment dropped sharply around week {spike_idx} — check comments from that period."

    if overall_slope > 0.15:
        return "rising", "Your video aged well — later audience loves it."
    if overall_slope < -0.15:
        return "falling", "Initial excitement faded — consider a follow-up video."
    return "flat", "Steady reception — reliable content for this topic."


def print_report(trend: pd.DataFrame, pattern: str, message: str) -> None:
    print("\n--- Trend ---")
    print(trend.to_string(index=False))
    print(f"\nPattern : {pattern}")
    print(f"Summary : {message}")
    print(
        "\nNote: trend is based on top-liked comments, not purely chronological — "
        "early timestamps may be underrepresented."
    )


def main(video_url) -> None:
    t0 = time.time()

    comments = fetch_comments(video_url)
    print(f"Fetched {len(comments)} comments in {time.time() - t0:.1f}s")

    comments = sample_comments(comments)
    print(f"Sampled down to {len(comments)} comments for analysis")

    df = build_dataframe(comments)
    df = add_sentiment(df)
    df = clasify_statement_feedback_question_request(df)
    df = add_emotion(df)
    # df.to_csv("test.csv")

    trend = compute_trend(df)
    scores = trend.set_index("bucket")["sentiment_score"].reset_index(drop=True)
    pattern, message = detect_pattern(scores)

    print_report(trend, pattern, message)


if __name__ == "__main__":
    main("https://www.youtube.com/watch?v=5keqGmhEwaU")
