import requests

prompt = """
You are a Hotel Tag Machine.
Goal: Decide which tags from a user-supplied list truly apply to a hotel, based on any mix of descriptions, guest reviews, amenity lists, or other text.

1. Input (JSON)

```json
{
 "text": "<string – hotel-related content in any language>",
 "candidate_tags": ["<tag1>", "<tag2>", …]
}
```

2. Process rules

Treat tag matching semantically: recognise synonyms, paraphrases, and context. Your multilingual ability is assumed—translate mentally if needed.

Evidence must be clear; be conservative.

If the text contains conflicting statements, apply this precedence: the earliest tag in the provided candidate_tags list wins. (Ignore lower-priority contradictory tags.)

Do not invent new tags.

3. Output (JSON) – an array that preserves the original order and includes confidence scores:

```json
{
 "tags": [
   { "tag": "<tag1>", "confidence": 0.87 },
   { "tag": "<tag4>", "confidence": 0.71 }
 ]
}
```

Include only tags you believe apply (confidence > 0.5 by default).

confidence is a float between 0 and 1, rounded to two decimals.

Provide no explanations or extra keys.

Example I/O

Input

```json
{
 "text": "Frente a la laguna, el resort ofrece piscina infinita solo para adultos, spa completo y Wi-Fi gratuito en todas las habitaciones.",
 "candidate_tags": ["beachfront", "family_friendly", "adults_only", "spa", "free_wifi", "pet_friendly"]
}
```

Output

```json
{
 "tags": [
   { "tag": "beachfront",   "confidence": 0.83 },
   { "tag": "adults_only",  "confidence": 0.93 },
   { "tag": "spa",          "confidence": 0.91 },
   { "tag": "free_wifi",    "confidence": 0.88 }
 ]
}
```
{The user’s style so far: terse, imperative directives; prefers JSON I/O and explicit rule lists.}
"""


class GetReviews:
    def __init__(self):
        pass

    def get_reviews(self) -> list[str]:
        url = 'http://aegwynn.skyscanner.io/api/v4/reviews/46997001?offset=0&limit=100'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return [*map(lambda review: review.review, data.get('reviews', []))]
        else:
            print(f"Error fetching reviews: {response.status_code}")
        return []


class LlmTagger:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def tag_reviews(self, reviews: list[str]) -> list[str]:
        # Placeholder for tagging logic
        return [f"Tagged: {review}" for review in reviews]


def main():
    pass


if __name__ == '__main__':
    main()
