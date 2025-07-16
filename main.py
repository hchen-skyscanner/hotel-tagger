import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce

import requests
from openai import OpenAI

LOCAL_PROMPT = """
You are a Hotel Tag Machine.Goal: Determine which tags from a user-supplied list genuinely apply to a hotel, using any mix of descriptions, reviews, amenity lists, or other text snippets.

1 · Input (JSON)

```json
{
 "texts": ["<string 1>", "<string 2>", …],   // one or more texts in any language
 "candidate_tags": ["<tag1>", "<tag2>", …]
}
```

2 · Process rules

Loop through every element in texts; treat all snippets as one evidence pool.

Match tags semantically—recognise synonyms, paraphrases, and context (multilingual understanding assumed).

Assign a tag only when evidence is clear.

If contradictory statements appear, precedence = first appearance in candidate_tags; lower-priority conflicting tags are ignored.

Never invent new tags.

3 · Output (JSON) – ordered list plus confidence:

```json
{
 "tags": [
   { "tag": "<tag1>", "confidence": 0.87 },
   { "tag": "<tag4>", "confidence": 0.71 }
 ]
}
```

Include only tags with confidence > 0.50 (default threshold).

confidence = float between 0 and 1, rounded to two decimals.

No explanations or extra keys.

Example I/OInput

```json
{
 "texts": [
   "Frente a la laguna, el resort ofrece piscina infinita solo para adultos.",
   "Hay un spa de servicio completo y Wi-Fi gratuito en todas las habitaciones."
 ],
 "candidate_tags": ["beachfront", "family_friendly", "adults_only", "spa", "free_wifi", "pet_friendly"]
}
```

Output

```json
{
 "tags": [
   { "tag": "beachfront",  "confidence": 0.83 },
   { "tag": "adults_only", "confidence": 0.93 },
   { "tag": "spa",         "confidence": 0.91 },
   { "tag": "free_wifi",   "confidence": 0.88 }
 ]
}
```

{The user’s style so far: concise, imperative specs; prefers JSON I/O and explicit rule lists.}
"""

DEFAULT_TAG_LIST = [
    "Capsule Hotel",
    "Ryokan Hotel",
    "With Onsen or Saunas",
    "Direct transportation to airport",
    "Of Great deal",
    "Pub Hotels",
    "With rooftop pool",
    "Riverside stays",
    "Mall& transit connected",
    "Ottoman Mansion Hotels",
    "With Rooftop-view Terrence",
    "Waterfront stays",
    "Loft-style",
    "Capsule/Pod Hotels",
    "With Skyline Views",
    "Balconies with City Views",
    "Beachfront stays",
    "Arts (Boutique) & Design hotels",
    "Palazzo style hotels",
    "Monastery Stays"
]


class ReviewsFetcher:
    def __init__(self):
        pass

    def get_reviews(self, hotel_id: str, offset: int, limit: int) -> list[str]:
        url = f'http://aegwynn.skyscanner.io/api/v4/reviews/{hotel_id}?offset={offset}&limit={limit}'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return [*map(lambda review: review.review, data.get('reviews', []))]
        else:
            print(f"Error fetching reviews: {response.status_code}")
        return []


class LlmTagger:
    def __init__(self,
                 api_key: str,
                 tag_list: list[str],
                 model_name: str = "gpt-4o"):
        self._model_name: str = model_name
        self._api_key: str = api_key
        self._llm_input_dict: dict = {
            "texts": [],
            "candidate_tags": tag_list
        }
        self._client = OpenAI(
            # This is the default and can be omitted
            api_key=self._api_key,
        )

    def tag_reviews(self, reviews: list[str]) -> list[str]:
        if len(reviews) == 0:
            return []

        self._llm_input_dict["texts"] = reviews

        input_text = json.dumps(self._llm_input_dict)
        response_text = self._call_llm(input_text)
        return json.loads(response_text)

    def _call_llm(self, intput_text: str) -> str:
        response = self._client.responses.create(
            model=self._model_name,
            instructions=LOCAL_PROMPT,
            input=intput_text,
        )

        return response.output_text


def get_tags_from_hotel(
        hotel_id: str,
        confidence: float,
        reviews_fetcher: ReviewsFetcher,
        tagger: LlmTagger) -> list[str]:
    reviews = reviews_fetcher.get_reviews(hotel_id, 0, 100)
    if not reviews:
        print(f"No reviews found for hotel ID: {hotel_id}")
        return []
    tagged_reviews = tagger.tag_reviews(reviews)
    if not tagged_reviews:
        print(f"No tags found for hotel ID: {hotel_id}")
        return []

    res = map(
        lambda t: t["tag"] if float(t["confidence"]) >= confidence else None,
        tagged_reviews
    )
    return list(filter(lambda x: x is not None, res))


def save_to_json(tags: dict[str, list[str]], filename: str):
    with open(filename, 'a') as f:
        json.dump(tags, f)


def worker_process(hotel_id: str,
                   confidence: float,
                   reviews_fetcher: ReviewsFetcher,
                   tagger: LlmTagger) -> dict[str: list[str]]:
    print(f"Processing hotel ID: {hotel_id}")
    tags = get_tags_from_hotel(
        hotel_id,
        confidence,
        reviews_fetcher,
        tagger
    )

    if not tags:
        print(f"No tags found for hotel ID: {hotel_id}")
        return []

    print(f"Tags for hotel ID {hotel_id}: {tags}")
    return {hotel_id: tags}


def main():
    parser = argparse.ArgumentParser(description='Tag hotel reviews using LLM.')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--hotel_id', type=str, required=True, help='Hotel ID to fetch reviews for')
    parser.add_argument('--output_file', type=str, default='tags_output.json', help='Output file name for tags')
    parser.add_argument('--confidence', type=float, default=0.50, help='Confidence threshold for tags')
    args = parser.parse_args()

    hotel_ids = [s.strip() for s in args.hotel_id.split(",")]
    reviews_fetcher = ReviewsFetcher()
    tagger = LlmTagger(
        api_key=args.api_key,
        tag_list=DEFAULT_TAG_LIST
    )

    with ThreadPoolExecutor(10) as thread_executor:
        futures = [
            thread_executor.submit(
                worker_process,
                hotel_id,
                args.confidence,
                reviews_fetcher,
                tagger
            ) for hotel_id in hotel_ids
        ]

        results = [f.result() for f in as_completed(futures)]
        tags = reduce(lambda acc, y: acc.update(y), results, {})
        save_to_json(tags, args.output_file)


if __name__ == '__main__':
    main()
