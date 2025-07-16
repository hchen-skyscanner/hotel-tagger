import json
import os

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


def main():
    pass


if __name__ == '__main__':
    main()
