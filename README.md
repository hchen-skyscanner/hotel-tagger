# Hotel Tagger

Hotel Tagger is a Python tool that uses OpenAI's LLMs to semantically tag hotels based on their reviews and descriptions. It fetches reviews for specified hotel IDs, applies a set of candidate tags, and outputs the most relevant tags with confidence scores.

## Features
- Fetches hotel reviews from a remote API
- Uses OpenAI LLM (default: GPT-4o) to assign tags
- Supports custom tag lists
- Multithreaded processing for multiple hotels
- Outputs tags with confidence scores in JSON format

## Usage

### Prerequisites
- Python 3.8+
- OpenAI API key
- Required Python packages: `requests`, `openai`

Install dependencies:
```bash
pip install requests openai
```

### Command Line Arguments
- `--api_key` (required): Your OpenAI API key
- `--hotel_id` (required): Hotel ID(s) to process (comma-separated for multiple)
- `--output_file`: Output JSON file (default: `tags_output.json`)
- `--confidence`: Confidence threshold for tags (default: `0.50`)

### Example
```bash
python main.py --api_key YOUR_OPENAI_KEY --hotel_id 12345,67890 --output_file results.json --confidence 0.7
```

## How It Works
1. Fetches reviews for each hotel ID from the Skyscanner API
2. Sends reviews and candidate tags to the LLM with a custom prompt
3. Receives tags with confidence scores, filters by threshold
4. Saves results to a JSON file

## Customization
- Modify `DEFAULT_TAG_LIST` in `main.py` to change candidate tags
- Adjust the prompt in `LOCAL_PROMPT` for different tagging rules

## Output Format
The output JSON contains hotel IDs mapped to lists of tags:
```json
{
  "12345": ["beachfront", "spa", "free_wifi"],
  "67890": ["family_friendly", "pet_friendly"]
}
```

## License
MIT
