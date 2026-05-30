"""Configuration for the API layer (LLM NL-query parser + serving constants)."""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Serving ---
MAX_RESULTS = 50
NL_CANDIDATES = 50       # candidate pool before post-filtering

# --- LLM NL-query parser (provider-agnostic) ---
# Provider is env-configured. The default URL targets a Gemini-compatible
# generateContent endpoint; point LLM_API_URL at another provider and adjust the
# request/response shaping in nlq._parse_llm to switch. Optional — nlq falls back
# to a local regex parser when no key is set, so the endpoint always works offline.
LLM_API_KEY = (
    os.environ.get("LLM_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
)
LLM_API_URL = os.environ.get(
    "LLM_API_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
)
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "15"))       # seconds

LLM_SYSTEM_PROMPT = """\
You are a product search query parser for an Indian fashion e-commerce site.
Extract structured filters from the query and return ONLY a valid JSON object — no markdown, no explanation, no extra text.

Schema:
{"free_text": "search terms", "brand": "brand or null", "category": "category or null", "colour": "colour or null", "min_price": number or null, "max_price": number or null}

Rules:
- free_text: the core product search terms (remove price/brand keywords)
- brand: only if explicitly mentioned (e.g. "by Nike", "Nike shoes")
- category: product type (e.g. shoes, saree, dress, jeans, kurta, shirt)
- colour: only if a colour word is present
- min_price: from "above/over/more than X"
- max_price: from "under/below/less than X"
- Use null for absent fields, not empty strings

Examples:
Input: Nike shoes under 5000
{"free_text":"shoes","brand":"Nike","category":"shoes","colour":null,"min_price":null,"max_price":5000}
Input: red dress by Zara above 1000
{"free_text":"dress","brand":"Zara","category":"dress","colour":"red","min_price":1000,"max_price":null}
Input: saree under 500
{"free_text":"saree","brand":null,"category":"saree","colour":null,"min_price":null,"max_price":500}
Input: formal shirts
{"free_text":"formal shirts","brand":null,"category":"shirts","colour":null,"min_price":null,"max_price":null}
Input: black jeans under 3000
{"free_text":"jeans","brand":null,"category":"jeans","colour":"black","min_price":null,"max_price":3000}
Input: blue kurta by FabIndia
{"free_text":"kurta","brand":"FabIndia","category":"kurta","colour":"blue","min_price":null,"max_price":null}

Now parse this query and return ONLY the JSON:"""

# Colour keywords for the local regex fallback parser
COLOURS = (
    "black|white|red|blue|green|yellow|pink|purple|orange|brown|grey|gray"
    "|beige|navy|maroon|cream|gold|silver|teal|coral|ivory|khaki|olive|tan"
    "|turquoise|magenta|lavender|burgundy|peach|rust|mint|indigo|charcoal"
    "|multicolour|multicolor|multi"
)
