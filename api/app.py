from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
from io import BytesIO
from google.cloud import storage
import requests
import base64
import json
import os

app = Flask(__name__)

CANDLESTICK_OPENAI_API_KEY = os.environ.get("CANDLESTICK_OPENAI_API_KEY")
RIZZ_OPENAI_API_KEY = os.environ.get("RIZZ_OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")

OPENAI_PROMPT_FILE_PATH = "openai_prompt.txt"
PERPLEXITY_PROMPT_FILE_PATH = "perplexity_prompt.txt"
RIZZ_PROMPT_FILE_PATH = "rizz_prompt.txt"

OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"
PERPLEXITY_BASE_URL = "https://api.perplexity.ai/chat/completions"


def get_txt_file(filename):
    bucket_name = "gen_ai_prompts"
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(filename)

    text = blob.download_as_text()
    return text


def get_instagram_profile_pic_and_name(instagram_handle):
    url = 'https://instagram.com/{}'.format(instagram_handle)
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch profile for {instagram_handle}. Status code: {response.status_code}")
        return None, None

    soup = BeautifulSoup(response.content, 'html.parser')

    # Get profile pic URL
    profile_pic_tag = soup.find('meta', attrs={'property': 'og:image'})
    profile_pic_url = profile_pic_tag['content'] if profile_pic_tag else None

    # Get display name
    title_tag = soup.find('meta', attrs={'property': 'og:title'})
    display_name = title_tag['content'].split('(@')[0].strip() if title_tag else "Unknown"

    if profile_pic_url:
        pic_response = requests.get(profile_pic_url)
        if pic_response.status_code == 200:
            # Convert to base64
            image_bytes = BytesIO(pic_response.content)
            base64_str = base64.b64encode(image_bytes.read()).decode('utf-8')
            return base64_str, display_name
        else:
            print("Failed to download profile picture.")
            return None, display_name
    else:
        print("Couldn't find profile picture.")
        return None, display_name


@app.route("/getChartAnalysis", methods=["POST"])
def get_chart_analysis():
    try:
        data = request.get_json()
        if not data or "base64Image" not in data:
            return jsonify({"error": "Missing 'base64Image' in JSON body."}), 400

        base64_image = data["base64Image"]
        prompt = get_txt_file(OPENAI_PROMPT_FILE_PATH)

        payload = {
            "model": "gpt-4o-mini-2024-07-18",
            "temperature": 0.5,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "chart_analysis",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "boolean",
                                "description": "Indicates if the uploaded image is a trading chart (true) or not (false)"
                            },
                            "result": {
                                "type": "object",
                                "properties": {
                                    "ticker": {
                                        "type": "string",
                                        "description": "Ticker symbol present in input image"
                                    },
                                    "features": {
                                        "type": "object",
                                        "properties": {
                                            "generalTrends": {
                                                "type": "object",
                                                "properties": {
                                                    "trendDirection": {
                                                        "type": "string",
                                                        "enum": ["up", "down", "sideways"],
                                                        "description": "Overall trend direction of the asset"
                                                    },
                                                    "trendStrength": {
                                                        "type": "string",
                                                        "enum": ["weak", "moderate", "strong"],
                                                        "description": "Strength of the detected trend"
                                                    },
                                                    "volume": {
                                                        "type": "string",
                                                        "enum": ["low", "medium", "high"],
                                                        "description": "Relative trading volume of the asset"
                                                    },
                                                    "volatility": {
                                                        "type": "string",
                                                        "enum": ["low", "medium", "high"],
                                                        "description": "Current market volatility."
                                                    },
                                                    "analysis": {
                                                        "type": "string",
                                                        "description": "Start with a sentence that includes the ticker name and the current price. Then assess the current market structure. Identify whether the price is trending, consolidating, or reversing. Evaluate how the current price action fits within the larger trend. Identify signs of momentum exhaustion or build up and explain the reasoning behind it. Analyze volatility expansion or contraction. Highlight key liquidity zones where price is likely to see significant reactions. Identify Smart Money concepts such as previous swing highs/lows, order blocks, or fair value gaps. Explain where traders might look for reversals, liquidity grabs, or trend continuations. This analysis should be thorough and insightful."
                                                    }
                                                },
                                                "required": ["trendDirection", "trendStrength", "volume", "volatility", "analysis"],
                                                "additionalProperties": False
                                            },
                                            "supportResistance": {
                                                "type": "object",
                                                "properties": {
                                                    "supportLevels": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                        "description": "List of price points (1-2) representing support levels present in input image."
                                                    },
                                                    "resistanceLevels": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                        "description": "List of price points (1-2) representing resistance levels present in input image"
                                                    },
                                                    "analysis": {
                                                        "type": "string",
                                                        "description": "Only provide analysis if support & resistance levels are clearly drawn on the chart or can be inferred from market structure. DO NOT USE THE CURRENT PRICE LINE AS A SUPPORT OR RESISTANCE LEVEL. Identify historical price reaction zones, including swing highs, swing lows, etc. that align with current price action. Assess whether price is approaching strong resistance (selling pressure) or strong support (buying demand), or neither. Explain whether these levels are being tested repeatedly, strengthening their validity, or if they are likely to break due to declining reaction strength. Provide ACCURATE numbers for support & resistance levels."
                                                    }
                                                },
                                                "required": ["supportLevels", "resistanceLevels", "analysis"],
                                                "additionalProperties": False
                                            },
                                            "candlestickPatterns": {
                                                "type": "object",
                                                "properties": {
                                                    "recognizedPatterns": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "patternName": {
                                                                    "type": "string",
                                                                    "description": "Name of the recognized candlestick pattern."
                                                                },
                                                                "analysis": {
                                                                    "type": "string",
                                                                    "description": "Identify all candlestick patterns visible in the uploaded chart. For each pattern, describe its structure and explain what it typically signals in terms of market direction - whether reversal, continuation, or indecision. Mention the context in which each pattern appears, such as its location relative to recent price action or trend. If multiple patterns form in close proximity, note any significant clusters or sequences that may suggest a stronger shift in market sentiment."
                                                                }
                                                            },
                                                            "required": ["patternName", "analysis"],
                                                            "additionalProperties": False
                                                        }
                                                    }
                                                },
                                                "required": ["recognizedPatterns"],
                                                "additionalProperties": False
                                            },
                                            "indicatorAnalyses": {
                                                "type": "object",
                                                "properties": {
                                                    "selectedIndicators": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "indicatorName": {
                                                                    "type": "string",
                                                                    "description": "Type of indicator present in the input image"
                                                                },
                                                                "analysis": {
                                                                    "type": "string",
                                                                    "description": "Identify all technical indicators present in the uploaded chart. For each indicator, identify key signals such as overbought/oversold conditions, trend strength or divergence, etc. Note how these align with the current price action. Additionally, assess how the indicators interact when combined, determining if they confirm or reinforce each others' signals for a more robust view of market sentiment."
                                                                }
                                                            },
                                                            "required": ["indicatorName", "analysis"],
                                                            "additionalProperties": False
                                                        }
                                                    }
                                                },
                                                "required": ["selectedIndicators"],
                                                "additionalProperties": False
                                            },
                                            "futureMarketPrediction": {
                                                "type": "object",
                                                "properties": {
                                                    "timeHorizon": {
                                                        "type": "string",
                                                        "enum": ["short_term", "medium_term", "long_term"],
                                                        "description": "Timeframe for the market prediction"
                                                    },
                                                    "analysis": {
                                                        "type": "string",
                                                        "description": "Identify the timeframe (i.e. daily, hourly, weekly) in the uploaded chart. Choose a time horizon (short, medium, long) that offers the strongest confluence with the timeframe. Then, assess the potential market direction by utilizing the technical indicators, candlestick patterns, liquidity zones, order blocks, fair value gaps, etc. that were previously analyzed. Explain why this prediction makes sense given the context. If you anticipate any breakouts, mention those and explain why using realistic target ranges. If you anticipate and trend reversals, mention those and explain why by pinpointing potential reversal levels."
                                                    }
                                                },
                                                "required": ["timeHorizon", "analysis"],
                                                "additionalProperties": False
                                            },
                                            "potentialTradeSetup": {
                                                "type": "object",
                                                "properties": {
                                                    "entryTargetPrice": {
                                                        "type": "string",
                                                        "description": "Recommended full price price for trade entry"
                                                    },
                                                    "stopLossPrice": {
                                                        "type": "string",
                                                        "description": "Recommended full price for stop loss"
                                                    },
                                                    "analysis": {
                                                        "type": "string",
                                                        "description": "Present a potential trade setup based on the prior analysis. Define a clear entry level and a stop loss. Explain the rationale behind these choices, and why this setup could work based on the previously analyzed technical indicators, candlestick patterns, liquidity zones, order blocks, fair value gaps, etc. Also, describe how traders should adjust stops or scale out based on evolving price action, and provide guidelines on when to take profits. If additional opportunities exist, mention them briefly after outlining the primary setup."
                                                    }
                                                },
                                                "required": ["entryTargetPrice", "stopLossPrice", "analysis"],
                                                "additionalProperties": False
                                            }
                                        },
                                        "required": [
                                            "generalTrends",
                                            "supportResistance",
                                            "candlestickPatterns",
                                            "indicatorAnalyses",
                                            "futureMarketPrediction",
                                            "potentialTradeSetup"
                                        ],
                                        "additionalProperties": False
                                    }
                                },
                                "required": ["ticker", "features"],
                                "additionalProperties": False
                            }
                        },
                        "required": ["status", "result"],
                        "additionalProperties": False
                    },
                    "strict": True
                },
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CANDLESTICK_OPENAI_API_KEY}"
        }

        response = requests.post(OPENAI_BASE_URL, json=payload, headers=headers, timeout=30)
        if response.status_code != 200:
            return jsonify({"error": "OpenAI API error", "details": response.text}), 500

        openai_json = response.json()
        if (
            "choices" in openai_json and
            len(openai_json["choices"]) > 0 and
            "message" in openai_json["choices"][0] and
            "content" in openai_json["choices"][0]["message"]
        ):
            content_str = openai_json["choices"][0]["message"]["content"]
            try:
                parsed_content = json.loads(content_str)
            except json.JSONDecodeError:
                return jsonify({
                    "error": "Could not parse JSON from OpenAI response",
                    "raw_content": content_str
                }), 500

            if "status" not in parsed_content:
                return jsonify({
                    "error": "Missing 'status' field in parsed content",
                    "parsed_content": parsed_content
                }), 500

            if parsed_content["status"]:
                if "result" not in parsed_content:
                    return jsonify({
                        "error": "Missing 'result' field in parsed content",
                        "parsed_content": parsed_content
                    }), 500
                return jsonify(parsed_content["result"]), 200
            else:
                return jsonify({
                    "error": "The provided image is not a valid trading chart"
                }), 400

        else:
            return jsonify({
                "error": "Invalid response format from OpenAI",
                "openai_json": openai_json
            }), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/getArticles", methods=["POST"])
def get_articles():
    try:
        data = request.get_json()
        if not data or "userPrompt" not in data:
            return jsonify({"error": "Missing 'userPrompt' in JSON body."}), 400

        user_prompt = data["userPrompt"]
        prompt = get_txt_file(PERPLEXITY_PROMPT_FILE_PATH)

        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "article_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "articles": {
                                "type": "array",
                                "description": "List of relevant news articles",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {
                                            "type": "string",
                                            "description": "The headline of the news article"
                                        },
                                        "summary": {
                                            "type": "string",
                                            "description": "A brief summary explaining the article's relevance to the chart analysis"
                                        },
                                        "link": {
                                            "type": "string",
                                            "description": "The link to the news article"
                                        }
                                    },
                                    "required": ["title", "summary", "link"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["articles"],
                        "additionalProperties": False
                    }
                }
            },
            "temperature": 0.0,
            "top_p": 0.9,
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "month",
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1
        }
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(PERPLEXITY_BASE_URL, json=payload, headers=headers, timeout=30)
        if response.status_code != 200:
            return jsonify({"error": "Perplexity API error", "details": response.text}), 500

        response_json = response.json()
        if ("choices" in response_json and
            isinstance(response_json["choices"], list) and
            len(response_json["choices"]) > 0 and
            "message" in response_json["choices"][0] and
            "content" in response_json["choices"][0]["message"]):

            content_str = response_json["choices"][0]["message"]["content"]

            cleaned_content = content_str.strip()
            cleaned_content = cleaned_content.replace("json", "")
            cleaned_content = cleaned_content.replace("`", "")

            try:
                parsed = json.loads(cleaned_content)
                return jsonify(parsed), 200
            except json.JSONDecodeError:
                return jsonify({
                    "error": "Could not parse JSON from Perplexity response",
                    "raw_content": content_str
                }), 500
        else:
            return jsonify({
                "error": "Invalid response format from Perplexity",
                "raw_response": response_json
            }), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/getResponses", methods=["POST"])
def generate_response():
    try:
        data = request.get_json()
        if not data or "base64Image" not in data:
            return jsonify({"error": "Missing 'base64Image' in request JSON."}), 400

        instagram_handle = data["instagramHandle"]
        description = data["description"]
        base64_image = data["base64Image"]

        if len(instagram_handle) > 0:
            base64_image_str, display_name = get_instagram_profile_pic_and_name(instagram_handle=instagram_handle)

        rizz_prompt = get_txt_file(RIZZ_PROMPT_FILE_PATH)

        if len(display_name) > 0:
            rizz_prompt += f"\nBelow is user's description of their situationship:\n{description} with {display_name}"
        else:
            rizz_prompt += f"\nBelow is user's description of their situationship:\n{description}"

        payload = {
            "model": "gpt-4o-mini-2024-07-18",
            "temperature": 0.8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": rizz_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "decode_situationship",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "responses": {
                                "type": "array",
                                "description": "List of exactly 4 categorized convo responses",
                                "minItems": 4,
                                "maxItems": 4,
                                "items": {
                                    "type": "object",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "description": "Text message response pulled from the conversation"
                                            },
                                            "category": {
                                                "type": "string",
                                                "enum": ["rizz", "nsfw", "romantic", "end it"],
                                                "description": "Category that best describes the tone or intent of the response"
                                            }
                                        },
                                        "required": ["text", "category"],
                                        "additionalProperties": False
                                }
                            },
                            "interestLevel": {
                                "type": "number",
                                "description": "Score from 0 to 10 indicating how interested the other person seems based on message tone, effort, and engagement"
                            },
                            "redFlags": {
                                "type": "string",
                                "description": "A 3–4 sentence summary highlighting the most concerning behaviors or signals in the conversation, such as breadcrumbing, lovebombing, mixed signals, emotional unavailability"
                            },
                            "greenFlags": {
                                "type": "string",
                                "description": "A 3–4 sentence summary highlighting positive behaviors in the conversation, such as signs of real interest, emotional availability, consistency"
                            }
                        },
                        "required": ["responses", "interestLevel", "redFlags", "greenFlags"],
                        "additionalProperties": False
                    }
                }
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RIZZ_OPENAI_API_KEY}"
        }

        resp = requests.post(OPENAI_BASE_URL, json=payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            return jsonify({"error": resp.text}), 500

        response_data = resp.json()
        if (
            "choices" in response_data and
            len(response_data["choices"]) > 0 and
            "message" in response_data["choices"][0] and
            "content" in response_data["choices"][0]["message"]
        ):
            raw_content = response_data["choices"][0]["message"]["content"]

            try:
                parsed_responses = json.loads(raw_content)
                if base64_image_str:
                    parsed_responses['profile_image_base64'] = base64_image_str
                    parsed_responses['display_name'] = display_name

                return jsonify(parsed_responses), 200
            except json.JSONDecodeError:
                return jsonify({
                    "error": "Could not parse JSON from model response",
                    "raw_content": raw_content
                }), 500
        else:
            return jsonify({
                "error": "Invalid structure in response",
                "response_data": response_data
            }), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
