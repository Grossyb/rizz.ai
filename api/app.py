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


@app.route("/getChartAnalysis", methods=["POST"])
def get_chart_analysis():
    try:
        data = request.get_json()
        if not data or "base64Image" not in data:
            return jsonify({"error": "Missing 'base64Image' in JSON body."}), 400

        base64_image = data["base64Image"]
        prompt = get_txt_file(OPENAI_PROMPT_FILE_PATH)

        trading_styles = ", ".join(data.get("tradingStyles", ["Not Sure"]))
        risk = data.get("risk", "Not Sure")

        prompt = f"User trading style(s): {trading_styles}. User risk preference: {risk}.\n\n{prompt}"
        app.logger.debug(prompt)

        payload = {
            "model": "gpt-5-mini",
            "temperature": 0.6,
            "top_p": 0.9,
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
                                                        "description": "Start with a sentence that includes the ticker name and the current price (if legible, otherwise say not legible). Then assess the current market structure. Identify whether the price is trending, consolidating, or reversing. Evaluate how the current price action fits within the larger trend. Identify signs of momentum exhaustion or build up and explain the reasoning behind it. Analyze volatility expansion or contraction. Highlight key liquidity zones where price is likely to see significant reactions. Mention Smart Money concepts (previous swing highs/lows, order blocks, fair value gaps) only if clearly visible on the chart."
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
                                                        "description": "Only provide analysis if support & resistance levels are clearly drawn or confidently inferred from visible market structure. Do not use the current price line as a support or resistance level. Provide accurate numbers only if legible; otherwise leave supportLevels and resistanceLevels arrays empty. Explain if reactions are strengthening or weakening."
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
                                                                    "description": "Identify candlestick patterns only if their full structure is clearly visible. For each, describe its structure and typical implication in context. If no clear patterns are visible, return an empty recognizedPatterns array."
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
                                                                    "description": "Only include indicators that are visibly present on the chart (e.g., RSI, MACD, moving averages, VWAP, Bollinger Bands). If no indicators are visible, return an empty selectedIndicators array. For each indicator, describe its signal in context and end with a"
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
                                                        "description": "Identify the timeframe if it is clearly visible; if not, state 'timeframe uncertain.' Choose a time horizon (short, medium, long) consistent with the chart. Assess likely market direction using visible price action, candlestick patterns, liquidity zones, order blocks, or fair value gaps. Mention breakouts or reversals only if supported by what is on the chart."
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
                                                        "description": "Present one potential trade setup aligned to the user's trading style(s) and risk if provided, otherwise default to a balanced setup. Define a clear entry and stop loss only if exact prices are legible; otherwise describe them relative to visible structures (e.g., below swing low, at order block edge). Explain rationale, risk management, and profit taking. If timeframe mismatches the user's style, note that."
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

        response = requests.post(OPENAI_BASE_URL, json=payload, headers=headers, timeout=60)
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

        response = requests.post(PERPLEXITY_BASE_URL, json=payload, headers=headers, timeout=60)
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

        description = data["description"]
        name = data["name"]
        base64_image = data["base64Image"]

        rizz_prompt = get_txt_file(RIZZ_PROMPT_FILE_PATH)

        if len(name) > 0:
            rizz_prompt += f"\nBelow is user's description of their situationship:\n{description} with {name}"
        else:
            rizz_prompt += f"\nBelow is user's description of their situationship:\n{description}"

        payload = {
            "model": "gpt-5-mini",
            "temperature": 0.7,
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
                            "breakdown": {
                                "type": "string",
                                "description": "Analyze the screenshot and user input describing the situationship. Identify emotional cues, contradictions, message tone, ghosting patterns, power dynamics, or mismatched effort. Summarize what’s really going on in a concise paragraph and give a direct recommendation (e.g., keep going, cut it off, call it out, etc.)."
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
                        "required": ["responses", "interestLevel", "breakdown", "redFlags", "greenFlags"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RIZZ_OPENAI_API_KEY}"
        }

        resp = requests.post(OPENAI_BASE_URL, json=payload, headers=headers, timeout=60)
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
