from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import pytesseract
from pytesseract import Output
import cv2
import re
import os
import openai

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

def format_conversation(conversation):
    formatted_conversation = ""
    for message in conversation:
        if message['sender'] == "you":
            formatted_conversation += "Person #2: {}\n".format(message['text'])
        else:
            formatted_conversation += "Person #1: {}\n".format(message['text'])
    return formatted_conversation

def generate_prompt(messages, tone):
    prompt = "You are an AI assistant helping with Tinder and Bumble conversations. You will generate responses for Person #2, a teenager who is looking to attract a partner. Ignore any text messages that do not look like typical text messages, such as messages that might be cell service info or labels from a chat UI."
    for message in messages:
        if message['sender'] == "you":
            prompt += "Person #2: {}\n".format(message['text'])
        else:
            prompt += "Person #1: {}\n".format(message['text'])

    prompt += "\nGenerate a unique, witty, and flirtatious response that includes jokes, puns, sarcasm, or playful banter. The response should sound like the viral conversations we see on dating apps like Tinder and Bumble. Keep the tone lighthearted and playful, and make sure the responses are appropriate for a teenage audience."
    return prompt

def only_special(text):
    count = 0
    for char in text:
        if not char.isalpha():
            count += 1

    return True if count == len(text) else False

def is_valid_text(text):
    if len(text) == 0:
        return False
    if text.isspace():
        return False
    elif 'iMessage' in text:
        return False
    elif '@' in text:
        return False
    elif only_special(text):
        return False
    else:
        return True

@app.route('/extract_text', methods=['POST'])
def extract_text():
    """
    Extract text from an image using Tesseract OCR via a Flask endpoint.

    Expects a POST request with an image file attached.

    Returns:
        JSON: Extracted text from the image.
    """
    try:
        # Get the uploaded file from the request
        image_file = request.files['image']

        # Open the image
        image = Image.open(image_file)
        image = ImageOps.grayscale(image)

        # Perform OCR using Tesseract
        d = pytesseract.image_to_data(image, output_type=Output.DICT)
        n_boxes = len(d['level'])
        x_coords = []
        texts = []
        curr_block = d['block_num'][0]
        curr_text = d['text'][0]
        curr_x = d['left'][0]
        for i in range(1, n_boxes):
            if d['block_num'][i] != curr_block:
                # next block started
                x = d['left'][i]
                x_coords.append(curr_x)
                texts.append(curr_text)
                curr_text = ""
                curr_x = d['left'][i]
                curr_block = d['block_num'][i]
            else:
                text = d['text'][i]
                curr_text += " " + d['text'][i]

        x_mean = sum(x_coords) / len(x_coords)
        n_texts = len(texts)
        messages = []
        for i in range(n_texts):
            message = {}
            message['id'] = i
            message['text'] = texts[i].strip()
            message['sender'] = 'you' if x_coords[i] > x_mean else 'other'
            if len(message['text'].replace(' ', '')) == 0:
                continue
            elif message['text'] == 'Sent' or message['text'] == 'Delivered':
                continue
            else:
                messages.append(message)

        # Return the extracted text as JSON
        return jsonify(messages), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

@app.route('/generate_responses', methods=['POST'])
def generate_responses():
    conversation = request.json['conversation']
    # prompts = []
    # prompt_1 = generate_prompt(messages=messages, tone="mysterious")
    # prompt_2 = generate_prompt(messages=messages, tone="flirtatious")
    # prompt_3 = generate_prompt(messages=messages, tone="funny and humorous. Surprise Person #1 with a clever comeback or joke")
    # prompt_4 = generate_prompt(messages=messages, tone="teasing")
    #
    # prompts.append(prompt_1)
    # prompts.append(prompt_2)
    # prompts.append(prompt_3)
    # prompts.append(prompt_4)

    messages=[
        {"role": "system", "content": "You are an AI assistant helping with Tinder and Bumble conversations. You will generate witty responses for Person #2, a teenager who would like to date Person #1. Ignore any text messages that do not look like typical text messages, such as messages that might be cell service info or labels from a chat UI."},
        {"role": "system", "content": format_conversation(conversation)},
        {"role": "system", "content": "Generate four unique, witty, and flirtatious responses that include jokes, sarcasm, and playful banter. The responses should sound like the viral conversations we see on dating apps like Tinder and Bumble. Each response should be a sentence or shorter and don't be too specific in your wording."},
        {"role": "system", "content": "Begin each response with a --"}
    ]
    responses = []
    text = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages, temperature=0.9, max_tokens=3500)['choices'][0]['message']['content']
    texts = text.split('--')
    for i in range(len(texts)):
        response = {}
        response['id'] = i
        response['text'] = texts[i].strip()
        if len(response['text']) > 0 and is_valid_text(response['text']):
            app.logger.debug(response['text'])
            responses.append(response)
    return jsonify(responses), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
