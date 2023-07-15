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

def only_special(text):
    count = 0
    for char in text:
        if not char.isalpha():
            count += 1

    return True if count == len(text) else False


def majority_digits(text):
    num_digits = 0
    text = text.strip()
    for element in text:
        if element.isdigit():
            num_digits += 1
    maj = (len(text) / 4)
    if num_digits >= maj:
        return True
    else:
        return False

def is_gibberish(text):
    num_gibberish = 0
    elements = text.split(' ')
    for element in elements:
        if len(element) < 3:
            num_gibberish += 1
    maj = (len(elements) / 2) + 1
    if num_gibberish >= maj:
        return True
    else:
        return False


def is_valid_text(text):
    if is_gibberish(text):
        return False
    elif '=' in text:
        return False
    elif '>' in text:
        return False
    elif '<' in text:
        return False
    elif 'Tre' in text:
        return False
    elif majority_digits(text):
        return False
    if text.lower() == 'space':
        return False
    if len(text) == 0:
        return False
    if text.isspace():
        return False
    elif text.lower() == 'sent' or text == 'delivered':
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
            if is_valid_text(message['text']):
                app.logger.debug(message['text'])
                messages.append(message)

        # Return the extracted text as JSON
        return jsonify(messages), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

@app.route('/generate_responses', methods=['POST'])
def generate_responses():
    conversation = request.json['conversation']

    messages=[
        {"role": "system", "content": "You are an AI dating assistant that is a master in charming or seducing a potential romantic partner. Your responses are smooth and full of swagger. They are flirtatious and witty yet effortless. Analyze the conversation below and pretend you are Person #2. Do not craft your responses based on anything that might sound like gibberish."},
        {"role": "system", "content": format_conversation(conversation)},
        {"role": "system", "content": "Generate four unique responses to the conversation above that will woo Person #1. Each response one sentence or shorter. The first one should be more playful and teasing, the second flirtatious, the third more joking and witty, and the fourth intriguing. Only use emojis for two of the responses."},
        {"role": "system", "content": "Begin each response with a --"}
    ]
    responses = []
    text = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages, temperature=0.9, max_tokens=3500)['choices'][0]['message']['content']
    texts = text.split('--')
    for i in range(len(texts)):
        response = {}
        response['id'] = i
        response['text'] = texts[i].strip()
        if len(response['text']) > 0:
            responses.append(response)
    return jsonify(responses), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
