from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
from pytesseract import Output
import cv2
import re
import openai

app = Flask(__name__)

open_ai_key = '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'
openai.api_key = open_ai_key

def format_conversation(conversation):
    formatted_conversation = ""
    for message in conversation:
        if message['sender'] == "you":
            formatted_conversation += "Person #2: {}\n".format(message['text'])
        else:
            formatted_conversation += "Person #1: {}\n".format(message['text'])
    return formatted_conversation

def generate_prompt(messages, tone):
    prompt = "You are an AI assistant helping with Tinder and Bumble conversations. You will generate witty responses for Person #2, a young adult who is looking to attract a partner. Ignore any text messages that do not look like typical text messages, such as messages that might be cell service info or labels from a chat UI."
    for message in messages:
        if message['sender'] == "you":
            prompt += "Person #2: {}\n".format(message['text'])
        else:
            prompt += "Person #1: {}\n".format(message['text'])

    prompt += "\nGenerate a response for Person #2 that is {}. Be clever yet concise in your response. The response should based on the conversation context and tone, and should be suitable for Person #2 to send in the conversation.".format(tone)
    return prompt

def check_word(text):
    m = re.match("\w+[!?,().]+$", text)
    if m is not None:
        return True
    return False

@app.route('/extract_text', methods=['POST'])
def extract_text():
    """
    Extract text from an image using Tesseract OCR via a Flask endpoint.

    Expects a POST request with an image file attached.

    Returns:
        JSON: Extracted text from the image.
    """
    pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
    try:
        # Get the uploaded file from the request
        image_file = request.files['image']

        # Open the image
        image = Image.open(image_file)

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
            app.logger.debug('TEXT: {}, COORD: {}, MEAN: {}'.format(texts[i], x_coords[i], x_mean))
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
        {"role": "system", "content": "You are an AI assistant helping with Tinder and Bumble conversations. You will generate witty responses for Person #2, a young adult who is looking to attract a partner. Ignore any text messages that do not look like typical text messages, such as messages that might be cell service info or labels from a chat UI."},
        {"role": "system", "content": format_conversation(conversation)},
        {"role": "system", "content": "Generate four witty and unique responses for Person #2 that are sarcastic or flirtatious, and make sense as the next text in the conversation. Keep the responses a sentence or shorter and do not include parentheses."},
        {"role": "system", "content": "Begin each response with a --"}
    ]
    responses = []
    text = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages, temperature=0.7, max_tokens=3500)['choices'][0]['message']['content']
    texts = text.split('--')
    for i in range(len(texts)):
        response = {}
        response['id'] = i
        response['text'] = texts[i].strip()
        if len(response['text']) > 0:
            responses.append(response)
    app.logger.debug(responses)
    return jsonify(responses), 200

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1',port=4040)