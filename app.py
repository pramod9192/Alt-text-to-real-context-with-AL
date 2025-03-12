from flask import Flask, render_template, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io
import requests
from google import genai

app = Flask(__name__)

# Load the processors and models for different types of images
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_model.to(device)

# Gemini API client
gemini_api_key = "AIzaSyDmRC3KHctpbqyErbiLbisp9VUeD5Qrvkg"
client = genai.Client(api_key=gemini_api_key)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    
    # Determine the type of image based on a parameter or heuristic (for simplicity, using a string input)
    image_type = request.form.get('image_type', 'general')

    if image_type == 'medical':
        # Medical-specific image captioning model (using the same BLIP model for now)
        processor = blip_processor
        model = blip_model
    elif image_type == 'sports':
        # Placeholder for a sports-specific model, can be changed to any other model (e.g., fine-tuned for sports images)
        processor = blip_processor
        model = blip_model
    else:
        # Default to general image captioning model
        processor = blip_processor
        model = blip_model
    
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Run the model to generate the caption
    outputs = model.generate(**inputs)

    # Decode the caption
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    # Call Gemini API to process the caption
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=caption
    )

    result = response.text
    return jsonify({'caption': caption, 'gemini_response': result})

if __name__ == '__main__':
    app.run(debug=True)
