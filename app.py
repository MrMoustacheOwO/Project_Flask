from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import io
import os

app = Flask(__name__)

# Helper function to decode base64 image
def decode_image(image_data):
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None

# Helper function to encode image to base64
def encode_image(img):
    try:
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f'data:image/png;base64,{img_base64}'
    except Exception as e:
        return None

@app.route('/')
def index():
    return render_template('editor.html')

@app.route('/api/resize', methods=['POST'])
def resize():
    try:
        data = request.get_json()
        img = decode_image(data['image'])
        
        if img is None:
            return jsonify({'success': False, 'message': 'Ошибка декодирования изображения'})
        
        width = int(data.get('width', 800))
        height = int(data.get('height', 600))
        interpolation = data.get('interpolation', 'linear')
        
        # Choose interpolation method
        interp_method = cv2.INTER_LINEAR  # default
        if interpolation == 'nearest':
            interp_method = cv2.INTER_NEAREST
        elif interpolation == 'cubic':
            interp_method = cv2.INTER_CUBIC
        
        resized = cv2.resize(img, (width, height), interpolation=interp_method)
        result = encode_image(resized)
        
        return jsonify({'success': True, 'image': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/crop', methods=['POST'])
def crop():
    try:
        data = request.get_json()
        img = decode_image(data['image'])
        
        if img is None:
            return jsonify({'success': False, 'message': 'Ошибка декодирования изображения'})
        
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        width = int(data.get('width', 100))
        height = int(data.get('height', 100))
        
        # Check boundaries
        img_height, img_width = img.shape[:2]
        
        if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
            return jsonify({'success': False, 'message': 'Координаты вырезки выходят за границы изображения'})
        
        cropped = img[y:y+height, x:x+width]
        result = encode_image(cropped)
        
        return jsonify({'success': True, 'image': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/flip', methods=['POST'])
def flip():
    try:
        data = request.get_json()
        img = decode_image(data['image'])
        
        if img is None:
            return jsonify({'success': False, 'message': 'Ошибка декодирования изображения'})
        
        direction = data.get('direction', 'horizontal')
        
        if direction == 'horizontal':
            flipped = cv2.flip(img, 1)  # Flip horizontally
        elif direction == 'vertical':
            flipped = cv2.flip(img, 0)  # Flip vertically
        elif direction == 'both':
            flipped = cv2.flip(img, -1)  # Flip both
        else:
            return jsonify({'success': False, 'message': 'Неизвестное направление отражения'})
        
        result = encode_image(flipped)
        return jsonify({'success': True, 'image': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/rotate', methods=['POST'])
def rotate():
    try:
        data = request.get_json()
        img = decode_image(data['image'])
        
        if img is None:
            return jsonify({'success': False, 'message': 'Ошибка декодирования изображения'})
        
        angle = float(data.get('angle', 0))
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation with bilinear interpolation
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        result = encode_image(rotated)
        return jsonify({'success': True, 'image': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/brightness-contrast', methods=['POST'])
def brightness_contrast():
    try:
        data = request.get_json()
        img = decode_image(data['image'])
        
        if img is None:
            return jsonify({'success': False, 'message': 'Ошибка декодирования изображения'})
        
        brightness = int(data.get('brightness', 0))
        contrast = int(data.get('contrast', 0))
        
        # Apply brightness
        if brightness != 0:
            img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)
        
        # Apply contrast
        if contrast != 0:
            alpha = 1 + contrast / 100.0
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        
        result = encode_image(img)
        return jsonify({'success': True, 'image': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/color-balance', methods=['POST'])
def color_balance():
    try:
        data = request.get_json()
        img = decode_image(data['image'])
        
        if img is None:
            return jsonify({'success': False, 'message': 'Ошибка декодирования изображения'})
        
        red = int(data.get('red', 0))
        green = int(data.get('green', 0))
        blue = int(data.get('blue', 0))
        
        # Split channels
        b, g, r = cv2.split(img)
        
        # Adjust channels
        if red != 0:
            r = cv2.convertScaleAbs(r, alpha=1, beta=red)
        if green != 0:
            g = cv2.convertScaleAbs(g, alpha=1, beta=green)
        if blue != 0:
            b = cv2.convertScaleAbs(b, alpha=1, beta=blue)
        
        # Merge channels
        result_img = cv2.merge([b, g, r])
        result = encode_image(result_img)
        
        return jsonify({'success': True, 'image': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/noise', methods=['POST'])
def add_noise():
    try:
        data = request.get_json()
        img = decode_image(data['image'])
        
        if img is None:
            return jsonify({'success': False, 'message': 'Ошибка декодирования изображения'})
        
        noise_type = data.get('type', 'gaussian')
        img_float = img.astype(np.float32) / 255.0
        
        if noise_type == 'gaussian':
            # Add Gaussian noise
            noise = np.random.normal(0, 0.05, img_float.shape)
            noisy = img_float + noise
        elif noise_type == 'salt_pepper':
            # Add salt and pepper noise
            noisy = img_float.copy()
            s_vs_p = 0.5
            amount = 0.05
            
            # Salt
            num_salt = np.ceil(amount * img_float.size * s_vs_p)
            coords = [np.random.randint(0, i, int(num_salt)) for i in img_float.shape]
            noisy[tuple(coords)] = 1
            
            # Pepper
            num_pepper = np.ceil(amount * img_float.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i, int(num_pepper)) for i in img_float.shape]
            noisy[tuple(coords)] = 0
        else:
            return jsonify({'success': False, 'message': 'Неизвестный тип шума'})
        
        # Clip values and convert back
        noisy = np.clip(noisy, 0, 1) * 255
        result_img = noisy.astype(np.uint8)
        
        result = encode_image(result_img)
        return jsonify({'success': True, 'image': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/blur', methods=['POST'])
def blur():
    try:
        data = request.get_json()
        img = decode_image(data['image'])
        
        if img is None:
            return jsonify({'success': False, 'message': 'Ошибка декодирования изображения'})
        
        blur_type = data.get('type', 'gaussian')
        kernel = int(data.get('kernel', 5))
        
        # Ensure kernel is odd
        if kernel % 2 == 0:
            kernel += 1
        
        if blur_type == 'average':
            blurred = cv2.blur(img, (kernel, kernel))
        elif blur_type == 'gaussian':
            blurred = cv2.GaussianBlur(img, (kernel, kernel), 0)
        elif blur_type == 'median':
            blurred = cv2.medianBlur(img, kernel)
        else:
            return jsonify({'success': False, 'message': 'Неизвестный тип размытия'})
        
        result = encode_image(blurred)
        return jsonify({'success': True, 'image': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/save', methods=['POST'])
def save_image():
    try:
        data = request.get_json()
        img = decode_image(data['image'])
        
        if img is None:
            return jsonify({'success': False, 'message': 'Ошибка декодирования изображения'}), 400
        
        filename = data.get('filename', 'image')
        file_format = data.get('format', 'png')
        quality = int(data.get('quality', 95))
        
        # Create filename
        full_filename = f"{filename}.{file_format}"
        
        # Prepare output
        output = BytesIO()
        
        if file_format == 'jpg':
            cv2.imwrite(output, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif file_format == 'png':
            cv2.imwrite(output, img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        elif file_format == 'tiff':
            cv2.imwrite(output, img)
        elif file_format == 'bmp':
            cv2.imwrite(output, img)
        else:
            return jsonify({'success': False, 'message': 'Неподдерживаемый формат'}), 400
        
        # Try with PIL Image as fallback
        try:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if file_format == 'jpg':
                pil_img.save(output, format='JPEG', quality=quality)
            elif file_format == 'png':
                pil_img.save(output, format='PNG')
            elif file_format == 'tiff':
                pil_img.save(output, format='TIFF')
            elif file_format == 'bmp':
                pil_img.save(output, format='BMP')
        except:
            pass
        
        output.seek(0)
        return send_file(output, mimetype=f'image/{file_format}', as_attachment=True, download_name=full_filename)
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
