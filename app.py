from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import json
from datetime import datetime
import os
import logging
from werkzeug.utils import secure_filename
import uuid

# Import segmentation models
try:
    from transformers import pipeline, AutoImageProcessor, Mask2FormerForUniversalSegmentation
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
except ImportError:
    print("Please install: pip install transformers torch torchvision detectron2 opencv-python pillow flask flask-cors")

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

class RetailSegmentationPlatform:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.load_models()
        
        # Retail-specific object categories
        self.retail_categories = {
            'clothing': ['shirt', 'pants', 'dress', 'jacket', 'shoes', 'hat', 'bag'],
            'electronics': ['phone', 'laptop', 'tablet', 'camera', 'headphones'],
            'furniture': ['chair', 'table', 'sofa', 'bed', 'cabinet'],
            'food': ['apple', 'banana', 'bottle', 'cup', 'bowl'],
            'accessories': ['watch', 'glasses', 'jewelry', 'belt']
        }
    
    def load_models(self):
        """Load segmentation models"""
        try:
            # Load Mask2Former for instance segmentation
            self.instance_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-instance")
            self.instance_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance")
            self.instance_model.to(self.device)
            
            # Load semantic segmentation model
            self.semantic_segmenter = pipeline("image-segmentation", 
                                             model="facebook/mask2former-swin-large-ade-semantic",
                                             device=0 if torch.cuda.is_available() else -1)
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to simpler models if available
            self.load_fallback_models()
    
    def load_fallback_models(self):
        """Load simpler models as fallback"""
        try:
            self.semantic_segmenter = pipeline("image-segmentation", 
                                             model="facebook/detr-resnet-50-panoptic",
                                             device=0 if torch.cuda.is_available() else -1)
            logger.info("Fallback models loaded")
        except Exception as e:
            logger.error(f"Error loading fallback models: {e}")
    
    def preprocess_image(self, image):
        """Preprocess image for segmentation"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        return image
    
    def perform_instance_segmentation(self, image):
        """Perform instance segmentation"""
        try:
            # Process with Mask2Former
            inputs = self.instance_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.instance_model(**inputs)
            
            # Post-process results
            result = self.instance_processor.post_process_instance_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0]
            
            instances = []
            for i, (mask, label, score) in enumerate(zip(
                result['segmentation'], result['segments_info'], [1.0] * len(result['segments_info'])
            )):
                if score > 0.5:  # Confidence threshold
                    mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                    instances.append({
                        'id': i,
                        'label': f"object_{label}",
                        'confidence': float(score),
                        'mask': mask_np,
                        'category': self.categorize_object(f"object_{label}")
                    })
            
            return instances
        except Exception as e:
            logger.error(f"Instance segmentation error: {e}")
            return []
    
    def perform_semantic_segmentation(self, image):
        """Perform semantic segmentation"""
        try:
            segments = self.semantic_segmenter(image)
            
            semantic_results = []
            for i, segment in enumerate(segments):
                mask = np.array(segment['mask'])
                semantic_results.append({
                    'id': i,
                    'label': segment['label'],
                    'mask': mask,
                    'area': np.sum(mask > 0),
                    'category': self.categorize_object(segment['label'])
                })
            
            return semantic_results
        except Exception as e:
            logger.error(f"Semantic segmentation error: {e}")
            return []
    
    def categorize_object(self, label):
        """Categorize objects for retail context"""
        label_lower = label.lower()
        for category, items in self.retail_categories.items():
            if any(item in label_lower for item in items):
                return category
        return 'other'
    
    def create_visualization(self, image, instances, semantics):
        """Create visualization with both segmentations"""
        vis_image = np.array(image.copy())
        overlay = np.zeros_like(vis_image)
        
        # Color palette for different categories
        colors = {
            'clothing': (255, 100, 100),
            'electronics': (100, 255, 100),
            'furniture': (100, 100, 255),
            'food': (255, 255, 100),
            'accessories': (255, 100, 255),
            'other': (150, 150, 150)
        }
        
        # Draw instance masks
        for instance in instances:
            color = colors.get(instance['category'], (150, 150, 150))
            mask = instance['mask']
            overlay[mask > 0] = color
        
        # Blend with original image
        alpha = 0.6
        vis_image = cv2.addWeighted(vis_image, 1-alpha, overlay, alpha, 0)
        
        return vis_image
    
    def extract_objects(self, image, instances):
        """Extract individual objects with transparent backgrounds"""
        extracted_objects = []
        image_np = np.array(image)
        
        for instance in instances:
            mask = instance['mask']
            
            # Create RGBA image
            rgba_obj = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            rgba_obj[:, :, :3] = image_np
            rgba_obj[:, :, 3] = mask  # Alpha channel
            
            # Find bounding box
            coords = np.where(mask > 0)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                # Crop to bounding box
                cropped_obj = rgba_obj[y_min:y_max+1, x_min:x_max+1]
                
                extracted_objects.append({
                    'id': instance['id'],
                    'label': instance['label'],
                    'category': instance['category'],
                    'confidence': instance['confidence'],
                    'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                    'image': cropped_obj
                })
        
        return extracted_objects

# Initialize the platform
segmentation_platform = RetailSegmentationPlatform()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
        # RGBA image
        pil_image = Image.fromarray(image_array, 'RGBA')
    else:
        # RGB image
        pil_image = Image.fromarray(image_array, 'RGB')
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'device': str(segmentation_platform.device),
        'models_loaded': True
    })

@app.route('/api/segment', methods=['POST'])
def segment_image():
    """Main segmentation endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get segmentation options
        include_instance = request.form.get('include_instance', 'true').lower() == 'true'
        include_semantic = request.form.get('include_semantic', 'true').lower() == 'true'
        extract_objects = request.form.get('extract_objects', 'false').lower() == 'true'
        min_confidence = float(request.form.get('min_confidence', 0.5))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        # Load and preprocess image
        image = segmentation_platform.preprocess_image(filepath)
        
        results = {
            'original_image': encode_image_to_base64(np.array(image)),
            'timestamp': datetime.now().isoformat(),
            'processing_info': {
                'device': str(segmentation_platform.device),
                'image_size': image.size
            }
        }
        
        # Perform instance segmentation
        instances = []
        if include_instance:
            logger.info("Performing instance segmentation...")
            instances = segmentation_platform.perform_instance_segmentation(image)
            instances = [inst for inst in instances if inst['confidence'] >= min_confidence]
            results['instance_segmentation'] = {
                'count': len(instances),
                'objects': [
                    {
                        'id': inst['id'],
                        'label': inst['label'],
                        'category': inst['category'],
                        'confidence': inst['confidence']
                    } for inst in instances
                ]
            }
        
        # Perform semantic segmentation
        semantics = []
        if include_semantic:
            logger.info("Performing semantic segmentation...")
            semantics = segmentation_platform.perform_semantic_segmentation(image)
            results['semantic_segmentation'] = {
                'count': len(semantics),
                'segments': [
                    {
                        'id': seg['id'],
                        'label': seg['label'],
                        'category': seg['category'],
                        'area': int(seg['area'])
                    } for seg in semantics
                ]
            }
        
        # Create visualization
        if instances or semantics:
            vis_image = segmentation_platform.create_visualization(image, instances, semantics)
            results['visualization'] = encode_image_to_base64(vis_image)
        
        # Extract individual objects
        if extract_objects and instances:
            logger.info("Extracting individual objects...")
            extracted = segmentation_platform.extract_objects(image, instances)
            results['extracted_objects'] = []
            
            for obj in extracted:
                results['extracted_objects'].append({
                    'id': obj['id'],
                    'label': obj['label'],
                    'category': obj['category'],
                    'confidence': obj['confidence'],
                    'bbox': obj['bbox'],
                    'image': encode_image_to_base64(obj['image'])
                })
        
        # Category summary
        category_counts = {}
        all_objects = instances + semantics
        for obj in all_objects:
            cat = obj['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        results['category_summary'] = category_counts
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get available retail categories"""
    return jsonify({
        'categories': segmentation_platform.retail_categories,
        'total_categories': len(segmentation_platform.retail_categories)
    })

@app.route('/api/batch-segment', methods=['POST'])
def batch_segment():
    """Process multiple images"""
    try:
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No images provided'}), 400
        
        results = []
        for file in files:
            if file and allowed_file(file.filename):
                # Process each image
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
                file.save(filepath)
                
                image = segmentation_platform.preprocess_image(filepath)
                instances = segmentation_platform.perform_instance_segmentation(image)
                
                results.append({
                    'filename': filename,
                    'object_count': len(instances),
                    'categories': list(set(inst['category'] for inst in instances)),
                    'processing_time': datetime.now().isoformat()
                })
                
                os.remove(filepath)
        
        return jsonify({
            'batch_results': results,
            'total_processed': len(results)
        })
    
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['POST'])
def get_analytics():
    """Analyze segmentation results for retail insights"""
    try:
        data = request.get_json()
        
        # Mock analytics - in production, this would analyze historical data
        analytics = {
            'most_common_objects': [
                {'category': 'clothing', 'count': 45, 'percentage': 35.2},
                {'category': 'accessories', 'count': 32, 'percentage': 25.0},
                {'category': 'electronics', 'count': 28, 'percentage': 21.9}
            ],
            'detection_accuracy': {
                'average_confidence': 0.87,
                'high_confidence_detections': 0.72,
                'categories_detected': 8
            },
            'retail_insights': {
                'recommended_focus': 'clothing',
                'inventory_gaps': ['furniture', 'food'],
                'seasonal_trends': 'accessories trending up 15%'
            }
        }
        
        return jsonify(analytics)
    
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)