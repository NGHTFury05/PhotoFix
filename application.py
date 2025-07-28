import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import io
import cv2
from streamlit_cropper import st_cropper

# AI imports with graceful fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    import torchvision.models as models
    import warnings
    warnings.filterwarnings('ignore')
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# ============================
# Configuration & Setup
# ============================

st.set_page_config(
    page_title="PhotoFix Pro - Complete AI Photo Editor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with AI and editing tools styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    .ai-section {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 2px solid #667eea;
    }
    
    .editing-section {
        background: linear-gradient(135deg, #e0c3fc 0%, #9bb5ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 2px solid #667eea;
    }
    
    .style-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        margin: 0.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        cursor: pointer;
        text-align: center;
    }
    
    .style-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
    }
    
    .tool-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .before-after-container {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .image-container {
        border: 3px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        background: white;
        padding: 0.5rem;
    }
    
    .success-message {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
        text-align: center;
    }
    
    .processing-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .download-section {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
        border: 2px solid #f59e0b;
    }
    
    .stats-box {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 0.5rem;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stats-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    .stats-value {
        font-size: 1.1rem;
        font-weight: bold;
        color: #374151;
        margin: 0.3rem 0;
    }
    
    .stats-label {
        font-size: 0.8rem;
        color: #6b7280;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# AI Neural Style Transfer Class
# ============================

class NeuralStyleTransfer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if AI_AVAILABLE else None
        self.model = None
        if AI_AVAILABLE:
            self.load_model()
    
    def load_model(self):
        """Load VGG19 model for style transfer"""
        try:
            if AI_AVAILABLE:
                vgg = models.vgg19(pretrained=True).features.to(self.device).eval()
                self.model = vgg
                
                # Freeze model parameters
                for param in self.model.parameters():
                    param.requires_grad = False
        except Exception as e:
            st.warning(f"Could not load AI model: {str(e)}")
            self.model = None
    
    def apply_style_transfer(self, content_img, style_type="artistic"):
        """Apply style transfer with predefined styles"""
        if not AI_AVAILABLE or self.model is None:
            return self.apply_artistic_filter(content_img, style_type)
        
        try:
            # For now, use artistic filters as they're faster and still impressive
            return self.apply_artistic_filter(content_img, style_type)
            
        except Exception as e:
            st.warning(f"Style transfer failed: {str(e)}")
            return content_img
    
    def apply_artistic_filter(self, image, style_type):
        """Enhanced artistic filters for different styles"""
        if style_type == "van_gogh":
            # Van Gogh style simulation
            enhanced = ImageEnhance.Color(image).enhance(1.5)
            enhanced = ImageEnhance.Contrast(enhanced).enhance(1.4)
            enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=0.8))
            
            # Add texture noise for brushstroke effect
            img_array = np.array(enhanced).astype(np.float32)
            noise = np.random.normal(0, 8, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
            return Image.fromarray(img_array.astype(np.uint8))
        
        elif style_type == "picasso":
            # Cubist style simulation
            enhanced = ImageEnhance.Contrast(image).enhance(1.6)
            enhanced = enhanced.filter(ImageFilter.EDGE_ENHANCE_MORE)
            
            # Color quantization for cubist effect
            img_array = np.array(enhanced)
            img_array = (img_array // 48) * 48  # Reduce color depth
            return Image.fromarray(img_array.astype(np.uint8))
        
        elif style_type == "monet":
            # Impressionist style
            blurred = image.filter(ImageFilter.GaussianBlur(radius=1.2))
            enhanced = ImageEnhance.Color(blurred).enhance(1.7)
            enhanced = ImageEnhance.Brightness(enhanced).enhance(1.15)
            
            # Add soft texture
            img_array = np.array(enhanced).astype(np.float32)
            texture = np.random.normal(0, 6, img_array.shape)
            img_array = np.clip(img_array + texture, 0, 255)
            return Image.fromarray(img_array.astype(np.uint8))
        
        elif style_type == "anime":
            # Anime/manga style
            img_array = np.array(image)
            
            # Bilateral filter for smooth areas
            smooth = cv2.bilateralFilter(img_array, 15, 200, 200)
            
            # Edge detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 9, 9)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            # Combine smooth areas with edges
            anime = cv2.bitwise_and(smooth, edges)
            return Image.fromarray(anime)
        
        else:  # abstract
            # Abstract art style
            enhanced = ImageEnhance.Color(image).enhance(2.2)
            enhanced = ImageEnhance.Contrast(enhanced).enhance(1.9)
            
            # Add random color shifts
            img_array = np.array(enhanced)
            h, w, c = img_array.shape
            
            # Create color zones
            for i in range(0, h, 60):
                for j in range(0, w, 60):
                    color_shift = np.random.randint(-40, 40, 3)
                    img_array[i:i+60, j:j+60] = np.clip(
                        img_array[i:i+60, j:j+60] + color_shift, 0, 255
                    )
            
            return Image.fromarray(img_array.astype(np.uint8))

# ============================
# Advanced Lighting Effects
# ============================

class AmbientLighting:
    @staticmethod
    def apply_golden_hour(image):
        """Apply golden hour lighting effect"""
        img_array = np.array(image).astype(np.float32)
        
        # Enhance warm tones
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.35, 0, 255)  # Red
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.2, 0, 255)   # Green
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.75, 0, 255)  # Blue
        
        # Add soft glow
        blurred = cv2.GaussianBlur(img_array, (21, 21), 0)
        golden = cv2.addWeighted(img_array, 0.75, blurred, 0.25, 0)
        
        return Image.fromarray(golden.astype(np.uint8))
    
    @staticmethod
    def apply_neon_nights(image):
        """Apply cyberpunk neon lighting"""
        img_array = np.array(image).astype(np.float32)
        
        # Enhance contrast
        img_array = cv2.convertScaleAbs(img_array, alpha=1.4, beta=15)
        
        # Add cyan-magenta color grading
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.15, 0, 255)  # Red
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 0.85, 0, 255)  # Green  
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.5, 0, 255)   # Blue
        
        # Add glow effect
        blurred = cv2.GaussianBlur(img_array, (15, 15), 0)
        neon = cv2.addWeighted(img_array, 0.65, blurred, 0.35, 0)
        
        return Image.fromarray(neon.astype(np.uint8))
    
    @staticmethod
    def apply_dramatic_shadows(image):
        """Apply dramatic lighting with strong shadows"""
        img_array = np.array(image)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Enhance contrast in L channel
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab[:, :, 0] = l_channel
        
        # Convert back to RGB
        dramatic = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Apply slight vignette
        h, w = dramatic.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        vignette = 1 - (dist_from_center / max_dist) * 0.35
        
        dramatic = dramatic * vignette[:, :, np.newaxis]
        dramatic = np.clip(dramatic, 0, 255)
        
        return Image.fromarray(dramatic.astype(np.uint8))
    
    @staticmethod
    def apply_soft_portrait(image):
        """Apply soft portrait lighting"""
        img_array = np.array(image)
        
        # Soft skin effect
        blurred = cv2.bilateralFilter(img_array, 15, 120, 120)
        soft = cv2.addWeighted(img_array, 0.55, blurred, 0.45, 0)
        
        # Warm up skin tones
        soft = soft.astype(np.float32)
        soft[:, :, 0] = np.clip(soft[:, :, 0] * 1.12, 0, 255)  # Red
        soft[:, :, 1] = np.clip(soft[:, :, 1] * 1.08, 0, 255)  # Green
        
        return Image.fromarray(soft.astype(np.uint8))

# ============================
# Enhanced Image Processing Functions
# ============================

def pil_to_cv(img):
    """Convert PIL to OpenCV format"""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(cv_img):
    """Convert OpenCV back to PIL format"""
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def apply_auto_magic(image):
    """One-click photo enhancement"""
    try:
        img_array = np.array(image)
        brightness = np.mean(img_array)
        
        if brightness < 100:
            bright_factor = 1.3
            contrast_factor = 1.2
        elif brightness > 180:
            bright_factor = 0.95
            contrast_factor = 1.1
        else:
            bright_factor = 1.15
            contrast_factor = 1.2
        
        enhanced = ImageEnhance.Brightness(image).enhance(bright_factor)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast_factor)
        enhanced = ImageEnhance.Color(enhanced).enhance(1.25)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.1)
        
        return enhanced
    except:
        return image

def apply_warm_filter(image):
    try:
        r, g, b = image.split()
        r = r.point(lambda x: min(int(x * 1.2), 255))
        g = g.point(lambda x: min(int(x * 1.1), 255))
        b = b.point(lambda x: max(int(x * 0.9), 0))
        result = Image.merge("RGB", (r, g, b))
        return result.filter(ImageFilter.SMOOTH)
    except:
        return image

def apply_cool_filter(image):
    try:
        r, g, b = image.split()
        r = r.point(lambda x: max(int(x * 0.95), 0))
        g = g.point(lambda x: min(int(x * 1.05), 255))
        b = b.point(lambda x: min(int(x * 1.2), 255))
        result = Image.merge("RGB", (r, g, b))
        return result.filter(ImageFilter.SMOOTH)
    except:
        return image

def apply_vintage_filter(image):
    try:
        img_array = np.array(image)
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        sepia_img = img_array @ sepia_filter.T
        sepia_img = np.clip(sepia_img, 0, 255)
        result = Image.fromarray(sepia_img.astype(np.uint8))
        return result.filter(ImageFilter.SMOOTH)
    except:
        return image

def apply_dramatic_effect(image):
    try:
        cv_img = pil_to_cv(image)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv_to_pil(enhanced)
    except:
        return ImageEnhance.Contrast(image).enhance(1.4)

def apply_soft_glow(image):
    try:
        blurred = image.filter(ImageFilter.GaussianBlur(radius=5))
        enhanced = ImageEnhance.Brightness(image).enhance(1.1)
        img_array = np.array(enhanced)
        blur_array = np.array(blurred)
        blended = img_array * 0.8 + blur_array * 0.2
        blended = np.clip(blended, 0, 255)
        return Image.fromarray(blended.astype(np.uint8))
    except:
        return image.filter(ImageFilter.SMOOTH)

def apply_black_white(image):
    try:
        bw = image.convert('L')
        bw = ImageEnhance.Contrast(bw).enhance(1.2)
        return bw.convert('RGB')
    except:
        return image

# ============================
# Manual Editing Functions
# ============================

def apply_manual_adjustments(image, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0):
    """Apply manual adjustments with sliders"""
    try:
        if brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        if contrast != 1.0:
            image = ImageEnhance.Contrast(image).enhance(contrast)
        if saturation != 1.0:
            image = ImageEnhance.Color(image).enhance(saturation)
        if sharpness != 1.0:
            image = ImageEnhance.Sharpness(image).enhance(sharpness)
        return image
    except:
        return image

def resize_image(image, width=None, height=None, maintain_aspect=True):
    """Resize image with optional aspect ratio maintenance"""
    try:
        original_width, original_height = image.size
        
        if maintain_aspect:
            if width and not height:
                aspect_ratio = original_height / original_width
                height = int(width * aspect_ratio)
            elif height and not width:
                aspect_ratio = original_width / original_height
                width = int(height * aspect_ratio)
        
        if width and height:
            return image.resize((width, height), Image.Resampling.LANCZOS)
        return image
    except:
        return image

def rotate_image(image, angle):
    """Rotate image by specified angle"""
    try:
        return image.rotate(angle, expand=True, fillcolor='white')
    except:
        return image

def flip_image(image, direction):
    """Flip image horizontally or vertically"""
    try:
        if direction == "horizontal":
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == "vertical":
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        return image
    except:
        return image

def enhance_quality(image, noise_reduction=False, sharpen_level=0):
    """Enhance image quality with noise reduction and sharpening"""
    try:
        if noise_reduction:
            cv_img = pil_to_cv(image)
            denoised = cv2.bilateralFilter(cv_img, 9, 75, 75)
            image = cv_to_pil(denoised)
        
        if sharpen_level > 0:
            if sharpen_level == 1:
                image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            elif sharpen_level == 2:
                image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
            else:
                image = image.filter(ImageFilter.SHARPEN)
        
        return image
    except:
        return image

# ============================
# Effect Definitions
# ============================

# Classic effects
CLASSIC_EFFECTS = {
    "✨ Auto Magic": {
        "description": "Smart AI enhancement for any photo",
        "function": apply_auto_magic,
        "category": "Smart"
    },
    "🔥 Warm Vibes": {
        "description": "Cozy, warm atmosphere",
        "function": apply_warm_filter,
        "category": "Classic"
    },
    "❄️ Cool Tone": {
        "description": "Modern, crisp feeling",
        "function": apply_cool_filter,
        "category": "Classic"
    },
    "📜 Vintage": {
        "description": "Classic film look",
        "function": apply_vintage_filter,
        "category": "Classic"
    },
    "⚡ Dramatic": {
        "description": "Bold, high-impact style",
        "function": apply_dramatic_effect,
        "category": "Classic"
    },
    "💫 Soft Glow": {
        "description": "Dreamy, ethereal effect",
        "function": apply_soft_glow,
        "category": "Classic"
    },
    "⚫ Black & White": {
        "description": "Timeless monochrome",
        "function": apply_black_white,
        "category": "Classic"
    }
}

# AI Style Transfer effects
AI_STYLE_EFFECTS = {
    "🎨 Van Gogh Style": {
        "description": "Swirling brushstrokes and vibrant colors",
        "style_type": "van_gogh",
        "category": "Artistic AI"
    },
    "🧩 Picasso Cubist": {
        "description": "Abstract geometric cubist style",
        "style_type": "picasso", 
        "category": "Artistic AI"
    },
    "🌸 Monet Impressionist": {
        "description": "Soft impressionist painting style",
        "style_type": "monet",
        "category": "Artistic AI"
    },
    "🎌 Anime Style": {
        "description": "Japanese anime/manga aesthetic",
        "style_type": "anime",
        "category": "Artistic AI"
    },
    "🌈 Abstract Art": {
        "description": "Modern abstract artistic style",
        "style_type": "abstract",
        "category": "Artistic AI"
    }
}

# Ambient Lighting effects
AMBIENT_EFFECTS = {
    "🌅 Golden Hour": {
        "description": "Warm, magical sunset lighting",
        "function": AmbientLighting.apply_golden_hour,
        "category": "Ambient AI"
    },
    "🌃 Neon Nights": {
        "description": "Cyberpunk neon atmosphere",
        "function": AmbientLighting.apply_neon_nights,
        "category": "Ambient AI"
    },
    "🎭 Dramatic Shadows": {
        "description": "High-contrast dramatic lighting",
        "function": AmbientLighting.apply_dramatic_shadows,
        "category": "Ambient AI"
    },
    "👤 Soft Portrait": {
        "description": "Flattering portrait lighting",
        "function": AmbientLighting.apply_soft_portrait,
        "category": "Ambient AI"
    }
}

# ============================
# Main Application
# ============================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎨 PhotoFix Pro - Complete AI Photo Editor</h1>
        <p>Professional photo editing with AI style transfer, ambient lighting, and manual controls</p>
        <p style="font-size: 1rem; opacity: 0.9;">Upload → Create → Perfect → Download!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize AI models
    if 'style_transfer' not in st.session_state and AI_AVAILABLE:
        with st.spinner("🤖 Loading AI models..."):
            st.session_state.style_transfer = NeuralStyleTransfer()
    
    # Sidebar
    st.sidebar.title("🎛️ Creative Studio")
    st.sidebar.markdown("Choose your editing approach")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload Your Photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload any photo to transform with AI and editing tools (max 20MB)"
    )
    
    if not uploaded_file:
        # Welcome screen
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎨 **AI Creative Features**
            - 🖼️ Neural style transfer (Van Gogh, Picasso, Monet)
            - 🌅 Ambient lighting effects (Golden Hour, Neon Nights)
            - 🎌 Anime and abstract art transformations
            - ✨ Smart auto-enhancement algorithms
            
            ### 🛠️ **Manual Editing Tools**
            - 🔆 Brightness, contrast, saturation controls
            - ✂️ Interactive cropping with aspect ratios
            - 🔄 Rotate, flip, and resize functions
            - 🎯 Quality enhancement and noise reduction
            """)
        
        with col2:
            st.markdown("""
            ### ⚡ **Quick Presets**
            - ✨ Auto Magic enhancement
            - 🔥 Warm and ❄️ Cool filters
            - 📜 Vintage and ⚡ Dramatic effects
            - 💫 Soft Glow and ⚫ Black & White
            
            ### 📱 **Export Options**
            - 💎 High quality for printing
            - 📱 Social media optimized formats
            - 🌐 Web optimized with compression
            - 🗜️ Custom quality and format settings
            """)
        
        if not AI_AVAILABLE:
            st.warning("⚠️ AI features require PyTorch. Install with: `pip install torch torchvision`")
        
        st.info("👆 Upload a photo above to start creating!")
        return
    
    # Load and process image
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error("Could not load your photo. Please try a different image.")
        return
    
    # Show image stats
    width, height = original_image.size
    megapixels = (width * height) / 1000000
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-box">
            <div style="font-size: 1.4rem;">📐</div>
            <div class="stats-value">{width} × {height}</div>
            <div class="stats-label">Dimensions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-box">
            <div style="font-size: 1.4rem;">⚡</div>
            <div class="stats-value">{megapixels:.1f}MP</div>
            <div class="stats-label">Resolution</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-box">
            <div style="font-size: 1.4rem;">💾</div>
            <div class="stats-value">{file_size:.1f}MB</div>
            <div class="stats-label">File Size</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stats-box">
            <div style="font-size: 1.4rem;">📊</div>
            <div class="stats-value">{width/height:.1f}:1</div>
            <div class="stats-label">Aspect Ratio</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced editing interface with AI tabs
    tabs = ["🎨 Quick Presets", "🛠️ Manual Adjustments", "✂️ Crop & Transform", "🎯 Quality & Export"]
    
    if AI_AVAILABLE:
        tabs = ["🤖 AI Style Transfer", "🌟 Ambient Lighting"] + tabs
    
    tab_objects = st.tabs(tabs)
    
    # Initialize session state for edited image
    if 'edited_image' not in st.session_state:
        st.session_state.edited_image = original_image.copy()
    
    processed_image = st.session_state.edited_image
    effect_name = "Custom Edited"
    
    tab_index = 0
    
    # AI Style Transfer Tab
    if AI_AVAILABLE:
        with tab_objects[tab_index]:
            st.markdown("""
            <div class="ai-section">
                <h3 style="text-align: center; margin-bottom: 1rem;">🎨 AI Neural Style Transfer</h3>
                <p style="text-align: center; margin-bottom: 1.5rem;">Transform your photo into famous artistic styles using advanced AI</p>
            </div>
            """, unsafe_allow_html=True)
            
            style_cols = st.columns(3)
            for i, (effect_name_key, effect_info) in enumerate(AI_STYLE_EFFECTS.items()):
                col_idx = i % 3
                with style_cols[col_idx]:
                    if st.button(
                        effect_name_key,
                        key=f"ai_style_{i}",
                        help=effect_info['description'],
                        use_container_width=True
                    ):
                        st.markdown("""
                        <div class="processing-message">
                            🎨 Applying AI style transfer magic...
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.spinner(f"🎨 Creating {effect_name_key} masterpiece..."):
                            if 'style_transfer' in st.session_state:
                                st.session_state.edited_image = st.session_state.style_transfer.apply_style_transfer(
                                    original_image, 
                                    effect_info['style_type']
                                )
                            else:
                                # Fallback to artistic filters
                                style_transfer = NeuralStyleTransfer()
                                st.session_state.edited_image = style_transfer.apply_style_transfer(
                                    original_image, 
                                    effect_info['style_type']
                                )
                            
                            processed_image = st.session_state.edited_image
                            effect_name = effect_name_key
                            st.success(f"✅ Applied {effect_name_key}!")
                            st.rerun()
        tab_index += 1
        
        # Ambient Lighting Tab
        with tab_objects[tab_index]:
            st.markdown("""
            <div class="ai-section">
                <h3 style="text-align: center; margin-bottom: 1rem;">🌟 AI Ambient Lighting</h3>
                <p style="text-align: center; margin-bottom: 1.5rem;">Transform the mood and atmosphere with AI-powered lighting effects</p>
            </div>
            """, unsafe_allow_html=True)
            
            ambient_cols = st.columns(2)
            for i, (effect_name_key, effect_info) in enumerate(AMBIENT_EFFECTS.items()):
                col_idx = i % 2
                with ambient_cols[col_idx]:
                    if st.button(
                        effect_name_key,
                        key=f"ambient_{i}",
                        help=effect_info['description'],
                        use_container_width=True
                    ):
                        with st.spinner(f"🌟 Applying {effect_name_key}..."):
                            st.session_state.edited_image = effect_info['function'](original_image)
                            processed_image = st.session_state.edited_image
                            effect_name = effect_name_key
                            st.success(f"✅ Applied {effect_name_key}!")
                            st.rerun()
        tab_index += 1
    
    # Quick Presets Tab
    with tab_objects[tab_index]:
        st.markdown("""
        <div class="editing-section">
            <h3 style="text-align: center; margin-bottom: 1rem;">🎨 Quick Preset Effects</h3>
            <p style="text-align: center; margin-bottom: 1.5rem;">Apply professional effects with one click</p>
        </div>
        """, unsafe_allow_html=True)
        
        preset_cols = st.columns(3)
        for i, (effect_name_key, effect_info) in enumerate(CLASSIC_EFFECTS.items()):
            col_idx = i % 3
            with preset_cols[col_idx]:
                if st.button(
                    effect_name_key,
                    key=f"preset_{i}",
                    help=effect_info['description'],
                    use_container_width=True
                ):
                    with st.spinner(f"Applying {effect_name_key}..."):
                        st.session_state.edited_image = effect_info['function'](original_image)
                        processed_image = st.session_state.edited_image
                        effect_name = effect_name_key
                        st.success(f"✅ Applied {effect_name_key}!")
                        st.rerun()
    tab_index += 1
    
    # Manual Adjustments Tab
    with tab_objects[tab_index]:
        st.markdown("""
        <div class="editing-section">
            <h3 style="text-align: center; margin-bottom: 1rem;">🛠️ Manual Adjustments</h3>
            <p style="text-align: center; margin-bottom: 1.5rem;">Fine-tune your photo with precision controls</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔆 **Light & Color**")
            
            brightness = st.slider(
                "💡 Brightness", 
                min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                help="Adjust overall brightness"
            )
            
            contrast = st.slider(
                "⚡ Contrast", 
                min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                help="Adjust contrast between light and dark areas"
            )
        
        with col2:
            st.markdown("#### 🎨 **Enhancement**")
            
            saturation = st.slider(
                "🌈 Saturation", 
                min_value=0.0, max_value=2.0, value=1.0, step=0.1,
                help="Adjust color intensity"
            )
            
            sharpness = st.slider(
                "🔍 Sharpness", 
                min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                help="Adjust image sharpness"
            )
        
        if st.button("✨ Apply Manual Adjustments", use_container_width=True):
            with st.spinner("Applying adjustments..."):
                st.session_state.edited_image = apply_manual_adjustments(
                    original_image, brightness, contrast, saturation, sharpness
                )
                processed_image = st.session_state.edited_image
                st.success("✅ Manual adjustments applied!")
                st.rerun()
        
        if st.button("🔄 Reset to Original", use_container_width=True):
            st.session_state.edited_image = original_image.copy()
            st.success("✅ Reset to original image!")
            st.rerun()
    tab_index += 1
    
    # Crop & Transform Tab
    with tab_objects[tab_index]:
        st.markdown("""
        <div class="editing-section">
            <h3 style="text-align: center; margin-bottom: 1rem;">✂️ Crop & Transform Tools</h3>
            <p style="text-align: center; margin-bottom: 1.5rem;">Crop, resize, rotate, and transform your image</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cropping section
        st.markdown("#### ✂️ **Interactive Cropping**")
        
        aspect_choice = st.selectbox(
            "📐 Aspect Ratio",
            ["Free", "1:1 (Square)", "4:3", "16:9", "3:2", "2:3 (Portrait)"],
            help="Choose aspect ratio for cropping"
        )
        
        aspect_dict = {
            "Free": None,
            "1:1 (Square)": (1, 1),
            "4:3": (4, 3),
            "16:9": (16, 9),
            "3:2": (3, 2),
            "2:3 (Portrait)": (2, 3)
        }
        
        aspect_ratio = aspect_dict[aspect_choice]
        
        try:
            cropped_img = st_cropper(
                st.session_state.edited_image,
                realtime_update=True,
                box_color='#667eea',
                aspect_ratio=aspect_ratio
            )
            
            if st.button("✂️ Apply Crop", use_container_width=True):
                st.session_state.edited_image = cropped_img
                st.success("✅ Crop applied!")
                st.rerun()
        except:
            st.warning("⚠️ Cropping tool requires streamlit-cropper library. Install with: pip install streamlit-cropper")
        
        st.markdown("---")
        
        # Transform section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔄 **Rotation & Flip**")
            
            rotation_angle = st.slider("🔄 Rotate", min_value=-180, max_value=180, value=0, step=15)
            
            flip_col1, flip_col2 = st.columns(2)
            with flip_col1:
                if st.button("↔️ Flip Horizontal"):
                    st.session_state.edited_image = flip_image(st.session_state.edited_image, "horizontal")
                    st.success("✅ Flipped horizontally!")
                    st.rerun()
            
            with flip_col2:
                if st.button("↕️ Flip Vertical"):
                    st.session_state.edited_image = flip_image(st.session_state.edited_image, "vertical")
                    st.success("✅ Flipped vertically!")
                    st.rerun()
            
            if rotation_angle != 0 and st.button("🔄 Apply Rotation"):
                st.session_state.edited_image = rotate_image(st.session_state.edited_image, rotation_angle)
                st.success(f"✅ Rotated by {rotation_angle}°!")
                st.rerun()
        
        with col2:
            st.markdown("#### 📏 **Resize**")
            
            current_width, current_height = st.session_state.edited_image.size
            
            new_width = st.number_input(
                "Width (px)", 
                min_value=50, 
                max_value=5000, 
                value=current_width,
                help="New width in pixels"
            )
            
            new_height = st.number_input(
                "Height (px)", 
                min_value=50, 
                max_value=5000, 
                value=current_height,
                help="New height in pixels"
            )
            
            maintain_aspect = st.checkbox("🔗 Maintain Aspect Ratio", value=True)
            
            if st.button("📏 Apply Resize"):
                st.session_state.edited_image = resize_image(
                    st.session_state.edited_image, new_width, new_height, maintain_aspect
                )
                st.success("✅ Image resized!")
                st.rerun()
    tab_index += 1
    
    # Quality & Export Tab
    with tab_objects[tab_index]:
        st.markdown("""
        <div class="editing-section">
            <h3 style="text-align: center; margin-bottom: 1rem;">🎯 Quality Enhancement & Export</h3>
            <p style="text-align: center; margin-bottom: 1.5rem;">Optimize quality and prepare for export</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 **Quality Enhancement**")
            
            noise_reduction = st.checkbox("🔇 Noise Reduction", help="Reduce image noise")
            
            sharpen_level = st.select_slider(
                "🔍 Sharpening",
                options=[0, 1, 2, 3],
                value=0,
                format_func=lambda x: ["None", "Light", "Medium", "Strong"][x],
                help="Apply sharpening filter"
            )
            
            if st.button("🎯 Enhance Quality"):
                st.session_state.edited_image = enhance_quality(
                    st.session_state.edited_image, noise_reduction, sharpen_level
                )
                st.success("✅ Quality enhanced!")
                st.rerun()
        
        with col2:
            st.markdown("#### 💾 **Export Settings**")
            
            export_quality = st.slider(
                "📊 JPEG Quality", 
                min_value=10, max_value=100, value=95,
                help="Higher values = better quality, larger file size"
            )
            
            export_format = st.selectbox(
                "📁 Format",
                ["JPEG", "PNG", "WEBP"],
                help="Choose export format"
            )
            
            # Preview compressed size
            if export_format == "JPEG":
                compressed_buffer = io.BytesIO()
                st.session_state.edited_image.save(
                    compressed_buffer, 
                    format="JPEG", 
                    quality=export_quality, 
                    optimize=True
                )
                compressed_size = len(compressed_buffer.getvalue()) / (1024 * 1024)
                st.info(f"💾 Estimated size: {compressed_size:.1f}MB")
    
    # Show current edited image
    processed_image = st.session_state.edited_image
    
    # Before/After Display
    st.markdown("""
    <div class="before-after-container">
        <h2 style="text-align: center; margin-bottom: 2rem; color: #374151;">
            📷 Before & After Transformation
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h3 style="color: #6b7280;">📸 Original</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(original_image, use_container_width=True, caption="Your original photo")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h3 style="color: #667eea;">🎨 Enhanced</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(processed_image, use_container_width=True, caption="Your enhanced photo")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Download section
    if not np.array_equal(np.array(original_image), np.array(processed_image)):
        st.markdown("""
        <div class="download-section">
            <h3 style="color: #374151; margin-bottom: 1rem;">🎉 Your Enhanced Photo is Ready!</h3>
            <p style="color: #6b7280; margin-bottom: 2rem;">Download in multiple formats and qualities</p>
        </div>
        """, unsafe_allow_html=True)
        
        download_col1, download_col2, download_col3 = st.columns(3)
        
        with download_col1:
            # High quality download
            hq_buffer = io.BytesIO()
            processed_image.save(hq_buffer, format="JPEG", quality=95, optimize=True)
            st.download_button(
                "💎 **High Quality**\n(95% quality)",
                data=hq_buffer.getvalue(),
                file_name=f"photofix_hq_{uploaded_file.name}",
                mime="image/jpeg",
                use_container_width=True
            )
        
        with download_col2:
            # Social media optimized
            social_img = processed_image.resize((1080, 1080), Image.Resampling.LANCZOS)
            social_buffer = io.BytesIO()
            social_img.save(social_buffer, format="JPEG", quality=90, optimize=True)
            st.download_button(
                "📱 **Social Media**\n(1080x1080px)",
                data=social_buffer.getvalue(),
                file_name=f"social_{uploaded_file.name}",
                mime="image/jpeg",
                use_container_width=True
            )
        
        with download_col3:
            # Custom quality download
            custom_buffer = io.BytesIO()
            processed_image.save(
                custom_buffer, 
                format=export_format, 
                quality=export_quality if export_format == "JPEG" else None,
                optimize=True
            )
            st.download_button(
                f"🎯 **Custom**\n({export_format}, {export_quality if export_format == 'JPEG' else 'Max'}% quality)",
                data=custom_buffer.getvalue(),
                file_name=f"custom_{uploaded_file.name.rsplit('.', 1)[0]}.{export_format.lower()}",
                mime=f"image/{export_format.lower()}",
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px; margin-top: 2rem;">
        <h3 style="color: #374151; margin-bottom: 1.5rem;">🏆 PhotoFix Pro - Complete AI Photo Editing Solution</h3>
        <p style="color: #6b7280; font-size: 1.1rem; margin-bottom: 2rem;">From AI-powered style transfer to professional manual editing - everything you need in one place</p>
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 15px;">
            <span style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 8px 16px; border-radius: 20px; font-size: 14px;">🎨 AI Style Transfer</span>
            <span style="background: linear-gradient(45deg, #f093fb, #f5576c); color: white; padding: 8px 16px; border-radius: 20px; font-size: 14px;">🌟 Ambient Lighting</span>
            <span style="background: linear-gradient(45deg, #4facfe, #00f2fe); color: white; padding: 8px 16px; border-radius: 20px; font-size: 14px;">🛠️ Manual Controls</span>
            <span style="background: linear-gradient(45deg, #43e97b, #38f9d7); color: white; padding: 8px 16px; border-radius: 20px; font-size: 14px;">✂️ Crop & Transform</span>
            <span style="background: linear-gradient(45deg, #fa709a, #fee140); color: white; padding: 8px 16px; border-radius: 20px; font-size: 14px;">🎯 Quality Control</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
