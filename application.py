import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import io
import cv2
from streamlit_cropper import st_cropper

# ============================
# Configuration & Setup
# ============================

st.set_page_config(
    page_title="PhotoFix Pro - Complete Photo Editor",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with editing tools styling
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
    
    .editing-section {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 2px solid #667eea;
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
    
    .slider-label {
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

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
# New Manual Editing Functions
# ============================

def apply_manual_adjustments(image, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0):
    """Apply manual adjustments with sliders"""
    try:
        # Apply adjustments in order
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
                # Calculate height based on width
                aspect_ratio = original_height / original_width
                height = int(width * aspect_ratio)
            elif height and not width:
                # Calculate width based on height
                aspect_ratio = original_width / original_height
                width = int(height * aspect_ratio)
            elif width and height:
                # Use provided dimensions
                pass
        
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
            # Apply bilateral filter for noise reduction
            cv_img = pil_to_cv(image)
            denoised = cv2.bilateralFilter(cv_img, 9, 75, 75)
            image = cv_to_pil(denoised)
        
        if sharpen_level > 0:
            # Apply unsharp mask
            if sharpen_level == 1:
                image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            elif sharpen_level == 2:
                image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
            else:
                image = image.filter(ImageFilter.SHARPEN)
        
        return image
    except:
        return image

def compress_image(image, quality=95, format="JPEG"):
    """Compress image with specified quality"""
    try:
        buffer = io.BytesIO()
        if format.upper() == "JPEG":
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
        elif format.upper() == "PNG":
            image.save(buffer, format="PNG", optimize=True)
        else:
            image.save(buffer, format=format)
        
        buffer.seek(0)
        return Image.open(buffer)
    except:
        return image

# ============================
# Effect Presets
# ============================

SIMPLE_EFFECTS = {
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

# ============================
# Main Application
# ============================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📸 PhotoFix Pro - Complete Photo Editor</h1>
        <p>Professional photo editing with presets, manual controls, and advanced tools</p>
        <p style="font-size: 1rem; opacity: 0.9;">Upload → Edit → Perfect → Download!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Editing Tools")
    st.sidebar.markdown("Choose your editing approach")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload Your Photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload any photo to edit (max 20MB)"
    )
    
    if not uploaded_file:
        # Welcome screen
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎨 **Quick Presets**
            - ✨ Auto Magic enhancement
            - 🔥 Warm and ❄️ Cool filters
            - 📜 Vintage and ⚡ Dramatic effects
            - 💫 Soft Glow and ⚫ Black & White
            
            ### 🛠️ **Manual Editing Tools**
            - 🔆 Brightness and contrast control
            - 🌈 Saturation and sharpness adjustment
            - ↕️ Resize and crop tools
            - 🔄 Rotate and flip functions
            """)
        
        with col2:
            st.markdown("""
            ### ✂️ **Advanced Tools**
            - 📐 Interactive cropping with aspect ratios
            - 📏 Custom resize with aspect ratio lock
            - 🎯 Quality enhancement and noise reduction
            - 💾 Compression control for different uses
            
            ### 📱 **Export Options**
            - 💎 High quality for printing
            - 📱 Social media optimized
            - 🌐 Web optimized with compression
            - 🗜️ Custom quality settings
            """)
        
        st.info("👆 Upload a photo above to start editing!")
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
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
    
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
    
    # Editing interface tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎨 Quick Presets", 
        "🛠️ Manual Adjustments", 
        "✂️ Crop & Transform", 
        "🎯 Quality & Export"
    ])
    
    # Initialize session state for edited image
    if 'edited_image' not in st.session_state:
        st.session_state.edited_image = original_image.copy()
    
    processed_image = st.session_state.edited_image
    effect_name = "Custom Edited"
    
    with tab1:
        st.markdown("""
        <div class="editing-section">
            <h3 style="text-align: center; margin-bottom: 1rem;">🎨 Quick Preset Effects</h3>
            <p style="text-align: center; margin-bottom: 1.5rem;">Apply professional effects with one click</p>
        </div>
        """, unsafe_allow_html=True)
        
        preset_cols = st.columns(3)
        for i, (effect_name_key, effect_info) in enumerate(SIMPLE_EFFECTS.items()):
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
    
    with tab2:
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
        
        # Reset button
        if st.button("🔄 Reset to Original", use_container_width=True):
            st.session_state.edited_image = original_image.copy()
            st.success("✅ Reset to original image!")
            st.rerun()
    
    with tab3:
        st.markdown("""
        <div class="editing-section">
            <h3 style="text-align: center; margin-bottom: 1rem;">✂️ Crop & Transform Tools</h3>
            <p style="text-align: center; margin-bottom: 1.5rem;">Crop, resize, rotate, and transform your image</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cropping section
        st.markdown("#### ✂️ **Interactive Cropping**")
        
        # Aspect ratio selection
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
        
        # Interactive cropper
        try:
            cropped_img = st_cropper(
                st.session_state.edited_image,
                realtime_update=True,
                box_color='#0000FF',
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
    
    with tab4:
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
            📷 Before & After Comparison
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
            <h3 style="color: #667eea;">✨ Edited</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(processed_image, use_container_width=True, caption="Your edited photo")
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
        <h3 style="color: #374151; margin-bottom: 1.5rem;">🏆 PhotoFix Pro - Complete Photo Editing Solution</h3>
        <p style="color: #6b7280; font-size: 1.1rem; margin-bottom: 2rem;">From quick presets to professional manual editing - everything you need in one place</p>
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 15px;">
            <span style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 8px 16px; border-radius: 20px; font-size: 14px;">🎨 Quick Presets</span>
            <span style="background: linear-gradient(45deg, #f093fb, #f5576c); color: white; padding: 8px 16px; border-radius: 20px; font-size: 14px;">🛠️ Manual Controls</span>
            <span style="background: linear-gradient(45deg, #4facfe, #00f2fe); color: white; padding: 8px 16px; border-radius: 20px; font-size: 14px;">✂️ Crop & Transform</span>
            <span style="background: linear-gradient(45deg, #43e97b, #38f9d7); color: white; padding: 8px 16px; border-radius: 20px; font-size: 14px;">🎯 Quality Control</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
