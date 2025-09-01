"""
ArtVLM API Testing Interface

A comprehensive Streamlit application for testing the ArtVLM API with structured
feedback collection for model improvement.
"""

import streamlit as st
import requests
import json
import io
from PIL import Image
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ArtVLM Tester",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .feedback-form {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .result-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)





def call_artvlm_api(image_file, analysis_mode: str, api_endpoint: str, question: Optional[str] = None) -> Dict[str, Any]:
    """
    Call the ArtVLM API with the provided parameters.
    
    Args:
        image_file: Uploaded image file
        analysis_mode: Selected analysis mode
        api_endpoint: API endpoint URL
        question: Optional question for VQA mode
        
    Returns:
        API response as dictionary
    """
    try:
        # Prepare form data for multipart upload
        files = {
            'image': ('image.jpg', image_file, 'image/jpeg')
        }
        
        data = {
            'analysis_mode': analysis_mode
        }
        
        # Add question for VQA mode
        if analysis_mode == "Art Historical VQA" and question:
            data["question"] = question
        
        # Make API request with multipart form data
        response = requests.post(
            f"{api_endpoint}/analyze",
            files=files,
            data=data,
            timeout=30
        )
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse JSON response
        result = response.json()
        
        logger.info(f"API call successful for mode: {analysis_mode}")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise ValueError(f"API request failed: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse API response: {e}")
        raise ValueError(f"Invalid API response format: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in API call: {e}")
        raise ValueError(f"Unexpected error: {e}")


def display_stylometry_results(response: Dict[str, Any]):
    """Display stylometry and forgery detection results."""
    st.subheader("🎨 Stylometry & Forgery Detection Results")
    
    # Create metrics container
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'authenticity_score' in response:
                st.metric(
                    label="Authenticity Score",
                    value=f"{response['authenticity_score']:.1%}",
                    help="Confidence that the artwork matches the artist's known style"
                )
        
        with col2:
            if 'confidence' in response:
                st.metric(
                    label="Overall Confidence",
                    value=f"{response['confidence']:.1%}",
                    help="Confidence in the analysis"
                )
        
        with col3:
            if 'is_outlier' in response:
                status = "⚠️ Outlier" if response['is_outlier'] else "✅ Normal"
                st.metric(
                    label="Analysis Status",
                    value=status,
                    help="Whether the artwork is an outlier or normal"
                )
    
    # Display artist and style information
    if 'artist' in response:
        st.write(f"**Predicted Artist:** {response['artist']}")
    
    if 'style' in response:
        st.write(f"**Predicted Style:** {response['style']}")
    
    if 'period' in response:
        st.write(f"**Predicted Period:** {response['period']}")
    
    # Display forgery detection result
    if 'is_outlier' in response:
        if response['is_outlier']:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ Potential Forgery or Misattribution Detected</h4>
                <p>The analysis suggests this artwork may not be by the attributed artist or may be a forgery.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
                <h4>✅ Consistent with Artist's Style</h4>
                <p>The analysis indicates this artwork is consistent with the artist's known style and techniques.</p>
            </div>
            """, unsafe_allow_html=True)


def display_vqa_results(response: Dict[str, Any], question: str):
    """Display VQA results."""
    st.subheader("❓ Art Historical VQA Results")
    
    # Display question and answer
    st.markdown(f"**Question:** {question}")
    
    if 'answer' in response:
        st.info(response['answer'])
    
    # Display confidence
    if 'confidence' in response:
        st.metric(
            label="Answer Confidence",
            value=f"{response['confidence']:.1%}",
            help="Confidence in the generated answer"
        )
    
    # Display mode information
    if 'mode' in response:
        st.write(f"**Analysis Mode:** {response['mode']}")


def display_captioning_results(response: Dict[str, Any]):
    """Display expert-level captioning results."""
    st.subheader("📝 Expert-Level Art Historical Analysis")
    
    # Display caption
    if 'caption' in response:
        st.markdown("**Generated Caption:**")
        st.info(response['caption'])
    
    # Display style analysis
    if 'style_analysis' in response:
        st.markdown("**Style Analysis:**")
        st.write(response['style_analysis'])
    
    # Display confidence
    if 'confidence' in response:
        st.metric(
            label="Caption Confidence",
            value=f"{response['confidence']:.1%}",
            help="Confidence in the generated caption"
        )
    
    # Display mode information
    if 'mode' in response:
        st.write(f"**Analysis Mode:** {response['mode']}")


def collect_feedback(analysis_mode: str, response: Dict[str, Any], question: Optional[str] = None):
    """Collect structured feedback from the user."""
    st.markdown("---")
    st.subheader("📊 Feedback & Evaluation")
    st.write("Please rate the quality of the generated analysis to help improve the model.")
    
    with st.form("feedback_form"):
        st.markdown("""
        <div class="feedback-form">
        """, unsafe_allow_html=True)
        
        # Evaluation criteria
        fluency_score = st.slider(
            "Fluency",
            min_value=1,
            max_value=5,
            value=3,
            help="Is the text grammatically correct and natural-sounding?"
        )
        
        relevance_score = st.slider(
            "Relevance & Accuracy",
            min_value=1,
            max_value=5,
            value=3,
            help="Is the information factually correct and relevant to the image and/or question?"
        )
        
        detail_score = st.slider(
            "Descriptiveness & Insight",
            min_value=1,
            max_value=5,
            value=3,
            help="Does the text provide insightful, specific details about the artwork's style, context, or meaning?"
        )
        
        # Qualitative comments
        comments = st.text_area(
            "Additional Comments (Optional)",
            placeholder="Please provide specific examples of what worked well or what could be improved...",
            help="Share specific feedback about the analysis quality, accuracy, or suggestions for improvement"
        )
        
        # Submit button
        submitted = st.form_submit_button("Submit Feedback")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if submitted:
            # Store feedback in session state
            feedback_data = {
                "timestamp": datetime.now().isoformat(),
                "analysis_mode": analysis_mode,
                "question": question,
                "response": response,
                "scores": {
                    "fluency": fluency_score,
                    "relevance_accuracy": relevance_score,
                    "descriptiveness_insight": detail_score
                },
                "comments": comments,
                "average_score": (fluency_score + relevance_score + detail_score) / 3
            }
            
            # Store in session state
            if "feedback_history" not in st.session_state:
                st.session_state.feedback_history = []
            
            st.session_state.feedback_history.append(feedback_data)
            
            # Display success message
            st.success("✅ Thank you for your feedback! Your evaluation has been recorded.")
            
            # Display feedback summary
            st.write("**Feedback Summary:**")
            st.write(f"- Fluency: {fluency_score}/5")
            st.write(f"- Relevance & Accuracy: {relevance_score}/5")
            st.write(f"- Descriptiveness & Insight: {detail_score}/5")
            st.write(f"- Average Score: {feedback_data['average_score']:.1f}/5")
            
            if comments:
                st.write(f"- Comments: {comments}")


def display_feedback_history():
    """Display feedback history."""
    if "feedback_history" in st.session_state and st.session_state.feedback_history:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 Feedback History")
        
        # Show recent feedback
        recent_feedback = st.session_state.feedback_history[-5:]  # Last 5 entries
        
        for i, feedback in enumerate(reversed(recent_feedback), 1):
            with st.sidebar.expander(f"Feedback {i} - {feedback['analysis_mode']}"):
                st.write(f"**Date:** {feedback['timestamp'][:19]}")
                st.write(f"**Mode:** {feedback['analysis_mode']}")
                st.write(f"**Average Score:** {feedback['average_score']:.1f}/5")
                
                if feedback['question']:
                    st.write(f"**Question:** {feedback['question']}")
                
                if feedback['comments']:
                    st.write(f"**Comments:** {feedback['comments']}")


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🎨 ArtVLM API Tester</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if "api_response" not in st.session_state:
        st.session_state.api_response = None
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    
    # Create two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Control Panel")
        
        # API Endpoint Input
        api_endpoint = st.text_input(
            "API Endpoint URL",
            value=st.session_state.get("api_endpoint", "http://localhost:8000"),
            placeholder="http://localhost:8000",
            help="Enter the ArtVLM API endpoint URL. Use 'http://localhost:8000' for local development."
        )
        
        # Store in session state
        st.session_state.api_endpoint = api_endpoint
        
        # Test API Connection
        if st.button("🔗 Test API Connection", help="Test if the API endpoint is reachable"):
            with st.spinner("Testing connection..."):
                try:
                    import requests
                    response = requests.get(f"{api_endpoint}/health", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ API connection successful!")
                        st.json(response.json())
                    else:
                        st.error(f"❌ API returned status code: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Connection failed. Make sure the API server is running on the specified endpoint.")
                except Exception as e:
                    st.error(f"❌ Connection error: {e}")
        
        # Image Uploader
        uploaded_file = st.file_uploader(
            "Upload Artwork Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image file (JPG, PNG) to analyze"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_image = uploaded_file
            
            # Display uploaded image info
            image = Image.open(uploaded_file)
            st.write(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**File Size:** {uploaded_file.size / 1024:.1f} KB")
        
        # Analysis Mode Selector
        analysis_mode = st.selectbox(
            "Select Analysis Mode",
            options=[
                "Stylometry & Forgery Detection",
                "Art Historical VQA",
                "Expert-Level Captioning"
            ],
            help="Choose the type of analysis to perform"
        )
        
        # VQA Question Input (conditional)
        question = None
        if analysis_mode == "Art Historical VQA":
            question = st.text_area(
                "Enter your question about the artwork",
                placeholder="e.g., What style is this artwork executed in?",
                help="Ask a specific question about the artwork's style, content, or context"
            )
        
        # Analyze Button
        analyze_button = st.button(
            "🔍 Analyze Artwork",
            type="primary",
            disabled=not (uploaded_file and api_endpoint),
            help="Click to analyze the uploaded artwork"
        )
        
        # Sidebar for feedback history
        display_feedback_history()
    
    with col2:
        st.markdown("### 📊 Results & Analysis")
        
        # Display uploaded image
        if st.session_state.uploaded_image is not None:
            st.image(st.session_state.uploaded_image, caption="Uploaded Artwork", use_column_width=True)
        
        # Handle analysis request
        if analyze_button and uploaded_file and api_endpoint:
            with st.spinner("Analyzing artwork..."):
                try:
                    # Call API
                    response = call_artvlm_api(
                        image_file=uploaded_file,
                        analysis_mode=analysis_mode,
                        api_endpoint=api_endpoint,
                        question=question
                    )
                    
                    # Store response in session state
                    st.session_state.api_response = response
                    
                    # Debug: Show raw response
                    with st.expander("🔍 Debug: Raw API Response"):
                        st.json(response)
                    
                    # Display results based on analysis mode
                    if analysis_mode == "Stylometry & Forgery Detection":
                        display_stylometry_results(response)
                    elif analysis_mode == "Art Historical VQA":
                        display_vqa_results(response, question or "No question provided")
                    elif analysis_mode == "Expert-Level Captioning":
                        display_captioning_results(response)
                    
                    # Collect feedback
                    collect_feedback(analysis_mode, response, question)
                    
                except ValueError as e:
                    st.error(f"❌ Analysis failed: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")
                    logger.error(f"Unexpected error in analysis: {e}")
        
        # Display cached results if available
        elif st.session_state.api_response is not None:
            st.info("📋 Displaying previous analysis results. Upload a new image and click 'Analyze Artwork' to perform a new analysis.")
            
            # Display cached results
            response = st.session_state.api_response
            if analysis_mode == "Stylometry & Forgery Detection":
                display_stylometry_results(response)
            elif analysis_mode == "Art Historical VQA":
                display_vqa_results(response, question or "No question provided")
            elif analysis_mode == "Expert-Level Captioning":
                display_captioning_results(response)
        
        # Initial state message
        else:
            st.info("👆 Upload an image and configure the API endpoint to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>ArtVLM API Tester | Built with Streamlit | Feedback data helps improve model performance</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
