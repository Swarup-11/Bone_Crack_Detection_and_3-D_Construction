import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image

# Function to display image using matplotlib
def display_image(img, title="Image"):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    ax.set_title(title)
    st.pyplot(fig)

# Simple rule-based chatbot response
def chatbot_response(user_input):
    user_input = user_input.lower()
    if "crack" in user_input:
        return "Bone cracks are detected by analyzing contours in the X-ray image."
    elif "treatment" in user_input:
        return "If the number of cracks is small, plaster is recommended. Otherwise, rod surgery may be required."
    elif "3d" in user_input:
        return "We use intensity-based 3D reconstruction to model the bone surface from the X-ray."
    elif "hello" in user_input or "hi" in user_input:
        return "Hello! How can I assist you with bone crack detection today?"
    else:
        return "I'm still learning! Please ask about cracks, treatment, or 3D reconstruction."

# Main Streamlit App
def main():
    st.title("Bone Crack Detection & 3D Reconstruction")
    st.write("Upload an image to detect bone cracks and get treatment suggestions.")

    # Chatbot in sidebar
    st.sidebar.title("💬 AI Chatbot")
    user_query = st.sidebar.text_input("Ask me anything:")
    if user_query:
        response = chatbot_response(user_query)
        st.sidebar.write("🤖", response)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if image is None:
            st.error("Error: Unable to load the image.")
            return

        st.subheader("Loaded Image")
        display_image(image)

        # Detect button
        if st.button("Detect"):
            # Processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

            contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            num_cracks = len(contours)
            st.success(f"Number of Cracks Detected: {num_cracks}")

            if num_cracks < 10:
                st.info("🩹 Plaster is recommended.")
            else:
                st.warning("🔩 You may need rod surgery.")

            # 3D Reconstruction
            st.subheader("3D Bone Surface Reconstruction")
            depth_map = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            height, width = gray.shape
            X, Y = np.meshgrid(range(width), range(height))

            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=depth_map)])
            fig.update_layout(
                title="3D Bone Structure",
                scene=dict(
                    aspectmode='data',
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Intensity'
                )
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
