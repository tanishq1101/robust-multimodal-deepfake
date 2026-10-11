"""Gradio demo.  (Phase 7)

Upload a video -> overall real/fake verdict, visual score, audio score,
Grad-CAM heatmaps on the face crops and the spectrogram.

Run (after training):  python app/app.py --checkpoint <path>
"""

import gradio as gr


def analyse(video_path: str):
    raise gr.Error("The demo will be available after the models are trained (Phase 7).")


demo = gr.Interface(
    fn=analyse,
    inputs=gr.Video(label="Video"),
    outputs=gr.JSON(label="Result"),
    title="Robust Multimodal Deepfake Analysis",
)

if __name__ == "__main__":
    demo.launch()
