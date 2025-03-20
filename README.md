

# HumAware-VAD: Humming-Aware Voice Activity Detection

## 📌 Overview
**HumAware-VAD** is a fine-tuned version of the **[Silero-VAD](https://github.com/snakers4/silero-vad/tree/master)** model, trained to distinguish **humming from actual speech**. Standard Voice Activity Detection (VAD) models, including Silero-VAD, often misclassify humming as speech, leading to inaccurate speech segmentation. HumAware-VAD improves upon this by leveraging a custom dataset (**[HumSpeechBlend](https://huggingface.co/datasets/CuriousMonkey7/HumSpeechBlend)**) to enhance speech detection accuracy in the presence of humming.

## 🎯 Purpose
The primary goal of **HumAware-VAD** is to:
- Reduce **false positives** where humming is mistakenly detected as speech.
- Enhance **speech segmentation accuracy** in real-world applications.
- Improve VAD performance for tasks involving **music, background noise, and vocal sounds**.

## 🗂️ Model Details
- **Base Model**: [Silero-VAD](https://github.com/snakers4/silero-vad/tree/master)
- **Fine-tuning Dataset**: [HumSpeechBlend](https://huggingface.co/datasets/CuriousMonkey7/HumSpeechBlend)
- **Format**: JIT (TorchScript)
- **Framework**: PyTorch
- **Inference Speed**: Real-time

## ⚠️ Limitations
- The model may **miss speech detection** if the user speaks too softly.
- Works best for detecting **"mhm" humming sounds**.
- May also work for sounds like **"la la la"** or **"da da da"**, but with **varying accuracy**.

<!-- 
## 📥 Download & Usage
### 🔹 Install Dependencies
```bash
pip install humaware-vad
```

### 🔹 Load the Model
```python

```

### 🔹 Run Inference
```python
import torchaudio

waveform, sample_rate = torchaudio.load("data/0000.wav")
out = vad_model(waveform)
print("VAD Output:", out)
``` -->
<!-- 
## 🏆 Performance
Compared to the base Silero-VAD model, **HumAware-VAD** demonstrates:
✅ **Lower false positives for humming**
✅ **Better segmentation of speech in mixed audio**
✅ **Maintained real-time inference capabilities**

## 📊 Applications
- **Automatic Speech Recognition (ASR) Preprocessing**
- **Noise-Robust VAD Systems**
- **Speech Enhancement & Separation**
- **Call Center & Voice Communication Filtering** -->

## 📄 Citation
If you use this model, please cite it accordingly.

```
@model{HumAwareVAD2025,
  author = {Sourabh Saini},
  title = {HumAware-VAD: Humming-Aware Voice Activity Detection},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/CuriousMonkey7/HumAware-VAD}
}
```