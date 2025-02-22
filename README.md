# **AeroReformer: Aerial Referring Transformer for UAV-based Referring Image Segmentation**

🚀 **AeroReformer** is a novel framework for **UAV-based referring image segmentation (RIS)**, designed to address the unique challenges of aerial imagery, such as complex spatial scales, occlusions, and varying object orientations.  

Our method integrates **multi-head vision-language fusion (MHVLFM)** and **multi-scale rotation-aware fusion (MSRAFM)** to achieve superior segmentation performance compared to existing RIS approaches.  

The datasets and code will be made publicly available at our **[GitHub repository](https://github.com/lironui/UAV-RS)**.

---

## **📝 Paper Status**
Our research paper detailing **AeroReformer** is currently in preparation and will be released soon. Stay tuned for updates!  

---

## **📌 Model Overview**
**AeroReformer** is a **transformer-based vision-language model** designed for **referring segmentation in UAV imagery**. It automatically **localizes and segments objects** based on **natural language descriptions**, overcoming the limitations of existing RIS models in aerial datasets.  

### **🔹 Key Features**
✅ **Automatic Annotation Pipeline**: Utilizes open-source UAV segmentation datasets and large language models (LLMs) to generate textual descriptions.  
✅ **Multi-Head Vision-Language Fusion (MHVLFM)**: Enhances cross-modal understanding for precise segmentation.  
✅ **Multi-Scale Rotation-Aware Fusion (MSRAFM)**: Improves robustness to aerial scene variations.  
✅ **State-of-the-Art Performance**: Sets a new benchmark in UAV-based referring segmentation on multiple datasets.  

---

## **📥 Installation & Usage**
To load and use the **AeroReformer** model from Hugging Face:  

### **1️⃣ Install Dependencies**
```bash
pip install transformers torch huggingface_hub
