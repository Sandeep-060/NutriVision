# NutriVision 🥗

### Multi-Food Recognition & Nutrition Explorer

NutriVision is a deep-learning computer vision application that analyzes a meal image containing multiple food items, identifies food regions, lets the user review the predictions, and calculates nutrition from user-entered quantities.

The application is designed as a **user-in-the-loop system**: the model provides food-region predictions, while the user remains in control of the final food selection and quantity.

## 🚀 Live Demo

**Live App:** https://nutri-vision.streamlit.app/

**GitHub:** https://github.com/Sandeep-060/NutriVision


## ✨ Features

* Multi-food meal image analysis
* Semantic segmentation using SegFormer B2
* Food-region and bounding-box extraction
* FoodSeg103-based 103-class vocabulary
* User-in-the-loop prediction review
* Confirm, Change, or Unsupported for each detection
* Manual food quantity entry in grams
* Nutrition calculation for calories, protein, carbohydrates, fat, and fiber
* Unsupported detections excluded from nutrition calculations
* Streamlit web interface
* Deployed model loaded from Hugging Face at runtime

## 🧠 How It Works

```text
Meal Image
    ↓
Image Preprocessing
    ↓
SegFormer B2
    ↓
Semantic Segmentation Mask
    ↓
Connected Components
    ↓
Food Regions + Bounding Boxes
    ↓
FoodSeg103 Class IDs → Food Names
    ↓
User Review(Confirm,Change,Unsupported)
          ↓
   Supported Foods
          ↓
   User Enters Grams
          ↓
   Nutrition Reference
          ↓
    Meal Nutrition Totals
```

## 🔬 Machine Learning

NutriVision uses **SegFormer B2** with a `512 × 512` input and `104` output classes:

* 1 background class
* 103 FoodSeg103 food/ingredient classes

The model produces a **semantic segmentation mask**. Food regions and bounding boxes are then obtained through connected-component post-processing.

The application does **not** use native object-detection or instance-segmentation bounding boxes.

### Model Deployment

The trained Keras model is approximately 140 MB and is hosted separately on Hugging Face rather than being committed to GitHub.

At runtime:

```text
Streamlit App
     ↓
Hugging Face Hub
     ↓
nutrivision_segformer_b2.keras
     ↓
Keras / KerasHub
     ↓
SegFormer B2
```

**Model:** NutriVision SegFormer B2 — Hugging Face

## 📊 Training & Evaluation

Production training used a custom TensorFlow/Keras `GradientTape` training loop with:

* AdamW optimizer
* Warmup followed by polynomial learning-rate decay
* Early stopping
* FP32 training
* Batch size of 4

### Dataset Split

* Training images: **4,479**
* Development images: **500**
* Held-out evaluation images: **2,135**

Best recorded development mIoU:

**0.3555**

Final held-out evaluation:

| Metric        | Result |
| ------------- | -----: |
| Images        |  2,135 |
| IoU threshold |   0.50 |
| Precision     | 0.3603 |
| Recall        | 0.4852 |
| F1            | 0.4135 |
| Mean IoU      | 0.8612 |

The model can produce meaningful food regions, but missed and extra regions can occur. The application therefore uses a review workflow before including predictions in the nutrition calculation.

## 🥗 Nutrition Pipeline

Nutrition information is kept separate from the vision model.

```text
FoodSeg103 Class ID
        ↓
Nutrition Reference
        +
User-entered grams
        ↓
Per-100g Scaling
        ↓
Individual Food Nutrition
        ↓
Meal Nutrition Totals
```

The nutrition reference data is stored in:

```text
data/processed/nutrition/nutrition_reference.csv
```

The runtime calculation logic is implemented in:

```text
src/nutrition/calculator.py
```

The system intentionally does **not** estimate food weight from the image. The user provides the consumed quantity in grams.

## 👤 User Review Workflow

Each detected food region can be reviewed by the user:

* **Confirm** — accept the predicted food.
* **Change** — select another supported FoodSeg103 class.
* **Unsupported** — exclude the detection from the nutrition workflow.

`Unsupported` is a review status, not an additional food class.

This allows the application to handle imperfect model predictions without automatically treating every detected region as a valid food item.

## 🗂️ Project Structure

```text
NutriVision/
│
├── app/
│   ├── app.py
│   ├── style.css
│   └── assets/
│
├── data/
│   ├── raw/
│   │   ├── FoodSeg103/
│   │   └── nutrition/
│   └── processed/
│       ├── foodseg103_segmentation/
│       └── nutrition/
├── models/
│   └── segmentation/
│       ├── nutrivision_segformer_b2.keras
│       └── nutrivision_segformer_b2.weights.h5   
│
├── notebooks/
│   ├── 1_segmentation_data_audit.ipynb
│   ├── 2_production_training_and_dev_evaluation.ipynb
│   ├── 3_final_evaluation_and_export.ipynb
│   ├── 4_final_model_evaluation.ipynb
│   └── 5_nutrition_data_engineering.ipynb
│
├── src/
│   ├── segmentation/
│   │   ├── data.py
│   │   ├── model.py
│   │   ├── inference.py
│   │   └── postprocess.py
│   │
│   └── nutrition/
│       └── calculator.py
│
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

* Python
* TensorFlow
* Keras
* KerasHub
* SegFormer B2
* FoodSeg103
* NumPy
* pandas
* SciPy
* Pillow
* Streamlit
* Git / GitHub
* Hugging Face Hub

## ▶️ Run Locally

### Clone the repository

```bash
git clone https://github.com/Sandeep-060/NutriVision.git
cd NutriVision
```

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

### Start the application

```powershell
streamlit run app/app.py
```

The application downloads the trained model from Hugging Face when it is first required.

## ⚠️ Limitations

* The segmentation model can produce missed or extra regions.
* Bounding boxes are derived from semantic segmentation and connected components; they are not native object-detection boxes.
* Nutrition values depend on the underlying reference data and user-entered quantities.
* Portion size and weight are not estimated from the image.
* The supported food vocabulary is limited to the FoodSeg103-based reference set.
* Nutrition calculations should be treated as estimates rather than precise dietary measurements.

## 👨‍💻 Author

**Sandeep**

GitHub: https://github.com/Sandeep-060/NutriVision

