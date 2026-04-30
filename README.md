# Real Waste Image Classification

A reproducible deep learning workflow for classifying real-world landfill waste images across multiple categories, with robust model training and a Streamlit demo web app.

---

## Dataset: RealWaste

This project uses the **RealWaste** dataset:

- **License:** CC BY-NC-SA 4.0  
- **Source:** [sam-single/realwaste](https://github.com/sam-single/realwaste)

**Category breakdown:**

| Label                | Images |
|----------------------|--------|
| Cardboard            | 461    |
| Food Organics        | 411    |
| Glass                | 420    |
| Metal                | 790    |
| Miscellaneous Trash  | 495    |
| Paper                | 500    |
| Plastic              | 921    |
| Textile Trash        | 318    |
| Vegetation           | 436    |


---

## Project Structure

- `app.ipynb`, `Final_UI.ipynb`  
  Streamlit page for real-time inference, demo, and UI. The included Streamlit app (`app.py` written within the notebook) lets users upload an image and receive a predicted waste category instantly.
- `custom_model.ipynb`  
  Building and training a custom ResNet-inspired convolutional neural network using TensorFlow/Keras.
- `DenseNet121.ipynb`  
  (Transfer learning approach using DenseNet121; pipeline and evaluation similar to MobileNetV2.)
- `Real_Waste_Classification_using_Transfer_Learning_(MobileNetV2).ipynb`  
  Modern transfer learning pipeline with MobileNetV2 as the feature extractor.
- Additional scripts/notebooks as helpers.

---

## Approach & Features

- **Preprocessing & Augmentation:**  
  - Uses `ImageDataGenerator` for 80/20 train/validation split
  - On-the-fly: Rescale, rotation, zoom, horizontal flip for more robust models

- **Class Imbalance Handling:**  
  - Computes and applies class weights automatically

- **Model Architectures:**  
  - Custom deep residual (ResNet-inspired) model
  - Transfer-learning with MobileNetV2 and DenseNet121

- **Evaluation:**  
  - Accuracy, precision, recall, f1-score (per class and averaged)
  - Confusion matrix visualization
  - Accuracy/Loss training and validation plots

- **Easy Inference and Deployment:**  
  - Save your trained models as `.h5` files  
  - Streamlit app for anyone to demo predictions quickly and visually

---

## Example Results

Custom Model Example (Validation):
```
           precision    recall  f1-score   support
Cardboard       0.34      0.33      0.33        92
Food Organics   0.75      0.77      0.76        82
Glass           0.67      0.86      0.75        84
Metal           0.57      0.48      0.52       158
Misc Trash      0.46      0.25      0.33        99
Paper           0.64      0.41      0.50       100
Plastic         0.47      0.46      0.47       184
Textile Trash   0.29      0.67      0.40        63
Vegetation      0.92      0.95      0.94        87

accuracy                            0.54       949
```
MobileNetV2 Example (Validation):
```
           precision    recall  f1-score   support
Cardboard       0.44      0.25      0.32        92
Food Organics   0.76      0.87      0.81        82
Glass           0.80      0.80      0.80        84
Metal           0.60      0.76      0.67       158
Misc Trash      0.44      0.54      0.48        99
Paper           0.64      0.58      0.61       100
Plastic         0.66      0.49      0.56       184
Textile Trash   0.46      0.60      0.52        63
Vegetation      0.89      0.91      0.90        87

accuracy                            0.63       949
```
_More details and plots are inside the notebooks._

---

## How to Run

1. **Clone this repository**
    ```bash
    git clone https://github.com/MuskanKarodiya/Real-Waste-Image-Classification.git
    cd Real-Waste-Image-Classification
    ```
2. **Download & Prepare the Dataset**
    - Download: [sam-single/realwaste](https://github.com/sam-single/realwaste)
    - Place it in a structured folder, e.g.:
      ```
      RealWaste/
        Cardboard/
        Food Organics/
        Glass/
        ...
        Vegetation/
      ```
3. **Open & Run Notebooks**
    - In Jupyter or Google Colab, start by running:
      - `custom_model.ipynb` — custom CNN, or
      - `Real_Waste_Classification_using_Transfer_Learning_(MobileNetV2).ipynb`
      - (Experiment with `DenseNet121.ipynb` for advanced transfer learning)
    - Set the `data_dir` path as needed inside each notebook.
    - Train, evaluate, and save your model.

4. **Demo with Streamlit App**
    - Run `app.ipynb` or `Final_UI.ipynb` in Colab or locally (or run the generated `app.py` with Streamlit).
    - Follow instructions to upload an image and preview predictions.
    - For local runs:
      ```bash
      streamlit run app.py
      ```

---

## Citation

> **RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning.**  
> Dataset License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## License

- **Code:** MIT (unless specified otherwise)
- **Dataset:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## Acknowledgements

- [sam-single/realwaste](https://github.com/sam-single/realwaste) for dataset and labeling protocol
- TensorFlow, Keras, scikit-learn
- All open-source contributors and the ML community!

---
