# 🏥 Health Insurance Cost Prediction (Deep Learning)

This project uses a **Deep Neural Network (DNN)** built with **TensorFlow/Keras** to predict **medical insurance expenses** based on demographic and lifestyle factors such as **age, sex, BMI, number of children, smoking habits, and region**.  

The dataset comes from [freeCodeCamp - Health Costs Dataset](https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv).

---

## 📂 Project Structure
- `insurance.csv` → Dataset  
- `health_costs.py` or `.ipynb` → Model training & evaluation code  
- `README.md` → Project documentation  

---

## ⚙️ Features
- Data preprocessing with **pandas** and **scikit-learn**  
- Feature engineering (e.g., **BMI × Smoker**, **Age × BMI** interactions)  
- Train/test split with **scaling**  
- Deep Neural Network with:
  - Hidden layers: 128 → 64 → 32 units (ReLU activation)  
  - Output: 1 unit (regression)  
  - Optimizer: Adam  
  - Loss: Mean Squared Error (MSE)  
- **Early stopping** for better generalization  
- Evaluation using **Mean Absolute Error (MAE)**  

---

## 🧪 Model Performance
- **Metric**: Mean Absolute Error (MAE)  
- **Target**: MAE < 3500 USD  
- ✅ Model achieves this goal  

---

## 📊 Results
- Predictions are close to actual expenses  
- Scatter plot of **Predicted vs True Values** shows points aligning along the diagonal  

---

## 🚀 How to Run
1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/health-insurance-prediction.git
   cd health-insurance-prediction
2. Install dependencies
   pip install -r requirements.txt
3. Run the script/notebook
   
## 📦 Dependencies

- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow-docs

## 📈 Future Improvements

- Hyperparameter tuning (learning rate, batch size, architecture)
- Add Dropout / Batch Normalization for regularization
- Compare with other regressors (Random Forest, XGBoost)
- Deploy model using FastAPI / Flask

## 📜 License

This project is licensed under the MIT License.

## 🙌 Acknowledgments

- freeCodeCamp for the dataset & challenge
- TensorFlow team for documentation & examples
