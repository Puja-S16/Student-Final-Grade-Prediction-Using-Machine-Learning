# Student Final Grade Prediction

Student Final Grade Prediction is a machine learning web application built using **Python, Streamlit, Scikit-learn, Pandas, and Matplotlib**. The application predicts a student's final grade (G3) based on academic performance, study habits, attendance, and personal factors using a **Random Forest Regressor**.

## Features

- Predicts the student's final grade (G3)
- Interactive Streamlit user interface
- Random Forest Regression model
- Displays model performance metrics (R² Score and Mean Squared Error)
- Feature Importance visualization
- Correlation Heatmap for dataset analysis
- Fast model loading using Streamlit caching

---

## Technology Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib

---

## Project Structure

```text
StudentGradePrediction/
│
├── app.py
├── student_mat_cleaned.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The application uses the **Student Performance Dataset**, which contains information about students' academic, social, and behavioral characteristics.

### Input Features

- Study Time
- Past Failures
- Absences
- First Period Grade (G1)
- Second Period Grade (G2)
- Health
- Free Time
- Family Relationship Quality

### Target Variable

- Final Grade (G3)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/StudentGradePrediction.git
```

Navigate to the project directory:

```bash
cd StudentGradePrediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Workflow

1. Load the student performance dataset.
2. Select relevant academic and behavioral features.
3. Split the dataset into training and testing sets.
4. Train a Random Forest Regressor.
5. Evaluate the model using R² Score and Mean Squared Error.
6. Accept student inputs through the Streamlit interface.
7. Predict the final grade (G3).
8. Display feature importance and dataset correlation heatmap.

---

## Model

**Algorithm Used**

- Random Forest Regressor

### Evaluation Metrics

- R² Score
- Mean Squared Error (MSE)

---

## Input Parameters

| Feature | Description |
|---------|-------------|
| Study Time | Weekly study time |
| Past Failures | Number of previous class failures |
| Absences | Number of school absences |
| G1 | First Period Grade |
| G2 | Second Period Grade |
| Health | Current health status |
| Free Time | Free time after school |
| Family Relationship | Quality of family relationships |

---

## Output

The application predicts the student's **Final Grade (G3)** on a scale of **0–20**.

---

## Visualizations

- Feature Importance Chart
- Correlation Heatmap

---

## Future Enhancements

- Compare multiple regression models
- Hyperparameter tuning
- Model export using Joblib
- Student performance dashboard
- Grade category prediction (Excellent, Good, Average, Poor)
- Model deployment using Docker
- REST API integration

---

## Author

**Puja Sharma**

GitHub: https://github.com/Puja-S16

---

## License

This project is developed for educational and academic purposes.
