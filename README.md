

---

# Loan Credit Approval App

A **Loan Credit Approval Prediction Application** built using **Python**, **Streamlit**, and a **GradientBoostingClassifier** model trained on a dataset from **OpenML**. The app predicts whether a loan should be approved based on applicant details and runs inside a **Docker container** for easy deployment.

---

## 🚀 Features

* Simple and intuitive **Streamlit** interface.
* Predicts loan approval based on applicant data.
* **GradientBoostingClassifier** for high accuracy.
* Fully **Dockerized** — just pull and run.

---

## 📂 Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── model.pkl              # Trained ML model
├── Dockerfile             # Docker configuration
└── README.md              # Project documentation
```

---

## 🐳 Running the App with Docker

Since the app is already containerized, simply run:

```bash
docker run -p 8501:8501 <your-docker-image-name>
```

Then open your browser at:

```
http://localhost:8501
```

---

## 📊 Model Information

* **Algorithm:** GradientBoostingClassifier
* **Dataset:** Loan Credit Dataset from OpenML
* **Prediction Output:**

  * ✅ Loan Approved
  * ❌ Loan Denied

---

## 🧪 Example Input for Testing

When using the app, you will be prompted to enter applicant details.
Here’s an example set of values you can use to test:

| Field                    | Example Value |
| ------------------------ | ------------- |
| Age                      | 35            |
| Income                   | 60000         |
| Loan Amount              | 15000         |
| Loan Duration (months)   | 36            |
| Credit Score             | 720           |
| Number of Existing Loans | 1             |
| Employment Status        | Employed      |
| Marital Status           | Married       |
| Education Level          | Bachelor's    |

**Expected Output:**
`Loan Approved ✅`

---

## ✨ Possible Improvements

* Add multiple ML model support.
* Improve UI/UX for better user interaction.
* Deploy to cloud platforms like AWS, Azure, or Heroku.

---


