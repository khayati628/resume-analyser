# Resume Analyser

A small Streamlit app that reads a resume (PDF or DOCX) and predicts which job category it belongs to, along with a confidence score and the top 3 closest categories.

Built with TF-IDF features and a Logistic Regression classifier trained on a labelled resume dataset.

## Features

- Upload a `.pdf` or `.docx` resume
- Text extraction with `pypdf` and `python-docx`
- Predicts one of 24 job categories (IT, HR, Finance, Chef, Aviation, etc.)
- Shows confidence and a top 3 chart
- Preview of the extracted text

## Project structure

```
resume-analyser/
├── app.py              # Streamlit UI
├── train.py            # trains the model and saves it to models/
├── utils.py            # text cleaning and PDF/DOCX extraction
├── models/
│   ├── tfidf.pkl       # fitted TF-IDF vectorizer
│   └── classifier.pkl  # trained Logistic Regression model
├── data/
│   └── Resume.csv      # dataset (not committed, see below)
├── requirements.txt
└── README.md
```

## Setup

```bash
<<<<<<< HEAD
git clone https://github.com/khayati628/resume-analyser
=======
git clone https://github.com/<your-username>/resume-analyser.git
>>>>>>> 5a3d41720784b47890e77369b55655d82b011e19
cd resume-analyser

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

The trained model files are already included in `models/`, so you can start the app straight away:

```bash
streamlit run app.py
```

## Retrain the model

1. Put the dataset at `data/Resume.csv` (see below).
2. Run:

```bash
python train.py
```

This prints test accuracy and a per-category report, then saves fresh files into `models/`.

## Dataset

The model is trained on the Resume Dataset from Kaggle (`Resume.csv`, about 2,400 resumes across 24 categories). Download it and place it at `data/Resume.csv`. The file is about 56 MB, so it is listed in `.gitignore` and not pushed to GitHub.

Columns used: `Resume_str` (resume text) and `Category` (label).

## How it works

1. Clean the text: lowercase, strip HTML and URLs, keep letters only
2. Convert to TF-IDF features (3,000 features, English stop words removed, sublinear TF)
3. Classify with Logistic Regression (`C=10`)
4. Return the highest-probability categories

## Results

On a stratified 80/20 split, the model reaches about **69% accuracy** across 24 classes. Large categories like HR, IT, Accountant, Chef and Aviation do well, while small or vague ones (BPO, Automobile, Consultant, Arts) are weak because there are few examples and they overlap with other categories.

## Limitations

- Scanned or image-only PDFs have no extractable text and are not supported
- The label set is fixed to the dataset's 24 categories (there is no "Data Science" class, for example)
- Confidence scores are spread across 24 classes, so a correct prediction can still show a modest percentage

## Ideas for improvement

- Try a linear SVM or embeddings-based model (about 71% in a quick test with a linear SVM and bigrams)
- Add OCR for scanned resumes
- Add skill extraction and a job-description match score

## Tech stack

Python, Streamlit, scikit-learn, pandas, pypdf, python-docx
