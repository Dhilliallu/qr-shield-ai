# QR Shield AI

QR Shield AI is a local research MVP for QR image screening. It includes a
baseline image classifier, QR decoding, heuristic URL review, dataset audit,
and an interactive Streamlit interface. It does not provide external
threat-intelligence lookups or a production security verdict.

## Project objectives

- Prepare a clear project structure for later dataset and model development.
- Make the initial training settings easy to find and update.
- Check that the Python environment and expected dataset directory are present.
- Avoid committing private or large dataset and model files.

## Technology stack

Python, PyTorch, Torchvision, OpenCV, NumPy, Pandas, Matplotlib, Seaborn,
Scikit-learn, Pillow, tqdm, Pyzbar, XGBoost, SHAP, FastAPI, Uvicorn, Requests,
Tldextract, and Streamlit.

## Folder structure

```text
fake qr/
├── dataset/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   ├── test/
│   │   ├── real/
│   │   └── fake/
│   ├── benign_qr_images_500/
│   └── malicious_qr_images_500/
├── notebooks/
├── models/
├── src/
│   ├── check_environment.py
│   ├── prepare_dataset.py
│   ├── analyze_errors.py
│   ├── scan_qr.py
│   ├── url_risk.py
│   ├── evaluate_external.py
│   ├── audit_dataset.py
│   ├── model.py
│   └── train.py
├── backend/
├── frontend/
│   └── app.py
├── .streamlit/
│   └── config.toml
├── tests/
│   ├── test_prepare_dataset.py
│   ├── test_analyze_errors.py
│   ├── test_url_risk.py
│   ├── test_scan_qr.py
│   ├── test_evaluate_external.py
│   ├── test_audit_dataset.py
│   └── test_model.py
├── reports/
│   ├── test_error_analysis.csv
│   └── dataset_audit.json
├── .gitignore
├── config.py
├── README.md
├── run_qr_shield.bat
└── requirements.txt
```

The source folders contain the provided dataset. Preparation maps
`benign_qr_images_500` to `real` and `malicious_qr_images_500` to `fake`; those
labels come from the folder names and are not independently verified. No sample
or fabricated data is created.

## Installation

Commands below are for Windows PowerShell, run from this project folder.

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation for this terminal, allow it for the current
   process and activate again:

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .\.venv\Scripts\Activate.ps1
   ```

2. Install PyTorch and Torchvision separately. Select the Windows, Pip, Python,
   and CPU or CUDA options that match your machine on the
   [official PyTorch installation page](https://pytorch.org/get-started/locally/),
   then run the command it provides in the activated environment. CUDA support
   also depends on a compatible NVIDIA driver.

3. Install the project dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

   PyTorch and Torchvision are included in `requirements.txt`; installing the
   hardware-appropriate builds first lets pip reuse them when they satisfy
   those package requirements.

## Validate the environment

With the virtual environment activated, run:

```powershell
python .\src\check_environment.py
```

The report displays the Python version, whether PyTorch, OpenCV, and NumPy can
be imported, CUDA availability and GPU name when available, and whether the
dataset directory exists. A present dataset folder does not mean it contains
data.

## Prepare the existing dataset

The preparation script uses the two source folders under `dataset/`, checks
that every source image can be decoded and that there are no exact duplicate
files, then resizes images to `224 x 224` and writes them into the
`train/`, `val/`, and `test/` class folders. It uses a reproducible 70/15/15
split (seed `42`) separately for each class. Source images are not changed.
The script refuses to write if any output class folder already contains files.

Run it from the project folder with the virtual environment activated:

```powershell
python -m src.prepare_dataset
```

Run the focused split tests with:

```powershell
python -m unittest discover -s tests
```

## Train the baseline CNN

After preparing the dataset, run the baseline classifier from the project
folder with the virtual environment activated:

```powershell
python -m src.train
```

The script trains a small CNN from scratch on `dataset/train`, selects the
checkpoint with the best validation accuracy (using validation loss to break
ties), then evaluates that checkpoint once on `dataset/test`. It prints loss,
accuracy, per-class precision/recall/F1, and a confusion matrix, and saves the
checkpoint as `models/baseline_cnn.pth`. Results measure this dataset only;
they are not evidence of performance on unseen QR sources or real-world
security. The output class names inherit the supplied source-folder labels.

## Analyze test-set errors

After training, list each test image's true label, predicted label, and
uncalibrated softmax scores with:

```powershell
python -m src.analyze_errors
```

The script writes `reports/test_error_analysis.csv` and prints each
misclassification. On the current 150-image test split, the model got 120
correct (80.0%), but misclassified 25 of 75 `fake` images as `real` and 5 of 75
`real` images as `fake`. In particular, a fake-to-real error rate of 33.3% is a
serious limitation for a security detector. Some of these incorrect
fake-to-real predictions have softmax scores between 66.5% and 72.9%; these
scores are not calibrated probabilities and must not be interpreted as a
reliable safety confidence.

The report identifies which images were misclassified, but image appearance
alone does not establish why. QR codes can have similar visual patterns while
encoding different destinations; this classifier does not decode or inspect
their payloads.

## Current limitations

- The dataset has 500 images per source folder, and the split contains 75
  images per class in the test set. Metrics on this small set are uncertain.
- `real` and `fake` labels are inherited from the provided folder names; they
  have not been independently verified against the QR payload or destination.
- The test images come from the provided dataset, not a separate external
  source. Performance on new generators, image quality, print/display capture,
  or deployment conditions is unknown.
- This model learns from QR image pixels only. It does not decode QR payloads,
  assess URLs, or prove that a QR code or destination is safe. QR decoding and
  basic URL warning signals are separate tools, not learned model predictions.
- The test split has now been used for error analysis. Keep a separate,
  untouched external dataset for future unbiased evaluation; do not tune the
  model against this test set and continue treating it as independent.
- The external evaluation workflow checks for exact image-file hash overlap
  with the project's available images, but cannot verify dataset provenance,
  label correctness, or resized/edited duplicate images.
- Do not use this baseline alone to approve or reject QR codes in a
  security-sensitive workflow.

## Evaluate on a separate dataset

The current project does not contain an independent external evaluation
dataset. To evaluate one you have sourced and labeled, arrange it outside the
project's `dataset/` directory:

```text
external-qr-dataset/
├── fake/
│   └── ... image files ...
└── real/
    └── ... image files ...
```

Labels must be independently verified against the QR payload and destination;
the evaluator cannot verify labels or provenance. Run:

```powershell
python -m src.evaluate_external C:\path\to\external-qr-dataset
```

The command loads the existing checkpoint for inference only. It refuses to
evaluate folders inside or overlapping the project's dataset tree, duplicate
image bytes within the external set, and exact byte-identical images found in
the current source/train/validation/test data. These checks do not detect
resized or otherwise modified duplicates. The JSON output includes dataset
counts, loss, accuracy, per-class precision/recall/F1, confusion matrix, and
per-image predictions. To save it:

```powershell
python -m src.evaluate_external C:\path\to\external-qr-dataset --output .\reports\external-evaluation.json
```

Do not use the external results to tune the model and then continue to describe
the same data as an untouched independent evaluation set. No external
evaluation metrics are reported until real, independently sourced data is
provided.

## Audit the current dataset

Run a local QR and duplicate-risk audit with:

```powershell
python -m src.audit_dataset
```

The script writes `reports/dataset_audit.json`. It reports source image counts,
QR decode counts, dimensions, exact file duplicates, pixel-identical
duplicates, identical decoded payloads appearing under both source labels, and
payload hashes shared between prepared train/validation/test splits. QR
payloads are hashed in the report rather than printed; no URL is opened,
resolved, or contacted. The report explicitly marks source-folder labels as
not independently verified. Exact and pixel-identical hash checks do not find
resized or otherwise modified duplicates, and matching payloads alone do not
prove that a label is wrong or right.

Audit of the current supplied images found 1,000 readable source files and
decoded 988 QR payloads (496/500 in `benign_qr_images_500`, 492/500 in
`malicious_qr_images_500`). Twelve QR images were not decoded. No exact file
duplicates, pixel-identical duplicates, exact decoded-payload matches across
the two source labels, or exact decoded-payload matches across prepared splits
were found. All decoded payload strings looked like web addresses by a simple
prefix check, which does not establish destination safety or validate the
labels. The source dimensions also differ by label: the benign-named folder
contains 290–450 pixel images while the malicious-named folder contains
290–1010 pixel images. Although preprocessing resizes images to 224 x 224, this
source imbalance warrants review for possible dataset-generation or image-style
bias. These observations cannot establish the provenance or correctness of
either label without trusted source records.

## Scan a QR image and inspect its URL

Run the combined scanner with an image path:

```powershell
python -m src.scan_qr .\path\to\qr-image.png
```

The JSON report includes the CNN's visual label and uncalibrated class scores,
QR decode status, and basic URL warning signals such as HTTP instead of HTTPS,
embedded credentials, IP-address hosts, punycode labels, and non-standard
ports. These are separate signals: the CNN does not verify the URL, and URL
warnings do not change the CNN result. If the decoded text is not an HTTP(S)
URL, it is reported as non-web content. The payload is hidden by default; use
`--show-payload` only when you explicitly want it included:

```powershell
python -m src.scan_qr .\path\to\qr-image.png --show-payload
```

Save the JSON report to a file with:

```powershell
python -m src.scan_qr .\path\to\qr-image.png --output .\reports\scan.json
```

This analysis is entirely local: it does not open the URL, make a DNS lookup,
follow redirects, or query a reputation service. Warnings are heuristic review
signals, not a verdict that a URL is malicious. No warnings do not mean a URL
is safe. QR decoding failure is recorded while still returning the CNN's image
classification. Treat scanned QR destinations as untrusted and inspect them
before opening.

## Interactive local interface

### Quick start (this project environment)

Double-click `run_qr_shield.bat`, or run it from PowerShell:

```powershell
.\run_qr_shield.bat
```

The launcher uses `.venv311` when present, otherwise `.venv`. Keep the
terminal window open while using the app and visit the local address it
prints (normally `http://localhost:8501`).

### Manual launch

Install the requirements in the selected virtual environment, then launch the
Streamlit interface from the project folder:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run .\frontend\app.py
```

Open the local address printed by Streamlit in your browser. Upload up to five
QR images per batch (each up to 10 MB) and select **Analyze**. The classifier
is loaded once per batch. Each image gets separate CNN and QR/URL results, an
individual JSON report, and the batch has a downloadable CSV summary. Decoded
payloads are hidden by default; they can be opted into individual JSON reports
but are never included in the CSV. The **Model & dataset** tab shows saved
test-set results and the dataset audit; **How to use** has quick-start
instructions. Uploaded images are held in memory and temporary copies are
removed after analysis. The UI does not make requests to QR destinations. It
is a local development interface, not a hardened public service.

To run the current Windows environment without activating it first:

```powershell
.\.venv311\Scripts\python.exe -m streamlit run .\frontend\app.py
```

### Deploy to Streamlit Community Cloud

The existing GitHub remote in the parent folder is a different project. Use a
dedicated public GitHub repository for QR Shield AI, with the contents of this
project folder at the repository root; do not publish the parent folder.

The repository needs the app source, `requirements.txt`, reports, and
`models/baseline_cnn.pth`. The project `.gitignore` excludes local virtual
environments and dataset images while allowing this small checkpoint. Review
the Git status before pushing to confirm no dataset images, environment files,
or credentials are included.

On Streamlit Community Cloud, create an app from that repository, select the
`main` branch, set the app file to `frontend/app.py`, and choose Python 3.11.
No secrets are required. The dashboard reads the committed test and audit
reports; the scanner uses the committed checkpoint.

**Cloud privacy:** scans are processed by the hosting server, not your PC. Do
not upload QR images containing private or sensitive information. The app
does not open destinations, but it is a public research demo without user
authentication or production security controls. Model predictions and URL
warnings are not safety verdicts.

## Initial configuration

`config.py` defines the project, dataset, and model paths; image size
`224 x 224`; batch size `32`; `20` epochs; learning rate `0.0001`; `2` classes;
and automatic CUDA/CPU device selection.
