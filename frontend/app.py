import copy
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from config import DEVICE, PROJECT_ROOT
from src.scan_qr import DEFAULT_CHECKPOINT, build_scan_report


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"}
AUDIT_REPORT_PATH = PROJECT_ROOT / "reports" / "dataset_audit.json"
TEST_REPORT_PATH = PROJECT_ROOT / "reports" / "test_error_analysis.csv"


def apply_custom_styles() -> None:
    st.markdown(
        """
<style>
:root {
    --qr-ink: #e8f2f5;
    --qr-muted: #9ab0bb;
    --qr-teal: #51e0c0;
    --qr-border: rgba(153, 193, 205, 0.16);
}
.stApp {
    background:
        radial-gradient(ellipse at 82% 0%, rgba(32, 113, 123, 0.19), transparent 34rem),
        linear-gradient(145deg, #08131d 0%, #0b1822 52%, #0d1d27 100%);
}
[data-testid="stMainBlockContainer"] {
    max-width: 1240px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}
[data-testid="stHeader"] {
    background: rgba(8, 19, 29, 0.72);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1923 0%, #0a151f 100%);
}
[data-testid="stSidebar"] > div:first-child {
    border-right: 1px solid var(--qr-border);
}
h1, h2, h3 {
    color: var(--qr-ink);
    letter-spacing: -0.035em;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"] {
    color: var(--qr-muted);
}
.qr-hero {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
    overflow: hidden;
    margin: 0.4rem 0 1.35rem;
    padding: clamp(1.6rem, 4vw, 3rem);
    border: 1px solid rgba(81, 224, 192, 0.24);
    border-radius: 24px;
    background:
        radial-gradient(circle at 78% 45%, rgba(81, 224, 192, 0.13), transparent 15rem),
        linear-gradient(115deg, rgba(18, 42, 52, 0.96), rgba(13, 30, 42, 0.94));
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.2);
}
.qr-hero-copy { position: relative; z-index: 1; max-width: 680px; }
.qr-hero-kicker {
    margin: 0 0 0.8rem;
    color: var(--qr-teal);
    font-size: 0.76rem;
    font-weight: 750;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}
.qr-hero h1 {
    margin: 0;
    color: #f1f8fa;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    line-height: 1.05;
}
.qr-hero-copy > p:last-child {
    max-width: 610px;
    margin: 1rem 0 0;
    color: #b4c7ce;
    font-size: 1.04rem;
    line-height: 1.7;
}
.qr-hero-mark {
    display: grid;
    width: 138px;
    height: 138px;
    flex: 0 0 138px;
    place-items: center;
    border: 1px solid rgba(81, 224, 192, 0.38);
    border-radius: 36px;
    background: linear-gradient(145deg, rgba(81, 224, 192, 0.17), rgba(81, 224, 192, 0.04));
    box-shadow: 0 0 64px rgba(81, 224, 192, 0.1), inset 0 0 28px rgba(81, 224, 192, 0.08);
    color: var(--qr-teal);
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.08em;
}
[data-testid="stTabs"] [role="tablist"] {
    gap: 0.45rem;
    border-bottom: 1px solid var(--qr-border);
}
[data-testid="stTabs"] button[role="tab"] {
    min-height: 3rem;
    border-radius: 12px 12px 0 0;
    color: #a9bdc6;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--qr-teal);
    background: rgba(81, 224, 192, 0.08);
}
[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed rgba(81, 224, 192, 0.46);
    border-radius: 16px;
    background: rgba(81, 224, 192, 0.035);
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--qr-teal);
    background: rgba(81, 224, 192, 0.07);
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--qr-border);
    border-radius: 18px;
    background: rgba(16, 34, 45, 0.62);
}
[data-testid="stMetric"] {
    padding: 1rem 1.1rem;
    border: 1px solid var(--qr-border);
    border-radius: 16px;
    background: rgba(16, 34, 45, 0.72);
}
[data-testid="stMetricLabel"] p { color: #a9bdc6; }
[data-testid="stMetricValue"] { color: #e9f7f5; }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stImage"] img { border: 1px solid var(--qr-border); border-radius: 16px; }
button[kind="primary"] {
    min-height: 3rem;
    border: 1px solid rgba(81, 224, 192, 0.35);
    border-radius: 12px;
    background: linear-gradient(100deg, #168f83, #16a68d);
    color: #f4fffc;
    font-weight: 700;
    transition: filter 120ms ease, transform 120ms ease;
}
button[kind="primary"] p,
button[kind="primary"] span {
    color: #f4fffc;
}
button[kind="primary"]:hover { filter: brightness(1.12); transform: translateY(-1px); }
button:focus-visible { outline: 2px solid var(--qr-teal) !important; outline-offset: 3px; }
@media (max-width: 700px) {
    .qr-hero { padding: 1.5rem; border-radius: 18px; }
    .qr-hero-mark { display: none; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def load_project_reports() -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    audit: dict[str, Any] | None = None
    test_rows: list[dict[str, str]] = []
    try:
        if AUDIT_REPORT_PATH.is_file():
            audit = json.loads(AUDIT_REPORT_PATH.read_text(encoding="utf-8"))
        if TEST_REPORT_PATH.is_file():
            with TEST_REPORT_PATH.open(newline="", encoding="utf-8") as report_file:
                test_rows = list(csv.DictReader(report_file))
    except (OSError, json.JSONDecodeError, csv.Error) as error:
        st.warning(f"Could not load a saved project report: {error}")
    return audit, test_rows


def calculate_test_summary(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    if not rows:
        return None

    labels = ("fake", "real")
    confusion = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for row in rows:
        actual = row["true_label"]
        predicted = row["predicted_label"]
        if actual in confusion and predicted in confusion[actual]:
            confusion[actual][predicted] += 1

    count = len(rows)
    correct = sum(row["correct"].lower() == "true" for row in rows)
    return {
        "sample_count": count,
        "accuracy": correct / count,
        "fake_missed": confusion["fake"]["real"],
        "fake_count": sum(confusion["fake"].values()),
        "real_false_alarms": confusion["real"]["fake"],
        "real_count": sum(confusion["real"].values()),
        "confusion": confusion,
    }


def render_scan_tab() -> None:
    st.subheader("Scan a QR image")
    st.write(
        "The CNN's image classification and the decoded QR/URL review are "
        "separate signals. Neither is a safety verdict."
    )
    upload_column, info_column = st.columns([1.35, 0.85], gap="large")
    with upload_column:
        uploaded_file = st.file_uploader(
            "Choose a QR image",
            type=sorted(ALLOWED_EXTENSIONS),
            help="Supported image files up to 10 MB.",
            key="qr_upload",
        )

    with info_column:
        with st.container(border=True):
            st.markdown("#### A safer way to review")
            st.markdown("**01 · Analyze the image**  \nThe app processes your upload.")
            st.markdown("**02 · Check the signals**  \nModel and URL findings stay separate.")
            st.markdown("**03 · Stay in control**  \nThe decoded payload is hidden by default.")
            st.caption(
                "When hosted online, your image is sent to the app host for "
                "analysis. QR destinations are never contacted."
            )

    if uploaded_file is None:
        st.info("Choose an image above to begin your local scan.")
        return

    image_bytes = uploaded_file.getvalue()
    upload_key = hashlib.sha256(image_bytes).hexdigest()
    if st.session_state.get("scan_upload_key") != upload_key:
        st.session_state.pop("scan_report", None)
        st.session_state["scan_upload_key"] = upload_key

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        st.error("This image is larger than the 10 MB upload limit.")
        return

    with upload_column:
        st.image(image_bytes, caption=uploaded_file.name, width="stretch")
        show_payload = st.checkbox(
            "Include the full decoded payload in the report",
            value=False,
            help="The payload may contain private information. It is hidden by default.",
        )

    if not DEFAULT_CHECKPOINT.is_file():
        st.error(f"Model checkpoint not found: {DEFAULT_CHECKPOINT}")
        return

    if upload_column.button(
        "Analyze QR image",
        type="primary",
        width="stretch",
    ):
        suffix = Path(uploaded_file.name).suffix.lower()
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                prefix="qr-shield-",
                delete=False,
            ) as temporary_file:
                temporary_file.write(image_bytes)
                temp_path = Path(temporary_file.name)

            with st.spinner("Analyzing locally..."):
                report = build_scan_report(
                    temp_path,
                    show_payload=show_payload,
                )
            report["image"] = uploaded_file.name
            st.session_state["scan_report"] = report
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
            st.error(f"Analysis failed: {error}")
            return
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    report = st.session_state.get("scan_report")
    if not report:
        return

    visual = report["visual_classification"]
    qr = report["qr"]
    st.divider()
    visual_column, review_column = st.columns(2, gap="large")
    with visual_column:
        with st.container(border=True):
            st.subheader("Visual classification")
            prediction = visual["predicted_label"]
            if prediction == "fake":
                st.error(f"CNN prediction: **{prediction.upper()}**")
            else:
                st.info(f"CNN prediction: **{prediction.upper()}**")
            st.caption(visual["score_note"])
            st.bar_chart(visual["scores"], horizontal=True)

    with review_column:
        with st.container(border=True):
            st.subheader("QR and URL review")
            if qr["decoded"]:
                assessment = qr["url_assessment"]
                st.write(f"Decoded QR code ({qr['payload_length']} payload characters).")
                st.write(f"Assessment: **{assessment['status']}**")
                if assessment["host"]:
                    st.write(f"Host: `{assessment['host']}`")
                if assessment["warnings"]:
                    for warning in assessment["warnings"]:
                        st.warning(warning)
                else:
                    st.info(
                        "No configured heuristic warning found. This does not mean "
                        "the destination is safe."
                    )
                if show_payload and "payload" in qr:
                    st.code(qr["payload"], language=None)
            else:
                st.warning(f"QR decode failed: {qr['error']}")
                st.caption("The CNN image classification is still shown above.")

    with st.expander("Important limitations", expanded=True):
        for limitation in report["limitations"]:
            st.write(f"- {limitation}")

    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    if not show_payload:
        report_for_download = copy.deepcopy(report)
        report_for_download["qr"].pop("payload", None)
        report_json = json.dumps(report_for_download, indent=2, ensure_ascii=False)
    st.download_button(
        "Download JSON report",
        data=report_json,
        file_name="qr_shield_scan_report.json",
        mime="application/json",
        width="stretch",
    )


def render_dashboard_tab(
    audit: dict[str, Any] | None,
    test_rows: list[dict[str, str]],
) -> None:
    st.subheader("Project results")
    summary = calculate_test_summary(test_rows)
    source_totals = audit.get("source_totals", {}) if audit else {}

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Test accuracy",
        f"{summary['accuracy']:.1%}" if summary else "Not available",
    )
    metric_columns[1].metric(
        "Fake missed as real",
        f"{summary['fake_missed']}/{summary['fake_count']}" if summary else "Not available",
        help="These fake-labeled test images were classified as real.",
    )
    metric_columns[2].metric(
        "QRs decoded",
        (
            f"{source_totals.get('decoded_qr_count', 0)}/"
            f"{source_totals.get('image_count', 0)}"
            if audit
            else "Not available"
        ),
    )
    metric_columns[3].metric(
        "Undecodable QR images",
        str(source_totals.get("undecodable_qr_count", "Not available")),
    )

    if summary:
        st.warning(
            f"On this test split, the model missed {summary['fake_missed']} of "
            f"{summary['fake_count']} fake-labeled images. This baseline is not "
            "reliable for security decisions."
        )
        st.markdown("**Confusion matrix** — rows are actual labels; columns are predictions.")
        st.dataframe(
            [
                {
                    "Actual label": label,
                    "Predicted fake": summary["confusion"][label]["fake"],
                    "Predicted real": summary["confusion"][label]["real"],
                }
                for label in ("fake", "real")
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("Test-set results are not available. Run `python -m src.train` first.")

    st.subheader("Dataset audit")
    if audit:
        label_columns = st.columns(2)
        for column, label in zip(label_columns, ("real", "fake")):
            details = audit["by_label"][label]
            column.markdown(f"**{label.title()}-labeled source**")
            column.write(
                f"{details['decoded_qr_count']}/{details['source_image_count']} "
                f"QRs decoded; {details['undecodable_qr_count']} undecodable."
            )
        duplicates = audit["duplicate_checks"]
        st.write(
            "Exact file duplicate groups: "
            f"{len(duplicates['exact_file_duplicate_groups'])} · "
            "pixel-identical groups: "
            f"{len(duplicates['pixel_identical_duplicate_groups'])} · "
            "cross-label payload matches: "
            f"{len(duplicates['cross_label_identical_payloads'])}"
        )
        st.caption(
            "Folder labels have not been independently verified. QR decode "
            "success does not establish destination safety."
        )
        st.download_button(
            "Download dataset audit JSON",
            data=json.dumps(audit, indent=2, ensure_ascii=False),
            file_name="dataset_audit.json",
            mime="application/json",
            key="download_audit",
        )
    else:
        st.info("Dataset audit is unavailable. Run `python -m src.audit_dataset`.")


st.set_page_config(
    page_title="QR Shield AI",
    page_icon="🛡️",
    layout="wide",
)

apply_custom_styles()
st.markdown(
    """
<section class="qr-hero">
  <div class="qr-hero-copy">
    <p class="qr-hero-kicker">Private by design · Local analysis</p>
    <h1>Check before<br>you scan.</h1>
    <p>Review a QR image, its decoded destination, and the model's visual signal
       in one place. Images are processed by this app; QR destinations are never contacted.</p>
  </div>
  <div class="qr-hero-mark" aria-hidden="true">QR</div>
</section>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "Research prototype only. The image model has missed fake-labeled QR codes "
    "in testing. Do not use it to make security decisions."
)

with st.sidebar:
    st.markdown("### QR Shield AI")
    st.caption("RESEARCH SCREENING PROTOTYPE")
    st.divider()
    st.markdown("**System status**")
    if DEFAULT_CHECKPOINT.is_file():
        st.success("Baseline CNN checkpoint ready")
    else:
        st.error("Model checkpoint not found")
    st.write("Device selected by PyTorch:")
    st.code(str(DEVICE))
    st.write("QR and URL checks run locally. No DNS, browsing, or reputation lookup.")

audit_report, test_report_rows = load_project_reports()
scan_tab, dashboard_tab, guide_tab = st.tabs(
    ["Scan a QR", "Model & dataset", "How to use"]
)

with scan_tab:
    render_scan_tab()

with dashboard_tab:
    render_dashboard_tab(audit_report, test_report_rows)

with guide_tab:
    st.subheader("Run QR Shield AI")
    st.markdown(
        """
1. Choose **Scan a QR** and upload an image (maximum 10 MB).
2. Select **Analyze QR image**.
3. Review the CNN prediction separately from the QR/URL signals.
4. Download the JSON report if needed.

The decoded payload is hidden unless you opt in. Never open an unfamiliar QR
destination based only on this prototype's output. If hosted online, uploaded
images are sent to the hosting server for analysis.
        """
    )
    st.code(
        r".\.venv311\Scripts\python.exe -m streamlit run .\frontend\app.py",
        language="powershell",
    )
    st.markdown(
        "For model training, use `python -m src.train`. "
        "For dataset auditing, use `python -m src.audit_dataset`. "
        "For a separate labeled dataset, see the external-evaluation guide in README."
    )
