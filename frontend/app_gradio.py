# frontend/app_gradio.py
# Gradio 6.x compatible frontend for Agentic Data Quality Auditor (CV)
# - Centered header + slogan
# - Form-style steps
# - Required fields enforced (dataset path + output dir)
# - Strict dataset validation (must contain happy/ and sad/)
# - Runs REAL agent via main.py using the SAME venv python (sys.executable)
# - Shows report + plots
# - Download buttons: TXT report, JSON report, cleaned dataset ZIP (if --fix)

import sys
import subprocess
import shutil
from pathlib import Path
import gradio as gr

# ---------------------------------------------------------
# Make project root importable (so paths are stable)
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _count_images(folder: Path) -> int:
    if not folder.exists() or not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)


def validate_dataset_structure(dataset_path: str):
    """
    Returns:
      ok (bool),
      msg (str),
      happy_count (int),
      sad_count (int)
    """
    ds = Path((dataset_path or "").strip().strip('"'))

    if not str(ds):
        return False, "Dataset path is required.", 0, 0

    if not ds.exists():
        return False, f"Dataset path does not exist: {ds}", 0, 0

    if not ds.is_dir():
        return False, f"Dataset path is not a folder: {ds}", 0, 0

    happy_dir = ds / "happy"
    sad_dir = ds / "sad"

    if not happy_dir.exists() or not happy_dir.is_dir():
        return False, "Missing folder: dataset/happy", 0, 0

    if not sad_dir.exists() or not sad_dir.is_dir():
        return False, "Missing folder: dataset/sad", 0, 0

    happy_count = _count_images(happy_dir)
    sad_count = _count_images(sad_dir)

    if happy_count == 0:
        return False, "No images found inside dataset/happy", happy_count, sad_count
    if sad_count == 0:
        return False, "No images found inside dataset/sad", happy_count, sad_count

    msg = (
        f"✅ Dataset looks valid.\n"
        f"• happy: {happy_count} images\n"
        f"• sad: {sad_count} images"
    )
    return True, msg, happy_count, sad_count


def run_agent_cli_stream(dataset_path, output_dir, do_fix, do_eval, eval_epochs, no_viz):
    dataset_path = (dataset_path or "").strip().strip('"')
    output_dir = (output_dir or "output").strip().strip('"')

    ok, msg, _, _ = validate_dataset_structure(dataset_path)
    if not ok:
        yield f"❌ {msg}", [], None, None, None
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(ROOT_DIR / "main.py"),
        "--dataset", dataset_path,
        "--output", output_dir,
    ]
    if do_fix:
        cmd.append("--fix")
    if do_eval:
        cmd.extend(["--evaluate", "--eval-epochs", str(int(eval_epochs))])
    if no_viz:
        cmd.append("--no-viz")

    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    live_log = ""
    for line in process.stdout:
        live_log += line
        yield live_log, [], None, None, None   # stream logs

    process.wait()

    if process.returncode != 0:
        yield live_log + "\n❌ Agent crashed.", [], None, None, None
        return

    # After completion → load files
    txt_report = out_dir / "quality_report.txt"
    json_report = out_dir / "quality_report.json"

    report_text = txt_report.read_text(encoding="utf-8", errors="replace") if txt_report.exists() else live_log
    plot_files = [] if no_viz else [str(p) for p in out_dir.glob("*.png")]

    cleaned_zip = None
    cleaned_dir = ROOT_DIR / "cleaned_dataset"
    if do_fix and cleaned_dir.exists():
        zip_base = out_dir / "cleaned_dataset"
        cleaned_zip = shutil.make_archive(str(zip_base), "zip", cleaned_dir)

    yield report_text, plot_files, str(txt_report), str(json_report), cleaned_zip



# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
def build_app():
    with gr.Blocks(title="Agentic Data Quality Auditor") as demo:
        # Centered header + slogan
        gr.Markdown(
            """
            <div style="text-align:center; margin-top: 10px; margin-bottom: 18px;">
              <h1 style="margin: 0; padding: 0;">Agentic Data Quality Auditor for Computer Vision</h1>
              <div style="margin-top: 8px; font-size: 16px;">
                Audit → Fix → Verify your dataset before training
              </div>
            </div>
            """
        )

        # STEP 1
        with gr.Group():
            gr.Markdown("### Step 1 — Dataset Configuration")

            dataset_path = gr.Textbox(
                label="Dataset folder path",
                placeholder=r"Example: D:\...\agentic-data-quality-auditor-cv\dataset",
                lines=1,
            )

            output_dir = gr.Textbox(
                label="Output folder",
                placeholder="Example: output",
                value="output",
                lines=1,
            )

            dataset_status = gr.Textbox(
                label="Dataset check",
                value="Fill dataset path to validate...",
                lines=3,
                interactive=False,
            )

        # STEP 2
        with gr.Group():
            gr.Markdown("### Step 2 — Options (Optional)")

            with gr.Row():
                do_fix = gr.Checkbox(label="Apply auto-fixes", value=False)
                do_eval = gr.Checkbox(label="Evaluate model", value=False)
                no_viz = gr.Checkbox(label="Skip visualizations", value=False)

            eval_epochs = gr.Slider(
                label="Evaluation epochs",
                minimum=1,
                maximum=30,
                value=5,
                step=1,
                visible=False,
            )

        def toggle_epochs(evaluate_checked):
            return gr.update(visible=bool(evaluate_checked))

        do_eval.change(toggle_epochs, inputs=[do_eval], outputs=[eval_epochs])

        # STEP 3
        with gr.Group():
            gr.Markdown("### Step 3 — Run")
            run_btn = gr.Button("Run Audit", variant="primary", interactive=False)
            run_hint = gr.Markdown("")

        # Enable Run only when dataset valid AND output filled
        def validate_form(ds_path, out_dir):
            out_ok = bool((out_dir or "").strip())
            ok, msg, _, _ = validate_dataset_structure(ds_path)

            # Update dataset status box
            status_text = msg if ok else f"❌ {msg}"

            # Enable run only if all required valid
            can_run = ok and out_ok
            hint = "" if can_run else "Please provide a valid dataset path and output folder."

            return status_text, gr.update(interactive=can_run), hint

        dataset_path.change(
            validate_form,
            inputs=[dataset_path, output_dir],
            outputs=[dataset_status, run_btn, run_hint],
        )
        output_dir.change(
            validate_form,
            inputs=[dataset_path, output_dir],
            outputs=[dataset_status, run_btn, run_hint],
        )

        # RESULTS
        with gr.Group():
            gr.Markdown("### Results")

            report_out = gr.Textbox(
                label="Report",
                lines=18,
                interactive=False,
            )

            plots_out = gr.Gallery(
                label="Visualizations",
                columns=3,
                height="auto",
            )

            with gr.Row():
                txt_download = gr.File(label="Download report (TXT)")
                json_download = gr.File(label="Download report (JSON)")
                cleaned_zip = gr.File(label="Download cleaned dataset (ZIP)")

        # Run action
        def on_run(ds_path, out_dir, fix, evaluate, epochs, skip_viz):
            return run_agent_cli_stream(ds_path, out_dir, fix, evaluate, epochs, skip_viz)

        run_btn.click(
    fn=run_agent_cli_stream,
    inputs=[dataset_path, output_dir, do_fix, do_eval, eval_epochs, no_viz],
    outputs=[report_out, plots_out, txt_download, json_download, cleaned_zip],
        )

        gr.Markdown(
            """
            <div style="margin-top: 14px; font-size: 13px; opacity: 0.9;">
              <b>Required structure:</b> <code>dataset/happy/*</code> and <code>dataset/sad/*</code><br/>
              Tip: If evaluation is enabled and CUDA is available, it will use GPU automatically.
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
