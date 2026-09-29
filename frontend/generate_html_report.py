import os
import base64

def image_to_base64(img_path):
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    return ""

assets_dir = "/Users/cryptorth/Documents/ml/projects/frontend/assets"

cm_b64 = image_to_base64(os.path.join(assets_dir, "week4_confusion_matrices.png"))
roc_b64 = image_to_base64(os.path.join(assets_dir, "week4_roc_curves.png"))
pr_b64 = image_to_base64(os.path.join(assets_dir, "week4_pr_curves.png"))
lc_b64 = image_to_base64(os.path.join(assets_dir, "week4_learning_curves.png"))
adv_roc_b64 = image_to_base64(os.path.join(assets_dir, "week5_advanced_roc_curves.png"))
fi_b64 = image_to_base64(os.path.join(assets_dir, "week5_feature_importance.png"))
thresh_b64 = image_to_base64(os.path.join(assets_dir, "week5_threshold_tuning.png"))

template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CardioSense — Complete ML Project Report & Technical Walkthrough</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  
  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
  <script>mermaid.initialize({startOnLoad: true, theme: 'dark'});</script>
  <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

  <style>
    :root {
      --bg: #090a0f;
      --card-bg: #12141c;
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #ff4d6d;
      --accent-grad: linear-gradient(135deg, #ff4d6d 0%, #ff758c 100%);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --success: #10b981;
      --warning: #f59e0b;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      line-height: 1.65;
      padding: 40px 20px;
    }

    .container {
      max-width: 1080px;
      margin: 0 auto;
    }

    .report-header {
      text-align: center;
      padding-bottom: 30px;
      margin-bottom: 40px;
      border-bottom: 1px solid var(--card-border);
    }

    .report-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 16px;
      border-radius: 999px;
      background: rgba(255, 77, 109, 0.12);
      border: 1px solid rgba(255, 77, 109, 0.3);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 16px;
    }

    h1 {
      font-family: 'Space Grotesk', sans-serif;
      font-size: clamp(30px, 4vw, 42px);
      font-weight: 700;
      color: #ffffff;
      margin-bottom: 12px;
      letter-spacing: -0.02em;
    }

    .report-subtitle {
      color: var(--text-muted);
      font-size: 16px;
      max-width: 680px;
      margin: 0 auto 24px auto;
    }

    .action-bar {
      display: flex;
      justify-content: center;
      gap: 12px;
      margin-top: 20px;
      flex-wrap: wrap;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 20px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      transition: all 0.2s ease;
      border: none;
    }

    .btn-primary {
      background: var(--accent-grad);
      color: #ffffff;
      box-shadow: 0 4px 16px rgba(255, 77, 109, 0.3);
    }

    .btn-primary:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 22px rgba(255, 77, 109, 0.45);
    }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--card-border);
      color: #ffffff;
    }

    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.12);
    }

    .section {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 32px;
      margin-bottom: 36px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    h2 {
      font-family: 'Space Grotesk', sans-serif;
      font-size: 22px;
      color: #ffffff;
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      gap: 10px;
      padding-bottom: 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    h2 i { color: var(--accent); }

    h3 {
      font-size: 16px;
      color: #e2e8f0;
      margin: 24px 0 12px 0;
    }

    p { margin-bottom: 16px; color: #cbd5e1; font-size: 14.5px; }

    .diagram-wrap {
      background: #0d0f17;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      margin: 20px 0;
      text-align: center;
    }

    .image-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 20px;
      margin: 20px 0;
    }

    @media (max-width: 600px) {
      .image-grid { grid-template-columns: 1fr; }
    }

    .figure-card {
      background: #0d0f17;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .figure-card img {
      width: 100%;
      height: auto;
      display: block;
    }

    .figure-caption {
      padding: 12px 16px;
      font-size: 12.5px;
      color: var(--text-muted);
      border-top: 1px solid rgba(255, 255, 255, 0.05);
      font-weight: 500;
    }

    .table-wrap {
      overflow-x: auto;
      margin: 20px 0;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13.5px;
      text-align: left;
    }

    th {
      background: rgba(255, 255, 255, 0.04);
      color: #94a3b8;
      font-weight: 600;
      padding: 12px 16px;
      border-bottom: 1px solid var(--card-border);
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.05em;
    }

    td {
      padding: 12px 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: #e2e8f0;
    }

    tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }

    .highlight-row td {
      font-weight: 700;
      color: #ffffff;
      background: rgba(255, 77, 109, 0.08);
    }

    .callout {
      background: rgba(255, 77, 109, 0.06);
      border-left: 4px solid var(--accent);
      padding: 16px 20px;
      border-radius: 0 8px 8px 0;
      margin: 20px 0;
    }

    .math-block {
      background: #0d0f17;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      margin: 16px 0;
      overflow-x: auto;
    }

    @media print {
      body { background: #ffffff; color: #000000; padding: 0; }
      .action-bar { display: none !important; }
      .section { background: #ffffff; border: 1px solid #ddd; box-shadow: none; break-inside: avoid; }
      h1, h2, h3 { color: #000000; }
      p, td, th { color: #333333; }
      .highlight-row td { background: #f0f0f0; }
    }
  </style>
</head>
<body>

  <div class="container">
    <header class="report-header">
      <div class="report-badge"><i class="fa-solid fa-heart-pulse"></i> Academic SOP Milestone Project</div>
      <h1>CardioSense Technical Walkthrough</h1>
      <p class="report-subtitle">
        Department of Computer Engineering | End-to-End Cardiovascular Disease Machine Learning System (Weeks 1 to 9).
      </p>

      <div class="action-bar">
        <button class="btn btn-primary" onclick="window.print()"><i class="fa-solid fa-print"></i> Save / Print as PDF</button>
        <a class="btn btn-secondary" href="http://localhost:8000" target="_blank"><i class="fa-solid fa-desktop"></i> Open Web UI (:8000)</a>
        <a class="btn btn-secondary" href="http://localhost:5001/api/health" target="_blank"><i class="fa-solid fa-server"></i> API Health (:5001)</a>
      </div>
    </header>

    <div class="section">
      <h2><i class="fa-solid fa-sitemap"></i> 1. System Architecture & High-Level Overview</h2>
      <p>CardioSense operates across a 3-tier architecture uniting ML research, backend model serving, and a client-side interface.</p>
      
      <div class="diagram-wrap">
        <div class="mermaid">
        graph LR
          subgraph Tier1 [Tier 1: ML Pipeline]
            A[70,000 Records] --> B[Data Cleaning]
            B --> C[Scratch Logistic Reg]
            B --> D[6 Baseline Models]
            D --> E[5-Fold CV & Tuning]
            E --> F[Stacking Ensemble]
            F --> G[Threshold 0.35]
          end
          subgraph Tier2 [Tier 2: Flask API]
            G --> H[app.py :5001]
            H --> I[POST /api/predict]
          end
          subgraph Tier3 [Tier 3: Web UI]
            J[Frontend UI :8000] -->|Async POST| I
            I -->|JSON Score & Breakdown| J
          end
        </div>
      </div>
    </div>

    <div class="section">
      <h2><i class="fa-solid fa-chart-line"></i> 2. Week 4 Model Benchmarks & Diagnostic Plots</h2>
      <p>Performance evaluated on hold-out test set (13,724 records) across 7 classification algorithms.</p>
      
      <div class="image-grid">
        <div class="figure-card">
          <img src="IMG_CM" alt="Confusion Matrices">
          <div class="figure-caption">Figure 2.1: Multi-Model Confusion Matrices (True Negatives, False Positives, False Negatives, True Positives)</div>
        </div>
        <div class="figure-card">
          <img src="IMG_ROC" alt="ROC Curves">
          <div class="figure-caption">Figure 2.2: Receiver Operating Characteristic (ROC) Comparison Curves</div>
        </div>
        <div class="figure-card">
          <img src="IMG_PR" alt="Precision-Recall Curves">
          <div class="figure-caption">Figure 2.3: Precision-Recall Curves across Baseline Classifiers</div>
        </div>
        <div class="figure-card">
          <img src="IMG_LC" alt="Learning Curves">
          <div class="figure-caption">Figure 2.4: Learning Curves for Overfitting vs Underfitting Diagnosis (10% to 100% Training Samples)</div>
        </div>
      </div>

      <h3>Hold-Out Test Evaluation Benchmark Table</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model Name</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>Specificity</th>
              <th>F1-Score</th>
              <th>ROC-AUC</th>
              <th>Overfit Gap</th>
            </tr>
          </thead>
          <tbody>
            <tr class="highlight-row">
              <td>Stacking Meta-Classifier</td>
              <td>73.52%</td>
              <td>0.7508</td>
              <td>0.6958</td>
              <td>0.7735</td>
              <td>0.7222</td>
              <td>0.8037</td>
              <td>+1.18%</td>
            </tr>
            <tr class="highlight-row">
              <td>Tuned XGBoost Booster</td>
              <td>73.44%</td>
              <td>0.7513</td>
              <td>0.6914</td>
              <td>0.7712</td>
              <td>0.7201</td>
              <td>0.8033</td>
              <td>+1.21%</td>
            </tr>
            <tr>
              <td>Gradient Boosting (GBDT)</td>
              <td>73.48%</td>
              <td>0.7515</td>
              <td>0.6922</td>
              <td>0.7725</td>
              <td>0.7206</td>
              <td>0.8031</td>
              <td>+1.32%</td>
            </tr>
            <tr>
              <td>Random Forest</td>
              <td>73.01%</td>
              <td>0.7482</td>
              <td>0.6865</td>
              <td>0.7714</td>
              <td>0.7160</td>
              <td>0.7996</td>
              <td>+2.84%</td>
            </tr>
            <tr>
              <td>Decision Tree</td>
              <td>72.82%</td>
              <td>0.7551</td>
              <td>0.6657</td>
              <td>0.7891</td>
              <td>0.7075</td>
              <td>0.7854</td>
              <td>+2.05%</td>
            </tr>
            <tr class="highlight-row">
              <td>Scratch Logistic Reg</td>
              <td>72.73%</td>
              <td>0.7410</td>
              <td>0.6768</td>
              <td>0.7766</td>
              <td>0.7074</td>
              <td>0.7901</td>
              <td>+0.07%</td>
            </tr>
            <tr>
              <td>Naive Bayes (Gaussian)</td>
              <td>70.92%</td>
              <td>0.7411</td>
              <td>0.6276</td>
              <td>0.7888</td>
              <td>0.6796</td>
              <td>0.7797</td>
              <td>+0.18%</td>
            </tr>
            <tr>
              <td>K-Nearest Neighbors</td>
              <td>70.85%</td>
              <td>0.7118</td>
              <td>0.6896</td>
              <td>0.7269</td>
              <td>0.7005</td>
              <td>0.7632</td>
              <td>+7.21%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="section">
      <h2><i class="fa-solid fa-sliders"></i> 3. Week 5 Ensembles, Feature Importance & Decision Optimization</h2>
      
      <div class="image-grid">
        <div class="figure-card">
          <img src="IMG_ADV_ROC" alt="Advanced ROC Curves">
          <div class="figure-caption">Figure 3.1: ROC Curves for Advanced Ensembles (Stacking, Voting, Tuned Trees)</div>
        </div>
        <div class="figure-card">
          <img src="IMG_FI" alt="Feature Importance">
          <div class="figure-caption">Figure 3.2: Feature Importance Breakdown (XGBoost Gain vs Random Forest Gini)</div>
        </div>
      </div>

      <div class="figure-card" style="margin-top: 20px;">
        <img src="IMG_THRESH" alt="Threshold Optimization Curve">
        <div class="figure-caption">Figure 3.3: Clinical Decision Threshold Sweep (0.10 to 0.90) demonstrating optimal Recall at 0.35</div>
      </div>

      <div class="callout">
        <strong>Clinical Screening Rationale:</strong> At the default threshold of 0.50, the model misses 30.86% of diseased patients. Adjusting to the calibrated screening threshold of <strong>0.35</strong> elevates Recall to <strong>83.92%</strong> (F1 = 0.7410), capturing 84% of true cardiovascular disease cases.
      </div>
    </div>

    <div class="section">
      <h2><i class="fa-solid fa-square-root-variable"></i> 4. Mathematical Foundations (Scratch Logistic Regression)</h2>
      <p>Implemented without scikit-learn to fulfill mandatory academic SOP Phase 3 criteria:</p>

      <div class="math-block">
        $$\\text{Log-Odds:} \\quad z = \\mathbf{w}^T \\mathbf{x} + b = \\sum_{j=1}^{n} w_j x_j + b$$
        $$\\text{Sigmoid Activation:} \\quad \\hat{y} = \\sigma(z) = \\frac{1}{1 + e^{-z}}$$
        $$\\text{Log-Loss:} \\quad J(\\mathbf{w}, b) = -\\frac{1}{m} \\sum_{i=1}^{m} \\left[ y^{(i)} \\log(\\hat{y}^{(i)}) + (1 - y^{(i)}) \\log(1 - \\hat{y}^{(i)}) \\right]$$
        $$\\text{Gradient Updates:} \\quad \\mathbf{w} \\leftarrow \\mathbf{w} - \\alpha \\frac{1}{m}\\mathbf{X}^T(\\mathbf{\\hat{y}} - \\mathbf{y}), \\quad b \\leftarrow b - \\alpha \\frac{1}{m}\\sum_{i=1}^{m}(\\hat{y}^{(i)} - y^{(i)})$$
      </div>
    </div>

    <div class="section">
      <h2><i class="fa-solid fa-laptop-code"></i> 5. Running & Interacting with the Live System</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Service</th>
              <th>Endpoint / Path</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Frontend Web UI</strong></td>
              <td><a href="http://localhost:8000" target="_blank" style="color:var(--accent);">http://localhost:8000</a></td>
              <td>Interactive dashboard with sliders, model selector, score animation</td>
            </tr>
            <tr>
              <td><strong>Flask REST API</strong></td>
              <td><a href="http://localhost:5001" target="_blank" style="color:var(--accent);">http://localhost:5001</a></td>
              <td>Production inference engine (<code>/api/predict</code>, <code>/api/health</code>, <code>/api/metrics</code>)</td>
            </tr>
            <tr>
              <td><strong>Jupyter Notebook</strong></td>
              <td><code>cardioDemo.ipynb</code></td>
              <td>103 pre-executed modular snippets with all charts and tables rendered</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

  </div>

</body>
</html>
"""

html_final = template.replace("IMG_CM", cm_b64) \
                     .replace("IMG_ROC", roc_b64) \
                     .replace("IMG_PR", pr_b64) \
                     .replace("IMG_LC", lc_b64) \
                     .replace("IMG_ADV_ROC", adv_roc_b64) \
                     .replace("IMG_FI", fi_b64) \
                     .replace("IMG_THRESH", thresh_b64)

target_paths = [
    "/Users/cryptorth/Documents/ml/projects/frontend/walkthrough.html"
]

for p in target_paths:
    with open(p, "w", encoding="utf-8") as f:
        f.write(html_final)

print("✅ Generated walkthrough.html successfully!")
