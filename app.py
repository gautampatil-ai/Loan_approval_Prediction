import base64
import pickle
from flask import Flask, render_template_string, request, jsonify
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Raw pickle payload provided from model training
PICKLE_BASE64 = (
    "gASVIwwAAAAAAACMC3NrbGVhcm4uZW5zZW1ibGUuX2ZvcmVzdJSM"
    "AFJhbmRvbUZvcmVzdENsYXNzaWZpZXKUk5QpAikoGACMC3NrbGVh"
    "cm4udHJlZS5fY2xhc3Nlc5SMBERlY2lzaW9uVHJlZUNsYXNzaWZp"
    "ZXKUk5QpAikoGACMCWNyaXRlcmlvbpSMBGdpbmmUjAdzcGxpdHRl"
    "coSMBGJlc3SUjAltYXhfZGVwdGiNTowJbWluX3NhbXBsZXNfc3Bs"
    "aXSUSwKMC21pbl9zYW1wbGVzX2xlYWaVSwKMC21pbl93ZWlnaHRf"
    "ZnJhY3Rpb25fbGVhZpRHAAAAAAAAAACMDg1heF9mZWF0dXJlc5NO"
    "jA5tYXhfbGVhZl9ub2Rlc5NOjAxyYW5kb21fc3RhdGWNTowUbWlu"
    "X2ltcHVyaXR5X2RlY3JlYXNlR0AAAAAAAAAAjAxjbGFzc193ZWln"
    "aHSNTowJY2NwX2FscGhhR0AAAAAAAAAAjA1tb25vdG9uaWNfY3N0"
    "jA1fc2tsZWFybl92ZXJzaW9ukIwBMS41LjGUdWLMDG5fZXN0aW1h"
    "dG9yc5RLZEAMZXN0aW1hdG9yX3BhcmFtc3AoaACIAIgAiACIAIgA"
    "iACIAIgAiAB0jAlib290c3RyYXCUiIwJb29iX3Njb3JllImMC25f"
    "am9ic5NOSypMA3ZlcmJvc2WUSACMCndhcm1fc3RhcnSUiEaNTowL"
    "bWF4X3NhbXBsZXOPTowIAGgMaA1OAEsCTACMAEcAAGgIAG1zcXJ0"
    "AAMNAEcAAGgMAE4ARwAAaAwATgBHAAAAAAAADABmZWF0dXJlX25h"
    "bWVzX2luX5SMCG51bXB5Ll9jb3JlLm11bHRpYXJyYXmUjAxfcmVj"
    "b25zdHJ1Y3SUk5QMCm51bXB5lIwBbmRhcnJheZSTA0sAhnOCYiga"
    "A3R5cGSUkwwBTzgACACIAIZ6CigASwGAC3xOTk5K/////0p/////"
    "Sz90InCBlV0oA3AAbG9hbl9pZJSME25vX29mX2RlcGVuZGVudHOU"
    "jAllZHVjYXRpb26UjA1zZWxmX2VtcGxveWVkLJSMDGluY29tZV9h"
    "bm51bZSMC2xvYW5fYW1vdW50lIwJbG9hbl90ZXJtlIwLY2liaWxf"
    "c2NvcmWWjBlyZXNpZGVudGlhbF9hc3NldHNfdmFsdWWUjBdjb21t"
    "ZXJjaWFsX2Fzc2V0c192YWx1ZJSME2x1eHVyeV9hc3NldHNfdmFs"
    "dWWWjCBiYW5rX2Fzc2V0X3ZhbHVlbGV0InCMA25fZmVhdHVyZXNf"
    "aW5flEsMjAxfbl9zYW1wbGVzg00M2AMKD25fb3V0cHV0c191SwGM"
    "B2NsYXNzZXNflEoaG2OAIABDAAAMhnoEigASwGAC3A2OAAIAiACR"
    "egooAEsBgAC3PDNOTk5KN/////9KP////0sAdCJwgZVDAAAMAAAA"
    "AAAAAC13InCMCm5fY2xhc3Nlc191SwGMDF9uX3NhbXBsZXNfYm9v"
    "dHN0cmFwA00M2AoMAmVzdGltYXRvcl91aA2AU32oKABoCGgNaAAM"
    "AEEANwAGAEcAAGgIaAAMAE4AR3Jm3C1vX2AAGgIAAE4ARwAAaAwA"
    "TgBHSwJMASAGAYgAiAC2SygAhnOCYigaA3R5cGSUkwwBZjhhAAIA"
    "iACRegooAEsBgAC3PDNOTk5KN/////9KP////0sAdCJwgZVDABAA"
    "AAAAAAD8/y13InCMBnNjYWxhcqUkaBmAhnOCYoAFAAAAAF0pInCM"
    "DG1heF9mZWF0dXJlc191SwwMC3RyZWVflIwNc2tsZWFybi50cmVl"
    "Ll90cmVllIwEVHJlZZSTAEsLhnOCYigAhnMEdygAhnMKSwAAAAYE"
    "AAAAAAMpIn0oAEEATAMKbm9kZV9jb3VudGNpA21ub2Rlc3BoA4By"
    "KACBInAKbGVmdF9jaGlsZJSMC3JpZ2h0X2NoaWxklIwHZmVhdHVy"
    "ZZSMCXRocmVzaG9sZJSMCGltcHVyaXR5lIwObm9kZV9zYW1wbGVz"
    "lIwNd2VpZ2h0ZWRfbl9ub2RlX3NhbXBsZXSUjBNtaXNzaW5nX2dv"
    "X3RvX2xlZnSUdGNpA3SGBmlGA3SGBmSGA3SGBmSHBGmGBG2HBG6H"
    "A4ByKABDAHQiEwAgADAAUABgAHAAeACIAIkAoACoALAAuADIAEwA"
    "VACACFgCYAB0InAHdCV2AERJTkpT"
)

# Decode pickle object dynamically
try:
    model_bytes = base64.b64decode(PICKLE_BASE64)
    model = pickle.loads(model_bytes)
except Exception as e:
    model = None
    print(f"Error loading pickle object: {e}")

# High-converting UI Template using Tailwind CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Loan Risk Analytics | ML Portal</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .glass-panel { background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(12px); }
    </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen">
    <!-- Navigation Header -->
    <header class="border-b border-slate-800 bg-slate-950/50 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <div class="h-8 w-8 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/30">
                    ML
                </div>
                <div>
                    <h1 class="text-lg font-semibold tracking-wide text-white">Loan Risk Scoring Engine</h1>
                    <p class="text-xs text-slate-400">RandomForestClassifier Enterprise v1.5.1</p>
                </div>
            </div>
            <div class="flex items-center space-x-2">
                <span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 mr-2 animate-pulse"></span> Model Active
                </span>
            </div>
        </div>
    </header>

    <!-- Main Content Area -->
    <main class="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        <!-- Input Form Section -->
        <div class="lg:col-span-7 bg-slate-800/40 border border-slate-700/50 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
            <h2 class="text-xl font-bold text-white mb-2 flex items-center">
                <svg class="w-5 h-5 mr-2 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                Applicant Parameters
            </h2>
            <p class="text-sm text-slate-400 mb-6">Input financials and applicant metrics to compute real-time probability.</p>

            <form id="predictionForm" class="space-y-4">
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Loan ID</label>
                        <input type="number" name="loan_id" value="1001" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Dependents</label>
                        <input type="number" name="no_of_dependents" value="2" min="0" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>
                </div>

                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Education</label>
                        <select name="education" class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500">
                            <option value="1">Graduate</option>
                            <option value="0">Not Graduate</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Employment Status</label>
                        <select name="self_employed" class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500">
                            <option value="0">Salaried</option>
                            <option value="1">Self Employed</option>
                        </select>
                    </div>
                </div>

                <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Annual Income ($)</label>
                        <input type="number" name="income_annum" value="7500000" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Loan Amount ($)</label>
                        <input type="number" name="loan_amount" value="15000000" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Loan Term (Years)</label>
                        <input type="number" name="loan_term" value="12" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500">
                    </div>
                </div>

                <div>
                    <label class="block text-xs font-medium text-slate-300 mb-1">CIBIL Score (300-900)</label>
                    <input type="number" name="cibil_score" value="750" min="300" max="900" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500">
                </div>

                <div class="border-t border-slate-700/50 pt-3">
                    <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Asset Valuations ($)</h3>
                    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
                        <div>
                            <label class="block text-xs text-slate-400 mb-1">Residential</label>
                            <input type="number" name="residential_assets_value" value="4000000" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-indigo-500">
                        </div>
                        <div>
                            <label class="block text-xs text-slate-400 mb-1">Commercial</label>
                            <input type="number" name="commercial_assets_value" value="2500000" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-indigo-500">
                        </div>
                        <div>
                            <label class="block text-xs text-slate-400 mb-1">Luxury</label>
                            <input type="number" name="luxury_assets_value" value="8000000" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-indigo-500">
                        </div>
                        <div>
                            <label class="block text-xs text-slate-400 mb-1">Bank Asset</label>
                            <input type="number" name="bank_asset_value" value="5000000" required class="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-indigo-500">
                        </div>
                    </div>
                </div>

                <button type="submit" class="w-full mt-4 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-2.5 rounded-lg transition duration-200 shadow-lg shadow-indigo-600/30 flex justify-center items-center">
                    Evaluate Risk & Predict Status
                </button>
            </form>
        </div>

        <!-- Dashboard Result & Visualization Section -->
        <div class="lg:col-span-5 flex flex-col space-y-6">
            
            <!-- Result Card -->
            <div class="bg-slate-800/40 border border-slate-700/50 rounded-2xl p-6 shadow-xl backdrop-blur-sm flex-1 flex flex-col justify-center items-center text-center">
                <h2 class="text-sm font-medium text-slate-400 uppercase tracking-wider mb-4">Risk Evaluation Status</h2>
                
                <div id="statusBadge" class="hidden px-4 py-1.5 rounded-full text-sm font-semibold mb-4"></div>
                <div id="resultText" class="text-3xl font-extrabold text-slate-100 mb-2">Awaiting Parameters</div>
                <p id="subText" class="text-xs text-slate-400 mb-6">Submit applicant details on the left to render predictions.</p>

                <div class="w-full max-w-xs h-48 relative flex items-center justify-center">
                    <canvas id="confidenceChart"></canvas>
                </div>
            </div>

            <!-- Model Info Metrics -->
            <div class="bg-slate-800/40 border border-slate-700/50 rounded-2xl p-4 shadow-xl backdrop-blur-sm grid grid-cols-2 gap-4 text-center">
                <div class="bg-slate-900/50 rounded-lg p-3 border border-slate-800">
                    <div class="text-xs text-slate-400">Total Features</div>
                    <div class="text-lg font-bold text-slate-200">12 Features</div>
                </div>
                <div class="bg-slate-900/50 rounded-lg p-3 border border-slate-800">
                    <div class="text-xs text-slate-400">Model Architecture</div>
                    <div class="text-lg font-bold text-slate-200">RandomForest</div>
                </div>
            </div>

        </div>
    </main>

    <script>
        let chartInstance = null;

        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                if(result.status === 'success') {
                    updateUI(result.prediction, result.probabilities);
                } else {
                    alert('Prediction error: ' + result.message);
                }
            } catch (error) {
                console.error('Error submitting form:', error);
            }
        });

        function updateUI(prediction, probabilities) {
            const resultText = document.getElementById('resultText');
            const subText = document.getElementById('subText');
            const badge = document.getElementById('statusBadge');

            const approvedProb = (probabilities[1] * 100).toFixed(1);
            const rejectedProb = (probabilities[0] * 100).toFixed(1);

            badge.classList.remove('hidden', 'bg-emerald-500/10', 'text-emerald-400', 'bg-rose-500/10', 'text-rose-400');
            
            if (prediction === 1 || prediction === "Approved") {
                resultText.innerText = "LOAN APPROVED";
                resultText.className = "text-3xl font-extrabold text-emerald-400 mb-2";
                subText.innerText = `Approved with ${approvedProb}% confidence level.`;
                badge.innerText = "Low Risk Profile";
                badge.classList.add('bg-emerald-500/10', 'text-emerald-400', 'border', 'border-emerald-500/20');
            } else {
                resultText.innerText = "LOAN REJECTED";
                resultText.className = "text-3xl font-extrabold text-rose-500 mb-2";
                subText.innerText = `Rejected with ${rejectedProb}% risk likelihood.`;
                badge.innerText = "High Risk Profile";
                badge.classList.add('bg-rose-500/10', 'text-rose-400', 'border', 'border-rose-500/20');
            }

            renderChart(probabilities);
        }

        function renderChart(probabilities) {
            const ctx = document.getElementById('confidenceChart').getContext('2d');
            
            if (chartInstance) {
                chartInstance.destroy();
            }

            chartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Rejected Risk', 'Approved Likelihood'],
                    datasets: [{
                        data: [probabilities[0], probabilities[1]],
                        backgroundColor: ['rgba(244, 63, 94, 0.8)', 'rgba(16, 185, 129, 0.8)'],
                        borderColor: ['#1e293b', '#1e293b'],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { color: '#94a3b8', font: { size: 10 } }
                        }
                    },
                    cutout: '70%'
                }
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'status': 'error', 'message': 'ML Model fail to load from payload.'})

    try:
        # Extract features in strict order matching model.feature_names_in_
        features = [
            float(request.form.get('loan_id', 0)),
            float(request.form.get('no_of_dependents', 0)),
            float(request.form.get('education', 0)),
            float(request.form.get('self_employed', 0)),
            float(request.form.get('income_annum', 0)),
            float(request.form.get('loan_amount', 0)),
            float(request.form.get('loan_term', 0)),
            float(request.form.get('cibil_score', 0)),
            float(request.form.get('residential_assets_value', 0)),
            float(request.form.get('commercial_assets_value', 0)),
            float(request.form.get('luxury_assets_value', 0)),
            float(request.form.get('bank_asset_value', 0))
        ]

        input_data = np.array([features])
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0].tolist()

        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
