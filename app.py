from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained RandomForest Model
MODEL_PATH = "coursename_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Course Name Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Error loading model file '{MODEL_PATH}': {e}")

# Modern Single-File UI Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MHTCET Course Predictor Portal</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: radial-gradient(circle at 50% -20%, #064e3b, #0f172a, #020617);
        }

        .glass-card {
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .glow-btn {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 0 25px rgba(16, 185, 129, 0.35);
        }

        .glow-btn:hover {
            box-shadow: 0 0 40px rgba(16, 185, 129, 0.75);
            transform: translateY(-2px);
        }

        @keyframes pulseGlow {
            0%, 100% { opacity: 0.3; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.08); }
        }

        .ambient-glow-1 {
            animation: pulseGlow 7s infinite ease-in-out;
        }

        .ambient-glow-2 {
            animation: pulseGlow 9s infinite ease-in-out 2s;
        }
    </style>
</head>
<body class="min-h-screen text-slate-100 flex items-center justify-center p-4 sm:p-6 relative overflow-x-hidden">

    <!-- Ambient background light effects -->
    <div class="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-emerald-600/20 rounded-full blur-[140px] pointer-events-none ambient-glow-1"></div>
    <div class="absolute bottom-10 right-10 w-[400px] h-[400px] bg-teal-600/20 rounded-full blur-[120px] pointer-events-none ambient-glow-2"></div>

    <div class="w-full max-w-4xl z-10 my-8">
        <!-- Header Section -->
        <div class="text-center mb-10">
            <div class="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs sm:text-sm font-semibold tracking-wide uppercase mb-4">
                <i data-lucide="sparkles" class="w-4 h-4"></i> MHTCET Admission AI Predictor
            </div>
            <h1 class="text-4xl sm:text-5xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-emerald-200 via-teal-300 to-cyan-400 tracking-tight">
                Course Name Predictor
            </h1>
            <p class="text-slate-400 mt-3 text-sm sm:text-base max-w-xl mx-auto">
                Fill in candidate details to predict target engineering course allotment using Machine Learning.
            </p>
        </div>

        <!-- Glassmorphism Form Card -->
        <div class="glass-card rounded-3xl p-6 sm:p-10 shadow-2xl relative">
            <form id="predictorForm" class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <!-- 1. Merit Number -->
                <div>
                    <label class="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">1. Merit Number</label>
                    <div class="relative">
                        <i data-lucide="hash" class="absolute left-3.5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" name="merit_num" required placeholder="e.g. 15420" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-xl pl-11 pr-4 py-3.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition">
                    </div>
                </div>

                <!-- 2. MHTCET Percentile -->
                <div>
                    <label class="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">2. MHTCET Percentile</label>
                    <div class="relative">
                        <i data-lucide="percent" class="absolute left-3.5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" step="0.0001" name="percentile" required placeholder="e.g. 96.8540" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-xl pl-11 pr-4 py-3.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition">
                    </div>
                </div>

                <!-- 3. Candidate Name Code -->
                <div>
                    <label class="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">3. Candidate Name Code</label>
                    <div class="relative">
                        <i data-lucide="user" class="absolute left-3.5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" name="name_code" required placeholder="Encoded integer ID" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-xl pl-11 pr-4 py-3.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition">
                    </div>
                </div>

                <!-- 4. Gender Code -->
                <div>
                    <label class="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">4. Gender Code</label>
                    <div class="relative">
                        <i data-lucide="users" class="absolute left-3.5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" name="gender_code" required placeholder="0 for Male, 1 for Female" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-xl pl-11 pr-4 py-3.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition">
                    </div>
                </div>

                <!-- 5. Category Code -->
                <div>
                    <label class="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">5. Category Code</label>
                    <div class="relative">
                        <i data-lucide="layers" class="absolute left-3.5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" name="category_code" required placeholder="Encoded category integer" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-xl pl-11 pr-4 py-3.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition">
                    </div>
                </div>

                <!-- 6. Seat Alloted Code -->
                <div>
                    <label class="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">6. Seat Allotted Code</label>
                    <div class="relative">
                        <i data-lucide="check-square" class="absolute left-3.5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" name="seat_alloted" required placeholder="Encoded seat type integer" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-xl pl-11 pr-4 py-3.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition">
                    </div>
                </div>

                <!-- 7. Seat Number Code -->
                <div>
                    <label class="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">7. Seat Number Code</label>
                    <div class="relative">
                        <i data-lucide="award" class="absolute left-3.5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" name="seat_num" required placeholder="Encoded seat number ID" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-xl pl-11 pr-4 py-3.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition">
                    </div>
                </div>

                <!-- 8. Institute Name Code -->
                <div>
                    <label class="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">8. Institute Name Code</label>
                    <div class="relative">
                        <i data-lucide="building-2" class="absolute left-3.5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" name="institute_code" required placeholder="Encoded institute ID" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-xl pl-11 pr-4 py-3.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition">
                    </div>
                </div>

                <!-- Submit Button -->
                <div class="md:col-span-2 mt-4">
                    <button type="submit" class="glow-btn w-full bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 text-slate-950 font-extrabold py-4 rounded-xl flex items-center justify-center gap-2 transition text-base uppercase tracking-wider">
                        <i data-lucide="cpu" class="w-5 h-5"></i> Predict Course Name
                    </button>
                </div>
            </form>

            <!-- Loading Spinner -->
            <div id="loading" class="hidden text-center my-8">
                <div class="inline-block animate-spin rounded-full h-10 w-10 border-b-2 border-emerald-400"></div>
                <p class="text-sm text-slate-400 mt-3 font-medium">Running prediction through RandomForest Model...</p>
            </div>

            <!-- Result Box -->
            <div id="resultBox" class="hidden mt-8 p-6 bg-gradient-to-br from-emerald-950/60 to-slate-900 border border-emerald-500/40 rounded-2xl text-center shadow-lg">
                <p class="text-xs uppercase font-bold text-emerald-400 tracking-widest">Predicted Course Code / Class</p>
                <div id="resultValue" class="text-3xl sm:text-4xl font-black text-slate-100 mt-2"></div>
            </div>
        </div>
    </div>

    <script>
        lucide.createIcons();

        document.getElementById("predictorForm").addEventListener("submit", async function (e) {
            e.preventDefault();

            const form = e.target;
            const formData = new FormData(form);
            const loading = document.getElementById("loading");
            const resultBox = document.getElementById("resultBox");
            const resultValue = document.getElementById("resultValue");

            loading.classList.remove("hidden");
            resultBox.classList.add("hidden");

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();
                loading.classList.add("hidden");

                if (data.success) {
                    resultValue.innerText = data.prediction;
                    resultBox.classList.remove("hidden");
                } else {
                    alert("Prediction Error: " + data.error);
                }
            } catch (err) {
                loading.classList.add("hidden");
                alert("Server connection failed. Make sure app.py is running.");
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model file 'coursename_model.pkl' is not loaded on server."}), 500

    try:
        # Extract inputs corresponding to the model's 8 expected feature columns
        merit_num = float(request.form.get("merit_num", 0))
        percentile = float(request.form.get("percentile", 0))
        name_code = float(request.form.get("name_code", 0))
        gender_code = float(request.form.get("gender_code", 0))
        category_code = float(request.form.get("category_code", 0))
        seat_alloted = float(request.form.get("seat_alloted", 0))
        seat_num = float(request.form.get("seat_num", 0))
        institute_code = float(request.form.get("institute_code", 0))

        # Format input array as (1, 8)
        features = np.array([[
            merit_num,
            percentile,
            name_code,
            gender_code,
            category_code,
            seat_alloted,
            seat_num,
            institute_code
        ]])

        # Execute Model Prediction
        prediction = model.predict(features)[0]

        return jsonify({
            "success": True, 
            "prediction": str(prediction)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
