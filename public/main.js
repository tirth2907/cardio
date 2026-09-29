/**
 * CardioSense — Multi-Page Navigation & Real ML Prediction Engine
 */

document.addEventListener("DOMContentLoaded", () => {
    // Model parameters trained on cardio_train.csv
    const MODEL_WEIGHTS = {
        features: ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'pulse_pressure'],
        coefficients: [0.34844, -0.00928, -0.09994, 0.29477, 0.45451, 0.38436, 0.33281, -0.06437, -0.04628, -0.05387, -0.09037, -0.13229, 0.34019],
        intercept: 0.02968,
        scaler_means: [53.27278, 1.34782, 164.42847, 74.15171, 126.68886, 81.30077, 1.36213, 1.22446, 0.08841, 0.05330, 0.80387, 27.46608, 45.38809],
        scaler_scales: [6.75972, 0.47628, 7.89143, 14.23569, 16.74058, 9.44433, 0.67650, 0.57017, 0.28390, 0.22464, 0.39707, 5.22646, 11.69559],
        xgb_importance: {
            "ap_hi": 0.6388,
            "cholesterol": 0.1188,
            "age_years": 0.0597,
            "active": 0.0264,
            "smoke": 0.0262,
            "ap_lo": 0.0205,
            "alco": 0.0205,
            "gluc": 0.0187,
            "bmi": 0.0173,
            "weight": 0.0164,
            "pulse_pressure": 0.0141,
            "gender": 0.0124,
            "height": 0.0102
        }
    };

    // Dynamic API URL: points to local Flask (:5001) during local dev, or same-origin on Vercel
    const isLocalhost = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") && window.location.port !== "";
    const API_BASE_URL = isLocalhost ? (window.location.port === "5001" ? "" : "http://localhost:5001") : "";
    let isBackendOnline = false;

    // ── DOM Elements ────────────────────────────────────────────────────────
    const burgerBtn = document.getElementById("burger-btn");
    const mobileOverlay = document.getElementById("mobile-overlay");
    const mobileMenu = document.getElementById("mobile-menu");
    const ctaStartBtn = document.getElementById("cta-start");
    const headerActionBtn = document.getElementById("btn-header-action");
    const backendStatusText = document.getElementById("backend-status-text");
    const headerStatusDot = document.getElementById("header-status-dot");
    const mobileActionBtn = document.getElementById("m-btn-action");
    const logoBtn = document.getElementById("logo-btn");

    // Panels
    const heroContent = document.getElementById("hero-content");
    const assessmentPanel = document.getElementById("assessment-panel");
    const breakdownPanel = document.getElementById("breakdown-panel");
    const contactPanel = document.getElementById("contact-panel");

    const btnCloseAssessment = document.getElementById("btn-close-assessment");
    const btnCloseBreakdown = document.getElementById("btn-close-breakdown");
    const btnCloseContact = document.getElementById("btn-close-contact");

    const btnComputeRisk = document.getElementById("btn-compute-risk");
    const resultPanel = document.getElementById("result-panel");
    const sourceBadge = document.getElementById("source-badge");
    const modelBadge = document.getElementById("model-badge");

    // Sliders
    const inputAge = document.getElementById("input-age");
    const inputHeight = document.getElementById("input-height");
    const inputWeight = document.getElementById("input-weight");
    const inputSbp = document.getElementById("input-sbp");
    const inputDbp = document.getElementById("input-dbp");
    const inputHr = document.getElementById("input-hr");

    // Selects
    const selectGender = document.getElementById("select-gender");
    const selectCholesterol = document.getElementById("select-cholesterol");
    const selectGluc = document.getElementById("select-gluc");
    const selectSmoke = document.getElementById("select-smoke");
    const selectAlco = document.getElementById("select-alco");
    const selectActive = document.getElementById("select-active");
    const selectModelType = document.getElementById("select-model-type");

    // Display Badges & Cards
    const valAge = document.getElementById("val-age");
    const valHeight = document.getElementById("val-height");
    const valWeight = document.getElementById("val-weight");
    const valSbp = document.getElementById("val-sbp");
    const valDbp = document.getElementById("val-dbp");
    const valHr = document.getElementById("val-hr");

    const cardValBmi = document.getElementById("card-val-bmi");
    const cardValBp = document.getElementById("card-val-bp");
    const cardValPp = document.getElementById("card-val-pp");
    const cardValAge = document.getElementById("card-val-age");

    // Results
    const resultScore = document.getElementById("result-score");
    const resultVerdict = document.getElementById("result-verdict");
    const riskBarFill = document.getElementById("risk-bar-fill");
    const riskBarPointer = document.getElementById("risk-bar-pointer");
    const factorsGrid = document.getElementById("factors-grid");

    // ── 1. Real-Time Inputs & BMI Calculation ────────────────────────────────
    function updateInputDisplays() {
        const age = parseFloat(inputAge.value);
        const height = parseFloat(inputHeight.value);
        const weight = parseFloat(inputWeight.value);
        const sbp = parseFloat(inputSbp.value);
        const dbp = parseFloat(inputDbp.value);
        const hr = parseFloat(inputHr.value);

        valAge.textContent = `${age} yrs`;
        valHeight.textContent = `${height} cm`;
        valWeight.textContent = `${weight} kg`;
        valSbp.textContent = `${sbp} mmHg`;
        valDbp.textContent = `${dbp} mmHg`;
        valHr.textContent = `${hr} bpm`;

        const heightM = height / 100;
        const bmi = (weight / (heightM * heightM)).toFixed(1);
        const pp = Math.max(0, sbp - dbp);

        cardValBmi.innerHTML = `${bmi}<span class="card-unit">kg/m²</span>`;
        cardValBp.innerHTML = `${sbp}/${dbp}<span class="card-unit">mmHg</span>`;
        cardValPp.innerHTML = `${pp}<span class="card-unit">mmHg</span>`;
        cardValAge.innerHTML = `${age}<span class="card-unit">yrs</span>`;
    }

    [inputAge, inputHeight, inputWeight, inputSbp, inputDbp, inputHr].forEach(inp => {
        inp.addEventListener("input", updateInputDisplays);
    });

    updateInputDisplays();

    // ── 2. Real ML Model Prediction Engine ──────────────────────────────────
    function computeMLRisk() {
        const ageYears = parseFloat(inputAge.value);
        const gender = parseFloat(selectGender.value);
        const height = parseFloat(inputHeight.value);
        const weight = parseFloat(inputWeight.value);
        const apHi = parseFloat(inputSbp.value);
        const apLo = parseFloat(inputDbp.value);
        const cholesterol = parseFloat(selectCholesterol.value);
        const gluc = parseFloat(selectGluc.value);
        const smoke = parseFloat(selectSmoke.value);
        const alco = parseFloat(selectAlco.value);
        const active = parseFloat(selectActive.value);

        const heightM = height / 100;
        const bmi = weight / (heightM * heightM);
        const pulsePressure = Math.max(0, apHi - apLo);

        const featureValues = [
            ageYears, gender, height, weight, apHi, apLo,
            cholesterol, gluc, smoke, alco, active, bmi, pulsePressure
        ];

        let logit = MODEL_WEIGHTS.intercept;
        for (let i = 0; i < featureValues.length; i++) {
            const mean = MODEL_WEIGHTS.scaler_means[i];
            const std = MODEL_WEIGHTS.scaler_scales[i];
            const zScore = (featureValues[i] - mean) / std;
            logit += MODEL_WEIGHTS.coefficients[i] * zScore;
        }

        const probability = 100 / (1 + Math.exp(-logit));
        const riskPct = Math.min(99.9, Math.max(0.1, Math.round(probability * 10) / 10));

        return { riskPct };
    }

    const featureDisplayNames = {
        "ap_hi": "Systolic Blood Pressure",
        "cholesterol": "Cholesterol Level",
        "age_years": "Patient Age",
        "bmi": "Body Mass Index (BMI)",
        "ap_lo": "Diastolic Blood Pressure",
        "active": "Physical Activity",
        "smoke": "Smoking Status",
        "gluc": "Fasting Glucose",
        "pulse_pressure": "Pulse Pressure",
        "weight": "Body Weight",
        "alco": "Alcohol Intake",
        "gender": "Biological Sex",
        "height": "Height"
    };

    const featureColors = {
        "ap_hi": "#f43f5e",
        "cholesterol": "#f59e0b",
        "age_years": "#6366f1",
        "bmi": "#8b5cf6",
        "ap_lo": "#ec4899",
        "active": "#10b981",
        "smoke": "#ef4444",
        "gluc": "#3b82f6"
    };

    // ── Backend Health Monitor ──────────────────────────────────────────────
    async function checkBackendHealth() {
        try {
            const res = await fetch(`${API_BASE_URL}/api/health`, { method: "GET", cache: "no-store" });
            if (res.ok) {
                const data = await res.json();
                isBackendOnline = true;
                if (backendStatusText) backendStatusText.textContent = `Python API Online (${data.version || "v2.4"})`;
                if (headerStatusDot) {
                    headerStatusDot.style.background = "#10b981";
                    headerStatusDot.style.boxShadow = "0 0 10px #10b981";
                }
                return true;
            }
        } catch (e) {
            // Backend offline
        }
        isBackendOnline = false;
        if (backendStatusText) backendStatusText.textContent = "Client In-Browser ML";
        if (headerStatusDot) {
            headerStatusDot.style.background = "#f59e0b";
            headerStatusDot.style.boxShadow = "0 0 10px #f59e0b";
        }
        return false;
    }

    // Initial check & interval
    checkBackendHealth();
    setInterval(checkBackendHealth, 15000);

    // ── Real-Time Prediction (REST API + Dual-Mode Fallback) ─────────────────
    btnComputeRisk.addEventListener("click", async () => {
        const modelType = selectModelType ? selectModelType.value : "xgboost";
        const modelLabel = modelType === "stacking" ? "Stacking Meta-Ensemble" : "Tuned XGBoost Booster";

        btnComputeRisk.innerHTML = `<i class="fa-solid fa-server fa-spin"></i> &nbsp; Evaluating via ${modelLabel}...`;
        btnComputeRisk.disabled = true;

        const payload = {
            age: parseFloat(inputAge.value),
            gender: parseInt(selectGender.value),
            height: parseFloat(inputHeight.value),
            weight: parseFloat(inputWeight.value),
            ap_hi: parseFloat(inputSbp.value),
            ap_lo: parseFloat(inputDbp.value),
            cholesterol: parseInt(selectCholesterol.value),
            gluc: parseInt(selectGluc.value),
            smoke: parseInt(selectSmoke.value),
            alco: parseInt(selectAlco.value),
            active: parseInt(selectActive.value),
            model_type: modelType
        };

        let riskPct = 0;
        let level = "Optimal Risk";
        let verdictCls = "verdict-low";
        let usedModelName = modelLabel;
        let isFromApi = false;
        let featureList = [];

        try {
            const response = await fetch(`${API_BASE_URL}/api/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                const data = await response.json();
                if (data.status === "success") {
                    riskPct = data.risk_percentage;
                    level = data.risk_level;
                    verdictCls = data.verdict_class;
                    usedModelName = data.model_used;
                    isFromApi = true;
                    featureList = data.feature_contributions || [];
                }
            }
        } catch (err) {
            console.warn("Backend API unavailable, using in-browser inference engine.", err);
        }

        // Fallback to in-browser weights if API is offline
        if (!isFromApi) {
            const fallbackResult = computeMLRisk();
            riskPct = fallbackResult.riskPct;
            usedModelName = "In-Browser Logistic Engine v2.4";

            if (riskPct < 35) {
                level = "Optimal Risk (< 35%)";
                verdictCls = "verdict-low";
            } else if (riskPct < 65) {
                level = "Moderate Risk (35 - 65%)";
                verdictCls = "verdict-mod";
            } else {
                level = "Elevated Risk (> 65%)";
                verdictCls = "verdict-high";
            }
        }

        // Reset CTA button
        btnComputeRisk.innerHTML = '⟡ &nbsp; Calculate ML Risk Probability';
        btnComputeRisk.disabled = false;

        // Update Source Badges
        if (sourceBadge) {
            sourceBadge.className = isFromApi ? "source-badge" : "source-badge fallback";
            sourceBadge.innerHTML = isFromApi 
                ? '<i class="fa-solid fa-server"></i> Live Python REST API'
                : '<i class="fa-solid fa-microchip"></i> Client-Side ML Fallback';
        }
        if (modelBadge) {
            modelBadge.textContent = usedModelName;
        }

        // Update Score & Verdict
        resultScore.innerHTML = `0.0<span class="pct-sym">%</span>`;
        resultVerdict.textContent = level;
        resultVerdict.className = `result-verdict ${verdictCls}`;
        resultPanel.classList.remove("hidden");

        // Animate Score Counter
        const duration = 900;
        const startTime = performance.now();

        function animateScore(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeProgress = 1 - Math.pow(1 - progress, 3);
            const currentVal = (easeProgress * riskPct).toFixed(1);

            resultScore.innerHTML = `${currentVal}<span class="pct-sym">%</span>`;

            if (progress < 1) {
                requestAnimationFrame(animateScore);
            } else {
                resultScore.innerHTML = `${riskPct.toFixed(1)}<span class="pct-sym">%</span>`;
            }
        }
        requestAnimationFrame(animateScore);

        riskBarFill.style.width = `${Math.min(100, Math.max(0, riskPct))}%`;
        riskBarPointer.style.left = `${Math.min(100, Math.max(0, riskPct))}%`;

        // Render Factors Breakdown
        factorsGrid.innerHTML = "";

        if (featureList && featureList.length > 0) {
            featureList.slice(0, 6).forEach(f => {
                const displayName = f.label;
                const importancePct = (f.importance * 100).toFixed(1);
                const color = featureColors[f.feature] || "#38bdf8";

                const item = document.createElement("div");
                item.className = "factor-item";
                item.innerHTML = `
                    <div class="factor-name">${displayName}</div>
                    <div class="factor-bar-track">
                        <div class="factor-bar-fill" style="width: 0%; background: ${color}; opacity: 0.85;"></div>
                    </div>
                    <div class="factor-impact" style="color: ${color}">${importancePct}% Feature Weight</div>
                `;
                factorsGrid.appendChild(item);

                setTimeout(() => {
                    const fill = item.querySelector(".factor-bar-fill");
                    if (fill) fill.style.width = `${Math.min(100, Math.max(12, f.importance * 140))}%`;
                }, 100);
            });
        } else {
            // Local fallback factor rendering
            const sortedFeatures = Object.keys(MODEL_WEIGHTS.xgb_importance);
            sortedFeatures.slice(0, 6).forEach(featKey => {
                const displayName = featureDisplayNames[featKey] || featKey;
                const importance = (MODEL_WEIGHTS.xgb_importance[featKey] * 100).toFixed(1);
                const color = featureColors[featKey] || "#ffffff";

                const item = document.createElement("div");
                item.className = "factor-item";
                item.innerHTML = `
                    <div class="factor-name">${displayName}</div>
                    <div class="factor-bar-track">
                        <div class="factor-bar-fill" style="width: 0%; background: ${color}; opacity: 0.85;"></div>
                    </div>
                    <div class="factor-impact" style="color: ${color}">${importance}% Model Weight</div>
                `;
                factorsGrid.appendChild(item);

                setTimeout(() => {
                    const fill = item.querySelector(".factor-bar-fill");
                    if (fill) fill.style.width = `${Math.min(100, Math.max(12, importance * 1.5))}%`;
                }, 100);
            });
        }

        setTimeout(() => {
            resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 120);
    });

    // ── 3. Multi-Page Navigation View Switcher ────────────────────────────────
    function showPage(pageId) {
        // Smoothly scroll window to top when changing views
        window.scrollTo({ top: 0, behavior: "smooth" });

        // Hide all page panels
        heroContent.classList.add("hidden-view");
        assessmentPanel.classList.add("hidden");
        breakdownPanel.classList.add("hidden");
        contactPanel.classList.add("hidden");

        assessmentPanel.setAttribute("aria-hidden", "true");
        breakdownPanel.setAttribute("aria-hidden", "true");
        contactPanel.setAttribute("aria-hidden", "true");

        // Deactivate all nav links
        document.querySelectorAll(".nav-link, .mobile-link").forEach(l => l.classList.remove("active"));

        // Activate requested page
        if (pageId === "assessment") {
            assessmentPanel.classList.remove("hidden");
            assessmentPanel.setAttribute("aria-hidden", "false");
            document.getElementById("nav-assessment")?.classList.add("active");
            document.getElementById("m-nav-assessment")?.classList.add("active");
        } else if (pageId === "breakdown") {
            breakdownPanel.classList.remove("hidden");
            breakdownPanel.setAttribute("aria-hidden", "false");
            document.getElementById("nav-breakdown")?.classList.add("active");
            document.getElementById("m-nav-breakdown")?.classList.add("active");
        } else if (pageId === "contact") {
            contactPanel.classList.remove("hidden");
            contactPanel.setAttribute("aria-hidden", "false");
            document.getElementById("nav-contact")?.classList.add("active");
            document.getElementById("m-nav-contact")?.classList.add("active");
        } else {
            // Home page default
            heroContent.classList.remove("hidden-view");
            document.getElementById("nav-home")?.classList.add("active");
            document.getElementById("m-nav-home")?.classList.add("active");
        }

        closeMobileMenu();
    }

    // Attach click events to nav links
    document.querySelectorAll(".nav-link, .mobile-link").forEach(link => {
        link.addEventListener("click", (e) => {
            const href = link.getAttribute("href");
            if (href.startsWith("#")) {
                e.preventDefault();
                const pageId = href.replace("#", "");
                showPage(pageId);
            }
        });
    });

    ctaStartBtn.addEventListener("click", () => showPage("assessment"));
    headerActionBtn.addEventListener("click", () => showPage("assessment"));
    mobileActionBtn.addEventListener("click", () => showPage("assessment"));
    logoBtn.addEventListener("click", () => showPage("home"));

    btnCloseAssessment.addEventListener("click", () => showPage("home"));
    btnCloseBreakdown.addEventListener("click", () => showPage("home"));
    btnCloseContact.addEventListener("click", () => showPage("home"));

    // ── 4. Mobile Menu Controls ─────────────────────────────────────────────
    function toggleMobileMenu() {
        if (document.body.classList.contains("menu-open")) {
            closeMobileMenu();
        } else {
            openMobileMenu();
        }
    }

    function openMobileMenu() {
        document.body.classList.add("menu-open");
        mobileOverlay.classList.add("active");
        mobileMenu.classList.add("active");
        mobileOverlay.setAttribute("aria-hidden", "false");
        mobileMenu.setAttribute("aria-hidden", "false");
        burgerBtn.setAttribute("aria-expanded", "true");
    }

    function closeMobileMenu() {
        document.body.classList.remove("menu-open");
        mobileOverlay.classList.remove("active");
        mobileMenu.classList.remove("active");
        mobileOverlay.setAttribute("aria-hidden", "true");
        mobileMenu.setAttribute("aria-hidden", "true");
        burgerBtn.setAttribute("aria-expanded", "false");
    }

    burgerBtn.addEventListener("click", toggleMobileMenu);
    mobileOverlay.addEventListener("click", closeMobileMenu);

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            closeMobileMenu();
            showPage("home");
        }
    });

    window.addEventListener("resize", () => {
        if (window.innerWidth > 720) {
            closeMobileMenu();
        }
    });

    // ── 5. Stats Footer Count-Up ──────────────────────────────────────────────
    function easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    function animateStatsCounter(el, index) {
        const targetVal = parseFloat(el.getAttribute("data-target"));
        const decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
        const duration = 1500 + index * 80;
        const startDelay = 480 + index * 90;

        setTimeout(() => {
            const startTime = performance.now();

            function updateCount(currentTime) {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const eased = easeOutCubic(progress);
                const currentVal = (eased * targetVal).toFixed(decimals);

                el.textContent = currentVal;

                if (progress < 1) {
                    requestAnimationFrame(updateCount);
                } else {
                    el.textContent = targetVal.toFixed(decimals);
                }
            }

            requestAnimationFrame(updateCount);
        }, startDelay);
    }

    const statCounts = document.querySelectorAll(".stat-count");
    let animated = false;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !animated) {
                animated = true;
                statCounts.forEach((el, idx) => animateStatsCounter(el, idx));
            }
        });
    }, { threshold: 0.25 });

    const statsFooter = document.getElementById("stats-footer");
    if (statsFooter) {
        observer.observe(statsFooter);
    }

    // ── 6. Custom Dropdown Initializer ─────────────────────────────────────────
    function initCustomDropdowns() {
        const nativeSelects = document.querySelectorAll("select.custom-select");
        
        nativeSelects.forEach(select => {
            // Create wrapper
            const wrapper = document.createElement("div");
            wrapper.className = "custom-dropdown-wrapper";
            
            // Insert wrapper right before native select, then move select inside
            select.parentNode.insertBefore(wrapper, select);
            wrapper.appendChild(select);
            
            // Create Trigger
            const trigger = document.createElement("div");
            trigger.className = "custom-dropdown-trigger";
            
            const selectedOptionText = select.options[select.selectedIndex]?.text || "Select...";
            trigger.innerHTML = `
                <span class="custom-dropdown-value">${selectedOptionText}</span>
                <i class="fa-solid fa-chevron-down custom-dropdown-icon"></i>
            `;
            wrapper.appendChild(trigger);
            
            // Create Options Container
            const optionsContainer = document.createElement("div");
            optionsContainer.className = "custom-dropdown-options";
            
            Array.from(select.options).forEach((option, index) => {
                const optDiv = document.createElement("div");
                optDiv.className = "custom-dropdown-option";
                if (index === select.selectedIndex) {
                    optDiv.classList.add("selected");
                }
                
                optDiv.innerHTML = `
                    <span>${option.text}</span>
                    <i class="fa-solid fa-check custom-dropdown-check"></i>
                `;
                
                optDiv.addEventListener("click", (e) => {
                    e.stopPropagation(); // prevent trigger click from firing
                    // Update native select
                    select.selectedIndex = index;
                    
                    // Dispatch change event to trigger ML logic if needed
                    select.dispatchEvent(new Event("change"));
                    
                    // Update trigger text
                    trigger.querySelector(".custom-dropdown-value").textContent = option.text;
                    
                    // Update classes
                    optionsContainer.querySelectorAll(".custom-dropdown-option").forEach(el => el.classList.remove("selected"));
                    optDiv.classList.add("selected");
                    
                    // Close dropdown
                    wrapper.classList.remove("open");
                });
                
                optionsContainer.appendChild(optDiv);
            });
            
            wrapper.appendChild(optionsContainer);
            
            // Toggle dropdown on trigger click
            trigger.addEventListener("click", (e) => {
                e.stopPropagation();
                // Close all other open dropdowns first
                document.querySelectorAll(".custom-dropdown-wrapper.open").forEach(el => {
                    if (el !== wrapper) el.classList.remove("open");
                });
                wrapper.classList.toggle("open");
            });
        });
        
        // Close dropdowns when clicking outside
        document.addEventListener("click", () => {
            document.querySelectorAll(".custom-dropdown-wrapper.open").forEach(el => {
                el.classList.remove("open");
            });
        });
    }
    
    // Initialize custom dropdowns
    initCustomDropdowns();

    // ── 7. Text Scramble Effect for Headline ──────────────────────────────────
    class TextScramble {
        constructor(el) {
            this.el = el;
            this.chars = '!<>-_\\/[]{}—=+*^?#________';
            this.update = this.update.bind(this);
        }
        setText(newText) {
            const oldText = this.el.innerText;
            const length = Math.max(oldText.length, newText.length);
            const promise = new Promise((resolve) => this.resolve = resolve);
            this.queue = [];
            for (let i = 0; i < length; i++) {
                const from = oldText[i] || '';
                const to = newText[i] || '';
                const start = Math.floor(Math.random() * 15);
                const end = start + Math.floor(Math.random() * 15);
                this.queue.push({ from, to, start, end, char: '' });
            }
            cancelAnimationFrame(this.frameRequest);
            this.frame = 0;
            this.update();
            return promise;
        }
        update() {
            let output = '';
            let complete = 0;
            for (let i = 0; i < this.queue.length; i++) {
                let { from, to, start, end, char } = this.queue[i];
                if (this.frame >= end) {
                    complete++;
                    output += to;
                } else if (this.frame >= start) {
                    if (!char || Math.random() < 0.28) {
                        char = this.randomChar();
                        this.queue[i].char = char;
                    }
                    output += `<span style="display:inline;color:var(--accent);opacity:0.85;">${char}</span>`;
                } else {
                    output += from;
                }
            }
            this.el.innerHTML = output;
            if (complete === this.queue.length) {
                this.resolve();
            } else {
                this.frameRequest = requestAnimationFrame(this.update);
                this.frame++;
            }
        }
        randomChar() {
            return this.chars[Math.floor(Math.random() * this.chars.length)];
        }
    }

    const line1 = document.querySelector(".headline .line-1");
    const line2 = document.querySelector(".headline .line-2");
    if (line1 && line2) {
        setTimeout(() => {
            const fx1 = new TextScramble(line1);
            const fx2 = new TextScramble(line2);
            fx1.setText("Intelligence");
            setTimeout(() => fx2.setText("Designed To Evolve"), 220);
        }, 450);
    }

    // ── 8. Custom Medical Cursor & Hover Physics ──────────────────────────────
    const cursorDot = document.createElement("div");
    cursorDot.className = "custom-cursor-dot";
    const cursorRing = document.createElement("div");
    cursorRing.className = "custom-cursor-ring";
    document.body.appendChild(cursorDot);
    document.body.appendChild(cursorRing);

    let mouseX = -100, mouseY = -100;
    let ringX = -100, ringY = -100;

    window.addEventListener("mousemove", (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
        cursorDot.style.transform = `translate3d(${mouseX}px, ${mouseY}px, 0) translate(-50%, -50%)`;
    });

    function renderCursorRing() {
        ringX += (mouseX - ringX) * 0.18;
        ringY += (mouseY - ringY) * 0.18;
        cursorRing.style.transform = `translate3d(${ringX}px, ${ringY}px, 0) translate(-50%, -50%)`;
        requestAnimationFrame(renderCursorRing);
    }
    requestAnimationFrame(renderCursorRing);

    document.addEventListener("mouseover", (e) => {
        const target = e.target.closest("button, a, input, select, .custom-dropdown-trigger, .custom-dropdown-option, .weight-row, .slider-input");
        if (target) {
            document.body.classList.add("cursor-hover");
        } else {
            document.body.classList.remove("cursor-hover");
        }
    });

    // ── 9. Magnetic CTA Buttons ────────────────────────────────────────────────
    const magneticElements = document.querySelectorAll(".cta-btn, .signin-pill, .predict-submit-btn, .logo-btn");
    magneticElements.forEach(el => {
        el.classList.add("magnetic-btn");
        el.addEventListener("mousemove", (e) => {
            const rect = el.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;
            const deltaX = (e.clientX - centerX) * 0.35;
            const deltaY = (e.clientY - centerY) * 0.35;
            el.style.transform = `translate3d(${deltaX}px, ${deltaY}px, 0) scale(1.02)`;
        });

        el.addEventListener("mouseleave", () => {
            el.style.transform = `translate3d(0, 0, 0) scale(1)`;
        });
    });

    // ── 10. Feature Weight Tooltips (Breakdown Panel) ──────────────────────────
    const tooltipsData = {
        "Systolic Blood Pressure (ap_hi)": "Primary indicator of vascular pressure. Values >140 mmHg strongly correlate with hypertension & vascular strain.",
        "Total Cholesterol Level": "High LDL cholesterol causes arterial plaque buildup, restricting cardiac blood flow.",
        "Patient Age (Years)": "Vascular stiffness increases naturally with age, raising baseline cardiovascular risk.",
        "Physical Activity Status": "Sedentary lifestyle reduces heart muscular efficiency and metabolic clearance.",
        "Smoking Status": "Nicotine damages endothelial cell lining, accelerating atherosclerosis formation.",
        "Diastolic Blood Pressure (ap_lo)": "Reflects arterial resistance during resting phase of the cardiac cycle."
    };

    document.querySelectorAll(".weight-row").forEach(row => {
        const nameEl = row.querySelector(".weight-name");
        if (nameEl && tooltipsData[nameEl.textContent.trim()]) {
            const tooltip = document.createElement("div");
            tooltip.className = "weight-tooltip";
            tooltip.textContent = tooltipsData[nameEl.textContent.trim()];
            row.appendChild(tooltip);
        }
    });
});
