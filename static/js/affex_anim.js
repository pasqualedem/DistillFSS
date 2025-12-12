document.addEventListener('DOMContentLoaded', () => {

    // --- Grid Population ---
    const containers = document.querySelectorAll('.afx-tensor-face, .afx-tensor-depth-1, .afx-tensor-depth-2');
    containers.forEach(container => {
        if (container.children.length > 0) return;
        for (let i = 0; i < 25; i++) {
            let div = document.createElement('div');
            container.appendChild(div);
        }
    });

    // --- State Variables ---
    let step = 0;
    const caption = document.getElementById('afx-caption');
    const stage = document.querySelector('.afx-stage'); // Used for class toggling
    const btn_next = document.getElementById('afx-btn-next');
    const btn_previous = document.getElementById('afx-btn-previous');
    const resetBtn = document.getElementById('afx-btn-reset');
    const phaseLabel = document.getElementById('afx-phase-label');

    // --- Steps Logic ---
    const steps = [
        {
            text: "Step 1: Feature Extraction. (Standard FSS Model)",
            action: () => {
                stage.classList.add('afx-state-features');
                phaseLabel.innerText = "Standard FSS Model";
                stage.classList.add('afx-phase-fss');
                btn_previous.disabled = false;
            }
        },
        {
            text: "Step 2: Dense Matching. 4D Correlation Tensors (Visually represented as thick 3D stacks).",
            action: () => stage.classList.add('afx-state-matching')
        },
        {
            text: "Step 3: ROI Averaging. Collapsing Tensors to 2D Maps over Query ROI. (AffEx Starts Here)",
            action: () => {
                stage.classList.add('afx-state-average');
                stage.classList.remove('afx-phase-fss');
                stage.classList.add('afx-phase-affex');
                phaseLabel.innerText = "Affinity Explainer (AffEx)";
            }
        },
        {
            text: "Step 4: Softmax Normalization. Red = Match, Blue = Mismatch.",
            action: () => stage.classList.add('afx-state-softmax')
        },
        {
            text: "Step 5: Feature Ablation Weights. Calculating layer importance (w).",
            action: () => stage.classList.add('afx-state-weights')
        },
        {
            text: "Step 6: Aggregation. Weighted summation into final maps.",
            action: () => stage.classList.add('afx-state-aggregate')
        },
        {
            text: "Step 7: Projection. Final Attribution Map highlights Black cat (Red) and marks Dog as irrelevant (Blue).",
            action: () => {
                stage.classList.add('afx-state-result');
                btn_next.innerText = "Done";
                btn_next.disabled = true;
            }
        }
    ];

    if (btn_next) {
        btn_next.addEventListener('click', () => {
            if (step < steps.length) {
                caption.innerText = steps[step].text;
                steps[step].action();
                step++;
            }
        });
    }

    if (btn_previous) {
        btn_previous.addEventListener('click', () => {

            if (step > 0) {
                step--; // Decrement first

                // 1. Nuke all classes strictly
                const classesToRemove = [
                    'afx-state-features', 'afx-state-matching', 'afx-state-average',
                    'afx-state-softmax', 'afx-state-weights', 'afx-state-aggregate',
                    'afx-state-result', 'afx-phase-fss', 'afx-phase-affex'
                ];
                stage.classList.remove(...classesToRemove);

                // 2. Re-run actions to rebuild state up to current step
                // We loop through the steps we have "completed" to get back to current state
                // Note: If step is 1, we want the state of step 1 (index 0) to be active.
                // The 'Next' logic executes action for index 0, then makes step = 1.
                // So if step = 1 now, we need to execute action 0.
                try {
                    for (let i = 0; i < step; i++) {
                        console.log("Re-running action for step index:", i);
                        steps[i].action();
                    }
                } catch (e) {
                    console.error("Error re-running actions:", e);
                }

                // 3. Update Caption
                // If step is 1, we are "at" Step 1 (index 0).
                if (step > 0) {
                    caption.innerText = steps[step - 1].text;
                } else {
                    caption.innerText = "Initialize: Query (Car) vs Support Set (Car, Plant)";
                    phaseLabel.innerText = ""; // Clear phase label on full reset
                    btn_previous.disabled = true;
                }

                // 4. Reset buttons
                btn_next.innerText = "Next";
                btn_next.disabled = false;

                // 5. Fix Phase Label Logic for navigating backwards
                // If we just went back to step 0, label is empty (handled above)
                // If we went back to step 1 or 2, it is FSS
                if (step > 0 && step <= 2) {
                    phaseLabel.innerText = "Standard FSS Model";
                } else if (step > 2) {
                    phaseLabel.innerText = "Affinity Explainer (AffEx)";
                }
            }
        });
    } else {
        console.error("Button #afx-btn-previous not found in DOM!");
    }
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            step = 0;
            const classesToRemove = [
                'afx-state-features', 'afx-state-matching', 'afx-state-average',
                'afx-state-softmax', 'afx-state-weights', 'afx-state-aggregate',
                'afx-state-result', 'afx-phase-fss', 'afx-phase-affex'
            ];
            stage.classList.remove(...classesToRemove);

            caption.innerText = "Initialize: Query (Car) vs Support Set (Car, Plant)";
            phaseLabel.innerText = "";
            btn_next.innerText = "Next";
            btn_next.disabled = false;
            btn_previous.disabled = true;
        });
    }

    // --- FIXED SCALING LOGIC ---
    function scaleAnimation() {
        const container = document.getElementById('affex-anim-container');
        const scaler = document.querySelector('.afx-scaler');

        if (!container || !scaler) return;

        // Fixed design dimensions
        const designWidth = 1100;
        const designHeight = 650;

        // Get the actual available width from the parent container
        const availableWidth = container.offsetWidth;

        // Calculate scale
        let scale = availableWidth / designWidth;

        // Optional: Cap scaling at 1.0 so it doesn't get blurry on huge screens
        if (scale > 1) scale = 1;

        // Apply Scale to the inner wrapper
        scaler.style.transform = `scale(${scale})`;

        // Set height of the outer container to match the scaled content
        container.style.height = `${designHeight * scale}px`;
    }

    window.addEventListener('resize', scaleAnimation);
    scaleAnimation();
    setTimeout(scaleAnimation, 200);
});