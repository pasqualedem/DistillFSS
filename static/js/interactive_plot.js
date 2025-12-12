// --- Configuration ---
// NOTE: Replace the 'data' arrays with your actual mIoU values for the graphs to match your paper.
const dataConfig = {
    'DAUC': {
        label: 'Deletion Curve',
        data: [0.5811669,
            0.51284564,
            0.5484883,
            0.40431026,
            0.2193716,
            0.16653515,
            0.1565787,
            0.15893225,
            0.15910959,
            0.1543064,
            0.1539258,
            0.15325716,
            0.15143031,
            0.1493394,
            0.14952768,
            0.15429284,
            0.15737456,
            0.16060656,
            0.16160175,
            0.16466144,
            0.17609452,
            0.17062636,
            0.1668528,
            0.16685116,
            0.16745687,
            0.15875204],
        color: 'rgb(255, 99, 132)',
        folder: 'dauc',
        insight: "Removing relevant pixels degrades performance."
    },
    'IAUC': {
        label: 'Insertion Curve',
        data: [0.15875204,
            0.35931292,
            0.46135718,
            0.7135555,
            0.750771,
            0.75030476,
            0.7464023,
            0.74061143,
            0.72855175,
            0.72018653,
            0.72231156,
            0.7368886,
            0.72815645,
            0.72655386,
            0.7386933,
            0.7244009,
            0.70574397,
            0.71499026,
            0.70063883,
            0.63119227,
            0.53338236,
            0.6043424,
            0.55835927,
            0.5383539,
            0.5346237,
            0.5811669],
        color: 'rgb(75, 192, 192)',
        folder: 'iauc',
        insight: "Adding top pixels recovers performance quickly."
    }
};

let currentMetric = 'DAUC';
let causalChart = null;

function initChart() {
    const ctx = document.getElementById('causalChart').getContext('2d');
    const steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24];

    causalChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: steps,
            datasets: [{
                label: dataConfig['DAUC'].label,
                data: dataConfig['DAUC'].data,
                borderColor: dataConfig['DAUC'].color,
                tension: 0.3,
                pointRadius: 5,
                pointHoverRadius: 7,
                pointBackgroundColor: dataConfig['DAUC'].color
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { display: false }, tooltip: { enabled: true } },
            scales: {
                y: { beginAtZero: true, max: 1.0, title: { display: true, text: 'mIoU' } },
                x: { title: { display: true, text: 'Perturbation Step' } }
            }
        }
    });
}

function updateVisualization(stepValue) {
    document.getElementById('step-label').innerText = stepValue;

    // 1. Update Chart Highlight
    if (causalChart) {
        const index = stepValue - 1;
        const pointRadii = new Array(24).fill(5);
        const pointColors = new Array(24).fill(dataConfig[currentMetric].color);

        pointRadii[index] = 10;
        pointColors[index] = 'rgb(50, 50, 50)';

        causalChart.data.datasets[0].pointRadius = pointRadii;
        causalChart.data.datasets[0].pointBackgroundColor = pointColors;
        causalChart.update('none');
    }

    // 2. Update Images
    // Pattern: static/images/{metric}/{metric}_mid{step}_{type}.png
    const metricLower = currentMetric.toLowerCase();
    const basePath = `./static/images/${metricLower}/${metricLower}_mid${stepValue}`;

    // Update Query Visuals
    document.getElementById('img-seg').src = `${basePath}_seg.png`;
    document.getElementById('img-probs').src = `${basePath}_probs.png`;

    // Update Support Visuals (Shots 0-4)
    for (let i = 0; i < 5; i++) {
        const shotId = `img-shot${i}`;
        const shotEl = document.getElementById(shotId);
        if (shotEl) {
            shotEl.src = `${basePath}_shot${i}.png`;
        }
    }
}

function switchMetric(metric) {
    currentMetric = metric;

    // Update Tabs
    document.getElementById('tab-dauc').classList.remove('is-active');
    document.getElementById('tab-iauc').classList.remove('is-active');
    document.getElementById(`tab-${metric.toLowerCase()}`).classList.add('is-active');

    // Update Chart Data
    if (causalChart) {
        causalChart.data.datasets[0].label = dataConfig[metric].label;
        causalChart.data.datasets[0].data = dataConfig[metric].data;
        causalChart.data.datasets[0].borderColor = dataConfig[metric].color;
        causalChart.data.datasets[0].pointBackgroundColor = dataConfig[metric].color;
    }

    // Reset Slider
    document.getElementById('step-slider').value = 0;
    updateVisualization(0);
}

document.addEventListener('DOMContentLoaded', function () {
    initChart();
    updateVisualization(0);
});