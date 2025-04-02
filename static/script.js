document.addEventListener('DOMContentLoaded', function() {
    const startBtn = document.getElementById('start-btn');
    const predictionText = document.getElementById('prediction-text');
    let modelRunning = false;

    startBtn.addEventListener('click', function() {
        if (!modelRunning) {
            // Disable the button
            startBtn.disabled = true;
            startBtn.textContent = "Model Running...";
            startBtn.classList.add('disabled');
            
            // Update status
            modelRunning = true;
            predictionText.textContent = "Model is running...";
            
           
        }
    });
});
