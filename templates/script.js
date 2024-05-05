function toggleSelection() {
    var container = document.getElementById("selection-container");
    var button = document.getElementById("show-selection-btn");
    if (container.style.display === "none") {
        container.style.display = "block";
        button.textContent = "Hide Selection";
    } else {
        container.style.display = "none";
        button.textContent = "Show Selection";
    }
}

// Update displayed study hours value
document.getElementById('study_hours').addEventListener('input', function() {
    document.getElementById('study_hours_value').textContent = this.value;
});

// Update displayed satisfaction value
document.getElementById('satisfaction').addEventListener('input', function() {
    document.getElementById('satisfaction_value').textContent = this.value;
});
