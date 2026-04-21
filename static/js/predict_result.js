document.addEventListener("DOMContentLoaded", function () {
    // Inicializácia tooltipov
    const marker = document.querySelector(".gauge-marker");
    const dot = document.querySelector(".gauge-dot");

    if (marker && dot) {
        let value = parseFloat(marker.dataset.left);
        if (Number.isNaN(value)) {
            value = 0;
        }

        value = Math.max(0, Math.min(100, value));
        marker.style.left = value + "%";
        dot.style.left = value + "%";

        let color;
        if (value < 33) {
            color = "#0a7b45";
        } else if (value < 66) {
            color = "#ffb800";
        } else {
            color = "#d92626";
        }

        dot.style.backgroundColor = color;
    }

    document.querySelectorAll(".gauge-dot, .gauge-threshold").forEach(function (element) {
        const value = parseFloat(element.dataset.left);
        const safeValue = Number.isNaN(value) ? 0 : Math.max(0, Math.min(100, value));
        element.style.left = safeValue + "%";
    });
});
