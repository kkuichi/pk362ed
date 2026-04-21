document.addEventListener("DOMContentLoaded", function () {
    // Inicializácia tooltipov pre informačné body a vstupné polia
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function (element) {
        bootstrap.Tooltip.getOrCreateInstance(element);
    });

    // Kontrola rozsahov pre číselné vstupy
    document.querySelectorAll(".range-input").forEach(function (input) {
        const tooltip = bootstrap.Tooltip.getOrCreateInstance(input);

        input.addEventListener("input", function () {
            const minDs = parseFloat(this.dataset.min);
            const maxDs = parseFloat(this.dataset.max);
            const value = parseFloat(this.value);

            this.classList.remove("is-invalid");
            this.setAttribute("title", "");
            tooltip.hide();

            if (this.value === "" || Number.isNaN(value)) {
                return;
            }

            if (value < 0) {
                this.classList.add("is-invalid");
                this.setAttribute("title", "Hodnota nemôže byť záporná");
                tooltip.show();
                return;
            }

            if (!Number.isNaN(minDs) && !Number.isNaN(maxDs) && (value < minDs || value > maxDs)) {
                this.classList.add("is-invalid");
                this.setAttribute("title", "Mimo rozsah datasetu");
                tooltip.show();
            }
        });
    });

    // Niektoré premenné sa zobrazujú len pre konkrétny typ imputácie.
    const modelSelect = document.getElementById("modelSelect");

    const knnOnlyFields = ["PDW_max", "HGB_min"];
    const miceOnlyFields = ["S_VITD_max", "S_VITD_first"];

    function setFieldState(fieldId, enabled) {
        const field = document.getElementById(fieldId);
        if (!field) {
            return;
        }

        field.disabled = !enabled;

        if (enabled) {
            field.classList.remove("field-locked");
        } else {
            field.classList.remove("is-invalid");
            field.value = "";
            field.setAttribute("title", "");
            field.classList.add("field-locked");
        }
    }

    function updateFieldsByModel() {
        if (!modelSelect) {
            return;
        }

        const selectedModel = modelSelect.value.toLowerCase();
        const isKNN = selectedModel.includes("knn");
        const isMICE = selectedModel.includes("mice");

        if (isKNN) {
            knnOnlyFields.forEach(function (fieldId) { setFieldState(fieldId, true); });
            miceOnlyFields.forEach(function (fieldId) { setFieldState(fieldId, false); });
        } else if (isMICE) {
            knnOnlyFields.forEach(function (fieldId) { setFieldState(fieldId, false); });
            miceOnlyFields.forEach(function (fieldId) { setFieldState(fieldId, true); });
        } else {
            [...knnOnlyFields, ...miceOnlyFields].forEach(function (fieldId) {
                setFieldState(fieldId, true);
            });
        }
    }

    if (modelSelect) {
        modelSelect.addEventListener("change", updateFieldsByModel);
        updateFieldsByModel();
    }
});
