(function () {
    // Tento skript implementuje jednoduchý lightbox pre zobrazenie obrázkov v plnej veľkosti.
    let images = [];
    let currentIndex = 0;

    function qs(sel, root = document) {
        return root.querySelector(sel);
    }

    function qsa(sel, root = document) {
        return Array.from(root.querySelectorAll(sel));
    }

    function isOpen() {
        const lb = qs("#lightbox");
        return lb && lb.classList.contains("open");
    }

    function openAt(index) {
        if (!images.length) return;

        currentIndex = ((index % images.length) + images.length) % images.length;

        const lb = qs("#lightbox");
        const img = qs("#lightbox-img");
        if (!lb || !img) return;

        img.src = images[currentIndex];
        lb.classList.add("open");
        lb.setAttribute("aria-hidden", "false");
        document.body.style.overflow = "hidden";
    }

    function close() {
        const lb = qs("#lightbox");
        const img = qs("#lightbox-img");
        if (!lb || !img) return;

        lb.classList.remove("open");
        lb.setAttribute("aria-hidden", "true");
        document.body.style.overflow = "";
        img.src = "";
    }

    function prev() {
        openAt(currentIndex - 1);
    }

    function next() {
        openAt(currentIndex + 1);
    }

    document.addEventListener("DOMContentLoaded", function () {
        const thumbs = qsa('img[data-lightbox]');
        images = thumbs.map(t => t.src);

        // klik na náhľad → otvor
        thumbs.forEach((img, idx) => {
            img.addEventListener("click", function () {
                openAt(idx);
            });
        });

        // ovládanie v lightboxe
        const btnPrev = qs("[data-lb-prev]");
        const btnNext = qs("[data-lb-next]");
        const btnClose = qs("[data-lb-close]");
        const lb = qs("#lightbox");

        if (btnPrev) btnPrev.addEventListener("click", function (e) { e.stopPropagation(); prev(); });
        if (btnNext) btnNext.addEventListener("click", function (e) { e.stopPropagation(); next(); });
        if (btnClose) btnClose.addEventListener("click", function (e) { e.stopPropagation(); close(); });

        // klik mimo obrázka zavrie
        if (lb) lb.addEventListener("click", close);

        // ESC + šípky
        document.addEventListener("keydown", function (e) {
            if (!isOpen()) return;

            if (e.key === "Escape") close();
            if (e.key === "ArrowLeft") prev();
            if (e.key === "ArrowRight") next();
        });
    });
})();
