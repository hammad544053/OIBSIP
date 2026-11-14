function toggleModal(show) {
    const modal = document.getElementById("modelModal");
    if (modal) {
        modal.style.display = show ? "flex" : "none";
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const menuBtn = document.getElementById("menu");
    const sidebar = document.getElementById("sidebar");
    if (menuBtn && sidebar) {
        menuBtn.addEventListener("click", () => sidebar.classList.toggle("open"));
    }

    const modal = document.getElementById("modelModal");
    if (modal) {
        modal.addEventListener("click", (e) => {
            if (e.target === modal) toggleModal(false);
        });
    }

    document.querySelectorAll(".flash").forEach((el) => {
        setTimeout(() => el.remove(), 5000);
    });
});