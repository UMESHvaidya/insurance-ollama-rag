
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("query-form");
    const loadingDiv = document.getElementById("loading");
    const resultsDiv = document.getElementById("results");
    const resultsContent = document.getElementById("results-content");
    const submitButton = document.getElementById("submit-btn");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const formData = new FormData(form);
        const fileInput = document.getElementById("file");
        const queryInput = document.getElementById("query");

        if (!fileInput.files[0] || !queryInput.value) {
            alert("Please select a file and enter a query.");
            return;
        }

        // Show loading indicator and disable button
        loadingDiv.classList.remove("hidden");
        resultsDiv.classList.add("hidden");
        submitButton.disabled = true;
        submitButton.textContent = "Analyzing...";

        try {
            const response = await fetch("/api/query", {
                method: "POST",
                body: formData,
            });

            const result = await response.json();

            if (response.ok) {
                displayResults(result);
            } else {
                displayError(result.error || "An unknown error occurred.");
            }
        } catch (error) {
            displayError("An error occurred while communicating with the server.");
        } finally {
            // Hide loading indicator and re-enable button
            loadingDiv.classList.add("hidden");
            submitButton.disabled = false;
            submitButton.textContent = "Analyze";
        }
    });

    function displayResults(data) {
        resultsContent.innerHTML = `
            <p><strong>Status:</strong> ${data.status}</p>
            <p><strong>Explanation:</strong> ${data.explanation}</p>
            <p><strong>Reference:</strong> ${data.reference}</p>
            <p><strong>Confidence:</strong> ${data.confidence}</p>
        `;
        resultsDiv.classList.remove("hidden");
    }

    function displayError(message) {
        resultsContent.innerHTML = `<p style="color: red;"><strong>Error:</strong> ${message}</p>`;
        resultsDiv.classList.remove("hidden");
    }
});
