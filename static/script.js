document.addEventListener('DOMContentLoaded', (event) => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const modelSelect = document.getElementById('model-select');
    const diagnosisResult = document.getElementById('diagnosis-result');
    const confidenceScore = document.getElementById('confidence-score');
    const errorMessage = document.getElementById('error-message');
    const submitButton = document.getElementById('submit-button');
    const uploadedImage = document.getElementById('uploaded-image');

    // Function to display the selected image
    function displayImage(file) {
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadedImage.src = e.target.result;
                uploadedImage.style.display = 'block'; // Show the image
            };
            reader.readAsDataURL(file); // Read the file as a data URL
        } else {
            uploadedImage.src = '';
            uploadedImage.style.display = 'none'; // Hide the image if no file
        }
    }

    // Add event listener to the file input for immediate preview
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            // Display the selected image right away
            displayImage(this.files[0]);

            // Reset diagnosis results when a new file is selected
            diagnosisResult.textContent = "Select a model, upload an image, and click 'Get Diagnosis' to see the result.";
            confidenceScore.textContent = "";
            errorMessage.textContent = "";
        });
    }

    // Add event listener for form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault(); // Prevent default form submission

            // Show "Processing..." and disable button
            diagnosisResult.textContent = "Processing...";
            confidenceScore.textContent = "";
            errorMessage.textContent = "";
            submitButton.disabled = true;

            if (fileInput.files.length === 0) {
                errorMessage.textContent = "Please select an image file.";
                diagnosisResult.textContent = "Select a model, upload an image, and click 'Get Diagnosis' to see the result.";
                uploadedImage.src = ''; // Clear image preview if error
                uploadedImage.style.display = 'none';
                submitButton.disabled = false;
                return;
            }

            const selectedFile = fileInput.files[0];
            // The image should already be displayed by the 'change' event,
            // so we don't strictly *need* to call displayImage here again,
            // but it doesn't hurt. Ensure it remains visible.

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('model_type', modelSelect.value);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    diagnosisResult.textContent = `Predicted Class: ${data.prediction}`;
                    confidenceScore.textContent = `Confidence: ${data.confidence}%`;
                } else {
                    errorMessage.textContent = `Error: ${data.error || 'Something went wrong.'}`;
                    diagnosisResult.textContent = "Select a model, upload an image, and click 'Get Diagnosis' to see the result.";
                }
            } catch (error) {
                errorMessage.textContent = `Network or server error: ${error.message}`;
                diagnosisResult.textContent = "Select a model, upload an image, and click 'Get Diagnosis' to see the result.";
            } finally {
                submitButton.disabled = false; // Re-enable button
            }
        });
    }
});