document.addEventListener('DOMContentLoaded', function() {
    fetch('./referrals.txt')  // Replace with the correct path to your text file
        .then(response => response.text())
        .then(text => {
            const lines = text.split('\n');
            const dropdown = document.getElementById('referralIndex');
            lines.forEach(line => {
                const [name, index] = line.split(',');
                if(name && index !== undefined) {
                    const option = document.createElement('option');
                    option.value = index.trim();
                    option.textContent = name.trim();
                    dropdown.appendChild(option);
                }
            });
        })
        .catch(error => {
            console.error('Error loading referral data:', error);
        });
});

document.getElementById('predictionForm').addEventListener('submit', function(event) {
    event.preventDefault();

    let pgsi = parseFloat(document.getElementById('pgsi').value);
    let core10 = parseFloat(document.getElementById('core10').value);
    let referralIndex = parseInt(document.getElementById('referralIndex').value, 10);


    fetch('http://192.168.1.240:31008/Prediction/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify({
            pgsi: pgsi,
            core10: core10,
            referralIndex: referralIndex
        })
    })
    .then(response => {
        if (response.ok) {
            return response.json();
        } else {
            throw new Error('Network response was not ok.');
        }
    })
    .then(data => {
        // Assuming the data is an object and we need its value
        let resultValue = Object.values(data)[0]; // Gets the first value of the object
        document.getElementById('predictionResult').innerHTML = `
            <div class="result-box">
                <strong>Suggested initial modality:</strong> ${resultValue}
            </div>
        `;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
