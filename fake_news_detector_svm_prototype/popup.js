document.addEventListener('DOMContentLoaded', function() {
  const checkButton = document.getElementById('checkCredibility');
  const resultDiv = document.getElementById('result');
  const probabilityDiv = document.getElementById('probability');
  const progressBarFill = document.querySelector('.progress-bar-fill');
  const loadingDiv = document.getElementById('loading');
  const summaryDiv = document.getElementById('summary');
  const tipsDiv = document.getElementById('tips');
  const wordInfluenceDiv = document.getElementById('wordInfluence');

  checkButton.addEventListener('click', function() {
      // Reset UI
      loadingDiv.classList.remove('hidden');
      resultDiv.textContent = '';
      probabilityDiv.textContent = '';
      summaryDiv.textContent = '';
      progressBarFill.style.width = '0%';
      tipsDiv.classList.add('hidden');
      wordInfluenceDiv.classList.add('hidden');

      // Get current tab
      chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
          if (!tabs[0]) {
              loadingDiv.classList.add('hidden');
              resultDiv.textContent = 'No active tab found';
              return;
          }

          const activeTab = tabs[0];

          // Get text from page
          chrome.tabs.sendMessage(activeTab.id, { action: "getText" }, function(response) {
              if (chrome.runtime.lastError) {
                  loadingDiv.classList.add('hidden');
                  resultDiv.textContent = 'Error: Please refresh the page and try again';
                  return;
              }

              if (!response || !response.text) {
                  loadingDiv.classList.add('hidden');
                  resultDiv.textContent = 'No text content found on page';
                  return;
              }

              // Send to backend
              fetch('http://localhost:5000/predict', {
                  method: 'POST',
                  headers: {
                      'Content-Type': 'application/json',
                  },
                  body: JSON.stringify({ text: response.text }),
              })
              .then(result => result.json())
              .then(data => {
                  // Update UI with results
                  loadingDiv.classList.add('hidden');
                  wordInfluenceDiv.classList.remove('hidden');

                  const credibilityClass = data.credibility.toLowerCase();
                  resultDiv.innerHTML = `Credibility: <span class="${credibilityClass}">${data.credibility}</span>`;
                  
                  const suspiciousPercentage = (data.suspicious_probability * 100).toFixed(2);
                  probabilityDiv.textContent = `Probability of being suspicious: ${suspiciousPercentage}%`;
                  progressBarFill.style.width = `${suspiciousPercentage}%`;
                  progressBarFill.style.backgroundColor = getColorForPercentage(data.suspicious_probability);

                  if (data.summary) {
                      summaryDiv.innerHTML = `<div class="${credibilityClass}-highlight">${data.summary}</div>`;
                  } else {
                      summaryDiv.textContent = 'No summary available.';
                  }
                  
                  tipsDiv.classList.remove('hidden');

                  animateProgressBar(0, suspiciousPercentage);
              })
              .catch(error => {
                  console.error('Error:', error);
                  loadingDiv.classList.add('hidden');
                  resultDiv.textContent = 'Error occurred while checking credibility';
              });
          });
      });
  });

  function getColorForPercentage(percentage) {
      const red = percentage > 0.5 ? 255 : Math.round(510 * percentage);
      const green = percentage < 0.5 ? 255 : Math.round(510 * (1 - percentage));
      return `rgb(${red}, ${green}, 0)`;
  }

  function animateProgressBar(current, target) {
      if (current <= target) {
          progressBarFill.style.width = `${current}%`;
          setTimeout(() => animateProgressBar(current + 1, target), 10);
      }
  }
});