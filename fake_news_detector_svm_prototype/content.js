// Add styles for word highlighting
const style = document.createElement('style');
style.textContent = `
    .credible-word {
        background-color: rgba(39, 174, 96, 0.3);
        border-radius: 3px;
        padding: 0 2px;
    }
    .suspicious-word {
        background-color: rgba(192, 57, 43, 0.3);
        border-radius: 3px;
        padding: 0 2px;
    }
`;
document.head.appendChild(style);

// Function to get the main content text from the page
function getPageText() {
    // Get text from article tags if they exist
    const articles = document.getElementsByTagName('article');
    if (articles.length > 0) {
        return articles[0].innerText;
    }

    // Get text from main tag if it exists
    const main = document.getElementsByTagName('main');
    if (main.length > 0) {
        return main[0].innerText;
    }

    // Fallback to body text
    return document.body.innerText;
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "getText") {
        const text = getPageText();
        sendResponse({ text: text });
    } 
    else if (request.action === "highlightWords") {
        try {
            // Remove existing highlights
            const highlights = document.querySelectorAll('.credible-word, .suspicious-word');
            highlights.forEach(el => {
                const parent = el.parentNode;
                parent.replaceChild(document.createTextNode(el.textContent), el);
            });

            // Skip if no words to highlight
            if (!request.credibleWords?.length && !request.suspiciousWords?.length) {
                console.warn('No words to highlight.');
                return;
            }

            const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );

            let node;
            while (node = walker.nextNode()) {
                let content = node.textContent;
                if (!content.trim()) continue;

                let newContent = content;

                // Highlight suspicious words
                if (request.suspiciousWords?.length) {
                    const suspiciousPattern = new RegExp(
                        '\\b(' + request.suspiciousWords.join('|') + ')\\b',
                        'gi'
                    );
                    newContent = newContent.replace(suspiciousPattern, 
                        match => `<span class="suspicious-word">${match}</span>`
                    );
                }

                // Highlight credible words
                if (request.credibleWords?.length) {
                    const crediblePattern = new RegExp(
                        '\\b(' + request.credibleWords.join('|') + ')\\b',
                        'gi'
                    );
                    newContent = newContent.replace(crediblePattern, 
                        match => `<span class="credible-word">${match}</span>`
                    );
                }

                if (newContent !== content) {
                    const span = document.createElement('span');
                    span.innerHTML = newContent;
                    node.parentNode.replaceChild(span, node);
                }
            }
        } catch (error) {
            console.error('Error highlighting words:', error);
        }
    }
    return true;
});