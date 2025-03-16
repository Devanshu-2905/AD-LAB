document.addEventListener('DOMContentLoaded', function() {
    const urlInput = document.getElementById('urlInput');
    const scrapeButton = document.getElementById('scrapeButton');
    const questionInput = document.getElementById('questionInput');
    const askButton = document.getElementById('askButton');
    const scrapeStatus = document.getElementById('scrapeStatus');
    const chatHistory = document.getElementById('chatHistory');

    // Disable ask button initially
    askButton.disabled = true;

    scrapeButton.addEventListener('click', async function() {
        const url = urlInput.value.trim();
        if (!url) {
            alert('Please enter a valid URL');
            return;
        }

        // Show loading state
        scrapeButton.disabled = true;
        scrapeStatus.innerHTML = '<div class="loading"></div> Scraping content...';

        try {
            const response = await fetch('/scrape', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: url })
            });

            const data = await response.json();

            if (data.status === 'success') {
                scrapeStatus.innerHTML = `<div class="alert alert-success">Content scraped successfully!</div>`;
                askButton.disabled = false;
            } else {
                scrapeStatus.innerHTML = `<div class="alert alert-danger">Error: ${data.message}</div>`;
            }
        } catch (error) {
            scrapeStatus.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        }

        scrapeButton.disabled = false;
    });

    askButton.addEventListener('click', async function() {
        const question = questionInput.value.trim();
        if (!question) {
            alert('Please enter a question');
            return;
        }

        // Show loading state
        askButton.disabled = true;
        addToChatHistory('question', question);

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question })
            });

            const data = await response.json();

            if (data.status === 'success') {
                addToChatHistory('answer', data.answer);
            } else {
                addToChatHistory('answer', `Error: ${data.message}`);
            }
        } catch (error) {
            addToChatHistory('answer', `Error: ${error.message}`);
        }

        // Reset input and enable button
        questionInput.value = '';
        askButton.disabled = false;
    });

    function addToChatHistory(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type}`;
        messageDiv.innerHTML = `<strong>${type === 'question' ? 'Q' : 'A'}:</strong> ${content}`;
        chatHistory.insertBefore(messageDiv, chatHistory.firstChild);
    }
});