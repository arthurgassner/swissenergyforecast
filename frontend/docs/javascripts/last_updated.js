async function fetchLastUpdated() {
    try {
        // Reuse the new custom forecast endpoint
        const response = await fetch('/api/forecast/custom/latest');
        if (!response.ok) {
            throw new Error('Network response was not ok: ' + response.statusText);
        }

        const data = await response.json();

        // The new schema uses "created_at" as an ISO string
        displayLastUpdated(data.created_at);
    } catch (error) {
        console.error('Error fetching timestamp:', error);
        document.getElementById('last-updated').textContent = 'Error fetching last updated time.';
    }
}

function displayLastUpdated(timestamp) {
    if (!timestamp) return;

    // Parse the ISO 8601 string directly
    const lastUpdatedDate = new Date(timestamp);
    const timeAgo = timeSince(lastUpdatedDate);
    document.getElementById('last-updated').textContent = `Last update: ${timeAgo}`;
}

function timeSince(date) {
    const seconds = Math.floor((new Date() - date) / 1000);

    const intervals = [
        { label: 'year', seconds: 31536000 },
        { label: 'month', seconds: 2592000 },
        { label: 'day', seconds: 86400 },
        { label: 'hour', seconds: 3600 },
        { label: 'minute', seconds: 60 },
        { label: 'second', seconds: 1 }
    ];

    for (const interval of intervals) {
        const count = Math.floor(seconds / interval.seconds);
        if (count >= 1) {
            return `${count} ${interval.label}${count > 1 ? 's' : ''} ago`;
        }
    }
    return "<1s ago";
}

// Initialize
document.addEventListener("DOMContentLoaded", fetchLastUpdated);