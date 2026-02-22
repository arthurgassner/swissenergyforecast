// Fetch our latest custom forecast data
async function fetchForecastData() {
  const response = await fetch('/api/forecast/custom');
  if (!response.ok) {
    throw new Error('Network response was not ok: ' + response.statusText);
  }
  return response.json();
}

// Fetch actual ENTSO-E loads data
async function fetchEntsoeLoads() {
  const response = await fetch('/api/loads?days=3&hours=1');
  if (!response.ok) {
    throw new Error('Network response was not ok: ' + response.statusText);
  }
  return response.json();
}

// Fetch official ENTSO-E forecast data using the range endpoint
async function fetchEntsoeForecast() {
  // Calculate timestamps for the last 3 days and 1 hour (73 hours total)
  const endTs = Math.floor(Date.now() / 1000);
  const startTs = endTs - (73 * 60 * 60);

  const response = await fetch(`/api/forecast/entsoe/range?start_ts=${startTs}&end_ts=${endTs}`);
  if (!response.ok) {
    throw new Error('Network response was not ok: ' + response.statusText);
  }
  return response.json();
}

// Create Plotly traces with 24h future time adjustment
function createTraces(forecastData, entsoeLoadsData, entsoeForecastData) {
  const oneDayInMilliseconds = 24 * 60 * 60 * 1000;

  const actualLoadTrace = {
    x: entsoeLoadsData.timestamps.map(t => new Date(new Date(t).getTime() + oneDayInMilliseconds)),
    y: entsoeLoadsData['day_later_loads'],
    mode: 'lines',
    type: 'scatter',
    name: 'Actual Load [MW]',
    connectgaps: false
  };

  const officialForecastTrace = {
    x: entsoeForecastData.timestamps.map(t => new Date(new Date(t).getTime() + oneDayInMilliseconds)),
    y: entsoeForecastData['day_later_predicted_loads'],
    mode: 'lines',
    type: 'scatter',
    name: "ENTSO-E's previous-day forecasted load [MW]",
    opacity: 0.3,
    connectgaps: false
  };

  const ourForecastTrace = {
    x: forecastData.timestamps.map(t => new Date(new Date(t).getTime() + oneDayInMilliseconds)),
    // If y is null, keep it null; otherwise round it
    y: forecastData.day_later_predicted_loads.map(y => y === null ? null : Math.round(y)),
    mode: 'lines',
    type: 'scatter',
    name: 'Our previous-day forecasted load [MW]',
    connectgaps: false
  };

  return [actualLoadTrace, officialForecastTrace, ourForecastTrace];
}

// Create Plotly layout with the vertical line and "Now" text
function createLayout() {
  const currentTime = new Date();
  return {
    title: 'Load and forecasted load [MW]',
    xaxis: { title: 'Time' },
    yaxis: { title: 'Load [MW]' },
    plot_bgcolor: '#1e1e1e',
    paper_bgcolor: '#1e1e1e',
    font: { color: '#ffffff' },
    legend: {
      orientation: 'h',
      yanchor: 'top',
      y: 1.2,
      xanchor: 'center',
      x: .85
    },
    shapes: [
      {
        type: 'line',
        x0: currentTime,
        x1: currentTime,
        y0: 0,
        y1: 1,
        xref: 'x',
        yref: 'paper',
        line: {
          color: 'rgba(255, 0, 0, 0.5)',
          width: 2,
          dash: 'dot'
        }
      }
    ],
    annotations: [
      {
        x: currentTime,
        y: 0,
        xref: 'x',
        yref: 'paper',
        text: 'Now',
        showarrow: false,
        xanchor: 'left',
        yanchor: 'bottom',
        font: {
          color: 'rgba(255, 0, 0, 0.9)',
          size: 12
        }
      }
    ]
  };
}

// Render the Plotly chart
function renderChart(forecastData, entsoeLoadsData, entsoeForecastData) {
  const traces = createTraces(forecastData, entsoeLoadsData, entsoeForecastData);
  const layout = createLayout();
  Plotly.newPlot('plotly-chart', traces, layout);
}

// Main function to fetch data and render chart
async function main() {
  try {
    // Fetch all three endpoints concurrently
    const [forecastData, entsoeLoadsData, entsoeForecastData] = await Promise.all([
      fetchForecastData(),
      fetchEntsoeLoads(),
      fetchEntsoeForecast()
    ]);

    renderChart(forecastData, entsoeLoadsData, entsoeForecastData);
  } catch (error) {
    console.error('Error fetching data:', error);
  }
}

// Initialize when DOM content is loaded
document.addEventListener("DOMContentLoaded", main);