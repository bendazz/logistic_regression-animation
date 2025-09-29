# Logistic Regression Animation (Front-end only)

A minimal, framework-free web app for students to practice logistic regression intuition on a 1D dataset. Students paste a small CSV dataset `(x,y)` and a sequence of parameter updates `(a,b)` to animate how the sigmoid curve fits the data.

## What it shows
- Two-class 1D dataset: blue points (class 0) at y=0 and red points (class 1) at y=1.
- A horizontal dashed reference line at y=0.5.
- An animated sigmoid curve `σ(a·x + b)` updated from a CSV of `(a,b)` values.

## Files
- `index.html` — Main page with controls and the plot
- `style.css` — Styling
- `app.js` — Logic: CSV parsing, plotting, and animation

## How to use
1. Open `index.html` in a browser.
2. Load the sample dataset or paste your own CSV into the first textarea. The plot updates automatically. The feature x is in [0,10].
	- CSV format: two columns named `x,y` (header optional)
	- Example:

	  x,y
	0.4,0
	1.7,0
	3.2,1
	8.1,1

3. Paste a CSV of parameter updates `(a,b)` into the second textarea.
	- CSV format: two columns `a,b` (header optional)
	- Each row is one animation frame.
	- Example:

	  a,b
	  0.0,0.0
	  0.4,-0.5
	  0.9,-0.8
	  1.3,-1.0
	  1.6,-1.2

4. Click "Start Animation" to animate the sigmoid over your x-range.
	- Use Pause/Reset to control playback.
	- Adjust the speed slider (milliseconds per frame).

## Notes
- The app uses Chart.js via CDN for plotting; everything else is plain JavaScript.
- The sample dataset (x,y) is generated from a logistic model with noise and is deterministic.
- Points are drawn at y=0 or y=1 to emphasize class membership.

## Attribution
Made for teaching purposes. No frameworks; front-end only.