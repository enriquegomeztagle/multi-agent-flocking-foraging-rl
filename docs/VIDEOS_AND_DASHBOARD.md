# Usage Guide: Videos and Dashboard

This document explains how to use the visualization features: video generation and interactive dashboard.

## 🎬 Video Generation

### Requirements

The dependencies are already in `requirements.txt`:

- `matplotlib>=3.8`
- `imageio>=2.34`
- `imageio-ffmpeg>=0.4.9`

### Basic Usage

#### 1. Generate video from trained RL model

```bash
PYTHONPATH=. python scripts/generate_video.py \
  --type rl \
  --model models/ppo_easy/final_model.zip \
  --config configs/env_easy.yaml \
  --output videos/easy_final.mp4 \
  --episodes 1 \
  --fps 20
```

#### 2. Generate video from Baseline Boids

```bash
PYTHONPATH=. python scripts/generate_video.py \
  --type baseline \
  --config configs/env_easy.yaml \
  --output videos/baseline_boids.mp4 \
  --episodes 1 \
  --fps 20
```

### Parameters

- `--type`: Agent type (`rl` or `baseline`)
- `--model`: Path to RL model (required if `--type rl`)
- `--config`: Path to YAML configuration file
- `--output`: Output video path (e.g., `videos/my_video.mp4`)
- `--episodes`: Number of episodes to record (default: 1)
- `--steps`: Maximum steps per episode (default: uses config value)
- `--fps`: Frames per second of the video (default: 10)
- `--seed`: Random seed (default: random)

### Advanced Examples

#### Multiple episodes

```bash
PYTHONPATH=. python scripts/generate_video.py \
  --type rl \
  --model models/ppo_easy_mode/best_model/best_model \
  --config configs/env_easy_mode.yaml \
  --output videos/easy_mode_5_episodes.mp4 \
  --episodes 5 \
  --fps 10
```

#### Faster video (higher FPS)

```bash
PYTHONPATH=. python scripts/generate_video.py \
  --type rl \
  --model models/ppo_easy_mode/best_model/best_model \
  --config configs/env_easy_mode.yaml \
  --output videos/easy_mode_fast.mp4 \
  --fps 20
```

#### Shorter episode

```bash
PYTHONPATH=. python scripts/generate_video.py \
  --type baseline \
  --config configs/env_easy_mode.yaml \
  --output videos/baseline_short.mp4 \
  --steps 500
```

### Video Features

The videos show:

- **Agents** (cyan circles) with direction (yellow arrows)
- **Resource patches** with colors according to stock level:
  - Green: high stock
  - Red: low stock
- **Real RL behavior**: Agents use trained neural networks with observation normalization for natural movement
- **Natural movement**: Proper VecNormalize loading ensures agents accelerate, turn, and coordinate naturally

## 📊 Interactive Dashboard with Streamlit

### Installation

Required dependencies:

```bash
pip install streamlit plotly pandas numpy scipy
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

**Note**: The dashboard uses `scipy` for statistical analysis (linear regression, etc.) and `pandas` for data manipulation.

### Run Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open automatically in your browser (usually at `http://localhost:8501`).

### Dashboard Features

#### 1. File Selection

- Left sidebar to select JSON result files from `results/` directory
- Automatic detection of file type (RL Agent or Baseline Boids)
- Displays configuration type and number of episodes

#### 2. Summary Metrics

The dashboard displays four key metrics at the top:

- **Mean Efficiency**: Percentage of theoretical maximum (with standard deviation)
- **Mean Intake**: Total amount of resources consumed (with standard deviation)
- **Mean Gini**: Equity measure (0 = perfect, 1 = unequal) (with standard deviation)
- **Polarization** or **Episodes Count**: Velocity alignment (if available) or total episodes evaluated

#### 3. Visualization Tabs

The dashboard has 5 main tabs for analysis:

**📊 Visual Summary** (Tab 1)

- Multi-metric box plots showing distribution of key metrics (Efficiency, Gini, Reward, Polarization)
- Performance tier breakdown (excellent, good, fair, poor) with color-coded bar chart
- Efficiency histogram with color-coded ranges (green ≥80%, cyan ≥60%, orange ≥40%, red <40%)
- Scatter plot: Efficiency vs Gini (Fairness) with reward coloring and intake sizing
- Statistical summary table with descriptive statistics (mean, std, min, max, quartiles, coefficient of variation)

**🧠 Learning Dynamics** (Tab 2)

- **Convergence and Stability**:
  - Rolling mean of efficiency with multiple window sizes (MA5, MA10, MA20)
  - Rolling standard deviation showing policy volatility
- **Episode-to-Episode Improvement**:
  - Improvement rate (percentage of episodes that improved)
  - Average improvement when performance increases
  - Average decline when performance decreases
  - Distribution histogram of episode-to-episode changes
  - Cumulative mean showing global convergence
- **Policy Consistency**:
  - Consistency rate (episodes within ±5% of previous)
  - Coefficient of variation trend (stability over time)
  - Maximum consecutive improvement streak
- **Performance Quartiles**: Scatter plot colored by performance quartiles (Q1-Q4)
- **Temporal Success Rate**: Rolling success rate based on median threshold

**📈 Trends** (Tab 3)

- Multi-metric time series plots with dual y-axes:
  - **Top panel**: Efficiency (with moving average) and Gini coefficient
  - **Bottom panel**: Reward (with moving average) and Steps
- **Trend Analysis**:
  - Linear regression for efficiency trend (slope, R², p-value)
  - Linear regression for reward trend (if available)
  - Stability change analysis comparing first half vs second half variability

**🎯 Best/Worst Episodes** (Tab 4)

- **Top 5 Best Episodes**: Table with efficiency, intake, Gini, reward (color-coded green gradient)
- **Bottom 5 Worst Episodes**: Table with same metrics (color-coded red gradient)
- **Average characteristics** for both best and worst episodes
- **Comparative Analysis**: Side-by-side comparison showing differences between top and bottom performers
- **Outlier Detection**: IQR-based outlier identification with statistics and outlier table

**📋 Data** (Tab 5)

- Complete episode-by-episode data table with:
  - Filter by minimum efficiency threshold (slider)
  - Sort by any column (episode, efficiency, Gini, intake, reward)
  - Sort order (ascending/descending)
  - Color-coded efficiency column (green ≥80%, cyan ≥60%, yellow ≥40%, red <40%)
- **Download Options**:
  - Download filtered data as CSV
  - Download complete JSON file

### Dashboard Usage

1. **Select file**: Use the sidebar to choose a JSON result file from `results/` directory
2. **Review summary**: Check the top metrics cards for quick overview
3. **Explore Visual Summary**: Start with Tab 1 to see overall distributions and performance tiers
4. **Analyze Learning Dynamics**: Use Tab 2 to understand convergence, stability, and policy consistency
5. **Examine Trends**: Check Tab 3 for temporal patterns and trend analysis
6. **Compare Extremes**: Review Tab 4 to see what makes best/worst episodes different
7. **Export Data**: Use Tab 5 to filter, sort, and download data as CSV or JSON

### Supported Files

The dashboard supports two types of JSON result files:

**RL Agent Results** (from evaluation scripts):

- `statistics` field with mean, std, etc.
- `all_episodes` or `episodes` field with list of episodes
- Each episode should have: `episode`, `efficiency_percent`, `intake`, `gini`, `reward`, `steps` (optional: `polarization`, `mean_neighbor_distance`)
- `configuration` field indicating the environment config
- `n_episodes` field with total episode count
- Optional: `theoretical_max`, `performance_tiers`

**Baseline Boids Results** (from baseline evaluation):

- `summary` field with summary metrics (mean_efficiency_percent, mean_intake, mean_gini, etc.)
- `episodes` field with list of episodes
- Each episode should have: `episode`, `efficiency_percent`, `intake`, `gini`
- `n_episodes` field with total episode count

## 🎨 Environment Rendering

The render is implemented in `env/render.py` and can be used directly:

```python
from env.flockforage_parallel import FlockForageParallel, EnvConfig

env = FlockForageParallel(EnvConfig(**config))

# Render in human mode (interactive)
env.render(mode="human")

# Render to RGB array (for videos)
frame = env.render(mode="rgb_array")
```

### Render Modes

- **`mode="human"`**: Shows interactive window with matplotlib
- **`mode="rgb_array"`**: Returns numpy array (H, W, 3) to save as image or video

## 🐛 Troubleshooting

### Error: "No module named 'imageio'"

```bash
pip install imageio imageio-ffmpeg
```

### Error: "No module named 'streamlit'"

```bash
pip install streamlit plotly pandas numpy scipy
```

### Error: "No module named 'scipy'"

The dashboard requires scipy for statistical analysis:

```bash
pip install scipy
```

### Video not generated

- Verify that `imageio-ffmpeg` is installed
- Make sure the output directory exists or can be created
- Check that the model and config paths are valid

### Dashboard not loading files

- Verify that the `results/` directory exists
- Make sure the JSON files have the correct format
- Check the Streamlit console for errors

### Render very slow

- Reduce the number of agents or patches
- Reduce the resolution (change `dpi` in `render_to_rgb_array`)
- Use lower `fps` in videos

## 📝 Notes

### Video Generation

- Videos are saved in MP4 format using matplotlib rendering and imageio encoding
- RL models use observation normalization (VecNormalize) for consistent behavior
  - Stochastic action sampling provides natural movement variation
Models automatically load observation normalization from `vecnormalize.pkl`

### Dashboard

- The dashboard is interactive and updates automatically when you change file selection
- Data is cached using `@st.cache_data` for better performance (reloads only when file changes)
- The dashboard automatically detects file type (RL Agent vs Baseline Boids)
- All visualizations are interactive (zoom, pan, hover for details)
- You can run multiple dashboard instances with different ports using `--server.port`:
  ```bash
  streamlit run dashboard.py --server.port 8502
  ```
