## Black Hole Ray Tracer (Python + Taichi)

This project is a simplified Python reimplementation of the [black_hole](https://github.com/kavan010/black_hole) repository by kavan010. The original C++/OpenGL project simulated gravitational lensing, an accretion disk, and warped spacetime grids around a black hole. This version recreates those effects using **Python** and **Taichi** preserving the core physics and visuals.

### Features
- Backward ray tracing in curved spacetime
- Schwarzschild black hole
- Thin accretion disk with gravitational + Doppler redshift
- Visualization of warped spacetime grid


### Setup
```bash
# Clone repo
git clone git@github.com:andrewqin29/taichi-blackhole-sim.git
cd taichi-blackhole-sim

# Install dependencies
pip install -r requirements.txt

# Run
python src/main.py
```

### Project Structure
- **main.py** — entrypoint, GUI loop
- **config.py** — simulation and rendering parameters
- **physics.py** — Schwarzschild metric, geodesic integrator, redshift calculations
- **scene.py** — camera rays, disk geometry, spacetime grid
- **kernel.py** — Taichi kernel that ties it all together

---

The result is a minimal visualization: a black hole shadow, a redshifted accretion disk, and warped grid lines showing spacetime curvature.
