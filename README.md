# Martingale Optimal Transport (MOT) for Option Pricing

This project focuses on solving linear programming (LP) problems with martingale transport structure, known as Martingale Optimal Transport (MOT), for pricing options in financial markets. The project implements several mathematical frameworks, including the Bregman projection and Sinkhorn-Knopp algorithm, to accurately price different types of options.

## Project Structure

The project has the following structure:

```
LAB PROJECT
├── notebooks
│   ├── MOT_pricing
│   │   ├── call_option_pricing.ipynb
│   │   └── lookback_option_pricing.ipynb
│   └── MOT_uniform_laws
│       ├── MOT_uniform_cost1.ipynb
│       ├── MOT_uniform_cost2.ipynb
│       ├── MOT_uniform_cost3.ipynb
│       ├── MOT_uniform_cost4.ipynb
│       └── MOT_uniform_math_model.ipynb
├── reports
│   ├── figure
│   │   ├── call_option_bregman.png
│   │   ├── call_sensitivity_rates.png
│   │   ├── call_sensitivity_stock.png
│   │   ├── call_sensitivity_strike.png
│   │   ├── call_sensitivity_vol.png
│   │   ├── call_sensitivity.png
│   │   ├── lookback_sensitivity.png
│   │   └── uniform_cost1_sinkhorn.png
│   └── uniform_cost1.png
│       ├── uniform_cost2.png
│       ├── uniform_cost3.png
│       └── uniform_cost4.png
├── videos
│   ├── distribution_evolution.mp4
│   ├── MOT_distribution_evolution.mp4
│   ├── MOT_transport_animation.mp4
│   └── OT_transport_animation.mp4
├── README.md
└── requirements.txt
```

## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/your-username/mot-option-pricing.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Navigate to the desired notebook in the `notebooks` directory and run the cells to execute the code.

## Notebooks

The project includes the following notebooks:

- `MOT_pricing/call_option_pricing.ipynb`: Notebook for pricing European call options using MOT.
- `MOT_pricing/lookback_option_pricing.ipynb`: Notebook for pricing lookback options using MOT.
- `MOT_uniform_laws/MOT_uniform_cost1.ipynb`: Notebook demonstrating MOT with uniform laws and cost function 1 - c(x, y) = |x - y|.
- `MOT_uniform_laws/MOT_uniform_cost2.ipynb`: Notebook demonstrating MOT with uniform laws and cost function 2 - c(x, y) = |x - y|**2.
- `MOT_uniform_laws/MOT_uniform_cost3.ipynb`: Notebook demonstrating MOT with uniform laws and cost function 3 - c(x, y) = xy.
- `MOT_uniform_laws/MOT_uniform_cost4.ipynb`: Notebook demonstrating MOT with uniform laws and cost function 4 - c(x, y, a) = y-a*max(x,y).
- `MOT_uniform_laws/MOT_uniform_math_model.ipynb`: Notebook checking the mathematical model for MOT with uniform laws and cost function 1 - c(x, y) = |x - y|.

## Reports and Figures

The `reports` directory contains figures and videos generated during the project.

### Figures
The `figures` directory contains:

- Sensitivity analysis plots for call options and lookback options.
- Plots of the transport plan for different cost functions with uniform laws.

## Videos

The `videos` directory contains animated visualizations related to the project:

- `distribution_evolution.mp4`: Animation showing the evolution of the distribution.
- `MOT_distribution_evolution.mp4`: Animation showing the evolution of the distribution in the MOT setting.
- `MOT_transport_animation.mp4`: Animation of the transport plan in the MOT setting.
- `OT_transport_animation.mp4`: Animation of the transport plan in the OT setting.

## Requirements

The required dependencies for running the project are listed in the `requirements.txt` file. Install them using `pip install -r requirements.txt`.

## Acknowledgments

This project is based on the work of Gaspard Monge [1] and Kantorovich [2] in Optimal Transport theory and its application to financial markets.

[1] Monge, G. (1781). Mémoire sur la théorie des déblais et des remblais. Histoire de l'Académie Royale des Sciences de Paris.
[2] Kantorovich, L. V. (1942). On the translocation of masses. In Dokl. Akad. Nauk SSSR (Vol. 37, No. 7-8, pp. 227-229).
[3] D. G. Hobson, “Robust hedging of the lookback option,” Finance and Stochastics, vol. 2, pp. 329–347, 1998.
[4] M. Beiglb ̈ock, P. Henry-Labordere, and F. Penkner, “Model-independent bounds for option prices—a mass
transport approach,” Finance and Stochastics, vol. 17, pp. 477–501, 2013.
[5] G. Guo and J. Obl ́oj, “Computational methods for martingale optimal transport problems,” The Annals of
Applied Probability, vol. 29, no. 6, pp. 3311–3347, 2019.
[6] S. Eckstein, G. Guo, T. Lim, and J. Obl ́oj, “Robust pricing and hedging of options on multiple assets and its
numerics,” SIAM Journal on Financial Mathematics, vol. 12, no. 1, pp. 158–188, 2021.
[7] J.-D. Benamou, G. Carlier, M. Cuturi, L. Nenna, and G. Peyr ́e, “Iterative bregman projections for regularized
transportation problems,” SIAM Journal on Scientific Computing, vol. 37, no. 2, pp. A1111–A1138, 2015.
[8] J.-D. Benamou and Y. Brenier, “A computational fluid mechanics solution to the monge-kantorovich mass
transfer problem,” Numerische Mathematik, vol. 84, no. 3, pp. 375–393, 2000.
[9] B. L ́evy, “A numerical algorithm for l {2} semi-discrete optimal transport in 3d,” ESAIM: Mathematical Mod-
elling and Numerical Analysis, vol. 49, no. 6, pp. 1693–1715, 2015.
