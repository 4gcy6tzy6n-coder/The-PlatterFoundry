# The PlatterFoundry

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

The PlatterFoundry is a next-generation network transport algorithm research platform designed for academic researchers and network engineers. It provides a systematic approach to developing, testing, and validating network transport algorithms with academic-grade rigor.

## 🚀 Features

- **Academic-Grade Experiments**: IEEE/ACM ToN standard experiment workflows
- **Advanced Estimation Algorithms**: Adaptive Kalman filter implementations
- **Intelligent Network Modeling**: Multi-state packet loss models with EM learning
- **Fairness Validation**: Multi-dimensional fairness verification framework
- **Production-Ready**: Canary deployment and safety systems
- **Comprehensive Analytics**: Statistical analysis, causal inference, and explainability

## 📂 Project Structure

```
The PlatterFoundry/
├── src/                          # Core source code
│   ├── core/                     # Algorithm implementations
│   │   ├── estimation/           # Adaptive estimators
│   │   └── request_heuristic/    # Network request predictions
│   ├── network/                  # Network modeling
│   │   └── models/               # Packet loss models
│   ├── experiments/              # Experiment framework
│   │   ├── analysis/             # Statistical & fairness analysis
│   │   ├── experiment_controller/# Experiment orchestration
│   │   └── visualization/        # Result visualization
│   ├── production/               # Production deployment
│   └── utils/                    # Utility tools
│       └── performance_monitor.py
├── examples/                     # Example implementations
│   ├── complete_research_experiment.py
│   ├── enhanced_research_experiment.py
│   ├── network_experiment.py
│   └── production_demo.py
├── experiments/                  # Experiment scripts
│   ├── run_all_experiments.py
│   ├── experiment_1_1_formal_final.py
│   ├── experiment_1_2_loss_model_final.py
│   ├── experiment_2_1_mcts_defect_discovery_final.py
│   ├── experiment_2_2_root_cause_analysis.py
│   ├── experiment_2_3_production_resilience.py
│   └── experiment_3_full_platform_ab_test.py
├── data/                         # Data files
│   └── safety_casebase.db
├── tests/                        # Test code
├── .gitignore                    # Git ignore file
└── requirements.txt              # Python dependencies
```

## 🏗️ Core Modules

### 1. Adaptive Estimation (`src/core/estimation/`)
```python
from src.core.estimation import AdaptiveEstimator

estimator = AdaptiveEstimator(
    name="Proposed",
    short_window=10,
    long_window=100,
    burst_threshold=2.0,
    enable_adaptive_noise=True
)
```

### 2. Network Modeling (`src/network/models/`)
```python
from src.network.models import ImprovedLossModel

model = ImprovedLossModel(config)
loss_sequence = model.generate(n_packets=10000)
```

### 3. Fairness Validation (`src/experiments/analysis/`)
```python
from src.experiments.analysis import FairnessValidator

validator = FairnessValidator()
validator.register_algorithm_results("TCP_CUBIC", cubic_results)
validator.register_algorithm_results("BBR", bbr_results)

report = validator.validate_all()
print(f"Fairness passed: {report['overall_passed']}")
```

### 4. Statistical Analysis (`src/experiments/analysis/`)
- ANOVA testing
- Bootstrap confidence intervals
- Kruskal-Wallis tests
- Effect size calculations

## 🧪 Experiments

The platform includes 6 experiments for comprehensive validation:

- **Experiment 1.1**: Adaptive estimator performance evaluation
- **Experiment 1.2**: Packet loss model accuracy comparison  
- **Experiment 2.1**: MCTS intelligent defect discovery
- **Experiment 2.2**: Root cause analysis & causal inference
- **Experiment 2.3**: Production resilience testing
- **Experiment 3**: Full platform A/B testing

### Run Experiments
```bash
cd experiments
python run_all_experiments.py
```

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/4gcy6tzy6n-coder/The-PlatterFoundry.git
cd The-PlatterFoundry

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Experiment
```bash
cd examples
python complete_research_experiment.py
```

### Run Individual Components
```bash
# Run specific experiment
cd experiments
python experiment_1_1_formal_final.py

# Run production demo
cd examples
python production_demo.py
```

## 📦 Requirements

- Python 3.8+
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- pulp (for optimization)

See `requirements.txt` for full dependencies.

## 📊 Results

The platform generates:
- Statistical analysis reports (CSV, TXT)
- Interactive visualizations (HTML)
- Academic-grade figures (PDF, PNG)
- Fairness validation reports
- Performance benchmarks

## 🔧 Usage Examples

### 1. Complete Research Experiment
```python
from examples.complete_research_experiment import run_complete_experiment

config = {
    "algorithms": ["TCP_CUBIC", "BBR", "QUIC"],
    "network_conditions": ["low_loss", "medium_loss", "high_loss"],
    "n_packets": 10000,
    "repeat_count": 30,
    "master_seed": 42
}

report = run_complete_experiment(**config)
```

### 2. Production Demo
```bash
python examples/production_demo.py
```

## 📈 Performance Benchmarks

Expected performance on standard test scenarios:
- Adaptive Estimator: < 5% RMSE on non-stationary networks
- Loss Model Accuracy: > 90% correlation with ground truth
- Fairness Validation: Complete resource balance check (< 5% variance)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions and support:
- Create an Issue on GitHub
- Email: [project maintainer]

## 🎯 Citation

If you use The PlatterFoundry in your research, please cite:

```bibtex
@article{PlatterFoundry,
  title={The PlatterFoundry: Next-Generation Network Transport Algorithm Research Platform},
  author={Yingliang Yao and Meng Xu and Yuanqi Liang and Liangdanlin Zhu and Yaoguang Wang and Ronglin Tang and Changbin Zhao and Mingyu Song and Bingshu Fan},
  journal={IEEE Network},
  year={2026},
  doi={10.1109/MNET.2026.3733134},
  url={https://github.com/4gcy6tzy6n-coder/The-PlatterFoundry}
}
```

## 🙏 Acknowledgments

- IEEE Network
- Network transport research community
- Open source contributors
```

---

**Status**: ✅ Project ready for production use