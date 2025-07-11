# PyTorch VGG Figures

## Description
The PyTorch VGG Figures project is a Python-based implementation that leverages PyTorch to create and visualize figures using VGG (Visual Geometry Group) neural network models. The project focuses on generating visualizations to demonstrate the behavior of VGG models, such as VGG16 and VGG19, on various image datasets, emphasizing image classification and feature visualization.

## Project Overview
This project, hosted at [https://github.com/MaksKroha/pytorch-vgg-figures](https://github.com/MaksKroha/pytorch-vgg-figures), utilizes PyTorch to implement VGG models for image classification and visualization tasks. It builds upon the VGG architectures (e.g., VGG16, VGG19) introduced in the "Very Deep Convolutional Networks for Large-Scale Image Recognition" paper. The project is designed to generate insightful figures, such as feature maps, decision boundaries, or classification outputs, to illustrate how VGG models process image data. It is suitable for educational purposes and as a foundation for advanced computer vision experiments.[](https://docs.pytorch.org/vision/main/models/vgg.html)

## Features
- Implementation of VGG models (VGG16, VGG19) using PyTorch
- Visualization of feature maps, decision boundaries, and classification results
- Support for custom image datasets and preprocessing pipelines
- Modular code structure for experimenting with different VGG configurations
- Integration with pre-trained VGG weights from the PyTorch model zoo
- Generation of high-quality plots using Matplotlib and Seaborn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MaksKroha/pytorch-vgg-figures.git
   ```
2. Navigate to the project directory:
   ```bash
   cd pytorch-vgg-figures
   ```
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure all dependencies are installed.
2. Run the main script to train or evaluate a VGG model and generate visualizations:
   ```bash
   python src/main.py
   ```
3. Check the `outputs/` directory for generated figures, such as feature maps or classification plots.

## Examples
To generate visualizations for a sample dataset using VGG16:
```bash
python src/main.py --model vgg16 --dataset data/sample_images.csv --plot feature_maps
```
This command loads a pre-trained VGG16 model, processes the dataset, and generates feature map visualizations saved in `outputs/`.

Example output:
```
Model: VGG16
Accuracy on test set: 73.2%
Feature maps saved to outputs/feature_maps_vgg16.png
```

## Dependencies
- Python 3.8+
- torch>=1.2.0
- torchvision>=0.4.0
- numpy>=1.19.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- seaborn>=0.11.0

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure
```
pytorch-vgg-figures/
├── data/                 # Sample image datasets
├── src/                  # Source code
│   ├── main.py           # Main script to run the project
│   ├── vgg.py            # VGG model implementation
│   ├── preprocess.py     # Image preprocessing functions
│   └── visualize.py      # Visualization functions for generating figures
├── outputs/              # Generated plots and results
├── tests/                # Unit tests
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## To-do / Roadmap
- Add support for additional VGG variants (e.g., VGG11, VGG13)
- Implement advanced visualizations (e.g., Grad-CAM, activation maximization)
- Support for transfer learning with custom datasets
- Optimize model training for larger datasets
- Integrate GPU acceleration for faster visualization generation

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
