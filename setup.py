from setuptools import find_packages, setup

setup(
    name="clarity-eeg",
    version="0.1.0",
    description="Deep learning models for EEG-based depression detection",
    author="Clarity Research Team",
    author_email="info@clarity-research.org",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "mne",
        "scikit-learn",
        "scikit-image",
        "torch",
        "torchvision",
        "torchaudio",
        "matplotlib",
        "seaborn",
        "pywavelets",
        "tqdm",
        "ipywidgets",
    ],
)
