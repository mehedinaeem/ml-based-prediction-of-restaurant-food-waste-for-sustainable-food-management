from setuptools import setup, find_packages

setup(
    name="ml_restaurant_food_waste",
    version="0.1.0",
    description="Machine learning pipeline for restaurant food waste prediction.",
    author="Your Name",
    author_email="you@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "plotly",
        "streamlit",
        "fastapi",
        "uvicorn",
        "joblib",
        "pyyaml",
    ],
    python_requires=">=3.8",
)
