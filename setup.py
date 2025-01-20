from setuptools import setup, find_packages

setup(
    name="compression-distance",
    version="0.1.0",
    description="A compression-based edit distance metric for text comparison",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Nicolas Devatine & Louis Abraham",
    author_email="nicolas.devatine@tiime.fr",
    url="https://github.com/NDV-tiime/CompressionDistance",
    packages=find_packages(include=["compression_distance", "compression_distance.*"]),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "tqdm",
        "evaluate",
        "nltk",
        "unidecode",
        "pydivsufsort",
        "Levenshtein",
        "pytest",
    ],
    extras_require={
        "experiments": [
            "evaluate",
            "datasets",
            "bert_score",
            "sacrebleu",
            "rouge_score",
            "cer",
        ],
    },
    classifiers=[],
    python_requires=">=3.8",
)
