# Compression Distance

A compression-based edit distance metric for comparing texts, as described in [Assessing Human Editing Effort on LLM-Generated Texts via Compression-Based Edit Distance](https://arxiv.org/abs/2412.17321).

## Installation

```bash
# Basic installation
pip install compression-distance

# With experiment dependencies
pip install compression-distance[experiments]
```

## Example usage

```python
from compression_distance import compression_distance

text1 = "Hello world"
text2 = "Hello there"
distance = compression_distance(text1, text2)
print(f"Distance: {distance}")
```

## Testing

```bash
pytest tests/
```

## Experiments

This repository also includes experiments comparing our metric with traditional ones (BLEU, TER, ROUGE, etc.) on two datasets:

- A **custom dataset** of accounting Q&A edits ([HuggingFace link](https://huggingface.co/datasets/Tiime/fr-qa-accounting-edits)).
- The **IWSLT2019** dataset ([GitHub link](https://github.com/carolscarton/iwslt2019)).

*For the IWSLT2019 dataset, download the data from the [official repository](https://github.com/carolscarton/iwslt2019) and place them in an "iwslt2019" folder at the root of this project.*

### Running the Experiments

1. First, install the experiment dependencies:
```bash
pip install -e ".[experiments]"
```

2. For the IWSLT2019 dataset, download the data from the [official repository](https://github.com/carolscarton/iwslt2019) and place them in an "iwslt2019" folder at the root of this project.

3. Run the experiments:
```bash
python experiments/run_experiments.py
```

You can modify the distance metrics list in `run_experiments.py` to include or exclude specific metrics.

### Prompts used for the synthetic dataset

The following prompts (in French) were used to generate LLM answers and edits:

#### Initial answer generation
```text
Tu dois répondre à une question de comptabilité posée dans un email. Je vais te donner la question <QUESTION>. 
Tu dois répondre à la question posée de manière détaillée et précise. Réponds avec uniquement le contenu de 
la réponse et rien d'autre.
```

#### **Normal** edits
```text
Tu dois répondre à une question de comptabilité posée dans un email. Je vais te donner la question <QUESTION>, 
une réponse générée par un modèle non spécialisé <LLM_ANSWER>, et des connaissances spécifiques et spécialisées 
qui permettent de répondre à cette question <KNOWLEDGE>. Tu dois éditer la réponse du modèle à partir des 
connaissances transmises afin d'améliorer la réponse quand cela est nécessaire. Tu peux modifier la réponse 
autant que tu le souhaites de manière à améliorer la réponse initiale avec ces connaissances. Tu ne dois pas 
copier/coller les connaissances dans la réponse mais utiliser de manière adaptée les éléments pertinents de 
ces connaissances afin de mettre à jour la réponse initiale. Réponds avec uniquement le contenu de la réponse 
et rien d'autre.
```

#### **Similar** edits
```text
Tu dois répondre à une question de comptabilité posée dans un email. Je vais te donner la question <QUESTION>, 
une réponse générée par un modèle non spécialisé <LLM_ANSWER>, et des connaissances spécifiques et spécialisées 
qui permettent de répondre à cette question <KNOWLEDGE>. Tu dois éditer la réponse du modèle à partir des 
connaissances transmises afin d'améliorer la réponse quand cela est nécessaire. Tu dois modifier la réponse 
de manière à garder la même structure et la même trame que la réponse initiale, en modificant si cela est 
nécessaire uniquement le fond à partir des connaissances transmises. Tu ne dois pas copier/coller les 
connaissances dans la réponse mais utiliser de manière adaptée les éléments pertinents de ces connaissances 
afin de mettre à jour la réponse initiale. Réponds avec uniquement le contenu de la réponse et rien d'autre.
```

#### **Fast** edits
```text
Tu dois répondre à une question de comptabilité posée dans un email. Je vais te donner la question <QUESTION>, 
une réponse générée par un modèle non spécialisé <LLM_ANSWER>, et des connaissances spécifiques et spécialisées 
qui permettent de répondre à cette question <KNOWLEDGE>. Tu dois éditer la réponse du modèle à partir des 
connaissances transmises afin d'améliorer la réponse quand cela est nécessaires. Tu dois prendre le moins de 
temps possible pour éditer la réponse initiale tout en réalisant la tâche correctement. Tu ne dois pas 
copier/coller les connaissances dans la réponse mais utiliser de manière adaptée les éléments pertinents de 
ces connaissances afin de mettre à jour la réponse initiale. Réponds avec uniquement le contenu de la réponse 
et rien d'autre.
```

## Citing

If you use this metric in your research, please cite:
```bibtex
@misc{devatine2024assessinghumaneditingeffort,
      title={Assessing Human Editing Effort on LLM-Generated Texts via Compression-Based Edit Distance}, 
      author={Nicolas Devatine and Louis Abraham},
      year={2024},
      eprint={2412.17321},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.17321}, 
}
```