# NLP-Tweet_analysis

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  

This is a machine learning project to evaluate a proof of concept using natural language processing for sentiment classification.

## Description  

We carried out a proof of concept to evaluate the performance improvements of the roBERTa model compared to the BERT model. We were inspired by the study of Liu, Y. et al. (2019) ROBERTA : *A robustly optimized BERT pretraining approach.* [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692).<br>
You can test the proof of concept by yourself in the [train](./train) file in order to visualize the improvements made by the roBERTa model. <br>
We also created a dashboard and an API to visualize the results and test the model.<br>
<br>
![roberta](./documentation/all-roberta-large-v1.png)

#### You can download data from this [link](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+7%C2%A0-+D%C3%A9tectez+les+Bad+Buzz+gr%C3%A2ce+au+Deep+Learning/sentiment140.zip)



## Table of Contents

- [Getting Started](#Getting-Started)
- [Launch Application](#Launch-Application)
- [Contributing](#Contributing)
- [License](#License)

## Getting Started

#### For the Proof of Concept

```bash
git clone https://github.com/HaDock404/ai-nlp-tweet_sentiment.git
cd ai-nlp-tweet_sentiment
pip install -r ./train/packages/requirements.txt
open ./train/notebook.ipynb
```  

#### For the Application

```bash
pip install -r ./production/app/packages/requirements.txt
```  

#### For the API

```bash
pip install -r ./production/api/packages/requirements.txt
```

## Launch API

```bash
uvicorn ./production/api/api:app --port 8080 --reload
```  

## Launch Application

```bash
uvicorn ./production/app/main:app --port 8081 --reload
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License  

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.