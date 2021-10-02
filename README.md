# Yelp Rating Prediction

Final project for Full Stack Deep Learning at UC Berkeley in Spring 2021. My project report can be found [here](https://raguvir.me/fsdl.pdf).

You can interact with the model by sending a cURL request:

```
curl -X POST https://yelp-model.onrender.com/predict --header 'Content-Type: application/json' --data-raw '{"review": "I love this restaurant."}'
```

## Project Vision

Currently, there exists a problem where between 2 distinct reviewers' perceptions of a 5-star (or any-star) rating (of a restaurant, for example) may be vastly different. Building a model that attempts to predict ratings from reviews could be useful to Yelp and other business review sites in an effort to standardize how businesses are rated. The above situation could potentially be mitigated by using the output of a model to recommend a rating to the reviewer prior to publishing the review. Based on model performance and business needs, these sites could even enforce ratings to be generated algorithmically. My intention is to create a model that predicts user ratings given text of a review, then deploy the model as a tool for suggesting ratings to users.
