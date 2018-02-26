# Suggestion-Engine
Apply 2D Gaussian Filter on Gaussian Distribution to suggest Nearest Neighbors

Steps that I used:
1. Create augmentation dimension from genres, keywords, and etc.
2. Calculate Euclidean Distance.
![alt text](http://mcla.ug/blog/images/EuclideanDistanceGraphic.jpg)

3. Sort based on 2D Gaussian filter for ratings, user counts and etc.
![alt text](CodeCogsEqn.png)

4. Apply Gaussian Distribution to check distance between 2 points in the distribution.
![alt text](http://ww2.tnstate.edu/ganter/BIO311-CH4-Eq1a.gif)

### Suggestion from Anime, you can download the data-set [here](https://www.kaggle.com/CooperUnion/anime-recommendations-database)
![alt text](anime.png)

### Suggestion from Game, you can download the data-set [here](https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings)
![alt text](game.png)

### Suggestion from Movie, you can download the data-set [here](https://www.kaggle.com/rounakbanik/the-movies-dataset)
![alt text](movie.png)
