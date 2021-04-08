# pol-disagreement
Political disagreement in non-political spaces


**Classifiers folder**

Contains content classifiers and training data relevant for the goals of the project, namely:

 - A disagreement classifier that determines if a text contains expressions of disagreement;
 - A political content classifier that determines if a text contains political content.
 - A political classifier based on titles: https://github.com/ercexpo/political_classification

These classifiers should generalize for Reddit, Facebook and Youtube text data (comments, posts, video descriptions).


**Scrapers folder**

Contains scrapers for social media content to be classified using the classifiers. In its current state, it contains the following:

- A Youtube comment scraper that scrapes comments and video data from a youtube channel playlist_id.




**To do:**
- Improve performane of disagreement classifier;
- Assess and improve performance of political content classifier
- Develop ideology classifier for left-right content
- Develop facebook post/comment scraper
- Collect list of subreddits related to main subreddits indicated in platform_categories.csv
