# NHL Shot Scraper/Plotter

This repo was used to generate the data and plots for my medium article, ["What would a hockey 2-point line look like?"](https://medium.com/@BlakeAtkinson/what-would-a-hockey-2-point-line-look-like-bf6b3192226a).

#### Python Libraries used

1. pickle
2. tqdm
3. matplotlib
4. scipy

#### Instructions

If you'd like to download the data yourself, follow these steps:

1. Run game_https.py to generate all the regular season API end point URLs for seasons 2014 to 2019. You can play with the seasons collected. Older seasons have less data available. The output is a pickle file.

2. Running scrape_shots.py will scrape the shot type, X-coordinate, and Y-coordinate for all the API end point URLs generated. The output is also a pickle file.

3. Once you have a shot_locations.pkl file, you're ready to play! I have two files that will plot different things. plot.py has functions that will plot hexbins, distance from goal, and scatter plot heatmaps. plot_contour.py will plot contour graphs.

NOTE! There are a couple of items that you have to run once in order for the functions to work. In plot.py, there's a distance from goal calculation that can run. If you're not interested in distance from goal, don't worry about it.

In plot_contour.py, I had to create rectangular inputs for matplotlib's contour plots. Run generate_shot_input() and generate_goal_input() once to generate pickle files that are rectangular in shape.

Lastly, this repo was only created to share the code that I wrote to plot graphs for the article. It's unlikely that I will update or maintain it.

Also, thank you National Hockey League for making this data available!




<!-- end -->
