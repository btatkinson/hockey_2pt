import pickle

start_season = 2014
end_season = 2019

game_types = {
    'preseason':False,
    'reg_season':True,
    'postseason':False,
    'all_star':False
}

# make year list of all years needed
# year is the year the season BEGAN
# 2017 for 2017-2018 season, etc.
years = list(range(start_season, (end_season)))

print("Years to be collected: ", years)

urls = []
for year in years:
    if(game_types['preseason']):
        # not implemented
        pass
    if(game_types['reg_season']):
        # yup
        if year >= 2017:
            gms = 1271
        else:
            gms = 1230

        for gm in range(gms):
            game_num = "0000" + str((gm+1))
            game_num = game_num[-4:]
            game_id = str(year)+"02"+game_num
            url = "https://statsapi.web.nhl.com/api/v1/game/"+game_id+"/feed/live"
            urls.append(url)
    if(game_types['postseason']):
        # not implemented
        pass
    if(game_types['all_star']):
        # not implemented
        pass

print("You've added " + str(len(urls)) + " game log urls to the output file.")
print("Writing a list of urls in a pickle file...")

with open('game_urls', 'wb') as fp:
    pickle.dump(urls, fp)

print("File outputted.")




# end
