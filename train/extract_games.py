# Packages
import os

# Directories
GAMES = 'games'

# Game count
count = 0

# Create dataset file
with open('games.js', 'w') as f: f.write('games = [\n')

# Process SGF files
for year in os.listdir(GAMES):
  year_path = './' + GAMES + '/' + year
  parse_year = int(year)
  if os.path.isdir(year_path) and parse_year == 2017:
    for dir in os.listdir(year_path):
      game_dir = year_path + '/' + dir
      if os.path.isdir(game_dir):
        for game in os.listdir(game_dir):
          if '.sgf' in game:
            game_path = game_dir + '/' + game
            with open(game_path) as f:
              try:
                sgf = f.read()
                moves = "'" + ''.join(sgf.split(']\n\n')[1].split('\n'))[:-1].split('C')[0] + "',"
                with open('games.js', 'a') as g: g.write(moves + '\n')
                count += 1
              except: pass

# Seal datates
with open('games.js', 'a') as g: g.write('];\n\nmodule.exports = games;')

# Print
print(f'Extracted {count} games to "games.js"')
