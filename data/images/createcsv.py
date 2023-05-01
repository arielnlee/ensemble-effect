# open the prompts.txt file, read lines
# create a csv file with the prompts in the first column and path to the image in the second column
# the path is just the index of the prompt in the prompts.txt file, in the images folder, e.g. images/0.jpg

import csv

with open('prompts.txt', 'r') as f:
    lines = f.readlines()
    with open('diffusion_prompts.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i, line in enumerate(lines):
            if i < 20000:
                writer.writerow([line.strip(), 'images/{}.png'.format(i)])
            if i >= 30000 and i < 50000:
                writer.writerow([line.strip(), 'images/{}.png'.format(i)])
            if i >= 60000:
                writer.writerow([line.strip(), 'images/{}.png'.format(i)])
