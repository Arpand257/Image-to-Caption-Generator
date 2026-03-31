import pandas as pd

data = []

with open("Flickr8k.token.txt", "r") as f:
    for line in f:
        img, caption = line.strip().split("\t")
        img = img.split("#")[0]
        data.append([img, caption])

df = pd.DataFrame(data, columns=["image", "caption"])
df.to_csv("captions.csv", index=False)