import json
from tqdm import tqdm

years = {"2019", "2020", "2021"}

r = open("./yelp_dataset/review.json", "r")
w = open("./yelp_dataset/review_sample.json", "w+")

for line in tqdm(r):
	review = json.loads(line)
	year = review["date"][:4]
	if year in years:
		w.write(json.dumps(review))
		w.write("\n")

r.close()
w.close()

print("Verifying sample was generated properly...")
f = open("./yelp_dataset/review_sample.json", "r")

for line in tqdm(f):
	review = json.loads(line)
	year = review["date"][:4]
	if year not in years:
		print(f"The following review should not be in the sample:\n{review}")
