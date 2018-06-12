import csv
import json

with open("./textme_review.json", 'r') as f:
    reviews = json.load(f)

f = csv.writer(open("textme_review.csv", "wb+"))

# Write CSV Header, If you dont need that, remove this line
keys = list(reviews[0].keys())
f.writerow(keys)

for review in reviews:
    row = []
    try:
        for key in keys:
            if type(review[key]) == str:
                row.append(u''.join((review[key],)).encode('utf-8').strip())
            else:
                row.append(review[key])
        f.writerow(row)
    except:
        continue
