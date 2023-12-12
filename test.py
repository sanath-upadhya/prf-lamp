import json

f = open('user_based/train_questions.json')

data = json.load(f)

print(data[0].keys())

lowest_profile = 100000000
lowest_id = -1
for ind_data in data:
    if (len(ind_data['profile']) < lowest_profile):
        print("Entering here")
        lowest_profile = len(ind_data['profile'])
        lowest_id = ind_data['id']


print("Lowest value of profile is " + str(lowest_profile))
print("Lowest ID is " + str(lowest_id))


# for i in data['task']:
#     print(i)

f.close()
