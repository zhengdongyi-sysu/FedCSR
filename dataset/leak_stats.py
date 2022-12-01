import codecs
import pdb
import random
import copy



def read_item(fname):
    item_number = 0
    with codecs.open(fname,"r",encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            item_number+=1
    print(fname+" has "+str(item_number)+" items!")

def read_user(fname):
    item_number = 0
    with codecs.open(fname,"r",encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            item_number+=1
    print(fname+" has "+str(item_number)+" users!")

def read_data(fname):
    item_number = 0
    mx = 0
    s_user = set()
    s_inter = set()
    s_border = set()
    mx_length = 0
    with codecs.open(fname,"r",encoding="utf-8") as fr:
        for line in fr:
            s_user.add(int(line.strip().split("\t")[0]))
            s_inter.add(int(line.strip().split("\t")[1]))
            line = line.strip().split("\t")[2:]
            s_border.add(int(line[-1].split("|")[0]))
            mx_length = max(mx_length, len(line))
            for w in line:
                w = w.split("|")
                mx = max(mx, int(w[0]))
            item_number+=1

    print(fname + " has max id item: " + str(max(s_border)))
    print(fname + " has " + str(item_number) + " interactions!")
    print(fname+" has user max: "+ str(max(s_user)))
    print(fname + " has inter max: " + str(max(s_user)))
    print(fname + " has border max: " + str(min(s_border)))
    print(fname + " has inter length max: " + str(mx_length))
    print()
    print()


def cmp(elem):
    return elem[0]

def leak_stats(train, valid, test):
    user_all = {}
    cnt = 0
    with codecs.open(train,"r",encoding="utf-8") as fr:
        for line in fr:
            cnt+=1
            user = int(line.strip().split("\t")[0])
            inter_id = int(line.strip().split("\t")[1])
            if user not in user_all:
                user_all[user] = []
            user_all[user].append((inter_id, line))
    print("train interactions :", cnt)


    for user in user_all:
        user_all[user].sort(key=cmp)

    cnt = 0
    no = 0

    with codecs.open(valid,"r",encoding="utf-8") as fr:
        for line in fr:
            cnt += 1
            user = int(line.strip().split("\t")[0])
            inter_id = int(line.strip().split("\t")[1])
            if user in user_all.keys():
                for w in user_all[user]:
                    if w[0] > inter_id:
                        no += 1
                        break
    print("validation interactions: ", cnt)
    print("leak rate:", no / cnt)


    cnt = 0
    no = 0
    with codecs.open(test,"r",encoding="utf-8") as fr:
        for line in fr:
            cnt += 1
            user = int(line.strip().split("\t")[0])
            inter_id = int(line.strip().split("\t")[1])
            if user in user_all.keys():
                for w in user_all[user]:
                    if w[0] > inter_id:
                        no += 1
                        break

    print("test interactions: ", cnt)
    print("leak rate:", no / cnt)





# dataset = "Food-Kitchen"
dataset = "Movie-Book"
# dataset = "Entertainment-Education"

read_item(dataset+"/Alist.txt")
read_item(dataset+"/Blist.txt")
read_user(dataset+"/userlist.txt")


leak_stats(dataset+"/traindata.txt", dataset+"/validdata.txt", dataset+"/testdata.txt")
leak_stats(dataset+"/traindata_new.txt", dataset+"/validdata_new.txt", dataset+"/testdata_new.txt")



"""
Food-Kitchen/Alist.txt has 29207 items!
Food-Kitchen/Blist.txt has 34886 items!

Food-Kitchen/userlist.txt has 16579 users!

Food-Kitchen/traindata.txt has 25766 interactions!
Food-Kitchen/validdata.txt has 7650 interactions!
Food-Kitchen/testdata.txt has 17280 interactions!


64093



Movie-Book/Alist.txt has 36845 items!
Movie-Book/Blist.txt has 63937 items!

Movie-Book/userlist.txt has 15352 users!

Movie-Book/traindata.txt has 44732 interactions!
Movie-Book/validdata.txt has 9274 interactions!
Movie-Book/testdata.txt has 19861 interactions!

100776

Food-Kitchen
34117 8406 8173

Movie-Book
58515 7644 7708




Food-Kitchen
2965
0.5632679738562092
0.1785300925925926

Movie-Book
4561
0.7169506146215225
0.3343235486632093
"""
