
movielines = {}
movieconversations = {};
with open("movie_lines.txt",encoding = "ISO-8859-1") as f:
    for line in f:
        linearray = line.split("+++$+++")
        key       = linearray[0].replace(" ","")
        movielines[key] = linearray[4]


with open("movie_conversations.txt",encoding = "ISO-8859-1") as f:
    for line in f:
        convarray     = line.split("+++$+++")
        conversstring = convarray[3]
        conversstring1 = conversstring.replace("[","")
        conversstring2 = conversstring1.replace("]","")
        conversations  = conversstring2.split(",");
        index=0;
        for i in range(len(conversations)):
            index = i;
            if(index % 2 != 0 ):
                break;
            key   =  conversations[index].replace("'","")
            key = key.replace("\n","")
            key   =  key.replace(" ","")
            key   = key.strip()
            value =  conversations[index+1].replace("'","")
            key = key.replace("\n", "")
            value = value.replace(" ","")
            value = value.strip()
            movieconversations[movielines[key]] = movielines[value]

with open("movie_dialogues_corpus.txt",'w') as f:
    for x in movieconversations:
        f.write(x+"+++$+++"+movieconversations[x]+"\n")








