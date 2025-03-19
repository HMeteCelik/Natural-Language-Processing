def initialVocabulary():
    
    return list("abcçdefgğhıijklmnoöprsştuüvyzwxq"+
                "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZWXQ"+
                "0123456789"+" "+
                "!'^#+$%&/{([)]=}*?\\_-<>|.:´,;`@€¨~\"é")


def merge_bigrams(sorted_merges,corp,merges):

    keys = list(sorted_merges.keys())[0]
    count = sorted_merges.get(keys)
    copy_corp = corp.copy()
    merges[keys] = count

    for word in range(len(copy_corp)):
        j = 0
        while j < len(copy_corp[word]) - 1:
            key = (copy_corp[word][j], copy_corp[word][j+1])
            if keys == key:  
                corp[word][j] = key[0] + key[1]  
                corp[word].pop(j+1)  
            else:
                j += 1 
    
    return corp, merges


def search_bigrams(corpus):
    
    bigrams = dict()
    for i in range(len(corpus)):  
        for j in range(len(corpus[i]) - 1):  
            key = (corpus[i][j], corpus[i][j+1])
            if key in bigrams:
                bigrams[key] += 1
            else:
                bigrams[key] = 1
    sorted_bigrams = dict(sorted(bigrams.items(), key=lambda item: (-item[1],item[0]),reverse=False))

    return sorted_bigrams


def bpeCorpus(corp, maxMergeCount):  

    Vocabulary = initialVocabulary()

    corp = [list(word) for word in corp]
    merges = dict()

    count = 0

    while count < maxMergeCount :a
        sorted_merges = search_bigrams(corp)
        if len(sorted_merges) == 0:
            break
        corp, merges = merge_bigrams(sorted_merges, corp, merges)
        count += 1
    tokenized_corpus = corp
    for key in merges.keys():
        keys = key[0]+key[1]
        Vocabulary.append(keys)
    new_merges = []
    for key in merges.keys():
        new_merges.append((key,merges.get(key)))
    
    return (new_merges, Vocabulary, tokenized_corpus)


def bpeFN(fileName, maxMergeCount=10):

    with open(fileName,"r") as file:
        message = file.read()
    message = " ".join(line for line in message.splitlines() if line.strip())
    corp = message.split(" ")

    corp = [f" {word}_" for word in corp]
    
    return bpeCorpus(corp,maxMergeCount)


def bpeTokenize(text, merges):

    words = text.split()
    tokenized_words = []
    
    for word in words:
        tokens = list(word) + ['_']  
        tokens = [' '] + tokens  
        
        while True:
            pair_frequencies = {(tokens[i], tokens[i+1]): i for i in range(len(tokens) - 1)}
            merge_found = False
            
            for merge in merges:
                if merge[0] in pair_frequencies:
                    index = pair_frequencies[merge[0]]
                    tokens = tokens[:index] + ["".join(merge[0])] + tokens[index+2:]
                    merge_found = True
                    break
            
            if not merge_found:
                break
        
        tokenized_words.append(tokens)
    
    return tokenized_words


def bpeFNToFile(infn, maxMergeCount=10, outfn="output.txt"):
        
    (Merges,Vocabulary,TokenizedCorpus)=bpeFN(infn, maxMergeCount)
    outfile = open(outfn,"w",encoding='utf-8')
    outfile.write("Merges:\n")
    outfile.write(str(Merges))
    outfile.write("\n\nVocabulary:\n")
    outfile.write(str(Vocabulary))
    outfile.write("\n\nTokenizedCorpus:\n")
    outfile.write(str(TokenizedCorpus))
    outfile.close()
