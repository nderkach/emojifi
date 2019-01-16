
#load gloVe embeddings from file

def read_glove_vecs():
    word_to_index = {}
    index_to_word = {}
    word_to_vec_map = {}
    index = 0
    
    gloveFile = open("./glove.twitter.27B.50d.txt", "r")
    for line in gloveFile:
        items = line.replace('/n', '').split(" ")
        word = items[0]
        embedding = [float(i) for i in items[1:]]
        word_to_index[word] = index
        index_to_word[index] = word
        word_to_vec_map[word] = embedding
        index = index + 1
    
    return (word_to_index, index_to_word, word_to_vec_map)