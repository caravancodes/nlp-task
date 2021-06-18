import nltk

nltk.download('wordnet')
nltk.download('punkt')

word_1 = 'pintar'
word_2 = 'makan'
word_3 = 'cinta'

gloss = 'gloss'
example = 'example'
arti_kata = 'arti kata'

sentences_1_first = "rupanya pencuri itu lebih pintar daripada polisi"
sentences_1_second = "mereka sudah pintar membuat baju sendiri"

sentences_2_first = "pembangunan jembatan ini makan waktu lama"
sentences_2_second = "upacara adat itu makan ongkos besar"

sentences_3_first = "orang tuaku cinta kepada kami semua"
sentences_3_second = "cinta kepada sesama makhluk"

word_ambigu = {
    word_1: [
        {gloss: arti_kata, example: sentences_1_first},
        {gloss: arti_kata, example: sentences_1_second}],
    word_2: [
        {gloss: arti_kata, example: sentences_2_first},
        {gloss: arti_kata, example: sentences_2_second}],
    word_3: [
        {gloss: arti_kata, example: sentences_3_first},
        {gloss: arti_kata, example: sentences_3_second}]}

length_dict = {key: len(value) for key, value in word_ambigu.items()}


def lesk(kata, sentence):
    length_dict = {key: len(value) for key, value in word_ambigu.items()}
    length_key = length_dict[kata]
    best_sense = word_ambigu[kata][0]
    max_overlap = 0
    for i in range(length_key):
        overlap = 0
        for each in nltk.word_tokenize(sentence):
            overlap += sum([1 for d in word_ambigu[kata][i].values() if each in d])
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = word_ambigu[kata][i]
    return best_sense


print(lesk(word_1, sentences_1_first))
print(lesk(word_1, sentences_1_second))

print(lesk(word_2, sentences_2_first))
print(lesk(word_2, sentences_2_second))

print(lesk(word_3, sentences_3_first))
print(lesk(word_3, sentences_3_second))
