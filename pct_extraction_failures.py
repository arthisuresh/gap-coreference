import spacy
import pandas as pd 

if __name__ == '__main__':
    test_set = pd.read_csv('gap-test.tsv', sep='\t')
    nlp = spacy.load('en_core_web_lg')
    total_a_or_b = 0
    total_extraction_failures = 0
    for i, row in test_set.iterrows():
        mention_a = row['A']
        mention_b = row['B']
        text = row['Text']
        processed_text = nlp(text)
        ents = [ent.text.lower() for ent in processed_text.ents]
        if row['A-coref']:
            total_a_or_b += 1
            if mention_a.lower() not in ents:
                total_extraction_failures += 1
        elif row['B-coref']:
            total_a_or_b += 1
            if mention_b.lower() not in ents:
                total_extraction_failures += 1
    print(100.0*total_extraction_failures/total_a_or_b)
    print(100.0*total_a_or_b/len(test_set))