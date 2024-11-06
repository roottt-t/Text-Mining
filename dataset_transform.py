from datasets import Dataset, DatasetDict, ClassLabel, Sequence, Value, Features

# Define mappings for POS, chunk, and NER tags (example mappings provided)
pos_tag_to_id = {
    '"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9,
    'CC': 10, 'CD': 11, 'DT': 12, 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17,
    'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23, 'NNS': 24, 
    'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31,
    'RBS': 32, 'RP': 33, 'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39,
    'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43, 'WP': 44, 'WP$': 45, 'WRB': 46
}

chunk_tag_to_id = {
    'O': 0, 'B-ADJP': 1, 'I-ADJP': 2, 'B-ADVP': 3, 'I-ADVP': 4, 'B-CONJP': 5, 'I-CONJP': 6,
    'B-INTJ': 7, 'I-INTJ': 8, 'B-LST': 9, 'I-LST': 10, 'B-NP': 11, 'I-NP': 12, 'B-PP': 13,
    'I-PP': 14, 'B-PRT': 15, 'I-PRT': 16, 'B-SBAR': 17, 'I-SBAR': 18, 'B-UCP': 19, 
    'I-UCP': 20, 'B-VP': 21, 'I-VP': 22
}

ner_tag_to_id = {
    'O': 0, 'B-ART': 1, 'I-ART': 2, 'B-PER': 3, 'I-PER': 4, 'B-MAT': 5, 'I-MAT': 6,
    'B-LOC': 7, 'I-LOC': 8, 'B-CON' : 9, 'I-CON': 10, 'B-SPE': 11, 'I-SPE': 12
}

# Parsing function
def parse_bio_file(file_path):
    sentence_id = 0
    datas = []
    
    with open(file_path, 'r') as file:
        tokens, pos_tags, ner_tags = [], [], []
        data = {'id': '', 'tokens': [], 'pos_tags': [], 'chunk_tags': [], 'ner_tags': []}
        for line in file:
            # print(line)
            
            # Check if it's a new sentence
            if line.strip() == "":
                # print (tokens)
                if tokens:
                    data['id']=str(sentence_id)
                    data['tokens']=tokens
                    data['pos_tags']=[pos_tag_to_id[tag] for tag in pos_tags]
                    data['ner_tags']=[ner_tag_to_id[tag] for tag in ner_tags]  # Use 'O' for NER if absent
                    data['chunk_tags']=[0]*len(tokens)  # Set chunk tags to 'O' by default
                    datas.append(data)
                    tokens, pos_tags, ner_tags = [], [], []
                    data = {'id': '', 'tokens': [], 'pos_tags': [], 'chunk_tags': [], 'ner_tags': []}
                    sentence_id += 1
                
            else:
                token, pos_tag, ner_tag = line.strip().split()
                tokens.append(token)
                pos_tags.append(pos_tag)
                ner_tags.append(ner_tag)
                

        # # Add the last sentence if any
        if tokens:
            data['id']=str(sentence_id)
            data['tokens']=tokens
            data['pos_tags']=[pos_tag_to_id[tag] for tag in pos_tags]
            data['ner_tags']=[ner_tag_to_id[tag] for tag in ner_tags]  # Use 'O' for NER if absent
            data['chunk_tags']=[0]*len(tokens)  # Set chunk tags to 'O' by default
            datas.append(data)
    return datas

# Load and parse the file
parsed_train_data = parse_bio_file('train.txt')
parsed_test_data = parse_bio_file('test.txt')
parsed_val_data = parse_bio_file('val.txt')

# Define features as in CoNLL2003
features = Features({
    'id': Value(dtype='string'),
    'tokens': Sequence(feature=Value(dtype='string')),
    'pos_tags': Sequence(feature=ClassLabel(names=list(pos_tag_to_id.keys()))),
    'chunk_tags': Sequence(feature=ClassLabel(names=list(chunk_tag_to_id.keys()))),
    'ner_tags': Sequence(feature=ClassLabel(names=list(ner_tag_to_id.keys())))
})


# Convert to Dataset
train_dataset = Dataset.from_list(parsed_train_data)
train_dataset = train_dataset.cast(features)

test_dataset = Dataset.from_list(parsed_test_data)
test_dataset = test_dataset.cast(features)

val_dataset = Dataset.from_list(parsed_val_data)
val_dataset = val_dataset.cast(features)


# Create the DatasetDict
raw_datasets = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
    'validation': val_dataset
})

# Display the structure
print(raw_datasets)
print(raw_datasets["train"][0])  # Display the first example
print(raw_datasets["train"].features)  # Display feature information
# print(raw_datasets["train"].features)  # Display feature information
