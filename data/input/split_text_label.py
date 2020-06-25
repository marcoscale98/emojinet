import json

#prendere semeval_train e semeval_test e dividere in
#semeval_train.text e semeval_train.label
#semeval_test.text e semeval_test.label

def get_text(path: str, path_writer : str):
    with open(path, 'r', encoding="utf-8") as reader:
        with open(path_writer,'w',encoding="utf-8") as writer:
            for line in reader:
                tweet = json.loads(line)
                writer.write(tweet['text_no_emoji'])
                writer.write('\n')

def get_labels(path: str, path_writer:str):
    with open(path, 'r', encoding="utf-8") as reader:
        with open(path_writer, 'w',encoding="utf-8") as writer:
            for line in reader:
                label = json.loads(line)
                writer.write(label['ground_truth_label'])
                writer.write('\n')

get_labels('semeval_test', 'semeval_test.labels')
get_text('semeval_test','semeval_test.text')