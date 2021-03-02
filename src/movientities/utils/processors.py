import os
from typing import List

class InputInstance(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

class NerPreProcessor:

    __jobs__ = ["train", "dev", "test"]

    @classmethod
    def __readfile(cls, filepath:str):
        """Reads txt file"""
        data = []
        label = []
        sentence = []
        with open(filepath) as f:
            for line in f.readlines():
                if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == "\n":
                    if len(sentence) > 0:
                        data.append((sentence, label))
                        label = []
                        sentence = []
                    continue
                splited = line.split(" ")
                sentence.append(splited[0])
                label.append(splited[-1][:-1])

        if len(sentence) > 0:
            data.append((sentence,label))
            sentence = []
            label = []
        return data

    @classmethod
    def _create_instances(cls, lines:List, job_type:str):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (job_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputInstance(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

    @classmethod
    def get_instances(cls, data_dir:str, job_type:str):
        assert job_type in NerPreProcessor.__jobs__, f"The jobtype is wrong must be one of -> {NerPreProcessor.__jobs__}"
        return NerPreProcessor._create_instances(NerPreProcessor.__readfile(os.path.join(data_dir, job_type + ".txt")), job_type)

    @classmethod
    def convert_instances_to_features(cls, examples, label_list, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        label_map = {label : i for i, label in enumerate(label_list,1)}

        features = []
        for (ex_index,example) in enumerate(examples):
            textlist = example.text_a.split(' ')
            labellist = example.label
            tokens = []
            labels = []
            valid = []
            label_mask = []
            for i, word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                        valid.append(1)
                        label_mask.append(1)
                    else:
                        valid.append(0)
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                valid = valid[0:(max_seq_length - 2)]
                label_mask = label_mask[0:(max_seq_length - 2)]
            ntokens = []
            segment_ids = []
            label_ids = []
            ntokens.append("[CLS]")
            segment_ids.append(0)
            valid.insert(0,1)
            label_mask.insert(0,1)
            label_ids.append(label_map["[CLS]"])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    label_ids.append(label_map[labels[i]])
            ntokens.append("[SEP]")
            segment_ids.append(0)
            valid.append(1)
            label_mask.append(1)
            label_ids.append(label_map["[SEP]"])
            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            label_mask = [1] * len(label_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                valid.append(1)
                label_mask.append(0)
            while len(label_ids) < max_seq_length:
                label_ids.append(0)
                label_mask.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length

            if ex_index < 5:
                print("*** Example ***")
                print("guid: %s" % (example.guid))
                print("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                print(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                # logger.info("label: %s (id = %d)" % (example.label, label_ids))

            features.append(
                    InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=label_ids,
                                valid_ids=valid,
                                label_mask=label_mask))
        return features