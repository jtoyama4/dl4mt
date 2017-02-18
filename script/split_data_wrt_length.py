#!/usr/bin/env python

def get_test_source_sentences():
    lens = []
    sources = []
    with open("/share/data/flickr30k/test.en","r") as f:
        for l in f.readlines():
            s = l.strip()
            lens.append(len(s.split(' ')))
            sources.append(s)
    return lens, sources


def main(test_file, out_prefix):
    lens, sources= get_test_source_sentences()
    outs = {}
    outs["5"]  = []
    outs["10"] = []
    outs["15"] = []
    outs["20"] = []
    outs["25"] = []
    outs["30"] = []

    with open(test_file) as f:
        for l, source_length in zip(f.readlines(), lens):
            if source_length <= 5:
                outs["5"].append(l)
            elif 5 < source_length <= 10:
                outs["10"].append(l)
            elif 10 < source_length <= 15:
                outs["15"].append(l)
            elif 15 < source_length <= 20:
                outs["20"].append(l)
            elif 20 < source_length <= 25:
                outs["25"].append(l)
            elif 25 < source_length <= 30:
                outs["30"].append(l)
            else:
                pass

    for i in ["5","10","15","20","25","30"]:
        #print(i, len(outs[i]))
        with open("{}.{}.txt".format(out_prefix,i),"w") as f:
            for l in outs[i]:
                f.write("{}".format(l))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="split data wrt source length")
    parser.add_argument("file", help="target file")
    parser.add_argument("out_prefix", help="output file prefix")

    args = parser.parse_args()
    main(args.file, args.out_prefix)

