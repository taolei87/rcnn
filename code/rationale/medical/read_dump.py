
import sys
import os
import json

with open(sys.argv[1]) as fin:
    for line in fin:
        if not line.strip(): continue
        x = json.loads(line)
        r, t = x["rationale"], x["text"]
        trimed = t.lstrip('_ ')
        removed = len(t)-len(trimed)
        r, t = r[removed:].split(), t[removed:].split()
        assert len(r) == len(t)
        print "Ground Truth:\t{}".format(int(x["y"][0]))
        print "Prediction:\t{}".format(x["p"][0])
        print "------------"
        print "Full Report:"
        print "\t"+" ".join(t)
        print "------------"
        print "Rationale:"
        print "\t"+" ".join(r)
        print ""
        print ""
