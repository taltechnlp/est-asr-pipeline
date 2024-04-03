#! /usr/bin/env python3

import json
import sys

x = json.load(sys.stdin)

for s in x["sections"]: 
  if "turns" in s:
    for t in s["turns"]: 
      print(x["speakers"][t["speaker"]].get("name", t["speaker"]).removeprefix("audio-") + ": " + t["transcript"])
