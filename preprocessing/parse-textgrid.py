import textgrid
import re
import pandas as pd
import os

textgrid_files = []

#procura na pasta catna-textgrid
for root, _, files in os.walk('catna-textgrid'):
    for fname in files:
        if fname.lower().endswith('.textgrid'):
            textgrid_files.append(os.path.join(root, fname))
textgrid_files.sort()
print(f"Found {len(textgrid_files)} TextGrid file(s) in 'catna-textgrid'.")
print("TextGrid files:", textgrid_files)

df = pd.DataFrame(columns=["File", "Speaker", "Start", "End", "Duration", "Text"])

for file in textgrid_files:
    tg = textgrid.TextGrid.fromFile(file)

    #catna
    speaker_regex = [r'(?i)^TB-L\d+$', r'(?i)^TB-L\d+-2$', r'(?i)^TB-loc$', r'(?i)^TB-oloc$', r'(?i)^TB-1\d+$']
    documenter_regex = r'(?i)^TB-Doc\d+$'

    #cm
    #speaker_regex = [r'(?i)^TB-L\d+-normal$']
    #documenter_regex = r'(?i)^TB-Doc\d+-normal$'

    speaker_list = [name for name in tg.getNames() if any(re.match(pattern, name) for pattern in speaker_regex)]
    documenter_list = [name for name in tg.getNames() if re.match(documenter_regex, name, re.IGNORECASE)]

    print(f"Processing file: {file}")
    print("Tiers: ", end="")
    print(", ".join(tg.getNames()))

    for tier in tg:
        if any(re.match(pattern, tier.name) for pattern in speaker_regex):
            print("Speaker:", tier.name)
        elif re.match(documenter_regex, tier.name):
            print("Documenter:", tier.name)
        else:
            continue

        row_list = []

        for interval in tier:
            if not interval.mark.strip():
                continue
            #if does not contain alphanumeric characters
            if not re.search(r'\w', interval.mark):
                continue
            row_list.append({
                "File": os.path.basename(file),
                "Speaker": tier.name,
                "Start": interval.minTime,
                "End": interval.maxTime,
                "Duration": interval.maxTime - interval.minTime,
                "Text": interval.mark
            })

        df = pd.concat([df, pd.DataFrame(row_list)], ignore_index=True)

    

df.to_csv(f'output_catna.csv', index=False)