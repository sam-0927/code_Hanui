def make_id(idx):
    mrd = idx // 25 + 1           # 1~3
    band = (idx % 25) // 5 + 1    # 1~5
    layer = (idx % 5) + 1         # 1~5
    return 'MRD '+str(mrd)+', '+str(band)+' band, '+str(layer)+''

    # if idx < 25:  # MRD 1
    #     mrd = 1
    #     x = idx
    # elif idx < 50:  # MRD 2
    #     mrd = 2
    #     x = idx - 25
    # else:  # MRD 3
    #     mrd = 3
    #     x = idx - 50

    # band = x // 5 + 1
    # layer = x % 5 + 1

    # return f"MRD {mrd}, {band} band, {layer} layer"
    
from collections import defaultdict
dict_ = defaultdict(int)
# dict_band = defaultdict(int)
# band_layer_dict = defaultdict(lambda: defaultdict(int))
# 3중 dict: mrd -> band -> layer -> count
mrd_band_layer_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
newf = open('sort_name.txt','w')
with open('sort.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        data = line.strip()
        if data.isdigit():
            name = make_id(int(data))
            dict_[name] += 1

            # 파싱
            parts = name.split(",")
            mrd   = parts[0].strip().split()[1]   # "MRD 2" → "2"
            band  = parts[1].strip().split()[0]   # "5 band" → "5"
            layer = parts[2].strip().split()[0]   # "3 layer" → "3"
            
            mrd_band_layer_dict[mrd][band][layer] += 1

            newf.write(name+'\n')
        else:
            newf.write(line)

    newf.write('\n=========================\n')
    newf.write('--- MRD별 Band/Layer 카운트 ---\n')
    for mrd in sorted(mrd_band_layer_dict.keys(), key=int):
        newf.write(f"MRD {mrd}:\n")
        print(f"MRD {mrd}:")
        for band in sorted(mrd_band_layer_dict[mrd].keys(), key=int):
            newf.write(f"  Band {band}:\n")
            print(f"  Band {band}:")
            for layer in sorted(mrd_band_layer_dict[mrd][band].keys(), key=int):
                count = mrd_band_layer_dict[mrd][band][layer]
                newf.write(f"    Layer {layer} : {count}\n")
                print(f"    Layer {layer} : {count}")

newf.write('\n=========================\n')
dict_ = dict(sorted(dict_.items(), key=lambda item: item[1]))
for k, v in dict_.items():
    print(k+' : '+str(v)+'\n')
    newf.write(k+' : '+str(v)+'\n')
# breakpoint()