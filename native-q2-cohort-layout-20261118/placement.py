"""Decode actual CTA placement from separate instrumented layouts and validate complete captures."""
import collections
import re


def read_maps(raw):
    pattern = re.compile(r'^Q2_LAYOUT_MAP fixture=(independent|partial) sample=([01]) major=([01]) x=(\d+) y=(\d+) row=(\d+) local=(\d+) sm=(\d+)$', re.M)
    data = collections.defaultdict(list)
    for name, seq, major, x, y, row, local, sm in pattern.findall(raw):
        data[name,int(seq),int(major)].append(tuple(map(int,(x,y,row,local,sm))))
    if len(data)!=8 or sum(map(len,data.values()))!=384 or raw.count("Q2_LAYOUT_MAP ")!=384:
        raise ValueError('missing or duplicate placement captures')
    result = []
    for (fixture,sample,major), records in sorted(data.items()):
        if len(records)!=48 or len({(v[0],v[1]) for v in records})!=48:
            raise ValueError('incomplete physical CTA grid')
        by_sm = collections.defaultdict(list)
        for x,y,row,local,sm in records:
            if not (0<=x<24 and 0<=y<2):
                raise ValueError('physical block outside fixed 24x2 grid')
            want=(y,x) if major else ((24*y+x)%2,(24*y+x)//2)
            if (row,local)!=want or not 0<=sm<65536:
                raise ValueError('incorrect observed logical task or SM')
            by_sm[sm].append((row,local))
        if len(by_sm)!=24 or any(len(v)!=2 for v in by_sm.values()):
            raise ValueError('witness did not observe two CTAs on each of 24 SMs')
        mixed = sum(pair[0][0]!=pair[1][0] for pair in by_sm.values())
        matched = sum(pair[0][0]!=pair[1][0] and pair[0][1]==pair[1][1] for pair in by_sm.values())
        result.append({'fixture':fixture,'sample':sample,'row_major':bool(major),'mixed_row_sms':mixed,
                       'same_row_sms':24-mixed,'same_local_tile_mixed_row_sms':matched,
                       'columns':['physical_x','physical_y','row','local_tile','smid'],'recorded_ctas':records})
    return result
