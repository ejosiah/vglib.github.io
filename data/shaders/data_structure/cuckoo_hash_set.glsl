#ifndef CUCK_KOO_HASH_SET
#define CUCK_KOO_HASH_SET

#include "cuckoo_hash_common.glsl"


layout(local_size_x = 1024) in;

layout(set = 0, binding = 0) buffer TableSsbo {
    uint table[];
};

layout(set = 0, binding = 1) buffer InsertState {
    uint state[];
};

layout(set = 0, binding = 2) buffer TableInsertLoc {
    uint locations[];
};

layout(set = 0, binding = 3) buffer KeySsbo {
    uint keys[];
};

layout(set = 0, binding = 4) buffer QueryResultSSbo {
    uint query_results[];
};

layout(push_constant) uniform Constants {
    uint tableSize;
    uint numItems;
};

uint get_entry(uint location) {
    return table[location];
}

bool hash_table_set_internal(bool prevInsertSucceeded, inout uint key, inout uint location) {
    if(prevInsertSucceeded) return true;
    location =  location == NULL_LOCATION ? hash1(key) : location;

    uint newKey = atomicExchange(table[location], key);
    if (newKey == KEY_EMPTY || newKey == key) {
        location = NULL_LOCATION;
        return true;
    }
    key = newKey;

    // item evicted. figure out where to reinsert this entry
    uint location1 = hash1(key);
    uint location2 = hash2(key);
    uint location3 = hash3(key);
    uint location4 = hash4(key);

    if( location == location1) location = location2;
    else if (location == location2) location = location3;
    else if (location == location3) location = location4;
    else location = location1;

    return false;
}

void hash_set_insert(uint gid) {
    if (gid >= numItems) return;

    uint key = keys[gid];
    uint location = locations[gid];
    bool prevInsertStatus = bool(state[gid]);
    bool inserted = hash_table_set_internal(prevInsertStatus, key, location);

    if(!inserted) {
        keys[gid] = key;
    }

    state[gid] = uint(inserted);
    locations[gid] = location;

}

void remove_duplicate_entries(uint gid) {
    if (gid >= numItems) return;

    uint key = keys[gid];

    if (key == KEY_EMPTY) return;

    uint location[4];
    location[0] = hash1(key);
    location[1] = hash2(key);
    location[2] = hash3(key);
    location[3] = hash4(key);

    uint entries[4];
    entries[0] =  table[location[0]];
    entries[1] =  table[location[1]];
    entries[2] =  table[location[2]];
    entries[3] =  table[location[3]];

    for(int i = 0; i < 4; ++i) {
        if(entries[i] == key) {
            for(int j = 0; j < 4; ++j) {
                if(i != j) {
                    if(entries[j] == key) {
                        table[location[j]] = KEY_EMPTY;
                    }
                }
            }
            break;
        }
    }

}

uint find(uint key) {
    uint location1 = hash1(key);
    uint location2 = hash2(key);
    uint location3 = hash3(key);
    uint location4 = hash4(key);

    uint entry = get_entry(location1);
    if(entry != key) {
        entry = get_entry(location2);

        if(entry != key) {
            entry = get_entry(location3);

            if(entry != key) {
                entry = get_entry(location4);

                if(entry != key) {
                    return NOT_FOUND;
                }
            }
        }
    }
    return entry;
}

uint hash_set_find(uint gid) {
    if (gid >= numItems) return NOT_FOUND;

    uint key = keys[gid];
    return find(key);
}



#endif // CUCK_KOO_HASH_SET