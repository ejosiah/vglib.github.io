/**
*  Cockoo hashing
* described in GPU Computing Gems, Chapter 4. Building an Efficient Hash Table on the GPU
* page 39
*/

#ifndef HASH_TABLE_GLSL
#define HASH_TABLE_GLSL

#define NULL_LOCATION 0xFFFFFFFFu
#define KEY_EMPTY 0xFFFFFFFFu
#define NOT_FOUND 0xFFFFFFFFu
#define p 334214459u
#define a1 100000u
#define b1 200u
#define a2 300000u
#define b2 489902u
#define a3 800000u
#define b3 10248089u
#define a4 9458373u
#define b4 1234838u
#define KEY 0
#define VALUE 1
#define TABLE_SIZE tableSize

#define hash1(key) ((a1 ^ key + b1) % p % TABLE_SIZE)
#define hash2(key) ((a2 ^ key + b2) % p % TABLE_SIZE)
#define hash3(key) ((a3 ^ key + b3) % p % TABLE_SIZE)
#define hash4(key) ((a4 ^ key + b4) % p % TABLE_SIZE)

#define get_entry(loc) uvec2(table[KEY].data[loc], table[VALUE].data[loc])
#define atmoic_exchage(loc, entry)  uvec2( \
    atomicExchange(table[KEY].data[loc], entry.x), \
    atomicExchange(table[VALUE].data[loc], entry.y))

#define get_key(entry) entry.x
#define get_value(entry) entry.y

layout(local_size_x = 1024) in;

layout(set = 0, binding = 0) buffer TableSsbo {
    uint data[];
} table[2];

layout(set = 0, binding = 1) buffer InsertState {
    uint state[];
};

layout(set = 0, binding = 2) buffer TableInsertLoc {
    uint locations[];
};

layout(set = 0, binding = 3) buffer KeySsbo {
    uint keys[];
};

layout(set = 0, binding = 4) buffer ValueSSbo {
    uint values[];
};

layout(set = 0, binding = 5) buffer QueryResultSSbo {
    uint query_results[];
};

layout(push_constant) uniform Constants {
    uint tableSize;
    uint numItems;
};


bool hash_table_insert_internal(bool prevInsertSucceeded, inout uvec2 entry, inout uint location) {
    if(prevInsertSucceeded) return true;
    uint key = get_key(entry);
    location =  location == NULL_LOCATION ? hash1(key) : location;

    entry = atmoic_exchage(location, entry);
    key = get_key(entry);
    if (key == KEY_EMPTY) {
        location = NULL_LOCATION;
        return true;
    }

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

void hash_table_insert(uint gid) {

    uvec2 entry = uvec2(keys[gid], values[gid]);
    uint location = locations[gid];
    bool prevInsertStatus = bool(state[gid]);
    bool inserted = hash_table_insert_internal(prevInsertStatus, entry, location);

    if(!inserted) {
        keys[gid] = entry.x;
        values[gid] = entry.y;
    }

    state[gid] = uint(inserted);
    locations[gid] = location;
}

uvec2 find(uint key) {
    uint location1 = hash1(key);
    uint location2 = hash2(key);
    uint location3 = hash3(key);
    uint location4 = hash4(key);

    uvec2 entry = get_entry(location1);
    if(get_key(entry) != key) {
        entry = get_entry(location2);

        if(get_key(entry) != key) {
            entry = get_entry(location3);

            if(get_key(entry) != key) {
                entry = get_entry(location4);

                if(get_key(entry) != key) {
                    return uvec2(0, NOT_FOUND);
                }
            }
        }
    }
    return entry;
}

uint hash_table_query(uint gid) {
    if (gid >= numItems) return NOT_FOUND;

    uint key = keys[gid];
    uvec2 entry = find(key);

    return get_value(entry);
}
#endif // HASH_TABLE_GLSL