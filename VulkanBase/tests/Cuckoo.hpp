#pragma once

#define NULL_LOCATION 0xFFFFFFFFu
#define KEY_EMPTY 0xFFFFFFFFu
#define NOT_FOUND 0xFFFFFFFFu

#define p 334214459
#define a1 100000u
#define b1 200u
#define a2 300000u
#define b2 489902u
#define a3 800000u
#define b3 10248089u
#define a4 9458373u
#define b4 1234838u

static constexpr uint32_t NUM_ITEMS{1 << 16};
static constexpr uint32_t TABLE_SIZE{4 * NUM_ITEMS};

static constexpr uint32_t hash1(uint32_t key) {
    return (a1 ^ key + b1) % p % TABLE_SIZE;
}

static constexpr uint32_t hash2(uint32_t key) {
    return (a2 ^ key + b2) % p % TABLE_SIZE;
}

static constexpr uint32_t hash3(uint32_t key) {
    return (a3 ^ key + b3) % p % TABLE_SIZE;
}

static constexpr uint32_t hash4(uint32_t key) {
    return (a4 ^ key + b4) % p % TABLE_SIZE;
}