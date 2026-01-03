/**
 * BLAKE3 NIF - Direct wrapper around official BLAKE3
 *
 * No extra layers - just BLAKE3 -> Erlang NIF
 */

#include <string.h>
#include <stdint.h>

#ifdef __GNUC__
  #define _SAVED_GNUC_ __GNUC__
  #undef __GNUC__
#endif

#include <erl_nif.h>

#ifdef _SAVED_GNUC_
  #define __GNUC__ _SAVED_GNUC_
  #undef _SAVED_GNUC_
#endif

#include "fp_lib/vendor/blake3.h"

/* Resource type for incremental hasher */
static ErlNifResourceType* BLAKE3_HASHER_TYPE = NULL;

static void hasher_dtor(ErlNifEnv* env, void* obj) {
    (void)env; (void)obj;
}

/* hash(binary) -> binary */
static ERL_NIF_TERM nif_hash(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary input;
    if (!enif_inspect_binary(env, argv[0], &input)) {
        return enif_make_badarg(env);
    }

    ErlNifBinary output;
    if (!enif_alloc_binary(BLAKE3_OUT_LEN, &output)) {
        return enif_make_badarg(env);
    }

    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, input.data, input.size);
    blake3_hasher_finalize(&hasher, output.data, BLAKE3_OUT_LEN);

    return enif_make_binary(env, &output);
}

/* hash_keyed(key, data) -> binary */
static ERL_NIF_TERM nif_hash_keyed(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary key, input;

    if (!enif_inspect_binary(env, argv[0], &key) || key.size != BLAKE3_KEY_LEN) {
        return enif_make_badarg(env);
    }
    if (!enif_inspect_binary(env, argv[1], &input)) {
        return enif_make_badarg(env);
    }

    ErlNifBinary output;
    if (!enif_alloc_binary(BLAKE3_OUT_LEN, &output)) {
        return enif_make_badarg(env);
    }

    blake3_hasher hasher;
    blake3_hasher_init_keyed(&hasher, key.data);
    blake3_hasher_update(&hasher, input.data, input.size);
    blake3_hasher_finalize(&hasher, output.data, BLAKE3_OUT_LEN);

    return enif_make_binary(env, &output);
}

/* derive_key(context, key_material) -> binary */
static ERL_NIF_TERM nif_derive_key(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary context, key_material;

    if (!enif_inspect_binary(env, argv[0], &context)) {
        return enif_make_badarg(env);
    }
    if (!enif_inspect_binary(env, argv[1], &key_material)) {
        return enif_make_badarg(env);
    }

    ErlNifBinary output;
    if (!enif_alloc_binary(BLAKE3_OUT_LEN, &output)) {
        return enif_make_badarg(env);
    }

    blake3_hasher hasher;
    blake3_hasher_init_derive_key_raw(&hasher, context.data, context.size);
    blake3_hasher_update(&hasher, key_material.data, key_material.size);
    blake3_hasher_finalize(&hasher, output.data, BLAKE3_OUT_LEN);

    return enif_make_binary(env, &output);
}

/* to_hex(hash) -> binary */
static ERL_NIF_TERM nif_to_hex(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary hash;
    if (!enif_inspect_binary(env, argv[0], &hash) || hash.size != BLAKE3_OUT_LEN) {
        return enif_make_badarg(env);
    }

    ErlNifBinary output;
    if (!enif_alloc_binary(64, &output)) {
        return enif_make_badarg(env);
    }

    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        output.data[i*2] = hex[(hash.data[i] >> 4) & 0xF];
        output.data[i*2+1] = hex[hash.data[i] & 0xF];
    }

    return enif_make_binary(env, &output);
}

/* compare(a, b) -> boolean */
static ERL_NIF_TERM nif_compare(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary a, b;

    if (!enif_inspect_binary(env, argv[0], &a) || a.size != BLAKE3_OUT_LEN) {
        return enif_make_badarg(env);
    }
    if (!enif_inspect_binary(env, argv[1], &b) || b.size != BLAKE3_OUT_LEN) {
        return enif_make_badarg(env);
    }

    /* Constant-time comparison */
    uint8_t diff = 0;
    for (int i = 0; i < BLAKE3_OUT_LEN; i++) {
        diff |= a.data[i] ^ b.data[i];
    }

    return diff == 0 ? enif_make_atom(env, "true") : enif_make_atom(env, "false");
}

/* Incremental API */
static ERL_NIF_TERM nif_hasher_new(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc; (void)argv;

    blake3_hasher* hasher = enif_alloc_resource(BLAKE3_HASHER_TYPE, sizeof(blake3_hasher));
    if (!hasher) return enif_make_badarg(env);

    blake3_hasher_init(hasher);

    ERL_NIF_TERM term = enif_make_resource(env, hasher);
    enif_release_resource(hasher);
    return term;
}

static ERL_NIF_TERM nif_hasher_new_keyed(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary key;
    if (!enif_inspect_binary(env, argv[0], &key) || key.size != BLAKE3_KEY_LEN) {
        return enif_make_badarg(env);
    }

    blake3_hasher* hasher = enif_alloc_resource(BLAKE3_HASHER_TYPE, sizeof(blake3_hasher));
    if (!hasher) return enif_make_badarg(env);

    blake3_hasher_init_keyed(hasher, key.data);

    ERL_NIF_TERM term = enif_make_resource(env, hasher);
    enif_release_resource(hasher);
    return term;
}

static ERL_NIF_TERM nif_hasher_update(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    blake3_hasher* hasher;
    ErlNifBinary data;

    if (!enif_get_resource(env, argv[0], BLAKE3_HASHER_TYPE, (void**)&hasher)) {
        return enif_make_badarg(env);
    }
    if (!enif_inspect_binary(env, argv[1], &data)) {
        return enif_make_badarg(env);
    }

    blake3_hasher_update(hasher, data.data, data.size);
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM nif_hasher_finalize(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    blake3_hasher* hasher;

    if (!enif_get_resource(env, argv[0], BLAKE3_HASHER_TYPE, (void**)&hasher)) {
        return enif_make_badarg(env);
    }

    ErlNifBinary output;
    if (!enif_alloc_binary(BLAKE3_OUT_LEN, &output)) {
        return enif_make_badarg(env);
    }

    blake3_hasher_finalize(hasher, output.data, BLAKE3_OUT_LEN);
    return enif_make_binary(env, &output);
}

static ERL_NIF_TERM nif_hasher_finalize_xof(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    blake3_hasher* hasher;
    unsigned int length;

    if (!enif_get_resource(env, argv[0], BLAKE3_HASHER_TYPE, (void**)&hasher)) {
        return enif_make_badarg(env);
    }
    if (!enif_get_uint(env, argv[1], &length) || length == 0 || length > 65536) {
        return enif_make_badarg(env);
    }

    ErlNifBinary output;
    if (!enif_alloc_binary(length, &output)) {
        return enif_make_badarg(env);
    }

    blake3_hasher_finalize(hasher, output.data, length);
    return enif_make_binary(env, &output);
}

/* NIF lifecycle */
static int load(ErlNifEnv* env, void** priv, ERL_NIF_TERM info) {
    (void)priv; (void)info;

    BLAKE3_HASHER_TYPE = enif_open_resource_type(
        env, NULL, "blake3_hasher", hasher_dtor,
        ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);

    return BLAKE3_HASHER_TYPE ? 0 : -1;
}

static ErlNifFunc nif_funcs[] = {
    {"hash", 1, nif_hash, 0},
    {"hash_keyed", 2, nif_hash_keyed, 0},
    {"derive_key", 2, nif_derive_key, 0},
    {"to_hex", 1, nif_to_hex, 0},
    {"compare", 2, nif_compare, 0},
    {"hasher_new", 0, nif_hasher_new, 0},
    {"hasher_new_keyed", 1, nif_hasher_new_keyed, 0},
    {"hasher_update", 2, nif_hasher_update, 0},
    {"hasher_finalize", 1, nif_hasher_finalize, 0},
    {"hasher_finalize_xof", 2, nif_hasher_finalize_xof, 0}
};

ERL_NIF_INIT(Elixir.MerkleDb.Blake3, nif_funcs, load, NULL, NULL, NULL)
