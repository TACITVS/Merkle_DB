/**
 * BLAKE3 NIF - Standalone NIF module for BLAKE3 cryptographic hashing
 *
 * This is a self-contained NIF module that exposes fp_blake3 functions to Elixir.
 * Loads as MerkleDb.Blake3Native module.
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

#include "fp_lib/include/fp_blake3.h"

/* ============================================================================
 * Resource Types
 * ============================================================================ */

static ErlNifResourceType* BLAKE3_HASHER_TYPE = NULL;

static void blake3_hasher_dtor(ErlNifEnv* env, void* obj) {
    (void)env;
    (void)obj;
}

/* ============================================================================
 * Simple API NIFs
 * ============================================================================ */

/**
 * hash(binary()) -> binary()
 */
static ERL_NIF_TERM nif_blake3_hash(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary input;
    if (!enif_inspect_binary(env, argv[0], &input)) {
        return enif_make_badarg(env);
    }

    ErlNifBinary output;
    if (!enif_alloc_binary(FP_BLAKE3_OUT_LEN, &output)) {
        return enif_make_badarg(env);
    }

    fp_blake3_hash(input.data, input.size, output.data);

    return enif_make_binary(env, &output);
}

/**
 * hash_keyed(key :: binary(), input :: binary()) -> binary()
 */
static ERL_NIF_TERM nif_blake3_hash_keyed(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary key;
    ErlNifBinary input;

    if (!enif_inspect_binary(env, argv[0], &key) || key.size != FP_BLAKE3_KEY_LEN) {
        return enif_make_badarg(env);
    }
    if (!enif_inspect_binary(env, argv[1], &input)) {
        return enif_make_badarg(env);
    }

    ErlNifBinary output;
    if (!enif_alloc_binary(FP_BLAKE3_OUT_LEN, &output)) {
        return enif_make_badarg(env);
    }

    fp_blake3_hash_keyed(key.data, input.data, input.size, output.data);

    return enif_make_binary(env, &output);
}

/**
 * derive_key(context :: binary(), key_material :: binary()) -> binary()
 */
static ERL_NIF_TERM nif_blake3_derive_key(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary context;
    ErlNifBinary key_material;

    if (!enif_inspect_binary(env, argv[0], &context)) {
        return enif_make_badarg(env);
    }
    if (!enif_inspect_binary(env, argv[1], &key_material)) {
        return enif_make_badarg(env);
    }

    ErlNifBinary output;
    if (!enif_alloc_binary(FP_BLAKE3_OUT_LEN, &output)) {
        return enif_make_badarg(env);
    }

    fp_blake3_derive_key((const char*)context.data, context.size,
                         key_material.data, key_material.size,
                         output.data);

    return enif_make_binary(env, &output);
}

/**
 * to_hex(hash :: binary()) -> binary()
 */
static ERL_NIF_TERM nif_blake3_to_hex(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary hash;
    if (!enif_inspect_binary(env, argv[0], &hash) || hash.size != FP_BLAKE3_OUT_LEN) {
        return enif_make_badarg(env);
    }

    ErlNifBinary output;
    if (!enif_alloc_binary(64, &output)) {
        return enif_make_badarg(env);
    }

    fp_blake3_to_hex(hash.data, (char*)output.data);

    return enif_make_binary(env, &output);
}

/**
 * compare(a :: binary(), b :: binary()) -> boolean()
 */
static ERL_NIF_TERM nif_blake3_compare(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary a, b;

    if (!enif_inspect_binary(env, argv[0], &a) || a.size != FP_BLAKE3_OUT_LEN) {
        return enif_make_badarg(env);
    }
    if (!enif_inspect_binary(env, argv[1], &b) || b.size != FP_BLAKE3_OUT_LEN) {
        return enif_make_badarg(env);
    }

    int result = fp_blake3_compare(a.data, b.data);

    return result == 0 ? enif_make_atom(env, "true") : enif_make_atom(env, "false");
}

/* ============================================================================
 * Incremental API NIFs
 * ============================================================================ */

/**
 * hasher_new() -> reference()
 */
static ERL_NIF_TERM nif_blake3_hasher_new(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    (void)argv;

    FpBlake3Hasher* hasher = enif_alloc_resource(BLAKE3_HASHER_TYPE, sizeof(FpBlake3Hasher));
    if (hasher == NULL) {
        return enif_make_badarg(env);
    }

    fp_blake3_hasher_init(hasher);

    ERL_NIF_TERM term = enif_make_resource(env, hasher);
    enif_release_resource(hasher);

    return term;
}

/**
 * hasher_new_keyed(key :: binary()) -> reference()
 */
static ERL_NIF_TERM nif_blake3_hasher_new_keyed(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary key;
    if (!enif_inspect_binary(env, argv[0], &key) || key.size != FP_BLAKE3_KEY_LEN) {
        return enif_make_badarg(env);
    }

    FpBlake3Hasher* hasher = enif_alloc_resource(BLAKE3_HASHER_TYPE, sizeof(FpBlake3Hasher));
    if (hasher == NULL) {
        return enif_make_badarg(env);
    }

    fp_blake3_hasher_init_keyed(hasher, key.data);

    ERL_NIF_TERM term = enif_make_resource(env, hasher);
    enif_release_resource(hasher);

    return term;
}

/**
 * hasher_update(hasher :: reference(), data :: binary()) -> :ok
 */
static ERL_NIF_TERM nif_blake3_hasher_update(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    FpBlake3Hasher* hasher;
    ErlNifBinary data;

    if (!enif_get_resource(env, argv[0], BLAKE3_HASHER_TYPE, (void**)&hasher)) {
        return enif_make_badarg(env);
    }
    if (!enif_inspect_binary(env, argv[1], &data)) {
        return enif_make_badarg(env);
    }

    fp_blake3_hasher_update(hasher, data.data, data.size);

    return enif_make_atom(env, "ok");
}

/**
 * hasher_finalize(hasher :: reference()) -> binary()
 */
static ERL_NIF_TERM nif_blake3_hasher_finalize(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    FpBlake3Hasher* hasher;

    if (!enif_get_resource(env, argv[0], BLAKE3_HASHER_TYPE, (void**)&hasher)) {
        return enif_make_badarg(env);
    }

    ErlNifBinary output;
    if (!enif_alloc_binary(FP_BLAKE3_OUT_LEN, &output)) {
        return enif_make_badarg(env);
    }

    fp_blake3_hasher_finalize(hasher, output.data);

    return enif_make_binary(env, &output);
}

/**
 * hasher_finalize_xof(hasher :: reference(), length :: integer()) -> binary()
 */
static ERL_NIF_TERM nif_blake3_hasher_finalize_xof(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    FpBlake3Hasher* hasher;
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

    fp_blake3_hasher_finalize_xof(hasher, output.data, length);

    return enif_make_binary(env, &output);
}

/* ============================================================================
 * NIF Lifecycle
 * ============================================================================ */

static int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info) {
    (void)priv_data;
    (void)load_info;

    BLAKE3_HASHER_TYPE = enif_open_resource_type(
        env, NULL, "blake3_hasher",
        blake3_hasher_dtor,
        ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER,
        NULL
    );

    if (BLAKE3_HASHER_TYPE == NULL) {
        return -1;
    }

    return 0;
}

static void unload(ErlNifEnv* env, void* priv_data) {
    (void)env;
    (void)priv_data;
}

/* ============================================================================
 * NIF Registration
 * ============================================================================ */

static ErlNifFunc nif_funcs[] = {
    /* Simple API */
    {"hash", 1, nif_blake3_hash, 0},
    {"hash_keyed", 2, nif_blake3_hash_keyed, 0},
    {"derive_key", 2, nif_blake3_derive_key, 0},
    {"to_hex", 1, nif_blake3_to_hex, 0},
    {"compare", 2, nif_blake3_compare, 0},

    /* Incremental API */
    {"hasher_new", 0, nif_blake3_hasher_new, 0},
    {"hasher_new_keyed", 1, nif_blake3_hasher_new_keyed, 0},
    {"hasher_update", 2, nif_blake3_hasher_update, 0},
    {"hasher_finalize", 1, nif_blake3_hasher_finalize, 0},
    {"hasher_finalize_xof", 2, nif_blake3_hasher_finalize_xof, 0}
};

ERL_NIF_INIT(Elixir.MerkleDb.Blake3, nif_funcs, load, NULL, NULL, unload)
